from __future__ import annotations
import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

mpl_config_dir = Path(__file__).resolve().parents[1] / "outputs" / ".matplotlib"
try:
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
except PermissionError:
    mpl_config_dir = Path(tempfile.gettempdir()) / "causal_av_tokenizer_matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
if not os.access(mpl_config_dir, os.W_OK):
    mpl_config_dir = Path(tempfile.gettempdir()) / "causal_av_tokenizer_matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from avtokenizer import AVWindowConfig, SynchronizedAVDataset
from avtokenizer.audio_autoencoder import AudioMelAutoencoder
from avtokenizer.video_autoencoder import psnr_from_l1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an audio log-mel autoencoder baseline.")
    parser.add_argument("--manifest", type=str, default="data/manifest.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/audio_mel_ae")
    parser.add_argument("--clip-seconds", type=float, default=2.0)
    parser.add_argument("--stride-seconds", type=float, default=0.5)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--audio-steps-per-frame", type=int, default=4)
    parser.add_argument("--latent-channels", type=int, default=64)
    parser.add_argument("--bottleneck", type=str, default="ae", choices=["ae", "fsq"])
    parser.add_argument("--fsq-levels", type=str, default="8,8,8,8,8,8,8,8")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    return parser.parse_args()


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_fsq_levels(levels: str) -> tuple[int, ...]:
    parsed = tuple(int(part.strip()) for part in levels.split(",") if part.strip())
    if not parsed:
        raise ValueError("--fsq-levels must contain at least one integer")
    return parsed


def mel_batch_to_image(batch: dict[str, object], device: torch.device) -> torch.Tensor:
    mel = batch["audio_mel"].to(device)
    batch_size, num_frames, n_mels, steps_per_frame = mel.shape
    return mel.permute(0, 2, 1, 3).reshape(batch_size, 1, n_mels, num_frames * steps_per_frame)


def run_epoch(
    *,
    model: AudioMelAutoencoder,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int = 0,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    metric_sums = {"loss": 0.0, "l1": 0.0, "psnr": 0.0}
    count = 0

    for step, batch in enumerate(loader, start=1):
        mel = mel_batch_to_image(batch, device)
        with torch.set_grad_enabled(is_train):
            recon = model(mel)
            loss = F.l1_loss(recon, mel)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        metric_sums["loss"] += float(loss.detach().cpu())
        metric_sums["l1"] += float(loss.detach().cpu())
        metric_sums["psnr"] += float(psnr_from_l1(loss.detach()).cpu())
        count += 1
        if max_batches and step >= max_batches:
            break

    return {key: value / count for key, value in metric_sums.items()}


@torch.no_grad()
def save_mel_reconstruction_grid(
    *,
    model: AudioMelAutoencoder,
    loader: DataLoader,
    device: torch.device,
    output: Path,
    max_items: int = 4,
) -> None:
    model.eval()
    batch = next(iter(loader))
    mel = mel_batch_to_image(batch, device)[:max_items]
    recon = model(mel).clamp_min(0.0)
    error = (mel - recon).abs()

    rows = [("original log-mel", mel), ("reconstruction", recon), ("absolute error", error)]
    num_cols = mel.shape[0]
    fig, axes = plt.subplots(nrows=3, ncols=num_cols, figsize=(num_cols * 3.2, 6.5), squeeze=False)
    for row_index, (label, tensor) in enumerate(rows):
        images = tensor.detach().cpu().squeeze(1).numpy()
        vmax = images.max() if row_index < 2 else None
        for col_index, image in enumerate(images):
            axes[row_index, col_index].imshow(
                image,
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                cmap="magma" if row_index < 2 else "gray",
                vmin=0.0,
                vmax=vmax,
            )
            axes[row_index, col_index].set_axis_off()
            if col_index == 0:
                axes[row_index, col_index].set_ylabel(label)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = choose_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = AVWindowConfig(
        clip_seconds=args.clip_seconds,
        stride_seconds=args.stride_seconds,
        fps=args.fps,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        audio_steps_per_frame=args.audio_steps_per_frame,
        include_video=False,
        include_audio=True,
    )
    train_dataset = SynchronizedAVDataset(manifest=args.manifest, split="train", config=config)
    val_dataset = SynchronizedAVDataset(manifest=args.manifest, split="val", config=config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = AudioMelAutoencoder(
        latent_channels=args.latent_channels,
        bottleneck=args.bottleneck,
        fsq_levels=parse_fsq_levels(args.fsq_levels),
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"device: {device}")
    print(f"audio bottleneck: {args.bottleneck}")
    print(f"train windows: {len(train_dataset)} | val windows: {len(val_dataset)}")

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
            max_batches=args.max_val_batches,
        )
        print(
            f"epoch {epoch:03d} | "
            f"train L1 {train_metrics['l1']:.4f} PSNR {train_metrics['psnr']:.2f} | "
            f"val L1 {val_metrics['l1']:.4f} PSNR {val_metrics['psnr']:.2f}"
        )
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            checkpoint = {
                "model": model.state_dict(),
                "config": vars(args),
                "best_val_l1": val_metrics["l1"],
            }
            torch.save(checkpoint, output_dir / "best.pt")
            save_mel_reconstruction_grid(
                model=model,
                loader=val_loader,
                device=device,
                output=output_dir / "mel_reconstruct.png",
            )

    print(f"saved checkpoint: {output_dir / 'best.pt'}")
    print(f"saved reconstructions: {output_dir / 'mel_reconstruct.png'}")


if __name__ == "__main__":
    main()
