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
from avtokenizer.video_autoencoder import VideoAutoencoder, psnr_from_l1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a frame-wise video autoencoder baseline.")
    parser.add_argument("--manifest", type=str, default="data/manifest.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/video_ae")
    parser.add_argument("--clip-seconds", type=float, default=2.0)
    parser.add_argument("--stride-seconds", type=float, default=0.5)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--latent-channels", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    return parser.parse_args()


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def flatten_video_batch(batch: dict[str, object], device: torch.device) -> torch.Tensor:
    video = batch["video"].to(device)
    batch_size, num_frames, channels, height, width = video.shape
    return video.reshape(batch_size * num_frames, channels, height, width)


def run_epoch(
    *,
    model: VideoAutoencoder,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    max_batches: int = 0,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)
    losses: list[float] = []
    psnrs: list[float] = []

    for step, batch in enumerate(loader, start=1):
        frames = flatten_video_batch(batch, device)
        with torch.set_grad_enabled(is_train):
            recon = model(frames)
            loss = F.l1_loss(recon, frames)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        losses.append(float(loss.detach().cpu()))
        psnrs.append(float(psnr_from_l1(loss.detach()).cpu()))
        if max_batches and step >= max_batches:
            break

    return sum(losses) / len(losses), sum(psnrs) / len(psnrs)


@torch.no_grad()
def save_reconstruction_grid(
    *,
    model: VideoAutoencoder,
    loader: DataLoader,
    device: torch.device,
    output: Path,
    max_items: int = 8,
) -> None:
    model.eval()
    batch = next(iter(loader))
    frames = flatten_video_batch(batch, device)[:max_items]
    recon = model(frames).clamp(0.0, 1.0)
    error = (frames - recon).abs().mean(dim=1, keepdim=True).repeat(1, 3, 1, 1).clamp(0.0, 1.0)

    rows = []
    for title, tensor in [("original", frames), ("reconstruction", recon), ("absolute error", error)]:
        images = tensor.detach().cpu().permute(0, 2, 3, 1).numpy()
        rows.append((title, images))

    fig, axes = plt.subplots(nrows=3, ncols=len(rows[0][1]), figsize=(len(rows[0][1]) * 1.5, 4.8))
    if len(rows[0][1]) == 1:
        axes = axes.reshape(3, 1)
    for row_index, (title, images) in enumerate(rows):
        for col_index, image in enumerate(images):
            axes[row_index, col_index].imshow(image)
            axes[row_index, col_index].set_axis_off()
            if col_index == 0:
                axes[row_index, col_index].set_ylabel(title)

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
        image_size=args.image_size,
        include_audio=False,
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

    model = VideoAutoencoder(latent_channels=args.latent_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"device: {device}")
    print(f"train windows: {len(train_dataset)} | val windows: {len(val_dataset)}")

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_psnr = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            max_batches=args.max_train_batches,
        )
        val_loss, val_psnr = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
        )
        print(
            f"epoch {epoch:03d} | "
            f"train L1 {train_loss:.4f} PSNR {train_psnr:.2f} | "
            f"val L1 {val_loss:.4f} PSNR {val_psnr:.2f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            checkpoint = {
                "model": model.state_dict(),
                "config": vars(args),
                "best_val_l1": best_val,
            }
            torch.save(checkpoint, output_dir / "best.pt")
            save_reconstruction_grid(
                model=model,
                loader=val_loader,
                device=device,
                output=output_dir / "reconstructions.png",
            )

    print(f"saved checkpoint: {output_dir / 'best.pt'}")
    print(f"saved reconstructions: {output_dir / 'reconstructions.png'}")


if __name__ == "__main__":
    main()
