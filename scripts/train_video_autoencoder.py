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
from avtokenizer.video_autoencoder import VideoAutoencoder, edge_loss, psnr_from_l1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a frame-wise video autoencoder baseline.")
    parser.add_argument("--manifest", type=str, default="data/manifest.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/video_ae")
    parser.add_argument("--clip-seconds", type=float, default=2.0)
    parser.add_argument("--stride-seconds", type=float, default=0.5)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--latent-channels", type=int, default=128)
    parser.add_argument("--bottleneck", type=str, default="ae", choices=["ae", "vae", "fsq"])
    parser.add_argument("--fsq-levels", type=str, default="8,8,8,8,8,8")
    parser.add_argument("--edge-loss-weight", type=float, default=0.1)
    parser.add_argument("--kl-weight", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=2)
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


def flatten_video_batch(batch: dict[str, object], device: torch.device) -> torch.Tensor:
    video = batch["video"].to(device)
    batch_size, num_frames, channels, height, width = video.shape
    return video.reshape(batch_size * num_frames, channels, height, width)


def parse_fsq_levels(levels: str) -> tuple[int, ...]:
    parsed = tuple(int(part.strip()) for part in levels.split(",") if part.strip())
    if not parsed:
        raise ValueError("--fsq-levels must contain at least one integer")
    return parsed


def compute_losses(
    *,
    model: VideoAutoencoder,
    frames: torch.Tensor,
    edge_weight: float,
    kl_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    recon, aux = model(frames, return_aux=True)
    l1 = F.l1_loss(recon, frames)
    edges = edge_loss(recon, frames)
    kl = aux["kl_loss"]
    total = l1 + edge_weight * edges + kl_weight * kl
    metrics = {
        "loss": float(total.detach().cpu()),
        "l1": float(l1.detach().cpu()),
        "edge": float(edges.detach().cpu()),
        "kl": float(kl.detach().cpu()),
        "psnr": float(psnr_from_l1(l1.detach()).cpu()),
    }
    return total, metrics


def run_epoch(
    *,
    model: VideoAutoencoder,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    edge_weight: float,
    kl_weight: float,
    max_batches: int = 0,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    metric_sums = {"loss": 0.0, "l1": 0.0, "edge": 0.0, "kl": 0.0, "psnr": 0.0}
    count = 0

    for step, batch in enumerate(loader, start=1):
        frames = flatten_video_batch(batch, device)
        with torch.set_grad_enabled(is_train):
            loss, metrics = compute_losses(
                model=model,
                frames=frames,
                edge_weight=edge_weight,
                kl_weight=kl_weight,
            )
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        for key in metric_sums:
            metric_sums[key] += metrics[key]
        count += 1
        if max_batches and step >= max_batches:
            break

    return {key: value / count for key, value in metric_sums.items()}


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

    fsq_levels = parse_fsq_levels(args.fsq_levels)
    model = VideoAutoencoder(
        latent_channels=args.latent_channels,
        bottleneck=args.bottleneck,
        fsq_levels=fsq_levels,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    print(f"device: {device}")
    print(f"bottleneck: {args.bottleneck}")
    print(f"train windows: {len(train_dataset)} | val windows: {len(val_dataset)}")

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            edge_weight=args.edge_loss_weight,
            kl_weight=args.kl_weight,
            max_batches=args.max_train_batches,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
            edge_weight=args.edge_loss_weight,
            kl_weight=args.kl_weight,
            max_batches=args.max_val_batches,
        )
        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f} L1 {train_metrics['l1']:.4f} "
            f"edge {train_metrics['edge']:.4f} KL {train_metrics['kl']:.4f} "
            f"PSNR {train_metrics['psnr']:.2f} | "
            f"val loss {val_metrics['loss']:.4f} L1 {val_metrics['l1']:.4f} "
            f"edge {val_metrics['edge']:.4f} KL {val_metrics['kl']:.4f} "
            f"PSNR {val_metrics['psnr']:.2f}"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            checkpoint = {
                "model": model.state_dict(),
                "config": vars(args),
                "best_val_loss": best_val,
                "best_val_l1": val_metrics["l1"],
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
