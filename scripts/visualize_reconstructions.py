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
    parser = argparse.ArgumentParser(description="Visualize video autoencoder reconstructions.")
    parser.add_argument("--checkpoint", type=str, default="outputs/video_ae/best.pt")
    parser.add_argument("--manifest", type=str, default="data/manifest.csv")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--output", type=str, default="outputs/video_ae/reconstructions_eval.png")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(checkpoint_path: Path, device: torch.device) -> tuple[VideoAutoencoder, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    latent_channels = int(config.get("latent_channels", 128))
    model = VideoAutoencoder(latent_channels=latent_channels).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model, config


def dataset_config_from_checkpoint(config: dict[str, object]) -> AVWindowConfig:
    return AVWindowConfig(
        clip_seconds=float(config.get("clip_seconds", 2.0)),
        stride_seconds=float(config.get("stride_seconds", 0.5)),
        fps=int(config.get("fps", 12)),
        image_size=int(config.get("image_size", 128)),
        include_audio=False,
    )


def flatten_video_batch(batch: dict[str, object], device: torch.device) -> torch.Tensor:
    video = batch["video"].to(device)
    batch_size, num_frames, channels, height, width = video.shape
    return video.reshape(batch_size * num_frames, channels, height, width)


@torch.no_grad()
def collect_reconstructions(
    *,
    model: VideoAutoencoder,
    loader: DataLoader,
    device: torch.device,
    max_items: int,
) -> tuple[torch.Tensor, torch.Tensor, float, float]:
    frames_list: list[torch.Tensor] = []
    recon_list: list[torch.Tensor] = []
    losses: list[float] = []
    psnrs: list[float] = []

    for batch in loader:
        frames = flatten_video_batch(batch, device)
        recon = model(frames).clamp(0.0, 1.0)
        loss = F.l1_loss(recon, frames)
        losses.append(float(loss.cpu()))
        psnrs.append(float(psnr_from_l1(loss).cpu()))

        remaining = max_items - sum(t.shape[0] for t in frames_list)
        if remaining > 0:
            frames_list.append(frames[:remaining].detach().cpu())
            recon_list.append(recon[:remaining].detach().cpu())
        if sum(t.shape[0] for t in frames_list) >= max_items:
            break

    frames_out = torch.cat(frames_list, dim=0)
    recon_out = torch.cat(recon_list, dim=0)
    return frames_out, recon_out, sum(losses) / len(losses), sum(psnrs) / len(psnrs)


def save_grid(
    *,
    frames: torch.Tensor,
    recon: torch.Tensor,
    output: Path,
    title: str,
) -> None:
    error = (frames - recon).abs().mean(dim=1, keepdim=True).repeat(1, 3, 1, 1).clamp(0.0, 1.0)
    rows = [
        ("original", frames),
        ("reconstruction", recon),
        ("absolute error", error),
    ]
    num_cols = frames.shape[0]
    fig, axes = plt.subplots(nrows=3, ncols=num_cols, figsize=(num_cols * 1.45, 4.8), squeeze=False)

    for row_index, (label, tensor) in enumerate(rows):
        images = tensor.permute(0, 2, 3, 1).numpy()
        for col_index, image in enumerate(images):
            axes[row_index, col_index].imshow(image)
            axes[row_index, col_index].set_axis_off()
            if col_index == 0:
                axes[row_index, col_index].set_ylabel(label)

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = choose_device()

    checkpoint_path = Path(args.checkpoint)
    model, checkpoint_config = load_model(checkpoint_path, device)
    dataset_config = dataset_config_from_checkpoint(checkpoint_config)
    dataset = SynchronizedAVDataset(manifest=args.manifest, split=args.split, config=dataset_config)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    frames, recon, l1, psnr = collect_reconstructions(
        model=model,
        loader=loader,
        device=device,
        max_items=args.max_items,
    )
    output = Path(args.output)
    save_grid(
        frames=frames,
        recon=recon,
        output=output,
        title=f"{args.split} reconstructions | L1 {l1:.4f} | PSNR {psnr:.2f} dB",
    )

    print(f"checkpoint: {checkpoint_path}")
    print(f"split: {args.split} | windows: {len(dataset)}")
    print(f"L1: {l1:.4f} | PSNR: {psnr:.2f} dB")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
