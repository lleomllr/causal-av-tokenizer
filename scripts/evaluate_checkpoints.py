from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from avtokenizer import AVWindowConfig, SynchronizedAVDataset
from avtokenizer.video_autoencoder import VideoAutoencoder, psnr_from_l1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate several video tokenizer checkpoints on the same validation windows."
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Checkpoint paths, optionally named as model_name=path/to/best.pt.",
    )
    parser.add_argument("--manifest", type=str, default="data/manifest.csv")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--output", type=str, default="outputs/checkpoint_eval.csv")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    return parser.parse_args()


def choose_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_checkpoint_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        name, path = spec.split("=", maxsplit=1)
        return name, Path(path)
    path = Path(spec)
    return path.parent.name or path.stem, path


def parse_fsq_levels(levels: object) -> tuple[int, ...]:
    if isinstance(levels, str):
        parsed = tuple(int(part.strip()) for part in levels.split(",") if part.strip())
        return parsed or (8, 8, 8, 8, 8, 8)
    if isinstance(levels, (list, tuple)):
        return tuple(int(level) for level in levels)
    return (8, 8, 8, 8, 8, 8)


def load_checkpoint(path: Path, device: torch.device) -> tuple[VideoAutoencoder, dict[str, object]]:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    model = VideoAutoencoder(
        latent_channels=int(config.get("latent_channels", 128)),
        bottleneck=str(config.get("bottleneck", "ae")),
        fsq_levels=parse_fsq_levels(config.get("fsq_levels", "8,8,8,8,8,8")),
    ).to(device)
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
def infer_latent_shape(model: VideoAutoencoder, frames: torch.Tensor) -> str:
    latents, _ = model.encode(frames[:1])
    _, channels, height, width = latents.shape
    return f"{channels}x{height}x{width}"


@torch.no_grad()
def evaluate_model(
    *,
    model: VideoAutoencoder,
    loader: DataLoader,
    device: torch.device,
    max_batches: int,
) -> tuple[float, float, str]:
    l1_values: list[float] = []
    psnr_values: list[float] = []
    latent_shape = ""

    for step, batch in enumerate(loader, start=1):
        frames = flatten_video_batch(batch, device)
        recon = model(frames).clamp(0.0, 1.0)
        l1 = F.l1_loss(recon, frames)
        l1_values.append(float(l1.cpu()))
        psnr_values.append(float(psnr_from_l1(l1).cpu()))
        if not latent_shape:
            latent_shape = infer_latent_shape(model, frames)
        if max_batches and step >= max_batches:
            break

    return sum(l1_values) / len(l1_values), sum(psnr_values) / len(psnr_values), latent_shape


def main() -> None:
    args = parse_args()
    device = choose_device()
    checkpoint_specs = [parse_checkpoint_spec(spec) for spec in args.checkpoints]
    if not checkpoint_specs:
        raise ValueError("At least one checkpoint is required")

    first_model, first_config = load_checkpoint(checkpoint_specs[0][1], device)
    del first_model
    dataset_config = dataset_config_from_checkpoint(first_config)
    dataset = SynchronizedAVDataset(manifest=args.manifest, split=args.split, config=dataset_config)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    rows: list[dict[str, str]] = []
    for model_name, checkpoint_path in checkpoint_specs:
        model, config = load_checkpoint(checkpoint_path, device)
        val_l1, val_psnr, latent_shape = evaluate_model(
            model=model,
            loader=loader,
            device=device,
            max_batches=args.max_batches,
        )
        bottleneck = str(config.get("bottleneck", "ae"))
        rows.append(
            {
                "model": model_name,
                "bottleneck": bottleneck,
                "latent_shape": latent_shape,
                "discrete": str(bottleneck == "fsq").lower(),
                "val_l1": f"{val_l1:.4f}",
                "val_psnr": f"{val_psnr:.2f}",
            }
        )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["model", "bottleneck", "latent_shape", "discrete", "val_l1", "val_psnr"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(",".join(fieldnames))
    for row in rows:
        print(",".join(row[field] for field in fieldnames))
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
