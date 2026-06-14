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
from avtokenizer.av_autoencoder import AudioVisualAutoencoder, frame_aligned_mel_to_image
from avtokenizer.video_autoencoder import edge_loss, psnr_from_l1


DEFAULT_CONDITIONS = (
    "normal",
    "audio_zeroed",
    "video_zeroed",
    "audio_shuffled",
    "video_shuffled",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate joint audio-video checkpoints with modality ablations."
    )
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        required=True,
        help="Joint AV checkpoints, optionally named as model_name=path/to/best.pt.",
    )
    parser.add_argument("--manifest", type=str, default="data/manifest.csv")
    parser.add_argument("--split", type=str, default="val")
    parser.add_argument("--output", type=str, default="outputs/av_ablations.csv")
    parser.add_argument("--conditions", nargs="+", default=list(DEFAULT_CONDITIONS))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
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


def parse_levels(value: object, fallback: str) -> tuple[int, ...]:
    if isinstance(value, str):
        parsed = tuple(int(part.strip()) for part in value.split(",") if part.strip())
        return parsed or parse_levels(fallback, fallback)
    if isinstance(value, (list, tuple)):
        return tuple(int(item) for item in value)
    return parse_levels(fallback, fallback)


def load_component_config(path: object) -> dict[str, object]:
    if not path:
        return {}
    checkpoint_path = Path(str(path))
    if not checkpoint_path.exists():
        return {}
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return checkpoint.get("config", {})


def build_model_from_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[AudioVisualAutoencoder, dict[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = checkpoint.get("config", {})
    video_config = load_component_config(config.get("video_checkpoint"))
    audio_config = load_component_config(config.get("audio_checkpoint"))

    model = AudioVisualAutoencoder(
        video_latent_channels=int(video_config.get("latent_channels", config.get("video_latent_channels", 128))),
        video_bottleneck=str(video_config.get("bottleneck", config.get("video_bottleneck", "fsq"))),
        video_fsq_levels=parse_levels(
            video_config.get("fsq_levels", config.get("video_fsq_levels", "8,8,8,8,8,8,8,8")),
            "8,8,8,8,8,8,8,8",
        ),
        audio_latent_channels=int(audio_config.get("latent_channels", config.get("audio_latent_channels", 64))),
        audio_bottleneck=str(audio_config.get("bottleneck", config.get("audio_bottleneck", "ae"))),
        audio_fsq_levels=parse_levels(
            audio_config.get("fsq_levels", config.get("audio_fsq_levels", "8,8,8,8,8,8,8,8")),
            "8,8,8,8,8,8,8,8",
        ),
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
        sample_rate=int(config.get("sample_rate", 16_000)),
        n_mels=int(config.get("n_mels", 80)),
        audio_steps_per_frame=int(config.get("audio_steps_per_frame", 4)),
        include_video=True,
        include_audio=True,
    )


def apply_condition(
    *,
    video: torch.Tensor,
    audio_mel: torch.Tensor,
    condition: str,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    if condition == "normal":
        return video, audio_mel
    if condition == "audio_zeroed":
        return video, torch.zeros_like(audio_mel)
    if condition == "video_zeroed":
        return torch.zeros_like(video), audio_mel
    if condition == "audio_shuffled":
        return video, audio_mel[batch_permutation(audio_mel.shape[0], audio_mel.device, generator)]
    if condition == "video_shuffled":
        return video[batch_permutation(video.shape[0], video.device, generator)], audio_mel
    raise ValueError(f"Unknown ablation condition: {condition}")


def batch_permutation(batch_size: int, device: torch.device, generator: torch.Generator) -> torch.Tensor:
    if batch_size < 2:
        return torch.arange(batch_size, device=device)
    permutation = torch.randperm(batch_size, generator=generator, device="cpu")
    if torch.equal(permutation, torch.arange(batch_size)):
        permutation = permutation.roll(1)
    return permutation.to(device)


@torch.no_grad()
def infer_latent_shapes(
    *,
    model: AudioVisualAutoencoder,
    batch: dict[str, object],
    device: torch.device,
) -> tuple[str, str]:
    video = batch["video"].to(device)
    audio_mel = batch["audio_mel"].to(device)
    batch_size, num_frames, channels, height, width = video.shape
    frames = video.reshape(batch_size * num_frames, channels, height, width)
    mel = frame_aligned_mel_to_image(audio_mel)
    video_latents, _ = model.video_model.encode(frames[:1])
    audio_latents, _ = model.audio_model.encode(mel[:1])
    _, video_channels, video_height, video_width = video_latents.shape
    _, audio_channels, audio_height, audio_width = audio_latents.shape
    return (
        f"{video_channels}x{video_height}x{video_width}",
        f"{audio_channels}x{audio_height}x{audio_width}",
    )


@torch.no_grad()
def evaluate_condition(
    *,
    model: AudioVisualAutoencoder,
    loader: DataLoader,
    device: torch.device,
    condition: str,
    video_edge_weight: float,
    audio_loss_weight: float,
    max_batches: int,
    seed: int,
) -> dict[str, float]:
    totals = {
        "loss": 0.0,
        "video_l1": 0.0,
        "video_edge": 0.0,
        "video_psnr": 0.0,
        "audio_l1": 0.0,
        "audio_psnr": 0.0,
    }
    count = 0
    generator = torch.Generator(device="cpu").manual_seed(seed)

    for step, batch in enumerate(loader, start=1):
        video = batch["video"].to(device)
        audio_mel = batch["audio_mel"].to(device)
        conditioned_video, conditioned_audio = apply_condition(
            video=video,
            audio_mel=audio_mel,
            condition=condition,
            generator=generator,
        )
        recon = model(conditioned_video, conditioned_audio)

        batch_size, num_frames, channels, height, width = video.shape
        target_frames = video.reshape(batch_size * num_frames, channels, height, width)
        recon_frames = recon["video"].reshape(batch_size * num_frames, channels, height, width)
        video_l1 = F.l1_loss(recon_frames, target_frames)
        video_edges = edge_loss(recon_frames, target_frames)
        audio_l1 = F.l1_loss(recon["audio_mel"], audio_mel)
        loss = video_l1 + video_edge_weight * video_edges + audio_loss_weight * audio_l1

        totals["loss"] += float(loss.cpu())
        totals["video_l1"] += float(video_l1.cpu())
        totals["video_edge"] += float(video_edges.cpu())
        totals["video_psnr"] += float(psnr_from_l1(video_l1).cpu())
        totals["audio_l1"] += float(audio_l1.cpu())
        totals["audio_psnr"] += float(psnr_from_l1(audio_l1).cpu())
        count += 1
        if max_batches and step >= max_batches:
            break

    if count == 0:
        raise ValueError("No batches were evaluated")
    return {key: value / count for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    device = choose_device()
    checkpoint_specs = [parse_checkpoint_spec(spec) for spec in args.checkpoints]

    first_model, first_config = build_model_from_checkpoint(checkpoint_specs[0][1], device)
    dataset_config = dataset_config_from_checkpoint(first_config)
    dataset = SynchronizedAVDataset(manifest=args.manifest, split=args.split, config=dataset_config)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    first_batch = next(iter(loader))
    first_video_shape, first_audio_shape = infer_latent_shapes(
        model=first_model,
        batch=first_batch,
        device=device,
    )
    del first_model

    print(f"device: {device}")
    print(f"{args.split} windows: {len(dataset)}")
    if args.batch_size < 2 and any("shuffled" in condition for condition in args.conditions):
        print("warning: shuffled ablations need --batch-size >= 2 to change cross-sample pairing")

    rows: list[dict[str, str]] = []
    for model_name, checkpoint_path in checkpoint_specs:
        model, config = build_model_from_checkpoint(checkpoint_path, device)
        video_latent_shape, audio_latent_shape = infer_latent_shapes(
            model=model,
            batch=first_batch,
            device=device,
        )
        video_edge_weight = float(config.get("video_edge_weight", 0.1))
        audio_loss_weight = float(config.get("audio_loss_weight", 1.0))
        video_bottleneck = str(config.get("video_bottleneck", "fsq"))
        audio_bottleneck = str(config.get("audio_bottleneck", "ae"))
        freeze_backbones = bool(config.get("freeze_backbones", False))

        if not video_latent_shape:
            video_latent_shape = first_video_shape
        if not audio_latent_shape:
            audio_latent_shape = first_audio_shape

        metrics_by_condition: dict[str, dict[str, float]] = {}
        for condition in args.conditions:
            metrics_by_condition[condition] = evaluate_condition(
                model=model,
                loader=loader,
                device=device,
                condition=condition,
                video_edge_weight=video_edge_weight,
                audio_loss_weight=audio_loss_weight,
                max_batches=args.max_batches,
                seed=args.seed,
            )

        normal_metrics = metrics_by_condition.get("normal")
        if normal_metrics is None:
            normal_metrics = next(iter(metrics_by_condition.values()))

        for condition, metrics in metrics_by_condition.items():
            row = {
                "model": model_name,
                "condition": condition,
                "freeze_backbones": str(freeze_backbones).lower(),
                "video_bottleneck": video_bottleneck,
                "audio_bottleneck": audio_bottleneck,
                "video_latent_shape": video_latent_shape,
                "audio_latent_shape": audio_latent_shape,
                "val_loss": f"{metrics['loss']:.4f}",
                "delta_loss": f"{metrics['loss'] - normal_metrics['loss']:+.4f}",
                "video_l1": f"{metrics['video_l1']:.4f}",
                "delta_video_l1": f"{metrics['video_l1'] - normal_metrics['video_l1']:+.4f}",
                "video_psnr": f"{metrics['video_psnr']:.2f}",
                "audio_l1": f"{metrics['audio_l1']:.4f}",
                "delta_audio_l1": f"{metrics['audio_l1'] - normal_metrics['audio_l1']:+.4f}",
                "audio_psnr": f"{metrics['audio_psnr']:.2f}",
            }
            rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "model",
        "condition",
        "freeze_backbones",
        "video_bottleneck",
        "audio_bottleneck",
        "video_latent_shape",
        "audio_latent_shape",
        "val_loss",
        "delta_loss",
        "video_l1",
        "delta_video_l1",
        "video_psnr",
        "audio_l1",
        "delta_audio_l1",
        "audio_psnr",
    ]
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
