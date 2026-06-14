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
from avtokenizer.av_autoencoder import AudioVisualAutoencoder, frame_aligned_mel_to_image
from avtokenizer.video_autoencoder import edge_loss, psnr_from_l1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight joint audio-video codec.")
    parser.add_argument("--manifest", type=str, default="data/manifest.csv")
    parser.add_argument("--output-dir", type=str, default="outputs/av_joint")
    parser.add_argument("--video-checkpoint", type=str, default=None)
    parser.add_argument("--audio-checkpoint", type=str, default=None)
    parser.add_argument(
        "--freeze-backbones",
        action="store_true",
        help="Freeze pretrained audio/video autoencoders and train only latent fusion layers.",
    )
    parser.add_argument("--clip-seconds", type=float, default=2.0)
    parser.add_argument("--stride-seconds", type=float, default=0.5)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--audio-steps-per-frame", type=int, default=4)
    parser.add_argument("--video-latent-channels", type=int, default=128)
    parser.add_argument("--video-bottleneck", type=str, default="fsq", choices=["ae", "vae", "fsq"])
    parser.add_argument("--video-fsq-levels", type=str, default="8,8,8,8,8,8,8,8,8,8,8,8,8,8,8,8")
    parser.add_argument("--audio-latent-channels", type=int, default=64)
    parser.add_argument("--audio-bottleneck", type=str, default="ae", choices=["ae", "fsq"])
    parser.add_argument("--audio-fsq-levels", type=str, default="8,8,8,8,8,8,8,8")
    parser.add_argument("--video-edge-weight", type=float, default=0.1)
    parser.add_argument("--audio-loss-weight", type=float, default=1.0)
    parser.add_argument(
        "--cross-loss-weight",
        type=float,
        default=0.0,
        help="Weight for explicit cross-modal reconstruction losses.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=5e-4)
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


def parse_levels(levels: str) -> tuple[int, ...]:
    parsed = tuple(int(part.strip()) for part in levels.split(",") if part.strip())
    if not parsed:
        raise ValueError("FSQ levels must contain at least one integer")
    return parsed


def load_component_config(path: str | None) -> dict[str, object]:
    if path is None:
        return {}
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return checkpoint.get("config", {})


def parse_checkpoint_levels(config: dict[str, object], key: str, fallback: str) -> tuple[int, ...]:
    value = config.get(key, fallback)
    if isinstance(value, str):
        return parse_levels(value)
    if isinstance(value, (tuple, list)):
        return tuple(int(item) for item in value)
    return parse_levels(fallback)


def build_model_from_args(args: argparse.Namespace, device: torch.device) -> AudioVisualAutoencoder:
    video_config = load_component_config(args.video_checkpoint)
    audio_config = load_component_config(args.audio_checkpoint)
    model = AudioVisualAutoencoder(
        video_latent_channels=int(video_config.get("latent_channels", args.video_latent_channels)),
        video_bottleneck=str(video_config.get("bottleneck", args.video_bottleneck)),
        video_fsq_levels=parse_checkpoint_levels(
            video_config,
            "fsq_levels",
            args.video_fsq_levels,
        ),
        audio_latent_channels=int(audio_config.get("latent_channels", args.audio_latent_channels)),
        audio_bottleneck=str(audio_config.get("bottleneck", args.audio_bottleneck)),
        audio_fsq_levels=parse_checkpoint_levels(
            audio_config,
            "fsq_levels",
            args.audio_fsq_levels,
        ),
    ).to(device)
    if args.video_checkpoint:
        checkpoint = torch.load(args.video_checkpoint, map_location=device, weights_only=False)
        model.video_model.load_state_dict(checkpoint["model"])
    if args.audio_checkpoint:
        checkpoint = torch.load(args.audio_checkpoint, map_location=device, weights_only=False)
        model.audio_model.load_state_dict(checkpoint["model"])
    return model


def freeze_backbone_parameters(model: AudioVisualAutoencoder) -> None:
    for module in (model.video_model, model.audio_model):
        for parameter in module.parameters():
            parameter.requires_grad = False


def trainable_parameters(model: AudioVisualAutoencoder) -> list[torch.nn.Parameter]:
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def count_parameters(model: AudioVisualAutoencoder) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return total, trainable


def compute_losses(
    *,
    model: AudioVisualAutoencoder,
    batch: dict[str, object],
    device: torch.device,
    video_edge_weight: float,
    audio_loss_weight: float,
    cross_loss_weight: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    video = batch["video"].to(device)
    audio_mel = batch["audio_mel"].to(device)
    recon = model(video, audio_mel)

    batch_size, num_frames, channels, height, width = video.shape
    video_frames = video.reshape(batch_size * num_frames, channels, height, width)
    recon_frames = recon["video"].reshape(batch_size * num_frames, channels, height, width)

    video_l1 = F.l1_loss(recon_frames, video_frames)
    video_edges = edge_loss(recon_frames, video_frames)
    audio_l1 = F.l1_loss(recon["audio_mel"], audio_mel)
    normal_loss = video_l1 + video_edge_weight * video_edges + audio_loss_weight * audio_l1

    video_from_audio_l1 = video_l1.new_tensor(0.0)
    audio_from_video_l1 = video_l1.new_tensor(0.0)
    cross_loss = video_l1.new_tensor(0.0)
    if cross_loss_weight > 0.0:
        video_from_audio = model(torch.zeros_like(video), audio_mel)
        video_from_audio_frames = video_from_audio["video"].reshape(batch_size * num_frames, channels, height, width)
        video_from_audio_l1 = F.l1_loss(video_from_audio_frames, video_frames)

        audio_from_video = model(video, torch.zeros_like(audio_mel))
        audio_from_video_l1 = F.l1_loss(audio_from_video["audio_mel"], audio_mel)
        cross_loss = video_from_audio_l1 + audio_loss_weight * audio_from_video_l1

    total = normal_loss + cross_loss_weight * cross_loss
    metrics = {
        "loss": float(total.detach().cpu()),
        "normal_loss": float(normal_loss.detach().cpu()),
        "cross_loss": float(cross_loss.detach().cpu()),
        "video_l1": float(video_l1.detach().cpu()),
        "video_edge": float(video_edges.detach().cpu()),
        "video_psnr": float(psnr_from_l1(video_l1.detach()).cpu()),
        "audio_l1": float(audio_l1.detach().cpu()),
        "audio_psnr": float(psnr_from_l1(audio_l1.detach()).cpu()),
        "video_from_audio_l1": float(video_from_audio_l1.detach().cpu()),
        "video_from_audio_psnr": float(psnr_from_l1(video_from_audio_l1.detach()).cpu()),
        "audio_from_video_l1": float(audio_from_video_l1.detach().cpu()),
        "audio_from_video_psnr": float(psnr_from_l1(audio_from_video_l1.detach()).cpu()),
    }
    return total, metrics


def run_epoch(
    *,
    model: AudioVisualAutoencoder,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    video_edge_weight: float,
    audio_loss_weight: float,
    cross_loss_weight: float,
    max_batches: int,
    freeze_backbones: bool,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    if is_train and freeze_backbones:
        model.video_model.eval()
        model.audio_model.eval()
    totals = {
        "loss": 0.0,
        "normal_loss": 0.0,
        "cross_loss": 0.0,
        "video_l1": 0.0,
        "video_edge": 0.0,
        "video_psnr": 0.0,
        "audio_l1": 0.0,
        "audio_psnr": 0.0,
        "video_from_audio_l1": 0.0,
        "video_from_audio_psnr": 0.0,
        "audio_from_video_l1": 0.0,
        "audio_from_video_psnr": 0.0,
    }
    count = 0
    for step, batch in enumerate(loader, start=1):
        with torch.set_grad_enabled(is_train):
            loss, metrics = compute_losses(
                model=model,
                batch=batch,
                device=device,
                video_edge_weight=video_edge_weight,
                audio_loss_weight=audio_loss_weight,
                cross_loss_weight=cross_loss_weight,
            )
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        for key, value in metrics.items():
            totals[key] += value
        count += 1
        if max_batches and step >= max_batches:
            break
    return {key: value / count for key, value in totals.items()}


@torch.no_grad()
def save_joint_preview(
    *,
    model: AudioVisualAutoencoder,
    loader: DataLoader,
    device: torch.device,
    output_dir: Path,
    max_frames: int = 8,
    save_cross_previews: bool = False,
) -> None:
    model.eval()
    batch = next(iter(loader))
    video = batch["video"].to(device)
    audio_mel = batch["audio_mel"].to(device)
    recon = model(video, audio_mel)

    frames = video[0, :max_frames]
    recon_frames = recon["video"][0, :max_frames].clamp(0.0, 1.0)
    frame_error = (frames - recon_frames).abs().mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    save_image_rows(
        rows=[
            ("video original", frames),
            ("video reconstruction", recon_frames),
            ("video absolute error", frame_error),
        ],
        output=output_dir / "video_reconstructions.png",
    )

    mel = frame_aligned_mel_to_image(audio_mel[:4])
    recon_mel = frame_aligned_mel_to_image(recon["audio_mel"][:4]).clamp_min(0.0)
    mel_error = (mel - recon_mel).abs()
    save_mel_rows(
        rows=[
            ("audio original", mel),
            ("audio reconstruction", recon_mel),
            ("audio absolute error", mel_error),
        ],
        output=output_dir / "audio_reconstructions.png",
    )

    if not save_cross_previews:
        return

    video_from_audio = model(torch.zeros_like(video), audio_mel)
    cross_recon_frames = video_from_audio["video"][0, :max_frames].clamp(0.0, 1.0)
    cross_frame_error = (frames - cross_recon_frames).abs().mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    save_image_rows(
        rows=[
            ("video target", frames),
            ("video from audio", cross_recon_frames),
            ("video cross error", cross_frame_error),
        ],
        output=output_dir / "video_from_audio.png",
    )

    audio_from_video = model(video, torch.zeros_like(audio_mel))
    cross_recon_mel = frame_aligned_mel_to_image(audio_from_video["audio_mel"][:4]).clamp_min(0.0)
    cross_mel_error = (mel - cross_recon_mel).abs()
    save_mel_rows(
        rows=[
            ("audio target", mel),
            ("audio from video", cross_recon_mel),
            ("audio cross error", cross_mel_error),
        ],
        output=output_dir / "audio_from_video.png",
    )


def save_image_rows(*, rows: list[tuple[str, torch.Tensor]], output: Path) -> None:
    num_cols = rows[0][1].shape[0]
    fig, axes = plt.subplots(nrows=len(rows), ncols=num_cols, figsize=(num_cols * 1.5, len(rows) * 1.7), squeeze=False)
    for row_index, (label, tensor) in enumerate(rows):
        images = tensor.detach().cpu().permute(0, 2, 3, 1).numpy()
        for col_index, image in enumerate(images):
            axes[row_index, col_index].imshow(image)
            axes[row_index, col_index].set_axis_off()
            if col_index == 0:
                axes[row_index, col_index].set_ylabel(label)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def save_mel_rows(*, rows: list[tuple[str, torch.Tensor]], output: Path) -> None:
    num_cols = rows[0][1].shape[0]
    fig, axes = plt.subplots(nrows=len(rows), ncols=num_cols, figsize=(num_cols * 3.2, len(rows) * 2.0), squeeze=False)
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
        image_size=args.image_size,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        audio_steps_per_frame=args.audio_steps_per_frame,
        include_video=True,
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

    model = build_model_from_args(args, device)
    if args.freeze_backbones:
        freeze_backbone_parameters(model)
    parameters = trainable_parameters(model)
    if not parameters:
        raise ValueError("No trainable parameters left. Disable --freeze-backbones or check the model definition.")
    optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    total_parameters, trainable_parameter_count = count_parameters(model)

    print(f"device: {device}")
    print(f"train windows: {len(train_dataset)} | val windows: {len(val_dataset)}")
    print(f"video checkpoint: {args.video_checkpoint}")
    print(f"audio checkpoint: {args.audio_checkpoint}")
    print(f"freeze backbones: {args.freeze_backbones}")
    print(f"cross loss weight: {args.cross_loss_weight}")
    print(f"trainable parameters: {trainable_parameter_count:,} / {total_parameters:,}")

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            video_edge_weight=args.video_edge_weight,
            audio_loss_weight=args.audio_loss_weight,
            cross_loss_weight=args.cross_loss_weight,
            max_batches=args.max_train_batches,
            freeze_backbones=args.freeze_backbones,
        )
        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            device=device,
            optimizer=None,
            video_edge_weight=args.video_edge_weight,
            audio_loss_weight=args.audio_loss_weight,
            cross_loss_weight=args.cross_loss_weight,
            max_batches=args.max_val_batches,
            freeze_backbones=args.freeze_backbones,
        )
        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f} "
            f"video L1 {train_metrics['video_l1']:.4f} PSNR {train_metrics['video_psnr']:.2f} "
            f"audio L1 {train_metrics['audio_l1']:.4f} PSNR {train_metrics['audio_psnr']:.2f} | "
            f"val loss {val_metrics['loss']:.4f} "
            f"video L1 {val_metrics['video_l1']:.4f} PSNR {val_metrics['video_psnr']:.2f} "
            f"audio L1 {val_metrics['audio_l1']:.4f} PSNR {val_metrics['audio_psnr']:.2f} "
            f"cross V<-A {val_metrics['video_from_audio_l1']:.4f} "
            f"A<-V {val_metrics['audio_from_video_l1']:.4f}"
        )
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "config": vars(args),
                    "best_val_loss": best_val,
                    "best_val_metrics": val_metrics,
                },
                output_dir / "best.pt",
            )
            save_joint_preview(
                model=model,
                loader=val_loader,
                device=device,
                output_dir=output_dir,
                save_cross_previews=args.cross_loss_weight > 0.0,
            )

    print(f"saved checkpoint: {output_dir / 'best.pt'}")
    print(f"saved video preview: {output_dir / 'video_reconstructions.png'}")
    print(f"saved audio preview: {output_dir / 'audio_reconstructions.png'}")
    if args.cross_loss_weight > 0.0:
        print(f"saved video-from-audio preview: {output_dir / 'video_from_audio.png'}")
        print(f"saved audio-from-video preview: {output_dir / 'audio_from_video.png'}")


if __name__ == "__main__":
    main()
