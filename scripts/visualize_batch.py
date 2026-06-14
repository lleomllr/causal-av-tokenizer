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
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from avtokenizer import AVWindowConfig, SynchronizedAVDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Save a visual QA grid for synchronized audio-video batches."
    )
    parser.add_argument("--root", type=str, default=None, help="Directory containing video files.")
    parser.add_argument("--manifest", type=str, default=None, help="CSV with path,duration_sec,split columns.")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--output", type=str, default="outputs/av_batch_preview.png")
    parser.add_argument("--clip-seconds", type=float, default=2.0)
    parser.add_argument("--stride-seconds", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--audio-steps-per-frame", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-frame-thumbs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-shuffle", action="store_true")
    return parser.parse_args()


def select_frame_indices(num_frames: int, max_frame_thumbs: int) -> list[int]:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    count = min(num_frames, max(1, max_frame_thumbs))
    if count == 1:
        return [0]
    return np.linspace(0, num_frames - 1, count).round().astype(int).tolist()


def frame_to_numpy(frame: torch.Tensor) -> np.ndarray:
    image = frame.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(image, 0.0, 1.0)


def make_contact_sheet(video: torch.Tensor, indices: list[int]) -> np.ndarray:
    frames = [frame_to_numpy(video[index]) for index in indices]
    return np.concatenate(frames, axis=1)


def flatten_mel(mel: torch.Tensor) -> np.ndarray:
    # [T, n_mels, K] -> [n_mels, T * K]
    mel_np = mel.detach().cpu().numpy()
    mel_np = np.transpose(mel_np, (1, 0, 2)).reshape(mel_np.shape[1], -1)
    return mel_np


def validate_alignment_shapes(batch: dict[str, object]) -> None:
    video = batch["video"]
    audio_mel = batch["audio_mel"]
    frame_times = batch["frame_times"]

    if video.ndim != 5:
        raise ValueError(f"Expected video [B, T, C, H, W], got {tuple(video.shape)}")
    if audio_mel.ndim != 4:
        raise ValueError(f"Expected audio_mel [B, T, n_mels, K], got {tuple(audio_mel.shape)}")
    if frame_times.ndim != 2:
        raise ValueError(f"Expected frame_times [B, T], got {tuple(frame_times.shape)}")
    if video.shape[0] != audio_mel.shape[0] or video.shape[0] != frame_times.shape[0]:
        raise ValueError("Batch dimensions do not match between video, audio_mel, and frame_times")
    if video.shape[1] != audio_mel.shape[1] or video.shape[1] != frame_times.shape[1]:
        raise ValueError("Temporal dimensions do not match between video, audio_mel, and frame_times")


def plot_batch(batch: dict[str, object], output: Path, max_frame_thumbs: int) -> None:
    validate_alignment_shapes(batch)

    video = batch["video"]
    audio_mel = batch["audio_mel"]
    frame_times = batch["frame_times"]
    batch_size, num_frames = video.shape[:2]
    frame_indices = select_frame_indices(num_frames, max_frame_thumbs)
    audio_steps_per_frame = audio_mel.shape[-1]

    fig, axes = plt.subplots(
        nrows=batch_size,
        ncols=2,
        figsize=(14, max(3.4, batch_size * 3.1)),
        gridspec_kw={"width_ratios": [1.6, 1.4]},
        squeeze=False,
    )

    for row in range(batch_size):
        sheet = make_contact_sheet(video[row], frame_indices)
        mel = flatten_mel(audio_mel[row])
        start_sec = float(batch["start_sec"][row])
        end_sec = float(batch["end_sec"][row])
        path = Path(batch["path"][row])

        video_ax = axes[row, 0]
        mel_ax = axes[row, 1]

        video_ax.imshow(sheet)
        video_ax.set_axis_off()
        video_ax.set_title(
            f"{path.name} | {start_sec:.2f}s -> {end_sec:.2f}s",
            fontsize=9,
        )
        thumb_width = sheet.shape[1] / len(frame_indices)
        for thumb_index, frame_index in enumerate(frame_indices):
            time_sec = float(frame_times[row, frame_index])
            video_ax.text(
                thumb_index * thumb_width + 4,
                12,
                f"{frame_index} | {time_sec:.2f}s",
                color="white",
                fontsize=7,
                ha="left",
                va="top",
                bbox={"facecolor": "black", "alpha": 0.45, "pad": 1, "edgecolor": "none"},
            )

        mel_ax.imshow(mel, origin="lower", aspect="auto", interpolation="nearest", cmap="magma")
        mel_ax.set_title("Frame-aligned log-mel spectrogram", fontsize=9)
        mel_ax.set_xlabel("mel time steps")
        mel_ax.set_ylabel("mel bins")

        for frame_index in frame_indices:
            x = frame_index * audio_steps_per_frame
            mel_ax.axvline(x=x, color="cyan", linewidth=0.8, alpha=0.75)
            time_sec = float(frame_times[row, frame_index])
            mel_ax.text(
                x,
                mel.shape[0] - 1,
                f"{frame_index}\n{time_sec:.2f}s",
                color="white",
                fontsize=6,
                ha="center",
                va="top",
                bbox={"facecolor": "black", "alpha": 0.35, "pad": 1, "edgecolor": "none"},
            )

    fig.suptitle(
        "Audio-video batch QA: selected frames and matching mel positions",
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    config = AVWindowConfig(
        clip_seconds=args.clip_seconds,
        stride_seconds=args.stride_seconds,
        fps=args.fps,
        image_size=args.image_size,
        sample_rate=args.sample_rate,
        n_mels=args.n_mels,
        audio_steps_per_frame=args.audio_steps_per_frame,
    )
    dataset = SynchronizedAVDataset(
        root=args.root,
        manifest=args.manifest,
        split=args.split,
        config=config,
    )
    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=not args.no_shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
    )

    batch = next(iter(loader))
    output = Path(args.output)
    plot_batch(batch, output, args.max_frame_thumbs)

    video_shape = tuple(batch["video"].shape)
    mel_shape = tuple(batch["audio_mel"].shape)
    print(f"saved: {output}")
    print(f"video: {video_shape}")
    print(f"audio_mel: {mel_shape}")
    print(
        "alignment structure: OK "
        f"({video_shape[1]} video frames, {mel_shape[-1]} mel steps per frame)"
    )
    print("visual check: cyan lines mark the mel positions corresponding to the displayed frames.")


if __name__ == "__main__":
    main()
