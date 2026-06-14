from __future__ import annotations
import argparse
from torch.utils.data import DataLoader
from avtokenizer import AVWindowConfig, SynchronizedAVDataset

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect synchronized audio-video batches.")
    parser.add_argument("--root", type=str, default=None, help="Directory containing video files.")
    parser.add_argument("--manifest", type=str, default=None, help="CSV with path,duration_sec,split columns.")
    parser.add_argument("--split", type=str, default=None)
    parser.add_argument("--clip-seconds", type=float, default=2.0)
    parser.add_argument("--stride-seconds", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--n-mels", type=int, default=80)
    parser.add_argument("--audio-steps-per-frame", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()

def main() -> None:
    arg = parse_args()
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
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    batch = next(iter(loader))
    print(f"dataset windows: {len(dataset)}")
    print(f"video: {tuple(batch['video'].shape)}")
    print(f"audio_mel: {tuple(batch['audio_mel'].shape)}")
    print(f"frame_times: {tuple(batch['frame_times'].shape)}")
    print(f"first path: {batch['path'][0]}")
    print(f"first window: {batch['start_sec'][0]:.3f}s -> {batch['end_sec'][0]:.3f}s")


if __name__ == "__main__":
    main()