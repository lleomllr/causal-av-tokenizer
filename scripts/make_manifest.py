from __future__ import annotations
import argparse
import csv
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import imageio_ffmpeg
from avtokenizer.data import VIDEO_EXT, list_video_files, window_starts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a CSV manifest with video durations.")
    parser.add_argument("--root", type=str, default="data/raw")
    parser.add_argument("--output", type=str, default="data/manifest.csv")
    parser.add_argument("--val-ratio", type=float, default=0.33)
    parser.add_argument("--clip-seconds", type=float, default=2.0)
    parser.add_argument("--stride-seconds", type=float, default=0.5)
    parser.add_argument("--window-level", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output = Path(args.output)
    videos = list_video_files(root, VIDEO_EXT)
    if not videos:
        raise ValueError(f"No videos found in {root.resolve()}")

    output.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    with output.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = ["path", "duration_sec", "split"]
        if args.window_level:
            fieldnames = ["path", "start_sec", "end_sec", "duration_sec", "split"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        train_count = 0
        val_count = 0
        for path in videos:
            _, duration_sec = imageio_ffmpeg.count_frames_and_secs(str(path))
            manifest_path = path.resolve()
            try:
                manifest_path = manifest_path.relative_to(output.parent.resolve())
            except ValueError:
                pass

            if args.window_level:
                starts = window_starts(duration_sec, args.clip_seconds, args.stride_seconds)
                for start_sec in starts:
                    split = "val" if rng.random() < args.val_ratio else "train"
                    if split == "val":
                        val_count += 1
                    else:
                        train_count += 1
                    writer.writerow(
                        {
                            "path": manifest_path.as_posix(),
                            "start_sec": f"{start_sec:.3f}",
                            "end_sec": f"{start_sec + args.clip_seconds:.3f}",
                            "duration_sec": f"{duration_sec:.3f}",
                            "split": split,
                        }
                    )
            else:
                split = "val" if rng.random() < args.val_ratio else "train"
                if split == "val":
                    val_count += 1
                else:
                    train_count += 1
                writer.writerow(
                    {
                        "path": manifest_path.as_posix(),
                        "duration_sec": f"{duration_sec:.3f}",
                        "split": split,
                    }
                )

    print(f"wrote {output}")
    print(f"videos: {len(videos)} | train rows: {train_count} | val rows: {val_count}")


if __name__ == "__main__":
    main()
