from __future__ import annotations
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import imageio_ffmpeg
from avtokenizer.data import VIDEO_EXT, list_video_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a CSV manifest with video durations.")
    parser.add_argument("--root", type=str, default="data/raw")
    parser.add_argument("--output", type=str, default="data/manifest.csv")
    parser.add_argument("--val-ratio", type=float, default=0.33)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root)
    output = Path(args.output)
    videos = list_video_files(root, VIDEO_EXT)
    if not videos:
        raise ValueError(f"No videos found in {root.resolve()}")

    output.parent.mkdir(parents=True, exist_ok=True)
    val_count = max(1, round(len(videos) * args.val_ratio)) if len(videos) > 1 else 0
    val_paths = set(videos[-val_count:]) if val_count else set()

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "duration_sec", "split"])
        writer.writeheader()
        for path in videos:
            _, duration_sec = imageio_ffmpeg.count_frames_and_secs(str(path))
            manifest_path = path.resolve()
            try:
                manifest_path = manifest_path.relative_to(output.parent.resolve())
            except ValueError:
                pass
            writer.writerow(
                {
                    "path": manifest_path.as_posix(),
                    "duration_sec": f"{duration_sec:.3f}",
                    "split": "val" if path in val_paths else "train",
                }
            )

    print(f"wrote {output}")
    print(f"videos: {len(videos)} | train: {len(videos) - len(val_paths)} | val: {len(val_paths)}")


if __name__ == "__main__":
    main()
