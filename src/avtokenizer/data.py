from __future__ import annotations
import csv 
import math 
import subprocess
from dataclasses import dataclass 
from pathlib import Path 
from typing import Iterable, Sequence 

try:
    import torch 
    import torch.nn.functional as F
    from torch.utils.data import Dataset 
except ModuleNotFoundError:
    torch = None 
    F = None 
    Dataset = object 
    
VIDEO_EXT = (".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v")


@dataclass(frozen=True)
class AVWindowConfig:
    clip_seconds: float = 2.0
    stride_seconds: float = 1.0
    fps: int = 12
    image_size: int = 128
    sample_rate: int = 16_000
    n_mels: int = 80
    audio_steps_per_frame: int = 4
    mel_win_length: int = 400
    mel_hop_length: int = 160 
    include_waveform: bool = False 
    extensions: tuple[str, ...] = VIDEO_EXT
    
    @property
    def num_frames(self) -> int: 
        return max(1, round(self.clip_seconds * self.fps))
    
    @property
    def num_audio_samples(self) -> int: 
        return max(1, round(self.clip_seconds * self.sample_rate))
    
    
@dataclass(frozen=True)
class AVWindow: 
    path: Path  
    start_sec: float 
    end_sec: float 
    split: str | None = None 
    
def list_video_files(root: str | Path, extensions: Sequence[str] = VIDEO_EXT) -> list[Path]:
    root = Path(root)
    suffixes = {ext.lower() for ext in extensions}
    return sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)

def window_starts(
    duration_sec: float | None, 
    clip_seconds: float, 
    stride_seconds: float, 
    *, 
    include_tail: bool = False, 
) -> list[float]:
    if clip_seconds <= 0: 
        raise ValueError("clip_seconds must be positive")
    if stride_seconds <= 0: 
        raise ValueError("stride_seconds must be positive")
    if duration_sec is None or duration_sec <= clip_seconds:
        return [0.0]
    
    last_start = max(0.0, duration_sec - clip_seconds)
    count = int(math.floor(last_start / stride_seconds)) + 1
    starts = [round(i * stride_seconds, 6) for i in range(count)]
    
    if include_tail and starts[-1] < last_start:
        starts.append(round(last_start, 6))
    return starts


def temporal_sample_indices(num_source_frames: int, num_target_frames: int) -> list[int]:
    if num_source_frames <= 0:
        raise ValueError("num_source_frames must be positive")
    if num_target_frames <= 0:
        raise ValueError("num_target_frames must be positive")
    if num_target_frames == 1:
        return [0]
    if num_source_frames == 1:
        return [0 for _ in range(num_target_frames)]

    scale = (num_source_frames - 1) / (num_target_frames - 1)
    return [min(num_source_frames - 1, round(i * scale)) for i in range(num_target_frames)]


def build_window_index(
    *,
    root: str | Path | None = None,
    manifest: str | Path | None = None,
    config: AVWindowConfig | None = None,
    split: str | None = None,
    include_tail: bool = False,
) -> list[AVWindow]:
    config = config or AVWindowConfig()
    rows: list[tuple[Path, float | None, str | None]] = []
    
    if manifest is not None: 
        manifest_path = Path(manifest)
        with manifest_path.open(newline="") as handle: 
            reader = csv.DictReader(handle)
            if "path" not in (reader.fieldnames or []):
                raise ValueError("Manifest must contain a path column")
            for row in reader: 
                row_split = row.get("split") or None
                if split is not None and row_split != split: 
                    continue 
                
                path = Path(row["path"])
                if not path.is_absolute():
                    path = manifest_path.parent / path
                duration = float(row["duration_sec"]) if row.get("duration_sec") else None
                rows.append((path, duration, row_split))
    if root is not None: 
        rows.extend((path, None, None) for path in list_video_files(root, config.extensions))
         
    if not rows:
        hints = ["No videos found."]
        if root is not None:
            hints.append(f"Scanned root: {Path(root).resolve()}")
        if manifest is not None:
            hints.append(f"Manifest: {Path(manifest).resolve()}")
        hints.append(f"Expected extensions: {', '.join(config.extensions)}")
        hints.append("Provide --root with video files, or --manifest with a path column.")
        raise ValueError(" ".join(hints))
      
    windows: list[AVWindow] = []
    for path, duration, row_split in rows: 
        for start in window_starts(
            duration, 
            config.clip_seconds, 
            config.stride_seconds, 
            include_tail=include_tail
        ):
            windows.append(
                AVWindow(
                    path= path, 
                    start_sec = start, 
                    end_sec = start + config.clip_seconds, 
                    split = row_split
                )
            )
    return windows


class SynchronizedAVDataset(Dataset):
    def __init__(
        self, 
        *,
        root: str | Path | None = None,
        manifest: str | Path | None = None,
        config: AVWindowConfig | None = None,
        split: str | None = None,
        include_tail: bool = False,
    ) -> None:
        _require_torch()
        self.config = config or AVWindowConfig()
        self.windows = build_window_index(
            root=root,
            manifest=manifest,
            config=self.config,
            split=split,
            include_tail=include_tail,
        )
        
    def __len__(self) -> int: 
        return len(self.windows)
    
    def __getitem__(self, index: int) -> dict[str, object]:
        window = self.windows[index]
        video, waveform, native_audio_rate, native_video_fps = _read_av_window(window, self.config)
        
        video = _prepare_video(video, self.config)
        waveform = _prepare_waveform(waveform, native_audio_rate, self.config)
        audio_mel = _waveform_to_frame_aligned_mel(waveform, self.config)
        
        frame_times = torch.linspace(
            window.start_sec, 
            window.end_sec, 
            steps = self.config.num_frames + 1, 
            dtype = torch.float32
        )[:-1]
        
        sample: dict[str, object] = {
            "video": video,
            "audio_mel": audio_mel,
            "frame_times": frame_times,
            "path": str(window.path),
            "start_sec": window.start_sec,
            "end_sec": window.end_sec,
            "native_video_fps": native_video_fps,
            "native_audio_rate": native_audio_rate,
        }
        if self.config.include_waveform:
            sample["audio_waveform"] = waveform
        return sample
    
    
def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "SynchronizedAVDataset requires torch, torchvision, and torchaudio"
            "Install this project with: pip install -e '.[dev]'"
        )
        

def _read_av_window(window: AVWindow, config: AVWindowConfig):
    _require_torch()
    try: 
        from torchvision.io import read_video
    except (ImportError, ModuleNotFoundError):
        read_video = None

    if read_video is None:
        return _read_av_window_with_ffmpeg(window, config)
    
    video, audio, info = read_video(
        str(window.path), 
        start_pts = window.start_sec, 
        end_pts = window.end_sec, 
        pts_unit = "sec", 
        output_format = "TCHW"
    )
    
    if video.numel() == 0: 
        raise RuntimeError(f"No video frames decoded from {window.path}")
    
    native_audio_rate = int(info.get("audio_fps") or 0)
    native_video_fps = float(info.get("video_fps") or 0.0)
    
    if audio.numel() == 0: 
        channels = 1 
        samples = max(1, round((window.end_sec - window.start_sec) * config.sample_rate))
        audio = torch.zeros(channels, samples, dtype=torch.float32)
        native_audio_rate = config.sample_rate
    return video, audio, native_audio_rate, native_video_fps


def _read_av_window_with_ffmpeg(window: AVWindow, config: AVWindowConfig):
    try:
        import imageio_ffmpeg
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "This torchvision version does not expose torchvision.io.read_video. "
            "Install the FFmpeg fallback with: pip install imageio-ffmpeg"
        ) from exc

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    duration = max(0.001, window.end_sec - window.start_sec)
    reader = imageio_ffmpeg.read_frames(
        str(window.path),
        pix_fmt="rgb24",
        input_params=["-ss", f"{window.start_sec:.6f}"],
        output_params=["-t", f"{duration:.6f}"],
    )
    try:
        metadata = next(reader)
        width, height = metadata["size"]
        native_video_fps = float(metadata.get("fps") or 0.0)
        frames = list(reader)
    finally:
        reader.close()

    frame_size = width * height * 3
    if frame_size <= 0 or not frames:
        raise RuntimeError(f"No video frames decoded from {window.path}")

    video_bytes = b"".join(frames)
    num_frames = len(video_bytes) // frame_size
    video = torch.frombuffer(bytearray(video_bytes), dtype=torch.uint8)
    video = video[: num_frames * frame_size].reshape(num_frames, height, width, 3)
    video = video.permute(0, 3, 1, 2).contiguous()

    audio_cmd = [
        ffmpeg,
        "-v",
        "error",
        "-ss",
        f"{window.start_sec:.6f}",
        "-t",
        f"{duration:.6f}",
        "-i",
        str(window.path),
        "-f",
        "f32le",
        "-ac",
        "1",
        "-ar",
        str(config.sample_rate),
        "pipe:1",
    ]
    try:
        audio_bytes = subprocess.check_output(audio_cmd, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        audio_bytes = b""

    if audio_bytes:
        audio = torch.frombuffer(bytearray(audio_bytes), dtype=torch.float32).unsqueeze(0)
    else:
        samples = max(1, round(duration * config.sample_rate))
        audio = torch.zeros(1, samples, dtype=torch.float32)

    return video, audio, config.sample_rate, native_video_fps


def _prepare_video(video, config: AVWindowConfig):
    video = video.float() / 255.0
    indices = temporal_sample_indices(video.shape[0], config.num_frames)
    video = video[torch.tensor(indices, dtype=torch.long)]
    
    if video.shape[-2:] != (config.image_size, config.image_size):
        video = F.interpolate(
            video, 
            size = (config.image_size, config.image_size), 
            mode = "bilinear", 
            align_corners=False
        )
    return video.clamp(0.0, 1.0)

def _prepare_waveform(waveform, native_audio_rate: int, config: AVWindowConfig):
    try:
        import torchaudio
    except ModuleNotFoundError as exc:
        raise RuntimeError("torchaudio is required to process audio") from exc

    waveform = waveform.float()
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if native_audio_rate <= 0:
        native_audio_rate = config.sample_rate
    if native_audio_rate != config.sample_rate:
        waveform = torchaudio.functional.resample(waveform, native_audio_rate, config.sample_rate)

    target = config.num_audio_samples
    if waveform.shape[-1] < target:
        waveform = F.pad(waveform, (0, target - waveform.shape[-1]))
    else:
        waveform = waveform[..., :target]
    return waveform


def _waveform_to_frame_aligned_mel(waveform, config: AVWindowConfig):
    try:
        import torchaudio
    except ModuleNotFoundError as exc:
        raise RuntimeError("torchaudio is required to compute mel features") from exc

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.sample_rate,
        n_fft=config.mel_win_length,
        win_length=config.mel_win_length,
        hop_length=config.mel_hop_length,
        n_mels=config.n_mels,
        center=False,
    )(waveform)
    mel = torch.log1p(mel).mean(dim=0, keepdim=True)

    target_steps = config.num_frames * config.audio_steps_per_frame
    mel = F.interpolate(mel, size=target_steps, mode="linear", align_corners=False)
    mel = mel.squeeze(0)
    mel = mel.reshape(config.n_mels, config.num_frames, config.audio_steps_per_frame)
    return mel.permute(1, 0, 2).contiguous()


def paths_from_windows(windows: Iterable[AVWindow]) -> list[str]:
    return [str(window.path) for window in windows]
