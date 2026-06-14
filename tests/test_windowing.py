import pytest
from avtokenizer.data import AVWindowConfig, build_window_index, temporal_sample_indices, window_starts

def test_window_starts_without_duration_returns_single_window():
    assert window_starts(None, clip_seconds=2.0, stride_seconds=1.0) == [0.0]
    
def test_window_starts_cover_non_overlapping_windows():
    assert window_starts(5.0, clip_seconds=2.0, stride_seconds=1.0) == [0.0, 1.0, 2.0, 3.0]


def test_window_starts_can_include_tail_window():
    assert window_starts(5.5, clip_seconds=2.0, stride_seconds=2.0, include_tail=True) == [0.0, 2.0, 3.5]


def test_temporal_sample_indices_downsample_evenly():
    assert temporal_sample_indices(num_source_frames=10, num_target_frames=5) == [0, 2, 4, 7, 9]


def test_temporal_sample_indices_repeat_for_upsampling():
    assert temporal_sample_indices(num_source_frames=2, num_target_frames=5) == [0, 0, 0, 1, 1]


def test_invalid_window_parameters_raise():
    with pytest.raises(ValueError):
        window_starts(10.0, clip_seconds=0.0, stride_seconds=1.0)
    with pytest.raises(ValueError):
        window_starts(10.0, clip_seconds=1.0, stride_seconds=0.0)


def test_build_window_index_from_manifest(tmp_path):
    video = tmp_path / "clip.mp4"
    video.touch()
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("path,duration_sec,split\nclip.mp4,4.0,train\n", encoding="utf-8")
    
    config = AVWindowConfig(clip_seconds=2.0, stride_seconds=1.0)
    windows = build_window_index(manifest=manifest, config=config, split="train")
    
    assert len(windows) == 3
    assert windows[0].path == video
    assert windows[-1].start_sec == 2.0
    

def test_build_window_index_from_window_manifest(tmp_path):
    video = tmp_path / "clip.mp4"
    video.touch()
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "path,start_sec,end_sec,duration_sec,split\n"
        "clip.mp4,1.5,3.5,10.0,train\n"
        "clip.mp4,4.0,6.0,10.0,val\n",
        encoding="utf-8",
    )

    windows = build_window_index(manifest=manifest, split="train")

    assert len(windows) == 1
    assert windows[0].path == video
    assert windows[0].start_sec == 1.5
    assert windows[0].end_sec == 3.5

    
