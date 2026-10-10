"""Continue Video contract: a graph clip chain can start from an existing video.

The source becomes Clip 0: its working copy and description go into the
chain, the first clip's context carries the source's encoded tail as a head
guide, and that clip's re-created opening is hidden so Final Decode can show
the source instead.
"""
from __future__ import annotations

import importlib
import json
import subprocess
import types

import pytest
import torch


class FileVideo:
    """What ComfyUI's Load Video hands downstream: a VIDEO backed by a file."""

    def __init__(self, path):
        self.path = path

    def get_stream_source(self):
        return str(self.path)


class TailVAE:
    def __init__(self, latent_t):
        self.latent_t = latent_t

    def encode(self, frames):
        return torch.zeros(1, 24, self.latent_t(int(frames.shape[0])), frames.shape[1] // 16, frames.shape[2] // 16)


class TailAudioVAE:
    audio_sample_rate = 32000

    def encode(self, waveform):
        return torch.zeros(1, 32, 2, round(waveform.shape[1] / self.audio_sample_rate * 40))


@pytest.fixture(scope="module")
def source_video(tmp_path_factory):
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    path = tmp_path_factory.mktemp("source") / "walk.mp4"
    subprocess.run([
        imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error",
        "-f", "lavfi", "-i", "testsrc=size=96x64:rate=30", "-f", "lavfi", "-i", "sine=frequency=440",
        "-t", "3", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest", str(path),
    ], check=True)
    return FileVideo(path)


@pytest.fixture
def nodes(ext):
    return types.SimpleNamespace(**ext.pkg.NODE_CLASS_MAPPINGS)


@pytest.fixture
def vaes(ext):
    extender = importlib.import_module(f"{ext.pkg.__name__}.extender")
    return TailVAE(extender._video_latent_t), TailAudioVAE()


def _av(height, width):
    import comfy.nested_tensor

    return {"samples": comfy.nested_tensor.NestedTensor((torch.rand(1, 24, 12, height, width), torch.rand(1, 32, 2, 70)))}


def _manifest(handle):
    with open(handle["manifest_path"], encoding="utf-8") as f:
        return json.load(f)


def _start(nodes, source_video, vaes, owner, width=64, height=64):
    return nodes.MiniMaxH3ContinueVideo().start(source_video, *vaes, width, height, "22", 0, unique_id=owner)


class DecodedClipVAE:
    def __init__(self, frames_from_latent_t):
        self.frames_from_latent_t = frames_from_latent_t

    def decode(self, latent):
        frames = self.frames_from_latent_t(int(latent.shape[2]))
        return torch.rand(frames, latent.shape[3] * 16, latent.shape[4] * 16, 3, generator=torch.Generator().manual_seed(3))


def test_continue_video_starts_a_chain_with_the_source_as_clip_0(nodes, source_video, vaes):
    handle, context = _start(nodes, source_video, vaes, "continue_start")

    source = _manifest(handle)["source_video"]
    assert (source["width"], source["height"], source["fps"]) == (64, 64, 24.0)
    assert source["frame_count"] == 72 and source["has_audio"]
    assert "run_mode" not in handle and handle["next_index"] == 0
    guide = context["minimax_source_guide"]
    assert tuple(guide.shape[-2:]) == (4, 4)


def test_the_first_clip_after_a_source_hides_its_recreated_opening(nodes, source_video, vaes):
    handle, _ = _start(nodes, source_video, vaes, "continue_join")

    out = nodes.MiniMaxH3MotionContextDiskJoin().join(
        samples=_av(4, 4), trim_frames=22, validated=False, run_mode="full_batch", fps=24.0, previous_cache=handle)

    segment = _manifest(out[0])["segments"][0]
    assert segment["visible_offset"] == 22
    assert segment["frames"] == segment["source_frames"] - 22
    assert segment["continued_from_source"] is True


def test_a_first_clip_of_another_size_is_refused(nodes, source_video, vaes):
    handle, _ = _start(nodes, source_video, vaes, "continue_size")

    with pytest.raises(ValueError, match="Continue Video made the source 64x64"):
        nodes.MiniMaxH3MotionContextDiskJoin().join(
            samples=_av(8, 8), trim_frames=22, validated=False, run_mode="full_batch", fps=24.0, previous_cache=handle)


def test_a_new_source_size_invalidates_the_generated_clips(nodes, source_video, vaes):
    handle, _ = _start(nodes, source_video, vaes, "continue_resize")
    nodes.MiniMaxH3MotionContextDiskJoin().join(
        samples=_av(4, 4), trim_frames=22, validated=False, run_mode="full_batch", fps=24.0, previous_cache=handle)

    handle, _ = _start(nodes, source_video, vaes, "continue_resize", width=96)

    manifest = _manifest(handle)
    assert manifest["segments"] == []
    assert manifest["source_video"]["width"] == 96


def test_a_hard_cut_chain_after_a_source_is_refused(nodes, source_video, vaes):
    handle, _ = _start(nodes, source_video, vaes, "continue_hard_cut")

    with pytest.raises(ValueError, match="motion_context, not hard_cut"):
        nodes.MiniMaxH3MotionContextDiskJoin().join(
            samples=_av(4, 4), trim_frames=22, validated=False, run_mode="full_batch", fps=24.0,
            previous_cache=handle, chain_mode="hard_cut")


def test_the_first_clip_after_a_source_asks_for_its_trim(nodes, source_video, vaes):
    handle, _ = _start(nodes, source_video, vaes, "continue_lazy")

    needed = nodes.MiniMaxH3MotionContextDiskJoin().check_lazy_status(
        samples=None, trim_frames=None, validated=False, run_mode="full_batch", fps=24.0, previous_cache=handle)

    assert "trim_frames" in needed



def test_the_first_clip_crossfades_in_from_the_source_and_shortens_it(ext, nodes, source_video, vaes):
    d = importlib.import_module(f"{ext.pkg.__name__}.motion_context_disk")
    handle, _ = _start(nodes, source_video, vaes, "continue_seam")
    out = nodes.MiniMaxH3MotionContextDiskJoin().join(
        samples=_av(4, 4), trim_frames=22, validated=False, run_mode="full_batch", fps=24.0, previous_cache=handle)
    manifest = _manifest(out[0])
    segments = manifest["segments"]
    shown = d._source_frame_count(manifest["source_video"]) + d._visible_frames(segments, 0)

    with torch.inference_mode():
        video, shift = d._render_one_final_video_segment(out[0]["data_path"], segments, 0, DecodedClipVAE(d._frames_from_video_t))

    lead = -shift
    assert d.SEAM_SKIP_FRAMES <= lead <= 22
    assert video.shape[0] == segments[0]["frames"] + lead
    d._record_seam_lead(segments, 0, segments[0], shift)
    manifest = d._sync_source_lead(out[0]["data_path"], dict(manifest, segments=segments))
    assert manifest["source_video"]["seam_lead"] == lead
    assert d._source_frame_count(manifest["source_video"]) == 72 - lead
    assert d._source_frame_count(manifest["source_video"]) + d._visible_frames(segments, 0) == shown
