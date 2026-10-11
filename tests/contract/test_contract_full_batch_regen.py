"""Full Batch contract: regenerating the last clip decodes only that clip.

Final Decode needs the new clip's seam, its final video and the browser
preview. One VideoVAE decode of the new clip covers all three; the clips
before it keep their final videos while their seams don't move.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import torch


class CountingVAE:
    def __init__(self, frames_from_latent_t):
        self.frames_from_latent_t = frames_from_latent_t
        self.decodes = 0

    def decode(self, latent):
        self.decodes += 1
        frames = self.frames_from_latent_t(int(latent.shape[2]))
        return torch.rand(frames, latent.shape[3] * 16, latent.shape[4] * 16, 3, generator=torch.Generator().manual_seed(5))


class SilentAudioVAE:
    audio_sample_rate = 32000

    def decode(self, latent):
        return torch.zeros(1, int(latent.shape[-1]) * 800, 2)


def _av(seed):
    import comfy.nested_tensor

    g = torch.Generator().manual_seed(seed)
    return {"samples": comfy.nested_tensor.NestedTensor((torch.rand(1, 24, 12, 6, 8, generator=g), torch.rand(1, 32, 2, 70, generator=g)))}


def _join(nodes, seed, previous=None):
    kwargs = {"trim_frames": 22, "previous_cache": previous} if previous is not None else {}
    return nodes["MiniMaxH3MotionContextDiskJoin"]().join(
        samples=_av(seed), validated=False, run_mode="full_batch", fps=24.0, unique_id="full_batch_regen", **kwargs)[0]


def _final_decode(nodes, handle, vae):
    return nodes["MiniMaxH3MotionContextDiskFinalDecode"]().export(
        cache=handle, vae=vae, audio_vae=SilentAudioVAE(), fps=24.0, filename_prefix="regen", output_directory="",
        codec="H.264", crf=17, preset="fast", audio_bitrate="192k", autoplay=False, unique_id="regen_final",
    )


def _frames(path):
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    return imageio_ffmpeg.count_frames_and_secs(str(path))[0]


def test_regenerating_the_last_clip_decodes_only_that_clip(ext):
    d = importlib.import_module(f"{ext.pkg.__name__}.motion_context_disk")
    nodes = ext.pkg.NODE_CLASS_MAPPINGS
    vae = CountingVAE(d._frames_from_video_t)
    first = _join(nodes, 1)
    second = _join(nodes, 2, first)
    third = _join(nodes, 3, second)
    with torch.inference_mode():
        _final_decode(nodes, third, vae)

    regenerated = _join(nodes, 4, second)
    vae.decodes = 0
    with torch.inference_mode():
        result = _final_decode(nodes, regenerated, vae)
    assert vae.decodes == 1

    manifest = json.loads(Path(regenerated["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["segments"][2]["seam_lead"] > 0
    total = d._final_frame_count(manifest["segments"])
    preview = d._decoded_preview_cache_path(Path(regenerated["data_path"]))
    assert _frames(preview) == total
    assert result["ui"]["h3_preview_info"][0]["preview_frames"] == total
    assert _frames(result["result"][0].get_stream_source()) == total


def _rerun_validated(nodes, count, run_mode="full_batch", owner="full_batch_regen"):
    """Run the first ``count`` clips again as validated, as when the clips after them are bypassed."""
    handle = None
    for _ in range(count):
        kwargs = {"previous_cache": handle} if handle is not None else {}
        handle = nodes["MiniMaxH3MotionContextDiskJoin"]().join(
            validated=True, run_mode=run_mode, fps=24.0, unique_id=owner, **kwargs)[0]
    return handle


def test_bypassed_clips_are_left_out_of_the_video_and_kept_in_the_cache(ext):
    d = importlib.import_module(f"{ext.pkg.__name__}.motion_context_disk")
    nodes = ext.pkg.NODE_CLASS_MAPPINGS
    vae = CountingVAE(d._frames_from_video_t)
    handle = None
    for seed in range(1, 5):
        handle = _join(nodes, seed, handle)
    with torch.inference_mode():
        _final_decode(nodes, handle, vae)

    manifest_path = Path(handle["manifest_path"])
    segments = json.loads(manifest_path.read_text(encoding="utf-8"))["segments"]

    # Clip 4 bypassed: only the last kept clip is decoded again, at full length.
    vae.decodes = 0
    with torch.inference_mode():
        result = _final_decode(nodes, _rerun_validated(nodes, 3), vae)
    assert vae.decodes == 1
    assert _frames(result["result"][0].get_stream_source()) == d._final_frame_count(segments[:3])
    assert len(json.loads(manifest_path.read_text(encoding="utf-8"))["segments"]) == 4

    # Clip 4 back on: it is read from the cache; clip 3 is cut for it again.
    vae.decodes = 0
    with torch.inference_mode():
        result = _final_decode(nodes, _rerun_validated(nodes, 4), vae)
    assert vae.decodes == 1
    assert _frames(result["result"][0].get_stream_source()) == d._final_frame_count(segments)


def test_hard_cut_chain_leaves_bypassed_clips_out_of_the_video(ext):
    d = importlib.import_module(f"{ext.pkg.__name__}.motion_context_disk")
    nodes = ext.pkg.NODE_CLASS_MAPPINGS
    vae = CountingVAE(d._frames_from_video_t)
    join = nodes["MiniMaxH3MotionContextDiskJoin"]()
    handle = None
    for seed in range(1, 4):
        kwargs = {"previous_cache": handle} if handle is not None else {"chain_mode": "hard_cut"}
        handle = join.join(samples=_av(seed), validated=False, run_mode="full_batch", fps=24.0,
                           unique_id="hard_cut_bypass", **kwargs)[0]
    with torch.inference_mode():
        _final_decode(nodes, handle, vae)
    segments = json.loads(Path(handle["manifest_path"]).read_text(encoding="utf-8"))["segments"]

    first_two = None
    for _ in range(2):
        kwargs = {"previous_cache": first_two} if first_two is not None else {}
        first_two = join.join(validated=True, run_mode="full_batch", fps=24.0, unique_id="hard_cut_bypass", **kwargs)[0]
    with torch.inference_mode():
        result = _final_decode(nodes, first_two, vae)
    assert _frames(result["result"][0].get_stream_source()) == sum(int(s["frames"]) for s in segments[:2])
    assert len(json.loads(Path(handle["manifest_path"]).read_text(encoding="utf-8"))["segments"]) == 3


def test_clip_by_clip_leaves_bypassed_clips_out_of_the_video(ext):
    d = importlib.import_module(f"{ext.pkg.__name__}.motion_context_disk")
    nodes = ext.pkg.NODE_CLASS_MAPPINGS
    vae = CountingVAE(d._frames_from_video_t)
    owner = "clip_by_clip_bypass"
    # One new clip per queue, validated before the next.
    for seed in range(1, 5):
        previous = _rerun_validated(nodes, seed - 1, "clip_by_clip", owner) if seed > 1 else None
        kwargs = {"trim_frames": 22, "previous_cache": previous} if previous is not None else {}
        handle = nodes["MiniMaxH3MotionContextDiskJoin"]().join(
            samples=_av(seed), validated=False, run_mode="clip_by_clip", fps=24.0, unique_id=owner, **kwargs)[0]
        with torch.inference_mode():
            _final_decode(nodes, handle, vae)
    with torch.inference_mode():
        _final_decode(nodes, _rerun_validated(nodes, 4, "clip_by_clip", owner), vae)
    manifest_path = Path(handle["manifest_path"])
    segments = json.loads(manifest_path.read_text(encoding="utf-8"))["segments"]

    with torch.inference_mode():
        result = _final_decode(nodes, _rerun_validated(nodes, 3, "clip_by_clip", owner), vae)
    assert _frames(result["result"][0].get_stream_source()) == d._final_frame_count(segments[:3])
    assert len(json.loads(manifest_path.read_text(encoding="utf-8"))["segments"]) == 4

    with torch.inference_mode():
        result = _final_decode(nodes, _rerun_validated(nodes, 4, "clip_by_clip", owner), vae)
    assert _frames(result["result"][0].get_stream_source()) == d._final_frame_count(segments)
