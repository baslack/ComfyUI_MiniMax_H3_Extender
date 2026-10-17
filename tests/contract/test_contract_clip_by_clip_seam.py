"""Clip by Clip preview contract: a crossfaded seam shows no frame twice.

With a crossfade, the next clip takes over a few frames (its seam lead) before
the previous clip ends. Clip by Clip commits a clip before that lead is known,
so every preview must still end up exactly as long as the chain.
"""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest
import torch


class DecodedClipVAE:
    def __init__(self, frames_from_latent_t):
        self.frames_from_latent_t = frames_from_latent_t

    def decode(self, latent):
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


def _join(nodes, owner, seed, previous=None):
    kwargs = {"trim_frames": 22, "previous_cache": previous} if previous is not None else {}
    return nodes["MiniMaxH3MotionContextDiskJoin"]().join(
        samples=_av(seed), validated=False, run_mode="full_batch", fps=24.0, unique_id=owner, **kwargs)[0]


def _validate(handle, *indices):
    path = Path(handle["manifest_path"])
    manifest = json.loads(path.read_text(encoding="utf-8"))
    for i in indices:
        manifest["segments"][i]["validated"] = True
    path.write_text(json.dumps(manifest), encoding="utf-8")


class CountingVAE(DecodedClipVAE):
    def __init__(self, frames_from_latent_t):
        super().__init__(frames_from_latent_t)
        self.decodes = 0

    def decode(self, latent):
        self.decodes += 1
        return super().decode(latent)


def _preview_frames(d, handle, vae=None):
    imageio_ffmpeg = pytest.importorskip("imageio_ffmpeg")
    manifest = json.loads(Path(handle["manifest_path"]).read_text(encoding="utf-8"))
    with torch.inference_mode():
        preview, total, *_ = d._export_live_candidate_preview(
            handle["data_path"], handle["manifest_path"], manifest, manifest["segments"],
            vae or DecodedClipVAE(d._frames_from_video_t), SilentAudioVAE(), 24.0, d._find_ffmpeg(), "clip_by_clip_seam")
    return imageio_ffmpeg.count_frames_and_secs(str(preview))[0], total


def test_clip_by_clip_previews_show_no_frame_twice(ext):
    d = importlib.import_module(f"{ext.pkg.__name__}.motion_context_disk")
    nodes = ext.pkg.NODE_CLASS_MAPPINGS
    first = _join(nodes, "clip_by_clip_seam", 1)
    _preview_frames(d, first)
    _validate(first, 0)
    second = _join(nodes, "clip_by_clip_seam", 2, first)

    shown, total = _preview_frames(d, second)
    assert shown == total
    assert json.loads(Path(second["manifest_path"]).read_text(encoding="utf-8"))["segments"][1]["seam_lead"] > 0

    _validate(second, 1)
    third = _join(nodes, "clip_by_clip_seam", 3, second)
    shown, total = _preview_frames(d, third)
    assert shown == total

    # A rebuilt committed preview must not cut the last committed clip twice.
    d._decoded_preview_video_cache_path(Path(third["data_path"])).unlink()
    shown, total = _preview_frames(d, third)
    assert shown == total


def test_clip_by_clip_previews_validated_clips_that_were_never_previewed(ext):
    # A new chain filled with clips already marked validated (a renamed project,
    # or a second chain built from cached clips) has no decoded audio for them yet.
    d = importlib.import_module(f"{ext.pkg.__name__}.motion_context_disk")
    nodes = ext.pkg.NODE_CLASS_MAPPINGS
    first = _join(nodes, "clip_by_clip_unpreviewed", 1)
    second = _join(nodes, "clip_by_clip_unpreviewed", 2, first)
    _validate(second, 0, 1)

    shown, total = _preview_frames(d, second)
    assert shown == total


def test_rerunning_the_candidate_keeps_the_committed_preview(ext):
    # The committed preview holds only the validated clips before the candidate, which a rerun of
    # the candidate doesn't change; rebuilding it would decode those clips again.
    d = importlib.import_module(f"{ext.pkg.__name__}.motion_context_disk")
    nodes = ext.pkg.NODE_CLASS_MAPPINGS
    first = _join(nodes, "rerun_candidate", 1)
    _preview_frames(d, first)
    _validate(first, 0)
    second = _join(nodes, "rerun_candidate", 2, first)
    _preview_frames(d, second)
    committed = d._decoded_preview_video_cache_path(Path(second["data_path"]))
    stamp = committed.stat().st_mtime_ns

    rerun = _join(nodes, "rerun_candidate", 3, first)
    assert committed.stat().st_mtime_ns == stamp
    shown, total = _preview_frames(d, rerun)
    assert shown == total
    assert committed.stat().st_mtime_ns == stamp

