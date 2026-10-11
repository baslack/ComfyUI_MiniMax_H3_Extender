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
