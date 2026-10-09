"""Final-decode seam contract.

Clips are cached through the public Disk Join node and rendered with a tiny
random-weight instance of ComfyUI's real MiniMax H3 VAE, wrapped to record
how much latent each decode asks for.
"""
from __future__ import annotations

import importlib
import json

import pytest
import torch

TAIL_TOKENS = 7  # one 5-token VAE chunk plus its 2 overlap tokens


class RecordingVAE:
    """ComfyUI VAE output contract: decode(latent) -> [B, T, H, W, C] float pixels."""

    def __init__(self, model):
        self.model = model
        self.decoded_tokens = []

    def decode(self, latent):
        self.decoded_tokens.append(int(latent.shape[2]))
        with torch.inference_mode():
            return self.model.decode(latent.float()).movedim(1, -1)


@pytest.fixture(scope="module")
def tiny_h3_vae():
    from comfy.ldm.minimax.vae import MiniMaxH3VideoVAE

    model = MiniMaxH3VideoVAE(ch=32, num_layers=1).eval()
    generator = torch.Generator().manual_seed(0)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.02)
    return model


def _av(video):
    import comfy.nested_tensor

    return {"samples": comfy.nested_tensor.NestedTensor([video, torch.zeros(1, 32, 2, 8)])}


def _cache_two_clips(ext, owner, first, second, trim_frames=22):
    join = ext.pkg.NODE_CLASS_MAPPINGS["MiniMaxH3MotionContextDiskJoin"]()
    handle = join.join(samples=_av(first), validated=True, run_mode="full_batch", fps=24.0, unique_id=owner)[0]
    handle = join.join(
        samples=_av(second), trim_frames=trim_frames, validated=True, run_mode="full_batch",
        fps=24.0, previous_cache=handle, unique_id=owner,
    )[0]
    with open(handle["manifest_path"], encoding="utf-8") as f:
        return handle["data_path"], json.load(f)["segments"]


def _render_second_clip(ext, vae, data_path, segments):
    disk = importlib.import_module(f"{ext.pkg.__name__}.motion_context_disk")
    video, _shift = disk._render_one_final_video_segment(data_path, segments, 1, vae)
    return video


def _clips(seed=1):
    generator = torch.Generator().manual_seed(seed)
    return torch.randn(1, 24, 22, 2, 2, generator=generator), torch.randn(1, 24, 12, 2, 2, generator=generator)


def test_second_clip_does_not_depend_on_previous_clip_beyond_its_tail(ext, tiny_h3_vae):
    first, second = _clips()
    long_path, long_segments = _cache_two_clips(ext, "seam_long", first, second)
    short_path, short_segments = _cache_two_clips(ext, "seam_short", first[:, :, -TAIL_TOKENS:].clone(), second)

    from_long = _render_second_clip(ext, RecordingVAE(tiny_h3_vae), long_path, long_segments)
    from_short = _render_second_clip(ext, RecordingVAE(tiny_h3_vae), short_path, short_segments)

    assert from_long.shape == from_short.shape
    assert torch.equal(from_long, from_short)


def test_seam_decode_reads_only_the_previous_clips_tail(ext, tiny_h3_vae):
    first, second = _clips(seed=2)
    data_path, segments = _cache_two_clips(ext, "seam_cost", first, second)
    vae = RecordingVAE(tiny_h3_vae)

    _render_second_clip(ext, vae, data_path, segments)

    assert vae.decoded_tokens == [TAIL_TOKENS + second.shape[2] - 2]  # trim 22 frames = 7 tokens, 5 kept as warm-up


def test_comfyui_h3_vae_decodes_time_chunks_independently(tiny_h3_vae):
    """The assumption the tail decode rests on: from a 5-token boundary on,
    decoding a suffix reproduces the full decode after the first 5 blended frames."""
    z = torch.randn(1, 24, 22, 2, 2, generator=torch.Generator().manual_seed(3))
    with torch.inference_mode():
        full = tiny_h3_vae.decode(z)
        tail = tiny_h3_vae.decode(z[:, :, -TAIL_TOKENS:])

    tail_frames = tail.shape[2]
    assert torch.equal(full[:, :, -tail_frames + 5:], tail[:, :, 5:])


@pytest.mark.gpu
def test_real_h3_vae_decodes_time_chunks_independently(ext):
    import comfy.model_management
    import comfy.sd
    import comfy.utils
    import folder_paths

    name = next((n for n in folder_paths.get_filename_list("vae") if "minimax_h3_video_vae" in n.replace("\\", "/")), None)
    if name is None:
        pytest.skip("no minimax_h3_video_vae checkpoint in models/vae")
    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(folder_paths.get_full_path_or_raise("vae", name)))
    z = torch.randn(1, 24, 22, 4, 6, generator=torch.Generator().manual_seed(4))

    full = vae.decode(z)
    tail = vae.decode(z[:, :, -TAIL_TOKENS:])
    del vae
    comfy.model_management.soft_empty_cache()

    assert torch.allclose(full[:, -tail.shape[1] + 5:], tail[:, 5:], atol=2e-3)
