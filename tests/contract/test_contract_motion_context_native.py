"""Motion Context on a ComfyUI build with native H3 guide anchors: works without patching ComfyUI."""
from __future__ import annotations

import pytest
import torch


def _av_latent(video_t=12, height=4, width=6, audio_t=65):
    import comfy.nested_tensor

    return {"samples": comfy.nested_tensor.NestedTensor([torch.randn(1, 24, video_t, height, width), torch.randn(1, 32, 2, audio_t)])}


def test_motion_context_anchors_the_previous_clip_without_patching_comfyui(ext, monkeypatch):
    import comfy.ldm.minimax.model as minimax_model
    import comfy.model_base as model_base

    if "frame_count" in minimax_model.PackedLayout.__init__.__code__.co_varnames:
        pytest.skip("this ComfyUI build predates native H3 guide anchors")
    stock_extra_conds = model_base.MiniMaxH3.extra_conds
    stock_layout_init = minimax_model.PackedLayout.__init__
    # Restore the stock functions afterwards even if this test catches a patch.
    monkeypatch.setattr(model_base.MiniMaxH3, "extra_conds", stock_extra_conds)
    monkeypatch.setattr(minimax_model.PackedLayout, "__init__", stock_layout_init)
    node = ext.pkg.NODE_CLASS_MAPPINGS["MiniMaxH3MotionContextRAM"]()

    conditioning, trim_frames, *_ = node.apply(
        [[torch.randn(1, 3, 8), {}]], _av_latent(), _av_latent(), "22", 0,
    )

    assert trim_frames == 22
    assert conditioning[0][1]["minimax_keyframes"]
    assert model_base.MiniMaxH3.extra_conds is stock_extra_conds
    assert minimax_model.PackedLayout.__init__ is stock_layout_init


def test_a_context_from_an_existing_video_anchors_its_tail_as_one_head_guide(ext):
    import comfy.ldm.minimax.model as minimax_model

    if "frame_count" in minimax_model.PackedLayout.__init__.__code__.co_varnames:
        pytest.skip("this ComfyUI build predates native H3 guide anchors")
    node = ext.pkg.NODE_CLASS_MAPPINGS["MiniMaxH3MotionContextRAM"]()
    context = _av_latent()
    guide = torch.randn(1, 24, 6, 4, 6)
    context["minimax_source_guide"] = guide
    last_frame = {"resolved_frame_index": 120, "latent": torch.randn(1, 24, 1, 4, 6)}

    conditioning, trim_frames, *_ = node.apply(
        [[torch.randn(1, 3, 8), {"minimax_keyframes": [last_frame]}]], _av_latent(), context, "22", 0,
    )

    keyframes = conditioning[0][1]["minimax_keyframes"]
    video = [k for k in keyframes if "latent" in k]
    assert trim_frames == 22
    assert video[0] is last_frame
    assert len(video) == 2 and video[1]["resolved_frame_index"] == 0 and torch.equal(video[1]["latent"], guide)
    assert any("audio_latent" in k for k in keyframes)
