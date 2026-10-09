"""Clip-chain contract: graph clips chain through Motion Context RAM.

The first clip has no previous clip, and a refined previous clip is shrunk to
the next clip's grid.
"""
from __future__ import annotations

import types

import pytest
import torch


def _av(height, width, seed=0):
    import comfy.nested_tensor

    g = torch.Generator().manual_seed(seed)
    video = torch.rand(1, 24, 7, height, width, generator=g)
    audio = torch.rand(1, 32, 2, 37, generator=g)
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


@pytest.fixture
def nodes(ext):
    return types.SimpleNamespace(**ext.pkg.NODE_CLASS_MAPPINGS)


def test_motion_context_without_a_previous_clip_passes_conditioning_through(nodes):
    conditioning = [[torch.zeros(1, 4, 8), {}]]
    out = nodes.MiniMaxH3MotionContextRAM().apply(conditioning, _av(4, 6))
    assert out[0] is conditioning
    assert out[1:4] == (0, 0, 0)


def test_motion_context_shrinks_a_refined_previous_clip_to_the_target_grid(nodes):
    conditioning = [[torch.zeros(1, 4, 8), {}]]
    out = nodes.MiniMaxH3MotionContextRAM().apply(conditioning, _av(4, 6), _av(8, 12, seed=5), "5", 0)
    video_blocks = [k["latent"] for k in out[0][0][1]["minimax_keyframes"] if "latent" in k]
    assert video_blocks and all(tuple(b.shape[-2:]) == (4, 6) for b in video_blocks)
    assert out[1] == 5
