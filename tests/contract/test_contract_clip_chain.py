"""Clip-chain contract: graph clips chain through Motion Context RAM and Disk Join.

The first clip has no previous clip, a refined previous clip is shrunk to the
next clip's grid, and Disk Join records a clip name and colour for Final
Decode without changing what the Extender writes.
"""
from __future__ import annotations

import json
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


def _manifest(handle):
    with open(handle["manifest_path"], encoding="utf-8") as f:
        return json.load(f)


def test_disk_join_stores_clip_name_and_colour_and_updates_them_on_validated_clips(nodes):
    join = nodes.MiniMaxH3MotionContextDiskJoin()
    common = dict(run_mode="full_batch", fps=24.0, unique_id="graph_nodes_named")
    handle = join.join(samples=_av(4, 6), validated=False, clip_name="intro", saturation=120.0,
                       contrast=100.0, brightness=90.0, **common)[0]
    segment = _manifest(handle)["segments"][0]
    assert segment["clip_name"] == "intro"
    assert segment["color_adjustment"] == {"saturation": 120.0, "contrast": 100.0, "brightness": 90.0}

    handle = join.join(samples=None, validated=True, clip_name="intro", saturation=80.0,
                       contrast=100.0, brightness=90.0, **common)[0]
    segment = _manifest(handle)["segments"][0]
    assert segment["color_adjustment"]["saturation"] == 80.0
    assert segment["final_video_dirty"] is True


def test_disk_join_called_like_the_extender_writes_no_clip_settings(nodes):
    handle = nodes.MiniMaxH3MotionContextDiskJoin().join(
        samples=_av(4, 6), validated=False, run_mode="full_batch", fps=24.0, unique_id="graph_nodes_plain")[0]
    segment = _manifest(handle)["segments"][0]
    assert "clip_name" not in segment and "color_adjustment" not in segment


def test_hard_cut_chain_hides_a_first_frame_copied_from_the_previous_clip(nodes):
    join = nodes.MiniMaxH3MotionContextDiskJoin()
    common = dict(validated=False, run_mode="full_batch", fps=24.0)
    first = join.join(samples=_av(4, 6), unique_id="clip_chain_cut", chain_mode="hard_cut", **common)[0]
    assert join.check_lazy_status(samples=None, previous_cache=first, **common) == ["samples"]
    second = join.join(samples=_av(4, 6, seed=3), previous_cache=first, first_frame_from_previous=True,
                       chain_mode="hard_cut", **common)[0]
    manifest = _manifest(second)
    assert manifest["sequence_mode"] == "ref2va_independent"
    a, b = manifest["segments"]
    assert (a["clip_id"], a["trim_frames"], a["visible_offset"], a["frames"]) == ("clip_1", 0, 0, 22)
    assert (b["clip_id"], b["trim_frames"], b["visible_offset"], b["source_frames"], b["frames"]) == ("clip_2", 0, 1, 22, 21)
    assert manifest["final_frame_count"] == 43


def test_first_frame_from_previous_can_change_on_a_validated_clip(nodes):
    join = nodes.MiniMaxH3MotionContextDiskJoin()
    common = dict(run_mode="full_batch", fps=24.0, chain_mode="hard_cut")
    first = join.join(samples=_av(4, 6), validated=True, unique_id="clip_chain_toggle", **common)[0]
    join.join(samples=_av(4, 6, seed=3), validated=True, previous_cache=first, first_frame_from_previous=True, **common)
    handle = join.join(samples=None, validated=True, previous_cache=first, first_frame_from_previous=False, **common)[0]
    segment = _manifest(handle)["segments"][1]
    assert (segment["visible_offset"], segment["frames"]) == (0, 22)


def test_changing_the_chain_mode_restarts_the_chain(nodes):
    join = nodes.MiniMaxH3MotionContextDiskJoin()
    common = dict(validated=False, run_mode="full_batch", fps=24.0, unique_id="clip_chain_switch")
    first = join.join(samples=_av(4, 6), chain_mode="motion_context", **common)[0]
    join.join(samples=_av(4, 6), previous_cache=first, trim_frames=5, validated=False, run_mode="full_batch", fps=24.0)
    handle = join.join(samples=_av(4, 6, seed=7), chain_mode="hard_cut", **common)[0]
    manifest = _manifest(handle)
    assert manifest["sequence_mode"] == "ref2va_independent"
    assert len(manifest["segments"]) == 1


def test_motion_context_keeps_keyframes_already_on_the_conditioning(nodes):
    last_frame = {"resolved_frame_index": 21, "latent": torch.zeros(1, 24, 1, 4, 6)}
    conditioning = [[torch.zeros(1, 4, 8), {"minimax_keyframes": [last_frame]}]]
    out = nodes.MiniMaxH3MotionContextRAM().apply(conditioning, _av(4, 6), _av(4, 6, seed=5), "5", 0)
    keyframes = out[0][0][1]["minimax_keyframes"]
    assert keyframes[0] is last_frame
    assert len(keyframes) > 1 and all(k is not last_frame for k in keyframes[1:])
