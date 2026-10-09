"""Seam contract: the next clip takes over where its re-created overlap is in step.

The next clip re-creates the previous clip's last frames (the motion-context
overlap), but not in step everywhere: it is often a frame ahead at the very
end, so cutting there makes the picture jump forward. Final Decode crossfades
from the previous clip into the next one around the overlap frame where the
two match best. The previous clip loses as many frames as the next one gains.
"""
from __future__ import annotations

import importlib

import pytest
import torch

PREV, WARM, CONTINUED = 22, 17, 12


class DecodedVAE:
    def __init__(self, frames):
        self.frames = frames

    def decode(self, chain):
        return self.frames


def _square(x):
    frame = torch.zeros(96, 448, 3)
    frame[32:64, x:x + 32] = 1.0
    return frame


def _pair(in_step_at):
    """A walks right; B's overlap copy runs one step ahead except at ``in_step_at``."""
    a = [8 * t for t in range(PREV)]
    b_overlap = [8 * t if t == in_step_at else 8 * (t + 1) for t in range(PREV - WARM, PREV)]
    b_new = [8 * (PREV + 1 + t) for t in range(CONTINUED)]
    frames = torch.stack([_square(x) for x in a + b_overlap + b_new])
    meta = {"previous_frames": PREV, "warmup_frames": WARM, "continued_frames": CONTINUED,
            "decode_frames": int(frames.shape[0])}
    return frames, meta


@pytest.mark.parametrize("fade", [6, 10])
def test_next_clip_takes_over_around_the_best_aligned_overlap_frame(ext, fade):
    d = importlib.import_module(f"{ext.pkg.__name__}.motion_context_disk")
    frames, meta = _pair(in_step_at=PREV - 6)
    source = frames.clone()

    with torch.inference_mode():
        current_raw, shift = d._decode_pair_video(DecodedVAE(frames), None, meta, fade)

    cut = PREV - 6 - fade // 2
    lead = PREV - cut
    assert shift == -lead
    assert current_raw.shape[0] == CONTINUED + lead
    for j in range(fade):
        a_weight = 1.0 - (j + 1) / (fade + 1)
        expected = torch.lerp(source[cut + j + WARM], source[cut + j], a_weight)
        assert torch.allclose(current_raw[j], expected, atol=1e-6)
    assert torch.equal(current_raw[fade:], source[cut + fade + WARM:PREV + WARM + CONTINUED])


def test_the_previous_clip_loses_what_the_next_clip_gains(ext):
    d = importlib.import_module(f"{ext.pkg.__name__}.motion_context_disk")
    segments = [
        {"frames": 124},
        {"frames": 124, "trim_frames": 22, "seam_lead": 9},
        {"frames": 124, "trim_frames": 22, "seam_lead": 5},
    ]
    visible = [d._visible_frames(segments, i) for i in range(3)]
    assert visible == [124 - 9, 102 + 9 - 5, 102 + 5]
    assert sum(visible) == d._final_frame_count(segments)


def test_no_crossfade_cuts_at_the_end_of_the_overlap(ext):
    d = importlib.import_module(f"{ext.pkg.__name__}.motion_context_disk")
    frames, meta = _pair(in_step_at=PREV - 6)
    source = frames.clone()

    with torch.inference_mode():
        current_raw, shift = d._decode_pair_video(DecodedVAE(frames), None, meta, 0)

    assert shift == 0
    assert torch.equal(current_raw, source[PREV + WARM:])


def test_a_crossfade_longer_than_the_overlap_allows_is_capped(ext):
    d = importlib.import_module(f"{ext.pkg.__name__}.motion_context_disk")
    frames, meta = _pair(in_step_at=PREV - 6)

    with torch.inference_mode():
        _, shift = d._decode_pair_video(DecodedVAE(frames), None, meta, 32)

    assert -shift == WARM - d.SEAM_SKIP_FRAMES


def test_changing_the_crossfade_recuts_only_the_seams_made_with_another_length(ext, tmp_path):
    d = importlib.import_module(f"{ext.pkg.__name__}.motion_context_disk")
    manifest_path = tmp_path / "chain.json"
    manifest = {"segments": [
        {"frames": 124, "decoded_mp4_blob": {}},
        {"frames": 124, "seam_lead": 8, "seam_crossfade_frames": 6, "decoded_mp4_blob": {}},
        {"frames": 124, "seam_lead": 9, "seam_crossfade_frames": 10, "decoded_mp4_blob": {}},
        {"frames": 124, "seam_lead": 9, "seam_crossfade_frames": 10, "decoded_mp4_blob": {}},
    ]}

    updated = d._set_seam_crossfade(manifest_path, manifest, 10)

    clip_0, clip_1, clip_2, clip_3 = updated["segments"]
    assert updated["seam_crossfade_frames"] == 10
    assert "seam_lead" not in clip_1 and clip_1["seam_crossfade_frames"] == 10
    for clip in (clip_0, clip_1):
        assert clip["final_video_dirty"] and "decoded_mp4_blob" not in clip
    for clip in (clip_2, clip_3):
        assert clip["seam_lead"] == 9 and "final_video_dirty" not in clip and "decoded_mp4_blob" in clip
    assert manifest_path.exists()

    manifest_path.unlink()
    assert d._set_seam_crossfade(manifest_path, updated, 10) is updated
    assert not manifest_path.exists()
