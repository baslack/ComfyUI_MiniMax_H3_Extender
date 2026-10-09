"""Seam contract: the next clip takes over where its re-created overlap is in step.

The next clip re-creates the previous clip's last frames (the motion-context
overlap), but not in step everywhere: it is often a frame ahead at the very
end, so cutting there makes the picture jump forward. Final Decode crossfades
from the previous clip into the next one around the overlap frame where the
two match best. The previous clip loses as many frames as the next one gains.
"""
from __future__ import annotations

import importlib

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


def test_next_clip_takes_over_around_the_best_aligned_overlap_frame(ext):
    d = importlib.import_module(f"{ext.pkg.__name__}.motion_context_disk")
    frames, meta = _pair(in_step_at=PREV - 6)
    source = frames.clone()

    with torch.inference_mode():
        _, previous_raw, current_raw, shift = d._decode_pair_video(DecodedVAE(frames), None, meta)

    fade = d.SEAM_CROSSFADE_FRAMES
    cut = PREV - 6 - fade // 2
    lead = PREV - cut
    assert shift == -lead
    assert previous_raw.shape[0] == PREV - lead
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
