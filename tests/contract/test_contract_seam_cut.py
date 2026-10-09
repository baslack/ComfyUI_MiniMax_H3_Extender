"""Seam contract: Final Decode starts the next clip right after the motion-context overlap.

The next clip's overlap frames re-create the previous clip's last frames, so
starting it any earlier shows one of those frames twice: a visible stall.
"""
from __future__ import annotations

import importlib

import torch


class DecodedVAE:
    def __init__(self, frames):
        self.frames = frames

    def decode(self, chain):
        return self.frames


def _walking_pair(prev_frames=22, warmup=17, continued=12):
    """A square walking right; the overlap replays A's tail, then B steps 1.4x further.

    This is the case the removed early-shift heuristic answered by starting B one
    frame early, on the replayed copy of A's last frame.
    """
    xs = [3 * t for t in range(prev_frames)]
    xs += [3 * t for t in range(prev_frames - warmup, prev_frames)]
    xs += [xs[prev_frames - 1] + 4.2 + 3 * t for t in range(continued)]
    frames = torch.zeros(len(xs), 64, 192, 3)
    for i, x in enumerate(xs):
        frames[i, 20:28, round(x):round(x) + 8] = 1.0
    meta = {"previous_frames": prev_frames, "warmup_frames": warmup, "continued_frames": continued,
            "decode_frames": len(xs)}
    return frames, meta


def test_next_clip_starts_right_after_the_overlap(ext):
    d = importlib.import_module(f"{ext.pkg.__name__}.motion_context_disk")
    frames, meta = _walking_pair()
    _, previous_raw, current_raw, shift = d._decode_pair_video(DecodedVAE(frames), None, meta)
    start = meta["previous_frames"] + meta["warmup_frames"]
    assert shift == 0
    assert torch.equal(previous_raw, frames[:meta["previous_frames"]])
    assert torch.equal(current_raw, frames[start:start + meta["continued_frames"]])
    assert not torch.equal(current_raw[0], previous_raw[-1])
