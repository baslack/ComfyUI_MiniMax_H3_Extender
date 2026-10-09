"""Seam contract: the next clip eases from the previous clip's colour into its own.

The next clip's copy of the previous clip's frames can sit on a different
brightness or colour. Final Decode shifts the next clip's first frames by the
difference between the previous clip's frame and the next clip's copy of that
same frame, fading the shift out, so the next clip keeps its own changes in
light. Later frames are untouched.
"""
from __future__ import annotations

import importlib

import pytest
import torch

PREV, WARM, CONTINUED = 8, 5, 20
A = (0.25, 0.25, 0.25)


class DecodedVAE:
    def __init__(self, frames):
        self.frames = frames

    def decode(self, chain):
        return self.frames


def _frames(colours):
    return torch.stack([torch.zeros(16, 24, 3) + torch.tensor(c) for c in colours])


CASES = {
    "brighter": ((0.30, 0.30, 0.30), [(0.30, 0.30, 0.30)] * CONTINUED),
    "brightening": ((0.30, 0.30, 0.30), [(0.31 + 0.01 * k,) * 3 for k in range(CONTINUED)]),
    "colour": ((0.25, 0.25, 0.31), [(0.25, 0.25, 0.31)] * CONTINUED),
}


@pytest.mark.parametrize("case", list(CASES))
def test_next_clip_eases_from_the_previous_clips_colour(ext, case):
    d = importlib.import_module(f"{ext.pkg.__name__}.motion_context_disk")
    copy, new = CASES[case]
    frames = _frames([A] * PREV + [copy] * WARM + new)
    meta = {"previous_frames": PREV, "warmup_frames": WARM, "continued_frames": CONTINUED,
            "decode_frames": int(frames.shape[0])}
    untouched = frames[PREV + WARM + 12:].clone()

    current, shift = d._decode_pair_video(DecodedVAE(frames), None, meta)

    assert shift == 0
    means = current[..., :3].mean(dim=(1, 2))
    for k in range(12):
        expected = torch.tensor([n + (a - c) * (1 - k / 12) for n, a, c in zip(new[k], A, copy)])
        assert torch.allclose(means[k], expected, atol=1e-4), (k, means[k], expected)
    assert torch.equal(current[12:], untouched)
