"""Seam contract: the next clip eases from the previous clip's colour into its own.

The first decoded frames of a continued clip can flash or sit on a different
brightness or colour. Final Decode moves each of them onto a straight path
from the previous clip's last frame to the next clip's settled colour, and
leaves the rest of the clip untouched.
"""
from __future__ import annotations

import importlib

import pytest
import torch


def _frames(colours):
    return torch.stack([torch.full((16, 24, 3), 0.0) + torch.tensor(c) for c in colours])


SETTLED = (0.32, 0.30, 0.28)
CASES = {
    "flash": [(0.42, 0.40, 0.38), (0.38, 0.36, 0.34)] + [SETTLED] * 18,
    "step": [SETTLED] * 20,
    "colour": [(0.30, 0.30, 0.36), (0.30, 0.30, 0.34)] + [SETTLED] * 18,
}


@pytest.mark.parametrize("case", list(CASES))
def test_next_clip_eases_from_the_previous_clips_last_frame(ext, case):
    d = importlib.import_module(f"{ext.pkg.__name__}.motion_context_disk")
    previous = _frames([(0.25, 0.25, 0.25)] * 4)
    current = _frames(CASES[case])
    untouched = current[12:].clone()

    d._correct_current_segment(previous, current)

    means = current[..., :3].mean(dim=(1, 2))
    for k in range(12):
        t = k / 12
        expected = torch.tensor([0.25 * (1 - t) + s * t for s in SETTLED])
        assert torch.allclose(means[k], expected, atol=1e-4), (k, means[k], expected)
    assert torch.equal(current[12:], untouched)
