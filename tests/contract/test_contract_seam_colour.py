"""Seam contract: the next clip takes on the previous clip's colour, then eases into its own.

The next clip's copy of the previous clip's frames can sit on a different
brightness or colour, and wobble from frame to frame. Where the two clips are
crossfaded, every next-clip frame is matched to the previous clip's frame at
that instant, so the light follows the previous clip. The last difference then
fades out, so the next clip keeps its own changes in light. Later frames are
untouched.
"""
from __future__ import annotations

import importlib

import pytest
import torch

A = (0.25, 0.25, 0.25)


class DecodedVAE:
    def __init__(self, frames):
        self.frames = frames

    def decode(self, chain):
        return self.frames


def _frames(colours):
    return torch.stack([torch.zeros(16, 24, 3) + torch.tensor(c) for c in colours])


def _decode(d, colours, prev, warm, continued):
    frames = _frames(colours)
    meta = {"previous_frames": prev, "warmup_frames": warm, "continued_frames": continued,
            "decode_frames": int(frames.shape[0])}
    source = frames.clone()
    current, shift = d._decode_pair_video(DecodedVAE(frames), None, meta)
    return source, current, shift


CASES = {
    "brighter": ((0.30, 0.30, 0.30), [(0.30, 0.30, 0.30)] * 20),
    "brightening": ((0.30, 0.30, 0.30), [(0.31 + 0.01 * k,) * 3 for k in range(20)]),
    "colour": ((0.25, 0.25, 0.31), [(0.25, 0.25, 0.31)] * 20),
}


@pytest.mark.parametrize("case", list(CASES))
def test_hard_seam_eases_from_the_previous_clips_colour(ext, case):
    d = importlib.import_module(f"{ext.pkg.__name__}.motion_context_disk")
    prev, warm = 8, 5
    copy, new = CASES[case]
    source, current, shift = _decode(d, [A] * prev + [copy] * warm + new, prev, warm, len(new))

    assert shift == 0
    means = current[..., :3].mean(dim=(1, 2))
    for k in range(12):
        expected = torch.tensor([n + (a - c) * (1 - (k + 1) / 13) for n, a, c in zip(new[k], A, copy)])
        assert torch.allclose(means[k], expected, atol=1e-4), (k, means[k], expected)
    assert torch.equal(current[12:], source[prev + warm + 12:])


def test_crossfade_follows_the_previous_clips_light(ext):
    d = importlib.import_module(f"{ext.pkg.__name__}.motion_context_disk")
    prev, warm, continued = 22, 17, 20
    a = [(0.20 + 0.01 * t, 0.20 + 0.01 * t, 0.22) for t in range(prev)]
    wobble = [0.03, 0.05, 0.02, 0.06]
    copies = [tuple(x + wobble[t % 4] for x in a[t]) for t in range(prev - warm, prev)]
    new = [(0.45, 0.44, 0.40)] * continued
    source, current, shift = _decode(d, a + copies + new, prev, warm, continued)

    lead = -shift
    fade = d.SEAM_CROSSFADE_FRAMES
    assert lead >= fade
    means = current[..., :3].mean(dim=(1, 2))
    for j in range(fade):
        assert torch.allclose(means[j], torch.tensor(a[prev - lead + j]), atol=1e-4), j
    last = torch.tensor(a[prev - lead + fade - 1]) - torch.tensor(copies[warm - lead + fade - 1])
    raw = source[prev + warm - lead:][..., :3].mean(dim=(1, 2))
    for k in range(12):
        expected = raw[fade + k] + last * (1 - (k + 1) / 13)
        assert torch.allclose(means[fade + k], expected, atol=1e-4), k
    assert torch.equal(current[fade + 12:], source[prev + warm - lead + fade + 12:])
