"""Graph-node contract: definitions, prompts and MiniMax H3 Encode feed ordinary
ComfyUI sampling.

MiniMax H3 Encode hands the composed text and the references, in core order,
to ComfyUI's own MiniMax H3 nodes; fakes here record what reaches the text
encoder and VAEs.
"""
from __future__ import annotations

import types
from fractions import Fraction

import pytest
import torch


def _seeded(*shape, seed=0):
    return torch.rand(*shape, generator=torch.Generator().manual_seed(seed))


class RecordingClip:
    def __init__(self):
        self.calls = []

    def tokenize(self, text, **kwargs):
        self.calls.append((text, kwargs))
        return {"text": text}

    def encode_from_tokens_scheduled(self, tokens):
        return [[torch.zeros(1, 4, 8), {}]]


class ShapeVAE:
    def encode(self, pixels):
        frames = pixels.shape[0]
        return torch.zeros(1, 24, 2 if frames <= 5 else ((frames - 5) // 17) * 5 + 2,
                           pixels.shape[1] // 16, pixels.shape[2] // 16)


class ShapeAudioVAE:
    def encode(self, waveform):
        return torch.zeros(1, 32, 2, 10)


class FakeVideo:
    def __init__(self, frames=30, fps=30, seed=0):
        self.components = types.SimpleNamespace(
            images=_seeded(frames, 64, 96, 3, seed=seed),
            frame_rate=Fraction(fps),
            audio={"waveform": torch.zeros(1, 2, 3200), "sample_rate": 32000},
        )

    def get_components(self):
        return self.components


@pytest.fixture
def nodes(ext):
    return types.SimpleNamespace(**{k: v for k, v in ext.pkg.NODE_CLASS_MAPPINGS.items()})


def _definition(nodes, name, kind, text="", **inputs):
    return nodes.MiniMaxH3Definition.execute(name=name, kind={"kind": kind, **inputs},
                                             definition=text, retention_note="")[0]


def test_encode_sends_the_composed_prompt_and_references_in_core_order(nodes):
    fred = _definition(nodes, "fred", "Subject", "is the man in <Picture 1>.",
                       images={"image1": _seeded(1, 64, 64, 3, seed=1), "image0": _seeded(1, 64, 64, 3, seed=2),
                               "image2": None},
                       videos={}, retention="fully_preserved")
    dolly = _definition(nodes, "dolly", "Video", "provides the camera move.", video=FakeVideo(),
                        role="structure_reference", retention="weak_reference",
                        soundtrack_as_audio=True, soundtrack_retention="fully_copy")
    voice = _definition(nodes, "voice", "Audio", "is a voice reference for <Subject fred> (S fred).",
                        audio={"waveform": torch.zeros(1, 2, 3200), "sample_rate": 32000},
                        role="voice_timbre", retention="reference")
    bundle = nodes.MiniMaxH3Definitions.execute(definitions={"definition0": fred, "definition1": dolly,
                                                             "definition2": voice})[0]
    prompt = nodes.MiniMaxH3Ref2VAPrompt.execute(
        definitions=bundle, task_types=[], summary="",
        detailed_description="[Shot 1] <Subject fred> (S fred) talks to <Video dolly> in the tone of <Audio voice>.",
        overall_soundscape="", non_diegetic_music="")[0]

    clip = RecordingClip()
    out = nodes.MiniMaxH3Encode.execute(clip=clip, vae=ShapeVAE(), prompt=prompt, width=1000, height=560,
                                        size_rounding="up", length=124, ref_image_size="match",
                                        audio_vae=ShapeAudioVAE())
    positive, latent, composed = out[0], out[1], out[2]

    assert composed == prompt["text"]
    text, kwargs = clip.calls[0]
    assert text == composed
    assert [item["type"] for item in kwargs["minimax_ref_items"]] == ["image", "image", "audio", "video", "audio"]
    assert [b["kind"] for b in positive[0][1]["minimax_refs"]] == ["image", "image", "video_audio", "audio"]
    assert "<Audio 1> is the synchronized audio track of <Video 1>." in composed
    assert "<Audio 2> is a voice reference for <Subject 1> (S1)." in composed
    video = latent["samples"].unbind()[0]
    assert tuple(video.shape[-2:]) == (576 // 16, 1024 // 16)


@pytest.mark.parametrize(("rounding", "size"), [("up", (576, 1024)), ("down", (544, 992))])
def test_size_rounding_snaps_to_the_32_pixel_grid(nodes, rounding, size):
    prompt = nodes.MiniMaxH3Ref2VAPrompt.execute(definitions=None, task_types=[], summary="",
                                                 detailed_description="[Shot 1] Rain.", overall_soundscape="",
                                                 non_diegetic_music="")[0]
    latent = nodes.MiniMaxH3Encode.execute(clip=RecordingClip(), vae=ShapeVAE(), prompt=prompt, width=1000,
                                           height=560, size_rounding=rounding, length=124,
                                           ref_image_size="match")[1]
    video = latent["samples"].unbind()[0]
    assert tuple(video.shape[-2:]) == (size[0] // 16, size[1] // 16)


def test_keyframe_encode_writes_the_fl2va_alignment_line(nodes):
    prompt = nodes.MiniMaxH3KeyframePrompt.execute(
        integrated_multimodal_description="[Shot 1] The cyclist opens the umbrella.",
        overall_soundscape="Rain.", non_diegetic_music="",
        first_frame=_seeded(1, 64, 64, 3, seed=3), last_frame=_seeded(1, 64, 64, 3, seed=4))[0]
    clip = RecordingClip()
    out = nodes.MiniMaxH3Encode.execute(clip=clip, vae=ShapeVAE(), prompt=prompt, width=1344, height=768,
                                        size_rounding="up", length=124, ref_image_size="match")
    assert out[2].startswith("How the reference pictures align with the target video — Picture 1 (from Shot 1)")
    assert "Picture 2 (from Shot 1) aligns with the 5.17-second mark" in out[2]
    assert out[2].split("\n\n")[1] == "integrated_multimodal_description: [Shot 1] The cyclist opens the umbrella."
    assert clip.calls[0][0] == out[2]
    assert [k["resolved_frame_index"] for k in out[0][0][1]["minimax_keyframes"]] == [0, 123]


def test_definitions_bundle_refuses_duplicate_names(nodes):
    a = _definition(nodes, "fred", "Subject", images={}, videos={}, retention="fully_preserved")
    b = _definition(nodes, "fred", "Subject", images={}, videos={}, retention="fully_preserved")
    with pytest.raises(ValueError, match="'fred' is used more than once"):
        nodes.MiniMaxH3Definitions.execute(definitions={"definition0": a, "definition1": b})
