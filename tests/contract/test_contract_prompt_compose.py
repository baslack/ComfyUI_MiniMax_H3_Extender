"""Prompt contract: composed prompts follow MiniMax's official rewrite formats.

Ref2VA: MiniMaxAI/MiniMax-H3 docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md.
T2VA/I2VA/FL2VA/L2VA: docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md.
Text names definitions (<Subject fred>) and speakers ((S fred)); every clip
gets its own numbering, in the order the core node presents media.
"""
from __future__ import annotations

import importlib

import pytest
import torch


@pytest.fixture(scope="module")
def pc(ext):
    return importlib.import_module(f"{ext.pkg.__name__}.prompt_compose")


def _image(seed):
    return torch.rand(1, 8, 8, 3, generator=torch.Generator().manual_seed(seed))


@pytest.fixture
def pier(pc):
    """Two people on a pier and a voice reference, plus an unused definition."""
    fred = pc.Definition("fred", "Subject", "is the tall man in <Picture 1>, with a grey beard.",
                         "fully_preserved", "grey beard and navy peacoat", images=[_image(1)])
    mara = pc.Definition("mara", "Subject", "is the young woman in <Picture 1>, with short red hair.",
                         "fully_preserved", "red hair and yellow raincoat", images=[_image(2)])
    voice = pc.Definition("fredvoice", "Audio", "is the voice-timbre reference for <Subject fred> (S fred).",
                          "reference", "gravelly timbre without copying the signal", role="voice_timbre",
                          audio={"waveform": torch.zeros(1, 2, 16), "sample_rate": 32000})
    ghost = pc.Definition("ghost", "Subject", "is never mentioned.", "fully_preserved", images=[_image(3)])
    return [fred, mara, voice, ghost]


PIER_DESCRIPTION = (
    "Live-action, cinematic, overcast light.\n"
    "[Shot 1] <Subject mara> (S mara) looks up from the pier and says, <d>[English] You're late.</d>\n"
    "<Subject fred> (S fred) replies in the gravelly timbre from <Audio fredvoice>, "
    "<d>[English] The tide was against me.</d>\n"
    "[Shot 2] At 00:04.000, the two (S fred, S mara) laugh together, <d>[English] Of course it was!</d>"
)


def test_ref2va_prompt_is_the_six_section_rewrite_with_per_clip_numbers(pc, pier):
    out = pc.compose_ref2va(pier, [], "<Subject fred> meets <Subject mara> on a pier.", PIER_DESCRIPTION,
                            "Gulls and lapping water.", "")
    assert out["text"] == (
        "subject_definitions:\n"
        "<Subject 1> is the tall man in <Picture 1>, with a grey beard.\n"
        "<Subject 2> is the young woman in <Picture 2>, with short red hair.\n"
        "<Audio 1> is the voice-timbre reference for <Subject 1> (S2).\n"
        "\n"
        "summary:\n"
        "[reference generation + audio reference] <Subject 1> meets <Subject 2> on a pier.\n"
        "\n"
        "retention_analysis:\n"
        "<Subject 1> (appears in [Shot 1]): fully_preserved - grey beard and navy peacoat\n"
        "<Subject 2> (appears in [Shot 1]): fully_preserved - red hair and yellow raincoat\n"
        "<Audio 1>: reference - gravelly timbre without copying the signal\n"
        "\n"
        "detailed_description:\n"
        "Live-action, cinematic, overcast light.\n"
        "[Shot 1] <Subject 2> (S1) looks up from the pier and says, <d>[English] You're late.</d>\n"
        "<Subject 1> (S2) replies in the gravelly timbre from <Audio 1>, <d>[English] The tide was against me.</d>\n"
        "[Shot 2] At 00:04.000, the two (S1,S2) laugh together, <d>[English] Of course it was!</d>\n"
        "\n"
        "overall_soundscape:\n"
        "Gulls and lapping water.\n"
        "\n"
        "non_diegetic_music:\n"
        "N/A"
    )
    assert [img is d.images[0] for img, d in zip(out["images"], pier[:2])] == [True, True]
    assert out["audios"] == [pier[2].audio]
    assert out["videos"] == []


def test_a_shared_definition_gets_each_clips_own_speaker_number(pc, pier):
    description = "[Shot 1] <Subject fred> (S fred) speaks first in the gravelly timbre of <Audio fredvoice>."
    out = pc.compose_ref2va(pier, [], "", description, "", "")
    assert "<Audio 1> is the voice-timbre reference for <Subject 1> (S1)." in out["text"]


def test_hand_typed_speaker_ids_pass_through(pc):
    out = pc.compose_ref2va([], [], "", "[Shot 1] A narrator (S1) says, <d>[English] Once.</d>", "", "")
    assert "[Shot 1] A narrator (S1) says" in out["text"]


@pytest.mark.parametrize(
    ("summary", "match"),
    [
        ("<Subject nobody> waits.", "<Subject nobody> doesn't match any definition"),
        ("<Picture fred> waits.", "<Picture fred> refers to a Subject definition"),
        ("(S fred) waits.", "speaker 'fred' in the summary never speaks"),
    ],
    ids=["unknown-name", "wrong-kind", "silent-speaker"],
)
def test_bad_references_fail_with_the_name_in_the_error(pc, pier, summary, match):
    with pytest.raises(ValueError, match=match):
        pc.compose_ref2va(pier, [], summary, "[Shot 1] Waves.", "", "")


def test_a_definition_naming_a_speaker_who_stays_silent_fails(pc, pier):
    with pytest.raises(ValueError, match="speaker 'fred' in definition 'fredvoice'"):
        pc.compose_ref2va(pier, [], "", "[Shot 1] <Subject mara> (S mara) hums near <Audio fredvoice>.", "", "")


def test_local_labels_point_at_the_definitions_own_attachments(pc):
    room = pc.Definition("room", "Subject", "is the cafe in <Picture 2>, lit as in <Picture 1>.",
                         "fully_preserved", images=[_image(4), _image(5)])
    shared = _image(6)
    a = pc.Definition("a", "Subject", "wears the coat in <Picture 1>.", "fully_preserved", images=[shared])
    b = pc.Definition("b", "Subject", "holds the coat in <Picture 1>.", "fully_preserved", images=[shared])
    out = pc.compose_ref2va([room, a, b], [], "", "[Shot 1] <Subject room>, <Subject a>, <Subject b>.", "", "")
    assert "<Subject 1> is the cafe in <Picture 2>, lit as in <Picture 1>." in out["text"]
    assert "<Subject 2> wears the coat in <Picture 3>." in out["text"]
    assert "<Subject 3> holds the coat in <Picture 3>." in out["text"]
    assert len(out["images"]) == 3 and out["images"][2] is shared


def test_a_local_label_beyond_the_attachments_fails(pc):
    a = pc.Definition("a", "Subject", "is in <Picture 2>.", "fully_preserved", images=[_image(7)])
    with pytest.raises(ValueError, match="'a' has no attached picture 2"):
        pc.compose_ref2va([a], [], "", "[Shot 1] <Subject a>.", "", "")


def test_media_follow_core_order_with_soundtracks_before_standalone_audio(pc):
    walk, dolly = object(), object()
    a = pc.Definition("a", "Subject", "walks as in <Video 1>.", "partially_preserved", videos=[walk])
    cam = pc.Definition("dolly", "Video", "provides the dolly move.", "weak_reference", role="structure_reference",
                        videos=[dolly], soundtrack_retention="fully_copy")
    music = pc.Definition("music", "Audio", "is the score.", "fully_copy", role="music_style", audio={"x": 1})
    description = "[Shot 1] <Subject a> walks; the camera follows <Video dolly> over <Audio dolly> and <Audio music>."
    out = pc.compose_ref2va([music, cam, a], [], "", description, "", "")
    assert out["videos"] == [(dolly, True), (walk, False)]
    assert out["audios"] == [{"x": 1}]
    assert "<Subject 1> walks as in <Video 2>." in out["text"]
    assert "<Video 1> provides the dolly move." in out["text"]
    assert "<Audio 1> is the synchronized audio track of <Video 1>." in out["text"]
    assert "<Audio 2> is the score." in out["text"]
    assert "over <Audio 1> and <Audio 2>." in out["text"]
    assert "<Video 1> ([Shot 1] structure reference): weak_reference\n" in out["text"]
    assert "<Audio 1>: fully_copy - the synchronized audio track of <Video 1>" in out["text"]
    assert "[reference generation + audio reuse]" in out["text"]


def test_task_types_and_picture_roles(pc):
    opening = pc.Definition("opening", "Picture", "is the first frame of [Shot 1].", "fully_preserved",
                            role="first_frame", images=[_image(8)])
    out = pc.compose_ref2va([opening], [], "", "[Shot 1] The shot begins from <Picture opening>.", "", "")
    assert "summary:\n[keyframe completion]" in out["text"]
    assert "<Picture 1> ([Shot 1] first frame): fully_preserved" in out["text"]
    chosen = pc.compose_ref2va([opening], ["video editing", "keyframe completion"], "x",
                               "[Shot 1] <Picture opening>.", "", "")
    assert "summary:\n[keyframe completion + video editing] x" in chosen["text"]


def test_keyframe_prompt_carries_the_guides_alignment_sentences(pc):
    body = pc.compose_keyframe("[Shot 1] She (S her) waves.\n[Shot 2] At 00:03.000, she turns.", "Wind.", "")
    assert body["body"] == (
        "integrated_multimodal_description: [Shot 1] She (S1) waves.\n[Shot 2] At 00:03.000, she turns.\n\n"
        "overall_soundscape: Wind.\n\n"
        "non_diegetic_music: N/A"
    )
    assert body["final_shot"] == 2
    assert pc.keyframe_instruction(True, True, 124, 2) == (
        "How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with the "
        "0.00-second mark of the target video; Picture 2 (from Shot 2) aligns with the 5.17-second mark of "
        "the target video."
    )
    assert pc.keyframe_instruction(True, False, 124, 2) == (
        "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
    )
    assert pc.keyframe_instruction(False, True, 192, 1) == (
        "How the reference pictures align with the target video — <Picture 1> (from [Shot 1]) aligns with the "
        "8.00-second mark of the target video."
    )
    assert pc.keyframe_instruction(False, False, 124, 1) == ""


@pytest.mark.parametrize("name", ["", "1st", "fred jones", "<fred>"])
def test_definition_names_must_be_usable_in_labels(pc, name):
    with pytest.raises(ValueError, match="must start with a letter"):
        pc.Definition(name, "Subject")


def test_duplicate_names_are_refused(pc):
    with pytest.raises(ValueError, match="'fred' is used more than once"):
        pc.check_unique_names([pc.Definition("fred", "Subject"), pc.Definition("fred", "Audio", audio={})])
