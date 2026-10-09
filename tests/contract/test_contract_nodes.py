"""Workflow-compatibility contract.

Saved workflows reference nodes by class key, store widget values by position
and links by input slot, so inputs may only ever be appended.
"""
from __future__ import annotations

import pytest

EXTENDER_REQUIRED = [
    "model", "clip", "vae", "run_mode", "width", "height", "ref_image_size", "steps",
    "sampler_name", "scheduler", "denoise", "context_length", "audio_context_length",
    "clips_json", "resolution_mode", "megapixels", "refs_json", "generation_mode",
    "motion_context", "refine_enabled", "refine_scale", "refine_steps", "refine_denoise",
]
EXTENDER_OPTIONAL = [
    "fl2va_model", "audio_vae", "ref_audio", "ref_audio_1", "ref_audio_2", "ref_audio_3",
    "ref_video_1", "ref_video_fps_1", "ref_video_audio_1",
    "ref_video_2", "ref_video_fps_2", "ref_video_audio_2",
    "ref_video_3", "ref_video_fps_3", "ref_video_audio_3",
    "ref_pack", "prompt_pack", "sigmas", "continue_existing_video",
]


@pytest.mark.parametrize(
    ("key", "returns"),
    [
        ("MiniMaxH3Extender", ("H3_MOTION_DISK_CACHE", "INT", "INT", "STRING", "FLOAT", "STRING")),
        ("MiniMaxH3MotionContextDiskFinalDecode", ("VIDEO", "VIDEO_EDITOR_BATCH")),
        ("MiniMaxH3MotionContextDiskJoin", ("H3_MOTION_DISK_CACHE", "LATENT", "INT", "INT", "STRING", "FLOAT", "STRING", "STRING")),
        ("MiniMaxH3MotionContextRAM", ("CONDITIONING", "INT", "INT", "INT", "STRING")),
        ("MiniMaxH3PromptPackBridge", ("H3_PROMPT_PACK", "INT")),
        ("MiniMaxH3ReferencePackBridge", ("H3_REF_PACK", "INT")),
        ("MiniMaxH3TailFromLatent", ("IMAGE", "AUDIO", "IMAGE", "INT", "FLOAT")),
    ],
)
def test_node_is_registered_with_stable_outputs(ext, key, returns):
    assert key in ext.pkg.NODE_CLASS_MAPPINGS
    assert tuple(ext.pkg.NODE_CLASS_MAPPINGS[key].RETURN_TYPES) == returns


def test_extender_widgets_keep_their_positions(ext):
    schema = ext.pkg.NODE_CLASS_MAPPINGS["MiniMaxH3Extender"].INPUT_TYPES()
    assert list(schema["required"])[: len(EXTENDER_REQUIRED)] == EXTENDER_REQUIRED


def test_extender_optional_sockets_only_grow_at_the_end(ext):
    schema = ext.pkg.NODE_CLASS_MAPPINGS["MiniMaxH3Extender"].INPUT_TYPES()
    assert list(schema["optional"])[: len(EXTENDER_OPTIONAL)] == EXTENDER_OPTIONAL


@pytest.mark.parametrize(
    ("key", "required", "optional"),
    [
        ("MiniMaxH3MotionContextDiskFinalDecode", [
            "cache", "vae", "audio_vae", "fps", "filename_prefix", "output_directory", "codec", "crf",
            "preset", "audio_bitrate", "autoplay", "auto_save_project", "save_individual_clips",
        ], []),
        ("MiniMaxH3MotionContextDiskJoin", ["samples", "validated", "run_mode", "fps"], ["previous_cache", "trim_frames"]),
        # context_latent is optional so the first clip of a chain runs without a previous clip.
        ("MiniMaxH3MotionContextRAM", ["conditioning", "latent", "context_length", "audio_context_length"], ["context_latent"]),
        ("MiniMaxH3TailFromLatent", ["samples", "vae", "audio_vae", "tail_seconds", "align_to_h3_grid"], []),
        ("MiniMaxH3ReferencePackBridge", [], [f"ref_{i}" for i in range(1, 10)]),
    ],
)
def test_other_nodes_keep_their_input_order(ext, key, required, optional):
    schema = ext.pkg.NODE_CLASS_MAPPINGS[key].INPUT_TYPES()
    assert list(schema.get("required", {}))[: len(required)] == required
    assert list(schema.get("optional", {}))[: len(optional)] == optional


def test_prompt_pack_bridge_keeps_numbered_prompt_slots(ext):
    schema = ext.pkg.NODE_CLASS_MAPPINGS["MiniMaxH3PromptPackBridge"].INPUT_TYPES()
    names = list(schema.get("required", {})) + list(schema.get("optional", {}))
    assert names[:128] == [f"prompt_{i}" for i in range(1, 129)]
