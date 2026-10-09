"""Final Decode contract: FL2VA and independent Ref2VA caches reach the hard-cut exporter.

Final Decode dispatches on the cache's sequence_mode. Every argument it passes
must be accepted by the exporter it dispatches to, or those modes fail before
decoding anything.
"""
from __future__ import annotations

import importlib
import inspect
import json

import pytest
import torch


def _av():
    import comfy.nested_tensor

    g = torch.Generator().manual_seed(0)
    video = torch.rand(1, 24, 7, 4, 6, generator=g)
    audio = torch.rand(1, 32, 2, 37, generator=g)
    return {"samples": comfy.nested_tensor.NestedTensor((video, audio))}


@pytest.mark.parametrize("sequence_mode", ["fl2va", "ref2va_independent"])
def test_final_decode_arguments_fit_the_hard_cut_exporter(ext, monkeypatch, sequence_mode):
    fl2va = importlib.import_module(f"{ext.pkg.__name__}.fl2va_engine")
    real = fl2va.export_fl2va_final
    calls = []

    def recording_export(**kwargs):
        calls.append(kwargs)
        return {"ui": {}, "result": (None, None)}

    monkeypatch.setattr(fl2va, "export_fl2va_final", recording_export)

    nodes = ext.pkg.NODE_CLASS_MAPPINGS
    handle = nodes["MiniMaxH3MotionContextDiskJoin"]().join(
        samples=_av(), validated=False, run_mode="full_batch", fps=24.0,
        unique_id=f"final_decode_modes_{sequence_mode}")[0]
    with open(handle["manifest_path"], encoding="utf-8") as f:
        manifest = json.load(f)
    manifest["sequence_mode"] = sequence_mode
    with open(handle["manifest_path"], "w", encoding="utf-8") as f:
        json.dump(manifest, f)

    nodes["MiniMaxH3MotionContextDiskFinalDecode"]().export(
        cache=handle, vae=None, audio_vae=None, fps=24.0, filename_prefix="modes", output_directory="",
        codec="H.264", crf=17, preset="fast", audio_bitrate="192k", autoplay=False,
        auto_save_project=False, save_individual_clips=False, unique_id="1", prompt={}, extra_pnginfo=None,
    )

    assert len(calls) == 1
    inspect.signature(real).bind(**calls[0])
