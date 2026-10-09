"""Core-patch contract: loading the package leaves ComfyUI's own MiniMax H3 code untouched."""
from __future__ import annotations


def test_importing_the_package_does_not_patch_comfyui(ext):
    import comfy.ldm.minimax.model as minimax_model
    import comfy.model_base as model_base

    assert model_base.MiniMaxH3.extra_conds.__module__ == "comfy.model_base"
    assert minimax_model.PackedLayout.__init__.__module__ == "comfy.ldm.minimax.model"
