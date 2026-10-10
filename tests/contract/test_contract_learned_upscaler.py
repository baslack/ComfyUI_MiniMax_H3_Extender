"""Latent Refine contract: upscaling comes from an Upscaler-Plus provider on `learned_upscaler`.

The provider contract (kind, api_version, upscale_clean_video) is owned by
Comfyui_Minimax_h3_latent_Upscaler-Plus; the stub below honours it, and the
cross-repo test checks the real provider when that package is available
(set H3_UPSCALER_PLUS_PATH to its checkout).
"""
from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

KIND = "minimax_h3_learned_latent_upscaler"
SOURCE = "huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler"


class StubProvider:
    kind = KIND
    api_version = 1
    model_name = "stub.safetensors"
    precision = "fp32"

    def __init__(self):
        self.calls = []

    def upscale_clean_video(self, video, *, target_h, target_w):
        self.calls.append((tuple(video.shape), target_h, target_w))
        return F.interpolate(video, size=(video.shape[2], target_h, target_w), mode="nearest")


def _engine(ext):
    return importlib.import_module(f"{ext.pkg.__name__}.latent_refine_engine")


def _joint(video_shape=(1, 24, 7, 48, 84), audio_t=100):
    import comfy.nested_tensor

    video = torch.randn(*video_shape)
    audio = torch.randn(1, 32, 2, audio_t)
    return video, audio, {"samples": comfy.nested_tensor.NestedTensor([video, audio])}


def _extend_with_refine(ext, learned_upscaler):
    node = ext.pkg.NODE_CLASS_MAPPINGS["MiniMaxH3Extender"]()
    kwargs = {} if learned_upscaler is None else {"learned_upscaler": learned_upscaler}
    return node.extend(
        model=None, clip=None, vae=None, run_mode="single_clip", width=1344, height=768,
        ref_image_size="match", steps=8, sampler_name="euler", scheduler="simple", denoise=1.0,
        context_length="22", audio_context_length=0, clips_json="", refine_enabled=True,
        unique_id="refine_contract", **kwargs,
    )


def test_learned_upscaler_is_the_last_optional_socket(ext):
    optional = ext.pkg.NODE_CLASS_MAPPINGS["MiniMaxH3Extender"].INPUT_TYPES()["optional"]

    assert list(optional)[-1] == "learned_upscaler"
    assert optional["learned_upscaler"][0] == "H3_LATENT_UPSCALER"


@pytest.mark.parametrize(
    "provider",
    [
        None,
        type("WrongKind", (), {"kind": "something_else", "api_version": 1})(),
        type("FutureApi", (), {"kind": KIND, "api_version": 2})(),
    ],
    ids=["missing", "wrong-kind", "wrong-api-version"],
)
def test_refine_without_a_usable_provider_says_what_to_get_and_where(ext, provider):
    with pytest.raises(ValueError) as error:
        _extend_with_refine(ext, provider)

    message = str(error.value)
    assert "learned_upscaler" in message
    assert SOURCE in message
    assert "latent_upscale_models" in message


def test_refine_upscale_uses_the_provider_and_keeps_audio_exact(ext):
    engine = _engine(ext)
    video, audio, sampled = _joint()
    provider = StubProvider()

    latent, width, height, first_audio = engine.upscale_for_refine(sampled, 1.5, provider)

    assert (width, height) == engine.target_dimensions(84 * 16, 48 * 16, 1.5)
    assert provider.calls == [((1, 24, 7, 48, 84), height // 16, width // 16)]
    up_video, kept_audio = latent["samples"].unbind()
    assert up_video.shape == (1, 24, 7, height // 16, width // 16)
    assert torch.equal(kept_audio, audio)
    assert torch.equal(first_audio, audio)


def _upscaler_plus():
    path = os.environ.get("H3_UPSCALER_PLUS_PATH")
    if not path:
        pytest.skip("set H3_UPSCALER_PLUS_PATH to a Comfyui_Minimax_h3_latent_Upscaler-Plus checkout")
    root = Path(path)
    name = "h3_upscaler_plus_for_extender_contract"
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, root / "__init__.py", submodule_search_locations=[str(root)])
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
    return (
        importlib.import_module(f"{name}.nodes.minimax_h3_handoff_provider"),
        importlib.import_module(f"{name}.nodes.minimax_h3_latent_upscaler_3d"),
    )


def test_real_upscaler_plus_provider_drives_the_refine_upscale(ext, tmp_path, monkeypatch):
    from safetensors.torch import save_file
    import folder_paths

    provider_module, lbh = _upscaler_plus()
    model = lbh.LatentResizer3D(in_channels=24, in_blocks=1, out_blocks=1, channels=32, dropout=0.0)
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(tmp_path / "tiny.safetensors"))
    paths, extensions = folder_paths.folder_names_and_paths["latent_upscale_models"]
    monkeypatch.setitem(folder_paths.folder_names_and_paths, "latent_upscale_models", ([str(tmp_path), *paths], extensions))
    provider = provider_module.H3LatentUpscalerProvider(model_name="tiny.safetensors", device="cpu", precision="fp32")

    assert (provider.kind, provider.api_version) == (KIND, 1)
    video, audio, sampled = _joint(video_shape=(1, 24, 2, 4, 6), audio_t=20)
    latent, width, height, _ = _engine(ext).upscale_for_refine(sampled, 2.0, provider)

    up_video, kept_audio = latent["samples"].unbind()
    assert up_video.shape == (1, 24, 2, height // 16, width // 16)
    assert torch.equal(kept_audio, audio)
