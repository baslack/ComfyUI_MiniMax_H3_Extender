from __future__ import annotations

import comfy.model_management
import comfy.nested_tensor

from .motion_context_ram import _streams_from_latent

VAE_DOWNSAMPLE = 16


def target_dimensions(width: int, height: int, scale: float):
    scale = max(1.0, float(scale))
    w = max(32, int(round((int(width) * scale) / 32.0)) * 32)
    h = max(32, int(round((int(height) * scale) / 32.0)) * 32)
    return w, h


def upscale_for_refine(sampled, scale: float, learned_upscaler):
    """Upscale only the H3 video latent with the learned upscaler; preserve first-pass audio exactly."""
    video, audio = _streams_from_latent(sampled, "samples")
    width, height = target_dimensions(video.shape[4] * VAE_DOWNSAMPLE, video.shape[3] * VAE_DOWNSAMPLE, scale)
    up_video = learned_upscaler.upscale_clean_video(
        video, target_h=height // VAE_DOWNSAMPLE, target_w=width // VAE_DOWNSAMPLE
    )
    audio = audio.to(device=comfy.model_management.intermediate_device())
    up_video = up_video.to(device=audio.device)
    latent = sampled.copy()
    latent["samples"] = comfy.nested_tensor.NestedTensor((up_video, audio))
    return latent, width, height, audio


def preserve_first_pass_audio(refined, first_pass_audio):
    video, _ = _streams_from_latent(refined, "samples")
    out = refined.copy()
    out["samples"] = comfy.nested_tensor.NestedTensor(
        (video, first_pass_audio.to(device=video.device))
    )
    return out
