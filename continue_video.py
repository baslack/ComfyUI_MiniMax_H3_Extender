"""MiniMax H3 Continue Video: start a clip chain from an existing video.

The source becomes the chain's Clip 0, exactly as in the Extender node: a
24 fps working copy at the chain's size is kept with the cache, Final Decode
puts it in front of the generated clips, and the first clip continues from its
tail through Motion Context RAM.
"""
import time

from .extender import (
    FPS,
    _build_continue_context_latent,
    _clear_continue_source_derivatives,
    _continue_source_working_path,
    _continue_video_from_native_input,
    _normalize_continue_source,
)
from .motion_context_disk import (
    CACHE_TYPE,
    _make_handle,
    _manifest_for_first,
    _truncate_chain,
    _write_json_atomic,
)
from .motion_context_ram import SOURCE_GUIDE_KEY


class MiniMaxH3ContinueVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {"tooltip": "The video to continue, from Load Video."}),
                "vae": ("VAE",),
                "audio_vae": ("VAE",),
                "width": ("INT", {"default": 1280, "min": 32, "max": 4096, "step": 16, "tooltip": "The clips' final size; the source is resized to it."}),
                "height": ("INT", {"default": 704, "min": 32, "max": 4096, "step": 16}),
                "context_length": (["22", "5", "39", "56"], {"default": "22", "tooltip": "Set the first clip's Motion Context RAM to the same value."}),
                "audio_context_length": ("INT", {"default": 0, "min": 0, "max": 240, "step": 1}),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = (CACHE_TYPE, "LATENT")
    RETURN_NAMES = ("cache", "context_latent")
    FUNCTION = "start"
    CATEGORY = "MiniMax H3"
    DESCRIPTION = (
        "Continue an existing video: connect cache to the first clip's Disk Join (previous_cache) "
        "and context_latent to its Motion Context RAM."
    )

    def start(self, video, vae, audio_vae, width, height, context_length="22", audio_context_length=0, unique_id=None):
        desc = _continue_video_from_native_input(video)
        data_path, manifest_path, manifest = _manifest_for_first(unique_id if unique_id is not None else "continue_video", FPS)

        source = manifest.get("source_video") if isinstance(manifest.get("source_video"), dict) else {}
        if (
            source.get("id") != desc["id"]
            or (int(source.get("width", 0)), int(source.get("height", 0))) != (int(width), int(height))
            or not _continue_source_working_path(data_path).exists()
        ):
            # A different source or size invalidates every generated clip.
            if manifest.get("segments"):
                manifest = _truncate_chain(data_path, manifest_path, manifest, 0)
            _clear_continue_source_derivatives(data_path)
            source = _normalize_continue_source(data_path, desc, int(width), int(height), "manual")
            manifest = dict(manifest)
            manifest["source_video"] = source
            manifest["updated_at"] = time.time()
            _write_json_atomic(manifest_path, manifest)

        context, guide = _build_continue_context_latent(
            vae, audio_vae, data_path, source, int(context_length), int(audio_context_length)
        )
        context[SOURCE_GUIDE_KEY] = guide

        handle = _make_handle(data_path, manifest_path, manifest, "full_batch", next_index=0)
        # The first clip's Disk Join keeps its own run mode.
        handle.pop("run_mode")
        return (handle, context)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3ContinueVideo": MiniMaxH3ContinueVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3ContinueVideo": "MiniMax H3 Continue Video",
}
