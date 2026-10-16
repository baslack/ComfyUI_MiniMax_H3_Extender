"""MiniMax H3 Project: names a clip chain's cache, and manages the caches on disk.

Without a project, a chain's cache is named after the node that starts it, so
two workflows with the same node ids share one cache. A project name keeps
each workflow's clips in their own files.

The node's buttons list, wipe, rename, archive and restore caches through the
routes below; the ``h3cache`` command line does the same with ComfyUI closed.
"""
import asyncio
import time
from pathlib import Path

from . import cache_library
from .motion_context_disk import PROJECT_TYPE, _ensure_cache_root, _project_name

try:
    import folder_paths
    from aiohttp import web
    from server import PromptServer
except Exception:  # static tests outside ComfyUI
    folder_paths = None
    web = None
    PromptServer = None


def _archive_dir():
    return Path(folder_paths.get_output_directory()) / "h3_projects"


class MiniMaxH3Project:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "name": ("STRING", {"default": "", "tooltip": "The cache's name: letters, digits, '-' and '_'. Workflows with the same name share one cache."}),
            },
        }

    RETURN_TYPES = (PROJECT_TYPE, "STRING")
    RETURN_NAMES = ("project", "latent_directory")
    OUTPUT_TOOLTIPS = ("Connect to the chain's first Disk Join, or to Continue Video.",
                       "A folder beside the chain's cache files for this project's Latent Caches (scripts_nodes), "
                       "so they are renamed, wiped, archived and restored with it.")
    FUNCTION = "project"
    CATEGORY = "MiniMax H3"
    DESCRIPTION = "Names the clip chain's cache, and gives the project a folder for its Latent Caches."

    def project(self, name):
        name = _project_name(name)
        return (name, str(_ensure_cache_root() / f"chain_{name}.latents"))


def _change_refused():
    """A wipe or rename while a prompt runs could pull files from under it."""
    running, _ = PromptServer.instance.prompt_queue.get_current_queue_volatile()
    if running:
        return web.json_response({"ok": False, "error": "A prompt is running. Wait for it to finish, or cancel it."}, status=409)
    return None


def _forget_results():
    # Nodes cached in memory still point at the old files; make the next run start fresh.
    queue = PromptServer.instance.prompt_queue
    queue.set_flag("unload_models", False)
    queue.set_flag("free_memory", True)


if web is not None and PromptServer is not None and getattr(PromptServer, "instance", None) is not None:
    @PromptServer.instance.routes.get("/h3_extender/caches")
    async def h3_extender_caches(request):
        return web.json_response({"ok": True, "caches": cache_library.list_caches(_ensure_cache_root())})

    @PromptServer.instance.routes.post("/h3_extender/caches/wipe")
    async def h3_extender_cache_wipe(request):
        refused = _change_refused()
        if refused is not None:
            return refused
        body = await request.json()
        try:
            freed = cache_library.wipe(_ensure_cache_root(), str(body.get("name") or ""))
        except FileNotFoundError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=404)
        _forget_results()
        return web.json_response({"ok": True, "freed_bytes": freed})

    @PromptServer.instance.routes.post("/h3_extender/caches/rename")
    async def h3_extender_cache_rename(request):
        refused = _change_refused()
        if refused is not None:
            return refused
        body = await request.json()
        try:
            name = cache_library.rename(_ensure_cache_root(), str(body.get("old") or ""), str(body.get("new") or ""))
        except (ValueError, FileNotFoundError, FileExistsError) as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        _forget_results()
        return web.json_response({"ok": True, "name": name})

    @PromptServer.instance.routes.post("/h3_extender/caches/archive")
    async def h3_extender_cache_archive(request):
        refused = _change_refused()
        if refused is not None:
            return refused
        body = await request.json()
        name = str(body.get("name") or "")
        workflow = body.get("workflow")
        if not isinstance(workflow, dict):
            return web.json_response({"ok": False, "error": "The workflow is missing."}, status=400)
        out = _archive_dir() / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}{cache_library.ARCHIVE_SUFFIX}"
        try:
            result = await asyncio.to_thread(
                cache_library.archive, _ensure_cache_root(), name, workflow, folder_paths.get_input_directory(), out
            )
        except FileNotFoundError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=404)
        except OSError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=507)
        return web.json_response({"ok": True, **result})

    @PromptServer.instance.routes.get("/h3_extender/caches/archives")
    async def h3_extender_cache_archives(request):
        return web.json_response({"ok": True, "folder": str(_archive_dir()), "archives": cache_library.list_archives(_archive_dir())})

    @PromptServer.instance.routes.post("/h3_extender/caches/restore")
    async def h3_extender_cache_restore(request):
        refused = _change_refused()
        if refused is not None:
            return refused
        body = await request.json()
        file = str(body.get("file") or "")
        path = _archive_dir() / file
        if Path(file).name != file or not file.endswith(cache_library.ARCHIVE_SUFFIX) or not path.is_file():
            return web.json_response({"ok": False, "error": f"No archive named '{file}' in {_archive_dir()}."}, status=404)
        workflows = Path(folder_paths.get_user_directory()) / "default" / "workflows"
        try:
            result = await asyncio.to_thread(
                cache_library.restore, _ensure_cache_root(), path, folder_paths.get_input_directory(), workflows,
                str(body.get("name") or "") or None,
            )
        except (ValueError, FileExistsError) as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=400)
        except OSError as exc:
            return web.json_response({"ok": False, "error": str(exc)}, status=507)
        return web.json_response({"ok": True, **result})


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3Project": MiniMaxH3Project,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3Project": "MiniMax H3 Project",
}
