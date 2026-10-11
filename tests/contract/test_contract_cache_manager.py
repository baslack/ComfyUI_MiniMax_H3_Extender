"""Cache manager contract: list, wipe and rename clip caches.

The Project node's buttons go through the routes, and the h3cache command
line works with ComfyUI closed. Wipe and rename refuse while something could
be using the files: a running prompt (routes) or a running ComfyUI
(command line).
"""
from __future__ import annotations

import socket
import subprocess
import sys
from pathlib import Path

import torch

from http_helpers import body_json, call


def _av(seed):
    import comfy.nested_tensor

    g = torch.Generator().manual_seed(seed)
    return {"samples": comfy.nested_tensor.NestedTensor((torch.rand(1, 24, 12, 6, 8, generator=g), torch.rand(1, 32, 2, 70, generator=g)))}


def _start(ext, project, **kwargs):
    kwargs.setdefault("samples", _av(1))
    kwargs.setdefault("validated", False)
    return ext.pkg.NODE_CLASS_MAPPINGS["MiniMaxH3MotionContextDiskJoin"]().join(
        run_mode="full_batch", fps=24.0, unique_id="cache_manager", project=project, **kwargs)[0]


def _caches(ext):
    return {c["name"]: c for c in body_json(call(ext.routes, "GET", "/h3_extender/caches"))["caches"]}


def _post(ext, path, **body):
    return call(ext.routes, "POST", path, json=body)


def test_routes_list_rename_and_wipe_a_cache(ext):
    handle = _start(ext, "manager-a")
    listed = _caches(ext)["manager-a"]
    assert listed["clips"] == 1 and listed["size_bytes"] > 0

    ext.queue.flags.clear()
    assert _post(ext, "/h3_extender/caches/rename", old="manager-a", new="manager-b").status == 200
    assert "manager-a" not in _caches(ext)
    # The renamed cache carries on under its new name: the clip is there to validate.
    validated = _start(ext, "manager-b", samples=None, validated=True)
    assert Path(validated["manifest_path"]).name == "chain_manager-b.json"
    assert ext.queue.flags == {"unload_models": False, "free_memory": True}

    ext.queue.flags.clear()
    response = _post(ext, "/h3_extender/caches/wipe", name="manager-b")
    assert response.status == 200 and body_json(response)["freed_bytes"] > 0
    assert not list(Path(handle["data_path"]).parent.glob("chain_manager-b.*"))
    assert ext.queue.flags == {"unload_models": False, "free_memory": True}


def test_routes_refuse_changes_while_a_prompt_runs(ext):
    _start(ext, "manager-busy")
    ext.queue.running.append("prompt")
    try:
        assert _post(ext, "/h3_extender/caches/wipe", name="manager-busy").status == 409
        assert _post(ext, "/h3_extender/caches/rename", old="manager-busy", new="manager-idle").status == 409
    finally:
        ext.queue.running.clear()
    assert "manager-busy" in _caches(ext)


def test_routes_refuse_a_taken_or_invalid_name_and_an_unknown_cache(ext):
    _start(ext, "manager-x")
    _start(ext, "manager-y")
    assert _post(ext, "/h3_extender/caches/rename", old="manager-x", new="manager-y").status == 400
    assert _post(ext, "/h3_extender/caches/rename", old="manager-x", new="../escape").status == 400
    assert _post(ext, "/h3_extender/caches/wipe", name="no-such-cache").status == 404
    assert {"manager-x", "manager-y"} <= set(_caches(ext))


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _cli(ext, comfy_dirs, port, *args):
    cache_dir = Path(comfy_dirs.user) / "h3_extender_cache"
    return subprocess.run(
        [sys.executable, str(ext.root / "h3cache.py"), "--cache-dir", str(cache_dir), "--port", str(port), *args],
        capture_output=True, text=True,
    )


def test_command_line_lists_renames_and_wipes(ext, comfy_dirs):
    _start(ext, "cli-a")
    port = _free_port()
    assert "cli-a" in _cli(ext, comfy_dirs, port, "list").stdout

    dry = _cli(ext, comfy_dirs, port, "wipe", "cli-a")
    assert dry.returncode == 1 and "--yes" in dry.stdout and "cli-a" in _caches(ext)

    assert _cli(ext, comfy_dirs, port, "rename", "cli-a", "cli-b").returncode == 0
    assert "cli-b" in _caches(ext) and "cli-a" not in _caches(ext)
    assert _cli(ext, comfy_dirs, port, "wipe", "cli-b", "--yes").returncode == 0
    assert "cli-b" not in _caches(ext)


def test_command_line_refuses_changes_while_comfyui_runs(ext, comfy_dirs):
    _start(ext, "cli-busy")
    with socket.socket() as comfyui:
        comfyui.bind(("127.0.0.1", 0))
        comfyui.listen()
        port = comfyui.getsockname()[1]
        refused = _cli(ext, comfy_dirs, port, "wipe", "cli-busy", "--yes")
        assert refused.returncode == 1 and "ComfyUI is running" in refused.stderr
        assert _cli(ext, comfy_dirs, port, "rename", "cli-busy", "cli-idle").returncode == 1
        assert _cli(ext, comfy_dirs, port, "list").returncode == 0
    assert "cli-busy" in _caches(ext)
