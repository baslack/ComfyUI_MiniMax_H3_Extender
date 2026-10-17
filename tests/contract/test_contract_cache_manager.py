"""Cache manager contract: list, wipe, rename, archive and restore clip caches.

The Project node's buttons go through the routes, and the h3cache command
line works with ComfyUI closed. Changes refuse while something could be
using the files: a running prompt (routes) or a running ComfyUI (command
line). An archive carries a cache, its workflow and the inputs it loads, but
not the previews and decoded videos the next Final Decode makes again.
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import zipfile
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
    return subprocess.run(
        [sys.executable, str(ext.root / "h3cache.py"), "--comfyui", str(Path(comfy_dirs.user).parent), "--port", str(port), *args],
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


def _workflow(project, image):
    return {
        "id": "workflow-id",
        "nodes": [
            {"id": 1, "type": "MiniMaxH3Project", "widgets_values": [project], "widgets_values_named": {"name": project}},
            {"id": 2, "type": "LoadImage", "widgets_values": [image, "image"]},
        ],
        "links": [],
        "definitions": {"subgraphs": []},
    }


def _archive(ext, name, image):
    response = _post(ext, "/h3_extender/caches/archive", name=name, workflow=_workflow(name, image))
    assert response.status == 200
    return Path(body_json(response)["path"])


def test_archive_holds_the_cache_its_workflow_and_inputs_but_not_rebuilt_files(ext, comfy_dirs):
    cache_dir = Path(_start(ext, "archive-a")["data_path"]).parent
    # What Final Decode leaves next to the latents; the next run makes it again.
    (cache_dir / "chain_archive-a.preview.mp4").write_bytes(b"preview")
    (cache_dir / "chain_archive-a.final.video").mkdir()
    (cache_dir / "chain_archive-a.final.video" / "ref2va_0000.mp4").write_bytes(b"final")
    (Path(comfy_dirs.input) / "archive-a-ref.png").write_bytes(b"reference")
    (Path(comfy_dirs.input) / "unused.png").write_bytes(b"unused")

    names = set(zipfile.ZipFile(_archive(ext, "archive-a", "archive-a-ref.png")).namelist())
    assert {"archive.json", "workflow.json", "inputs/archive-a-ref.png",
            "cache/chain_archive-a.h3cache", "cache/chain_archive-a.json"} <= names
    assert not [n for n in names if ".preview." in n or ".final.video" in n or "unused" in n]


def test_restore_brings_back_the_cache_inputs_and_workflow(ext, comfy_dirs):
    _start(ext, "archive-b")
    image = Path(comfy_dirs.input) / "archive-b-ref.png"
    image.write_bytes(b"original reference")
    archive = _archive(ext, "archive-b", "archive-b-ref.png")
    image.write_bytes(b"a different picture with the same name")

    response = _post(ext, "/h3_extender/caches/restore", file=archive.name, name="archive-b-copy")
    assert response.status == 200
    result = body_json(response)
    assert result["name"] == "archive-b-copy"
    assert result["renamed_inputs"] == {"archive-b-ref.png": "archive-b-ref_restored1.png"}
    assert (Path(comfy_dirs.input) / "archive-b-ref_restored1.png").read_bytes() == b"original reference"
    assert image.read_bytes() == b"a different picture with the same name"

    workflow = json.loads((Path(comfy_dirs.user) / "default" / "workflows" / result["workflow"]).read_text(encoding="utf-8"))
    nodes = {n["type"]: n for n in workflow["nodes"]}
    assert nodes["MiniMaxH3Project"]["widgets_values"] == ["archive-b-copy"]
    assert nodes["LoadImage"]["widgets_values"][0] == "archive-b-ref_restored1.png"
    assert workflow["id"] != "workflow-id"
    # The restored cache carries on: its clip is there to validate.
    validated = _start(ext, "archive-b-copy", samples=None, validated=True)
    assert Path(validated["manifest_path"]).name == "chain_archive-b-copy.json"


def test_restore_refuses_a_taken_name_a_file_elsewhere_and_a_running_prompt(ext):
    _start(ext, "archive-c")
    archive = _archive(ext, "archive-c", "none.png")
    assert _post(ext, "/h3_extender/caches/restore", file=archive.name).status == 400
    assert _post(ext, "/h3_extender/caches/restore", file="../" + archive.name).status == 404
    ext.queue.running.append("prompt")
    try:
        assert _post(ext, "/h3_extender/caches/restore", file=archive.name, name="archive-c2").status == 409
        assert _post(ext, "/h3_extender/caches/archive", name="archive-c", workflow=_workflow("archive-c", "none.png")).status == 409
    finally:
        ext.queue.running.clear()
    assert "archive-c2" not in _caches(ext)


def test_command_line_archives_and_restores(ext, comfy_dirs, tmp_path):
    _start(ext, "cli-arch")
    workflow = tmp_path / "cli-arch.json"
    workflow.write_text(json.dumps(_workflow("cli-arch", "none.png")), encoding="utf-8")
    port = _free_port()
    out = tmp_path / "cli-arch.h3project"
    assert _cli(ext, comfy_dirs, port, "archive", "cli-arch", "--workflow", str(workflow), "--out", str(out)).returncode == 0

    with socket.socket() as comfyui:
        comfyui.bind(("127.0.0.1", 0))
        comfyui.listen()
        busy = comfyui.getsockname()[1]
        refused = _cli(ext, comfy_dirs, busy, "restore", str(out), "--name", "cli-arch-2")
        assert refused.returncode == 1 and "ComfyUI is running" in refused.stderr
        # Archiving only reads, so it works while ComfyUI runs.
        assert _cli(ext, comfy_dirs, busy, "archive", "cli-arch", "--workflow", str(workflow),
                    "--out", str(tmp_path / "again.h3project")).returncode == 0

    assert _cli(ext, comfy_dirs, port, "restore", str(out), "--name", "cli-arch-2").returncode == 0
    assert _caches(ext)["cli-arch-2"]["clips"] == 1

    # A ComfyUI that has never made a cache has no cache folder yet.
    fresh = tmp_path / "fresh-comfyui"
    restored = subprocess.run(
        [sys.executable, str(ext.root / "h3cache.py"), "--comfyui", str(fresh), "--port", str(port), "restore", str(out)],
        capture_output=True, text=True,
    )
    assert restored.returncode == 0, restored.stderr
    assert (fresh / "user" / "default" / "h3_extender_cache" / "chain_cli-arch.json").is_file()
    assert (fresh / "user" / "default" / "workflows" / "cli-arch.json").is_file()


def test_a_project_s_latent_caches_go_with_it(ext, comfy_dirs):
    project = ext.pkg.NODE_CLASS_MAPPINGS["MiniMaxH3Project"]()
    name, latents, _ = project.project("latents-a")
    _start(ext, name)
    latents = Path(latents)
    assert latents.parent == Path(comfy_dirs.user) / "default" / "h3_extender_cache"
    latents.mkdir()
    (latents / "clip_1.latent").write_bytes(b"first pass")
    with_latents = _caches(ext)["latents-a"]["size_bytes"]

    assert _post(ext, "/h3_extender/caches/rename", old="latents-a", new="latents-b").status == 200
    moved = Path(project.project("latents-b")[1]) / "clip_1.latent"
    assert moved.read_bytes() == b"first pass" and not latents.exists()
    assert _caches(ext)["latents-b"]["size_bytes"] == with_latents

    archive = _archive(ext, "latents-b", "none.png")
    assert "cache/chain_latents-b.latents/clip_1.latent" in zipfile.ZipFile(archive).namelist()
    assert _post(ext, "/h3_extender/caches/restore", file=archive.name, name="latents-c").status == 200
    restored = Path(project.project("latents-c")[1]) / "clip_1.latent"
    assert restored.read_bytes() == b"first pass"

    assert _post(ext, "/h3_extender/caches/wipe", name="latents-c").status == 200
    assert not restored.parent.exists()


def test_rename_project_moves_both_chains_and_the_latent_caches(ext):
    project = ext.pkg.NODE_CLASS_MAPPINGS["MiniMaxH3Project"]()
    name, latents, refined = project.project("whole-a")
    assert refined == "whole-a_refined"
    _start(ext, name)
    _start(ext, refined)
    Path(latents).mkdir()
    (Path(latents) / "clip_1.latent").write_bytes(b"first pass")

    ext.queue.flags.clear()
    response = _post(ext, "/h3_extender/caches/rename_project", old="whole-a", new="whole-b")
    assert response.status == 200
    assert sorted(body_json(response)["renamed"]) == ["whole-a", "whole-a_refined"]
    caches = _caches(ext)
    assert {"whole-b", "whole-b_refined"} <= set(caches) and not {"whole-a", "whole-a_refined"} & set(caches)
    assert (Path(project.project("whole-b")[1]) / "clip_1.latent").read_bytes() == b"first pass"
    assert ext.queue.flags == {"unload_models": False, "free_memory": True}


def test_rename_project_refuses_a_taken_name_without_moving_anything(ext):
    _start(ext, "whole-c")
    _start(ext, "whole-c_refined")
    _start(ext, "whole-d_refined")
    assert _post(ext, "/h3_extender/caches/rename_project", old="whole-c", new="whole-d").status == 400
    assert {"whole-c", "whole-c_refined", "whole-d_refined"} <= set(_caches(ext))
    assert _post(ext, "/h3_extender/caches/rename_project", old="whole-c", new="../escape").status == 400
    ext.queue.running.append("prompt")
    try:
        assert _post(ext, "/h3_extender/caches/rename_project", old="whole-c", new="whole-e").status == 409
    finally:
        ext.queue.running.clear()


def test_rename_project_without_caches_only_renames(ext):
    response = _post(ext, "/h3_extender/caches/rename_project", old="never-ran", new="never-ran-b")
    assert response.status == 200 and body_json(response) == {"ok": True, "name": "never-ran-b", "renamed": []}


def test_project_inputs_are_naming_only_for_latent_cache(ext):
    # scripts_nodes' Latent Cache leaves inputs with this option out of its upstream hash, so a
    # renamed project doesn't re-run clips whose latents are cached.
    nodes = ext.pkg.NODE_CLASS_MAPPINGS
    for node in ("MiniMaxH3MotionContextDiskJoin", "MiniMaxH3ContinueVideo"):
        assert nodes[node].INPUT_TYPES()["optional"]["project"][1]["latent_cache_ignore"] is True
