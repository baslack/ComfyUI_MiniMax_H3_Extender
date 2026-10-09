"""Portable project (.ext) contract: round trip, and refusal of hostile or broken archives."""
from __future__ import annotations

import io
import json
import zipfile

import pytest

from http_helpers import body_json, call, multipart, png_bytes

PROMPT = "a fennec-eared girl with long blonde hair walks through falling snow"


def _upload_ref(ext):
    response = call(ext.routes, "POST", "/h3_extender/ref/upload", data=multipart(ref_file=("ref.png", png_bytes())))
    assert response.status == 200, response.body
    return body_json(response)["ref"]


def _export(ext, project, owner="export_node"):
    prepared = call(
        ext.routes, "POST", "/h3_extender/project/prepare_save",
        json={"owner_id": owner, "project_name": "Contract", "project": project},
    )
    assert prepared.status == 200, prepared.body
    downloaded = call(ext.routes, "GET", "/h3_extender/project/download", params={"token": body_json(prepared)["token"]})
    assert downloaded.status == 200
    return downloaded.body


def _load(ext, archive, owner="import_node"):
    return call(
        ext.routes, "POST", "/h3_extender/project/load",
        data=multipart(owner_id=owner, project_file=("project.ext", archive)),
    )


def _zip(entries):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buffer.getvalue()


def _project_json(project=None, **overrides):
    meta = {"format": "MiniMax H3 Extender Project", "format_version": 4, "project": project or {"extender": {"clips": [{"prompt": "x"}]}}}
    meta.update(overrides)
    return json.dumps(meta)


def _files_under(*roots):
    return {p for root in roots for p in root.rglob("*") if p.is_file()}


def test_project_round_trips_prompts_and_embedded_references(ext):
    ref = _upload_ref(ext)
    archive = _export(ext, {"extender": {"clips": [{"prompt": PROMPT}], "references": [ref]}})

    response = _load(ext, archive)

    assert response.status == 200, response.body
    loaded = body_json(response)
    assert loaded["ok"] is True
    assert loaded["project"]["extender"]["clips"][0]["prompt"] == PROMPT
    assert loaded["references"]["count"] == 1


@pytest.mark.parametrize(
    "entries",
    [
        pytest.param({"project.json": _project_json(), "../escaped.txt": "x"}, id="parent-traversal"),
        pytest.param({"project.json": _project_json(), "cache/../../escaped.txt": "x"}, id="nested-traversal"),
        pytest.param({"project.json": _project_json(), "/escaped.txt": "x"}, id="absolute-path"),
        pytest.param({"readme.txt": "no project.json"}, id="missing-project-json"),
        pytest.param({"project.json": _project_json(format="Something Else")}, id="foreign-format"),
        pytest.param({"project.json": _project_json(format_version=99)}, id="future-version"),
    ],
)
def test_hostile_or_broken_archives_are_refused_without_writing_anything(ext, comfy_dirs, tmp_path, entries):
    before = _files_under(ext.root, comfy_dirs.user, tmp_path.parent)

    response = _load(ext, _zip(entries))

    assert response.status == 400
    assert body_json(response)["ok"] is False
    assert _files_under(ext.root, comfy_dirs.user, tmp_path.parent) == before


def test_embedded_media_that_fails_its_hash_is_refused(ext):
    media_id = "b" * 64
    project = {"extender": {"clips": [{
        "prompt": "x",
        "local_refs": {"videos": [{"slot": 1, "media": {"id": media_id, "kind": "video", "original_name": "v.mp4"}}]},
    }]}}
    archive = _zip({"project.json": _project_json(project), f"local_refs/media/{media_id}.bin": b"not the hashed bytes"})

    response = _load(ext, archive)

    assert response.status == 400
    assert "integrity" in body_json(response)["error"]


def test_a_file_that_is_not_a_zip_is_refused(ext):
    response = _load(ext, b"definitely not a zip archive")

    assert response.status == 400
    assert body_json(response)["ok"] is False
