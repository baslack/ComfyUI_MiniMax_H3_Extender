"""Import space contract: a project that would fill the cache drive is refused before anything is written.

Free space is injected at the OS boundary (shutil.disk_usage) so the test
needs no multi-GB files.
"""
from __future__ import annotations

import io
import json
import shutil
import types
import zipfile

import pytest

from http_helpers import body_json, call, multipart

MARGIN = 1024 * 1024 * 1024


@pytest.fixture
def free_space(monkeypatch):
    def set_free(nbytes):
        real = shutil.disk_usage

        def disk_usage(path):
            usage = real(path)
            return types.SimpleNamespace(total=usage.total, used=usage.used, free=nbytes)

        monkeypatch.setattr(shutil, "disk_usage", disk_usage)

    return set_free


def _archive(payload_bytes):
    meta = {"format": "MiniMax H3 Extender Project", "format_version": 4, "project": {"extender": {"clips": [{"prompt": "x"}]}}}
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("project.json", json.dumps(meta))
        zf.writestr("padding.bin", b"\0" * payload_bytes)  # compresses to almost nothing
    return buffer.getvalue()


def _files(*roots):
    return {p for root in roots for p in root.rglob("*") if p.is_file()}


def _load(ext, archive):
    return call(
        ext.routes, "POST", "/h3_extender/project/load",
        data=multipart(owner_id="space_node", project_file=("project.ext", archive)),
    )


def test_archive_whose_contents_would_not_fit_is_refused_without_writing(ext, comfy_dirs, free_space):
    archive = _archive(4 * 1024 * 1024)
    free_space(MARGIN + 2 * 1024 * 1024)  # room for the upload, not for 4 MiB unpacked
    before = _files(ext.root, comfy_dirs.user)

    response = _load(ext, archive)

    assert response.status == 400
    assert "free" in body_json(response)["error"]
    assert _files(ext.root, comfy_dirs.user) == before


def test_upload_that_would_not_fit_is_refused(ext, comfy_dirs, free_space):
    free_space(MARGIN - 1)
    before = _files(ext.root, comfy_dirs.user)

    response = _load(ext, _archive(16))

    assert response.status == 400
    assert "free" in body_json(response)["error"]
    assert _files(ext.root, comfy_dirs.user) == before


def test_archive_that_fits_still_imports(ext, free_space):
    free_space(MARGIN + 64 * 1024 * 1024)

    response = _load(ext, _archive(1024 * 1024))

    assert response.status == 200, response.body
    assert body_json(response)["ok"] is True
