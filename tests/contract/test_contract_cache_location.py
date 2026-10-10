"""Storage contract: project data lives in ComfyUI's user directory, never inside the node package."""
from __future__ import annotations

from http_helpers import body_json, call, multipart, png_bytes


def _files(root):
    return {p for p in root.rglob("*") if p.is_file()}


def test_uploaded_reference_is_stored_under_the_comfyui_user_directory(ext, comfy_dirs):
    package_before = _files(ext.root)
    user_before = _files(comfy_dirs.user)

    response = call(
        ext.routes, "POST", "/h3_extender/ref/upload",
        data=multipart(ref_file=("ref.png", png_bytes(40, 30, (10, 200, 90)))),
    )

    assert response.status == 200, response.body
    ref_id = body_json(response)["ref"]["id"]
    new_user_files = _files(comfy_dirs.user) - user_before
    assert any(ref_id in p.name for p in new_user_files)
    assert _files(ext.root) == package_before
