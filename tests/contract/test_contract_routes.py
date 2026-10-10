"""HTTP route contract: what the frontend calls, and what the routes must never serve or write."""
from __future__ import annotations

import pytest

from http_helpers import body_json, call, multipart, png_bytes

HEX64 = "a" * 64


@pytest.fixture
def uploaded_ref(ext):
    response = call(ext.routes, "POST", "/h3_extender/ref/upload", data=multipart(ref_file=("ref.png", png_bytes())))
    assert response.status == 200, response.body
    return body_json(response)["ref"]


def test_reference_upload_returns_a_content_id_that_serves_the_image(ext, uploaded_ref):
    assert len(uploaded_ref["id"]) == 64

    response = call(ext.routes, "GET", "/h3_extender/ref/image", params={"id": uploaded_ref["id"]})

    assert response.status == 200
    assert response.body.startswith(b"\x89PNG")


@pytest.mark.parametrize("ref_id", ["../../secret", "..\\..\\secret", "A" * 63, "g" * 64, ""])
def test_reference_image_rejects_ids_that_are_not_content_hashes(ext, ref_id):
    response = call(ext.routes, "GET", "/h3_extender/ref/image", params={"id": ref_id})
    assert response.status == 400


def test_unknown_reference_id_is_not_found(ext):
    assert call(ext.routes, "GET", "/h3_extender/ref/image", params={"id": HEX64}).status == 404


@pytest.mark.parametrize("media_id", ["..%2F..%2Fsecret", "secret", "A" * 64])
def test_local_media_rejects_ids_that_are_not_content_hashes(ext, media_id):
    assert call(ext.routes, "GET", f"/h3_extender/local_media/{media_id}").status in (400, 404)


def test_fl2va_frame_route_never_serves_a_path_outside_its_cache(ext, comfy_dirs):
    (comfy_dirs.user / "secret.continuity.png").write_bytes(png_bytes())

    for owner_id, clip_id in [("../../user", "secret"), ("..\\..", "..\\user\\secret"), ("x", "../../../user/secret")]:
        response = call(
            ext.routes, "GET", "/h3_extender/fl2va/last_frame", params={"owner_id": owner_id, "clip_id": clip_id},
        )
        assert response.status == 404


def test_continue_video_source_serves_input_files_only(ext, comfy_dirs):
    (comfy_dirs.input / "clip.mp4").write_bytes(b"input video bytes")
    (comfy_dirs.user / "private.mp4").write_bytes(b"private bytes")
    (comfy_dirs.output / "render.mp4").write_bytes(b"output bytes")

    served = call(ext.routes, "GET", "/h3_extender/continue_video/source", params={"file": "clip.mp4"})
    assert served.status == 200 and served.body == b"input video bytes"

    for name in ["../user/private.mp4", "..\\user\\private.mp4", "render.mp4 [output]", str(comfy_dirs.user / "private.mp4")]:
        response = call(ext.routes, "GET", "/h3_extender/continue_video/source", params={"file": name})
        assert response.status == 400, name
        assert b"private bytes" not in response.body and b"output bytes" not in response.body


@pytest.mark.parametrize(
    "payload",
    [
        {"filename": "../evil.mp4", "type": "temp"},
        {"filename": "h3_motion_preview_x_0.mp4", "type": "output"},
        {"filename": "h3_motion_preview_x_0.mp4", "type": "temp", "subfolder": ".."},
        {"filename": "anything.mp4", "type": "temp"},
    ],
)
def test_save_preview_only_accepts_the_extender_temp_preview(ext, comfy_dirs, payload):
    before = sorted(p.name for p in comfy_dirs.output.iterdir())

    response = call(ext.routes, "POST", "/h3_extender/save_preview", json=payload)

    assert response.status == 400
    assert sorted(p.name for p in comfy_dirs.output.iterdir()) == before


def test_project_download_token_works_exactly_once(ext):
    prepared = call(
        ext.routes, "POST", "/h3_extender/project/prepare_save",
        json={"owner_id": "token_node", "project_name": "Once", "project": {"extender": {"clips": [{"prompt": "x"}]}}},
    )
    assert prepared.status == 200, prepared.body
    token = body_json(prepared)["token"]

    first = call(ext.routes, "GET", "/h3_extender/project/download", params={"token": token})
    second = call(ext.routes, "GET", "/h3_extender/project/download", params={"token": token})

    assert first.status == 200 and first.body[:2] == b"PK"
    assert second.status == 404
