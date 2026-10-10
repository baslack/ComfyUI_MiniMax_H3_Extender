"""Synchronous helpers for calling the package's routes through aiohttp's test client."""
from __future__ import annotations

import asyncio
import io
import json
import types

from PIL import Image


def call(routes, method, path, **kwargs):
    """Serve `routes` on a loopback test server, make one request, return namespace(status, headers, body)."""
    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer

    async def go():
        app = web.Application()
        app.add_routes(routes)
        async with TestClient(TestServer(app)) as client:
            response = await client.request(method, path, **kwargs)
            body = await response.read()
            return types.SimpleNamespace(status=response.status, headers=dict(response.headers), body=body)

    return asyncio.run(go())


def body_json(response):
    return json.loads(response.body.decode("utf-8"))


def png_bytes(width=32, height=24, color=(200, 120, 40)):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format="PNG")
    return buffer.getvalue()


def multipart(**fields):
    """aiohttp FormData; bytes values become file parts named `<field>.bin` unless given as (name, bytes)."""
    from aiohttp import FormData

    form = FormData()
    for name, value in fields.items():
        if isinstance(value, tuple):
            filename, data = value
            form.add_field(name, data, filename=filename)
        elif isinstance(value, bytes):
            form.add_field(name, value, filename=f"{name}.bin")
        else:
            form.add_field(name, str(value))
    return form
