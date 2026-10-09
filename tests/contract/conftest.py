"""Harness for the contract tests.

The package is loaded the way ComfyUI loads custom_nodes/<pkg>/__init__.py,
from a throwaway copy so nothing it writes lands in the checkout. ComfyUI's
user/input/output/temp directories point at a temp tree, and the HTTP routes
the package registers are collected on a stub PromptServer for aiohttp's
test client.
"""
from __future__ import annotations

import importlib.util
import ipaddress
import os
import shutil
import socket
import sys
import types
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "h3_extender_contract"

comfyui_path = os.environ.get("COMFYUI_PATH")
if comfyui_path and comfyui_path not in sys.path:
    sys.path.insert(0, comfyui_path)

# Without CUDA (or with it hidden via CUDA_VISIBLE_DEVICES=""), prime ComfyUI's
# CLI args with --cpu before anything imports comfy.model_management.
if comfyui_path and not torch.cuda.is_available():
    original_argv = sys.argv[:]
    try:
        sys.argv[:] = [original_argv[0], "--cpu"]
        import comfy.options

        comfy.options.enable_args_parsing()
        import comfy.cli_args  # noqa: F401
    finally:
        sys.argv[:] = original_argv

    import comfy_kitchen

    # Probing int8 attention queries a CUDA device and fails on CPU-only runs.
    comfy_kitchen.int8_attention_is_available = lambda: False


def pytest_configure(config):
    config.addinivalue_line("markers", "gpu: needs CUDA and real MiniMax H3 weights")


def pytest_collection_modifyitems(config, items):
    if torch.cuda.is_available():
        return
    skip = pytest.mark.skip(reason="CUDA is not available")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)


_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex


def _is_loopback(address) -> bool:
    host = address[0] if isinstance(address, tuple) else address
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Nothing in this package may reach the network; loopback stays usable for the test server."""

    def connect(self, address):
        if not _is_loopback(address):
            raise RuntimeError(f"tests must not open network connections (attempted {address!r})")
        return _real_connect(self, address)

    def connect_ex(self, address):
        if not _is_loopback(address):
            raise RuntimeError(f"tests must not open network connections (attempted {address!r})")
        return _real_connect_ex(self, address)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)


@pytest.fixture(scope="session")
def comfy_dirs(tmp_path_factory):
    folder_paths = pytest.importorskip("folder_paths", reason="set COMFYUI_PATH to a ComfyUI checkout")
    base = tmp_path_factory.mktemp("comfyui")
    dirs = types.SimpleNamespace(**{name: base / name for name in ("user", "input", "output", "temp")})
    for path in vars(dirs).values():
        path.mkdir()
    folder_paths.set_user_directory(str(dirs.user))
    folder_paths.set_input_directory(str(dirs.input))
    folder_paths.set_output_directory(str(dirs.output))
    folder_paths.set_temp_directory(str(dirs.temp))
    return dirs


@pytest.fixture(scope="session")
def ext(comfy_dirs, tmp_path_factory):
    """namespace(pkg, root, routes): the loaded package, its copy on disk, and its HTTP routes."""
    from aiohttp import web

    root = tmp_path_factory.mktemp("custom_nodes") / "ComfyUI_MiniMax_H3_Extender"
    shutil.copytree(
        ROOT, root,
        ignore=shutil.ignore_patterns(".git", "cache", "tests", "Workflow", "*.zip", "__pycache__", ".pytest_cache"),
    )
    routes = web.RouteTableDef()
    server = types.ModuleType("server")
    server.PromptServer = type(
        "PromptServer", (), {"instance": types.SimpleNamespace(routes=routes, send_sync=lambda *a, **k: None)}
    )
    previous_server = sys.modules.get("server")
    sys.modules["server"] = server
    try:
        spec = importlib.util.spec_from_file_location(
            PACKAGE, root / "__init__.py", submodule_search_locations=[str(root)]
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[PACKAGE] = module
        spec.loader.exec_module(module)
    finally:
        if previous_server is None:
            sys.modules.pop("server", None)
        else:
            sys.modules["server"] = previous_server
    return types.SimpleNamespace(pkg=module, root=root, routes=routes)
