"""The clip caches on disk: list, wipe and rename them.

A chain's cache is every entry in the cache folder named ``chain_<name>.*``:
the latents, the manifest (``.json``), decoded audio, previews and the
per-clip video folders. Cache names never contain a dot, so that prefix
belongs to one chain only.

Standard library only: the ``h3cache`` command line imports this module
without loading ComfyUI.
"""
import json
import os
import re
import shutil
import uuid
from pathlib import Path

PROJECT_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9_-]*[A-Za-z0-9])?")


def project_name(name):
    """A project's name, used as is in its cache file names."""
    name = str(name).strip()
    if not PROJECT_NAME.fullmatch(name):
        raise ValueError(
            f"MiniMax H3 Project: '{name}' is not a valid project name. Use letters, digits, "
            "'-' and '_', starting and ending with a letter or digit."
        )
    return name


def _entries(root):
    """Map each cache name in ``root`` to its files and folders."""
    caches = {}
    root = Path(root)
    if not root.is_dir():
        return caches
    for path in root.iterdir():
        if path.name.startswith("chain_") and "." in path.name:
            caches.setdefault(path.name[len("chain_"):].split(".", 1)[0], []).append(path)
    return caches


def _size(path):
    if path.is_dir():
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return path.stat().st_size


def _paths(root, name):
    paths = _entries(root).get(str(name))
    if not paths:
        raise FileNotFoundError(f"MiniMax H3 cache: no cache named '{name}'.")
    return paths


def list_caches(root):
    """Every cache in ``root``, most recently updated first."""
    caches = []
    for name, paths in _entries(root).items():
        manifest_path = Path(root) / f"chain_{name}.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            manifest = {}
        segments = manifest.get("segments") or []
        caches.append({
            "name": name,
            "clips": len(segments),
            "validated": sum(1 for s in segments if s.get("validated")),
            "size_bytes": sum(_size(p) for p in paths),
            "updated_at": float(manifest.get("updated_at") or max(p.stat().st_mtime for p in paths)),
        })
    return sorted(caches, key=lambda c: c["updated_at"], reverse=True)


def wipe(root, name):
    """Delete one cache. Returns the bytes freed."""
    paths = _paths(root, name)
    freed = sum(_size(p) for p in paths)
    for path in paths:
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    return freed


def rename(root, old, new):
    """Give a cache a new name, for example to adopt one a workflow made without a project."""
    new = project_name(new)
    paths = _paths(root, old)
    if new in _entries(root):
        raise FileExistsError(f"MiniMax H3 cache: a cache named '{new}' already exists.")
    prefix = f"chain_{old}."
    for path in paths:
        path.rename(path.with_name(f"chain_{new}." + path.name[len(prefix):]))
    manifest_path = Path(root) / f"chain_{new}.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["owner_id"] = new
        tmp = manifest_path.with_name(f"{manifest_path.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, manifest_path)
    return new
