"""The clip caches on disk: list, wipe, rename, archive and restore them.

A chain's cache is every entry in the cache folder named ``chain_<name>.*``:
the latents, the manifest (``.json``), decoded audio, previews and the
per-clip video folders. Cache names never contain a dot, so that prefix
belongs to one chain only.

An archive (``.h3project``, a zip) holds a cache, the workflow that made it
and the input files the workflow loads. Previews and decoded videos are
left out: the next Final Decode makes them again from the latents.

Standard library only: the ``h3cache`` command line imports this module
without loading ComfyUI.
"""
import hashlib
import json
import os
import re
import shutil
import time
import uuid
import zipfile
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
    _set_owner(root, new)
    return new


def _set_owner(root, name):
    manifest_path = Path(root) / f"chain_{name}.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["owner_id"] = name
        tmp = manifest_path.with_name(f"{manifest_path.name}.{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(tmp, manifest_path)


ARCHIVE_FORMAT = "minimax-h3-project/1"
ARCHIVE_SUFFIX = ".h3project"
# Made again by the next Final Decode, so not archived.
_REBUILT = (".final.video", ".preview.mp4", ".preview.video.mp4", ".source.preview.video.mp4")
_DECODED_VIDEO = (".mp4", ".mkv")
_SCRATCH = (".tmp", ".log", ".concat.txt")
_HEADROOM = 1024 ** 3


def _archived_files(root, name):
    prefix_len = len(f"chain_{name}")
    for path in _paths(root, name):
        if path.name[prefix_len:] in _REBUILT or path.name.endswith(_SCRATCH):
            continue
        if path.is_dir():
            yield from (f for f in sorted(path.rglob("*")) if f.is_file() and not f.name.endswith(_DECODED_VIDEO + _SCRATCH))
        else:
            yield path


def _workflow_nodes(workflow):
    yield from workflow.get("nodes", [])
    for subgraph in (workflow.get("definitions") or {}).get("subgraphs", []):
        yield from subgraph.get("nodes", [])


def _input_files(workflow, input_dir):
    """The files in ``input_dir`` that the workflow's widgets name, such as Load Image's image."""
    input_dir = Path(input_dir).resolve()
    found = {}
    for node in _workflow_nodes(workflow):
        values = node.get("widgets_values")
        for value in values.values() if isinstance(values, dict) else values or []:
            if not isinstance(value, str) or not value.strip():
                continue
            rel = value.removesuffix(" [input]")
            try:
                path = (input_dir / rel).resolve()
            except (OSError, ValueError):
                continue
            if path.is_file() and path.is_relative_to(input_dir):
                found[Path(rel).as_posix()] = path
    return found


def _check_space(folder, needed):
    free = shutil.disk_usage(folder).free
    if free < needed + _HEADROOM:
        raise OSError(
            f"MiniMax H3 archive: needs {needed / 1024 ** 3:.1f} GB plus 1 GB to spare in {folder}, "
            f"but only {free / 1024 ** 3:.1f} GB is free."
        )


def archive(root, name, workflow, input_dir, out_path):
    """Write cache ``name``, ``workflow`` and the input files it loads to ``out_path``."""
    files = list(_archived_files(root, name))
    inputs = _input_files(workflow, input_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _check_space(out_path.parent, sum(f.stat().st_size for f in files) + sum(p.stat().st_size for p in inputs.values()))
    meta = {"format": ARCHIVE_FORMAT, "name": name, "created_at": time.time(), "inputs": sorted(inputs)}
    tmp = out_path.with_name(out_path.name + ".tmp")
    try:
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
            z.writestr("archive.json", json.dumps(meta, indent=2))
            z.writestr("workflow.json", json.dumps(workflow, ensure_ascii=False))
            for f in files:
                z.write(f, "cache/" + f.relative_to(root).as_posix())
            for rel, path in inputs.items():
                z.write(path, "inputs/" + rel)
        os.replace(tmp, out_path)
    finally:
        tmp.unlink(missing_ok=True)
    return {"path": str(out_path), "size_bytes": out_path.stat().st_size, "inputs": sorted(inputs)}


def archive_info(path):
    """An archive's name, date and size, or None if the file isn't one."""
    try:
        with zipfile.ZipFile(path) as z:
            meta = json.loads(z.read("archive.json"))
    except (OSError, KeyError, ValueError, zipfile.BadZipFile):
        return None
    if meta.get("format") != ARCHIVE_FORMAT:
        return None
    return {"file": Path(path).name, "name": meta.get("name"), "created_at": meta.get("created_at"),
            "size_bytes": Path(path).stat().st_size}


def list_archives(folder):
    folder = Path(folder)
    infos = [archive_info(p) for p in folder.glob(f"*{ARCHIVE_SUFFIX}")] if folder.is_dir() else []
    return sorted((i for i in infos if i), key=lambda i: i["created_at"] or 0, reverse=True)


def _digest(stream):
    h = hashlib.sha256()
    for block in iter(lambda: stream.read(1 << 20), b""):
        h.update(block)
    return h.hexdigest()


def _contained(base, rel):
    path = (Path(base) / rel).resolve()
    if not path.is_relative_to(Path(base).resolve()):
        raise ValueError(f"MiniMax H3 archive: '{rel}' is outside {base}.")
    return path


def _restore_input(z, rel, input_dir):
    """Put an archived input back. A different file already there keeps its name; this one gets a new one."""
    member = "inputs/" + rel
    target = _contained(input_dir, rel)
    if target.exists():
        with z.open(member) as a, open(target, "rb") as b:
            if _digest(a) == _digest(b):
                return rel
        n = 1
        while target.with_name(f"{target.stem}_restored{n}{target.suffix}").exists():
            n += 1
        target = target.with_name(f"{target.stem}_restored{n}{target.suffix}")
    target.parent.mkdir(parents=True, exist_ok=True)
    with z.open(member) as src, open(target, "wb") as dst:
        shutil.copyfileobj(src, dst, 1 << 20)
    return target.relative_to(Path(input_dir).resolve()).as_posix()


def _rewrite_workflow(workflow, name, renamed_inputs):
    for node in _workflow_nodes(workflow):
        if node.get("type") == "MiniMaxH3Project":
            node["widgets_values"] = [name]
            if isinstance(node.get("widgets_values_named"), dict):
                node["widgets_values_named"]["name"] = name
        for key in ("widgets_values", "widgets_values_named"):
            values = node.get(key)
            items = values.items() if isinstance(values, dict) else enumerate(values or [])
            for k, value in list(items):
                if isinstance(value, str) and value.removesuffix(" [input]") in renamed_inputs:
                    rel = value.removesuffix(" [input]")
                    values[k] = value.replace(rel, renamed_inputs[rel], 1)
    # A new id, so the restored workflow opens beside the original instead of replacing it.
    workflow["id"] = str(uuid.uuid4())
    return workflow


def restore(root, archive_path, input_dir, workflows_dir, name=None):
    """Unpack an archive: its cache under ``name`` (default: the archived name), its inputs and its workflow."""
    with zipfile.ZipFile(archive_path) as z:
        meta = json.loads(z.read("archive.json"))
        if meta.get("format") != ARCHIVE_FORMAT:
            raise ValueError(f"MiniMax H3 archive: {Path(archive_path).name} is not a MiniMax H3 project archive.")
        old = project_name(meta["name"])
        new = project_name(name) if name else old
        if new in _entries(root):
            raise FileExistsError(f"MiniMax H3 cache: a cache named '{new}' already exists.")
        Path(root).mkdir(parents=True, exist_ok=True)
        _check_space(root, sum(m.file_size for m in z.infolist()))
        prefix = f"cache/chain_{old}."
        try:
            for member in z.infolist():
                if member.is_dir() or not member.filename.startswith(prefix):
                    continue
                target = _contained(root, f"chain_{new}." + member.filename[len(prefix):])
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
            _set_owner(root, new)
        except BaseException:
            if new in _entries(root):
                wipe(root, new)
            raise
        renamed = {}
        for rel in meta.get("inputs", []):
            restored = _restore_input(z, rel, input_dir)
            if restored != rel:
                renamed[rel] = restored
        workflow = _rewrite_workflow(json.loads(z.read("workflow.json")), new, renamed)

    workflows_dir = Path(workflows_dir)
    workflows_dir.mkdir(parents=True, exist_ok=True)
    workflow_path = workflows_dir / f"{new}.json"
    n = 1
    while workflow_path.exists():
        workflow_path = workflows_dir / (f"{new} (restored).json" if n == 1 else f"{new} (restored {n}).json")
        n += 1
    workflow_path.write_text(json.dumps(workflow, ensure_ascii=False), encoding="utf-8")
    return {"name": new, "workflow": workflow_path.name, "renamed_inputs": renamed}
