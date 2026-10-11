"""Project contract: a project name gives a clip chain its own cache files.

Without one, a chain's cache is named after the node that starts it, so two
workflows with the same node ids share a cache.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch


def _av(seed):
    import comfy.nested_tensor

    g = torch.Generator().manual_seed(seed)
    return {"samples": comfy.nested_tensor.NestedTensor((torch.rand(1, 24, 12, 6, 8, generator=g), torch.rand(1, 32, 2, 70, generator=g)))}


def _join(nodes, owner, project=None, previous=None, seed=1):
    kwargs = {"project": project} if project is not None else {}
    if previous is not None:
        kwargs.update(previous_cache=previous, trim_frames=22)
    return nodes["MiniMaxH3MotionContextDiskJoin"]().join(
        samples=_av(seed), validated=False, run_mode="full_batch", fps=24.0, unique_id=owner, **kwargs)[0]


def _clips(handle):
    return len(json.loads(Path(handle["manifest_path"]).read_text(encoding="utf-8"))["segments"])


def test_projects_with_the_same_node_ids_keep_separate_caches(ext):
    nodes = ext.pkg.NODE_CLASS_MAPPINGS
    project = nodes["MiniMaxH3Project"]()
    cabin = _join(nodes, "44:26", project.project("eridah-cabin")[0])
    sample = _join(nodes, "44:26", project.project("ref2va_sample")[0], seed=2)

    assert Path(cabin["manifest_path"]).name == "chain_eridah-cabin.json"
    assert Path(sample["manifest_path"]).name == "chain_ref2va_sample.json"
    assert _clips(cabin) == _clips(sample) == 1


def test_without_a_project_the_cache_is_named_after_the_first_join(ext):
    handle = _join(ext.pkg.NODE_CLASS_MAPPINGS, "45:26")
    assert Path(handle["manifest_path"]).name == "chain_45_26.json"


def test_only_the_chain_s_first_join_reads_the_project(ext):
    nodes = ext.pkg.NODE_CLASS_MAPPINGS
    first = _join(nodes, "46:26", "chain-a")
    second = _join(nodes, "47:26", "chain-b", previous=first, seed=2)

    assert second["manifest_path"] == first["manifest_path"]
    assert _clips(second) == 2


@pytest.mark.parametrize("name", ["", "has space", "../escape", "a/b", "-lead", "trail_", "dot.name"])
def test_invalid_project_names_are_refused(ext, name):
    nodes = ext.pkg.NODE_CLASS_MAPPINGS
    with pytest.raises(ValueError):
        nodes["MiniMaxH3Project"]().project(name)
    with pytest.raises(ValueError):
        _join(nodes, "48:26", name)
