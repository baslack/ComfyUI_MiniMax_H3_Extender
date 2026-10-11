"""h3cache: list, wipe, rename, archive and restore MiniMax H3 clip caches.

Run it with ComfyUI's Python, for example:

    python custom_nodes/ComfyUI_MiniMax_H3_Extender/h3cache.py list
    python custom_nodes/ComfyUI_MiniMax_H3_Extender/h3cache.py wipe old-test --yes
    python custom_nodes/ComfyUI_MiniMax_H3_Extender/h3cache.py rename 44_26 my-project
    python custom_nodes/ComfyUI_MiniMax_H3_Extender/h3cache.py archive my-project --workflow user/default/workflows/my.json
    python custom_nodes/ComfyUI_MiniMax_H3_Extender/h3cache.py restore output/h3_projects/my-project_20261011_120000.h3project

Wipe, rename and restore refuse while ComfyUI is running, since a render
could be using the files; use the MiniMax H3 Project node's buttons then.
"""
import argparse
import json
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cache_library  # noqa: E402

DEFAULT_COMFYUI = Path(__file__).resolve().parents[2]


def comfyui_running(host, port):
    try:
        socket.create_connection((host, port), timeout=1).close()
        return True
    except OSError:
        return False


def size_text(n):
    return f"{n / 1024 ** 3:.2f} GB" if n >= 1024 ** 3 else f"{n / 1024 ** 2:.1f} MB"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="h3cache", description="List, wipe, rename, archive and restore MiniMax H3 clip caches.")
    parser.add_argument("--comfyui", type=Path, default=DEFAULT_COMFYUI, help=f"the ComfyUI folder (default: {DEFAULT_COMFYUI})")
    parser.add_argument("--cache-dir", type=Path, help="default: <comfyui>/user/h3_extender_cache")
    parser.add_argument("--host", default="127.0.0.1", help="where ComfyUI listens (default 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8188, help="ComfyUI's port (default 8188)")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="list every cache")
    wipe = commands.add_parser("wipe", help="delete one cache")
    wipe.add_argument("name")
    wipe.add_argument("--yes", action="store_true", help="really delete it; without this, only show what would go")
    rename = commands.add_parser("rename", help="rename one cache")
    rename.add_argument("old")
    rename.add_argument("new")
    archive = commands.add_parser("archive", help="pack a cache, its workflow and the inputs it loads into a .h3project")
    archive.add_argument("name")
    archive.add_argument("--workflow", type=Path, required=True, help="the workflow file that uses this cache")
    archive.add_argument("--out", type=Path, help="default: <comfyui>/output/h3_projects/<name>_<date>.h3project")
    restore = commands.add_parser("restore", help="unpack a .h3project")
    restore.add_argument("file", type=Path)
    restore.add_argument("--name", help="restore the cache under this name instead of the archived one")
    args = parser.parse_args(argv)

    cache_dir = args.cache_dir or args.comfyui / "user" / "h3_extender_cache"
    input_dir = args.comfyui / "input"

    if args.command == "list":
        caches = cache_library.list_caches(cache_dir)
        if not caches:
            print(f"No caches in {cache_dir}")
        for c in caches:
            updated = time.strftime("%Y-%m-%d %H:%M", time.localtime(c["updated_at"]))
            print(f"{c['name']:32} {c['validated']:>2}/{c['clips']:<2} clips validated  {size_text(c['size_bytes']):>10}  {updated}")
        return 0

    try:
        if args.command == "archive":
            # Only reads; don't archive a project while it is rendering.
            out = args.out or args.comfyui / "output" / "h3_projects" / f"{args.name}_{time.strftime('%Y%m%d_%H%M%S')}{cache_library.ARCHIVE_SUFFIX}"
            workflow = json.loads(args.workflow.read_text(encoding="utf-8"))
            result = cache_library.archive(cache_dir, args.name, workflow, input_dir, out)
            print(f"Archived '{args.name}' with {len(result['inputs'])} input files to {result['path']} ({size_text(result['size_bytes'])}).")
            return 0

        if comfyui_running(args.host, args.port):
            print(f"ComfyUI is running on {args.host}:{args.port}; use the MiniMax H3 Project node, or close ComfyUI first.", file=sys.stderr)
            return 1
        if args.command == "wipe":
            if not args.yes:
                caches = {c["name"]: c for c in cache_library.list_caches(cache_dir)}
                if args.name not in caches:
                    raise FileNotFoundError(f"MiniMax H3 cache: no cache named '{args.name}'.")
                c = caches[args.name]
                print(f"Would delete '{args.name}': {c['clips']} clips, {size_text(c['size_bytes'])}. Add --yes to delete it.")
                return 1
            print(f"Deleted '{args.name}', {size_text(cache_library.wipe(cache_dir, args.name))} freed.")
        elif args.command == "rename":
            print(f"Renamed '{args.old}' to '{cache_library.rename(cache_dir, args.old, args.new)}'.")
        else:
            result = cache_library.restore(cache_dir, args.file, input_dir, args.comfyui / "user" / "default" / "workflows", args.name)
            print(f"Restored '{result['name']}'; its workflow is {result['workflow']}.")
            for old, new in result["renamed_inputs"].items():
                print(f"  input {old} was already there with other contents; restored as {new}")
    except (OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
