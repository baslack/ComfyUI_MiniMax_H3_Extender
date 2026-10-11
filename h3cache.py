"""h3cache: list, wipe and rename MiniMax H3 clip caches.

Run it with ComfyUI's Python, for example:

    python custom_nodes/ComfyUI_MiniMax_H3_Extender/h3cache.py list
    python custom_nodes/ComfyUI_MiniMax_H3_Extender/h3cache.py wipe old-test --yes
    python custom_nodes/ComfyUI_MiniMax_H3_Extender/h3cache.py rename 44_26 my-project

Wipe and rename refuse while ComfyUI is running, since a render could be
using the files; use the MiniMax H3 Project node's buttons then.
"""
import argparse
import socket
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cache_library  # noqa: E402

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / "user" / "h3_extender_cache"


def comfyui_running(host, port):
    try:
        socket.create_connection((host, port), timeout=1).close()
        return True
    except OSError:
        return False


def size_text(n):
    return f"{n / 1024 ** 3:.2f} GB" if n >= 1024 ** 3 else f"{n / 1024 ** 2:.1f} MB"


def main(argv=None):
    parser = argparse.ArgumentParser(prog="h3cache", description="List, wipe and rename MiniMax H3 clip caches.")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help=f"default: {DEFAULT_CACHE_DIR}")
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
    args = parser.parse_args(argv)

    if args.command == "list":
        caches = cache_library.list_caches(args.cache_dir)
        if not caches:
            print(f"No caches in {args.cache_dir}")
        for c in caches:
            updated = time.strftime("%Y-%m-%d %H:%M", time.localtime(c["updated_at"]))
            print(f"{c['name']:32} {c['validated']:>2}/{c['clips']:<2} clips validated  {size_text(c['size_bytes']):>10}  {updated}")
        return 0

    if comfyui_running(args.host, args.port):
        print(f"ComfyUI is running on {args.host}:{args.port}; use the MiniMax H3 Project node, or close ComfyUI first.", file=sys.stderr)
        return 1
    try:
        if args.command == "wipe":
            if not args.yes:
                caches = {c["name"]: c for c in cache_library.list_caches(args.cache_dir)}
                if args.name not in caches:
                    raise FileNotFoundError(f"MiniMax H3 cache: no cache named '{args.name}'.")
                c = caches[args.name]
                print(f"Would delete '{args.name}': {c['clips']} clips, {size_text(c['size_bytes'])}. Add --yes to delete it.")
                return 1
            print(f"Deleted '{args.name}', {size_text(cache_library.wipe(args.cache_dir, args.name))} freed.")
        else:
            print(f"Renamed '{args.old}' to '{cache_library.rename(args.cache_dir, args.old, args.new)}'.")
    except (ValueError, FileNotFoundError, FileExistsError) as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
