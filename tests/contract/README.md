# Contract tests

Run locally against a ComfyUI checkout with ComfyUI's own Python; `pytest` is
the only extra package.

```powershell
cd <this repo>
$env:COMFYUI_PATH = "<ComfyUI checkout>"
$env:PYTHONPATH   = "$env:COMFYUI_PATH;$PWD"
$env:CUDA_VISIBLE_DEVICES = "-1"                  # hide the GPU (not "": that deletes the variable)
<ComfyUI python> -m pytest tests/contract -m "not gpu"
```

To also run the `gpu` tests, `Remove-Item Env:CUDA_VISIBLE_DEVICES` and drop
`-m "not gpu"`, and only while ComfyUI isn't generating.

- Tests go through the package's public surface only: node schemas (saved
  workflows depend on input order), the HTTP routes the frontend calls,
  portable `.ext` archives, and observable effects such as files written or
  ComfyUI code left unpatched.
- The package is loaded from a throwaway copy, and ComfyUI's user, input,
  output and temp directories point at a temp tree, so a run never writes to
  the checkout or to a real ComfyUI install.
- Any non-loopback network connection fails the test.

Upstream's own tests in `tests/` are left as upstream wrote them.
`tests/test_project_autosave.py::...export_hooks_after_video` already fails
upstream: it expects `auto_save_project` to be the last Final Decode input,
but `save_individual_clips` was appended after it.
