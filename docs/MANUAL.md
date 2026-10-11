# MiniMax H3 Extender — Manual

The Extender builds long MiniMax H3 videos (with audio) out of a sequence of
clips. Each clip is generated, cached on disk, optionally validated, and the
final decode assembles all clips into one video with seam and audio
correction.

## Requirements

- ComfyUI with native MiniMax H3 support, plus the H3 model, text encoder,
  video VAE and audio VAE.
- `ffmpeg` (from the `imageio-ffmpeg` package in `requirements.txt`, or on `PATH`).
- For **Latent Refine** only:
  - [Comfyui_Minimax_h3_latent_Upscaler-Plus](https://github.com/xmarre/Comfyui_Minimax_h3_latent_Upscaler-Plus)
  - an H3 latent upscaler checkpoint from
    [huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)
    in `ComfyUI/models/latent_upscale_models/` (subfolders are fine)

Nothing is downloaded or installed automatically.

## Minimal workflow

1. Load the H3 model, CLIP and VAEs.
2. Add **MiniMax H3 Extender**, connect `model`, `clip`, `vae` and `audio_vae`.
3. Add **Final Decode / Preview** (`MiniMaxH3MotionContextDiskFinalDecode`) and
   connect the Extender's `cache` output, the video VAE and the audio VAE.
4. Write a prompt on each clip card, then queue.

## Modes

| Mode | What it does | Use when |
|---|---|---|
| **Ref2VA, Motion ON** (default) | Each clip continues from the end of the previous one (Motion Context). | One continuous shot. Validation is a chain: un-validating a clip un-validates everything after it. |
| **Ref2VA, Motion OFF** | Clips are independent shots. | You want to insert or re-render any clip without touching the others. No continuity between clips. |
| **FL2VA** | Each clip is driven by first/last frames and up to 3 image guides at exact frames. Needs the `fl2va_model` input. | Planned visual timelines: First → Guide 1–3 → Last. |

## Run modes

- **clip_by_clip** (default): generates the first unvalidated clip and stops.
  Typical loop: generate → preview → retry → validate → next clip.
- **full_batch**: generates every clip that isn't validated in one run. Can be
  interrupted; computed clips are kept and the run resumes later (also after
  project Save/Load).

## Main settings

| Input | Default | Notes |
|---|---|---|
| `steps`, `sampler_name`, `scheduler`, `denoise` | 4, euler, simple, 1.0 | First-pass sampling. An optional `sigmas` input overrides steps/scheduler. |
| `resolution_mode` | `auto_from_ref` | Auto uses Ref 1's aspect ratio at `megapixels`; `manual` uses `width`/`height` (multiples of 32). |
| `ref_image_size` | `match` | `match` scales references to the output area; `max` keeps more detail but is slower. |
| `context_length` | 22 | Frames of the previous clip carried into the next (Motion Context): 5, 22, 39 or 56. |
| `audio_context_length` | 0 | Audio context in latent steps; 0 matches the video context. |

Per clip (on the card): prompt, name, duration, seed and seed mode
(randomize / fixed / increment / decrement), LoRAs with strengths, local
Picture/Video/Audio references, and a colour editor (saturation, contrast,
brightness; applied only to the decoded video).

## References

- Up to **9 pictures** (`<Picture N>`), **3 videos** (`<Video N>`) and
  **3 audio** clips (`<Audio N>`). Slot numbers never shift.
- Pictures are loaded into the node's internal reference manager and saved
  with projects. **Reference Pack Bridge** maps external `IMAGE`s into those
  slots; **Prompt Pack Bridge** feeds prompts from other nodes.
- Reference videos go in `ref_video_N`; connect `ref_video_fps_N` when the
  source isn't 24 fps and `ref_video_audio_N` for its soundtrack.
- Start every clip prompt with the same `subject_definitions` block to keep
  identities and environment consistent across clips.

## Continue an existing video

Connect a file-backed **Load Video** to `continue_existing_video`. It becomes
a locked Clip 0 that the first generated clip continues from. Not available
together with Latent Refine.

## Latent Refine

Turn on `refine_enabled` and connect a **MiniMax H3 Latent Upscaler Provider
(3D)** to `learned_upscaler`. Each clip is upscaled by `refine_scale` (1.0–2.0)
and re-sampled for `refine_steps` at `refine_denoise`. Changing any refine
setting, or the provider's checkpoint/precision, invalidates refined clips.

## Final Decode / Preview

Decodes the cached clips, corrects seams between them, rebuilds the audio
once, and writes the video (H.264, H.265 or FFV1). Options include autoplay,
**auto-save project** after a full batch, and **save individual clips**. The
node also has a native `VIDEO` output and a **Save Preview** button that saves
the current preview with workflow metadata.

With motion context, each clip blends into the next over **crossfade_frames**
frames (default 10), around the point where the two clips are in step; 0 cuts
at the end of the overlap. Changing it re-cuts the seams without sampling
again.

The video ends at the last Disk Join the chain ran through. Clips cached
after it, for example in bypassed groups, are left out and kept, and are used
again when their groups are turned back on. That holds while the clips
before them are read from the cache: a clip that runs again unvalidated
deletes every clip after it, since they continued the old version. The last
clip that ran is cut again when its neighbour leaves or comes back: Full
Batch decodes it again each way; Clip by Clip decodes it when the clips
after it are left out and trims its preview when they come back.

## Projects

- **Save Project** writes a portable `.ext` archive: prompts, settings,
  references (embedded), validation state and the latent cache.
- **Load Project** restores it, also on another machine. Imports are refused
  if they would not fit on the cache drive (1 GiB is kept free).
- **New Project** clears clips, references and cache but keeps node settings.

## Where data lives

Cache, references and project temp files are in
`ComfyUI/user/h3_extender_cache/`. They survive restarts and reinstalls of
the node; delete that folder (or use **New Project**) to reclaim space.

A graph clip chain keeps its files as `chain_<name>.*`, named by the
**MiniMax H3 Project** node connected to the chain's first Disk Join (or to
Continue Video). Names use letters, digits, `-` and `_`. Without a project
the chain is named after that node's id, so two workflows with the same node
ids, such as a workflow and its copy, share one cache. Give each workflow its
own project name to keep their clips apart.

The Project node shows its cache's clips, size and last update, and has
buttons to **Wipe this cache** and to open **All caches**, which lists every
cache with **Use** (point this node at it), **Rename** and **Wipe**. Use
Rename to adopt a cache made without a project, such as `44_26`. Wipe and
Rename refuse while a prompt is running.

With ComfyUI closed, `h3cache.py` in this node's folder does the same from a
terminal, using ComfyUI's Python:

```
python custom_nodes/ComfyUI_MiniMax_H3_Extender/h3cache.py list
python custom_nodes/ComfyUI_MiniMax_H3_Extender/h3cache.py rename 44_26 my-project
python custom_nodes/ComfyUI_MiniMax_H3_Extender/h3cache.py wipe my-project --yes
```

`wipe` without `--yes` only shows what it would delete. Wipe and rename
refuse while ComfyUI is running (`--port` if it isn't on 8188); `list`
always works.

**Archive** packs a project into one `.h3project` file: its cache, the
workflow as it is now and every file in `input/` the workflow loads.
Previews and decoded videos are left out; the next Final Decode makes them
again from the latents, about one VAE decode per clip. **Restore** unpacks
one, under its archived name or a new one:

- the cache never replaces an existing one;
- an input file that is already there with the same contents is reused; a
  different file with that name is kept, and the archived one is restored
  as `<name>_restored1.<ext>` with the workflow pointing at it;
- the workflow is saved to your workflows with its Project node set to the
  restored name.

The node archives to `output/h3_projects/` and restores from there. From a
terminal:

```
python custom_nodes/ComfyUI_MiniMax_H3_Extender/h3cache.py archive my-project --workflow user/default/workflows/my.json
python custom_nodes/ComfyUI_MiniMax_H3_Extender/h3cache.py restore my-project_20261011_120000.h3project --name my-project-2
```

Archive only reads, so it works while ComfyUI runs, but don't archive a
project that is rendering. Restore refuses while ComfyUI is running.

## Troubleshooting

- **"Latent Refine needs a learned upscaler"**: see *Requirements*; connect the
  provider to `learned_upscaler`.
- **Clips keep regenerating**: a prompt, seed, duration, reference, model,
  sampling or refine change invalidated them (and, with Motion ON, every clip
  after them).
- **Out of VRAM in final decode**: seams decode only the previous clip's last
  7 latent frames plus the next clip; lower resolution or clip length if it
  still doesn't fit.

## Graph nodes

The prompt and encode nodes (MiniMax H3 Definition, Definitions, Ref2VA
Prompt, Keyframe Prompt and Encode) and their sample workflows live in
[H3PromptCompositionNodes](https://github.com/baslack/H3PromptCompositionNodes).
This fork keeps what they need to chain clips:

- **Motion Context RAM:** `context_latent` is optional, so the first clip runs
  without a previous clip, and a larger (refined) context latent is shrunk to
  the clip's size.
- **Disk Join:** `clip_name` names the clip's file in individual clip exports;
  `saturation`, `contrast` and `brightness` set its colour in Final Decode.
  `chain_mode` `hard_cut` joins independent clips with plain cuts, and
  `first_frame_from_previous` hides a first frame handed over from the
  previous clip.
- **Continue Video** starts a chain from an existing video, which becomes
  Clip 0 as with the Extender node. Connect a file-backed **Load Video**,
  the VAEs and the clips' final width and height. Its `cache` goes to the
  first clip's Disk Join as `previous_cache`, and its `context_latent` to
  that clip's Motion Context RAM, set to the same `context_length`. Final
  Decode puts the source in front of the clips and crossfades it into the
  first clip like any other seam (`crossfade_frames`). Use a
  `motion_context` chain. `fit` handles a source of another aspect ratio:
  `crop` (default) fills the frame and cuts the overflow, `pad` fits it
  inside black bars, `stretch` distorts it.
- **Continue Video with refine:** the source is resized to the final size, so
  with refine on, give Continue Video the refined size: the base size times
  the refine scale, rounded up to a multiple of 32 (1248x704 at 1.5 is
  1888x1056). The first clip's base pass shrinks the source tail to its
  size, and its refine pass uses it at full size.

## Tests

See `tests/contract/README.md` for the local contract test suite.
