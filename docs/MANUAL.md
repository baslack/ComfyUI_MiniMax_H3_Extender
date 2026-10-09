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

## Troubleshooting

- **"Latent Refine needs a learned upscaler"**: see *Requirements*; connect the
  provider to `learned_upscaler`.
- **Clips keep regenerating**: a prompt, seed, duration, reference, model,
  sampling or refine change invalidated them (and, with Motion ON, every clip
  after them).
- **Out of VRAM in final decode**: seams decode only the previous clip's last
  7 latent frames plus the next clip; lower resolution or clip length if it
  still doesn't fit.

## Graph nodes (experimental)

An alternative to the Extender node: each step is an ordinary node, so clips
use ComfyUI's own samplers, schedulers, noise and LoRA loaders. The Extender
node is unchanged. See `Workflow/MiniMax_H3_Graph_Ref2VA.json` and
`Workflow/MiniMax_H3_Graph_FL2VA.json`.

| Node | Does |
|---|---|
| **MiniMax H3 Definition** | One named reference. `kind` is Subject (any number of images and videos), Picture, Video (optionally with its soundtrack) or Audio, with the guide's role and retention choices. |
| **MiniMax H3 Definitions** | Bundles definitions once for every clip. Names must be unique. |
| **MiniMax H3 Ref2VA Prompt** | One clip's prompt in MiniMax's six-section Ref2VA format. |
| **MiniMax H3 Keyframe Prompt** | T2VA / I2VA / FL2VA / L2VA prompt; the connected first/last frames pick the task. |
| **MiniMax H3 Encode** | Prompt + references → `positive`, `latent` and `composed_prompt`, the exact text the model gets. |

**Writing prompts.**

- Refer to definitions by name: `<Subject fred>`, `<Picture opening>`,
  `<Video dolly>`, `<Audio fredvoice>`, and to a Video definition's soundtrack
  as `<Audio dolly>`.
- Inside a definition, `<Picture 1>` / `<Video 1>` mean that node's own first
  attached image or video.
- Name speakers: `(S fred)`, or `(S fred, S mara)` when they speak together.
- Write `detailed_description` in the guide's form: a style sentence, then
  `[Shot 1] …`, `[Shot 2] At 00:03.000, …`.

Each clip numbers only what its text mentions, in the order the model reads
it, and speakers in the order they first speak in that clip. An unknown name,
a name used with the wrong kind, or a speaker who never speaks in the clip
stops the run with an error naming it. The `retention_analysis` lines,
including which shots each subject appears in, and the summary's task tags are
filled in for you; pick `task_types` to set the tags yourself.

**Clips.**

- **Building a clip:** encode → **Motion Context RAM** (previous clip's
  `cached_samples` as `context_latent`; leave it unconnected for clip 1) →
  `BasicGuider` → `SamplerCustomAdvanced` → **Disk Join**. Use
  `BasicGuider`; H3 does badly with CFG guidance.
- **Reusing a clip:** keep the clip as a subgraph and duplicate it for the
  next clip.
- **Disk Join settings:** `clip_name` names the clip's file in individual clip
  exports. `saturation`, `contrast` and `brightness` set its colour in Final
  Decode.
- **Refine:** refine every clip at the same scale, or none. Feed the next
  clip's refine pass a second Motion Context RAM with the encoder's
  `positive`. A refined previous clip is shrunk automatically for the base
  pass.
- **FL2VA clips:** these have no motion context. Wire `trim_frames` to 0 (for
  example from a `PrimitiveInt`); unconnected, Disk Join assumes 22.

Continuing an existing video is not available with the graph nodes yet.

## Tests

See `tests/contract/README.md` for the local contract test suite.
