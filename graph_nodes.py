"""Composable MiniMax H3 prompt and encode nodes.

Definitions name the references a clip can use, a prompt node composes one
clip's prompt in the official format, and MiniMaxH3Encode turns it into
conditioning plus an empty AV latent through the core MiniMax H3 nodes.
Sampling, motion context, refine and caching stay ordinary graph nodes.
"""
from comfy_api.latest import io
from comfy_extras.nodes_minimax_h3 import (
    FPS,
    MiniMaxH3ImageToVideo,
    MiniMaxH3ReferenceToVideo,
    temporal_shape,
)
import nodes

from .prompt_compose import (
    AudioRetention,
    AudioRole,
    Definition,
    Kind,
    PictureRole,
    Retention,
    SizeRounding,
    TaskType,
    VideoRole,
    check_unique_names,
    compose_keyframe,
    compose_ref2va,
    keyframe_instruction,
)

H3Def = io.Custom("H3_DEF")
H3Defs = io.Custom("H3_DEFS")
H3Prompt = io.Custom("H3_PROMPT")

CATEGORY = "MiniMax H3"
SIZE_MULTIPLE = 32


def _connected(group, prefix):
    group = group or {}
    keys = sorted((k for k in group if group[k] is not None), key=lambda k: int(k[len(prefix):]))
    return [group[k] for k in keys]


def _video_at_24fps(video, with_soundtrack, label):
    components = video.get_components()
    frames = components.images
    fps = float(components.frame_rate)
    if abs(fps - FPS) > 1e-6 and frames.shape[0] > 1:
        count = max(1, round(frames.shape[0] * FPS / fps))
        last = frames.shape[0] - 1
        frames = frames[[min(last, round(i * fps / FPS)) for i in range(count)]]
    soundtrack = None
    if with_soundtrack:
        soundtrack = components.audio
        if soundtrack is None:
            raise ValueError(f"MiniMax H3 Encode: {label} has soundtrack_as_audio on but the video has no audio.")
    return frames, soundtrack


def _snap(value, rounding):
    if rounding == SizeRounding.DOWN.value:
        return max(SIZE_MULTIPLE, value // SIZE_MULTIPLE * SIZE_MULTIPLE)
    return -(-value // SIZE_MULTIPLE) * SIZE_MULTIPLE


class MiniMaxH3Definition(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        def retention(options):
            return io.Combo.Input("retention", options=options, tooltip="How the target video keeps this reference (retention_analysis).")

        return io.Schema(
            node_id="MiniMaxH3Definition",
            display_name="MiniMax H3 Definition",
            category=CATEGORY,
            description="A named reference for MiniMax H3 Ref2VA prompts. Refer to it in prompt text as <Subject name>, <Picture name>, <Video name> or <Audio name>.",
            inputs=[
                io.String.Input("name", default="subject",
                    tooltip="Used in prompt text as <Kind name>. Letters, digits, _ and -; must start with a letter or _."),
                io.DynamicCombo.Input("kind", options=[
                    io.DynamicCombo.Option(Kind.SUBJECT.value, [
                        io.Autogrow.Input("images", optional=True, template=io.Autogrow.TemplatePrefix(
                            input=io.Image.Input("image"), prefix="image", min=0, max=9)),
                        io.Autogrow.Input("videos", optional=True, template=io.Autogrow.TemplatePrefix(
                            input=io.Video.Input("video"), prefix="video", min=0, max=3)),
                        retention(Retention),
                    ]),
                    io.DynamicCombo.Option(Kind.PICTURE.value, [
                        io.Image.Input("image"),
                        io.Combo.Input("role", options=PictureRole),
                        retention(Retention),
                    ]),
                    io.DynamicCombo.Option(Kind.VIDEO.value, [
                        io.Video.Input("video"),
                        io.Combo.Input("role", options=VideoRole),
                        retention(Retention),
                        io.Boolean.Input("soundtrack_as_audio", default=False,
                            tooltip="Also reference this video's soundtrack, as <Audio name>."),
                        io.Combo.Input("soundtrack_retention", options=AudioRetention),
                    ]),
                    io.DynamicCombo.Option(Kind.AUDIO.value, [
                        io.Audio.Input("audio"),
                        io.Combo.Input("role", options=AudioRole),
                        retention(AudioRetention),
                    ]),
                ]),
                io.String.Input("definition", multiline=True,
                    tooltip="The rest of the subject_definitions line, e.g. 'is the young woman in <Picture 1>, with...'. "
                            "<Picture k> / <Video k> mean this node's k-th attachment."),
                io.String.Input("retention_note", default="", tooltip="What is kept, after the retention marker."),
            ],
            outputs=[H3Def.Output(display_name="definition")],
        )

    @classmethod
    def execute(cls, name, kind, definition, retention_note) -> io.NodeOutput:
        selected = kind["kind"]
        common = {"name": name, "kind": selected, "text": definition, "note": retention_note,
                  "retention": kind["retention"], "role": kind.get("role")}
        if selected == Kind.SUBJECT.value:
            d = Definition(images=_connected(kind.get("images"), "image"),
                           videos=_connected(kind.get("videos"), "video"), **common)
        elif selected == Kind.PICTURE.value:
            d = Definition(images=[kind["image"]], **common)
        elif selected == Kind.VIDEO.value:
            d = Definition(videos=[kind["video"]],
                           soundtrack_retention=kind["soundtrack_retention"] if kind["soundtrack_as_audio"] else None,
                           **common)
        else:
            d = Definition(audio=kind["audio"], **common)
        return io.NodeOutput(d)


class MiniMaxH3Definitions(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3Definitions",
            display_name="MiniMax H3 Definitions",
            category=CATEGORY,
            description="Bundle definitions once for every clip. Each clip uses only the definitions its prompt names.",
            inputs=[
                io.Autogrow.Input("definitions", template=io.Autogrow.TemplatePrefix(
                    input=H3Def.Input("definition"), prefix="definition", min=1, max=64)),
            ],
            outputs=[H3Defs.Output(display_name="definitions")],
        )

    @classmethod
    def execute(cls, definitions) -> io.NodeOutput:
        bundle = _connected(definitions, "definition")
        check_unique_names(bundle)
        return io.NodeOutput(bundle)


class MiniMaxH3Ref2VAPrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3Ref2VAPrompt",
            display_name="MiniMax H3 Ref2VA Prompt",
            category=CATEGORY,
            description="One clip's Ref2VA prompt in the official six-section format. Refer to definitions as <Kind name> "
                        "and to speakers as (S name); numbers are assigned for this clip.",
            inputs=[
                H3Defs.Input("definitions", optional=True),
                io.MultiCombo.Input("task_types", options=[t.value for t in TaskType], default=[],
                    placeholder="derive from definitions",
                    tooltip="Summary prefix. Leave empty to derive it from the referenced definitions' roles."),
                io.String.Input("summary", multiline=True),
                io.String.Input("detailed_description", multiline=True,
                    tooltip="Style sentence, then [Shot 1] ..., [Shot 2] At 00:03.000, ... in playback order."),
                io.String.Input("overall_soundscape", multiline=True),
                io.String.Input("non_diegetic_music", multiline=True, tooltip="Empty renders N/A."),
            ],
            outputs=[H3Prompt.Output(display_name="prompt")],
        )

    @classmethod
    def execute(cls, task_types, summary, detailed_description, overall_soundscape, non_diegetic_music,
                definitions=None) -> io.NodeOutput:
        return io.NodeOutput(compose_ref2va(definitions or [], task_types, summary, detailed_description,
                                            overall_soundscape, non_diegetic_music))


class MiniMaxH3KeyframePrompt(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3KeyframePrompt",
            display_name="MiniMax H3 Keyframe Prompt",
            category=CATEGORY,
            description="T2VA / I2VA / FL2VA / L2VA prompt. The connected frames pick the task; the alignment line is "
                        "written by MiniMax H3 Encode from the clip length.",
            inputs=[
                io.Image.Input("first_frame", optional=True),
                io.Image.Input("last_frame", optional=True),
                io.String.Input("integrated_multimodal_description", multiline=True,
                    tooltip="[Shot 1] ... describing the path from the first frame to the last, not the frames themselves."),
                io.String.Input("overall_soundscape", multiline=True),
                io.String.Input("non_diegetic_music", multiline=True, tooltip="Empty renders N/A."),
            ],
            outputs=[H3Prompt.Output(display_name="prompt")],
        )

    @classmethod
    def execute(cls, integrated_multimodal_description, overall_soundscape, non_diegetic_music,
                first_frame=None, last_frame=None) -> io.NodeOutput:
        return io.NodeOutput(compose_keyframe(integrated_multimodal_description, overall_soundscape,
                                              non_diegetic_music, first_frame, last_frame))


class MiniMaxH3Encode(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3Encode",
            display_name="MiniMax H3 Encode",
            category=CATEGORY,
            description="Encode a MiniMax H3 prompt with its references into conditioning and an empty AV latent.",
            inputs=[
                io.Clip.Input("clip"),
                io.Vae.Input("vae"),
                io.Vae.Input("audio_vae", optional=True, tooltip="Needed when the prompt references audio."),
                H3Prompt.Input("prompt"),
                io.Int.Input("width", default=1344, min=32, max=nodes.MAX_RESOLUTION,
                    tooltip="Connect a resolution node. Snapped to a multiple of 32 by size_rounding."),
                io.Int.Input("height", default=768, min=32, max=nodes.MAX_RESOLUTION),
                io.Combo.Input("size_rounding", options=SizeRounding, default=SizeRounding.UP,
                    tooltip="up: never smaller than requested. down: never larger."),
                io.Int.Input("length", default=124, min=5, max=3600, step=17,
                    tooltip="Frame count at 24 fps, snapped up to the model's 17k+5 grid (124 = ~5s)."),
                io.Combo.Input("ref_image_size", options=["match", "max"], default="match",
                    tooltip="'match' scales reference images to the output area; 'max' keeps up to 2048px short edge."),
            ],
            outputs=[
                io.Conditioning.Output(display_name="positive"),
                io.Latent.Output(display_name="latent"),
                io.String.Output(display_name="composed_prompt"),
            ],
        )

    @classmethod
    def execute(cls, clip, vae, prompt, width, height, size_rounding, length, ref_image_size,
                audio_vae=None) -> io.NodeOutput:
        width = _snap(width, size_rounding)
        height = _snap(height, size_rounding)

        if prompt["mode"] == "keyframe":
            has_first = prompt["first_frame"] is not None
            has_last = prompt["last_frame"] is not None
            frame_count = temporal_shape(length)[0]
            instruction = keyframe_instruction(has_first, has_last, frame_count, prompt["final_shot"])
            text = f"{instruction}\n\n{prompt['body']}" if instruction else prompt["body"]
            out = MiniMaxH3ImageToVideo.execute(clip, vae, text, width, height, length,
                                                first_frame=prompt["first_frame"], last_frame=prompt["last_frame"])
            return io.NodeOutput(out[0], out[1], text)

        ref_videos = {}
        ref_video_audios = {}
        for i, (video, with_soundtrack) in enumerate(prompt["videos"], 1):
            frames, soundtrack = _video_at_24fps(video, with_soundtrack, f"reference video {i}")
            ref_videos[f"ref_video_{i}"] = frames
            if soundtrack is not None:
                ref_video_audios[f"ref_video_audio_{i}"] = soundtrack
        out = MiniMaxH3ReferenceToVideo.execute(
            clip, prompt["text"], width, height, length, ref_image_size=ref_image_size,
            vae=vae, audio_vae=audio_vae,
            ref_images={f"ref_image_{i}": img for i, img in enumerate(prompt["images"], 1)},
            ref_videos=ref_videos,
            ref_video_audios=ref_video_audios,
            ref_audios={f"ref_audio_{i}": a for i, a in enumerate(prompt["audios"], 1)},
        )
        return io.NodeOutput(out[0], out[1], prompt["text"])


_NODES = [MiniMaxH3Definition, MiniMaxH3Definitions, MiniMaxH3Ref2VAPrompt, MiniMaxH3KeyframePrompt, MiniMaxH3Encode]

NODE_CLASS_MAPPINGS = {n.GET_SCHEMA().node_id: n for n in _NODES}
NODE_DISPLAY_NAME_MAPPINGS = {n.GET_SCHEMA().node_id: n.GET_SCHEMA().display_name for n in _NODES}
