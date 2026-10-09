"""MiniMax H3 prompt composition in the official rewrite formats.

Prompt text refers to definitions by name (<Subject fred>, <Picture opening>,
<Video dolly>, <Audio fredvoice>) and to speakers as (S fred) or
(S fred, S mara). Each clip is its own target video, so composing a clip
numbers only what that clip uses.

Formats: MiniMaxAI/MiniMax-H3 docs/VIDEO_PROMPT_WRITING_GUIDE_ref_en.md
(Ref2VA) and docs/VIDEO_PROMPT_WRITING_GUIDE_base_en.md (T2VA/I2VA/FL2VA/L2VA).
"""
import re
from enum import Enum

FPS = 24
NAME_PATTERN = r"[A-Za-z_][A-Za-z0-9_-]*"
_NAME_RE = re.compile(rf"^{NAME_PATTERN}$")
_LABEL_RE = re.compile(rf"<(Subject|Picture|Video|Audio) ({NAME_PATTERN})>")
_LOCAL_RE = re.compile(r"<(Picture|Video) (\d+)>")
_SPEAKER_RE = re.compile(rf"\(S {NAME_PATTERN}(?:\s*,\s*S {NAME_PATTERN})*\)")
_SHOT_RE = re.compile(r"\[Shot (\d+)\]")


class Kind(str, Enum):
    SUBJECT = "Subject"
    PICTURE = "Picture"
    VIDEO = "Video"
    AUDIO = "Audio"


class Retention(str, Enum):
    FULLY_PRESERVED = "fully_preserved"
    PARTIALLY_PRESERVED = "partially_preserved"
    ATTRIBUTE_TRANSFER = "attribute_transfer"
    WEAK_REFERENCE = "weak_reference"


class AudioRetention(str, Enum):
    FULLY_COPY = "fully_copy"
    PARTIALLY_COPY = "partially_copy"
    REFERENCE = "reference"
    WEAK_REFERENCE = "weak_reference"


class PictureRole(str, Enum):
    FIRST_FRAME = "first_frame"
    KEYFRAME = "keyframe"
    LAST_FRAME = "last_frame"
    EDITED_KEYFRAME = "edited_keyframe"
    COMPOSITION_ANCHOR = "composition_anchor"
    STORYBOARD = "storyboard"


class VideoRole(str, Enum):
    EDIT_SOURCE = "edit_source"
    CONTINUATION_SOURCE = "continuation_source"
    STRUCTURE_REFERENCE = "structure_reference"


class AudioRole(str, Enum):
    VOICE_TIMBRE = "voice_timbre"
    MUSIC_STYLE = "music_style"
    DIALOGUE_REUSE = "dialogue_reuse"
    SOUND_EFFECTS = "sound_effects"
    BEAT = "beat"
    FULL_TRACK = "full_track"


class TaskType(str, Enum):
    KEYFRAME_COMPLETION = "keyframe completion"
    REFERENCE_GENERATION = "reference generation"
    VIDEO_EDITING = "video editing"
    VIDEO_CONTINUATION = "video continuation"
    AUDIO_REUSE = "audio reuse"
    AUDIO_REFERENCE = "audio reference"


class SizeRounding(str, Enum):
    UP = "up"
    DOWN = "down"


_FRAME_ROLES = {PictureRole.FIRST_FRAME, PictureRole.KEYFRAME, PictureRole.LAST_FRAME, PictureRole.EDITED_KEYFRAME}
_VIDEO_TASKS = {
    VideoRole.EDIT_SOURCE: TaskType.VIDEO_EDITING,
    VideoRole.CONTINUATION_SOURCE: TaskType.VIDEO_CONTINUATION,
    VideoRole.STRUCTURE_REFERENCE: TaskType.REFERENCE_GENERATION,
}
_KIND_ORDER = {Kind.SUBJECT: 0, Kind.PICTURE: 1, Kind.VIDEO: 2, Kind.AUDIO: 3}


class Definition:
    """One named reference. Media are kept as given; Video definitions with a
    soundtrack_retention also expose their soundtrack as <Audio name>."""

    def __init__(self, name, kind, text="", retention=None, note="", role=None,
                 images=(), videos=(), audio=None, soundtrack_retention=None):
        name = str(name).strip()
        if not _NAME_RE.match(name):
            raise ValueError(
                f"MiniMax H3 definition name {name!r} must start with a letter or _ "
                "and contain only letters, digits, _ and -."
            )
        self.name = name
        self.kind = Kind(kind)
        self.text = str(text).strip()
        self.retention = retention
        self.note = str(note).strip()
        self.role = role
        self.images = list(images)
        self.videos = list(videos)
        self.audio = audio
        self.soundtrack_retention = soundtrack_retention

        needs = {
            Kind.PICTURE: ("image", len(self.images)),
            Kind.VIDEO: ("video", len(self.videos)),
            Kind.AUDIO: ("audio", 1 if self.audio is not None else 0),
        }.get(self.kind)
        if needs is not None and needs[1] != 1:
            raise ValueError(f"MiniMax H3 {self.kind.value} definition '{name}' needs exactly one {needs[0]}.")


def check_unique_names(definitions):
    seen = set()
    for d in definitions:
        if d.name in seen:
            raise ValueError(f"MiniMax H3 definitions: the name '{d.name}' is used more than once.")
        seen.add(d.name)


def _index_of(items, item):
    for i, seen in enumerate(items):
        if seen is item:
            return i
    items.append(item)
    return len(items) - 1


def _pick(by_name, texts):
    picked = set()
    pending = list(texts)
    while pending:
        for kind, name in _LABEL_RE.findall(pending.pop()):
            d = by_name.get(name)
            if d is None:
                raise ValueError(f"MiniMax H3 prompt: <{kind} {name}> doesn't match any definition.")
            if name not in picked:
                picked.add(name)
                pending += [d.text, d.note]
    return picked


def _speaker_names(token):
    return [part.strip()[2:].strip() for part in token[1:-1].split(",")]


def speaker_numbers(description):
    """Number speakers in order of first appearance, as the guide numbers vocal events."""
    numbers = {}
    for token in _SPEAKER_RE.findall(description):
        for name in _speaker_names(token):
            numbers.setdefault(name, len(numbers) + 1)
    return numbers


def resolve_speakers(text, numbers, where):
    def sub(m):
        found = []
        for name in _speaker_names(m.group(0)):
            if name not in numbers:
                raise ValueError(
                    f"MiniMax H3 prompt: speaker '{name}' in {where} never speaks in this clip's description."
                )
            found.append(numbers[name])
        return "(" + ",".join(f"S{n}" for n in sorted(set(found))) + ")"
    return _SPEAKER_RE.sub(sub, text)


def final_shot(description):
    shots = [int(n) for n in _SHOT_RE.findall(description)]
    return max(shots) if shots else 1


def _shots_mentioning(description, label):
    parts = _SHOT_RE.split(description)
    return [int(parts[i]) for i in range(1, len(parts) - 1, 2) if label in parts[i + 1]]


def _shot_list(shots):
    return ", ".join(f"[Shot {n}]" for n in shots)


def _retention_line(label, paren, marker, note):
    head = f"{label} ({paren})" if paren else label
    return f"{head}: {marker} - {note}" if note else f"{head}: {marker}"


def _derive_task_types(definitions):
    found = set()
    for d in definitions:
        if d.kind is Kind.SUBJECT:
            found.add(TaskType.REFERENCE_GENERATION)
        elif d.kind is Kind.PICTURE:
            found.add(TaskType.KEYFRAME_COMPLETION if PictureRole(d.role) in _FRAME_ROLES else TaskType.REFERENCE_GENERATION)
        elif d.kind is Kind.VIDEO:
            found.add(_VIDEO_TASKS[VideoRole(d.role)])
        retention = d.retention if d.kind is Kind.AUDIO else d.soundtrack_retention
        if retention is not None and d.kind in (Kind.AUDIO, Kind.VIDEO):
            copied = AudioRetention(retention) in (AudioRetention.FULLY_COPY, AudioRetention.PARTIALLY_COPY)
            found.add(TaskType.AUDIO_REUSE if copied else TaskType.AUDIO_REFERENCE)
    return [t for t in TaskType if t in found]


def compose_ref2va(definitions, task_types, summary, detailed_description,
                   overall_soundscape, non_diegetic_music):
    """Render the six-section Ref2VA rewrite and collect its media in core order.

    Returns {"mode": "ref2va", "text", "images", "videos", "audios"}: images
    are IMAGE tensors, videos are (video, soundtrack_as_audio) pairs and audios
    are AUDIO dicts, each list in the order MiniMaxH3ReferenceToVideo numbers
    <Picture N>, <Video N> and <Audio N>.
    """
    by_name = {d.name: d for d in definitions}
    fields = [summary, detailed_description, overall_soundscape, non_diegetic_music]
    picked_names = _pick(by_name, fields)
    picked = [d for d in definitions if d.name in picked_names]

    labels = {}
    local = {}
    for n, d in enumerate((d for d in picked if d.kind is Kind.SUBJECT), 1):
        labels[(Kind.SUBJECT.value, d.name)] = n

    images = []
    videos = []
    soundtracks = []
    for d in picked:
        pictures = [_index_of(images, img) + 1 for img in d.images]
        clips = []
        for video in d.videos:
            index = _index_of(videos, video)
            if index == len(soundtracks):
                soundtracks.append(None)
            if d.kind is Kind.VIDEO and d.soundtrack_retention is not None:
                soundtracks[index] = d.name
            clips.append(index + 1)
        local[d.name] = {"Picture": pictures, "Video": clips}
        if d.kind is Kind.PICTURE:
            labels[(Kind.PICTURE.value, d.name)] = pictures[0]
        elif d.kind is Kind.VIDEO:
            labels[(Kind.VIDEO.value, d.name)] = clips[0]

    audio_count = 0
    for owner in soundtracks:
        if owner is not None:
            audio_count += 1
            labels[(Kind.AUDIO.value, owner)] = audio_count
    audios = []
    for d in picked:
        if d.kind is Kind.AUDIO:
            audios.append(d.audio)
            audio_count += 1
            labels[(Kind.AUDIO.value, d.name)] = audio_count

    speakers = speaker_numbers(detailed_description)

    def resolve(text, where, owner=None):
        if owner is not None:
            def local_sub(m):
                kind, k = m.group(1), int(m.group(2))
                numbers = local[owner.name][kind]
                if not 1 <= k <= len(numbers):
                    raise ValueError(
                        f"MiniMax H3 prompt: definition '{owner.name}' has no attached {kind.lower()} {k}."
                    )
                return f"<{kind} {numbers[k - 1]}>"
            text = _LOCAL_RE.sub(local_sub, text)

        def label_sub(m):
            kind, name = m.group(1), m.group(2)
            n = labels.get((kind, name))
            if n is None:
                d = by_name[name]
                if d.kind is Kind.VIDEO and kind == Kind.AUDIO.value:
                    raise ValueError(f"MiniMax H3 prompt: <Audio {name}> needs soundtrack_as_audio on video '{name}'.")
                raise ValueError(f"MiniMax H3 prompt: <{kind} {name}> refers to a {d.kind.value} definition.")
            return f"<{kind} {n}>"
        return resolve_speakers(_LABEL_RE.sub(label_sub, text), speakers, where)

    entries = []
    for d in picked:
        label = f"<{d.kind.value} {labels[(d.kind.value, d.name)]}>"
        text = resolve(d.text, f"definition '{d.name}'", d)
        note = resolve(d.note, f"definition '{d.name}'", d)
        shots = _shots_mentioning(detailed_description, f"<{d.kind.value} {d.name}>")
        if d.kind is Kind.SUBJECT:
            paren = f"appears in {_shot_list(shots)}" if shots else ""
        elif d.kind in (Kind.PICTURE, Kind.VIDEO):
            role = str(d.role).replace("_", " ")
            paren = f"{_shot_list(shots)} {role}" if shots else role
        else:
            paren = ""
        entries.append((_KIND_ORDER[d.kind], labels[(d.kind.value, d.name)],
                        f"{label} {text}".rstrip(), _retention_line(label, paren, d.retention, note)))
        if d.kind is Kind.VIDEO and d.soundtrack_retention is not None:
            audio = f"<Audio {labels[(Kind.AUDIO.value, d.name)]}>"
            entries.append((_KIND_ORDER[Kind.AUDIO], labels[(Kind.AUDIO.value, d.name)],
                            f"{audio} is the synchronized audio track of {label}.",
                            _retention_line(audio, "", d.soundtrack_retention, f"the synchronized audio track of {label}")))
    entries.sort(key=lambda e: (e[0], e[1]))

    tasks = [t for t in TaskType if t.value in set(task_types)] if task_types else _derive_task_types(picked)
    prefix = "[" + " + ".join(t.value for t in tasks) + "]" if tasks else ""
    summary_line = " ".join(s for s in (prefix, resolve(summary.strip(), "the summary")) if s)

    sections = [
        ("subject_definitions", "\n".join(e[2] for e in entries)),
        ("summary", summary_line),
        ("retention_analysis", "\n".join(e[3] for e in entries)),
        ("detailed_description", resolve(detailed_description.strip(), "the detailed description")),
        ("overall_soundscape", resolve(overall_soundscape.strip(), "the soundscape") or "N/A"),
        ("non_diegetic_music", resolve(non_diegetic_music.strip(), "the music") or "N/A"),
    ]
    return {
        "mode": "ref2va",
        "text": "\n\n".join(f"{name}:\n{body}" for name, body in sections),
        "images": images,
        "videos": [(video, owner is not None) for video, owner in zip(videos, soundtracks)],
        "audios": audios,
    }


def compose_keyframe(integrated_multimodal_description, overall_soundscape, non_diegetic_music,
                     first_frame=None, last_frame=None):
    """Body of a T2VA/I2VA/FL2VA/L2VA prompt; the alignment line needs the clip length, see keyframe_instruction."""
    description = integrated_multimodal_description.strip()
    speakers = speaker_numbers(description)
    body = "\n\n".join([
        "integrated_multimodal_description: " + resolve_speakers(description, speakers, "the description"),
        "overall_soundscape: " + (resolve_speakers(overall_soundscape.strip(), speakers, "the soundscape") or "N/A"),
        "non_diegetic_music: " + (resolve_speakers(non_diegetic_music.strip(), speakers, "the music") or "N/A"),
    ])
    return {
        "mode": "keyframe",
        "body": body,
        "first_frame": first_frame,
        "last_frame": last_frame,
        "final_shot": final_shot(description),
    }


def keyframe_instruction(has_first, has_last, frame_count, last_shot):
    """The guide's fixed first line for I2VA, FL2VA and L2VA; T2VA has none."""
    end = f"{frame_count / FPS:.2f}"
    if has_first and has_last:
        return ("How the reference pictures align with the target video — Picture 1 (from Shot 1) aligns with "
                f"the 0.00-second mark of the target video; Picture 2 (from Shot {last_shot}) aligns with the "
                f"{end}-second mark of the target video.")
    if has_first:
        return "For the target video, at 0.00 seconds into the target video, <Picture 1> (from [Shot 1]) is fully referenced."
    if has_last:
        return ("How the reference pictures align with the target video — <Picture 1> (from "
                f"[Shot {last_shot}]) aligns with the {end}-second mark of the target video.")
    return ""
