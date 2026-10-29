#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Pin one-time bring-up traces of Qwen's own Qwen3-TTS package (docs/models/qwen3-tts.md).

Runs the authors' `qwen-tts` (QwenLM/Qwen3-TTS at `022e286b`, `transformers==4.57.3`)
on the CPU in F32 at seed 0, one case per `Spec`, and writes small little-endian
files under `tests/fixtures/qwen3-tts/<case>/` with an `index.json` naming each
file's shape, dtype, rows, and SHA-256, the model revision, the package commit, and
the library versions. Every array is row-major, time first: a row is one frame (or
one upsampled step) of all channels.

Sections, each a function that adds files to a case:

- `codes`: the talker and code predictor sample the sentence's frames
  (`generate_custom_voice`, the authors' defaults), captured at the codec's door.
- `talker`: hooked during that same generation (`Spec.talker`): the token ids the
  processor returned, the prompt's rows as (text id, codec id) pairs derived from the
  config by `generate`'s rules and checked against the rows the talker received, the
  talker's input rows, its residual stream after layers 0, 13, 27 and after the final
  norm, the group-0 logits of the first frames, and the code predictor's logits for
  groups 1-15 of the first two frames. Replaying the pinned codes, nuclis reproduces
  every row: a sampled id is never compared.
- `codec`: the authors' codec decoder on the case's frames (`Spec.codec_frames`),
  hooked at each stage boundary; each stage pinned keeps its last rows
  (`Spec.codec_rows`). The `fox` case pins every stage of its first 24 frames; the
  `window` case decodes 112 frames, past the transformer's 72-frame window, and pins
  the last rows only.

The Base (voice cloning), `--clone`, one reference case and one case per mode:

- `reference`: the clip, spoken by the 0.6B CustomVoice (`REFERENCE`) and pinned
  as the 16-bit WAV it is read back from; then on the Base, through
  `create_voice_clone_prompt`, the speaker encoder's input mel, stages, and
  embedding, and the codec encoder's stages and codes.
- each mode (`xvector`, `icl`): `generate_voice_clone` on that item, with the
  talker section's hooks, the prompt's layout checked as for CustomVoice, the
  generated codes, and the samples after the reference's are cut.

Run from the repository root (the venv: docs/models/qwen3-tts.md):

    .reference/venv-qwen3-tts/bin/python scripts/speech-traces/qwen3_tts_trace.py [--cases NAME,...]
    .reference/venv-qwen3-tts/bin/python scripts/speech-traces/qwen3_tts_trace.py --clone

`--cases` re-runs only the named cases and keeps the others' index entries.
`--debug-dir DIR` also writes every hooked stage whole, unpinned, for localizing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "tests/fixtures/qwen3-tts"
MODELS = pathlib.Path(os.path.expanduser("~/.nuclis/models/Qwen"))
PACKAGE = "QwenLM/Qwen3-TTS"
PACKAGE_COMMIT = "022e286b98fbec7e1e916cb940cdf532cd9f488e"
SEED = 0
SAMPLES_PER_FRAME = 1920
# The authors decode longer clips in re-warmed chunks (docs/models/qwen3-tts.md).
CHUNK_FRAMES = 300

SIZES = {
    "0.6b": {"dir": "Qwen3-TTS-12Hz-0.6B-CustomVoice", "repo": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"},
    "1.7b": {"dir": "Qwen3-TTS-12Hz-1.7B-CustomVoice", "repo": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"},
}
SENTENCES = {
    "fox": "The quick brown fox jumps over the lazy dog.",
    "window": (
        "Speech synthesis turns written words into sound. The talker predicts one frame of codes "
        "at a time, and the codec decoder turns those frames back into a waveform."
    ),
}
# Codec stage boundaries: (name, the module hooked, which side of it). Rows are time
# steps; `pre_conv`'s input is the quantizer's output.
CODEC_HOOKS = [
    ("quantized", "pre_conv", "input"),
    ("pre_conv", "pre_conv", "output"),
    ("transformer", "pre_transformer", "output"),
    ("upsample.0", "upsample.0.1", "output"),
    ("upsample.1", "upsample.1.1", "output"),
    ("decoder.0", "decoder.0", "output"),
    ("decoder.1", "decoder.1", "output"),
    ("decoder.2", "decoder.2", "output"),
    ("decoder.3", "decoder.3", "output"),
    ("decoder.4", "decoder.4", "output"),
]
# Rows pinned per stage (the last ones; None: all), sized to keep the fixtures small.
EVERY_STAGE: dict[str, int | None] = {
    "quantized": None,
    "pre_conv": None,
    "transformer": None,
    "upsample.0": 32,
    "upsample.1": 32,
    "decoder.0": 16,
    "decoder.1": 32,
    "decoder.2": 64,
    "decoder.3": 128,
    "decoder.4": 256,
    "samples": None,
}
PAST_THE_WINDOW: dict[str, int | None] = {"transformer": 16, "samples": 2 * SAMPLES_PER_FRAME}
INSTRUCTIONS = {"warm": "Speak in a warm, cheerful tone."}
BASE = {"0.6b": {"dir": "Qwen3-TTS-12Hz-0.6B-Base", "repo": "Qwen/Qwen3-TTS-12Hz-0.6B-Base"}}
# The reference clip: a CustomVoice speaker other than the cases' `ryan`, so a clone is
# told from the default by ear.
REFERENCE = {
    "size": "0.6b",
    "speaker": "vivian",
    "language": "english",
    "text": "Every morning I walk along the river and listen to the birds.",
}
CLONE_MODES = ("xvector", "icl")
# Rows pinned of the encoders' stages (the last ones; None: all).
SPEAKER_STAGES: dict[str, int | None] = {"mel": 64, "blocks.0": 8, "blocks.3": 8, "mfa": 8}
CODEC_ENCODER_STAGES: dict[str, int | None] = {"seanet": 16, "transformer": 16, "downsample": 16}
# A codec-track marker in a clone prompt row: the speaker embedding instead of a codec id.
SPEAKER_ROW = -2
# The talker layers whose output (the residual stream) is pinned.
TALKER_LAYERS = (0, 13, 27)


@dataclass(frozen=True)
class Talker:
    """What the `talker` section pins."""

    # Prompt rows of each hidden stage kept, the last ones (None: all); every decode row is.
    prompt_rows: int | None
    # Decode steps whose talker rows are pinned; group-0 logits cover frames 0..frames.
    frames: int = 4
    # Frames whose predictor logits (groups 1-15) are pinned.
    predictor_frames: int = 2


@dataclass(frozen=True)
class Spec:
    size: str
    speaker: str
    language: str
    sentence: str
    # Frames the codec decodes from the start (None: all); a prefix decodes as itself.
    codec_frames: int | None
    # Stages pinned by the `codec` section; empty: the section does not run.
    codec_rows: dict[str, int | None]
    talker: Talker | None = None
    # A key of INSTRUCTIONS (the 1.7B only: the authors drop it on the 0.6B).
    instruct: str | None = None

    @property
    def name(self) -> str:
        suffix = f"-{self.instruct}" if self.instruct else ""
        return f"{self.size}-{self.speaker}-{self.sentence}{suffix}"


CASES = [
    Spec("0.6b", "ryan", "english", "fox", 24, EVERY_STAGE, Talker(prompt_rows=None)),
    # 112 frames: 40 past the window, ending in speech (the clip's tail is silence).
    Spec("0.6b", "ryan", "english", "window", 112, PAST_THE_WINDOW),
    # The 1.7B's codec file is the 0.6B's (byte-identical), so it pins the talker only.
    Spec("1.7b", "ryan", "english", "fox", None, {}, Talker(prompt_rows=4)),
    Spec("1.7b", "ryan", "english", "fox", None, {}, Talker(prompt_rows=4), instruct="warm"),
]


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


class Case:
    """One case's directory and its index entries."""

    def __init__(self, name: str, meta: dict[str, Any], debug: pathlib.Path | None):
        self.name = name
        self.dir = OUT / name
        self.meta = meta
        self.files: dict[str, dict[str, Any]] = {}
        self.debug = debug / name if debug else None

    def add(self, key: str, array: np.ndarray, dtype: str, note: str, rows: tuple[int, int] | None = None) -> None:
        """Writes `array` as `<key>.<dtype>` (`f32` or `u16`, little endian)."""
        np_dtype = {"f32": "<f4", "u16": "<u2"}[dtype]
        data = np.ascontiguousarray(array.astype(np_dtype))
        path = self.dir / f"{key}.{dtype}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data.tobytes())
        entry: dict[str, Any] = {
            "file": path.relative_to(OUT).as_posix(),
            "shape": list(data.shape),
            "dtype": dtype,
            "sha256": sha256(path),
            "note": note,
        }
        if rows is not None:
            entry["rows"] = list(rows)
        self.files[key] = entry

    def dump(self, key: str, array: np.ndarray) -> None:
        """Writes a whole stage under the debug directory, unpinned."""
        if self.debug is None:
            return
        self.debug.mkdir(parents=True, exist_ok=True)
        np.ascontiguousarray(array.astype("<f4")).tofile(self.debug / f"{key}.f32")


def time_major(tensor: Any, channels_first: bool) -> np.ndarray:
    """One batch item as [T, C]: a convolution's [1, C, T] is transposed, a transformer's [1, T, C] kept."""
    value = tensor.detach().float().cpu().numpy()
    assert value.shape[0] == 1, value.shape
    return value[0].T if channels_first else value[0]


class TalkerTrace:
    """Hooks on the talker and the code predictor for one generation, then the pinned files."""

    def __init__(self, tts: Any, pins: Talker):
        self.tts = tts
        self.pins = pins
        self.tokenized: list[list[int]] = []
        self.inputs: list[np.ndarray] = []
        self.layers: dict[int, list[np.ndarray]] = {i: [] for i in TALKER_LAYERS}
        self.norm: list[np.ndarray] = []
        self.logits: list[np.ndarray] = []
        self.predictor: list[np.ndarray] = []
        self.handles: list[Any] = []
        self.tokenize = tts._tokenize_texts

    def install(self) -> None:
        talker = self.tts.model.talker
        calls = self.pins.frames + 1  # the prefill, then one call per frame

        def rows(value: Any) -> np.ndarray:
            return value.detach().float().cpu().numpy()[0]

        def keep(store: list[np.ndarray], limit: int) -> Callable[..., None]:
            def record(_module: Any, _inputs: Any, output: Any) -> None:
                if len(store) < limit:
                    store.append(rows(output[0] if isinstance(output, tuple) else output))

            return record

        def inputs(_module: Any, _args: Any, kwargs: Any) -> None:
            if len(self.inputs) < calls:
                self.inputs.append(rows(kwargs["inputs_embeds"]))

        self.handles.append(talker.model.register_forward_pre_hook(inputs, with_kwargs=True))
        for i in TALKER_LAYERS:
            self.handles.append(talker.model.layers[i].register_forward_hook(keep(self.layers[i], calls)))
        self.handles.append(talker.model.norm.register_forward_hook(keep(self.norm, calls)))
        self.handles.append(talker.codec_head.register_forward_hook(keep(self.logits, calls)))
        # Each frame's predictor calls lm_head[0] (on the two prefill rows), then lm_head[1..14].
        heads = 15 * self.pins.predictor_frames
        for head in talker.code_predictor.lm_head:
            self.handles.append(head.register_forward_hook(keep(self.predictor, heads)))

        def tokenize(texts: list[str]) -> Any:
            out = self.tokenize(texts)
            self.tokenized.extend(ids[0].tolist() for ids in out)
            return out

        self.tts._tokenize_texts = tokenize

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.handles = []
        self.tts._tokenize_texts = self.tokenize

    def write(self, case: Case, spec: Spec) -> None:
        """Pins the captured rows; the prompt's layout is derived from the config and checked."""
        import torch

        text_ids = self.tokenized[0]
        instruct_ids = self.tokenized[1] if spec.instruct else []
        assert len(self.tokenized) == (2 if spec.instruct else 1), len(self.tokenized)
        layout = prompt_rows(self.tts.model.config, text_ids, instruct_ids, spec.speaker, spec.language)
        prompt = self.inputs[0]
        assert prompt.shape[0] == len(layout), (prompt.shape, len(layout))
        talker = self.tts.model.talker
        with torch.inference_mode():
            text = torch.tensor([[t for t, _ in layout]])
            codec = torch.tensor([[max(c, 0) for _, c in layout]])
            rebuilt = talker.text_projection(talker.get_text_embeddings()(text))
            has_codec = torch.tensor([[c >= 0 for _, c in layout]]).unsqueeze(-1)
            rebuilt = rebuilt + talker.get_input_embeddings()(codec) * has_codec
        rebuilt_np = rebuilt.float().numpy()[0]
        difference = float(np.max(np.abs(rebuilt_np - prompt)))
        assert difference <= 1e-5 * float(np.max(np.abs(prompt))), difference
        case.meta.update(
            {
                "text_ids": text_ids,
                "instruct_ids": instruct_ids,
                "prompt_rows": [list(r) for r in layout],
                "prompt_rows_note": "[text id, codec id] per prompt row, -1 for none: derived from config.json "
                "by generate's rules; rebuilt from them, the rows equal those the talker received",
                "prompt_rebuilt_max_abs": difference,
            }
        )
        self.pin(case)

    def pin(self, case: Case) -> None:
        """Pins the talker's rows, logits, and the predictor's logits of the captured generation."""
        prompt = self.inputs[0]
        case.meta["talker_frames"] = self.pins.frames
        case.meta["predictor_frames"] = self.pins.predictor_frames
        length = prompt.shape[0]
        total = length + self.pins.frames
        first = 0 if self.pins.prompt_rows is None else length - self.pins.prompt_rows
        whole = np.concatenate(self.inputs)
        assert whole.shape[0] == total, whole.shape
        case.add(
            "talker/inputs",
            whole,
            "f32",
            f"the talker's input rows: the prompt's {length}, then decode rows of frames 0..{self.pins.frames - 1}",
            (0, total),
        )
        stages = {f"talker/layer.{i}": self.layers[i] for i in TALKER_LAYERS} | {"talker/norm": self.norm}
        for key, store in stages.items():
            value = np.concatenate(store)
            assert value.shape[0] == total, (key, value.shape)
            what = "after the final norm" if key.endswith("norm") else f"after layer {key.rsplit('.', 1)[1]}"
            case.add(
                key,
                value[first:],
                "f32",
                f"the talker's rows {what}, rows {first}..{total} (prompt then decode)",
                (first, total),
            )
        logits = np.stack([call[-1] for call in self.logits])
        case.add(
            "talker/logits",
            logits,
            "f32",
            f"group-0 logits (codec_head, before every processor) of frames 0..{logits.shape[0] - 1}",
        )
        predictor = np.stack([call[-1] for call in self.predictor])
        assert predictor.shape == (15 * self.pins.predictor_frames, 2048), predictor.shape
        case.add(
            "predictor/logits",
            predictor,
            "f32",
            "the predictor's logits, row 15 * frame + group - 1, before temperature and top-k",
        )


def prompt_rows(
    config: Any, text_ids: list[int], instruct_ids: list[int], speaker: str, language: str
) -> list[tuple[int, int]]:
    """The non-streaming prompt as (text id, codec id) rows, -1 for an absent track (docs/models/qwen3-tts.md § (a))."""
    t = config.talker_config
    language = language.lower()
    language_id = None if language == "auto" else t.codec_language_id[language]
    dialect = t.spk_is_dialect[speaker.lower()]
    if language in ("chinese", "auto") and dialect:
        language_id = t.codec_language_id[dialect]
    if language_id is None:
        prefix = [t.codec_nothink_id, t.codec_think_bos_id, t.codec_think_eos_id]
    else:
        prefix = [t.codec_think_id, t.codec_think_bos_id, language_id, t.codec_think_eos_id]
    codec = [*prefix, t.spk_id[speaker.lower()], t.codec_pad_id]
    text = [config.tts_pad_token_id] * (len(codec) - 1) + [config.tts_bos_token_id]
    rows = [(i, -1) for i in instruct_ids] + [(i, -1) for i in text_ids[:3]]
    rows += list(zip(text, codec, strict=True))
    rows += [(i, t.codec_pad_id) for i in text_ids[3:-5]]
    rows += [(config.tts_eos_token_id, t.codec_pad_id), (config.tts_pad_token_id, t.codec_bos_id)]
    return rows


def section_codes(tts: Any, case: Case, spec: Spec, talker: TalkerTrace | None) -> np.ndarray:
    """Samples the sentence's codes with the authors' defaults; returns [frames, 16]."""
    import torch

    tokenizer = tts.model.speech_tokenizer
    captured: list[Any] = []
    decode = tokenizer.decode

    def capture(encoded: Any) -> Any:
        captured.append(encoded[0]["audio_codes"])
        return decode(encoded)

    tokenizer.decode = capture
    text = SENTENCES[spec.sentence]
    instruct = INSTRUCTIONS[spec.instruct] if spec.instruct else None
    if talker:
        talker.install()
    torch.manual_seed(SEED)
    try:
        wavs, rate = tts.generate_custom_voice(
            text=text, speaker=spec.speaker, language=spec.language, instruct=instruct
        )
    finally:
        tokenizer.decode = decode
        if talker:
            talker.remove()
    codes = captured[0].detach().cpu().numpy().astype(np.int64)
    assert codes.ndim == 2 and codes.shape[1] == 16 and codes.min() >= 0 and codes.max() < 2048, codes.shape
    assert rate == 24000 and wavs[0].shape[0] == codes.shape[0] * SAMPLES_PER_FRAME
    case.add("codes", codes, "u16", "every frame the talker and predictor sampled, [frame][group]")
    case.meta["frames_generated"] = int(codes.shape[0])
    case.dump("wav", wavs[0])
    return codes


def module_at(root: Any, dotted: str) -> Any:
    for part in dotted.split("."):
        root = root[int(part)] if part.isdigit() else getattr(root, part)
    return root


def section_codec(tts: Any, case: Case, spec: Spec, codes: np.ndarray) -> None:
    """The codec decoder on the case's frames, every stage hooked, the chosen rows pinned."""
    import torch

    frames = codes.shape[0] if spec.codec_frames is None else spec.codec_frames
    if codes.shape[0] < frames:
        raise SystemExit(f"{case.name}: {codes.shape[0]} frames generated, fewer than {frames}")
    if frames > CHUNK_FRAMES:
        raise SystemExit(f"{case.name}: {frames} frames, past the authors' {CHUNK_FRAMES}-frame chunk")
    tokenizer = tts.model.speech_tokenizer
    decoder = tokenizer.model.decoder
    stages: dict[str, np.ndarray] = {}
    handles = []

    def hook(name: str, side: str) -> Callable[..., None]:
        def record(_module: Any, inputs: Any, output: Any = None) -> None:
            value = inputs[0] if side == "input" else output
            transformer = hasattr(value, "last_hidden_state")
            stages[name] = time_major(value.last_hidden_state if transformer else value, not transformer)

        return record

    for name, path, side in CODEC_HOOKS:
        module = module_at(decoder, path)
        if side == "input":
            handles.append(module.register_forward_pre_hook(hook(name, side)))
        else:
            handles.append(module.register_forward_hook(hook(name, side)))
    wavs, _ = tokenizer.decode([{"audio_codes": torch.from_numpy(codes[:frames])}])
    for h in handles:
        h.remove()
    stages["samples"] = np.asarray(wavs[0], dtype=np.float32)
    assert stages["samples"].shape[0] == frames * SAMPLES_PER_FRAME

    for name, value in stages.items():
        case.dump(f"codec.{name}", value)
        if name not in spec.codec_rows:
            continue
        kept = spec.codec_rows[name]
        start = 0 if kept is None else value.shape[0] - kept
        what = "the clamped 24 kHz samples" if name == "samples" else name
        case.add(
            f"codec/{name}",
            value[start:],
            "f32",
            f"{what} of the first {frames} frames, rows {start}..{value.shape[0]}",
            (start, int(value.shape[0])),
        )

    case.meta["codec_frames"] = frames
    case.meta["codec_attention"] = decoder.config._attn_implementation
    if frames < codes.shape[0]:
        # The decoder is causal: the prefix's samples are the whole decode's, up to summation order.
        whole, _ = tokenizer.decode([{"audio_codes": torch.from_numpy(codes)}])
        prefix = np.asarray(whole[0])[: stages["samples"].shape[0]]
        case.meta["codec_prefix_vs_whole_max_abs"] = float(np.max(np.abs(prefix - stages["samples"])))


def clone_prompt_rows(
    config: Any, text_ids: list[int], ref_ids: list[int] | None, language: str, frames: int
) -> tuple[list[tuple[int, int, int]], list[int]]:
    """A clone prompt as (text id, codec id, reference frame) rows, -1 for none and
    SPEAKER_ROW for the embedding, and the trailing text ids (docs/models/qwen3-tts.md
    § Voice cloning (the Base))."""
    t = config.talker_config
    language = language.lower()
    if language == "auto":
        prefix = [t.codec_nothink_id, t.codec_think_bos_id, t.codec_think_eos_id]
    else:
        prefix = [t.codec_think_id, t.codec_think_bos_id, t.codec_language_id[language], t.codec_think_eos_id]
    codec = [*prefix, SPEAKER_ROW, t.codec_pad_id]
    text = [config.tts_pad_token_id] * (len(codec) - 1) + [config.tts_bos_token_id]
    rows = [(i, -1, -1) for i in text_ids[:3]] + [(x, c, -1) for x, c in zip(text, codec, strict=True)]
    if ref_ids is None:
        rows.append((text_ids[3], t.codec_bos_id, -1))
        return rows, [*text_ids[4:-5], config.tts_eos_token_id]
    track = [*ref_ids[3:-2], *text_ids[3:-5], config.tts_eos_token_id]
    codec_track = [(t.codec_bos_id, -1)] + [(-1, k) for k in range(frames)]
    if len(track) > len(codec_track):
        trailing = track[len(codec_track) :]
        track = track[: len(codec_track)]
    else:
        trailing = [config.tts_pad_token_id]
        track += [config.tts_pad_token_id] * (len(codec_track) - len(track))
    rows += [(x, c, k) for x, (c, k) in zip(track, codec_track, strict=True)]
    return rows, trailing


def rebuild_clone_prompt(tts: Any, rows: list[tuple[int, int, int]], embedding: Any, ref_codes: Any) -> Any:
    """The talker's input rows rebuilt from `rows`, the embedding, and the reference codes."""
    import torch

    talker = tts.model.talker
    text_rows = talker.text_projection(talker.get_text_embeddings()(torch.tensor([[max(x, 0) for x, _, _ in rows]])))
    out = []
    for i, (x, c, k) in enumerate(rows):
        row = text_rows[0, i] if x >= 0 else torch.zeros_like(text_rows[0, i])
        if c == SPEAKER_ROW:
            row = row + embedding
        elif c >= 0:
            row = row + talker.get_input_embeddings()(torch.tensor(c))
        if k >= 0:
            frame = ref_codes[k]
            row = row + talker.get_input_embeddings()(frame[0])
            for g in range(1, 16):
                row = row + talker.code_predictor.get_input_embeddings()[g - 1](frame[g])
        out.append(row)
    return torch.stack(out)


def section_reference(case: Case) -> pathlib.Path:
    """The reference clip, spoken by CustomVoice and written as 16-bit PCM; returns its path."""
    import soundfile as sf
    import torch
    from qwen_tts import Qwen3TTSModel

    size = SIZES[REFERENCE["size"]]
    voice = Qwen3TTSModel.from_pretrained(str(MODELS / size["dir"]), dtype=torch.float32, device_map="cpu")
    torch.manual_seed(SEED)
    wavs, rate = voice.generate_custom_voice(
        text=REFERENCE["text"], speaker=REFERENCE["speaker"], language=REFERENCE["language"]
    )
    del voice
    path = case.dir / "reference.wav"
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.clip(wavs[0], -1, 1), rate, subtype="PCM_16")
    case.files["reference"] = {
        "file": path.relative_to(OUT).as_posix(),
        "sha256": sha256(path),
        "note": f"the clip, 24 kHz mono 16-bit PCM: {size['repo']} speaker {REFERENCE['speaker']}, seed {SEED}",
    }
    case.meta.update({"reference_" + k: v for k, v in REFERENCE.items() if k != "size"})
    case.meta["reference_repo"] = size["repo"]
    return path


def section_encoders(tts: Any, case: Case, path: pathlib.Path) -> Any:
    """`create_voice_clone_prompt` on the clip, both encoders hooked; returns the ICL item."""
    model = tts.model
    stages: dict[str, np.ndarray] = {}
    handles = []

    def hook(name: str, channels_first: bool, side: str = "output") -> Callable[..., None]:
        def record(_module: Any, inputs: Any, output: Any = None) -> None:
            value = inputs[0] if side == "input" else output
            value = value.last_hidden_state if hasattr(value, "last_hidden_state") else value
            stages[name] = time_major(value, channels_first)

        return record

    speaker = model.speaker_encoder
    handles.append(speaker.register_forward_pre_hook(hook("mel", False, "input")))
    handles.append(speaker.blocks[0].register_forward_hook(hook("blocks.0", True)))
    handles.append(speaker.blocks[3].register_forward_hook(hook("blocks.3", True)))
    handles.append(speaker.mfa.register_forward_hook(hook("mfa", True)))
    handles.append(speaker.asp.register_forward_hook(hook("asp", True)))
    encoder = model.speech_tokenizer.model.encoder
    handles.append(encoder.encoder.register_forward_hook(hook("seanet", True)))
    handles.append(encoder.encoder_transformer.register_forward_hook(hook("transformer", False)))
    handles.append(encoder.downsample.register_forward_hook(hook("downsample", True)))
    try:
        (item,) = tts.create_voice_clone_prompt(ref_audio=str(path), ref_text=REFERENCE["text"])
    finally:
        for h in handles:
            h.remove()

    for name, kept in (SPEAKER_STAGES | CODEC_ENCODER_STAGES).items():
        value = stages[name]
        case.dump(name, value)
        start = 0 if kept is None else value.shape[0] - kept
        what = "the speaker encoder's input mel" if name == "mel" else name
        case.add(
            f"encoders/{name}",
            value[start:],
            "f32",
            f"{what}, rows {start}..{value.shape[0]}",
            (start, int(value.shape[0])),
        )
    case.add("encoders/asp", stages["asp"].reshape(-1), "f32", "the attentive pooling's output, [mean; std]")
    embedding = item.ref_spk_embedding.detach().float().cpu().numpy()
    case.add("embedding", embedding, "f32", "ref_spk_embedding, the speaker encoder's output")
    codes = item.ref_code.detach().cpu().numpy().astype(np.int64)
    assert codes.ndim == 2 and codes.shape[1] == 16, codes.shape
    case.add("codes", codes, "u16", "ref_code: the codec encoder's 16 groups per 12.5 Hz frame, [frame][group]")
    case.meta["reference_frames"] = int(codes.shape[0])
    case.meta["codec_encoder_attention"] = encoder.config._attn_implementation
    return item


def section_clone(tts: Any, case: Case, mode: str, item: Any, pins: Talker) -> None:
    """`generate_voice_clone` in `mode` on the reference item; pins the talker and the cut samples."""
    import copy

    import torch

    item = copy.copy(item)
    if mode == "xvector":
        item.ref_code, item.x_vector_only_mode, item.icl_mode = None, True, False
    tokenizer = tts.model.speech_tokenizer
    captured: list[Any] = []
    decode = tokenizer.decode

    def capture(encoded: Any) -> Any:
        captured.append(encoded[0]["audio_codes"])
        return decode(encoded)

    tokenizer.decode = capture
    trace = TalkerTrace(tts, pins)
    trace.install()
    torch.manual_seed(SEED)
    try:
        wavs, rate = tts.generate_voice_clone(
            text=SENTENCES["fox"], language=REFERENCE["language"], voice_clone_prompt=[item]
        )
    finally:
        tokenizer.decode = decode
        trace.remove()

    ref_codes = item.ref_code
    ref_len = 0 if ref_codes is None else int(ref_codes.shape[0])
    whole = captured[0].detach().cpu().numpy().astype(np.int64)
    codes = whole[ref_len:]
    assert rate == 24000 and wavs[0].shape[0] == codes.shape[0] * SAMPLES_PER_FRAME, (wavs[0].shape, codes.shape)
    assert whole.shape[0] <= CHUNK_FRAMES, whole.shape
    case.add("codes", codes, "u16", "every frame the talker and predictor sampled, [frame][group]")
    case.meta["frames_generated"] = int(codes.shape[0])

    # The item keeps its transcript in either mode (as `create_voice_clone_prompt` makes it),
    # so the reference is tokenized after the text in both; only ICL reads it.
    assert len(trace.tokenized) == 2, len(trace.tokenized)
    text_ids = trace.tokenized[0]
    ref_ids = trace.tokenized[1] if mode == "icl" else None
    rows, trailing = clone_prompt_rows(tts.model.config, text_ids, ref_ids, REFERENCE["language"], ref_len)
    prompt = trace.inputs[0]
    assert prompt.shape[0] == len(rows), (prompt.shape, len(rows))
    rebuilt = rebuild_clone_prompt(tts, rows, item.ref_spk_embedding.float(), ref_codes).float().numpy()
    difference = float(np.max(np.abs(rebuilt - prompt)))
    assert difference <= 1e-5 * float(np.max(np.abs(prompt))), difference
    case.meta.update(
        {
            "mode": mode,
            "text": SENTENCES["fox"],
            "text_ids": text_ids,
            "ref_ids": ref_ids,
            "prompt_rows": [list(r) for r in rows],
            "prompt_rows_note": "[text id, codec id, reference frame] per prompt row, -1 for none, "
            f"{SPEAKER_ROW} for the speaker embedding: rebuilt from them, the rows equal those the talker received",
            "trailing_ids": trailing,
            "prompt_rebuilt_max_abs": difference,
        }
    )
    trace.pin(case)
    samples = np.asarray(wavs[0], dtype=np.float32)
    case.dump("wav", samples)
    head, tail = samples[: 2 * SAMPLES_PER_FRAME], samples[-SAMPLES_PER_FRAME:]
    case.add("samples/head", head, "f32", "the first 2 frames' samples after the reference's are cut")
    case.add("samples/tail", tail, "f32", "the last frame's samples")


def run_clone(index: dict[str, Any], debug: pathlib.Path | None) -> None:
    """The reference case, then one case per mode, on the 0.6B Base."""
    import torch
    from qwen_tts import Qwen3TTSModel

    size = BASE[REFERENCE["size"]]
    model_dir = MODELS / size["dir"]
    sidecar = json.loads((model_dir / "model.safetensors.nuclis.json").read_text())
    codec_sha = json.loads((model_dir / "speech_tokenizer/model.safetensors.nuclis.json").read_text())["sha256"]
    base_meta = {
        "repo": size["repo"],
        "revision": sidecar["revision"],
        "model_sha256": sidecar["sha256"],
        "codec_sha256": codec_sha,
    }
    prefix = f"{REFERENCE['size']}-clone"
    reference = Case(f"{prefix}-reference", dict(base_meta), debug)
    with torch.inference_mode():
        path = section_reference(reference)
    tts = Qwen3TTSModel.from_pretrained(str(model_dir), dtype=torch.float32, device_map="cpu")
    tts.model.eval()
    with torch.inference_mode():
        item = section_encoders(tts, reference, path)
    index["cases"][reference.name] = {**reference.meta, "files": reference.files}
    print(f"{reference.name}: {reference.meta['reference_frames']} frames; {len(reference.files)} files")
    for mode in CLONE_MODES:
        case = Case(f"{prefix}-{mode}-fox", {**base_meta, "reference": reference.name}, debug)
        with torch.inference_mode():
            section_clone(tts, case, mode, item, Talker(prompt_rows=4))
        index["cases"][case.name] = {**case.meta, "files": case.files}
        print(f"{case.name}: {case.meta['frames_generated']} frames; {len(case.files)} files")


SECTIONS = {"codes", "talker", "codec"}


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--sections", default="codes,talker,codec", help="comma-separated, of: " + ", ".join(sorted(SECTIONS))
    )
    parser.add_argument("--cases", help="comma-separated case names (default: every case)")
    parser.add_argument("--debug-dir", type=pathlib.Path, help="also write every stage whole here (unpinned)")
    parser.add_argument("--clone", action="store_true", help="the Base's cases instead (voice cloning)")
    args = parser.parse_args()
    sections = set(args.sections.split(","))
    if not sections <= SECTIONS or "codes" not in sections:
        raise SystemExit(f"--sections must include codes and be drawn from {sorted(SECTIONS)}")
    chosen = set(args.cases.split(",")) if args.cases else {spec.name for spec in CASES}
    if unknown := chosen - {spec.name for spec in CASES}:
        raise SystemExit(f"--cases: no case named {sorted(unknown)}")

    import torch
    import transformers
    from qwen_tts import Qwen3TTSModel

    torch.set_num_threads(max(1, os.cpu_count() or 1))
    index_path = OUT / "index.json"
    index: dict[str, Any] = json.loads(index_path.read_text()) if index_path.exists() else {"cases": {}}
    index.update(
        {
            "$comment": "Written by scripts/speech-traces/qwen3_tts_trace.py; docs/models/qwen3-tts.md.",
            "package": PACKAGE,
            "package_commit": PACKAGE_COMMIT,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": np.__version__,
            "device": "cpu",
            "dtype": "float32",
            "seed": SEED,
        }
    )
    for spec in [] if args.clone else CASES:
        if spec.name not in chosen:
            continue
        size = SIZES[spec.size]
        model_dir = MODELS / size["dir"]
        sidecar = json.loads((model_dir / "model.safetensors.nuclis.json").read_text())
        codec_sha = json.loads((model_dir / "speech_tokenizer/model.safetensors.nuclis.json").read_text())["sha256"]
        meta: dict[str, Any] = {
            "repo": size["repo"],
            "revision": sidecar["revision"],
            "codec_sha256": codec_sha,
            "speaker": spec.speaker,
            "language": spec.language,
            "text": SENTENCES[spec.sentence],
        }
        if spec.instruct:
            meta["instruct"] = INSTRUCTIONS[spec.instruct]
        case = Case(spec.name, meta, args.debug_dir)
        tts = Qwen3TTSModel.from_pretrained(str(model_dir), dtype=torch.float32, device_map="cpu")
        tts.model.eval()
        talker = TalkerTrace(tts, spec.talker) if spec.talker and "talker" in sections else None
        if talker:
            meta["model_sha256"] = sidecar["sha256"]
        with torch.inference_mode():
            codes = section_codes(tts, case, spec, talker)
            if talker:
                talker.write(case, spec)
            if "codec" in sections and spec.codec_rows:
                section_codec(tts, case, spec, codes)
        index["cases"][spec.name] = {**meta, "files": case.files}
        print(f"{spec.name}: {meta.get('frames_generated')} frames; {len(case.files)} files")
        del tts
    if args.clone:
        run_clone(index, args.debug_dir)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=1) + "\n")


if __name__ == "__main__":
    main()
