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
- `codec`: the authors' codec decoder on the case's frames (`Spec.codec_frames`),
  hooked at each stage boundary; each stage pinned keeps its last rows
  (`Spec.codec_rows`). The `fox` case pins every stage of its first 24 frames; the
  `window` case decodes 112 frames, past the transformer's 72-frame window, and pins
  the last rows only.

Run from the repository root (the venv: docs/models/qwen3-tts.md):

    .reference/venv-qwen3-tts/bin/python scripts/speech-traces/qwen3_tts_trace.py

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


@dataclass(frozen=True)
class Spec:
    size: str
    speaker: str
    language: str
    sentence: str
    # Frames the codec decodes from the start (None: all); a prefix decodes as itself.
    codec_frames: int | None
    codec_rows: dict[str, int | None]

    @property
    def name(self) -> str:
        return f"{self.size}-{self.speaker}-{self.sentence}"


CASES = [
    Spec("0.6b", "ryan", "english", "fox", 24, EVERY_STAGE),
    # 112 frames: 40 past the window, ending in speech (the clip's tail is silence).
    Spec("0.6b", "ryan", "english", "window", 112, PAST_THE_WINDOW),
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


def section_codes(tts: Any, case: Case, speaker: str, language: str, text: str) -> np.ndarray:
    """Samples the sentence's codes with the authors' defaults; returns [frames, 16]."""
    import torch

    tokenizer = tts.model.speech_tokenizer
    captured: list[Any] = []
    decode = tokenizer.decode

    def capture(encoded: Any) -> Any:
        captured.append(encoded[0]["audio_codes"])
        return decode(encoded)

    tokenizer.decode = capture
    torch.manual_seed(SEED)
    try:
        wavs, rate = tts.generate_custom_voice(text=text, speaker=speaker, language=language)
    finally:
        tokenizer.decode = decode
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


SECTIONS = {"codes", "codec"}


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--sections", default="codes,codec", help="comma-separated, of: " + ", ".join(sorted(SECTIONS)))
    parser.add_argument("--debug-dir", type=pathlib.Path, help="also write every stage whole here (unpinned)")
    args = parser.parse_args()
    sections = set(args.sections.split(","))
    if not sections <= SECTIONS or "codes" not in sections:
        raise SystemExit(f"--sections must include codes and be drawn from {sorted(SECTIONS)}")

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
    for spec in CASES:
        size = SIZES[spec.size]
        model_dir = MODELS / size["dir"]
        revision = json.loads((model_dir / "model.safetensors.nuclis.json").read_text())["revision"]
        codec_sha = json.loads((model_dir / "speech_tokenizer/model.safetensors.nuclis.json").read_text())["sha256"]
        meta: dict[str, Any] = {
            "repo": size["repo"],
            "revision": revision,
            "codec_sha256": codec_sha,
            "speaker": spec.speaker,
            "language": spec.language,
            "text": SENTENCES[spec.sentence],
        }
        case = Case(spec.name, meta, args.debug_dir)
        tts = Qwen3TTSModel.from_pretrained(str(model_dir), dtype=torch.float32, device_map="cpu")
        tts.model.eval()
        with torch.inference_mode():
            codes = section_codes(tts, case, spec.speaker, spec.language, SENTENCES[spec.sentence])
            if "codec" in sections:
                section_codec(tts, case, spec, codes)
        index["cases"][spec.name] = {**meta, "files": case.files}
        print(f"{spec.name}: {meta.get('frames_generated')} frames; {len(case.files)} files")
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=1) + "\n")


if __name__ == "__main__":
    main()
