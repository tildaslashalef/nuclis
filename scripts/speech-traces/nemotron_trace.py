#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Pin Nemotron 3.5 ASR's bring-up trace from the authors' code.

Runs `Nemotron3_5AsrForRNNT` (transformers at c0d024fb, float32, CPU) on two
public clips cut from the speech corpus (`make speech-corpus`), one
LibriSpeech test-clean and one FLEURS German, and writes under
`tests/fixtures/nemotron-asr/`:

- each cut clip as 16-bit mono WAV, the input of everything below;
- the whole-clip log-mel features (all frames, the zeroed last one included);
- the subsampling's rows, and the rows after blocks 0, 11 and 23 and after
  the prompt block, at lookaheads 0 and 3: once over the whole clip, once
  streamed chunk by chunk as the authors' streaming path runs it (the
  processor's chunked features, the encoder's caches);
- `index.json`: every array's file, shape, dtype, SHA-256 and row indices,
  the clips' sources and licences, the versions, and the authors' own
  stream-versus-whole differences.

Each clip is cut to N = 25 + 32m mel frames, so both lookaheads end on a whole
chunk and the stream equals the whole clip; rows are pinned at their first
and last eight to keep the fixture small. More stages (the joint's logits,
the token ids) join by returning more arrays from `whole` and `streamed`.
Run once, in the venv of `.reference/` (docs/models/nemotron-asr.md § Front
end and encoder against the trace):

    .reference/venv-transformers/bin/python scripts/speech-traces/nemotron_trace.py
"""

from __future__ import annotations

import hashlib
import json
import pathlib
import wave
from dataclasses import dataclass
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
SPEECH = ROOT / ".reference/speech"
CORPUS = ROOT / "tests/fixtures/speech-corpus.json"
OUT = ROOT / "tests/fixtures/nemotron-asr"
MODEL = pathlib.Path.home() / ".nuclis/models/nvidia/nemotron-3.5-asr-streaming-0.6b"
MODEL_REPOSITORY = "nvidia/nemotron-3.5-asr-streaming-0.6b"
MODEL_REVISION = "ea30d66debe3740a08b573244286791d423d6b3e"
MODEL_SHA256 = "9eebdd6590289cb3030f310858f3df93256600a800a3e8200c5993d5f967e174"
TRANSFORMERS_COMMIT = "c0d024fbc873ab0325e1b40fc1bd69069609de65"

HOP = 160
LOOKAHEADS = (0, 3)
WHOLE_STAGES = ("block.0", "block.11", "block.23", "prompt")
STREAM_STAGES = ("block.0", "block.23", "prompt")
EDGE_ROWS = 8


@dataclass
class Clip:
    name: str
    # The corpus source the clip is cut from: its first clip long enough.
    source: str
    # The processor's `language`: a `prompt_dictionary` key.
    language: str
    # Mel frames N = 25 + 32m: whole chunks at r = 0 and r = 3.
    frames: int


CLIPS = [
    Clip("librispeech", "librispeech-test-clean", "auto", 505),
    Clip("fleurs-de", "fleurs-de_de-test", "de-DE", 505),
]


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def encoder_rows(frames: int) -> int:
    """Rows the encoder gives for N valid mel frames, f(f(f(N))) with f(n) = n // 2 + 1."""
    for _ in range(3):
        frames = frames // 2 + 1
    return frames


def choose(corpus: dict[str, Any], source: str, samples: int) -> dict[str, Any]:
    """The first corpus clip from `source` long enough to cut."""
    for clip in corpus["clips"]:
        if clip["source"] == source and clip["samples"] >= samples:
            return clip
    raise SystemExit(f"no {source} clip of {samples} samples in the corpus")


def cut(speech: pathlib.Path, entry: dict[str, Any], samples: int, path: pathlib.Path) -> np.ndarray:
    """Writes the clip's first `samples` as 16-bit WAV and returns them as the decoder reads them back."""
    import soundfile

    source = speech / entry["file"]
    if sha256(source.read_bytes()) != entry["sha256"]:
        raise SystemExit(f"{source}: not the pinned clip; run make speech-corpus")
    audio, rate = soundfile.read(str(source), dtype="float32", always_2d=False)
    if rate != 16000 or audio.ndim != 1:
        raise SystemExit(f"{source}: {rate} Hz, {audio.ndim} dims")
    pcm = np.clip(np.round(audio[:samples] * 32768.0), -32768, 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(pcm.tobytes())
    return pcm.astype(np.float32) / 32768.0


class Recorder:
    """Forward hooks that collect each stage's output rows, chunk after chunk."""

    def __init__(self, model: Any) -> None:
        self.rows: dict[str, list[np.ndarray]] = {}
        encoder = model.encoder
        self.handles = [encoder.subsampling.register_forward_hook(self.hook("subsampling"))]
        for i in (0, 11, 23):
            self.handles.append(encoder.layers[i].register_forward_hook(self.hook(f"block.{i}")))
        self.handles.append(model.prompt_projector.register_forward_hook(self.hook("prompt")))

    def hook(self, name: str) -> Any:
        def record(_module: Any, _inputs: Any, output: Any) -> None:
            value = output[0] if isinstance(output, (tuple, list)) else output
            self.rows.setdefault(name, []).append(value.detach().float().numpy()[0])

        return record

    def take(self) -> dict[str, np.ndarray]:
        out = {name: np.concatenate(parts, axis=0) for name, parts in self.rows.items()}
        self.rows = {}
        return out

    def close(self) -> None:
        for h in self.handles:
            h.remove()


def whole(model: Any, processor: Any, audio: np.ndarray, language: str, r: int) -> dict[str, np.ndarray]:
    import torch

    inputs = processor(audio, sampling_rate=16000, language=language)
    recorder = Recorder(model)
    with torch.inference_mode():
        model.get_audio_features(
            input_features=inputs["input_features"],
            attention_mask=inputs["attention_mask"],
            prompt_ids=inputs["prompt_ids"],
            num_lookahead_tokens=r,
        )
    stages = recorder.take()
    recorder.close()
    stages["features"] = inputs["input_features"][0].numpy()
    stages["valid_frames"] = np.array([int(inputs["attention_mask"].sum())])
    return stages


def streamed(model: Any, processor: Any, audio: np.ndarray, language: str, r: int) -> dict[str, np.ndarray]:
    """The authors' streaming path: the processor's chunk features (`center=True` first, then
    `center=False` with 400 samples of overlap), each chunk through the encoder with its caches."""
    import torch

    extractor = processor.feature_extractor
    prompt_ids = processor(audio[:4000], sampling_rate=16000, language=language)["prompt_ids"]
    first = 1 + 8 * r
    later = 8 * (r + 1)
    frames = len(audio) // HOP
    recorder = Recorder(model)
    pieces: list[np.ndarray] = []
    past = None
    padding = None
    done = 0
    while done < frames:
        if done == 0:
            n = first
            chunk = audio[: (n - 1) * HOP + extractor.win_length // 2]
            features = extractor(chunk, sampling_rate=16000, center=True)["input_features"][:, :n]
        else:
            n = later
            start = HOP * done - extractor.n_fft // 2
            end = start + n * HOP + extractor.win_length
            last = end > len(audio)
            # At r = 0 the second chunk starts 96 samples before the clip: the
            # zeros a centered pass reads there.
            piece = np.concatenate([np.zeros(max(0, -start), np.float32), audio[max(0, start) : end]])
            features = extractor(piece, sampling_rate=16000, center=False, is_last_audio_chunk=last)
            features = features["input_features"][:, :n]
        if features.shape[1] != n:
            raise SystemExit(f"chunk at frame {done}: {features.shape[1]} frames, wanted {n}")
        pieces.append(features[0].numpy())
        with torch.inference_mode():
            out = model.get_audio_features(
                input_features=features,
                prompt_ids=prompt_ids,
                past_key_values=past,
                padding_cache=padding,
                use_cache=True,
                num_lookahead_tokens=r,
            )
        past, padding = out.past_key_values, out.padding_cache
        done += n
    if done != frames:
        raise SystemExit(f"{frames} frames are not whole chunks at r = {r}")
    stages = recorder.take()
    recorder.close()
    stages["features"] = np.concatenate(pieces, axis=0)
    return stages


class Pin:
    def __init__(self, directory: pathlib.Path) -> None:
        self.directory = directory
        self.arrays: list[dict[str, Any]] = []

    def put(self, name: str, values: np.ndarray, rows: list[int] | None = None) -> None:
        if rows is not None:
            values = values[rows]
        data = np.ascontiguousarray(values.astype("<f4") if values.dtype.kind == "f" else values.astype("<i4"))
        path = self.directory / f"{name}.{'f32' if data.dtype.kind == 'f' else 'i32'}"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data.tobytes())
        entry: dict[str, Any] = {
            "name": name,
            "file": str(path.relative_to(OUT)),
            "shape": list(data.shape),
            "dtype": "f32" if data.dtype.kind == "f" else "i32",
            "sha256": sha256(data.tobytes()),
        }
        if rows is not None:
            entry["rows"] = rows
        self.arrays.append(entry)


def main() -> None:
    import argparse

    import librosa
    import torch
    import transformers
    from transformers import AutoProcessor, Nemotron3_5AsrForRNNT

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--speech", type=pathlib.Path, default=SPEECH, help="the fetched corpus (.reference/speech)")
    args = parser.parse_args()
    torch.manual_seed(0)
    corpus = json.loads(CORPUS.read_text())
    sources = {s["id"]: s for s in corpus["sources"]}
    processor = AutoProcessor.from_pretrained(str(MODEL))
    model = Nemotron3_5AsrForRNNT.from_pretrained(str(MODEL), dtype=torch.float32).eval()
    OUT.mkdir(parents=True, exist_ok=True)
    index: dict[str, Any] = {
        "$comment": "Nemotron 3.5 ASR bring-up trace: scripts/speech-traces/nemotron_trace.py. Arrays are "
        "little-endian, row-major; `rows` names the encoder rows pinned (of the clip's valid rows).",
        "model": {"repository": MODEL_REPOSITORY, "revision": MODEL_REVISION, "sha256": MODEL_SHA256},
        "transformers": {"version": transformers.__version__, "commit": TRANSFORMERS_COMMIT},
        "torch": torch.__version__,
        "librosa": librosa.__version__,
        "numpy": np.__version__,
        "clips": [],
    }
    for clip in CLIPS:
        samples = clip.frames * HOP + HOP // 2
        entry = choose(corpus, clip.source, samples)
        pin = Pin(OUT / clip.name)
        wav = OUT / clip.name / "clip.wav"
        wav.parent.mkdir(parents=True, exist_ok=True)
        audio = cut(args.speech, entry, samples, wav)
        rows = encoder_rows(clip.frames)
        selected = list(range(EDGE_ROWS)) + list(range(rows - EDGE_ROWS, rows))
        consistency: dict[str, Any] = {}
        record: dict[str, Any] = {
            "name": clip.name,
            "wav": str(wav.relative_to(OUT)),
            "wav_sha256": sha256(wav.read_bytes()),
            "samples": samples,
            "frames": clip.frames,
            "rows": rows,
            "language": clip.language,
            "prompt_id": processor.prompt_dictionary[clip.language],
            "source": {
                "corpus_id": entry["id"],
                "url": sources[entry["source"]]["url"],
                "member": entry["member"],
                "sha256": entry["sha256"],
                "license": sources[entry["source"]]["license"],
                "attribution": sources[entry["source"]]["attribution"],
                "cut": f"the first {samples} samples, rounded to 16-bit PCM",
            },
        }
        for r in LOOKAHEADS:
            full = whole(model, processor, audio, clip.language, r)
            if int(full["valid_frames"][0]) != clip.frames:
                raise SystemExit(f"{clip.name}: {full['valid_frames'][0]} valid frames, wanted {clip.frames}")
            stream = streamed(model, processor, audio, clip.language, r)
            if r == LOOKAHEADS[0]:
                pin.put("features", full["features"])
                pin.put("subsampling", full["subsampling"][:rows], selected)
            for stage in WHOLE_STAGES:
                pin.put(f"r{r}/whole/{stage}", full[stage][:rows], selected)
            for stage in STREAM_STAGES:
                pin.put(f"r{r}/stream/{stage}", stream[stage], selected)
            gaps = {"features": float(np.abs(stream["features"] - full["features"][: clip.frames]).max())}
            for stage in ("subsampling", *WHOLE_STAGES):
                gaps[stage] = float(np.abs(stream[stage] - full[stage][:rows]).max())
            consistency[f"r{r}"] = gaps
            print(clip.name, f"r={r}", {k: f"{v:.2e}" for k, v in gaps.items()})
        record["stream_minus_whole_max_abs"] = consistency
        record["arrays"] = pin.arrays
        index["clips"].append(record)
    (OUT / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    total = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())
    print(f"{total} bytes under {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
