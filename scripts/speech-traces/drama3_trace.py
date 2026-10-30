#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Pin one-time bring-up traces of Fish Audio Drama 3 (docs/models/drama3.md § The traces).

Runs Fish's own modules (fishaudio/fish-speech at `0f8a12e8`, unpacked at
`.reference/fish-speech-0f8a12e/`) on the CPU in F32, on weights dequantized from the
GGUF nuclis runs (`sakasegawa/drama-3-GGUF` `Drama-35B-A3B-v3-Q4_K.gguf`), and writes
small little-endian files under `tests/fixtures/drama3/<case>/` with an `index.json`
naming each file's shape, dtype, rows, and SHA-256, the GGUF digest, the package commit,
the library versions, the GGUF-to-Fish name map, and the layout checks. Every array is
row-major, time first: a row is one sample, frame, or prompt row of all channels.

Cases:

- `tokenizer`: a `tokenizers.Tokenizer` built from the GGUF's `drama3.tokenizer.*` alone
  (NFC, the regex split, byte level, BPE, the added ids), and the ids of a dozen strings.
- `codec-reference`: the 24 kHz clip pinned for Qwen3-TTS
  (`tests/fixtures/qwen3-tts/0.6b-clone-reference/reference.wav`), resampled as Fish's
  `encode_audio` does; Fish's DAC encoder (`DAC.encode`) at each stage boundary, the
  codes whole; then `DAC.from_indices` on those codes at each stage boundary and the
  samples. Before that, every ambiguous codec layout (the split stride convolutions,
  the depthwise kernels, the transposed convolutions and the upsample order) is decided
  by reconstruction quality, each candidate recorded.
- `codec-window`: the clip twice over, past the encoder transformer's 512-row and the
  quantizer transformers' 128-frame windows; the late stages and codes only.
- `clone-fox`: the `reference-bracket-v1` prompt Fish's `build_drama3_prompt` makes with
  the clip as Reference 1, then 24 continuation frames (the clip's first frames)
  teacher-forced; the backbone layer-streamed (one Fish `DecoderLayer` at a time, every
  expert dequantized) under each candidate convention of the norm shift, the codebook
  rows, the DeltaNet value-head order, and the query/gate split, decided by the
  likelihood of the clip's own codes; the winner's rows and logits are pinned. Then
  Fish's tower `prefill` over the same rows, its conventions decided the same way.

Run from the repository root (a throwaway venv; docs/models/drama3.md § The traces):

    .reference/venv-drama3/bin/python scripts/speech-traces/drama3_trace.py [--cases NAME,...]

`--cases` re-runs only the named cases (`clone-fox` reads `codec-reference`'s codes)
and keeps the others' index entries. `--debug-dir DIR` also writes every hooked stage
whole, unpinned, for localizing.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import mmap
import os
import pathlib
import struct
import sys
import time
import types
import unicodedata
import wave
from dataclasses import dataclass
from typing import Any

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
OUT = ROOT / "tests/fixtures/drama3"
FISH = ROOT / ".reference/fish-speech-0f8a12e"
PACKAGE = "fishaudio/fish-speech"
PACKAGE_COMMIT = "0f8a12e8a651062fdee9c33a894bdccd3a8513f1"
GGUF = pathlib.Path(os.path.expanduser("~/.nuclis/models/sakasegawa/drama-3-GGUF/Drama-35B-A3B-v3-Q4_K.gguf"))
GGUF_REPO = "sakasegawa/drama-3-GGUF"
GGUF_SHA256 = "2a57b14c9b208a2ff833ab7305b920ca8f154ad807004a52d96a86a90a55523e"
REFERENCE_WAV = ROOT / "tests/fixtures/qwen3-tts/0.6b-clone-reference/reference.wav"
# The clip's transcript (scripts/speech-traces/qwen3_tts_trace.py `REFERENCE`).
REFERENCE_TEXT = "Every morning I walk along the river and listen to the birds."
TEXT = "The quick brown fox jumps over the lazy dog."
CONTINUATION_FRAMES = 24
SAMPLES_PER_FRAME = 2048
# Backbone layers whose output (the residual stream) is pinned.
BACKBONE_LAYERS = (0, 3, 19, 39)
TOKENIZER_STRINGS = [
    "The quick brown fox jumps over the lazy dog.",
    "إزيك يا باشا؟ عامل إيه النهارده؟",
    "今天天气很好，我们去公园散步吧。",
    "Привет, как дела?",
    "It costs $1,234.56 on 2026-10-30 at 7:45pm, about 3x more.",
    "[sighs] I can't believe it. [laughs] Really?",
    "<whisper>Don't tell anyone.</whisper>",
    "<|speaker:1|>Hello there.<|speaker:0|> Hi!",
    "First line.\nSecond line.\n\n  Indented third.\r\n",
    "Café naïve — “quoted” résumé",
    "Hello,   world!  \t tabs   ",
    "<|im_start|>user\n<|speaker:0|> 你好<|im_end|>\n",
]

# --------------------------------------------------------------------------------------
# The GGUF, read with numpy (Q4_K, F16, F32).

_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
Q4_K, F16, F32 = 12, 1, 0


class Gguf:
    """Metadata and tensors of a GGUF; `get` returns numpy with the dims reversed (ne0 last)."""

    def __init__(self, path: pathlib.Path):
        self.file = path.open("rb")
        self.mm = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)
        assert self.mm[:4] == b"GGUF"
        self.pos = 4
        _version, tensors, keys = self._read("I"), self._read("Q"), self._read("Q")
        self.meta: dict[str, Any] = {}
        for _ in range(keys):
            key = self._string()
            self.meta[key] = self._value(self._read("I"))
        self.tensors: dict[str, tuple[list[int], int, int]] = {}
        for _ in range(tensors):
            name = self._string()
            dims = [self._read("Q") for _ in range(self._read("I"))]
            kind, offset = self._read("I"), self._read("Q")
            self.tensors[name] = (dims, kind, offset)
        align = self.meta.get("general.alignment", 32)
        self.base = (self.pos + align - 1) // align * align
        self.read: set[str] = set()

    def _read(self, fmt: str) -> Any:
        value = struct.unpack_from("<" + fmt, self.mm, self.pos)[0]
        self.pos += struct.calcsize(fmt)
        return value

    def _string(self) -> str:
        n = self._read("Q")
        value = self.mm[self.pos : self.pos + n].decode("utf-8")
        self.pos += n
        return value

    def _value(self, kind: int) -> Any:
        if kind == 8:
            return self._string()
        if kind == 9:
            inner, n = self._read("I"), self._read("Q")
            return [self._string() if inner == 8 else self._read(_SCALAR[inner]) for _ in range(n)]
        return self._read(_SCALAR[kind])

    def shape(self, name: str) -> list[int]:
        return list(reversed(self.tensors[name][0]))

    def get(self, name: str, rows: tuple[int, int] | None = None) -> np.ndarray:
        """F32 values shaped [ne_last, ..., ne0]; `rows` slices the flattened outer rows."""
        dims, kind, offset = self.tensors[name]
        self.read.add(name)
        ne0 = dims[0]
        outer = int(np.prod(dims[1:])) if len(dims) > 1 else 1
        r0, r1 = (0, outer) if rows is None else rows
        start = self.base + offset
        if kind == F32:
            a = np.frombuffer(self.mm, np.float32, (r1 - r0) * ne0, start + r0 * ne0 * 4).copy()
        elif kind == F16:
            a = np.frombuffer(self.mm, np.float16, (r1 - r0) * ne0, start + r0 * ne0 * 2).astype(np.float32)
        elif kind == Q4_K:
            row_bytes = ne0 // 256 * 144
            a = dequantize_q4_k(np.frombuffer(self.mm, np.uint8, (r1 - r0) * row_bytes, start + r0 * row_bytes))
        else:
            raise ValueError(f"{name}: encoding {kind}")
        if rows is not None or len(dims) == 1:
            return a.reshape(r1 - r0, ne0) if len(dims) > 1 else a
        return a.reshape(list(reversed(dims)))


def dequantize_q4_k(raw: np.ndarray) -> np.ndarray:
    """ggml's Q4_K: 256 values per 144-byte block, eight 32-value groups with 6-bit scales and mins."""
    b = raw.reshape(-1, 144)
    d = b[:, 0:2].copy().view(np.float16).astype(np.float32)
    dmin = b[:, 2:4].copy().view(np.float16).astype(np.float32)
    sc, qs = b[:, 4:16], b[:, 16:]
    scales = np.zeros((len(b), 8), np.float32)
    mins = np.zeros((len(b), 8), np.float32)
    for j in range(8):
        if j < 4:
            s, m = sc[:, j] & 63, sc[:, j + 4] & 63
        else:
            s = (sc[:, j + 4] & 0xF) | ((sc[:, j - 4] >> 6) << 4)
            m = (sc[:, j + 4] >> 4) | ((sc[:, j] >> 6) << 4)
        scales[:, j], mins[:, j] = s, m
    out = np.empty((len(b), 256), np.float32)
    for j in range(4):
        q = qs[:, 32 * j : 32 * j + 32]
        out[:, 64 * j : 64 * j + 32] = d * scales[:, 2 * j : 2 * j + 1] * (q & 0xF) - dmin * mins[:, 2 * j : 2 * j + 1]
        out[:, 64 * j + 32 : 64 * j + 64] = (
            d * scales[:, 2 * j + 1 : 2 * j + 2] * (q >> 4) - dmin * mins[:, 2 * j + 1 : 2 * j + 2]
        )
    return out.reshape(-1)


# --------------------------------------------------------------------------------------
# Fixture files.


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_array(array: np.ndarray, dtype: str = "<f4") -> str:
    return hashlib.sha256(np.ascontiguousarray(array.astype(dtype)).tobytes()).hexdigest()


def edge_rows(total: int, count: int) -> list[list[int]]:
    """The first and last `count` rows as [start, end) ranges (one range when they meet)."""
    if 2 * count >= total:
        return [[0, total]]
    return [[0, count], [total - count, total]]


class Case:
    """One case's directory and its index entries."""

    def __init__(self, name: str, debug: pathlib.Path | None):
        self.name = name
        self.dir = OUT / name
        self.meta: dict[str, Any] = {}
        self.files: dict[str, dict[str, Any]] = {}
        self.debug = debug / name if debug else None

    def add(self, key: str, array: np.ndarray, dtype: str, note: str, rows: list[list[int]] | None = None) -> None:
        """Writes `array` as `<key>.<dtype>` (`f32` or `i32`, little endian); `rows` are [start, end) ranges."""
        np_dtype = {"f32": "<f4", "i32": "<i4"}[dtype]
        data = np.ascontiguousarray(array.astype(np_dtype))
        if rows is not None:
            assert data.shape[0] == sum(e - s for s, e in rows), (key, data.shape, rows)
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
            entry["rows"] = rows
        self.files[key] = entry

    def add_edges(self, key: str, whole: np.ndarray, count: int, note: str) -> None:
        """Pins the first and last `count` rows of a time-first stage."""
        ranges = edge_rows(whole.shape[0], count)
        self.dump(key, whole)
        self.add(key, np.concatenate([whole[s:e] for s, e in ranges]), "f32", note, ranges)

    def dump(self, key: str, array: np.ndarray) -> None:
        if self.debug is None:
            return
        path = self.debug / f"{key}.f32"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.ascontiguousarray(array.astype("<f4")).tofile(path)

    def entry(self) -> dict[str, Any]:
        return {**self.meta, "files": self.files}


# --------------------------------------------------------------------------------------
# Fish's modules, imported from the unpacked tree.


def fish_imports() -> None:
    """Puts Fish's tree on the path; stubs `audiotools`, which only the DAC's unused helpers need.

    `dac` (descript-audio-codec, installed with --no-deps) is entered below its package
    `__init__`, which imports the training losses: Fish's DAC reads only `dac.nn.layers`
    (Snake1d), `dac.nn.quantize` (the residual VQ), and `dac.model.base` (CodecMixin).
    """
    import torch

    sys.path.insert(0, str(FISH))
    audiotools = types.ModuleType("audiotools")
    ml = types.ModuleType("audiotools.ml")
    ml.BaseModel = torch.nn.Module  # pyright: ignore[reportAttributeAccessIssue]
    audiotools.AudioSignal = object  # pyright: ignore[reportAttributeAccessIssue]
    audiotools.ml = ml  # pyright: ignore[reportAttributeAccessIssue]
    sys.modules["audiotools"] = audiotools
    sys.modules["audiotools.ml"] = ml
    spec = importlib.util.find_spec("dac")
    assert spec is not None and spec.submodule_search_locations, "descript-audio-codec is not installed"
    location = pathlib.Path(next(iter(spec.submodule_search_locations)))
    for name, sub in (("dac", ""), ("dac.nn", "nn"), ("dac.model", "model")):
        package = types.ModuleType(name)
        package.__path__ = [str(location / sub)]  # pyright: ignore[reportAttributeAccessIssue]
        sys.modules[name] = package


def fish_config(g: Gguf) -> Any:
    """Fish's FishOmniConfig from the GGUF's metadata (the checkpoint's config.json is not on disk)."""
    from fish_speech.models.text2semantic.fish_omni.config import (
        AudioDecoderConfig,
        FishOmniConfig,
        FishOmniTextConfig,
    )

    m = g.meta
    t = "drama3.text."
    a = "drama3.audio_decoder."
    text = FishOmniTextConfig(
        vocab_size=m[t + "vocab_size"],
        hidden_size=m[t + "hidden_size"],
        num_hidden_layers=m[t + "num_hidden_layers"],
        layer_types=list(m[t + "layer_types"]),
        num_attention_heads=m[t + "num_attention_heads"],
        num_key_value_heads=m[t + "num_key_value_heads"],
        head_dim=m[t + "head_dim"],
        rope_theta=m[t + "rope_theta"],
        partial_rotary_factor=m[t + "rotary_dim"] / m[t + "head_dim"],
        rms_norm_eps=m[t + "rms_norm_eps"],
        max_position_embeddings=m[t + "max_position_embeddings"],
        linear_conv_kernel_dim=m[t + "linear_conv_kernel_dim"],
        linear_key_head_dim=m[t + "linear_key_head_dim"],
        linear_value_head_dim=m[t + "linear_value_head_dim"],
        linear_num_key_heads=m[t + "linear_num_key_heads"],
        linear_num_value_heads=m[t + "linear_num_value_heads"],
        num_experts=m[t + "num_experts"],
        num_experts_per_tok=m[t + "num_experts_per_tok"],
        moe_intermediate_size=m[t + "moe_intermediate_size"],
        shared_expert_intermediate_size=m[t + "shared_expert_intermediate_size"],
        norm_topk_prob=m[t + "norm_topk_prob"],
    )
    audio = AudioDecoderConfig(
        dim=m[a + "dim"],
        intermediate_size=m[a + "intermediate_size"],
        n_head=m[a + "n_head"],
        n_local_heads=m[a + "n_local_heads"],
        head_dim=m[a + "head_dim"],
        norm_eps=m[a + "norm_eps"],
        rope_base=m[a + "rope_base"],
        rotary_percent=m[a + "rotary_dim"] / m[a + "head_dim"],
        vocab_size=m[a + "vocab_size"],
        num_codebooks=m[a + "num_codebooks"],
        shared_enrichment_layers=m[a + "shared_enrichment_layers"],
        vq_backbone_codebooks=m[a + "vq_backbone_codebooks"],
    )
    return FishOmniConfig(
        text=text,
        audio_decoder=audio,
        semantic_start_token_id=m["drama3.semantic_start_token_id"],
        semantic_end_token_id=m["drama3.semantic_end_token_id"],
        eos_token_id=m["drama3.eos_token_id"],
    )


# --------------------------------------------------------------------------------------
# The tokenizer.


def build_tokenizer(g: Gguf) -> Any:
    """A byte-level BPE tokenizer from `drama3.tokenizer.*` alone.

    The GGUF carries the tokens, merges, added ids and split regex; the rest is Qwen3.5's
    tokenizer.json shape (NFC normalizer, the split isolated, byte level without its own
    regex, BPE without dropout, affixes or ignore_merges, no post-processor).
    """
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, models, normalizers, pre_tokenizers

    tokens: list[str] = g.meta["drama3.tokenizer.tokens"]
    merges = [tuple(m.split(" ", 1)) for m in g.meta["drama3.tokenizer.merges"]]
    vocab = {t: i for i, t in enumerate(tokens)}
    assert len(vocab) == len(tokens), "duplicate tokens"
    tok = Tokenizer(
        models.BPE(
            vocab=vocab,
            merges=merges,  # pyright: ignore[reportArgumentType]
            dropout=None,
            unk_token=None,
            continuing_subword_prefix="",
            end_of_word_suffix="",
            fuse_unk=False,
            byte_fallback=False,
        )
    )
    tok.normalizer = normalizers.NFC()  # pyright: ignore[reportAttributeAccessIssue]
    tok.pre_tokenizer = pre_tokenizers.Sequence(  # pyright: ignore[reportAttributeAccessIssue]
        [
            pre_tokenizers.Split(Regex(g.meta["drama3.tokenizer.pre_tokenizer"]), behavior="isolated", invert=False),
            pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=False, use_regex=False),
        ]
    )
    tok.decoder = decoders.ByteLevel()  # pyright: ignore[reportAttributeAccessIssue]
    tok.add_special_tokens(
        [AddedToken(tokens[i], special=True, normalized=False) for i in g.meta["drama3.tokenizer.added_ids"]]
    )
    for i in g.meta["drama3.tokenizer.added_ids"]:
        assert tok.token_to_id(tokens[i]) == i
    return tok


def section_tokenizer(g: Gguf, tok: Any, case: Case) -> None:
    pinned = []
    for text in TOKENIZER_STRINGS:
        enc = tok.encode(text)
        assert tok.decode(enc.ids, skip_special_tokens=False) == unicodedata.normalize("NFC", text)
        pinned.append({"text": text, "ids": list(enc.ids), "tokens": list(enc.tokens)})
    added = g.meta["drama3.tokenizer.added_ids"]
    case.meta.update(
        {
            "source": "the GGUF's drama3.tokenizer.* only (no tokenizer.json on disk)",
            "shape": "NFC normalizer; Split(drama3.tokenizer.pre_tokenizer, isolated); ByteLevel(add_prefix_space=false, "
            "use_regex=false); BPE (no dropout, no affixes, ignore_merges=false); every drama3.tokenizer.added_ids token "
            "matched whole before normalization; no BOS or post-processor (Fish encodes each literal with "
            "Tokenizer.encode, add_special_tokens=True, which adds nothing here)",
            "vocab": len(g.meta["drama3.tokenizer.tokens"]),
            "merges": len(g.meta["drama3.tokenizer.merges"]),
            "added_ids": len(added),
            "strings": pinned,
        }
    )


# --------------------------------------------------------------------------------------
# The codec.


@dataclass(frozen=True)
class CodecLayout:
    """The GGUF codec layouts its names and shapes leave open."""

    # Rows of a stride convolution stored [out][taps * in]: tap-major (k * in + i) or channel-major (i * taps + k).
    stride_conv: str = "tap-major"
    # The ConvNeXt depthwise kernel stored [7][C] ("k-major") or [C][7] read as [7][C] ("c-major").
    depthwise: str = "k-major"
    # The quantizer's transposed convolutions stored [k][out][in] (as every other convolution) or [k][in][out].
    up_transpose: str = "k,out,in"
    # GGUF `q.up.N` is Fish's `upsample.N` (applied N-th), or `upsample.(1 - N)`.
    up_order: str = "applied"

    def name(self) -> str:
        return f"{self.stride_conv}/{self.depthwise}/{self.up_transpose}/{self.up_order}"


def conv_weight(a: np.ndarray) -> np.ndarray:
    """GGUF [k][out][in] (ne0 = in) to torch Conv1d [out][in][k]."""
    return a.transpose(1, 2, 0)


def stride_weight(a: np.ndarray, taps: int, layout: CodecLayout) -> np.ndarray:
    """GGUF [out][taps * in] to torch Conv1d [out][in][taps]."""
    out, width = a.shape
    if layout.stride_conv == "tap-major":
        return a.reshape(out, taps, width // taps).transpose(0, 2, 1)
    return a.reshape(out, width // taps, taps)


def codec_state(g: Gguf, codec: Any, layout: CodecLayout) -> tuple[dict[str, Any], dict[str, str]]:
    """Fish's DAC state dict from `codec.*`, and the map GGUF name -> Fish name(s)."""
    import torch

    state: dict[str, np.ndarray] = {}
    names: dict[str, str] = {}

    def put(fish: str, gguf: str, value: np.ndarray | None = None) -> None:
        state[fish] = g.get(gguf) if value is None else value
        names[gguf] = f"{names[gguf]} + {fish}" if gguf in names else fish

    def snake(fish: str, gguf: str) -> None:
        put(f"{fish}.alpha", f"{gguf}.alpha", g.get(f"{gguf}.alpha")[None, :, None])
        # inv_alpha = 1 / (alpha + 1e-9), Snake1d's own reciprocal; checked equal, not loaded.
        inv = g.get(f"{gguf}.inv_alpha")
        assert np.array_equal(inv, (1.0 / (g.get(f"{gguf}.alpha") + np.float32(1e-9))).astype(np.float32)), gguf
        names[f"{gguf}.inv_alpha"] = f"(1 / ({fish}.alpha + 1e-9), not a parameter)"

    def conv(fish: str, gguf: str) -> None:
        put(f"{fish}.weight", f"{gguf}.weight", conv_weight(g.get(f"{gguf}.weight")))
        put(f"{fish}.bias", f"{gguf}.bias")

    def residual_unit(fish: str, gguf: str) -> None:
        snake(f"{fish}.block.0", f"{gguf}.snake1")
        conv(f"{fish}.block.1.conv", f"{gguf}.conv1")
        snake(f"{fish}.block.2", f"{gguf}.snake2")
        conv(f"{fish}.block.3.conv", f"{gguf}.conv2")

    def transformer(fish: str, gguf: str, layers: int) -> None:
        for i in range(layers):
            f, q = f"{fish}.layers.{i}", f"{gguf}.blk.{i}"
            wqkv = np.concatenate([g.get(f"{q}.attn_{x}") for x in "qkv"])
            state[f"{f}.attention.wqkv.weight"] = wqkv
            for x in "qkv":
                names[f"{q}.attn_{x}"] = f"{f}.attention.wqkv.weight rows [q | k | v]"
            put(f"{f}.attention.wo.weight", f"{q}.attn_o")
            put(f"{f}.attention_norm.weight", f"{q}.attn_norm")
            put(f"{f}.attention_layer_scale.gamma", f"{q}.attn_scale")
            put(f"{f}.ffn_norm.weight", f"{q}.ffn_norm")
            put(f"{f}.feed_forward.w1.weight", f"{q}.ffn_gate")
            put(f"{f}.feed_forward.w3.weight", f"{q}.ffn_up")
            put(f"{f}.feed_forward.w2.weight", f"{q}.ffn_down")
            put(f"{f}.ffn_layer_scale.gamma", f"{q}.ffn_scale")
        put(f"{fish}.norm.weight", f"{gguf}.norm")

    def convnext(fish: str, gguf: str) -> None:
        dw = g.get(f"{gguf}.dwconv.weight")
        dw = dw.T if layout.depthwise == "k-major" else dw.reshape(dw.shape[1], dw.shape[0])
        put(f"{fish}.dwconv.conv.weight", f"{gguf}.dwconv.weight", dw[:, None, :])
        put(f"{fish}.dwconv.conv.bias", f"{gguf}.dwconv.bias")
        put(f"{fish}.norm.weight", f"{gguf}.norm.weight")
        put(f"{fish}.norm.bias", f"{gguf}.norm.bias")
        put(f"{fish}.pwconv1.weight", f"{gguf}.pw1.weight")
        put(f"{fish}.pwconv1.bias", f"{gguf}.pw1.bias")
        put(f"{fish}.pwconv2.weight", f"{gguf}.pw2.weight")
        put(f"{fish}.pwconv2.bias", f"{gguf}.pw2.bias")
        put(f"{fish}.gamma", f"{gguf}.gamma")

    def vq(fish: str, gguf: str) -> None:
        put(f"{fish}.in_proj.weight", f"{gguf}.in_proj.weight", g.get(f"{gguf}.in_proj.weight")[:, :, None])
        put(f"{fish}.in_proj.bias", f"{gguf}.in_proj.bias")
        put(f"{fish}.out_proj.weight", f"{gguf}.out_proj.weight", g.get(f"{gguf}.out_proj.weight")[:, :, None])
        put(f"{fish}.out_proj.bias", f"{gguf}.out_proj.bias")
        put(f"{fish}.codebook.weight", f"{gguf}.codebook")

    # Encoder: block.0 in_conv, block.1-4 EncoderBlocks, block.5 Snake, block.6 out_conv.
    conv("encoder.block.0.conv", "codec.enc.in_conv")
    for n, stride in enumerate(g.meta["drama3.codec.encoder_rates"]):
        f, q = f"encoder.block.{n + 1}.block", f"codec.enc.blk.{n}"
        for m in range(3):
            residual_unit(f"{f}.{m}", f"{q}.res.{m}")
        snake(f"{f}.3", f"{q}.snake")
        first = stride_weight(g.get(f"{q}.down.first"), stride, layout)
        second = stride_weight(g.get(f"{q}.down.second"), stride, layout)
        state[f"{f}.4.conv.weight"] = np.concatenate([first, second], axis=2)
        names[f"{q}.down.first"] = f"{f}.4.conv.weight taps 0..{stride - 1}"
        names[f"{q}.down.second"] = f"{f}.4.conv.weight taps {stride}..{2 * stride - 1}"
        put(f"{f}.4.conv.bias", f"{q}.down.bias")
    transformer("encoder.block.4.block.5", "codec.enc.blk.3.tf", g.meta["drama3.codec.encoder_transformer.layers"])
    snake("encoder.block.5", "codec.enc.out_snake")
    conv("encoder.block.6.conv", "codec.enc.out_conv")
    # Quantizer.
    factors = g.meta["drama3.codec.downsample_factors"]
    for n, factor in enumerate(factors):
        f, q = f"quantizer.downsample.{n}", f"codec.q.down.{n}"
        put(f"{f}.0.conv.weight", f"{q}.conv.weight", stride_weight(g.get(f"{q}.conv.weight"), factor, layout))
        put(f"{f}.0.conv.bias", f"{q}.conv.bias")
        convnext(f"{f}.1", f"{q}.convnext")
        index = n if layout.up_order == "applied" else len(factors) - 1 - n
        f, q = f"quantizer.upsample.{index}", f"codec.q.up.{n}"
        up = g.get(f"{q}.conv.weight")
        up = up.transpose(2, 1, 0) if layout.up_transpose == "k,out,in" else up.transpose(1, 2, 0)
        put(f"{f}.0.conv.weight", f"{q}.conv.weight", up)
        put(f"{f}.0.conv.bias", f"{q}.conv.bias")
        convnext(f"{f}.1", f"{q}.convnext")
    layers = g.meta["drama3.codec.quantizer_transformer.layers"]
    transformer("quantizer.pre_module", "codec.q.pre", layers)
    transformer("quantizer.post_module", "codec.q.post", layers)
    vq("quantizer.semantic_quantizer.quantizers.0", "codec.q.semantic")
    for n in range(g.meta["drama3.codec.residual_codebooks"]):
        vq(f"quantizer.quantizer.quantizers.{n}", f"codec.q.residual.{n}")
    # Decoder: model.0 in_conv, model.1-4 DecoderBlocks, model.5 Snake, model.6 out_conv, model.7 Tanh.
    conv("decoder.model.0.conv", "codec.dec.in_conv")
    for n in range(len(g.meta["drama3.codec.decoder_rates"])):
        f, q = f"decoder.model.{n + 1}.block", f"codec.dec.blk.{n}"
        snake(f"{f}.0", f"{q}.snake")
        # ConvTranspose1d [in][out][k] from GGUF [k][out][in].
        put(f"{f}.1.conv.weight", f"{q}.up.weight", g.get(f"{q}.up.weight").transpose(2, 1, 0))
        put(f"{f}.1.conv.bias", f"{q}.up.bias")
        for m in range(3):
            residual_unit(f"{f}.{m + 2}", f"{q}.res.{m}")
    snake("decoder.model.5", "codec.dec.out_snake")
    conv("decoder.model.6.conv", "codec.dec.out_conv")

    unused = {n for n in g.tensors if n.startswith("codec.")} - names.keys()
    assert not unused, f"codec tensors not mapped: {sorted(unused)[:8]}"
    params = dict(codec.named_parameters())
    assert params.keys() == state.keys(), (
        sorted(params.keys() - state.keys())[:8],
        sorted(state.keys() - params.keys())[:8],
    )
    tensors = {}
    for k, v in state.items():
        assert tuple(params[k].shape) == v.shape, (k, params[k].shape, v.shape)
        tensors[k] = torch.from_numpy(np.ascontiguousarray(v, dtype=np.float32))
    return tensors, names


def build_codec(g: Gguf, layout: CodecLayout) -> tuple[Any, dict[str, str]]:
    """Fish's DAC as `load_codec_model` makes it (the yaml through hydra, weight norm removed), F32."""
    import torch
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torch.nn.utils.parametrize import is_parametrized, remove_parametrizations

    torch.manual_seed(0)
    codec = instantiate(OmegaConf.load(str(FISH / "fish_speech/configs/modded_dac_vq.yaml")))
    assert codec.quantizer.pre_module is not codec.quantizer.post_module
    # The checkpoint's weight norm is folded in the GGUF: drop the parametrizations first,
    # leaving plain `weight` parameters, as load_codec_model ends.
    for module in list(codec.modules()):
        if is_parametrized(module, "weight"):
            remove_parametrizations(module, "weight", leave_parametrized=True)
        elif hasattr(module, "weight_g"):
            torch.nn.utils.remove_weight_norm(module)
    tensors, names = codec_state(g, codec, layout)
    missing, unexpected = codec.load_state_dict(tensors, strict=False)
    rebuilt = {
        f"{name}.{buffer}" if name else buffer
        for name, module in codec.named_modules()
        for buffer in module._non_persistent_buffers_set  # noqa: SLF001
    }
    assert not unexpected and set(missing) <= rebuilt, (missing, unexpected)
    return codec.eval().float(), names


def load_clip() -> tuple[np.ndarray, int]:
    """The 16-bit WAV as torchaudio.load reads it (int16 / 32768, channels averaged)."""
    with wave.open(str(REFERENCE_WAV)) as w:
        assert w.getsampwidth() == 2
        rate, channels = w.getframerate(), w.getnchannels()
        data = np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float32) / 32768.0
    return data.reshape(-1, channels).mean(axis=1), rate


def resample(samples: np.ndarray, rate: int, target: int) -> np.ndarray:
    """`encode_audio`'s torchaudio.functional.resample (sinc_interp_hann, width 6, rolloff 0.99)."""
    import torch
    import torchaudio

    return torchaudio.functional.resample(torch.from_numpy(samples)[None], rate, target)[0].numpy()


class Hooks:
    """Forward hooks keeping each named module's output (or input) as a time-first array."""

    def __init__(self, model: Any, spec: list[tuple[str, str, str]]):
        self.values: dict[str, np.ndarray] = {}
        self.handles = []
        for name, path, side in spec:
            module = model.get_submodule(path)

            def record(_m: Any, inputs: Any, output: Any, name: str = name, side: str = side) -> None:
                value = inputs[0] if side == "input" else output
                assert name not in self.values, f"{name} ran twice"
                v = value.detach().float().numpy()
                assert v.shape[0] == 1
                self.values[name] = v[0].T  # [1, C, T] -> [T, C]

            self.handles.append(module.register_forward_hook(record))

    def remove(self) -> None:
        for h in self.handles:
            h.remove()


ENCODER_HOOKS = [
    ("enc.in_conv", "encoder.block.0", "output"),
    ("enc.blk.0.res.0", "encoder.block.1.block.0", "output"),
    ("enc.blk.0", "encoder.block.1", "output"),
    ("enc.blk.1", "encoder.block.2", "output"),
    ("enc.blk.2", "encoder.block.3", "output"),
    ("enc.blk.3.down", "encoder.block.4.block.4", "output"),
    ("enc.blk.3", "encoder.block.4", "output"),
    ("enc.out", "encoder.block.6", "output"),
    ("q.down.0.conv", "quantizer.downsample.0.0", "output"),
    ("q.down.0", "quantizer.downsample.0", "output"),
    ("q.down.1", "quantizer.downsample.1", "output"),
    ("q.pre", "quantizer.pre_module", "output"),
]
DECODER_HOOKS = [
    ("q.z", "quantizer.post_module", "input"),
    ("q.post", "quantizer.post_module", "output"),
    ("q.up.0", "quantizer.upsample.0", "output"),
    ("q.up.1", "quantizer.upsample.1", "output"),
    ("dec.in_conv", "decoder.model.0", "output"),
    ("dec.blk.0.up", "decoder.model.1.block.1", "output"),
    ("dec.blk.0", "decoder.model.1", "output"),
    ("dec.blk.1", "decoder.model.2", "output"),
    ("dec.blk.2", "decoder.model.3", "output"),
    ("dec.blk.3", "decoder.model.4", "output"),
    ("dec.out_conv", "decoder.model.6", "output"),
]
# Rows pinned at each end of a stage, by the stage's rate.
EDGE = {
    "enc.in_conv": 32,
    "enc.blk.0.res.0": 32,
    "enc.blk.0": 32,
    "enc.blk.1": 16,
    "enc.blk.2": 16,
    "enc.blk.3.down": 8,
    "enc.blk.3": 8,
    "enc.out": 8,
    "q.down.0.conv": 8,
    "q.down.0": 8,
    "q.down.1": 8,
    "q.pre": 8,
    "q.z": 8,
    "q.post": 8,
    "q.up.0": 8,
    "q.up.1": 8,
    "dec.in_conv": 8,
    "dec.blk.0.up": 16,
    "dec.blk.0": 16,
    "dec.blk.1": 32,
    "dec.blk.2": 32,
    "dec.blk.3": 64,
    "dec.out_conv": 64,
}
STAGE_NOTES = {
    "enc.in_conv": "encoder.block.0, the causal 7-tap input conv (1 -> 64)",
    "enc.blk.0.res.0": "encoder.block.1.block.0, the first ResidualUnit (dilation 1) of encoder block 0",
    "enc.blk.0": "encoder block 0 (3 ResidualUnits, Snake, causal stride-2 conv 64 -> 128)",
    "enc.blk.1": "encoder block 1 (stride 4, 128 -> 256)",
    "enc.blk.2": "encoder block 2 (stride 8, 256 -> 512)",
    "enc.blk.3.down": "encoder block 3's stride-8 conv (512 -> 1024), before its transformer",
    "enc.blk.3": "encoder block 3 after its 4-layer causal windowed (512) transformer",
    "enc.out": "the encoder's output: Snake then the causal 3-tap conv (1024 -> 1024), the latent z",
    "q.down.0.conv": "quantizer.downsample.0's causal stride-2 conv",
    "q.down.0": "quantizer.downsample.0 after its ConvNeXt block",
    "q.down.1": "quantizer.downsample.1 (stride 2 + ConvNeXt): frame rate",
    "q.pre": "the quantizer's 8-layer causal windowed (128) pre-transformer: what the VQs read",
    "q.z": "the quantizer decode's sum of the 10 codebook vectors after their out_proj (post_module's input)",
    "q.post": "the 8-layer causal windowed (128) post-transformer",
    "q.up.0": "quantizer.upsample.0: causal transposed conv x2 + ConvNeXt",
    "q.up.1": "quantizer.upsample.1: causal transposed conv x2 + ConvNeXt; the decoder's input",
    "dec.in_conv": "decoder.model.0, the causal 7-tap conv (1024 -> 1536)",
    "dec.blk.0.up": "decoder block 0's Snake + causal transposed conv x8 (1536 -> 768)",
    "dec.blk.0": "decoder block 0 after its 3 ResidualUnits",
    "dec.blk.1": "decoder block 1 (x8, 768 -> 384)",
    "dec.blk.2": "decoder block 2 (x4, 384 -> 192)",
    "dec.blk.3": "decoder block 3 (x2, 192 -> 96)",
    "dec.out_conv": "Snake then the causal 7-tap conv (96 -> 1), before tanh",
}


def encode(codec: Any, samples: np.ndarray) -> tuple[Any, dict[str, np.ndarray]]:
    """`encode_audio` from the resampled samples: codes [10, frames] and the hooked stages."""
    import torch

    hooks = Hooks(codec, ENCODER_HOOKS)
    latents: list[np.ndarray] = []
    vqs = [codec.quantizer.semantic_quantizer.quantizers[0], *codec.quantizer.quantizer.quantizers]
    handles = [
        q.in_proj.register_forward_hook(lambda _m, _i, o: latents.append(o.detach().float().numpy()[0].T)) for q in vqs
    ]
    try:
        with torch.inference_mode():
            audio = torch.from_numpy(samples)[None, None]
            lengths = torch.tensor([len(samples)], dtype=torch.long)
            indices, feature_lengths = codec.encode(audio, lengths)
            codes = indices[0, :, : feature_lengths[0]].clone()
    finally:
        hooks.remove()
        for h in handles:
            h.remove()
    hooks.values["q.latents"] = np.concatenate(latents, axis=1)
    return codes, hooks.values


def decode(codec: Any, codes: Any) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """`decode_to_audio`: `from_indices` on [10, frames] codes; samples and the hooked stages."""
    import torch

    hooks = Hooks(codec, DECODER_HOOKS)
    try:
        with torch.inference_mode():
            audio = codec.from_indices(codes.clone()[None])
    finally:
        hooks.remove()
    return audio[0, 0].float().numpy(), hooks.values


def reconstruction(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    """Waveform SNR and mean log-magnitude STFT distance of y against x (y cut to len(x))."""
    import torch

    y = y[: len(x)]
    snr = 10 * math.log10(float(np.sum(x**2)) / max(float(np.sum((x - y) ** 2)), 1e-20))
    window = torch.hann_window(2048)

    def spec(v: np.ndarray) -> Any:
        return torch.stft(torch.from_numpy(v), 2048, 512, window=window, return_complex=True).abs()

    distance = float((torch.log(spec(x) + 1e-5) - torch.log(spec(y) + 1e-5)).abs().mean())
    return {"snr_db": round(snr, 3), "log_spectral_l1": round(distance, 4)}


def search_codec_layout(g: Gguf, samples: np.ndarray) -> tuple[CodecLayout, list[dict[str, Any]]]:
    """Every combination of the open layouts, scored by encode-then-decode reconstruction."""
    results = []
    for stride_conv in ("tap-major", "channel-major"):
        for depthwise in ("k-major", "c-major"):
            for up_transpose in ("k,out,in", "k,in,out"):
                for up_order in ("applied", "reversed"):
                    layout = CodecLayout(stride_conv, depthwise, up_transpose, up_order)
                    codec, _ = build_codec(g, layout)
                    codes, _ = encode(codec, samples)
                    audio, _ = decode(codec, codes)
                    score = reconstruction(samples, audio)
                    results.append({"layout": layout.name(), **score})
                    print(f"  codec layout {layout.name()}: {score}", flush=True)
                    del codec
    best = min(results, key=lambda r: r["log_spectral_l1"])
    parts = best["layout"].split("/")
    return CodecLayout(*parts), results


def pin_encoder(case: Case, samples: np.ndarray, codes: Any, stages: dict[str, np.ndarray], keys: list[str]) -> None:
    for key in keys:
        case.add_edges(f"encoder/{key}", stages[key], EDGE[key], f"{STAGE_NOTES[key]}; [rows, channels]")
    latents = stages["q.latents"]
    case.dump("encoder/q.latents", latents)
    case.add(
        "encoder/q.latents",
        latents,
        "f32",
        "each VQ's in_proj output z_e (the 8-dim latent before the L2-normalized nearest-codeword search), "
        "[frame][codebook * 8 + d], codebook 0 the semantic one, 1-9 the residual ones (each reads the residual "
        "left by the earlier ones)",
    )
    case.add(
        "codes",
        codes.numpy().T,
        "i32",
        "the codes DAC.encode returns, [frame][codebook]: 0 semantic (< 4096), 1-9 residual (< 1024)",
    )


def section_codec(g: Gguf, case: Case, window: Case | None, search: bool) -> tuple[Any, CodecLayout]:
    """The codec cases; returns the clip's codes [10, frames] and the layout used."""
    clip, rate = load_clip()
    sample_rate = g.meta["speech.sample_rate"]
    samples = resample(clip, rate, sample_rate)
    if search:
        layout, results = search_codec_layout(g, samples)
    else:
        layout, results = CodecLayout(), []
    codec, names = build_codec(g, layout)
    codes, stages = encode(codec, samples)
    audio, dec_stages = decode(codec, codes)
    frames = codes.shape[1]
    assert frames == math.ceil(len(samples) / SAMPLES_PER_FRAME) and len(audio) == frames * SAMPLES_PER_FRAME
    case.meta.update(
        {
            "clip": REFERENCE_WAV.relative_to(ROOT).as_posix(),
            "clip_sha256": sha256(REFERENCE_WAV),
            "clip_rate": rate,
            "clip_samples": len(clip),
            "transcript": REFERENCE_TEXT,
            "sample_rate": sample_rate,
            "samples": len(samples),
            "frames": frames,
            "padding": f"DAC.encode pads the samples on the right with zeros to {frames * SAMPLES_PER_FRAME} "
            f"(a multiple of {SAMPLES_PER_FRAME}); frames = ceil(samples / {SAMPLES_PER_FRAME})",
            "layout": layout.name(),
            "layout_search": results,
            "reconstruction": reconstruction(samples, audio),
            "decoder_samples": len(audio),
            "decoder_samples_sha256": sha256_array(audio),
            "decoder_samples_rms": float(np.sqrt(np.mean(audio**2))),
        }
    )
    case.add(
        "samples",
        samples,
        "f32",
        f"the clip resampled {rate} -> {sample_rate} Hz by torchaudio.functional.resample defaults",
    )
    pin_encoder(case, samples, codes, stages, [k for k, _, _ in ENCODER_HOOKS])
    for key, _, _ in DECODER_HOOKS:
        case.add_edges(
            f"decoder/{key}",
            dec_stages[key],
            EDGE[key],
            f"from_indices on `codes`: {STAGE_NOTES[key]}; [rows, channels]",
        )
    case.dump("decoder/samples", audio)
    n = 2 * SAMPLES_PER_FRAME
    case.add(
        "decoder/samples",
        np.concatenate([audio[:n], audio[-n:]]),
        "f32",
        "from_indices's waveform (after tanh), its first and last 2 frames; the whole's SHA-256 and RMS are in the case",
        edge_rows(len(audio), n),
    )
    if window is not None:
        doubled = np.concatenate([samples, samples])
        wcodes, wstages = encode(codec, doubled)
        waudio, wdec = decode(codec, wcodes)
        window.meta.update(
            {
                "input": "codec-reference's samples twice over, end to end",
                "samples": len(doubled),
                "frames": int(wcodes.shape[1]),
                "windows": "the encoder transformer runs over "
                f"{wstages['enc.blk.3'].shape[0]} rows (window 512); the quantizer transformers over "
                f"{wcodes.shape[1]} frames (window 128)",
                "reconstruction": reconstruction(doubled, waudio),
                "decoder_samples_sha256": sha256_array(waudio),
            }
        )
        for key in ("enc.blk.3", "q.pre"):
            window.dump(f"encoder/{key}", wstages[key])
            ranges = [[wstages[key].shape[0] - 16, wstages[key].shape[0]]]
            window.add(f"encoder/{key}", wstages[key][-16:], "f32", f"the last 16 rows of {STAGE_NOTES[key]}", ranges)
        window.add("codes", wcodes.numpy().T, "i32", "DAC.encode's codes of the doubled clip, [frame][codebook]")
        window.dump("decoder/q.post", wdec["q.post"])
        rows = wdec["q.post"].shape[0]
        window.add(
            "decoder/q.post",
            wdec["q.post"][-16:],
            "f32",
            "from_indices on `codes`: the last 16 rows of the post-transformer",
            [[rows - 16, rows]],
        )
        n = 2 * SAMPLES_PER_FRAME
        window.add(
            "decoder/samples",
            waudio[-n:],
            "f32",
            "the last 2 frames of from_indices's waveform",
            [[len(waudio) - n, len(waudio)]],
        )
    case.meta["names"] = names
    del codec
    return codes, layout


# --------------------------------------------------------------------------------------
# The backbone and the tower.


@dataclass(frozen=True)
class Convention:
    """One reading of the GGUF's backbone/tower layout."""

    # The zero-centred norms stored as 1 + w (Fish's weight = stored - 1), or as w.
    shifted: bool = True
    # ad.codebook_embd row (k - 1) * 4096 + c is Fish's codebook k >= 1 ("k-1"), or row k * 4096 + c for k <= 8 ("k").
    codebooks: str = "k-1"
    # The DeltaNet value heads in Fish's grouped order (value head h reads key head h // 2), or llama.cpp's tiled order
    # (GGUF value head j reads key head j % 16; Fish head h is GGUF head (h % 2) * 16 + h // 2).
    value_heads: str = "grouped"
    # attn_q / attn_gate as named, or swapped.
    swap_query_gate: bool = False

    def name(self) -> str:
        return (
            f"{'shifted' if self.shifted else 'unshifted'}/codebooks {self.codebooks}/{self.value_heads}"
            f"{'/q-gate swapped' if self.swap_query_gate else ''}"
        )


def value_head_order(c: Convention, heads: int) -> np.ndarray:
    """Fish value head h -> GGUF value head."""
    h = np.arange(heads)
    if c.value_heads == "grouped":
        return h
    return (h % 2) * (heads // 2) + h // 2


def tensor(a: np.ndarray) -> Any:
    import torch

    return torch.from_numpy(np.ascontiguousarray(a, dtype=np.float32))


def layer_weights(g: Gguf, i: int, cfg: Any, c: Convention) -> dict[str, np.ndarray]:
    """Fish DecoderLayer parameters (all but the experts) of GGUF `blk.i` under convention `c`."""
    p = f"blk.{i}."
    shift = np.float32(1.0) if c.shifted else np.float32(0.0)
    w: dict[str, np.ndarray] = {
        "input_layernorm.weight": g.get(p + "attn_norm") - shift,
        "post_attention_layernorm.weight": g.get(p + "post_attn_norm") - shift,
        "mlp.gate.weight": g.get(p + "ffn_gate_inp"),
        "mlp.shared_expert.gate_proj.weight": g.get(p + "ffn_gate_shexp"),
        "mlp.shared_expert.up_proj.weight": g.get(p + "ffn_up_shexp"),
        "mlp.shared_expert.down_proj.weight": g.get(p + "ffn_down_shexp"),
        "mlp.shared_expert_gate.weight": g.get(p + "ffn_gate_inp_shexp"),
    }
    if cfg.layer_types[i] == "full_attention":
        heads, dim = cfg.num_attention_heads, cfg.head_dim
        q = g.get(p + "attn_q").reshape(heads, dim, -1)
        gate = g.get(p + "attn_gate").reshape(heads, dim, -1)
        if c.swap_query_gate:
            q, gate = gate, q
        # Fish's q_proj rows are each head's [query, gate].
        w["self_attn.q_proj.weight"] = np.concatenate([q, gate], axis=1).reshape(2 * heads * dim, -1)
        w["self_attn.k_proj.weight"] = g.get(p + "attn_k")
        w["self_attn.v_proj.weight"] = g.get(p + "attn_v")
        w["self_attn.o_proj.weight"] = g.get(p + "attn_output")
        w["self_attn.q_norm.weight"] = g.get(p + "attn_q_norm") - shift
        w["self_attn.k_norm.weight"] = g.get(p + "attn_k_norm") - shift
        return w
    vh, vd = cfg.linear_num_value_heads, cfg.linear_value_head_dim
    key = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    order = value_head_order(c, vh)

    def heads_rows(a: np.ndarray) -> np.ndarray:
        return a.reshape(vh, -1, *a.shape[1:])[order].reshape(a.shape)

    qkv = g.get(p + "ssm_in_qkv")
    w["linear_attn.in_proj_qkv.weight"] = np.concatenate([qkv[: 2 * key], heads_rows(qkv[2 * key :])])
    w["linear_attn.in_proj_z.weight"] = heads_rows(g.get(p + "ssm_in_z"))
    w["linear_attn.in_proj_b.weight"] = heads_rows(g.get(p + "ssm_in_b"))
    w["linear_attn.in_proj_a.weight"] = heads_rows(g.get(p + "ssm_in_a"))
    conv = g.get(p + "ssm_conv1d")  # [channels, 4]
    w["linear_attn.conv1d.weight"] = np.concatenate([conv[: 2 * key], heads_rows(conv[2 * key :])])[:, None, :]
    w["linear_attn.dt_bias"] = g.get(p + "ssm_dt_bias")[order]
    w["linear_attn.A_log"] = g.get(p + "ssm_a_log")[order]
    w["linear_attn.norm.weight"] = g.get(p + "ssm_norm")
    out = g.get(p + "ssm_out")  # [hidden, value heads * dim]
    w["linear_attn.out_proj.weight"] = out.reshape(out.shape[0], vh, vd)[:, order].reshape(out.shape)
    return w


BACKBONE_NAMES = {
    "token_embd": "model.embed_tokens.weight (rows read per id)",
    "output": "lm_head.weight rows drama3.output.token_ids (EOS, then the semantic band)",
    "output_norm": "model.norm.weight + 1 (ZeroCenteredRMSNorm)",
    "blk.N.attn_norm": "model.layers.N.input_layernorm.weight + 1",
    "blk.N.post_attn_norm": "model.layers.N.post_attention_layernorm.weight + 1",
    "blk.N.attn_q, blk.N.attn_gate": "model.layers.N.self_attn.q_proj.weight: per head h, rows [h*512, h*512+256) "
    "= attn_q rows [h*256, h*256+256), rows [h*512+256, h*512+512) = attn_gate's",
    "blk.N.attn_k / attn_v / attn_output": "self_attn.k_proj / v_proj / o_proj",
    "blk.N.attn_q_norm / attn_k_norm": "self_attn.q_norm / k_norm .weight + 1",
    "blk.N.ssm_in_qkv": "linear_attn.in_proj_qkv.weight (rows q 2048 | k 2048 | v 4096)",
    "blk.N.ssm_in_z / ssm_in_b / ssm_in_a": "linear_attn.in_proj_z / in_proj_b / in_proj_a",
    "blk.N.ssm_conv1d": "linear_attn.conv1d.weight [8192, 1, 4] (GGUF [channel][tap])",
    "blk.N.ssm_dt_bias / ssm_a_log": "linear_attn.dt_bias / A_log (A_log raw, not -exp)",
    "blk.N.ssm_norm": "linear_attn.norm.weight (GatedRMSNorm, a plain weight)",
    "blk.N.ssm_out": "linear_attn.out_proj",
    "blk.N.ffn_gate_inp": "mlp.gate.weight (router, 256 rows)",
    "blk.N.ffn_gate_exps / ffn_up_exps": "mlp.experts.w13_weight[e][:512] / [512:] = GGUF rows [e*512, e*512+512)",
    "blk.N.ffn_down_exps": "mlp.experts.w2_weight[e] = GGUF rows [e*2048, e*2048+2048)",
    "blk.N.ffn_gate_shexp / ffn_up_shexp / ffn_down_shexp": "mlp.shared_expert.gate_proj / up_proj / down_proj",
    "blk.N.ffn_gate_inp_shexp": "mlp.shared_expert_gate.weight [1, 2048]",
}
TOWER_NAMES = {
    "ad.codebook_embd": "codebook_embeddings.weight rows [4096, 40960): GGUF row (k-1)*4096 + c is codebook k >= 1, "
    "code c (Fish's codebook 0 rows, never read, are absent)",
    "ad.shared.N.embd": "shared_layers.N.semantic_embedding.weight",
    "ad.layer.N.embd": "residual_layers.N.codebook_embedding.weight",
    "ad.{shared,layer}.N.enorm / hnorm / final_norm": "enorm / hnorm / final_layernorm .weight + 1",
    "ad.{shared,layer}.N.eh_proj": "eh_proj.weight [2048, 4096] (input [enorm(e) | hnorm(hidden)])",
    "ad.{shared,layer}.N.attn_norm / ffn_norm": "transformer_layer.attention_norm / ffn_norm .weight + 1",
    "ad.{shared,layer}.N.attn_q, attn_gate, attn_k, attn_v": "transformer_layer.attention.wqkv.weight rows "
    "[q 4096 | gate 4096 | k 512 | v 512]",
    "ad.{shared,layer}.N.attn_q_norm / attn_k_norm": "attention.q_norm / k_norm .weight + 1",
    "ad.{shared,layer}.N.attn_output": "attention.wo",
    "ad.{shared,layer}.N.ffn_gate / ffn_up / ffn_down": "feed_forward.w1 / w3 / w2",
    "ad.head.N": "codebook_heads.N.weight (predicts code N + 1)",
}


def fill_tower(g: Gguf, tower: Any, c: Convention) -> None:
    """Loads `ad.*` into Fish's FishOmniAudioDecoder under convention `c`, every parameter once."""
    import torch

    shift = np.float32(1.0) if c.shifted else np.float32(0.0)
    cfg = tower.config
    params = dict(tower.named_parameters())
    loaded: set[str] = set()

    def put(name: str, value: np.ndarray) -> None:
        assert name not in loaded and tuple(params[name].shape) == value.shape, (name, value.shape)
        with torch.no_grad():
            params[name].copy_(tensor(value))
        loaded.add(name)

    book = g.get("ad.codebook_embd")
    full = np.zeros((cfg.vocab_size * cfg.num_codebooks, cfg.dim), np.float32)
    if c.codebooks == "k-1":
        full[cfg.vocab_size :] = book
    else:
        full[: book.shape[0]] = book
    put("codebook_embeddings.weight", full)
    for kind, prefix, count, emb in (
        ("shared_layers", "ad.shared", cfg.shared_enrichment_layers, "semantic_embedding"),
        ("residual_layers", "ad.layer", cfg.num_codebooks - 1, "codebook_embedding"),
    ):
        for n in range(count):
            f, q = f"{kind}.{n}", f"{prefix}.{n}"
            put(f"{f}.{emb}.weight", g.get(f"{q}.embd"))
            put(f"{f}.enorm.weight", g.get(f"{q}.enorm") - shift)
            put(f"{f}.hnorm.weight", g.get(f"{q}.hnorm") - shift)
            put(f"{f}.eh_proj.weight", g.get(f"{q}.eh_proj"))
            put(f"{f}.final_layernorm.weight", g.get(f"{q}.final_norm") - shift)
            t = f"{f}.transformer_layer"
            put(f"{t}.attention_norm.weight", g.get(f"{q}.attn_norm") - shift)
            put(f"{t}.ffn_norm.weight", g.get(f"{q}.ffn_norm") - shift)
            query, gate = g.get(f"{q}.attn_q"), g.get(f"{q}.attn_gate")
            if c.swap_query_gate:
                query, gate = gate, query
            put(f"{t}.attention.wqkv.weight", np.concatenate([query, gate, g.get(f"{q}.attn_k"), g.get(f"{q}.attn_v")]))
            put(f"{t}.attention.wo.weight", g.get(f"{q}.attn_output"))
            put(f"{t}.attention.q_norm.weight", g.get(f"{q}.attn_q_norm") - shift)
            put(f"{t}.attention.k_norm.weight", g.get(f"{q}.attn_k_norm") - shift)
            put(f"{t}.feed_forward.w1.weight", g.get(f"{q}.ffn_gate"))
            put(f"{t}.feed_forward.w3.weight", g.get(f"{q}.ffn_up"))
            put(f"{t}.feed_forward.w2.weight", g.get(f"{q}.ffn_down"))
    for n in range(cfg.num_codebooks - 1):
        put(f"codebook_heads.{n}.weight", g.get(f"ad.head.{n}"))
    assert loaded == params.keys(), sorted(params.keys() - loaded)[:8]


def build_frames(tok: Any, config: Any, codes: Any) -> tuple[Any, dict[str, Any]]:
    """Fish's prompt rows plus the teacher-forced continuation; [R, 11] int64 and the layout."""
    import torch
    from fish_speech.models.text2semantic.fish_omni.generate import build_drama3_prompt

    prompt = build_drama3_prompt(TEXT, tok, config, prompt_texts=[REFERENCE_TEXT], prompt_tokens=[codes])
    cont = codes[:, :CONTINUATION_FRAMES].to(torch.long)
    main = cont[:1] + config.semantic_start_token_id
    frames = torch.cat([prompt, torch.cat([main, cont]).T])
    semantic = (frames[:, 0] >= config.semantic_start_token_id) & (frames[:, 0] <= config.semantic_end_token_id)
    rows = semantic.nonzero()[:, 0].tolist()
    reference = [r for r in rows if r < len(prompt)]
    assert reference == list(range(reference[0], reference[0] + codes.shape[1])), "reference rows not contiguous"
    layout = {
        "prompt_rows": len(prompt),
        "rows": len(frames),
        "reference_rows": [reference[0], reference[-1] + 1],
        "continuation_rows": [len(prompt), len(frames)],
    }
    return frames, layout


def exact_delta_rule(
    query: Any, key: Any, value: Any, decay: Any, beta: Any, initial_state: Any = None, chunk_size: int = 64
) -> Any:
    """Fish's gated delta rule, recurrent row by row in F64, for `chunk_gated_delta_rule`.

    Fish's chunked F32 form (64-row chunks, two triangular solves) rounds by about 2e-6
    at layer 2, which flips a near-tied expert choice of the clone prompt (layer 3, row
    34: 0.41913772 against 0.41913646); the exact rule makes the trace a fixed point.
    State: decayed by exp(decay), corrected by beta·(v − Sᵀk); q and k L2-normalized,
    q scaled by key_dim^-0.5; the output in the input's dtype, the state in F32.
    """
    import torch

    del chunk_size
    out_dtype = query.dtype
    batch, length, heads, key_dim = key.shape
    q, k, v, b, d = (t.transpose(1, 2).to(torch.float64).contiguous() for t in (query, key, value, beta, decay))
    q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6) * key_dim**-0.5
    k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    state = (
        v.new_zeros(batch, heads, key_dim, v.shape[-1]) if initial_state is None else initial_state.to(torch.float64)
    )
    out = torch.empty_like(v)
    for i in range(length):
        state = state * d[:, :, i].exp()[..., None, None]
        kt = k[:, :, i]
        remembered = (state * kt.unsqueeze(-1)).sum(-2)
        delta = (v[:, :, i] - remembered) * b[:, :, i].unsqueeze(-1)
        state = state + kt.unsqueeze(-1) * delta.unsqueeze(-2)
        out[:, :, i] = (state * q[:, :, i].unsqueeze(-1)).sum(-2)
    return out.transpose(1, 2).contiguous().to(out_dtype), state.to(torch.float32)


def section_clone(g: Gguf, tok: Any, case: Case, codes: Any, debug: bool) -> None:
    import torch
    import torch.nn.functional as F
    from fish_speech.models.text2semantic.fish_omni.backbone import DecoderLayer
    from fish_speech.models.text2semantic.fish_omni.layers import LayerContext, ZeroCenteredRMSNorm
    from fish_speech.models.text2semantic.fish_omni.tower import FishOmniAudioDecoder
    from fish_speech.models.text2semantic.fish_omni import layers as fish_layers

    fish_layers.chunk_gated_delta_rule = exact_delta_rule  # pyright: ignore[reportAttributeAccessIssue]

    config = fish_config(g)
    cfg = config.text
    frames, layout = build_frames(tok, config, codes)
    R, Rp = layout["rows"], layout["prompt_rows"]
    ref0, ref1 = layout["reference_rows"]
    token_ids = list(g.meta["drama3.output.token_ids"])
    column = {t: i for i, t in enumerate(token_ids)}
    print(f"clone: {R} rows ({Rp} prompt, reference {ref0}..{ref1})", flush=True)

    # Inputs: token_embd rows, then the tower's embed_backbone (Fish's code) per codebook reading.
    ids = frames[:, 0].tolist()
    emb = np.stack([g.get("token_embd", (t, t + 1))[0] for t in ids])
    tower = FishOmniAudioDecoder(config, dtype=torch.float32).eval()
    readings = {"k-1": Convention(), "k": Convention(codebooks="k")}
    inputs = {}
    with torch.inference_mode():
        for key, c in readings.items():
            fill_tower(g, tower, c)
            inputs[key] = tower.embed_backbone(frames, tensor(emb)).float()

    conventions = [
        Convention(value_heads=v, shifted=s, codebooks=b, swap_query_gate=q)
        for v in ("grouped", "tiled")
        for s, b, q in ((True, "k-1", False), (False, "k-1", False), (True, "k", False), (True, "k-1", True))
    ]
    streams = {c: inputs[c.codebooks][None].clone() for c in conventions}
    kept: dict[Convention, dict[int, np.ndarray]] = {c: {} for c in conventions}
    positions = torch.arange(R)
    # LayerContext as FishOmniBackbone.run_layers builds it for a whole sequence from position 0, no caches.
    exponents = torch.arange(0, cfg.rotary_dim, 2, dtype=torch.float32)
    frequencies = 1.0 / (cfg.rope_theta ** (exponents / cfg.rotary_dim))
    angles = positions.float()[:, None] * frequencies[None, :]
    angles = torch.cat((angles, angles), dim=-1)
    context = LayerContext(
        input_pos=positions, cos=angles.cos(), sin=angles.sin(), kv_len=R, continue_state=False, use_cache=False
    )
    experts, inter = cfg.num_experts, cfg.moe_intermediate_size
    for i in range(cfg.num_hidden_layers):
        started = time.perf_counter()
        layer = DecoderLayer(cfg, i, fp8=False, dtype=torch.float32).eval()
        p = f"blk.{i}."
        with torch.no_grad():
            w13 = layer.mlp.experts.w13_weight
            w13[:, :inter].copy_(tensor(g.get(p + "ffn_gate_exps").reshape(experts, inter, -1)))
            w13[:, inter:].copy_(tensor(g.get(p + "ffn_up_exps").reshape(experts, inter, -1)))
            layer.mlp.experts.w2_weight.copy_(
                tensor(g.get(p + "ffn_down_exps").reshape(experts, cfg.hidden_size, inter))
            )
        params = dict(layer.named_parameters())
        for c in conventions:
            weights = layer_weights(g, i, cfg, c)
            assert weights.keys() | {"mlp.experts.w13_weight", "mlp.experts.w2_weight"} == params.keys(), sorted(
                params.keys() - weights.keys()
            )
            with torch.no_grad():
                for name, value in weights.items():
                    assert tuple(params[name].shape) == value.shape, (name, value.shape)
                    params[name].copy_(tensor(value))
            with torch.inference_mode():
                streams[c] = layer(streams[c], context)
            if i in BACKBONE_LAYERS:
                kept[c][i] = streams[c][0].numpy().copy()
        del layer, w13, params
        print(f"  layer {i} ({cfg.layer_types[i]}) {time.perf_counter() - started:.1f}s", flush=True)

    head = tensor(np.stack([g.get("output", (t, t + 1))[0] for t in range(len(token_ids))]))
    final = ZeroCenteredRMSNorm(cfg.hidden_size, cfg.rms_norm_eps, dtype=torch.float32)
    normed, logits, checks = {}, {}, []
    targets_ref = torch.tensor([column[t] for t in ids[ref0:ref1]])
    targets_cont = torch.tensor([column[t] for t in ids[Rp:R]])
    for c in conventions:
        with torch.no_grad():
            final.weight.copy_(tensor(g.get("output_norm") - (1.0 if c.shifted else 0.0)))
        with torch.inference_mode():
            normed[c] = final(streams[c][0]).float()
            logits[c] = F.linear(normed[c], head)
        lp = torch.log_softmax(logits[c], -1)
        ref_lp = lp[ref0 - 1 : ref1 - 1].gather(1, targets_ref[:, None])[:, 0]
        cont_lp = lp[Rp - 1 : R - 1].gather(1, targets_cont[:, None])[:, 0]
        top1 = (lp[ref0 - 1 : ref1 - 1].argmax(-1) == targets_ref).float().mean()
        checks.append(
            {
                "convention": c.name(),
                "reference_mean_logprob": round(float(ref_lp.mean()), 4),
                "reference_top1": round(float(top1), 4),
                "continuation_mean_logprob": round(float(cont_lp.mean()), 4),
            }
        )
        print(f"  backbone {checks[-1]}", flush=True)
    best = max(conventions, key=lambda c: checks[conventions.index(c)]["reference_mean_logprob"])
    ranked = sorted(checks, key=lambda r: -r["reference_mean_logprob"])
    assert best.shifted and best.codebooks == "k-1" and not best.swap_query_gate, f"unexpected winner {best.name()}"
    case.meta["backbone_checks"] = checks
    case.meta["backbone_convention"] = best.name()
    print(f"  backbone winner {best.name()}, next {ranked[1]}", flush=True)

    # Pins: the winner's rows.
    pin = sorted(set(range(8)) | set(range(Rp - 8, Rp)) | set(range(Rp, R)))
    ranges: list[list[int]] = []
    for r in pin:
        if ranges and ranges[-1][1] == r:
            ranges[-1][1] = r + 1
        else:
            ranges.append([r, r + 1])
    case.meta.update(
        {
            "text": TEXT,
            "reference_text": REFERENCE_TEXT,
            **layout,
            "frames_note": "frames.i32 [row][main, c0..c9]: a text row is [token id, 0 x 10] (Fish's build_drama3_prompt), a "
            "semantic row [248099 + c0, c0..c9]; the prompt rows end at <|voice|>, then the continuation rows are the "
            f"reference's first {CONTINUATION_FRAMES} frames, teacher-forced",
            "pinned_rows": ranges,
            "positions": "row r is at position r (one sequence from 0)",
            "logits_columns": "column j is drama3.output.token_ids[j] (j 0: EOS 248046, j 1 + c: semantic code c)",
            "names": BACKBONE_NAMES | TOWER_NAMES,
        }
    )
    case.add("frames", frames.numpy(), "i32", "every row's frame, [row][main, c0..c9]")
    case.add(
        "backbone/inputs",
        inputs[best.codebooks].numpy()[pin],
        "f32",
        "embed_backbone: text rows token_embd[id]; semantic rows (token_embd[main] + codebook_embd[c1]) / sqrt(2)",
        ranges,
    )
    for i in BACKBONE_LAYERS:
        case.dump(f"backbone/layer.{i}", kept[best][i])
        case.add(
            f"backbone/layer.{i}",
            kept[best][i][pin],
            "f32",
            f"the residual stream after layer {i} ({cfg.layer_types[i]})",
            ranges,
        )
    case.add(
        "backbone/norm",
        normed[best].numpy()[pin],
        "f32",
        "after output_norm: the hidden the lm head and the tower read",
        ranges,
    )
    case.add(
        "backbone/logits",
        logits[best].numpy()[pin],
        "f32",
        "the 4,097 `output` logits per pinned row (row r predicts row r + 1's main token)",
        ranges,
    )
    if debug:
        case.dump("backbone/norm.all", normed[best].numpy())
        case.dump("backbone/logits.all", logits[best].numpy())

    # The tower: Fish's prefill over rows 0..R-2, labelled with frame r + 1.
    hidden = normed[best]
    tower_rows = [3, ref0 + 10, Rp - 1, Rp, Rp + 1, R - 2]
    tower_checks = []
    variants = [Convention(), Convention(shifted=False), Convention(codebooks="k"), Convention(swap_query_gate=True)]
    pinned_tower = {}
    for c in variants:
        fill_tower(g, tower, c)
        shared_out: list[Any] = []
        residual_out: list[Any] = []
        handles = [
            tower.shared_layers[-1].register_forward_hook(lambda _m, _i, o: shared_out.append(o.detach().clone()))
        ]
        handles += [
            layer.register_forward_hook(lambda _m, _i, o: residual_out.append(o.detach().clone()))
            for layer in tower.residual_layers
        ]
        with torch.inference_mode():
            tower.prefill(hidden[: R - 1], positions[: R - 1], frames[: R - 1], frames[1:R])
            heads = torch.stack(
                [h(x) for h, x in zip(tower.codebook_heads, residual_out, strict=True)], 1
            )  # [R-1, 9, 4096]
        for h in handles:
            h.remove()
        lp = torch.log_softmax(heads.float(), -1)
        nxt = frames[1:R, 2:]  # codes 1..9 of row r + 1
        picked = lp.gather(2, nxt[:, :, None])[:, :, 0]
        ref = picked[ref0 - 1 : ref1 - 1]
        tower_checks.append(
            {
                "convention": c.name(),
                "reference_mean_logprob": round(float(ref.mean()), 4),
                "reference_per_codebook": [round(float(v), 3) for v in ref.mean(0)],
                "reference_top1": round(
                    float((lp[ref0 - 1 : ref1 - 1].argmax(-1) == nxt[ref0 - 1 : ref1 - 1]).float().mean()), 4
                ),
                "continuation_mean_logprob": round(float(picked[Rp - 1 : R - 1].mean()), 4),
            }
        )
        print(f"  tower {tower_checks[-1]}", flush=True)
        if c == Convention():
            pinned_tower = {
                "shared": shared_out[0].numpy()[tower_rows],
                "residual": torch.stack(residual_out, 1).numpy()[tower_rows],
                "logits": heads.numpy()[tower_rows],
            }
    best_tower = max(tower_checks, key=lambda r: r["reference_mean_logprob"])
    assert best_tower["convention"] == Convention().name(), best_tower
    case.meta["tower_checks"] = tower_checks
    case.meta["tower_rows"] = tower_rows
    case.meta["tower_note"] = (
        "FishOmniAudioDecoder.prefill(hidden = backbone/norm rows 0..R-2, positions, frames[r], next_frames = frames[r + 1]): "
        "row r runs the shared layers on code c0 of frame r + 1 (layer 0 adding the summed codebook 2-9 embeddings of frame r "
        "when frame r is semantic), then residual layer j on c_j of frame r + 1; head j on residual layer j's output predicts "
        "c_{j+1} of frame r + 1. Every tower layer attends over all rows 0..r, text rows included (their e is 0)"
    )
    rr = [[r, r + 1] for r in tower_rows]
    case.add(
        "tower/shared",
        pinned_tower["shared"],
        "f32",
        "the shared stack's output (after shared layer 3) at tower_rows",
        rr,
    )
    case.add("tower/residual", pinned_tower["residual"], "f32", "[row][j] residual layer j's output at tower_rows", rr)
    case.add(
        "tower/logits",
        pinned_tower["logits"],
        "f32",
        "[row][j] codebook head j's 4,096 logits (code j + 1 of the next frame) at tower_rows",
        rr,
    )


# --------------------------------------------------------------------------------------


def verdicts(cases: dict[str, Any]) -> dict[str, Any]:
    """Each layout question and the reading the checks chose, with its score and the best rejected one."""
    out: dict[str, Any] = {}
    codec = cases["codec-reference"]
    scores = sorted(codec["layout_search"], key=lambda r: r["log_spectral_l1"])
    out["codec layout"] = {
        "chosen": codec["layout"],
        "measure": "encode then decode the reference clip: snr_db, log_spectral_l1",
        "chosen_score": scores[0],
        "best_rejected": scores[1],
    }
    clone = cases["clone-fox"]
    for network, key, base in (
        ("backbone", "backbone_checks", "shifted/codebooks k-1/grouped"),
        ("tower", "tower_checks", "shifted/codebooks k-1/grouped"),
    ):
        checks = {c["convention"]: c for c in clone[key]}
        rivals = {
            "norm shift (stored 1 + w)": "unshifted/codebooks k-1/grouped",
            "codebook rows (k - 1) * 4096 + c": "shifted/codebooks k/grouped",
            "query/gate split as named": "shifted/codebooks k-1/grouped/q-gate swapped",
        }
        if network == "backbone":
            rivals["DeltaNet value heads grouped (Fish's order, no permutation)"] = "shifted/codebooks k-1/tiled"
        for question, rival in rivals.items():
            out[f"{network}: {question}"] = {
                "measure": "mean log-prob of the reference's own codes, teacher-forced",
                "chosen": checks[base]["reference_mean_logprob"],
                "rejected": checks[rival]["reference_mean_logprob"],
                "rejected_convention": rival,
            }
    return out


CASES = ("tokenizer", "codec-reference", "codec-window", "clone-fox")


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--cases", help="comma-separated, of: " + ", ".join(CASES))
    parser.add_argument("--debug-dir", type=pathlib.Path, help="also write every stage whole here (unpinned)")
    parser.add_argument(
        "--no-layout-search", action="store_true", help="skip the codec layout search (use the decided layout)"
    )
    args = parser.parse_args()
    chosen = set(args.cases.split(",")) if args.cases else set(CASES)
    if unknown := chosen - set(CASES):
        raise SystemExit(f"--cases: no case named {sorted(unknown)}")

    import tokenizers
    import torch

    fish_imports()
    torch.set_num_threads(max(1, os.cpu_count() or 1))
    sidecar = json.loads(GGUF.with_name(GGUF.name + ".nuclis.json").read_text())
    assert sidecar["sha256"] == GGUF_SHA256, sidecar["sha256"]
    g = Gguf(GGUF)
    tok = build_tokenizer(g)
    index_path = OUT / "index.json"
    index: dict[str, Any] = json.loads(index_path.read_text()) if index_path.exists() else {"cases": {}}
    index.update(
        {
            "$comment": "Written by scripts/speech-traces/drama3_trace.py; docs/models/drama3.md § The traces.",
            "gguf": {"repo": GGUF_REPO, "revision": sidecar["revision"], "file": GGUF.name, "sha256": GGUF_SHA256},
            "package": PACKAGE,
            "package_commit": PACKAGE_COMMIT,
            "torch": torch.__version__,
            "tokenizers": tokenizers.__version__,
            "numpy": np.__version__,
            "device": "cpu",
            "dtype": "float32 (every GGUF tensor dequantized to F32: Q4_K by ggml's formula, F16 widened); the clone case's gated delta rule recurrent in float64 (exact_delta_rule)",
            "conventions": "Each file is little-endian, row-major, time first. `rows` lists the [start, end) ranges of the "
            "whole stage's rows the file holds, in order; a file without `rows` is whole.",
        }
    )
    if "tokenizer" in chosen:
        case = Case("tokenizer", args.debug_dir)
        section_tokenizer(g, tok, case)
        index["cases"]["tokenizer"] = case.entry()
    codes = None
    if chosen & {"codec-reference", "codec-window"}:
        ref = Case("codec-reference", args.debug_dir)
        window = Case("codec-window", args.debug_dir) if "codec-window" in chosen else None
        codes, _ = section_codec(g, ref, window, not args.no_layout_search)
        index["cases"]["codec-reference"] = ref.entry()
        if window is not None:
            index["cases"]["codec-window"] = window.entry()
        index_path.parent.mkdir(parents=True, exist_ok=True)
        index_path.write_text(json.dumps(index, indent=1, ensure_ascii=False) + "\n")
    if "clone-fox" in chosen:
        if codes is None:
            pinned = np.fromfile(OUT / "codec-reference/codes.i32", "<i4").reshape(-1, 10)
            codes = torch.from_numpy(pinned.T.astype(np.int64).copy())
        case = Case("clone-fox", args.debug_dir)
        case.meta["reference"] = "codec-reference (its codes are Reference 1's)"
        section_clone(g, tok, case, codes, args.debug_dir is not None)
        index["cases"]["clone-fox"] = case.entry()
        index["verdicts"] = verdicts(index["cases"])
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_path.write_text(json.dumps(index, indent=1, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
