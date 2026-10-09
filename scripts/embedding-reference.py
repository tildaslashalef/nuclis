#!/usr/bin/env python3
# pyright: reportMissingImports=false
# (numpy, torch, and sentence-transformers live only in the reference venv the docstring names.)
"""Record EmbeddingGemma 2's semantic oracle: Google's checkpoint, float32, on the CPU.

Never imported by the build. It needs the network once (about 1.4 GB) and
runs in the reference venv:

    uv venv -p 3.12 .reference/venv-embed
    VIRTUAL_ENV=.reference/venv-embed uv pip install 'sentence-transformers>=6.1' \
        transformers torch torchvision pillow numpy soundfile librosa
    HF_HOME=$PWD/.reference/hf-home .reference/venv-embed/bin/python scripts/embedding-reference.py

It embeds every case of tests/fixtures/embeddinggemma-inputs/inputs.json one
at a time (text parts are already rendered with their prefixes, so no
`prompt_name`), and writes tests/fixtures/embeddinggemma-vectors/:
`st-f32.json` (versions, per case its token ids run-length encoded as
`[id, count]` or `id`) with `st-f32.f32` (the unit vectors, 768 little-endian
F32 per case in case order), and per audio case the processor's log-mel
features under tests/fixtures/embeddinggemma-audio-features/.
"""

import argparse
import json
import pathlib
import platform
import sys

MODEL = "google/embeddinggemma-2"
REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
ROOT = pathlib.Path(__file__).resolve().parent.parent
INPUTS = ROOT / "tests/fixtures/embeddinggemma-inputs/inputs.json"
VECTORS = ROOT / "tests/fixtures/embeddinggemma-vectors"
FEATURES = ROOT / "tests/fixtures/embeddinggemma-audio-features"
PLACEHOLDER = {"image": "<|image|>", "audio": "<|audio|>"}


def st_input(parts):
    """One case as sentence-transformers takes it: a bare string, or the text
    with a placeholder per media part (no whitespace added) and the media in order."""
    if all("text" in p for p in parts):
        return "".join(p["text"] for p in parts)
    text, media = [], {"image": [], "audio": []}
    for p in parts:
        if "text" in p:
            text.append(p["text"])
            continue
        kind = next(iter(p))
        text.append(PLACEHOLDER[kind])
        media[kind].append(str(ROOT / p[kind]))
    out = {k: v[0] if len(v) == 1 else v for k, v in media.items() if v}
    if len(parts) > 1:
        out["text"] = "".join(text)
    return out


def runs(ids):
    out = []
    for i in ids:
        if out and isinstance(out[-1], list) and out[-1][0] == i:
            out[-1][1] += 1
        elif out and out[-1] == i:
            out[-1] = [i, 2]
        else:
            out.append(i)
    return out


def main():
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--only", nargs="*", help="case ids to run (default all; writes nothing)")
    args = parser.parse_args()

    import numpy as np
    import sentence_transformers
    import torch
    import transformers
    from sentence_transformers import SentenceTransformer

    torch.manual_seed(0)
    model = SentenceTransformer(MODEL, revision=REVISION, device="cpu", model_kwargs={"dtype": torch.float32})
    module = model[0]
    spec = json.loads(INPUTS.read_text())
    cases = [c for c in spec["cases"] if not args.only or c["id"] in args.only]
    vectors, records = [], []
    for case in cases:
        x = st_input(case["parts"])
        features = module.preprocess([x])
        ids = features["input_ids"][0].tolist()
        with torch.inference_mode():
            v = model.encode(x, convert_to_numpy=True, normalize_embeddings=True, batch_size=1)
        v = np.asarray(v, dtype=np.float32)
        if v.shape != (768,) or not np.isfinite(v).all():
            sys.exit(f"{case['id']}: unexpected vector {v.shape}")
        record = {"id": case["id"], "tokens": len(ids), "ids": runs(ids), "norm": float(np.linalg.norm(v))}
        if "input_features" in features:
            frames = int(features["input_features_mask"][0].sum())
            mel = features["input_features"][0][:frames].to(torch.float32).numpy()
            record["mel_frames"] = frames
            if not args.only:
                FEATURES.mkdir(parents=True, exist_ok=True)
                mel.astype("<f4").tofile(FEATURES / f"{case['id']}.f32")
        if "image_position_ids" in features:
            # The patch grid the processor chose (rows, columns), padding rows excluded.
            grid = features["image_position_ids"][0]
            real = grid[(grid >= 0).all(dim=-1)]
            record["patch_grid"] = [int(real[:, 1].max()) + 1, int(real[:, 0].max()) + 1]
        records.append(record)
        vectors.append(v)
        print(f"{case['id']:28s} {len(ids):5d} tokens  norm {record['norm']:.7f}", flush=True)
    if args.only:
        return
    VECTORS.mkdir(parents=True, exist_ok=True)
    np.stack(vectors).astype("<f4").tofile(VECTORS / "st-f32.f32")
    meta = {
        "source": f"{MODEL}@{REVISION}",
        "dtype": "float32",
        "device": "cpu",
        "versions": {
            "sentence_transformers": sentence_transformers.__version__,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "dimensions": 768,
        "cases": records,
    }
    (VECTORS / "st-f32.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1) + "\n")


if __name__ == "__main__":
    main()
