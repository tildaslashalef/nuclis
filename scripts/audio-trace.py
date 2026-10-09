#!/usr/bin/env python3
# pyright: reportMissingImports=false
"""Dump Google's audio-tower stages for one clip, for embeddinggemma-check's trace.

Runs `google/embeddinggemma-2` (float32, CPU, the pinned revision) on a clip and
writes each stage the CPU reference's observer names (`subsample`, `l0-ff1`,
`l0-attn`, `l0-lconv`, `layer-<i>`, `output`, `projected`) as little-endian F32
rows to `.zig-cache/audio-trace/<stage>.f32`. `zig build test-embeddinggemma --
MODEL audio --mmproj MMPROJ` then traces the clip that has a `media-0` fixture
(`audio.north`) against them. A debugging aid, not a fixture:

    HF_HOME=$PWD/.reference/hf-home .reference/venv-embed/bin/python scripts/audio-trace.py
"""

from __future__ import annotations

import importlib.util
import pathlib
from typing import Any

ROOT = pathlib.Path(__file__).resolve().parent.parent
CLIP = "tests/fixtures/embeddinggemma-inputs/speech-1.wav"
OUT = ROOT / ".zig-cache/audio-trace"


def main() -> None:
    import torch
    from sentence_transformers import SentenceTransformer

    spec = importlib.util.spec_from_file_location("reference", ROOT / "scripts/embedding-reference.py")
    assert spec and spec.loader
    reference: Any = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    model = SentenceTransformer(
        reference.MODEL, revision=reference.REVISION, device="cpu", model_kwargs={"dtype": torch.float32}
    )
    module: Any = model[0]
    features = module.preprocess([reference.st_input([{"audio": CLIP}])])
    tower = module.model.audio_tower
    stages: dict[str, Any] = {}

    def hook(name: str) -> Any:
        def record(_module: Any, _inputs: Any, output: Any) -> None:
            value = output[0] if isinstance(output, (tuple, list)) else output
            stages[name] = value.detach().float().numpy()

        return record

    tower.subsample_conv_projection.register_forward_hook(hook("subsample"))
    tower.layers[0].feed_forward1.register_forward_hook(hook("l0-ff1"))
    tower.layers[0].self_attn.register_forward_hook(hook("l0-attn"))
    tower.layers[0].lconv1d.register_forward_hook(hook("l0-lconv"))
    for i, layer in enumerate(tower.layers):
        layer.register_forward_hook(hook(f"layer-{i}"))
    tower.output_proj.register_forward_hook(hook("output"))
    with torch.inference_mode():
        out = tower(features["input_features"].float(), features["input_features_mask"], return_dict=True)
        stages["projected"] = module.model.embed_audio(inputs_embeds=out.last_hidden_state).float().numpy()
    OUT.mkdir(parents=True, exist_ok=True)
    for name, value in stages.items():
        rows = value.reshape(-1, value.shape[-1])
        rows.astype("<f4").tofile(OUT / f"{name}.f32")
        print(f"{name:10s} {rows.shape}")


if __name__ == "__main__":
    main()
