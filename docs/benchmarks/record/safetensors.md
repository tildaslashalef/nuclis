<!-- kind: record -->
# Safetensors: the record

The dated record behind [the reference](../../engine/safetensors.md).


## Evidence (2026-09-27)

- Laya (`convaiinnovations/laya` at `55cf4c4e`, root set): 206 tensors,
  205 F16 and 1 F32, header 21,536 bytes, 842,587,666 tensor bytes in a
  842,609,210-byte file, the same counts a Python read of the remote
  header gave. SmolLM2-135M: 272 BF16 tensors.
- Decoded values equal an independent Python `struct` read: Laya
  `temperature` (F32) `1 1 1`, `type_emb.weight` (F16) `-1.4257812
  -0.22570801 0.2331543 -1.34375 …`, SmolLM2 `model.norm.weight` (BF16)
  `1.75 1.8046875 1.765625 1.59375 …`.
- Real headers of five other repositories, written into sparse files of
  the true size so the tiling check runs against the real byte count, all
  parse: `Qwen/Qwen3-0.6B` (311 BF16), `Qwen/Qwen3-0.6B-FP8` (196 F8_E4M3,
  311 BF16), `openai/gpt-oss-20b` shard 0 (40 U8 MXFP4 blocks and scales,
  156 BF16), `answerdotai/ModernBERT-large` (173 F32), and
  `Qwen/Qwen2.5-7B-Instruct` shard 1 of 4 (82 BF16).
