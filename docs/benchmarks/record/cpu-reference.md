<!-- kind: record -->
# CPU numerical references: the record

The dated record behind [the reference](../../engine/cpu-reference.md).

## Pinned vector fixtures' generator (2026-09-15)

The committed [fixtures](../../../inference/src/backends/cpu/fixtures/vector.json)
come from llama.cpp `7620399f58aebfd2196b74021f9581bcf7218cb9`, its matching local
Release build, and one CPU thread, written by `scripts/cpu-vector-fixtures.py`
(retired; at tag `v0.11.0`); a re-pinned file's fixtures are made by nuclis
([models § Which executor makes which fixture](../../models/README.md#which-executor-makes-which-fixture)).

The script used the local C ABI to build and compute small graphs. Four L2 cases
cover a 3–4 vector, denominator clamping, zero, and tiny values. Four RoPE cases
use a synthetic 256-channel head at positions 0, 1, 127, and 32,767, with rotary
width 64, base 10,000,000, IMRoPE mode 40, sections `[11,11,10,0]`, positions
`[p,p,p,0]`, frequency scale 1, extension factor 0, and attention factor 1
([models § 4](../../models/README.md#4-validate-against-nuclis-itself) tells how
that build served as the reference).

Eleven test blocks cover convolution orientation and independent channels,
updated history, kernel one, the DeltaNet equations, beta/decay extremes,
rectangular state, replay from an explicitly copied checkpoint, and transactional
failure including overflow in a later channel. Three consecutive convolution and
DeltaNet steps are checked against [pinned fixtures](../../../inference/src/backends/cpu/fixtures/recurrent.json),
carrying our own state between steps. DeltaNet outputs and updated matrices use
absolute tolerance `1e-6`; the exactly representable convolution examples and
history shifts match exactly. The fixture generator calls `ggml_ssm_conv` and
`ggml_gated_delta_net` on the pinned one-thread CPU backend; the latter exports
both output and final state. Expected convolution history is the explicit window
shift, not an output of `ggml_ssm_conv`. The generator is
`scripts/cpu-vector-fixtures.py` (retired; at tag `v0.11.0`); no model or GPU is needed
to run them.

## Chunkwise DeltaNet evidence (2026-09-09)

Evidence (2026-09-09, four test blocks): against an F64-state sequential
loop on a model-shaped chunk (C = 64, 128×128 state, random gates in the
model's ranges) the outputs and carry are **bit-identical after the F32
cast** (bound 1e-9); against the production `delta`, which rounds state to
F32 per step, one chunk of 64 and two chunks of 40 + 24 with the state
carried between them agree to 2.4e-7 (bound 1e-4); the three pinned
sequential fixture steps run as one chunk reproduce the pinned outputs and
final state to 1e-6; validation failures and the single-token case match
`delta` exactly. The Metal kernel (`nu_delta_chunk`) is checked
against this function and against the sequential steps.
