<!-- kind: reference -->
# Vision through the companion projectors

How nuclis turns an image into embedding rows the language model consumes,
family by family: the shared contract, the preprocessing, each projector's
tensor facts, and how the language model feeds the rows. The traces, the
measurements, and the investigations behind it are in
[the record](../benchmarks/record/vision.md).

**Contents.**

- [The contract](#the-contract)
- [The Qwen3-VL projector](#the-qwen3-vl-projector)
- [Images in the chat](#images-in-the-chat)
- [Gemma 4's projectors](#gemma-4s-projectors)
- [Muse Glimmer's projector](#muse-glimmers-projector)
- [Encoder attention](#encoder-attention)

## The contract

A *projector file* is the catalogue's `mmproj` companion, a GGUF of
architecture `clip`. Its adapter lives in `inference/src/vision/` and, given
a decoded image, returns one *feature row* per output token at the language
model's width. The chat and `generate` place those rows into an *image span*:
a run of placeholder tokens in the prompt whose embedding rows the engine
overwrites with the features at prefill.

The package (`inference/src/vision/root.zig`):

- `image.zig` — `Rgb8` and `decode`: a P6 PPM parser for fixtures and tests
  on every platform, and on macOS an ImageIO bridge (`image_bridge.m`,
  `CGImageSourceCreateWithData` → RGB8; PNG, JPEG, HEIC, WebP, TIFF, GIF,
  BMP) behind the same C interface the Metal bridge uses. An 8-bit RGB
  image is copied as stored, alpha dropped unpremultiplied and no colour
  conversion, as stb_image and PIL's `convert("RGB")` read it; drawing it
  would premultiply a translucent pixel (it changed 12 % of the RGBA logo's
  values). Other layouts are drawn into an RGBA8 context. Bounds are host
  constants: image bytes ≤ 32 MiB, decoded pixels ≤ 64 M.
- `preprocess.zig` — the reference's smart size (`smartSize`: round each side
  to the patch·merge grid, then scale to the pixel bounds; a side held at
  one grid cell cuts the other, so the result never passes `max_pixels`), a
  Pillow-compatible separable resize in 22-bit fixed point with the
  **bicubic** or the **Lanczos** filter (`resize`, `resizeBicubic`,
  `resizeLanczos`; `resizeLetterbox` for the `PAD_CEIL` letterbox;
  `resizeWith(…, .torchvision)` rounds the weights as torchvision's native
  uint8 resize does, which EmbeddingGemma 2's processor calls), and
  `patches`, the channel-planar `[c][ky][kx]` layout the convolution reads,
  normalized by mean/std and rounded to F16 as the reference's im2col does.
- `qwen3vl.zig` / `qwen3vl_metal.zig` — the Qwen3-VL adapter (below).
- `gemma4.zig` / `gemma4_metal.zig` — Gemma 4's two adapters (below).
- `muse_glimmer.zig` / `muse_glimmer_metal.zig` — Muse Glimmer's adapter
  (below).
- `attention_metal.zig` — `Spans`, the encoders' bidirectional attention on
  Metal: per-row window bounds and query tiles over the materialized-score
  kernels (below).
- `plan_metal.zig` — what the three Metal plans share: `Weight` (a buffer
  and its matrix), `created` (a counted device buffer), `wrapped` and
  `weight` (mapped bytes in place), and `paddedDown` (an FFN down matrix
  copied with its rows zero-padded to the tile's columns).
- `projector.zig` — `Projector`, one loaded projector whatever its family:
  the file's projector type (`clip.projector_type` for Qwen and Muse,
  `clip.vision.projector_type` for Gemma) picks the adapter and the
  executor its CPU reference or Metal plan; `grid`, `prepare` (resize and
  patch layout), `encode`, `outputWidth`, `maxTokens`, and `bidirectional`
  are what the engine and the checks call.
- `Span { start, count, width_tokens, height_tokens }` — one image span; the
  engine's `locateImageSpans` pairs the runs of the profile's placeholder
  (`Profile.imagePlaceholder`: `<|image_pad|>`, `<|image|>`, `<|patch|>`) with the
  encoded images in order.

The prompt seam is in `engine.zig`: `Engine.loadVision(path, max_tokens)` loads a
projector on the engine's backend with the image token cap in effect
(below); `encodeImage(bytes)` decodes, preprocesses,
and runs the projector to `PreparedImage { width_tokens, height_tokens,
features }`; `profiles.Message.images` (a `[]const ImageRef`) makes the
profile render the family's markers with the right placeholder count;
`locateImageSpans` finds the runs (`mediaSpans` the runs of images and
audio clips together, [gemma4.md § Audio input](../models/gemma4.md#audio-input-on-the-e4b)); `runLoop`'s `images` argument carries the
spans and their concatenated features into `Model.prefillVision`. The Qwen
runtime substitutes a feature row for a token embedding per span row
(`stepImage`); the Metal plans copy the chunk's feature rows into `x_c`
instead of the embedding gather (Muse's before its weightless input norm,
which the reference applies to image rows too). Gemma 4's spans attend
bidirectionally, so they are fed differently (below). The projector's output width must
equal the artifact's `<architecture>.embedding_length`
(`VisionSourceMismatch`). Speculation is off on the vision prefill; after
it, decode is unchanged.

**The image token cap.** Each family has the reference's token range
(`Projector.tokenRange`: Qwen3.8 8–1,024, Gemma 4 70–1,120, Muse Glimmer
1–4,096), and the grid functions take the cap in effect.
`generation.image_max_tokens` sets it (`"auto"`, the default, is the
family's maximum; a count is clamped into the family's range, so one value
serves every model), `models.<name>.generation.image_max_tokens` overrides
it per model, and `--image-max-tokens auto|N` on `generate` and `agent`
overrides both (`Projector.limitTokens`, called by `loadVision`). A lower
cap is a smaller grid of the same image: fewer tokens, less detail, a
faster encode and prefill ([record, 2026-09-23](../benchmarks/record/vision.md#muse-glimmers-projector-2026-09-23)). A resumed chat re-encodes
its images at the current cap; the conversation is prefilled anew, so the
recorded grid binds nothing.

## The Qwen3-VL projector

The file is `unsloth/Qwen3.8-27B-GGUF/mmproj-BF16.gguf` (931 MB; 224 F32 +
110 BF16 tensors), `clip.projector_type = qwen3vl_merger`.

**Shapes.** 27 pre-LayerNorm blocks, hidden 1152, FFN 4304, 16 heads of 72,
patch 16, spatial merge 2, projection width 5120, image size 768, LayerNorm
eps 1e-6, mean/std 0.5. Output tokens `= (w/16/2) × (h/16/2)`, after the
smart size and the Pillow bicubic letterbox.

**The graph.**

1. **Patch embedding.** A 3-D convolution with temporal patch 2; for a still
   image the two temporal slices `v.patch_embd.weight` and `.weight.1` are
   summed (`patchKernel`, one `1152 × 768` F32 matrix), then
   `+ v.patch_embd.bias`.
2. **Learned positions.** `v.position_embd.weight` is a 48×48 grid,
   bilinearly resized with corners aligned to the image's patch grid
   (`positionRows`), added before the blocks.
3. **The merge walk.** Patch token `t` is patch `(x, y)` where blocks are
   raster order and inside a 2×2 block the order is top-left, top-right,
   bottom-left, bottom-right (`patchPosition`).
4. **Blocks.** LayerNorm, fused QKV (`+bias`), 2-D RoPE (pairs `[0,18)` turn
   with the patch's y, `[18,36)` with its x, at base 1e4), full
   bidirectional attention (scale `1/√72`), output projection, residual;
   LayerNorm, tanh-GELU FFN, residual.
5. **Post-norm and merger.** `v.post_ln`, then each merge block's four patch
   rows concatenated to 4608 through `mm.0` (GELU) and `mm.2` to 5120.

**M-RoPE on the language model.** An image span's rows carry the triple
`(t, t + i/nx, t + i%nx)` with `nx = width_tokens` and `t` the span's text
position; the span advances the text position by `max(width_tokens,
height_tokens)`. The rotary sections `[11, 11, 10]` interleave t, h, w over
the 32 rotary pairs (`cpu.rope.applyMultiAxis`, the Metal `fillChunkRope`
per-row table). Text rows carry `(p, p, p)`. The language model's attention
stays causal by slot: the merged raster token order equals a 2-D causal
mask. After a span the cache row and the rotary position differ, so the
decode step passes them separately (`qwen35_metal.zig`
`fullAttention(…, row, rope_position)`); why: [record,
2026-09-23](../benchmarks/record/vision.md#decode-after-an-image-2026-09-23).

**Executors and memory.** The Metal plan wraps the file's weight bytes in
place except the summed patch kernel and each FFN down matrix (4,304
columns padded to the tile's 4,352); activations are sized for
`max_patches = 4,096`. The projector runs as its own command buffer.

**Bonsai 2.** `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf` is this projector
re-encoded: matrices Q8_0, `ffn_down` F16 (its 4,304 columns are not a
multiple of Q8_0's 32-value block), vectors and the patch kernel F32.
`bind` accepts F32, F16, BF16, and Q8_0 matrices and an element encoding
for `ffn_down`. Bonsai's language model runs in a Hadamard-rotated basis:
token rows are rotated back after the embedding, and feature rows are fed
as they are. No trace is pinned for this file.

**CLI and gates.** `generate --image <path>` (repeatable, up to 8) attaches
images; `models.<name>.mmproj` names the projector, loaded on demand. The
gates are `qwen38-vision-metal` and `qwen38-vision-cpu`, each
`generation-check --vision-check` against `fixtures/qwen3vl-synthetic/`
(the synthetic 96×64 P6 image, its resized pixels, feature rows, and first
eight greedy tokens); they also step through the greedy tokens and compare
the logits with one batched prefill of the same tokens (bound 2e-2).
`synthetic-96x64-torchvision-{down,up}.ppm` pin
`resizeWith(…, .torchvision)`. `--vision-image <file> --vision-oracle
<dir>` runs the projector on any image against an oracle directory and
teacher-forces the language model.

**Limits.** Deepstack projectors and dynamic `image_min/max_pixels`
overrides are not supported.

## Images in the chat

**The path in.** A file dropped onto the terminal arrives as a bracketed
paste of its path (Terminal.app and iTerm2 backslash-escape spaces, some
terminals prefix `file://`); `editor.droppedPath` undoes that and the chat's
probe (`src/agent/root.zig` `dropProbe`) stats the file. An image by
extension becomes an `[image #N]` chip whose bytes in the buffer are the
marker and whose path waits in the editor's attachments; a UTF-8 text file
within the 128 KiB input limit becomes a `[file name, N lines]` chip over
its fenced content. A typed image path at submit reaches the same
`Editor.attachImage`. The editor never touches the file system: tests
supply the probe. At most 8 images per prompt; markers are single-digit, so
renumbering after a deletion rewrites bytes of equal length in place.

**The turn.** The chat reads each image (`engine.readImage`, 32 MiB) and
runs it through the loaded projector (`Engine.encodeImage`), giving a
`loop.Image`: path, file size, and the `PreparedImage`. The projector is
loaded on the first attachment (`visionAvailable`); an entry without one,
or one whose projector cannot be bound, refuses the chip with a notice and
leaves the paste as text. A failure reading or encoding an image is a
notice naming it, and the turn is not sent. The agent owns the images
beside the history item (`Item.images` and the profile's `ImageRef`s),
frees them with the item, and hands `Model.run` the images of every
rendered message in order. The completer counts the placeholder runs in
the remainder it prefills and pairs them with the tail of that list
(consumed text is a prefix of the render and compaction drops whole
earlier turns, so the images still rendered are a suffix).

**Speculation.** `runLoop` runs an image prefill without the drafter, whose
cache is stale from then on; the completer keeps speculation off while what
the session consumed holds an image or a clip
([agent.md § The agent's token cache](../app/agent.md#12-the-agents-token-cache)),
so a conversation that has fed an image decodes at the plain rate.

**What is shown and stored.** The transcript's user block gets one dim
detail row per image (`image #1: /path/shot.png (320×240 → 10×8 tokens)`)
and, where `tui.graphics.enabled` finds Ghostty or kitty in the environment
(`NUCLIS_NO_PREVIEW=1` disables it), a preview: the decoded image
box-filtered to a longest side of 512 px, sent as one kitty direct
transmission (`f=24`, base64 in 4096-byte chunks, `q=2`, `C=1`) over a box
of at most 12 rows, assuming 1:2 cells. Not under tmux (`TMUX` set), which
cannot scroll a picture it does not know about. The session file records a
user entry's images as `{path, width, height, width_tokens, height_tokens}`
and no pixels; `nuclis agent export` lists them under the prompt; a resumed
session decodes and encodes each image again and leaves a missing or
unreadable one out of the model's view with a notice.

**Limits.** Clipboard images (Cmd-V of a bitmap) are not taken; PDF is not
extracted. The preview is re-transmitted on a fold or resize replay.

## Gemma 4's projectors

Both place one token per 48×48 pixels in raster order, within 70 to 1,120
tokens, after the same smart size and PAD_CEIL bicubic letterbox as
Qwen's (align 48; `gemma4.gridFor`). The file's projector type
(`clip.vision.projector_type`) picks the adapter.

**The unified embedder (`gemma4uv`, the legacy 12B).** `mmproj-BF16.gguf`,
175 MB: ten vision tensors (the file's audio pair is not loaded). The patch
is a whole 48-pixel token. Per token: its 6,912 F32 pixel values (mean 0,
std 1) → LayerNorm `patch_norm.1` (eps 1e-5) → `patch_embd` (F32
3840×6912) + bias → LayerNorm `patch_norm.2` → + the x table's row
`i % columns` + the y table's row `i / columns` (`position_embd` F32
[3840, 1120, 2]) → LayerNorm `patch_norm.3` → RMS norm without a weight
(eps 1e-6) → `mm.input_projection` (BF16 3840×3840). No attention: the
language model does the vision work.

**The SigLIP encoder (`gemma4v`: 26B-A4B and E4B).** The 26B-A4B's
`mmproj-BF16.gguf` is 1.19 GB, 356 tensors, 27 blocks of width 1152; the
E4B's is 16 blocks of 12 heads of 64. 16-pixel patches of `2x − 1` rounded
to F16, the F32 patch kernel (no bias), + x/y tables; every norm RMS (eps
1e-6): `x += rms(W_o · attn(h))·attn_post_norm` with h = rms(x)·ln1, q/k/v
without biases, per-head RMS of q and k, 2-D NEOX RoPE base 100 (the first
half of the head turns with the patch's x, the second with its y), v
per-head RMS without a weight, and bidirectional attention at scale 1;
then `x += rms(W_down(act(W_gate h) ⊙ W_up h))·ffn_post_norm` with
h = rms(x)·ln2. The activation is `gelu_quick` (x·σ(1.702x)) for a file
that names none; `bind` refuses a file that names one, and the binding's
`activation` selects tanh for EmbeddingGemma 2
([embeddinggemma.md § Images](../models/embeddinggemma.md#images)).
After the blocks: a 3×3 average pool over the patch grid, × √width,
`(x − std_bias) ⊙ std_scale`, RMS norm without a weight, and
`mm.input_projection`. The Metal plan pools on the host between two
command buffers and pads the FFN down matrices to 4,352 columns;
activations are sized for 10,080 patches (~700 MB on the 26B-A4B).

**The language model.** The profile renders `<|image>` (255999) +
`count × <|image|>` (258880) + `<image|>` (258882) before the user text, no
newlines. Image rows are not scaled by √width (only token rows are). The
image is non-causal on the **sliding layers only**: a span row sees every
row of its span and the window below it; the global layers stay causal.

- `metal.Backend.attentionChunk` takes `span { begin, end }`: chunk rows in
  it see keys up to `position + end` instead of their own row (the
  row-split body; the register-reuse body refuses a span). The Metal plan
  feeds a span as one chunk with the span on its sliding layers; the chunk
  buffers grow to the largest image (`reserveRows`, ~0.3 MB per row).
- The CPU runtime's `prefillVision` runs a span as one batched pass per
  layer: every row writes its cache row, then each row attends over
  `[first, span end)` on a sliding layer and `[0, row]` on a global one.
- Gemma has no M-RoPE: the cache row, the rotary position, and the visible
  count are one value (`state.position` + row) on both executors; decode
  after an image is the ordinary step.

**Gates.** `gemma4-26b-a4b-vision-metal` and `gemma4-e4b-vision-metal`
(`verify`), `gemma4-26b-a4b-vision-cpu` and `gemma4-e4b-vision-cpu`
(`verify-cpu`): each `generation-check --vision-check` against its fixture
(`gemma4v-synthetic/`, `gemma4v-e4b-synthetic/`), a 3×2-token span. The
12B's `gemma4uv-synthetic/` fixture stays in the tree without a gate.

**Limits.** SigLIP rows track an implementation with BF16 activations only
as far as F32 activations track BF16 ones
([record, 2026-09-23](../benchmarks/record/vision.md#gemma-4s-projectors-2026-09-23)).
The 12B file's audio embedder is not loaded. A span must fit the context
(`ContextFull` otherwise).

## Muse Glimmer's projector

**The file.** `mmproj-kquant.gguf`, 1.40 GB, 809 tensors,
`clip.projector_type = muse-glimmer`, mean/std 0.5, LayerNorm eps 1e-5,
patch 14, merge 2. Fifty blocks: `ln1`/`ln2` with biases,
`attn_q`/`attn_k`/`attn_out` Q4_K and `attn_v` Q6_K (1536²) with biases,
`ffn_up` Q4_K [1536 → 8960] and `ffn_down` Q6_K [8960 → 1536] with biases
and no gate; `v.pre_ln`/`v.post_ln`; the patch kernel F32
[14, 14, 3, 1536] without a bias; the position table F32 [1536, 1024], a
32×32 grid; the adapter `mm.0`/`mm.1`/`mm.2` BF16 (6144 → 4096 → 4096 →
6656) without biases. `bind` refuses any other encoding. The inventory is
`fixtures/muse-glimmer-mmproj.json`.

**Preprocessing.** Up to 4,096 tokens (16,384 patches). The grid is the
aspect-ratio-preserving size on 28-pixel tokens (`muse_glimmer.gridFor`):
scale the counts down to the cap keeping the ratio, try floor and ceil of
each side, keep the pair under the cap whose `h/w` is closest to the
image's (ties to more tokens). The image is stretched to the grid with the
Lanczos filter (support 3), not letterboxed, normalized on the host, and
rounded to F16; patches raster, channel-planar.

**The encoder.** Per patch: conv + the position table resized to the patch
grid bilinearly with half-pixel centres (edges clamped). The rows are then
put in window order (`Layout`): 32×32-patch windows in raster order,
patches raster inside each, edge windows partial; every block runs in that
order. `pre_ln`; fifty blocks of `x += W_o·attn(LN(x)·ln1 + b) + b`,
`x += W_down·gelu_erf(W_up·(LN(x)·ln2 + b) + b) + b`, with q/k/v biased,
2-D RoPE in the adjacent-pair form, base 1e4: pair j < 24 turns channels
(2j, 2j+1) by the patch's column + 1 at `10000^(−j/24)`, pair j ≥ 24 by its
row + 1, and bidirectional attention at scale 1/√96 over the window's rows
on 37 blocks and over all rows on the 13 global ones (blocks 3, 7, …, 47
and 49); the exact erf GELU. `post_ln`, back to raster order, then the
interleaved pixel shuffle: merged token (ox, oy) holds at element `c·4 + s`
channel c of patch (2ox + rx, 2oy + ry), s = 2ry + rx. Adapter: `mm.0`, erf
GELU, `mm.1`, erf GELU, `mm.2`.

**Indices each kernel receives.** The encoder's row index is the
window-order index r; the rotary position of row r is
`(order[r] % width + 1, order[r] / width + 1)`; a windowed block's keys are
its window's `[begin, begin + count)`, a global block's `[0, n)`; the
shuffle reads row `row_of[patch]`. The language model has plain positions.

**Executors.** The CPU reference (`muse_glimmer.Runtime`) decodes each
weight row once for all patch rows, F64 accumulation. The Metal plan
(`muse_glimmer_metal.zig`) runs the batched K-quant and BF16 matmul tiles
(the patch kernel zero-padded from 588 to 640 columns), LayerNorm, bias,
`ropeRows(.adjacent)`, `nu_gelu_erf_inplace` (|error| ≤ 1.5e-7), and the
encoder attention below; the shuffle runs on the host between two command
buffers. Activations are sized for 16,384 patches, about 1.35 GB.

**The language model.** The profile renders `<|image_start|>` (200080) +
`count × <|patch|>` (200092) + `<|image_end|>` (200081) before the user's
text, no newlines; a user cannot type the markers. The spans are causal:
both executors' `prefillVision` feed a span's rows the features in place of
the embedding (the Metal plan chunks text runs and spans separately), and
the weightless input norm applies to them as to token rows.

**Gates and limits.** The family is legacy and its vision gates are
retired; `fixtures/muse-glimmer-synthetic/patches.f32` stays for a unit
test of the patch embedding. A full-size image costs tens of seconds
before the first token; a lower `image_max_tokens` is the lever. Why its
rows track another implementation only up to block 33: [record,
2026-09-23](../benchmarks/record/vision.md#muse-glimmers-projector-2026-09-23).

## Encoder attention

The vision encoders' bidirectional attention runs on the segments kernels
(`Backend.attentionSegmentsGrouped`, which materialize the scores in F32;
[metal-attention.md § Segments attention with materialized scores](metal-attention.md#segments-attention-with-materialized-scores))
through `attention_metal.Spans`: one window of all rows (Qwen3-VL, both
Gemma SigLIP encoders, Muse Glimmer's global blocks) or per-row windows
(Muse's windowed blocks), with query tiles across the window boundaries
and the padding rows past the patches zero. The kernels take any head width
that is a multiple of 8. Each plan holds a 128 MB score buffer, and an
image's attention runs in as many chunks as its strips need. `make
test-metal` checks the encoders' shapes against F64 attention. In
`nuclis agent` an image is encoded while the prompt is typed
([terminal.md](../app/terminal.md)). Timings before and after the move:
[record, 2026-10-27](../benchmarks/record/vision.md#encoders-at-the-materialized-score-attention-2026-10-27).
