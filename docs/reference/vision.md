# Vision through the companion projectors

How nuclis turns an image into embedding rows the language model consumes,
family by family. The shared contract, the preprocessing, each projector's
tensor facts and provenance, the pinned traces, and the memory. The units
that built it are in the [engineering log](../engineering-log.md): `MODL-21`
(Qwen3.8), `AGNT-15` (the chat), `MODL-24` (decode after an image),
`MODL-22` (Gemma 4), and `MODL-23` (Muse Glimmer).

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
  BMP) behind the same C interface the Metal bridge uses. Bounds are host
  constants: image bytes ≤ 32 MiB, decoded pixels ≤ 64 M.
- `preprocess.zig` — the reference's smart size (`smartSize`: round each side
  to the patch·merge grid, then scale to the pixel bounds), a
  Pillow-compatible separable resize in 22-bit fixed point with the
  **bicubic** or the **Lanczos** filter (`resize`, `resizeBicubic`,
  `resizeLanczos`; `resizeLetterbox` for the `PAD_CEIL` letterbox), and
  `patches`, the channel-planar `[c][ky][kx]` layout the convolution reads,
  normalized by mean/std and rounded to F16 as the reference's im2col does.
- `qwen3vl.zig` / `qwen3vl_metal.zig` — the Qwen3-VL adapter (below).
- `gemma4.zig` / `gemma4_metal.zig` — Gemma 4's two adapters (below).
- `muse_glimmer.zig` / `muse_glimmer_metal.zig` — Muse Glimmer's adapter
  (below).
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
`locateImageSpans` finds the runs; `runLoop`'s `images` argument carries the
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
faster encode and prefill (Muse's timings below). A resumed chat re-encodes
its images at the current cap; the conversation is prefilled anew, so the
recorded grid binds nothing.

## The Qwen3-VL projector (MODL-21, 2026-09-22)

The file is `unsloth/Qwen3.8-27B-GGUF/mmproj-BF16.gguf` (931 MB; 224 F32 +
110 BF16 tensors), `clip.projector_type = qwen3vl_merger`. Read from the
pinned llama.cpp `7620399f5` (`tools/mtmd/models/qwen3vl.cpp`, `clip.cpp`,
`mtmd-image.cpp`, `mtmd.cpp`).

**Shapes.** 27 pre-LayerNorm blocks, hidden 1152, FFN 4304, 16 heads of 72,
patch 16, spatial merge 2, projection width 5120, image size 768, LayerNorm
eps 1e-6, mean/std 0.5. No deepstack layers in this file. Output tokens
`= (w/16/2) × (h/16/2)`.

**The graph.**

1. **Patch embedding.** A 3-D convolution with temporal patch 2; for a still
   image the two temporal slices `v.patch_embd.weight` and
   `.weight.1` are summed (`patchKernel` sums them into one `1152 ×
   768` F32 matrix). Then `+ v.patch_embd.bias`.
2. **Learned positions.** `v.position_embd.weight` is a 48×48 grid,
   bilinearly resized with corners aligned to the image's patch grid
   (`positionRows`, `ggml_interpolate`), added before the blocks.
3. **The merge walk.** Patch token `t` is patch `(x, y)` where blocks are
   raster order and inside a 2×2 block the order is top-left, top-right,
   bottom-left, bottom-right (`patchPosition`).
4. **Blocks.** LayerNorm, fused QKV (`+bias`), 2-D RoPE (pairs `[0,18)` turn
   with the patch's y, `[18,36)` with its x, at `base 1e4`, the reference's
   `GGML_ROPE_TYPE_VISION`), full bidirectional attention (scale `1/√72`),
   output projection, residual; LayerNorm, GELU (**tanh** form) FFN, residual.
5. **Post-norm and merger.** `v.post_ln`, then each merge block's four patch
   rows concatenated to 4608 through `mm.0` (GELU) and `mm.2` to 5120.

**M-RoPE on the language model.** An image span's rows carry the triple
`(t, t + i/nx, t + i%nx)` with `nx = width_tokens` and `t` the span's text
position; the span advances the text position by `max(width_tokens,
height_tokens)`. The rotary sections `[11, 11, 10]` interleave t, h, w over
the 32 rotary pairs (`cpu.rope.applyMultiAxis`, the Metal `fillChunkRope`
per-row table). Text rows carry `(p, p, p)`, bit-identical to the pre-vision
schedule, so the trace gates are unchanged. The language model's attention
stays causal by slot: the merged raster token order equals the reference's
2-D causal mask, so no mask change is needed.

**The oracle** is `scripts/reference-vision.cpp` against libmtmd
(`llama-mtmd-cli`'s API): it tokenizes `<__media__>describe this image`, runs
the projector, and dumps the feature rows, the decoder positions, and the
first greedy tokens. The pinned fixture is a synthetic 96×64 P6 image
(`inference/src/vision/fixtures/synthetic-96x64.ppm`); its reference-resized
128×96 pixels, feature rows, and first eight greedy tokens are committed
under `fixtures/qwen3vl-synthetic/` and `synthetic-96x64-resized.ppm`.

**The measurement (2026-09-22).**

| check | result |
| --- | --- |
| projector rows, CPU and Metal vs the pinned oracle | 0.39 max abs, 3.9e-3 relative RMS |
| the reference's own CPU vs Metal projector | 0.36 max abs, 4.6e-3 relative RMS |
| the Pillow bicubic letterbox resize | byte-identical to the reference |
| the 8 greedy tokens on the pinned feature rows, both executors | identical to the oracle (1919 2099 369 264 4145 11 46596 1452) |

The projector's F32-on-CPU and F16-on-Metal outputs sit within the spread the
reference itself shows between its two executors, so the greedy tokens are
gated on the pinned feature rows to isolate the language-model path from the
projector's tolerance; both executors reproduce the oracle's eight exactly.
On a live `generate --image` the projector's own rows can flip a close call
(the second token here), as the reference's two backends would.

**Memory.** The 768×768 warmup grid is 2,304 patches; the fixture's 128×96 is
48. The Metal plan wraps the file's weight bytes in place except the summed
patch kernel and each FFN down matrix (4,304 columns padded to the tile's
4,352); activations are sized for `max_patches = 4,096`.

**CLI.** `generate --image <path>` (repeatable, up to 8) attaches images to
the prompt; the model entry's `models.<name>.mmproj` names the projector,
loaded on demand. `nuclis model ls` still shows the projector as a companion.
The gates are `qwen38-vision-metal` (Metal tier) and `qwen38-vision-cpu`
(CPU tier), each `generation-check --vision-check`.

**Limits.** Deepstack projectors and dynamic `image_min/max_pixels`
overrides are not in this unit. The projector runs as its own command
buffer, separate from the language model's.

**Bonsai 2 (MODL-25, 2026-09-23).** Its `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf`
is this projector with the same metadata and tensors, re-encoded: the
matrices Q8_0, `ffn_down` F16 (its 4,304 columns are not a multiple of
Q8_0's 32-value block), vectors and the patch kernel F32. `bind` accepts
F32, F16, BF16, and Q8_0 matrices (the generic decoders on both
executors) and an element encoding for `ffn_down`, which the Metal plan
pads by element; the Gemma adapters keep F32/BF16. Bonsai's language
model is the Qwen3.8 adapter in a Hadamard-rotated basis: token rows are
rotated back after the embedding, and feature rows are fed as they are,
already in the model's basis. The aerial photo at the 1,024-token cap
(1,029 prompt tokens): encode 18.1 s, prefill 13.7 s, and the caption
names the turquoise water, the boulders, the snow-capped mountains, and
the evergreens on the right shore. No oracle trace is pinned for this
file.

## Images in the chat (AGNT-15, 2026-09-23)

**The path in.** A file dropped onto the terminal arrives as a bracketed
paste of its path (Terminal.app and iTerm2 backslash-escape spaces, some
terminals prefix `file://`); `editor.droppedPath` undoes that and the chat's
probe (`src/agent/root.zig` `dropProbe`) stats the file. An image by
extension becomes an `[image #N]` chip whose bytes in the buffer are the
marker and whose path waits in the editor's attachments; a UTF-8 text file
within the 128 KiB input limit becomes a `[file name, N lines]` chip over
its fenced content, so the model reads a file `read_file` cannot reach.
`/image <path>` and a typed image path at submit reach the same
`Editor.attachImage`. The editor never touches the file system: tests
supply the probe. At most 8 images per prompt; markers are single-digit, so
renumbering after a deletion rewrites bytes of equal length in place.

**The turn.** At submit the chat reads each image (`engine.readImage`,
32 MiB) and runs it through the loaded projector (`Model.encode_image`, the
completer's `Engine.encodeImage`), giving a `loop.Image` — path, file size,
the `PreparedImage` with its grid, decoded size, and feature rows. The
projector is loaded on the first attachment (`visionAvailable`), and an
entry without one, or one whose projector cannot be bound (Gemma's until
MODL-22, Muse's until MODL-23, Bonsai's until MODL-25), refuses the chip with a notice and
leaves the paste as text. A failure reading or encoding an image is a
notice naming it and the turn is not sent. The agent owns the images beside
the history item (`Item.images` and the profile's `ImageRef`s), frees them
with the item (compaction, a new session), and hands `Model.run` the images
of every rendered message in order. The completer counts the placeholder
runs (`<|image_pad|>`) in the remainder it prefills, pairs them with the
tail of that list — consumed text is a prefix of the render and compaction
drops whole earlier turns, so the images still rendered are always a
suffix — concatenates their rows, and calls `engine.complete` with an
`ImagePrefill`. The session's per-row rope positions carry the M-RoPE
advance across later turns without engine changes.

**Speculation.** `runLoop` runs an image prefill without the drafter, whose
cache is stale from then on; the completer therefore turns speculation off
for the session once an image has been fed (`images_fed`) and `reset`
restores it. No entry with a projector has speculation on (ENGN-17), so
nothing measurable changes today.

**What is shown and stored.** The transcript's user block gets one dim
detail row per image (`image #1: /path/shot.png (320×240 → 10×8 tokens)`)
and, where `tui.graphics.enabled` finds Ghostty or kitty in the environment
(`NUCLIS_NO_PREVIEW=1` disables it), a preview: the decoded image
box-filtered to a longest side of 512 px, sent as one kitty direct
transmission (`f=24`, base64 in 4096-byte chunks, `q=2`, `C=1`) over a box
of at most 12 rows, fewer when less room stands above the live region,
whose aspect assumes 1:2 cells. Not under tmux (`TMUX` set): tmux redraws
lines itself and cannot scroll a picture it does not know about, so a
passed-through image stayed put over the text (seen 2026-09-23, TERM-11);
the harness therefore shows the detail row only, and the preview is looked
at by running the chat in Ghostty directly. The session file records a user entry's images as
`{path, width, height, width_tokens, height_tokens}` and no pixels (a turn
without images writes the older line); `/save` lists them under the prompt;
a resumed session decodes and encodes each image again and leaves a
missing or unreadable one out of the model's view with a notice, the text
keeping its marker.

**The evidence (2026-09-23, `qwen3.8-27b`, Metal, ctx 16384).** A dropped
320×240 synthetic scene (sky, sun, red rectangle, green ground) became
`[image #1]` and a 10×8-token span; the caption named the sky, the sun, and
the red rectangle (the ground it called a black strip); `/image` attached
the same file as a chip after `/image nope.png` was refused; a dropped
`README.md` became `[file README.md, 219 lines]`; the Gemma 4 12B entry
refused the drop with `no vision support yet for this model: MissingMetadata
loading the projector` (it captions since MODL-22); and the captioned session resumed with its detail
row replayed and the follow-up `What colour is the rectangle?` answered
`red` from the re-encoded image. The captures are under `.zig-cache/tui/`
(`chip`, `caption`, `filechip`, `imagecmd`, `refusal`, `replay`,
`resumed`). The preview's sequence, box, and downscale are pinned by unit
tests; its rendering was first seen the next day in a Ghostty window
attached to the harness's tmux session (TERM-11), where it drew correctly
and then failed to follow the scrolling text, which is why tmux is excluded.

**Limits.** Clipboard images (Cmd-V of a bitmap) are not taken: reaching
the macOS pasteboard is a bridge call. PDF is not extracted. A conversation
that has fed an image runs without speculation. The preview assumes 1:2 cells, is
re-transmitted on a fold or resize replay, and is not drawn under tmux.

## Decode after an image (MODL-24, 2026-09-23)

**The defect.** The Metal plan's decode step passed one number to its
full-attention layers for three jobs: the rotary angle, the cache row the
new key and value are written to, and the number of rows attended. Outside
an image they are equal. After an image span they are not: a 32×32-token
span occupies 1,024 rows but advances the rotary position by 32
(`Session.ropePosition`), so every generated token after it was written 992
rows early — over the image's own rows — and attended only to the rows
before that. The prefill was right, so the prompt's last logits matched and
the answer began plausibly, then drifted: a real portrait was described
with its hair pulled back, light skin, and the window on the wrong side.
The CPU reference kept the row and the rotary position apart and was never
affected. The fix passes the cache row and the rotary position separately
(`qwen35_metal.zig` `fullAttention(…, row, rope_position)`).

**How it was found.** `test-generation -- <model> --metal --vision-check
<mmproj> --vision-image <file> --vision-oracle <dir>` runs the projector on
any image against an oracle directory from `scripts/reference-vision.cpp`
and reports the rows' relative RMS with a per-block map of the token grid
(7.0e-3 on the photo, within the reference's own CPU-to-Metal spread, no
region above 2.5e-2: the projector was right), the prompt's last logits on
the oracle's rows (max abs 0.13, argmax equal), the greedy tokens, and a
teacher-forced pass over the oracle's continuation — the tool that isolated
it: agreement was 141/183 with disagreements up to 5.4 logits, while one
batched prefill of the same tokens agreed at every one of those positions.
After the fix: 182/183 (the one left is a tokenization boundary at the
end of the cut text, where the reference itself gives the token a logit
of 4.2), and the greedy answer on the photo matches the reference CLI's.

**The guard.** The vision gate (`qwen38-vision-metal`, `-cpu`) now also
steps through its greedy tokens and compares the logits with one batched
prefill of the same tokens (bound 2e-2): 2.8e-3 with the fix, 0.82 with it
reverted. The 4×3-token fixture alone could not catch the defect (a shift
of 8 rows left the eight greedy tokens unchanged); the new check does.

## Gemma 4's projectors (MODL-22, 2026-09-23)

Two different companions, read from the pinned `7620399f5`
(`tools/mtmd/models/gemma4uv.cpp`, `gemma4v.cpp`, `clip.cpp` `build_vit`
and the GEMMA4V/GEMMA4UV hparams at 1627–1640, `mtmd.cpp` 877–883 and
`mtmd_decode_use_non_causal`, `src/models/gemma4.cpp`,
`src/llama-kv-cache.cpp` 1640–1760). Both place one token per 48×48 pixels
in raster order, within the reference's bounds of 70 and 1,120 tokens
(`set_limit_image_tokens(70, 1120)`), after the same smart size and PAD_CEIL
bicubic letterbox as Qwen's (align 48); `gemma4.gridFor`.

**The 12B: the unified embedder (`gemma4uv`).** `mmproj-BF16.gguf`, 175 MB:
ten vision tensors (the file's audio pair, `gemma4ua`, is not loaded).
`clip.cpp` turns the file's patch 16 into **48** with merge 1, so the
"patch" is a whole token. Per token: its 6,912 F32 pixel values (the
im2col keeps the input's F32; mean 0, std 1) → LayerNorm `patch_norm.1`
(eps **1e-5**, PyTorch's default, not the file's 1e-6) → `patch_embd`
(F32 3840×6912) + bias → LayerNorm `patch_norm.2` → + the x table's row
`i % columns` + the y table's row `i / columns` (`position_embd` F32
[3840, 1120, 2]) → LayerNorm `patch_norm.3` → RMS norm without a weight
(eps 1e-6) → `mm.input_projection` (BF16 3840×3840). No attention: the
language model does the vision work.

**The 26B-A4B: the SigLIP encoder (`gemma4v`).** `mmproj-BF16.gguf`,
1.19 GB, 356 tensors. 16-pixel patches of `2x − 1` rounded to F16 (the
convolution's im2col), the F32 patch kernel (no bias), + x/y tables
(`position_embd` [1152, 10240, 2]); 27 blocks, every norm RMS (eps 1e-6):
`x += rms(W_o · attn(h))·attn_post_norm` with h = rms(x)·ln1, q/k/v
without biases, per-head RMS of q and k with their 72-wide weights, **2-D
NEOX RoPE base 100** (channels [0,36) pairs (j, j+18) turn with the patch's
x, channels [36,72) with its y), v per-head RMS without a weight, and
bidirectional attention at scale **1**; then
`x += rms(W_down(gelu_quick(W_gate h) ⊙ W_up h))·ffn_post_norm` with
h = rms(x)·ln2. The gate is **`gelu_quick`** (x·σ(1.702x)): the file
carries no `clip.use_gelu`/`use_silu`, so the reference's default applies
(its log says `ffn_op: gelu_quick`); `bind` refuses a file that names one.
After the blocks: a 3×3 average pool over the patch grid, × √1152,
`(x − std_bias) ⊙ std_scale`, RMS norm without a weight,
`mm.input_projection` (BF16 1152×2816). The Metal plan runs the attention
through the chunk attention kernel with the whole patch set as one
bidirectional span, pools on the host between two command buffers, and
pads the FFN down matrices to 4,352 columns; activations are sized for
10,080 patches.

**The language model.** The profile renders `<|image>` (255999) +
`count × <|image|>` (258880) + `<image|>` (258882) before the user text, no
newlines. Image rows are not scaled by √width (only token rows are,
`gemma4.cpp:160`). The reference decodes an image as one ubatch with
`causal_attn = false`, and Gemma's `LLAMA_NON_CAUSAL_TYPE_SWA_ONLY` makes
that non-causal on the **sliding layers only**: a span row sees every row of
its span (the window never masks keys above it, since the reference masks
only `query − key ≥ window`) and the window below it; the global layers
stay causal. So:

- `metal.Backend.attentionChunk` takes `span { begin, end }`: chunk rows in
  it see keys up to `position + end` instead of their own row (the
  row-split body; the register-reuse body refuses a span). The Metal plan
  feeds a span as **one chunk** with the span on its sliding layers; the
  chunk buffers are grown to the largest image (`reserveRows`, on
  `loadVision` or the first larger span; the backend's new `release` frees
  the old set).
- The CPU runtime's `prefillVision` runs a span as one batched pass per
  layer: every row projects and writes its cache row, then each row
  attends over `[first, span end)` on a sliding layer and `[0, row]` on a
  global one.
- **The indices each kernel receives.** Gemma has no M-RoPE: the cache
  row, the rotary position, and the visible count are one value
  (`state.position` + row) on both executors, in the span and after it;
  the span only moves a sliding row's *last visible key* from its own row
  to the span's end. Decode after an image is the ordinary step.

**The oracle and the fixtures.** `scripts/reference-vision.cpp` gains
`--image-min-tokens`/`--image-max-tokens` and a 2,048-row ubatch (the
reference requires a non-causal image in one ubatch); build it with
`clang++ -std=c++17 -I<llama.cpp>/{include,ggml/include,tools/mtmd}
-L<llama.cpp>/build/bin -lmtmd -lllama -lggml -lggml-base
-Wl,-rpath,<llama.cpp>/build/bin`. The prompt is
`<bos><|turn>user\n<__media__>describe this image<turn|>\n<|turn>model\n<|channel>thought\n<channel|>`,
our rendering with thinking off. The synthetic 96×64 fixture at
`--image-min-tokens 4` becomes 144×96, a **3×2-token** span on both
entries, so the CPU tier stays in minutes; `fixtures/gemma4uv-synthetic/`
and `gemma4v-synthetic/` hold the reference's Metal rows, its prompt
tokens, its 16 best last-position logits, and its 8 greedy tokens. The
large real image is the system wallpaper
`/System/Library/Wallpapers/.default/DefaultAerial.jpg` (3840×2160, a lake
with boulders, snow-capped mountains, pine trees on the right shore):
44×25 = **1,100 tokens**, near the maximum; it is not committed.

**The measurement (2026-09-23, Metal and CPU).**

| check | 12B (`gemma4uv`) | 26B-A4B (`gemma4v`) |
| --- | --- | --- |
| the reference's own CPU vs Metal rows, fixture | 0.058 max abs, 1.2e-3 rel RMS | 0.092, 1.26e-2 |
| our rows vs the reference's, fixture, Metal / CPU | 6.3e-5, 1.0e-6 / 1.5e-5, 2.4e-7 | 6.2e-2, 6.6e-3 / 6.2e-2, 6.6e-3 |
| prompt tokens vs the oracle's | equal (24) | equal (24) |
| best-16 logits on the pinned rows, Metal / CPU | 0.121 / 0.123 (argmax equal) | 2.6e-3 / 2.7e-3 |
| 8 greedy tokens, both executors | identical | identical |
| decode after the image vs one prefill, Metal / CPU | 1.1e-2 / 0 | 6.7e-3 / 0 |
| planted fault: no bidirectional span | best-16 logits off by 1.80 | off by 0.41, greedy flips at 4 |
| photo: our rows vs the reference's Metal rows | 0.36, 7.8e-3 (no block above 1.1e-2) | 16.8, 0.30 (see below) |
| photo: last logits on the oracle's rows | 0.27, argmax equal | 0.14, argmax equal |
| photo: teacher-forced agreement, the oracle's rows | **120/121** (miss at margin 0.03) | **90/91** (margin 0.00) |
| photo: teacher-forced agreement, our rows | **120/121** (margin 0.11) | **89/91** (margins 0.46, 0.04) |

The CPU tier (`gemma4-vision-cpu` 12 min, `gemma4-26b-a4b-vision-cpu`
4.6 min) agrees with Metal on every line. In the chat (the harness,
`--think off`, captures `gemma-caption`, `gemma-followup`, and
`gemma12-caption` under `.zig-cache/tui/`), the photo dropped as
`[image #1]` became a 44×25-token span; the 26B-A4B described the lake,
the boulders, the mountains, and "a single tree on a rocky shoreline",
and answered the follow-up "What stands on the right shore?" with "A
single tree stands on the rocky shoreline on the right."; the 12B, which
refused the drop before this unit, described the same scene with "a few
pine trees visible on the right". The 12B embedder matches the
reference to F32 rounding on the fixture; on the photo its 7.8e-3 is the
reference's own Metal matmul precision at 1,100 rows.

**The SigLIP rows on a large image.** On the photo our 26B rows sit 0.30
relative RMS from the reference's Metal rows and 0.30 from its CPU rows,
while those two differ by 4.9e-2. Traced block by block
(`reference-vision --trace layer_out-N`), each of our blocks fed the
reference's own input reproduces its output to 9e-5–1.3e-3 relative, far
below the block's own change (0.11–0.51): the function is the same, and
the gap is amplification through 27 blocks (7e-3 after the embedding,
1.5e-2 at block 13, 0.25 at block 26, worst over the uniform sky). The
reference's two backends share a systematic rounding we do not: ggml
multiplies BF16 weights with activations converted to BF16 (its BF16 dot
type), where we keep F32 activations. The caption is unaffected: 89/91
teacher-forced with our rows, the same first 16 greedy tokens, and
`generate` describes the lake, the boulders, the mountains, and the pine
tree on the right shore.

**Speed (Metal, the photo, 1,122-token prompt, `generate --json`).** The
12B embedder encodes 1,100 tokens in 0.06 s; the 26B's SigLIP in **17.1 s**
(the reference: 10.6 s), of which 13.7 s is the chunk attention kernel over
9,900 72-wide rows at under 1 TFLOP/s. Prefill of the prompt: 6.8 s (12B),
2.9 s (26B-A4B); decode 26 and 66 tok/s.

**Memory.** The 12B plan: ~130 MB of activations; the 26B's: ~700 MB of
activations for 10,080 patches plus the padded down matrices (270 MB).
Reserving a 1,120-row chunk grows the language model's chunk buffers from
the prompt chunk (256 or 512 rows) to 1,120 (~0.3 MB per row).

**Gates.** `gemma4-vision-metal`, `gemma4-26b-a4b-vision-metal` (Metal
tier), `gemma4-vision-cpu`, `gemma4-26b-a4b-vision-cpu` (CPU tier): each
`generation-check --vision-check` against its fixture. The photo is a
manual diagnostic (`--vision-image <file> --vision-oracle <dir>`), which
now also teacher-forces the language model on our own rows.

**Limits.** The SigLIP attention is the slow part of a large image. Its
rows track the reference only as far as F32 activations track BF16 ones.
The 12B file's audio embedder is not loaded. A span must fit the context
(`ContextFull` otherwise).

## Muse Glimmer's projector (MODL-23, 2026-09-23)

Read from the pinned `7620399f5` (`tools/mtmd/models/muse-glimmer.cpp`,
`clip.cpp` `build_vit` and its MUSE_GLIMMER hparams at 1683–1692 and
`set_input` at 4582–4645, `mtmd-image.cpp` 1678–1739, `mtmd.cpp` 711–716,
`src/models/muse-glimmer.cpp` 73–74) and the file.

**The file.** `mmproj-kquant.gguf`, 1.40 GB, 809 tensors,
`clip.projector_type = muse-glimmer`, mean/std 0.5, LayerNorm eps
**1e-5**, patch 14, merge 2. Fifty blocks: `ln1`/`ln2` with biases,
`attn_q`/`attn_k`/`attn_out` **Q4_K** and `attn_v` **Q6_K** (1536²) with
biases, `ffn_up` Q4_K [1536 → 8960] and `ffn_down` Q6_K [8960 → 1536] with
biases and **no gate**; `v.pre_ln`/`v.post_ln`; the patch kernel F32
[14, 14, 3, 1536] without a bias; the position table F32 [1536, 1024], a
32×32 grid; the adapter `mm.0`/`mm.1`/`mm.2` BF16 (6144 → 4096 → 4096 →
6656) without biases. `bind` refuses any other encoding.
`muse_glimmer_patch_temporal = 2` is set by the reference and read
nowhere. The inventory is `fixtures/muse-glimmer-mmproj.json`.

**Preprocessing.** Up to **4,096 tokens (16,384 patches)**
(`set_limit_image_tokens(1, 4096)`; the file's `image_size` 896 is only
the reference's warm-up). The grid is transformers'
`get_aspect_ratio_preserving_size` on 28-pixel tokens
(`muse_glimmer.gridFor`): scale the counts down to the cap keeping the
ratio, try floor and ceil of each side, keep the pair under the cap whose
`h/w` is closest to the image's (ties to more tokens). The image is then
**stretched** to the grid with Pillow's **Lanczos** filter (support 3,
`sinc(x)·sinc(x/3)`), not letterboxed, normalized on the host, and rounded
to F16 by the im2col; patches raster, channel-planar. The synthetic
96×64 fixture becomes 84×56 (3×2 tokens); the aerial photo 2380×1344
(85×48 = 4,080 tokens), 588×336 at a 256 cap (21×12), 24×42 at 1,024.
Our patches equal the reference's im2col **bit for bit** on the fixture
(14,112 values, a unit test) and on the photo as a PNG (9.6 M values).

**The encoder.** Per patch: conv + the position table resized to the patch
grid **bilinearly with half-pixel centres** (`pixel_offset 0.5`, edges
clamped; Qwen's is corners-aligned). Then the rows are put in **window
order** (`Layout`): 32×32-patch windows in raster order, patches raster
inside each, edge windows partial; every block runs in that order.
`pre_ln`; fifty blocks of `x += W_o·attn(LN(x)·ln1 + b) + b`,
`x += W_down·gelu_erf(W_up·(LN(x)·ln2 + b) + b) + b`, with q/k/v biased,
**2-D RoPE in the GGUF normal (adjacent-pair) form**, base 1e4: pair
j < 24 turns channels (2j, 2j+1) by the patch's column + 1 at
`10000^(−j/24)`, pair j ≥ 24 by its row + 1 (positions 1-indexed from the
patch's grid cell, never from its row), and bidirectional attention at
scale 1/√96 over the **window's rows** on 37 blocks and over **all rows**
on the 13 global ones (block 3, 7, …, 47 and 49); the **exact erf GELU**.
`post_ln`, back to raster order, then the **interleaved pixel shuffle**:
merged token (ox, oy) holds at element `c·4 + s` channel c of patch
(2ox + rx, 2oy + ry), s = 2ry + rx (Qwen's merge concatenates, `s·1536 +
c`; checked against the reference's trace). Adapter: `mm.0`, erf GELU,
`mm.1`, erf GELU, `mm.2`.

**Indices each kernel receives.** The encoder's row index is the
window-order index r, used by every matmul, norm, and attention row; the
rotary position of row r is `(order[r] % width + 1, order[r] / width + 1)`;
a windowed block's keys are its window's `[begin, begin + count)`, a
global block's `[0, n)`; the shuffle reads row `row_of[patch]`. The
language model has plain positions: an image row's cache row, rotary
position, and visible count are `state.position` + row, as for text.

**Executors.** The CPU reference (`muse_glimmer.Runtime`) decodes each
weight row once for all patch rows, F64 accumulation. The Metal plan
(`muse_glimmer_metal.zig`) runs the batched K-quant and BF16 matmul tiles
(the patch kernel zero-padded from 588 to 640 columns), LayerNorm, bias,
`ropeRows(.adjacent)`, a new `nu_gelu_erf_inplace` (Abramowitz & Stegun
7.1.26, |error| ≤ 1.5e-7; the CPU's `erf` is a series and continued
fraction to ~1e-14), and the chunk attention kernel with a bidirectional
span: once per window on its row slice, in window order (a dispatch
stores up to seven padding rows into the next window's output, which that
window then overwrites), and once over all rows in a global block. The
shuffle runs on the host between two command buffers. Activations are
sized for 16,384 patches: ~1.35 GB (input, six 101 MB row buffers, the
587 MB FFN buffer reused for the adapter, the output).

**The language model.** The profile renders `<|image_start|>` (200080) +
`count × <|patch|>` (200092) + `<|image_end|>` (200081) before the user's
text, no newlines, as the reference writes around the rows (the GGUF
template's own image part, a single `<|patch|>`, is not what it renders);
a user cannot type the markers. The spans are causal: both executors'
`prefillVision` feed a span's rows the features in place of the embedding
(the Metal plan chunks text runs and spans separately), and the weightless
input norm applies to them as to token rows.

**The reference disagrees with itself.** On the synthetic fixture its CPU
and Metal encoders differ by 1.4e-2 relative RMS after block 0 and 1.8e-2
after block 30, then **0.63 after block 33**: there the FFN turns one or
two patches into *sinks* on channel 1082 (Metal: patches 0 and 17 at
about −400; CPU: patch 0 alone at −910), and which patches become sinks
flips with rounding (the CPU's Q8_K activations, the Metal's F16). Its
final rows differ by 0.26. So rows are compared tightly only before block
33, and the language model's output is the check after it. Its language
model also differs from itself: a batched prompt (F16-rounded activations
in the matmuls) against one row per decode moves the best 16 last logits
by up to 0.59 with identical rows and greedy tokens. The fixture pins the
**one-row** logits (`reference-vision --n-batch 1`, the path the text
traces pin); our CPU and Metal agree with it, not with the batched run.

**The oracle.** `scripts/reference-vision.cpp` gained `--n-ctx N`,
`--vision-flash` (the projector's attention as the reference's flash
attention; without it a global block over 16K patches materializes a 17 GB
score matrix), and `--n-batch N`. Prompt: our rendering with thinking off
(`low`), `<|begin_of_text|><|start|>system<|message|>You are a helpful AI
assistant.\nKnowledge cutoff: 2026-01-04.\n\nReasoning strength:
low.\n\n# Valid recipients: "self", "user".<|eot|><|start|>user<|message|><__media__>describe
this image<|eot|><|start|>assistant`. `fixtures/muse-glimmer-synthetic/`
holds the reference's Metal rows, `patches.f32` (its im2col),
`layer-out-30.f32` (the residual after block 30, window order), the prompt
tokens, the one-row run's best 16 logits, and 8 greedy tokens.

**Measured** (M4 Pro, 2026-09-23; `muse-vision-metal`, one CPU run before
the logits were re-pinned to the one-row run (26.9 s for the projector,
about 30 s per language-model token), and `generation-check --vision-image
… --vision-oracle …` on the photo). The `muse-vision-cpu` gate itself was
not run to the end:

| check | CPU | Metal |
| --- | ---: | ---: |
| fixture: patches vs the reference's im2col | bit-exact | bit-exact |
| fixture: residual after block 30 (its own spread 1.8e-2) | 1.8e-3 | 1.8e-3 |
| fixture: projector rows, reported (its own spread 0.26) | 0.12 | 0.11 |
| fixture: prompt tokens | equal | equal |
| fixture: best 16 last logits on the pinned rows | 0.587 from the batched run, as Metal's 0.586 | 6.9e-3 from the one-row run, argmax equal |
| fixture: 8 greedy tokens | not run | equal |
| decode after the image, 7 steps vs one prefill (bound 2e-2) | not run | 7.9e-3 |
| planted fault: the step's rotary position + 1 | — | 1.57, greedy still equal |
| photo (PNG): residual after `pre_ln` / block 2 / block 30 | — | 2.4e-6 / 8.6e-4 / 2.2e-3 |
| photo (PNG): projector rows | — | 2.2e-2 |
| photo (PNG): teacher-forced agreement, our rows | — | **198/200** (margins 0.06, 0.07) |
| photo (PNG): teacher-forced agreement, the oracle's rows | — | 197/200 |

**The JPEG.** On the photo as a JPEG our patches differ from the
reference's in 24 % of values (max 0.024, relative RMS 9e-3): we decode
with ImageIO, the reference with stb_image, and their chroma upsampling
and IDCT differ by a few levels. That is 1.9e-2 after `pre_ln`, and the
sink blocks grow it to 0.37 on the rows; teacher-forced agreement is then
193/200 (margins 0.01–0.77), and the caption is unaffected. On identical
pixels (the PNG) every stage matches.

**Speed** (Metal, the photo, `generate --json`, which now reports
`image_milliseconds`; ReleaseFast, F16 KV, quiet machine):

| cap | prompt tokens | encode | language-model prefill |
| ---: | ---: | ---: | ---: |
| 1,024 | 1,062 | 10.1 s | 12.4 s |
| 2,048 | 2,094 | 25.8 s | 24.5 s |
| 4,096 (auto) | 4,134 | 47 s (was 67) | 51.0 s |

Profiled at 16,320 patches (`generation-check --vision-profile`): the 13
global blocks' chunk attention is 27.9 s (the kernel's ~0.76 TFLOP/s on
96-wide heads, as Gemma's SigLIP measured), the 666 window dispatches 4.4 s
(24.5 s on the scalar `attention_full` kernel, which the plan first used),
the K-quant matmuls ~12 s. In the chat (the harness, captures
`muse-chip`, `muse-encoding`, `muse-caption`, `muse-followup` under
`.zig-cache/tui/`) the dropped JPEG became an 85×48-token span; Muse
described the turquoise water of Lake Tahoe, the granite boulders, the
snow-dusted mountains, and "a rocky outcrop with a solitary pine tree" on
the right, and answered "What stands on the right shore?" from the image
with 14 new prompt tokens (Muse keeps its session; nothing is replayed).

**Limits.** The global blocks' attention is the encoder's limiter; a
full-size image costs about 100 s before the first token (a lower
`image_max_tokens` is the lever). Speculation stays off for the rest of a
conversation once an image is in it (the drafter never saw the image
rows), so Muse's default-on drafter is idle there. Rows track the
reference only up to the sink blocks, and only on the same pixels.
