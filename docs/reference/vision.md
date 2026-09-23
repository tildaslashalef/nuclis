# Vision through the companion projectors

How nuclis turns an image into embedding rows the language model consumes,
family by family. The shared contract, the preprocessing, each projector's
tensor facts and provenance, the pinned traces, and the memory. The plan
that introduced it is [TODO.md](../../TODO.md)'s vision theme; the units are
`MODL-21` (Qwen3.8), `MODL-22` (Gemma 4), and `MODL-23` (Muse Glimmer).

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
  Pillow-compatible separable **bicubic** resize in 22-bit fixed point
  (`resizeBicubic`, `resizeLetterbox` for the `PAD_CEIL` letterbox), and
  `patches`, the channel-planar `[c][ky][kx]` layout the convolution reads,
  normalized by mean/std and rounded to F16 as the reference's im2col does.
- `qwen3vl.zig` / `qwen3vl_metal.zig` — the Qwen3-VL adapter (below).
- `Span { start, count, width_tokens, height_tokens }` — one image span; the
  engine's `locateImageSpans` pairs the runs of `<|image_pad|>` with the
  encoded images in order.

The prompt seam is in `engine.zig`: `Engine.loadVision(path)` loads a
projector on the engine's backend; `encodeImage(bytes)` decodes, preprocesses,
and runs the projector to `PreparedImage { width_tokens, height_tokens,
features }`; `profiles.Message.images` (a `[]const ImageRef`) makes the
profile render the family's markers with the right placeholder count;
`locateImageSpans` finds the runs; `runLoop`'s `images` argument carries the
spans and their concatenated features into `Model.prefillVision`. The Qwen
runtime substitutes a feature row for a token embedding per span row
(`stepImage`); the Metal plan copies the chunk's feature rows into `x_c`
instead of the embedding gather. Speculation is off on the vision prefill;
after it, decode is unchanged.

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
entry without one, or one whose projector cannot be bound (Gemma's
`gemma4uv` today: `MissingMetadata`), refuses the chip with a notice and
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
of at most 12 rows whose aspect assumes 1:2 cells, wrapped in the tmux DCS
passthrough under `TMUX`. The session file records a user entry's images as
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
loading the projector`; and the captioned session resumed with its detail
row replayed and the follow-up `What colour is the rectangle?` answered
`red` from the re-encoded image. The captures are under `.zig-cache/tui/`
(`chip`, `caption`, `filechip`, `imagecmd`, `refusal`, `replay`,
`resumed`). The preview's sequence, box, and downscale are pinned by unit
tests; its rendering was not photographed (the harness's Ghostty window
lookup needs the screen-recording permission this session lacks).

**Limits.** Clipboard images (Cmd-V of a bitmap) are not taken: reaching
the macOS pasteboard is a bridge call. PDF is not extracted. A conversation
that has fed an image runs without speculation. The preview assumes 1:2
cells and is re-transmitted on a fold or resize replay.
