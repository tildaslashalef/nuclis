//! GPU-resident execution of the qwen35 schedule at the binding's shape. One command
//! buffer per token: the CPU supplies a token ID and reads back logits (or a
//! greedy token); every activation, norm, gate, cache write, and recurrent
//! update stays on the GPU. The schedule is the same as `qwen35_runtime.zig`,
//! which remains the CPU reference these results are compared against. On a
//! rotated (Bonsai) file the embedding row is un-rotated after the gather and
//! every activation a rotated weight reads is transformed in place first
//! (`ssm_alpha` / `ssm_beta` read the residual before its transform).
//!
//! Ownership: the plan borrows the mapped weights and the `Backend`; it owns
//! the session and the GPU buffers it creates. Session memory is wrapped once
//! as a shared buffer, so `reset()` is a CPU memset that is only valid after
//! `commit()` — which the synchronous backend guarantees.
const std = @import("std");
const model = @import("qwen35.zig");
const weights = @import("../runtime/weights.zig");
const session = @import("../runtime/session.zig");
const sampling = @import("../sampling/root.zig");
const metal = @import("../backends/metal/root.zig");
const Observer = @import("qwen35_runtime.zig").Observer;
const Tensor = @import("../formats/gguf.zig").Tensor;
const Buffer = metal.Buffer;
const Drafter = @import("../runtime/draft.zig").Drafter;
const dflash2 = @import("dflash2.zig");

pub const vocabulary = 248320;
/// Rows a single verify batch may hold: the largest draft block plus its
/// seed token and bonus. Sizes the device buffers the output head reads back
/// (about 16 MB of logits), not the layer activations, which are chunk-sized.
pub const max_verify_rows = 16;
/// Rows the verify tape holds: the adapter's proposal bound plus the seed. A
/// longer batch updates the recurrent state in place, and recovery falls back
/// to rewind and replay.
pub const max_draft_rows = model.max_draft_proposals + 1;
/// One DeltaNet layer's verify tape: the batch rows' convolution inputs, then
/// the `deltaRows` tape.
fn tapeShape(shape: model.Shape) metal.Backend.DeltaRowsShape {
    return .{ .qheads = 16, .vheads = shape.value_heads, .keys = 128, .values = 128, .count = 0, .capacity = max_draft_rows, .in_stride = shape.qkvWidth(), .gate_stride = shape.value_heads, .out_stride = shape.innerWidth(), .scale = 1.0 / @sqrt(@as(f32, 128)) };
}
fn tapeMixedFloats(shape: model.Shape) usize {
    return max_draft_rows * shape.qkvWidth();
}
fn tapeLayerFloats(shape: model.Shape) usize {
    return tapeMixedFloats(shape) + metal.Backend.deltaTapeFloats(tapeShape(shape));
}

const LayerConstants = struct {
    attention_norm: Buffer,
    post_attention_norm: Buffer,
    mixer: union(enum) {
        full_attention: AttentionConstants,
        delta_net: DeltaConstants,
    },
};
const AttentionConstants = struct { query_norm: Buffer, key_norm: Buffer };
const DeltaConstants = struct { convolution: Buffer, a: Buffer, time_bias: Buffer, norm: Buffer };

/// A weight matrix ready for dispatch: its GPU range and CPU-side descriptor.
const Weight = struct { buffer: Buffer, matrix: @import("../backends/cpu/root.zig").Matrix };

/// The rotated file's transform operands, uploaded once at init: the sign
/// vector per width (`model.Rotation.widths` order) and, when the fold used
/// the grouped value-head order, the gather map with the regathered scratch
/// `ssm_out` reads (one decode row, `padded` prefill rows).
const Rotation = struct {
    signs: [3]Buffer,
    value_map: ?Buffer,
    regrouped: Buffer,
    regrouped_c: Buffer,

    fn signsFor(self: Rotation, width: usize) !Buffer {
        for (model.Rotation.widths, self.signs) |w, signs| if (w == width) return signs;
        return error.InvalidShape;
    }
};

/// One DFlash2 block's device constants.
const FlashConstants = struct {
    attention_norm: Buffer,
    query_norm: Buffer,
    key_norm: Buffer,
    ffn_norm: Buffer,
    attention_base: Buffer,
    ffn_base: Buffer,
};

/// The DFlash2 drafter's device workspace and constants (`dflash2.zig`; the
/// CPU reference is `dflash2_runtime.zig`). It owns no attention layout: its
/// five caches are the session layouts from `Plan.draft_layer`. Its rope
/// table is its own: 128-wide heads rotated whole, where the target rotates
/// 64 of 256. `capture` receives the five target-layer input residuals of a
/// recorded chunk (one slab per slot); `stage` takes the commit rows back.
const Flash = struct {
    binding: dflash2.Binding,
    constants: [dflash2.block_count]FlashConstants,
    encoder_norm: Buffer,
    output_norm: Buffer,
    rope_table: Buffer,
    stage: Buffer, // padded(chunk) × hidden_width
    encoded: Buffer, // padded(chunk) × embedding
    inject_k: Buffer, // padded(chunk) × kv_width
    inject_v: Buffer,
    capture: Buffer, // block_count × chunk × embedding
    // The noise block, padded to the matmul tile of `block_size` rows.
    x: Buffer,
    normalized: Buffer,
    taps: Buffer, // padded × conv_projection
    prepared: Buffer,
    sub: Buffer,
    finished: Buffer,
    q: Buffer, // padded × query_width
    k: Buffer, // padded × kv_width
    v: Buffer,
    mixed: Buffer, // padded × query_width
    gate: Buffer, // padded × feed_forward
    up: Buffer,
    hidden: Buffer, // padded × embedding, each row's final normed hidden
    projected: Buffer, // padded × selector_rank
    logits: Buffer, // padded × vocabulary
    partials: Buffer,
    topk: [dflash2.block_size]metal.Backend.TopKBuffers,
    /// Host-visible selector rows: one predecessor and the candidates' successors.
    predecessor: Buffer,
    successors: Buffer,

    fn init(plan: *Plan, binding: dflash2.Binding, chunk: usize, capacity: usize) !Flash {
        var result: Flash = undefined;
        result.binding = binding;
        const view = binding.view;
        for (&result.constants, binding.layers) |*c, layer| c.* = .{
            .attention_norm = try constantFrom(plan, view, layer.attention_norm),
            .query_norm = try constantFrom(plan, view, layer.query_norm),
            .key_norm = try constantFrom(plan, view, layer.key_norm),
            .ffn_norm = try constantFrom(plan, view, layer.ffn_norm),
            .attention_base = try constantFrom(plan, view, layer.attention_conv.base),
            .ffn_base = try constantFrom(plan, view, layer.ffn_conv.base),
        };
        result.encoder_norm = try constantFrom(plan, view, binding.encoder_norm);
        result.output_norm = try constantFrom(plan, view, binding.output_norm);
        const b = plan.backend;
        result.rope_table = try b.create(capacity * (dflash2.head_size / 2) * 8);
        try metal.Backend.ropeTable(result.rope_table, capacity, dflash2.head_size, dflash2.rope_base, null);
        const chunk_pad = metal.Backend.matmulPadded(chunk);
        const block_pad = metal.Backend.matmulPadded(dflash2.block_size);
        result.stage = try b.create(chunk_pad * dflash2.hidden_width * 4);
        result.encoded = try b.create(chunk_pad * dflash2.embedding * 4);
        result.inject_k = try b.create(chunk_pad * dflash2.kv_width * 4);
        result.inject_v = try b.create(chunk_pad * dflash2.kv_width * 4);
        result.capture = try b.create(dflash2.block_count * chunk * dflash2.embedding * 4);
        inline for (.{ "x", "normalized", "prepared", "sub", "finished", "hidden" }) |field| @field(result, field) = try b.create(block_pad * dflash2.embedding * 4);
        result.taps = try b.create(block_pad * dflash2.conv_projection * 4);
        result.q = try b.create(block_pad * dflash2.query_width * 4);
        result.k = try b.create(block_pad * dflash2.kv_width * 4);
        result.v = try b.create(block_pad * dflash2.kv_width * 4);
        result.mixed = try b.create(block_pad * dflash2.query_width * 4);
        result.gate = try b.create(block_pad * dflash2.feed_forward * 4);
        result.up = try b.create(block_pad * dflash2.feed_forward * 4);
        result.projected = try b.create(block_pad * dflash2.selector_rank * 4);
        result.logits = try b.create(block_pad * vocabulary * 4);
        result.partials = try b.create(metal.Backend.attentionDecodePartials(dflash2.heads, dflash2.head_size) * 4);
        for (&result.topk) |*scratch| scratch.* = try b.topkBuffers(dflash2.selector_top_k);
        result.predecessor = try b.create(dflash2.selector_rank * 4);
        result.successors = try b.create(dflash2.selector_top_k * dflash2.selector_rank * 4);
        return result;
    }

    /// A small F32 tensor from the companion's file in its own device buffer.
    fn constantFrom(plan: *Plan, view: weights.View, tensor: *const Tensor) !Buffer {
        const values = try view.vector(plan.alloc, tensor);
        defer plan.alloc.free(values);
        if (values.len == 0) return error.InvalidShape;
        const buffer = try plan.backend.create(values.len * 4);
        @memcpy(buffer.floats(), values);
        return buffer;
    }

    /// A batched projection whose weight lives in the companion's file. The
    /// Q8_0 file's few-row projections take Q8_0's fragment tile by name:
    /// `matmul` routes Q8_0 to the generic tile for every other family.
    fn mmRows(self: *Flash, backend: *metal.Backend, tensor: *const Tensor, input: Buffer, in_stride: usize, output: Buffer, out_stride: usize, rows: usize) !void {
        const matrix = try self.binding.view.matrix(tensor);
        const buffer = try backend.wrap(matrix.bytes);
        if (matrix.encoding == 8 and rows <= metal.Backend.small_chunk_tokens and buffer.offset % 2 == 0 and (matrix.bytes.len / matrix.rows) % 2 == 0)
            return backend.matmulKernel(.matmul_q8_0_f2, buffer, matrix, input, in_stride, output, out_stride, rows);
        try backend.matmul(buffer, matrix, input, in_stride, output, out_stride, rows);
    }

    /// The `slot`-th capture slab's `rows` rows.
    fn captureSlice(self: *Flash, slot: usize, rows: usize) Buffer {
        return self.capture.slice(slot * self.capture.len / dflash2.block_count, rows * dflash2.embedding * 4);
    }

    fn bytes(self: *const Flash) usize {
        var total: usize = self.rope_table.len + self.stage.len + self.encoded.len + self.inject_k.len + self.inject_v.len + self.capture.len +
            self.x.len * 6 + self.taps.len + self.q.len + self.k.len + self.v.len + self.mixed.len + self.gate.len + self.up.len +
            self.projected.len + self.logits.len + self.partials.len + self.predecessor.len + self.successors.len;
        for (self.topk) |scratch| total += scratch.partial_values.len + scratch.partial_indices.len + scratch.values.len + scratch.indices.len + scratch.sums.len + scratch.flags.len;
        return total;
    }
};

/// The DFlash2 capture slot of target layer `index`, or null.
fn targetSlot(index: usize) ?usize {
    inline for (dflash2.target_layers, 0..) |layer, slot| if (index == layer) return slot;
    return null;
}

/// A DFlash2 head's per-head norm over `rows` heads of 128.
fn flashHeadNorm(rows: usize) metal.Backend.Norm {
    return .{ .rows = rows, .width = dflash2.head_size, .in_stride = dflash2.head_size, .out_stride = dflash2.head_size, .eps = dflash2.rms_epsilon };
}
fn flashNorm(rows: usize) metal.Backend.Norm {
    return .{ .rows = rows, .width = dflash2.embedding, .in_stride = dflash2.embedding, .out_stride = dflash2.embedding, .eps = dflash2.rms_epsilon };
}

pub const Plan = struct {
    alloc: std.mem.Allocator,
    backend: *metal.Backend,
    view: weights.View,
    binding: model.Binding,
    state: session.Session,
    state_buffer: Buffer,
    rotation: ?Rotation,
    constants: []LayerConstants,
    output_norm: Buffer,
    rope_table: Buffer,
    chunk_rope: Buffer,
    /// Set for a chunk of image feature rows, cleared otherwise; recordLayers
    /// copies these rows into `x_c` instead of embedding tokens.
    image_rows: ?Buffer = null,
    /// Device staging for a chunk's uploaded feature rows.
    image_scratch: Buffer,
    // Activations (F32 counts in comments).
    x: Buffer, // hidden
    normalized: Buffer, // hidden
    projected: Buffer, // hidden
    gate: Buffer, // ffn
    up: Buffer, // ffn
    qg: Buffer, // heads * 512
    q: Buffer, // heads * 256
    /// F32 key/value rows of the current token when the cache is F16:
    /// projected, normalized, and rotated here, then packed into the slot.
    /// With an F32 cache the projections write the slot directly.
    k: Buffer, // 1024
    v: Buffer, // 1024
    mixed: Buffer, // qkv
    convolved: Buffer, // qkv
    z: Buffer, // inner
    alpha: Buffer, // value heads
    beta: Buffer, // value heads
    mixed_out: Buffer, // max(heads * 256, inner)
    /// Flash-decoding partials: `[rows][heads][splits][2 + 256]` for up to
    /// `verify_max_rows` rows (a decode step uses the first), 25 MB,
    /// independent of the capacity.
    partials: Buffer,
    logits: Buffer, // vocabulary
    /// A verify batch's output head over every row: `max_verify_rows`
    /// vocabulary rows, its per-row argmax results, and the post-`output_norm`
    /// hidden rows the drafter's `commit` consumes. `topk_rows` holds one
    /// partial top-k scratch set per row for the sampled acceptance path; the
    /// logits stay resident for `readVerifyRow`'s fallback.
    verify_logits: Buffer,
    verify_argmax: Buffer,
    verify_hidden: Buffer,
    /// `prefillRows`' head output (`matmulPadded(chunk) × vocabulary`, about
    /// 250 MB), created on its first call so no other load pays for it.
    rows_logits: ?Buffer = null,
    topk_rows: [max_verify_rows]metal.Backend.TopKBuffers,
    argmax_values: Buffer,
    argmax_indices: Buffer,
    argmax_result: Buffer,
    /// Partial top-k scratch for sampled decoding; `sampling.TopK.capacity` entries.
    topk: metal.Backend.TopKBuffers,
    /// The history bit set as device words for `nu_penalize`, uploaded when
    /// the history's revision changed (one 31 KB copy per change, not per
    /// step). `penalty_revision` is the last uploaded revision.
    penalty_history: Buffer,
    penalty_revision: u64,
    /// Chunked prefill: up to `chunk` prompt tokens per command buffer
    /// through the batched matmul. The `_c` buffers hold `padded` token rows
    /// ([token][feature]); `step` keeps its own single-token buffers above, so
    /// the two paths never alias.
    chunk: usize,
    padded: usize,
    x_c: Buffer, // padded × hidden
    normalized_c: Buffer,
    projected_c: Buffer,
    gate_c: Buffer, // padded × ffn
    up_c: Buffer,
    qg_c: Buffer, // padded × heads × 512
    q_c: Buffer, // padded × heads × 256
    k_c: Buffer, // padded × 1024
    v_c: Buffer,
    /// Half copy of `q_c` for the F16 chunk attention (its operands share one type).
    q_c_h: Buffer, // padded × heads × 256 halves
    /// A `q8_0` cache's visible rows expanded to F16 for the chunk kernel:
    /// keys then values, capacity × 1024 halves each; null otherwise.
    kv_stage: ?Buffer,
    mixed_c: Buffer, // padded × qkv
    convolved_c: Buffer,
    z_c: Buffer, // padded × inner
    alpha_c: Buffer, // padded × value heads
    beta_c: Buffer,
    mixed_out_c: Buffer, // padded × max(heads * 256, inner)
    /// Post-`output_norm` rows of a prefill chunk when `prefill` is asked for
    /// them; a drafter's `commit` consumes the same rows, so it exists only
    /// with the draft.
    /// Created on first use unless a drafter loads it at init.
    prefill_hidden: ?Buffer, // chunk × hidden

    /// The embedded prediction head's workspace, allocated only when a
    /// drafter was requested. `draft_h` is the block's `h_nextn` (after
    /// `shared_head_norm`); `draft_chain` is the block's own hidden chaining
    /// `propose` positions; `draft_pending_h` is the target hidden of the
    /// last committed token. The block's attention cache is one more layout
    /// in `state` (`draft_layer`), so checkpoint/rewind cover it.
    draft_layer: usize,
    /// A drafter is loaded (the verify paths and the tape exist); `mtp` says
    /// which: the prediction block, else the DFlash2 drafter in `flash`.
    has_draft: bool,
    mtp: bool,
    flash: ?Flash,
    draft_constants: ?LayerConstants,
    draft_enorm: Buffer,
    draft_hnorm_w: Buffer,
    draft_head_norm: Buffer,
    draft_h: Buffer,
    draft_hnorm: Buffer,
    draft_concat: Buffer,
    draft_chain: Buffer,
    draft_pending_h: Buffer,
    draft_logits: Buffer,
    /// Batched-commit scratch: each committed row's previous target hidden and
    /// the `[enorm(embed); hnorm(h_prev)]` pair `eh_proj` reads. Sized by the
    /// padded chunk (`eh_proj`'s matmul reads padded rows); a commit never
    /// overlaps a main forward, so it reuses the chunk activation buffers too.
    draft_hprev_c: Buffer, // padded × hidden
    draft_concat_c: Buffer, // padded × 2 × hidden
    /// The verify tape, one `tape_layer_floats` slice per DeltaNet layer in
    /// layer order (only with a drafter). A batch of at most `max_draft_rows`
    /// leaves the recurrent state at the batch's start and its rows pending
    /// (`Session.pending_rows`) until `replayRows` applies the kept prefix.
    tape: Buffer,

    /// `chunk` bounds the tokens one `prefill` command buffer processes (and
    /// sizes its activation buffers: about 0.4 MB per token). `kv` is the
    /// attention cache precision: `f16` halves cache memory and the
    /// bytes attention reads per token, `q8_0` about halves it again; the
    /// recurrent state stays F32.
    pub fn init(alloc: std.mem.Allocator, backend: *metal.Backend, view: weights.View, binding: model.Binding, capacity: usize, chunk: usize, kv: session.Precision, checkpoint: bool, draft: bool) !Plan {
        if (chunk == 0 or chunk > 4096) return error.InvalidShape;
        const flash = draft and binding.dflash != null;
        const mtp = draft and !flash;
        if (mtp and binding.draft == null) return error.NoDraftBlock;
        const shape = binding.shape;
        const hidden = shape.hidden;
        const ffn = shape.ffn;
        const heads = shape.heads;
        const value_heads = shape.value_heads;
        const qkv = shape.qkvWidth();
        const inner = shape.innerWidth();
        const mixed_out_width = @max(shape.queryWidth(), inner);
        const text = shape.layers;
        var layouts: [model.Shape.max_layers + dflash2.block_count]session.Layout = undefined;
        for (binding.layers(), layouts[0..text]) |layer, *layout| layout.* = switch (layer.mixer) {
            .full_attention => .{ .attention = .{ .key_row = 1024, .value_row = 1024, .precision = kv } },
            .delta_net => .{ .recurrent = .{ .history = shape.deltaHistory(), .matrix = shape.deltaMatrix() } },
        };
        var extra: usize = 0;
        if (mtp) {
            layouts[text] = .{ .attention = .{ .key_row = 1024, .value_row = 1024, .precision = kv } };
            extra = 1;
        }
        if (flash) {
            for (layouts[text..][0..dflash2.block_count]) |*layout| layout.* = .{ .attention = .{ .key_row = dflash2.kv_width, .value_row = dflash2.kv_width, .precision = kv } };
            extra = dflash2.block_count;
        }
        var state = try session.Session.init(alloc, layouts[0 .. text + extra], capacity, checkpoint);
        errdefer state.deinit();
        const constants = try alloc.alloc(LayerConstants, text);
        errdefer alloc.free(constants);
        var self: Plan = undefined;
        self.alloc = alloc;
        self.backend = backend;
        self.view = view;
        self.binding = binding;
        self.state = state;
        self.constants = constants;
        self.state_buffer = try backend.wrap(state.memory);
        for (binding.layers(), constants) |layer, *c| {
            c.* = .{
                .attention_norm = try self.constant(layer.attention_norm),
                .post_attention_norm = try self.constant(layer.post_attention_norm),
                .mixer = switch (layer.mixer) {
                    .full_attention => |attn| .{ .full_attention = .{ .query_norm = try self.constant(attn.query_norm), .key_norm = try self.constant(attn.key_norm) } },
                    .delta_net => |linear| .{ .delta_net = .{ .convolution = try self.constant(linear.convolution), .a = try self.constant(linear.a), .time_bias = try self.constant(linear.time_bias), .norm = try self.constant(linear.norm) } },
                },
            };
        }
        self.output_norm = try self.constant(binding.output_norm);
        self.rope_table = try backend.create(capacity * 32 * 8);
        try metal.Backend.ropeTable(self.rope_table, capacity, 64, 1e7, null);
        // A per-row rotary table for chunks that contain image spans: the
        // scalar-position table above cannot express the M-RoPE triples, so
        // when a span exists the chunk fills this and rotates at position 0.
        self.chunk_rope = try backend.create(chunk * 32 * 8);
        self.image_rows = null;
        self.rows_logits = null;
        self.image_scratch = try backend.create(chunk * hidden * 4);
        self.x = try backend.create(hidden * 4);
        self.normalized = try backend.create(hidden * 4);
        self.projected = try backend.create(hidden * 4);
        self.gate = try backend.create(ffn * 4);
        self.up = try backend.create(ffn * 4);
        self.qg = try backend.create(heads * 512 * 4);
        self.q = try backend.create(heads * 256 * 4);
        self.k = try backend.create(1024 * 4);
        self.v = try backend.create(1024 * 4);
        // One shared projection allocation lets four weights and all DeltaNet
        // outputs fit the merged kernel's seven data bindings. The named slices
        // preserve the downstream operation interfaces and borrowed lifetimes.
        const delta_projections = try backend.create((qkv + inner + 2 * value_heads) * 4);
        self.mixed = delta_projections.slice(0, qkv * 4);
        self.convolved = try backend.create(qkv * 4);
        self.z = delta_projections.slice(qkv * 4, inner * 4);
        self.alpha = delta_projections.slice((qkv + inner + value_heads) * 4, value_heads * 4);
        self.beta = delta_projections.slice((qkv + inner) * 4, value_heads * 4);
        self.mixed_out = try backend.create(mixed_out_width * 4);
        self.partials = try backend.create(metal.Backend.attentionVerifyPartials(metal.Backend.verify_max_rows, heads, 256) * 4);
        self.logits = try backend.create(vocabulary * 4);
        // Verify scratch exists only with a drafter (the loop's only caller).
        // `matmul` writes `matmulPadded(tokens)` output rows regardless of the
        // token count, so the logits buffer is sized by the tile, not the
        // batch: about 64 MB, kept out of a non-speculative load.
        if (draft) {
            self.verify_logits = try backend.create(metal.Backend.matmulPadded(max_verify_rows) * vocabulary * 4);
            self.verify_argmax = try backend.create(max_verify_rows * 4);
            self.verify_hidden = try backend.create(max_verify_rows * hidden * 4);
            for (&self.topk_rows) |*scratch| scratch.* = try backend.topkBuffers(sampling.TopK.capacity);
        } else {
            self.verify_logits = try backend.create(4);
            self.verify_argmax = try backend.create(4);
            self.verify_hidden = try backend.create(4);
        }
        self.argmax_values = try backend.create(metal.Backend.argmax_partials * 4);
        self.argmax_indices = try backend.create(metal.Backend.argmax_partials * 4);
        self.argmax_result = try backend.create(4);
        self.topk = try backend.topkBuffers(sampling.TopK.capacity);
        self.penalty_history = try backend.create(@divCeil(vocabulary, 32) * 4);
        self.penalty_revision = std.math.maxInt(u64);
        self.chunk = chunk;
        self.padded = metal.Backend.matmulPadded(chunk);
        const n = self.padded;
        self.x_c = try backend.create(n * hidden * 4);
        self.normalized_c = try backend.create(n * hidden * 4);
        self.projected_c = try backend.create(n * hidden * 4);
        self.gate_c = try backend.create(n * ffn * 4);
        self.up_c = try backend.create(n * ffn * 4);
        self.qg_c = try backend.create(n * heads * 512 * 4);
        self.q_c = try backend.create(n * heads * 256 * 4);
        self.k_c = try backend.create(n * 1024 * 4);
        self.v_c = try backend.create(n * 1024 * 4);
        self.q_c_h = try backend.create(n * heads * 256 * 2);
        self.kv_stage = if (kv == .q8_0) try backend.create(2 * capacity * 1024 * 2) else null;
        self.mixed_c = try backend.create(n * qkv * 4);
        self.convolved_c = try backend.create(n * qkv * 4);
        self.z_c = try backend.create(n * inner * 4);
        self.alpha_c = try backend.create(n * value_heads * 4);
        self.beta_c = try backend.create(n * value_heads * 4);
        self.mixed_out_c = try backend.create(n * mixed_out_width * 4);
        self.prefill_hidden = if (mtp) try backend.create(chunk * hidden * 4) else null;
        self.rotation = null;
        if (binding.rotation) |rotation| {
            var signs: [3]Buffer = undefined;
            for (&signs, rotation.signs) |*buffer, source| {
                if (source.values.len == 0) return error.InvalidShape;
                buffer.* = try backend.create(source.values.len * 4);
                for (buffer.floats(), source.values) |*out, value| out.* = @floatFromInt(value.signed);
            }
            var value_map: ?Buffer = null;
            if (rotation.value_grouped) {
                // Grouped head `nk * 3 + rep` takes the mixer's tiled head `rep * 16 + nk`.
                const map = try backend.create(48 * 4);
                for (@as([*]u32, @ptrCast(@alignCast(map.host)))[0..48], 0..) |*m, g| m.* = @intCast((g % 3) * 16 + g / 3);
                value_map = map;
            }
            self.rotation = .{ .signs = signs, .value_map = value_map, .regrouped = try backend.create(6144 * 4), .regrouped_c = try backend.create(n * 6144 * 4) };
        }
        self.has_draft = draft;
        self.mtp = mtp;
        self.flash = null;
        self.tape = undefined; // only taped verifies (a drafter) read it
        self.draft_layer = text;
        self.draft_constants = null;
        if (draft) {
            var delta_layers: usize = 0;
            for (binding.layers()) |layer| delta_layers += @intFromBool(layer.mixer == .delta_net);
            self.tape = try backend.create(delta_layers * tapeLayerFloats(shape) * 4);
        }
        if (flash) self.flash = try Flash.init(&self, binding.dflash.?, chunk, capacity);
        if (mtp) {
            const block = binding.draft.?;
            self.draft_enorm = try self.constant(block.enorm);
            self.draft_hnorm_w = try self.constant(block.hnorm);
            self.draft_head_norm = try self.constant(block.shared_head_norm);
            self.draft_constants = .{
                .attention_norm = try self.constant(block.layer.attention_norm),
                .post_attention_norm = try self.constant(block.layer.post_attention_norm),
                .mixer = .{ .full_attention = .{
                    .query_norm = try self.constant(block.layer.mixer.full_attention.query_norm),
                    .key_norm = try self.constant(block.layer.mixer.full_attention.key_norm),
                } },
            };
            self.draft_h = try backend.create(hidden * 4);
            self.draft_hnorm = try backend.create(hidden * 4);
            self.draft_concat = try backend.create(2 * hidden * 4);
            self.draft_chain = try backend.create(hidden * 4);
            self.draft_pending_h = try backend.create(hidden * 4);
            self.draft_logits = try backend.create(vocabulary * 4);
            self.draft_hprev_c = try backend.create(n * hidden * 4);
            self.draft_concat_c = try backend.create(n * 2 * hidden * 4);
            @memset(self.draft_pending_h.floats(), 0);
        }
        return self;
    }
    pub fn deinit(self: *Plan) void {
        // GPU buffers are released with the backend; the session and the
        // constants table are ours.
        self.alloc.free(self.constants);
        self.backend.unwrap(self.state.memory);
        self.state.deinit();
        self.* = undefined;
    }
    pub fn reset(self: *Plan) void {
        self.state.reset();
        if (self.mtp) @memset(self.draft_pending_h.floats(), 0);
    }

    /// The width of a drafter's commit row: the five target-layer input
    /// residuals for DFlash2, else the post-`output_norm` hidden.
    fn commitWidth(self: *const Plan) usize {
        return if (self.flash != null) dflash2.hidden_width else self.binding.shape.hidden;
    }

    /// Assembles the captured slabs into `count` row-major commit rows. Valid
    /// after the command buffer committed.
    fn readCapture(self: *Plan, count: usize, out: []f32) void {
        const d = &(self.flash orelse return);
        for (0..dflash2.block_count) |slot| {
            const slab = d.captureSlice(slot, count).floats();
            for (0..count) |row| @memcpy(out[row * dflash2.hidden_width + slot * dflash2.embedding ..][0..dflash2.embedding], slab[row * dflash2.embedding ..][0..dflash2.embedding]);
        }
    }

    /// Small F32 tensor copied into its own GPU buffer once at init.
    fn constant(self: *Plan, tensor: *const Tensor) !Buffer {
        const values = try self.binding.viewOf(self.view, tensor).vector(self.alloc, tensor);
        defer self.alloc.free(values);
        if (values.len == 0) return error.InvalidShape;
        const buffer = try self.backend.create(values.len * 4);
        @memcpy(buffer.floats(), values);
        return buffer;
    }
    /// Tensors are wrapped one at a time (cached by address). Wrapping the whole
    /// 16 GB mapping as one buffer was measured to make GPU time unstable by
    /// 3-4x between runs, presumably from per-command-buffer residency work over
    /// the entire range; per-tensor buffers are steady.
    fn weight(self: *Plan, tensor: *const Tensor) !Weight {
        const matrix = try self.binding.viewOf(self.view, tensor).matrix(tensor);
        return .{ .buffer = try self.backend.wrap(matrix.bytes), .matrix = matrix };
    }
    /// The GPU range of one session region (a byte range of the wrapped block).
    fn stateSlice(self: *Plan, region: []const u8) Buffer {
        return self.state_buffer.slice(self.state.offsetOf(region), region.len);
    }
    fn stateFloats(self: *Plan, values: []const f32) Buffer {
        return self.stateSlice(std.mem.sliceAsBytes(values));
    }
    fn mm(self: *Plan, tensor: *const Tensor, input: Buffer, output: Buffer) !void {
        const w = try self.weight(tensor);
        try self.backend.matvec(w.buffer, w.matrix, input, output);
    }
    /// On a rotated file, transforms `rows` rows of `width` (packed at that
    /// stride) in place into the basis the rotated weights were folded for;
    /// no dispatch on a plain file.
    fn rotate(self: *Plan, data: Buffer, width: usize, rows: usize) !void {
        const rotation = self.rotation orelse return;
        try self.backend.hadamard(data, try rotation.signsFor(width), width, rows, width, false);
    }
    /// The activation `ssm_out` reads: the mixer output as is on a plain file;
    /// on a rotated one, transformed in place, or first regathered into the
    /// scratch of its path (`chunk`: the prefill rows) from the tiled to the
    /// grouped head order when the fold used it (`qwen35_runtime.rotateGrouped`).
    /// A rotation exists only at the 27B shape, hence its literal widths.
    fn rotateGrouped(self: *Plan, input: Buffer, rows: usize, chunk: bool) !Buffer {
        const rotation = self.rotation orelse return input;
        const map = rotation.value_map orelse {
            try self.rotate(input, 6144, rows);
            return input;
        };
        const scratch = if (chunk) rotation.regrouped_c else rotation.regrouped;
        try self.backend.gatherRows(scratch, input, map, 128, 48, rows, 6144, 6144);
        try self.rotate(scratch, 6144, rows);
        return scratch;
    }

    /// Merge only shapes the backend can encode as whole threadgroups and
    /// encodings it does not decline. A rejected or declined merge records
    /// no work; the standalone path remains the fallback.
    fn projections(self: *Plan, tensors: []const *const Tensor, outputs: []const Buffer, mode: metal.Backend.SegmentMode) !void {
        std.debug.assert(tensors.len == outputs.len and tensors.len <= 4);
        var segments: [4]metal.Backend.Segment = undefined;
        for (tensors, outputs, 0..) |tensor, output, i| {
            const w = try self.weight(tensor);
            segments[i] = .{ .weights = w.buffer, .matrix = w.matrix, .output = if (mode == .silu_mul_pair) outputs[0] else output };
        }
        self.backend.matvecSegments(segments[0..tensors.len], self.normalized, mode) catch |err| switch (err) {
            error.InvalidShape, error.MergeDeclined => {
                for (segments[0..tensors.len], outputs) |s, output| try self.backend.matvec(s.weights, s.matrix, self.normalized, output);
                if (mode == .silu_mul_pair) try self.backend.siluMul(outputs[0], outputs[1], segments[0].matrix.rows);
            },
            else => return err,
        };
    }

    /// Records and runs one token. `logits` receives the vocabulary scores when
    /// non-null; `greedy` receives the argmax when non-null (computed on the
    /// GPU, no logit readback); `topk` receives the best `TopK.capacity`
    /// logits and the softmax denominator for `topk.temperature` (2 KB read
    /// back instead of 1 MB; `readLogits` still works afterwards when the
    /// sampler must fall back). An observer's `check` runs between recorded
    /// layers at no GPU cost; its `layer` callback forces a commit after every
    /// layer (64 command buffers per token) and is for numerical traces only.
    pub fn step(self: *Plan, token: u32, logits: ?[]f32, greedy: ?*u32, topk: ?*sampling.TopK, penalties: ?sampling.Penalties, observer: ?Observer) !void {
        const hidden = self.binding.shape.hidden;
        const ffn = self.binding.shape.ffn;
        if (token >= vocabulary) return error.InvalidTokenId;
        if (logits) |out| if (out.len != vocabulary) return error.InvalidShape;
        if (topk) |top| if (!std.math.isFinite(top.temperature) or top.temperature <= 0) return error.InvalidShape;
        if (penalties) |p| try self.syncPenalties(p);
        try self.settle();
        try self.state.begin();
        errdefer self.state.fail();
        const b = self.backend;
        try b.begin();
        errdefer b.discard();
        const embedding = try self.weight(self.binding.token_embedding);
        try b.embed(embedding.buffer, embedding.matrix, token, self.x);
        if (self.rotation) |rotation| try b.hadamard(self.x, try rotation.signsFor(hidden), hidden, 1, hidden, true);
        for (self.binding.layers(), self.constants, 0..) |layer, c, il| {
            try b.rmsNorm(self.x, c.attention_norm, self.normalized, .{ .rows = 1, .width = hidden, .in_stride = hidden, .out_stride = hidden });
            switch (layer.mixer) {
                .full_attention => |attn| try self.fullAttention(attn, c.mixer.full_attention, il, self.state.position, self.state.ropePosition(self.state.position)),
                .delta_net => |linear| try self.linearAttention(linear, c.mixer.delta_net, il),
            }
            try b.addRmsNorm(self.x, self.projected, c.post_attention_norm, self.normalized, .{ .rows = 1, .width = hidden, .in_stride = hidden, .out_stride = hidden });
            try self.rotate(self.normalized, hidden, 1);
            try self.projections(&.{ layer.ffn_gate, layer.ffn_up }, &.{ self.gate, self.up }, .silu_mul_pair);
            try self.rotate(self.gate, ffn, 1);
            try self.mm(layer.ffn_down, self.gate, self.projected);
            try b.add(self.x, self.projected, hidden);
            if (observer) |o| {
                if (o.check) |check| try check(o.context);
                if (o.layer) |report| {
                    try b.commit();
                    for (self.x.floats()) |v| if (!std.math.isFinite(v)) return error.NonFiniteResult;
                    try report(o.context, il, self.x.floats());
                    try b.begin();
                }
            }
        }
        if (logits != null or greedy != null or topk != null) {
            try b.rmsNorm(self.x, self.output_norm, self.normalized, .{ .rows = 1, .width = hidden, .in_stride = hidden, .out_stride = hidden });
            try self.recordOutputs(greedy, topk, penalties);
        }
        try b.commit();
        try self.readOutputs(logits, greedy, topk, penalties != null);
        try self.state.commit();
    }

    /// Uploads the history words to the device when the set changed since the
    /// last upload; valid before the command buffer's first dispatch (the
    /// shared-storage buffer is only read by the GPU at `commit`).
    fn syncPenalties(self: *Plan, p: sampling.Penalties) !void {
        if (p.history.revision == self.penalty_revision) return;
        const words = @divCeil(vocabulary, 32);
        const out = @as([*]u32, @ptrCast(@alignCast(self.penalty_history.host)))[0..words];
        if (p.history.writeWords(out) != words) return error.InvalidShape;
        self.penalty_revision = p.history.revision;
    }

    /// Records the output head for the normalized last-token row in
    /// `self.normalized` and whichever readbacks were requested. `penalties`
    /// applies `nu_penalize` to the logits in place before the argmax/top-k,
    /// so both see the vector the CPU sampler would sort.
    fn recordOutputs(self: *Plan, greedy: ?*u32, topk: ?*sampling.TopK, penalties: ?sampling.Penalties) !void {
        const hidden = self.binding.shape.hidden;
        const b = self.backend;
        try self.rotate(self.normalized, hidden, 1);
        try self.mm(self.binding.output, self.normalized, self.logits);
        // In place, so only a selection readback may request it: a raw
        // `logits` readback must stay unpenalized for a CPU sampler.
        if (penalties) |p| if (greedy != null or topk != null) try b.penalize(self.logits, vocabulary, self.penalty_history, p.repetition, p.presence);
        if (greedy != null) try b.argmax(self.logits, vocabulary, self.argmax_values, self.argmax_indices, self.argmax_result);
        if (topk) |top| try b.topk(self.logits, vocabulary, sampling.TopK.capacity, top.temperature, self.topk);
    }
    /// Copies the requested readbacks out after `commit`. `penalized` marks a
    /// `topk` readback taken after the device penalties, which is what lets
    /// `Sampler.selectFrom` decide with penalties active.
    fn readOutputs(self: *Plan, logits: ?[]f32, greedy: ?*u32, topk: ?*sampling.TopK, penalized: bool) !void {
        if (logits) |out| try self.readLogits(out);
        if (greedy) |out| {
            const id = @as(*const u32, @ptrCast(@alignCast(self.argmax_result.host))).*;
            if (id >= vocabulary) return error.NonFiniteResult;
            out.* = id;
        }
        if (topk) |top| {
            top.penalized = penalized;
            const ids = @as([*]const u32, @ptrCast(@alignCast(self.topk.indices.host)))[0..sampling.TopK.capacity];
            for (ids) |id| if (id >= vocabulary) return error.NonFiniteResult;
            @memcpy(top.ids[0..], ids);
            @memcpy(top.values[0..], self.topk.values.floats()[0..sampling.TopK.capacity]);
            top.count = sampling.TopK.capacity;
            // Partition order is fixed, so this F64 sum is deterministic run to run.
            var total: f64 = 0;
            for (self.topk.sums.floats()[0..metal.Backend.topk_partials]) |partial| total += partial;
            top.total = total;
            top.finite = true;
            for (@as([*]const u32, @ptrCast(@alignCast(self.topk.flags.host)))[0..metal.Backend.topk_partials]) |flag| if (flag != 0) {
                top.finite = false;
            };
        }
    }

    /// Chunked prefill: consumes `tokens` in chunks of at most `chunk`,
    /// one command buffer per chunk, with the projections and feed-forward
    /// batched through the matmul kernel, attention as one causal tiled
    /// dispatch per layer, and DeltaNet as one chunkwise dispatch per
    /// layer; nothing steps per token inside a chunk.
    /// Readbacks (`logits`, `greedy`, `topk`) refer to the last token, as in
    /// `step`. The observer's `check` runs between layers of every chunk;
    /// its `layer` callback is per token by contract and is not supported
    /// here (`error.InvalidShape`): trace through `step`. Arithmetic order
    /// differs from `step` (tile accumulation), so results agree within a
    /// tolerance, not bit for bit; `generation-check --metal` measures it.
    ///
    /// `hidden_rows`, when given (`tokens.len × hidden`), receives every
    /// row's post-`output_norm` hidden: the rows the drafter's `commit`
    /// consumes, or a decision head's input.
    /// Prefills a prompt with image spans: text runs and image spans are
    /// chunked separately (a span never straddles a chunk), each span
    /// registered in the session so its rows carry M-RoPE positions, and the
    /// span's rows fed the projector's feature rows (`features`,
    /// `Σ span.count × hidden`) instead of token embeddings. `logits`, when
    /// given, receives the last row's. Speculation is off on this path.
    pub fn prefillVision(self: *Plan, tokens: []const u32, spans: []const model.VisionSpan, features: []const f32, logits: ?[]f32, greedy: ?*u32, topk: ?*sampling.TopK, penalties: ?sampling.Penalties, observer: ?Observer) !void {
        return self.prefillSpans(tokens, spans, features, logits, greedy, topk, penalties, null, observer);
    }

    /// Runs `tokens` from position 0 of a fresh session and writes every
    /// row's post-`output_norm` hidden into `out` (`tokens.len × hidden`),
    /// never computing the output head: a decision head's input. `spans`
    /// and `features` place image rows as `prefillVision` does.
    pub fn prefillHidden(self: *Plan, tokens: []const u32, spans: []const model.VisionSpan, features: []const f32, out: []f32) !void {
        // Under DFlash2 a chunk's hidden rows are the drafter's residuals.
        if (self.flash != null) return error.HiddenUnsupported;
        self.reset();
        try self.prefillSpans(tokens, spans, features, null, null, null, null, out, null);
    }

    /// `prefillVision` that also writes every row's post-`output_norm`
    /// hidden into `hidden_rows` (`tokens.len × hidden`) when given.
    fn prefillSpans(self: *Plan, tokens: []const u32, spans: []const model.VisionSpan, features: []const f32, logits: ?[]f32, greedy: ?*u32, topk: ?*sampling.TopK, penalties: ?sampling.Penalties, hidden_rows: ?[]f32, observer: ?Observer) !void {
        const hidden = self.binding.shape.hidden;
        if (tokens.len == 0) return error.InvalidShape;
        if (hidden_rows) |h| if (h.len != tokens.len * hidden) return error.InvalidShape;
        if (observer) |o| if (o.layer != null) return error.InvalidShape;
        for (tokens) |t| if (t >= vocabulary) return error.InvalidTokenId;
        try self.settle();
        if (self.state.status != .ready) return error.SessionNotReady;
        if (tokens.len > self.state.capacity - self.state.position) return error.ContextFull;
        var i: usize = 0;
        var si: usize = 0;
        var frow: usize = 0;
        while (i < tokens.len) {
            if (si < spans.len and spans[si].start == i) {
                const sp = spans[si];
                if (frow + sp.count > features.len / hidden) return error.InvalidShape;
                try self.state.addSpan(.{ .row = self.state.position, .count = sp.count, .advance = @max(sp.width_tokens, sp.height_tokens), .columns = sp.width_tokens });
                var done: usize = 0;
                while (done < sp.count) {
                    const c = @min(self.chunk, sp.count - done);
                    const last = i + c == tokens.len;
                    @memcpy(self.image_scratch.floats()[0 .. c * hidden], features[(frow + done) * hidden ..][0 .. c * hidden]);
                    self.image_rows = self.image_scratch;
                    defer self.image_rows = null;
                    try self.prefillChunk(tokens[i..][0..c], if (last) logits else null, if (last) greedy else null, if (last) topk else null, if (last) penalties else null, rowsOf(hidden_rows, i, c, hidden), observer);
                    done += c;
                    i += c;
                }
                frow += sp.count;
                si += 1;
            } else {
                const run_end = if (si < spans.len) spans[si].start else tokens.len;
                while (i < run_end) {
                    const c = @min(self.chunk, run_end - i);
                    const last = i + c == tokens.len;
                    try self.prefillChunk(tokens[i..][0..c], if (last) logits else null, if (last) greedy else null, if (last) topk else null, if (last) penalties else null, rowsOf(hidden_rows, i, c, hidden), observer);
                    i += c;
                }
            }
        }
    }

    pub fn prefill(self: *Plan, tokens: []const u32, logits: ?[]f32, greedy: ?*u32, topk: ?*sampling.TopK, penalties: ?sampling.Penalties, hidden_rows: ?[]f32, observer: ?Observer) !void {
        if (tokens.len == 0) return error.InvalidShape;
        if (observer) |o| if (o.layer != null) return error.InvalidShape;
        for (tokens) |t| if (t >= vocabulary) return error.InvalidTokenId;
        if (logits) |out| if (out.len != vocabulary) return error.InvalidShape;
        if (topk) |top| if (!std.math.isFinite(top.temperature) or top.temperature <= 0) return error.InvalidShape;
        const width = self.commitWidth();
        if (hidden_rows) |h| if (h.len != tokens.len * width) return error.InvalidShape;
        // The whole prompt must fit: a prompt is never half-consumed.
        try self.settle();
        if (self.state.status != .ready) return error.SessionNotReady;
        if (tokens.len > self.state.capacity - self.state.position) return error.ContextFull;
        var offset: usize = 0;
        while (offset < tokens.len) {
            const count = @min(self.chunk, tokens.len - offset);
            const last = offset + count == tokens.len;
            try self.prefillChunk(tokens[offset..][0..count], if (last) logits else null, if (last) greedy else null, if (last) topk else null, if (last) penalties else null, rowsOf(hidden_rows, offset, count, width), observer);
            offset += count;
            // The whole prompt is one `step` for the loop's hooks, so this is
            // the only place a caller can learn how far a long prefill has got.
            if (observer) |o| if (o.progress) |call| try call(o.context, .{ .phase = .prefill, .position = offset, .target = tokens.len });
        }
    }

    /// Rows `[start, start + count)` of an optional `[rows][width]` buffer.
    fn rowsOf(rows: ?[]f32, start: usize, count: usize, width: usize) ?[]f32 {
        const all = rows orelse return null;
        return all[start * width ..][0 .. count * width];
    }

    fn prefillChunk(self: *Plan, tokens: []const u32, logits: ?[]f32, greedy: ?*u32, topk: ?*sampling.TopK, penalties: ?sampling.Penalties, hidden_rows: ?[]f32, observer: ?Observer) !void {
        const hidden = self.binding.shape.hidden;
        const count = tokens.len;
        std.debug.assert(count >= 1 and count <= self.chunk);
        const want_outputs = logits != null or greedy != null or topk != null;
        const capture = hidden_rows != null and self.flash != null;
        const staging: ?Buffer = if (hidden_rows == null or capture) null else self.prefill_hidden orelse blk: {
            const created = try self.backend.create(self.chunk * hidden * 4);
            self.prefill_hidden = created;
            break :blk created;
        };
        if (penalties) |p| try self.syncPenalties(p);
        try self.state.beginChunk(count);
        errdefer self.state.fail();
        const b = self.backend;
        try b.begin();
        errdefer b.discard();
        try self.recordLayers(tokens, count, observer, false, capture);
        if (hidden_rows != null and !capture) {
            // Normalize every row, as `verify` does; the last row also serves
            // the output head when a readback was asked for.
            try b.rmsNorm(self.x_c, self.output_norm, self.normalized_c, .{ .rows = count, .width = hidden, .in_stride = hidden, .out_stride = hidden });
            try b.copy(staging.?, self.normalized_c, count * hidden);
            if (want_outputs) {
                try b.copy(self.normalized, self.normalized_c.slice((count - 1) * hidden * 4, hidden * 4), hidden);
                try self.recordOutputs(greedy, topk, penalties);
            }
        } else if (want_outputs) {
            const last_row = self.x_c.slice((count - 1) * hidden * 4, hidden * 4);
            try b.rmsNorm(last_row, self.output_norm, self.normalized, .{ .rows = 1, .width = hidden, .in_stride = hidden, .out_stride = hidden });
            try self.recordOutputs(greedy, topk, penalties);
        }
        try b.commit();
        try self.readOutputs(logits, greedy, topk, penalties != null);
        try self.state.commitChunk(count);
        if (hidden_rows) |dest| {
            if (capture) self.readCapture(count, dest) else @memcpy(dest, staging.?.floats()[0 .. count * hidden]);
        }
    }

    /// Evaluation prefill: consumes `tokens` in chunks, as `prefill` does, and
    /// runs the output head over every row, writing each row's logits to
    /// `rows` (`tokens.len × vocabulary`). Admitted whole before the first
    /// write.
    pub fn prefillRows(self: *Plan, tokens: []const u32, rows: []f32, observer: ?Observer) !void {
        const hidden = self.binding.shape.hidden;
        if (tokens.len == 0 or rows.len != tokens.len * vocabulary) return error.InvalidShape;
        if (observer) |o| if (o.layer != null) return error.InvalidShape;
        for (tokens) |t| if (t >= vocabulary) return error.InvalidTokenId;
        try self.settle();
        if (self.state.status != .ready) return error.SessionNotReady;
        if (tokens.len > self.state.capacity - self.state.position) return error.ContextFull;
        const b = self.backend;
        const out = self.rows_logits orelse try b.create(metal.Backend.matmulPadded(self.chunk) * vocabulary * 4);
        self.rows_logits = out;
        const head = try self.weight(self.binding.output);
        var offset: usize = 0;
        while (offset < tokens.len) {
            const count = @min(self.chunk, tokens.len - offset);
            try self.state.beginChunk(count);
            errdefer self.state.fail();
            try b.begin();
            errdefer b.discard();
            try self.recordLayers(tokens[offset..][0..count], count, observer, false, false);
            try b.rmsNorm(self.x_c, self.output_norm, self.normalized_c, .{ .rows = count, .width = hidden, .in_stride = hidden, .out_stride = hidden });
            try self.rotate(self.normalized_c, hidden, count);
            try b.matmul(head.buffer, head.matrix, self.normalized_c, hidden, out, vocabulary, count);
            try b.commit();
            @memcpy(rows[offset * vocabulary ..][0 .. count * vocabulary], out.floats()[0 .. count * vocabulary]);
            try self.state.commitChunk(count);
            offset += count;
            if (observer) |o| if (o.progress) |call| try call(o.context, .{ .phase = .prefill, .position = offset, .target = tokens.len });
        }
    }

    /// Records the embedding gather and every decoder layer over one admitted
    /// chunk of `count` tokens, leaving the final hidden rows in `x_c`. The
    /// caller owns the command buffer (`begin`/`commit`) and the state
    /// admission (`beginChunk`/`commitChunk`); `prefillChunk` and `verify`
    /// share it so the schedule is written once.
    /// `capture` copies each DFlash2 target layer's input rows into the
    /// drafter's slab for that slot.
    fn recordLayers(self: *Plan, tokens: []const u32, count: usize, observer: ?Observer, taped: bool, capture: bool) !void {
        const hidden = self.binding.shape.hidden;
        const ffn = self.binding.shape.ffn;
        const b = self.backend;
        if (self.image_rows) |features| {
            // A chunk of image span rows: the projector's feature rows are
            // already in the model's width and basis, so they are copied in
            // (no embed, no rotation), exactly the CPU runtime's `stepImage`.
            try b.copy(self.x_c.slice(0, count * hidden * 4), features.slice(0, count * hidden * 4), count * hidden);
        } else {
            const embedding = try self.weight(self.binding.token_embedding);
            for (tokens, 0..) |token, t| try b.embed(embedding.buffer, embedding.matrix, token, self.x_c.slice(t * hidden * 4, hidden * 4));
            if (self.rotation) |rotation| try b.hadamard(self.x_c, try rotation.signsFor(hidden), hidden, count, hidden, true);
        }
        const norm: metal.Backend.Norm = .{ .rows = count, .width = hidden, .in_stride = hidden, .out_stride = hidden };
        for (self.binding.layers(), self.constants, 0..) |layer, c, il| {
            if (capture) if (targetSlot(il)) |slot| try b.copy(self.flash.?.captureSlice(slot, count), self.x_c, count * hidden);
            try b.rmsNorm(self.x_c, c.attention_norm, self.normalized_c, norm);
            switch (layer.mixer) {
                .full_attention => |attn| try self.attentionChunk(attn, c.mixer.full_attention, il, count, self.state.position),
                .delta_net => |linear| try self.deltaChunk(linear, c.mixer.delta_net, il, count, taped),
            }
            try b.add(self.x_c, self.projected_c, count * hidden);
            try b.rmsNorm(self.x_c, c.post_attention_norm, self.normalized_c, norm);
            try self.rotate(self.normalized_c, hidden, count);
            try self.mmRows(layer.ffn_gate, self.normalized_c, hidden, self.gate_c, ffn, count);
            try self.mmRows(layer.ffn_up, self.normalized_c, hidden, self.up_c, ffn, count);
            try b.siluMul(self.gate_c, self.up_c, count * ffn);
            try self.rotate(self.gate_c, ffn, count);
            try self.mmRows(layer.ffn_down, self.gate_c, ffn, self.projected_c, hidden, count);
            try b.add(self.x_c, self.projected_c, count * hidden);
            if (observer) |o| if (o.check) |check| try check(o.context);
        }
    }

    /// Verifies a batch: records the layer stack once and computes the output
    /// head for every row. Exactly one of `rows` (`tokens.len × vocabulary`,
    /// the full logits of each position) and `tops` (one `sampling.TopK`
    /// per row, read back from the device partial top-k with every row's
    /// `temperature`) is given; the logits stay resident either way, so
    /// `readVerifyRow` can serve a row the readback could not decide.
    /// `h_rows` (`tokens.len × hidden`), when given, receives the rows'
    /// post-`output_norm` hidden. The whole batch is admitted before the
    /// first write, as `prefill` is.
    pub fn verify(self: *Plan, tokens: []const u32, rows: ?[]f32, tops: ?[]sampling.TopK, h_rows: ?[]f32, observer: ?Observer) !void {
        const hidden = self.binding.shape.hidden;
        if (!self.has_draft) return error.NoDraftBlock;
        if (observer) |o| if (o.layer != null) return error.InvalidShape;
        if (tokens.len == 0 or tokens.len > self.chunk or tokens.len > max_verify_rows) return error.InvalidShape;
        if (rows == null and tops == null) return error.InvalidShape;
        if (rows != null and tops != null) return error.InvalidShape;
        if (rows) |r| if (r.len != tokens.len * vocabulary) return error.InvalidShape;
        if (tops) |t| {
            if (t.len != tokens.len) return error.InvalidShape;
            for (t) |top| if (!std.math.isFinite(top.temperature) or top.temperature <= 0) return error.InvalidShape;
        }
        if (h_rows) |h| if (h.len != tokens.len * self.commitWidth()) return error.InvalidShape;
        for (tokens) |t| if (t >= vocabulary) return error.InvalidTokenId;
        try self.settle();
        if (self.state.status != .ready) return error.SessionNotReady;
        if (tokens.len > self.state.capacity - self.state.position) return error.ContextFull;
        const count = tokens.len;
        try self.state.beginChunk(count);
        errdefer self.state.fail();
        const b = self.backend;
        const taped = count <= max_draft_rows;
        const capture = h_rows != null and self.flash != null;
        try b.begin();
        errdefer b.discard();
        try self.recordLayers(tokens, count, observer, taped, capture);
        try b.rmsNorm(self.x_c, self.output_norm, self.normalized_c, .{ .rows = count, .width = hidden, .in_stride = hidden, .out_stride = hidden });
        if (h_rows != null and !capture) try b.copy(self.verify_hidden, self.normalized_c, count * hidden);
        try self.rotate(self.normalized_c, hidden, count);
        const head = try self.weight(self.binding.output);
        try b.matmul(head.buffer, head.matrix, self.normalized_c, hidden, self.verify_logits, vocabulary, count);
        if (tops) |out| {
            for (out, 0..) |top, i| try b.topk(self.verify_logits.slice(i * vocabulary * 4, vocabulary * 4), vocabulary, sampling.TopK.capacity, top.temperature, self.topk_rows[i]);
        }
        try b.commit();
        if (h_rows) |h| {
            if (capture) self.readCapture(count, h) else @memcpy(h, self.verify_hidden.floats()[0 .. count * hidden]);
        }
        if (rows) |r| @memcpy(r, self.verify_logits.floats()[0 .. count * vocabulary]);
        if (tops) |out| {
            for (out, 0..) |*top, i| try self.readVerifyTopK(i, top);
        }
        try self.state.commitChunk(count);
        if (taped) try self.state.deferRows(count);
    }

    /// One verify row's partial top-k readback from its own scratch set. The
    /// values are the raw logits (the penalties, when active, are the
    /// sampler's on the host and `selectFromHistory` checks the bound).
    fn readVerifyTopK(self: *Plan, row: usize, top: *sampling.TopK) !void {
        const scratch = self.topk_rows[row];
        const ids = @as([*]const u32, @ptrCast(@alignCast(scratch.indices.host)))[0..sampling.TopK.capacity];
        for (ids) |id| if (id >= vocabulary) return error.NonFiniteResult;
        @memcpy(top.ids[0..], ids);
        @memcpy(top.values[0..], scratch.values.floats()[0..sampling.TopK.capacity]);
        top.count = sampling.TopK.capacity;
        var total: f64 = 0;
        for (scratch.sums.floats()[0..metal.Backend.topk_partials]) |partial| total += partial;
        top.total = total;
        top.finite = true;
        for (@as([*]const u32, @ptrCast(@alignCast(scratch.flags.host)))[0..metal.Backend.topk_partials]) |flag| if (flag != 0) {
            top.finite = false;
        };
        top.penalized = false;
    }

    /// Copies one row of the last `verify`'s logits out of the shared buffer:
    /// the fallback for a row whose readback could not decide. Valid until
    /// the next verify.
    pub fn readVerifyRow(self: *Plan, row: usize, out: []f32) !void {
        if (row >= max_verify_rows or out.len != vocabulary) return error.InvalidShape;
        @memcpy(out, self.verify_logits.floats()[row * vocabulary ..][0..vocabulary]);
        for (out) |v| if (!std.math.isFinite(v)) return error.NonFiniteResult;
    }

    /// `verify`'s greedy sibling: per-row argmax read back (four bytes a row),
    /// no logit readback. `out` holds `tokens.len` token ids.
    pub fn verifyGreedy(self: *Plan, tokens: []const u32, out: []u32, h_rows: ?[]f32, observer: ?Observer) !void {
        const hidden = self.binding.shape.hidden;
        if (!self.has_draft) return error.NoDraftBlock;
        if (observer) |o| if (o.layer != null) return error.InvalidShape;
        if (tokens.len == 0 or tokens.len > self.chunk or tokens.len > max_verify_rows) return error.InvalidShape;
        if (out.len != tokens.len) return error.InvalidShape;
        if (h_rows) |h| if (h.len != tokens.len * self.commitWidth()) return error.InvalidShape;
        for (tokens) |t| if (t >= vocabulary) return error.InvalidTokenId;
        try self.settle();
        if (self.state.status != .ready) return error.SessionNotReady;
        if (tokens.len > self.state.capacity - self.state.position) return error.ContextFull;
        const count = tokens.len;
        try self.state.beginChunk(count);
        errdefer self.state.fail();
        const b = self.backend;
        const taped = count <= max_draft_rows;
        const capture = h_rows != null and self.flash != null;
        try b.begin();
        errdefer b.discard();
        try self.recordLayers(tokens, count, observer, taped, capture);
        try b.rmsNorm(self.x_c, self.output_norm, self.normalized_c, .{ .rows = count, .width = hidden, .in_stride = hidden, .out_stride = hidden });
        if (h_rows != null and !capture) try b.copy(self.verify_hidden, self.normalized_c, count * hidden);
        try self.rotate(self.normalized_c, hidden, count);
        const head = try self.weight(self.binding.output);
        try b.matmul(head.buffer, head.matrix, self.normalized_c, hidden, self.verify_logits, vocabulary, count);
        for (0..count) |i| try b.argmax(self.verify_logits.slice(i * vocabulary * 4, vocabulary * 4), vocabulary, self.argmax_values, self.argmax_indices, self.verify_argmax.slice(i * 4, 4));
        try b.commit();
        if (h_rows) |h| {
            if (capture) self.readCapture(count, h) else @memcpy(h, self.verify_hidden.floats()[0 .. count * hidden]);
        }
        const ids = @as([*]const u32, @ptrCast(@alignCast(self.verify_argmax.host)))[0..count];
        for (ids, out) |id, *token| {
            if (id >= vocabulary) return error.NonFiniteResult;
            token.* = id;
        }
        try self.state.commitChunk(count);
        if (taped) try self.state.deferRows(count);
    }

    /// Batched projection over `count` token rows.
    fn mmRows(self: *Plan, tensor: *const Tensor, input: Buffer, in_stride: usize, output: Buffer, out_stride: usize, count: usize) !void {
        const w = try self.weight(tensor);
        try self.backend.matmul(w.buffer, w.matrix, input, in_stride, output, out_stride, count);
    }

    /// Applies the first `kept` rows of the last verify's tape to the
    /// recurrent layers (the convolution history from the rows' inputs, the
    /// matrix by `deltaReplay`) and settles the session at the batch's start
    /// plus `kept`. `kept` is at most the pending rows; zero replays nothing.
    /// A failed replay leaves the session failed, as a failed forward does.
    pub fn replayRows(self: *Plan, kept: usize) !void {
        if (self.state.pending_rows == 0 or kept > self.state.pending_rows) return error.InvalidShape;
        if (kept > 0) {
            errdefer self.state.fail();
            const b = self.backend;
            try b.begin();
            errdefer b.discard();
            var shape = tapeShape(self.binding.shape);
            shape.count = kept;
            const qkv = self.binding.shape.qkvWidth();
            for (self.binding.layers(), 0..) |layer, il| if (layer.mixer == .delta_net) {
                const state = self.state.layers[il].recurrent;
                const tape = self.tapeLayer(il);
                const history = self.stateFloats(state.history);
                try b.convolutionHistory(history, tape.mixed, qkv, 4, kept, qkv);
                try b.deltaReplay(self.stateFloats(state.matrix), tape.delta, shape);
            };
            try b.commit();
        }
        try self.state.settleRows(kept);
    }
    /// Replays every pending row: what a forward needs before it reads or
    /// extends the state, when no `recover` chose a shorter prefix.
    fn settle(self: *Plan) !void {
        if (self.state.pending_rows > 0) try self.replayRows(self.state.pending_rows);
    }
    /// DeltaNet layer `il`'s slices of `tape`.
    fn tapeLayer(self: *Plan, il: usize) struct { mixed: Buffer, delta: Buffer } {
        var index: usize = 0;
        for (self.binding.layers()[0..il]) |layer| index += @intFromBool(layer.mixer == .delta_net);
        const layer_floats = tapeLayerFloats(self.binding.shape);
        const mixed_floats = tapeMixedFloats(self.binding.shape);
        const base = index * layer_floats * 4;
        return .{
            .mixed = self.tape.slice(base, mixed_floats * 4),
            .delta = self.tape.slice(base + mixed_floats * 4, (layer_floats - mixed_floats) * 4),
        };
    }

    /// The M-RoPE sections of `qwen35.rope.dimension_sections`: t, h, w over
    /// the 32 rotary pairs (11, 11, 10), the same as the CPU runtime's.
    const rope_sections = [3]usize{ 11, 11, 10 };
    /// Fills `chunk_rope` with each row's (cos, sin) pairs from its M-RoPE
    /// triple: pair `i` takes axis `i % 3`'s position (t, h, or w) and turns
    /// at `1e7^(-i/32)`, matching `cpu.rope.applyMultiAxis` and the plain
    /// table where the triple is uniform. F64 angles, as the shared table.
    fn fillChunkRope(self: *Plan, position: usize, count: usize) !void {
        const out = self.chunk_rope.floats();
        for (0..count) |r| {
            const triple = self.state.ropeTriple(position + r);
            for (0..32) |i| {
                const axis = i % 3;
                const p: f64 = if (i < 3 * rope_sections[axis]) @floatFromInt(triple[axis]) else 0;
                const theta = p * std.math.pow(f64, 1e7, -@as(f64, @floatFromInt(i)) / 32.0);
                out[(r * 32 + i) * 2] = @floatCast(@cos(theta));
                out[(r * 32 + i) * 2 + 1] = @floatCast(@sin(theta));
            }
        }
    }

    fn attentionChunk(self: *Plan, attn: model.FullAttention, c: AttentionConstants, il: usize, count: usize, position: usize) !void {
        const b = self.backend;
        const cache = self.state.layers[il].attention;
        const hidden = self.binding.shape.hidden;
        const heads = self.binding.shape.heads;
        try self.rotate(self.normalized_c, hidden, count);
        try self.mmRows(attn.query_and_gate, self.normalized_c, hidden, self.qg_c, heads * 512, count);
        try self.mmRows(attn.key, self.normalized_c, hidden, self.k_c, 1024, count);
        try self.mmRows(attn.value, self.normalized_c, hidden, self.v_c, 1024, count);
        try b.rmsNorm(self.qg_c, c.query_norm, self.q_c, .{ .rows = count * heads, .width = 256, .in_stride = 512, .out_stride = 256 });
        try b.rmsNorm(self.k_c, c.key_norm, self.k_c, .{ .rows = count * 4, .width = 256, .in_stride = 256, .out_stride = 256 });
        if (self.state.span_count == 0) {
            // The text fast path: consecutive positions from one shared table,
            // bit-identical to the pre-vision schedule (the trace gates).
            try b.ropeRows(self.q_c, self.rope_table, heads, 256, 64, position, count, heads * 256, .split_half);
            try b.ropeRows(self.k_c, self.rope_table, 4, 256, 64, position, count, 1024, .split_half);
        } else {
            // A chunk that overlaps an image span: each row's M-RoPE triple
            // (`Session.ropeTriple`) into the per-row table, rotated at 0.
            try self.fillChunkRope(position, count);
            try b.ropeRows(self.q_c, self.chunk_rope, heads, 256, 64, 0, count, heads * 256, .split_half);
            try b.ropeRows(self.k_c, self.chunk_rope, 4, 256, 64, 0, count, 1024, .split_half);
        }
        // The chunk's keys and values are contiguous in the cache; padding rows never leave the chunk buffers.
        const precision = cache.keys.precision;
        const k_rows = self.stateSlice(cache.keys.range(position, count));
        const v_rows = self.stateSlice(cache.values.range(position, count));
        // A verify batch or a drafter's commit (few rows) takes the split
        // pass with F32 queries; a prefill chunk the tiled body.
        const few = count <= metal.Backend.verify_max_rows;
        try b.storeRows(&.{ .{ .dst = k_rows, .src = self.k_c, .count = count * 1024 }, .{ .dst = v_rows, .src = self.v_c, .count = count * 1024 } }, precision);
        // Row t attends to cache[0 .. position + t]; padded rows of
        // q_c/mixed_out_c absorb the tiled body's tail.
        const total = position + count;
        var keys = self.stateSlice(cache.keys.range(0, total));
        var values = self.stateSlice(cache.values.range(0, total));
        var queries = self.q_c;
        var operands = precision;
        if (!few and precision != .f32) {
            // The F16 chunk kernel multiplies half operands: the cache rows
            // (a Q8_0 cache expanded first) and a half copy of the rotated
            // queries (padded rows included, as the kernel computes on them).
            if (precision == .q8_0) {
                const stage = self.kv_stage.?;
                const half_keys = stage.slice(0, total * 1024 * 2);
                const half_values = stage.slice(self.state.capacity * 1024 * 2, total * 1024 * 2);
                try b.unpackQ8(&.{ .{ .dst = half_keys, .src = keys, .count = total * 1024 }, .{ .dst = half_values, .src = values, .count = total * 1024 } });
                keys = half_keys;
                values = half_values;
                operands = .f16;
            }
            try b.packHalf(&.{.{ .dst = self.q_c_h, .src = self.q_c, .count = metal.Backend.attentionChunkRows(count) * heads * 256 }});
            queries = self.q_c_h;
        }
        const width = heads * 256;
        const shape: metal.Backend.AttentionChunkShape = .{ .query_heads = heads, .kv_heads = 4, .key_width = 256, .value_width = 256, .position = position, .count = count, .q_stride = width, .out_stride = width, .scale = 1.0 / 16.0, .precision = operands };
        if (few) try b.attentionVerify(keys, values, queries, self.partials, self.mixed_out_c, shape) else try b.attentionChunk(keys, values, queries, self.mixed_out_c, shape);
        try b.sigmoidGate(self.mixed_out_c, self.qg_c, count * heads, 256, 512, 256);
        try self.rotate(self.mixed_out_c, width, count);
        try self.mmRows(attn.output, self.mixed_out_c, width, self.projected_c, hidden, count);
    }

    fn deltaChunk(self: *Plan, linear: model.DeltaNet, c: DeltaConstants, il: usize, count: usize, taped: bool) !void {
        const b = self.backend;
        const state = self.state.layers[il].recurrent;
        const s = self.binding.shape;
        const hidden = s.hidden;
        const vh = s.value_heads;
        const qkv = s.qkvWidth();
        const inner = s.innerWidth();
        // The gating projections read the residual before its transform.
        try self.mmRows(linear.beta, self.normalized_c, hidden, self.beta_c, vh, count);
        try self.mmRows(linear.alpha, self.normalized_c, hidden, self.alpha_c, vh, count);
        try self.rotate(self.normalized_c, hidden, count);
        try self.mmRows(linear.qkv, self.normalized_c, hidden, self.mixed_c, qkv, count);
        try self.mmRows(linear.gate, self.normalized_c, hidden, self.z_c, inner, count);
        const history = self.stateFloats(state.history);
        try b.convolutionRows(history, self.mixed_c, c.convolution, self.convolved_c, qkv, 4, count, qkv);
        // A taped verify leaves the history and the matrix at the batch's
        // start; `replayRows` advances both over the kept rows.
        const tape = if (taped) self.tapeLayer(il) else null;
        if (tape) |t| try b.copy(t.mixed, self.mixed_c, count * qkv) else try b.convolutionHistory(history, self.mixed_c, qkv, 4, count, qkv);
        try b.silu(self.convolved_c, count * qkv);
        try b.l2NormRows(self.convolved_c, count, 32, 128, 128, qkv, 1e-6);
        try b.deltaGatesRows(self.alpha_c, self.beta_c, c.a, c.time_bias, vh, count);
        if (tape) |t| {
            var shape = tapeShape(s);
            shape.count = count;
            try b.deltaRows(self.stateFloats(state.matrix), self.convolved_c, self.alpha_c, self.beta_c, self.mixed_out_c, t.delta, shape);
        } else {
            // One chunkwise dispatch for the whole chunk: the WY form over
            // 32-token sub-chunks with the state carried in place.
            try b.deltaChunk(self.stateFloats(state.matrix), self.convolved_c, self.alpha_c, self.beta_c, self.mixed_out_c, .{ .qheads = 16, .vheads = vh, .keys = 128, .values = 128, .count = count, .in_stride = qkv, .gate_stride = vh, .out_stride = inner, .scale = 1.0 / @sqrt(@as(f32, 128)) });
        }
        try b.rmsNorm(self.mixed_out_c, c.norm, self.mixed_out_c, .{ .rows = count * vh, .width = 128, .in_stride = 128, .out_stride = 128, .silu_multiplier = .{ .buffer = self.z_c, .stride = 128 } });
        const out_input = try self.rotateGrouped(self.mixed_out_c, count, true);
        try self.mmRows(linear.output, out_input, inner, self.projected_c, hidden, count);
    }

    /// Copies the last step's logits out of the shared buffer. Valid after a
    /// `step` that computed them (any non-null `logits`/`greedy`/`topk`) and
    /// until the next step; the sampler's fallback path uses it. When a
    /// selection was requested with penalties, the vector is the penalized
    /// one (the raw vector is not kept), so the sampler must not penalize it
    /// again.
    pub fn readLogits(self: *Plan, out: []f32) !void {
        if (out.len != vocabulary) return error.InvalidShape;
        @memcpy(out, self.logits.floats());
        for (out) |v| if (!std.math.isFinite(v)) return error.NonFiniteResult;
    }

    // --- the prediction block ------------------------------------------

    /// One row of the block at `position`: pair `token` with `h_prev`,
    /// project, run the full-attention layer over the block's own cache, the
    /// FFN, `shared_head_norm`, and the shared output head. The block's
    /// `h_nextn` lands in `draft_h`, its own hidden for chaining in
    /// `draft_chain`. `h_prev` is a device row of `hidden` floats; `greedy`
    /// and `logits` are read only when given, and `pmax` (the top candidate's
    /// softmax probability, `1 / Σ exp(l − max)` at temperature 1) with them.
    pub fn draftForward(self: *Plan, h_prev: Buffer, token: u32, position: usize, greedy: ?*u32, pmax: ?*f32, logits: ?[]f32) !void {
        const hidden = self.binding.shape.hidden;
        const ffn = self.binding.shape.ffn;
        const block = self.binding.draft orelse return error.NoDraftBlock;
        const constants = self.draft_constants orelse return error.NoDraftBlock;
        if (token >= vocabulary or position >= self.state.capacity or h_prev.len < hidden * 4) return error.InvalidShape;
        if (logits) |out| if (out.len != vocabulary) return error.InvalidShape;
        const b = self.backend;
        const single: metal.Backend.Norm = .{ .rows = 1, .width = hidden, .in_stride = hidden, .out_stride = hidden };
        try b.begin();
        errdefer b.discard();
        const embedding = try self.weight(self.binding.token_embedding);
        try b.embed(embedding.buffer, embedding.matrix, token, self.x);
        // [enorm(embed(x_p)); hnorm(h_{p-1})], projected by eh_proj.
        try b.rmsNorm(self.x, self.draft_enorm, self.normalized, single);
        try b.rmsNorm(h_prev, self.draft_hnorm_w, self.draft_hnorm, single);
        try b.copy(self.draft_concat.slice(0, hidden * 4), self.normalized, hidden);
        try b.copy(self.draft_concat.slice(hidden * 4, hidden * 4), self.draft_hnorm, hidden);
        try self.mm(block.eh_proj, self.draft_concat, self.x);
        try b.rmsNorm(self.x, constants.attention_norm, self.normalized, single);
        // The drafter runs only on sessions without image spans, where the
        // row is the rotary position.
        try self.fullAttention(block.layer.mixer.full_attention, constants.mixer.full_attention, self.draft_layer, position, position);
        try b.add(self.x, self.projected, hidden);
        try b.rmsNorm(self.x, constants.post_attention_norm, self.normalized, single);
        try self.mm(block.layer.ffn_gate, self.normalized, self.gate);
        try self.mm(block.layer.ffn_up, self.normalized, self.up);
        try b.siluMul(self.gate, self.up, ffn);
        try self.mm(block.layer.ffn_down, self.gate, self.projected);
        try b.add(self.x, self.projected, hidden);
        try b.rmsNorm(self.x, self.draft_head_norm, self.draft_h, single);
        // The block's own hidden chains the next proposed position.
        try b.copy(self.draft_chain, self.draft_h, hidden);
        try self.mm(self.binding.output, self.draft_h, self.draft_logits);
        // `pmax` needs the argmax and one top-1 top-k pass (whose F64 sum of
        // the partials is Σ exp(l − max) at T = 1) in the same command buffer.
        if (greedy != null or logits != null or pmax != null) try b.argmax(self.draft_logits, vocabulary, self.argmax_values, self.argmax_indices, self.argmax_result);
        if (pmax != null) try b.topk(self.draft_logits, vocabulary, 1, 1.0, self.topk);
        try b.commit();
        if (greedy) |out| {
            out.* = @as(*const u32, @ptrCast(@alignCast(self.argmax_result.host))).*;
            if (out.* >= vocabulary) return error.NonFiniteResult;
        }
        if (pmax) |out| {
            var total: f64 = 0;
            for (self.topk.sums.floats()[0..metal.Backend.topk_partials]) |partial| total += partial;
            var finite = true;
            for (@as([*]const u32, @ptrCast(@alignCast(self.topk.flags.host)))[0..metal.Backend.topk_partials]) |flag| if (flag != 0) {
                finite = false;
            };
            out.* = if (finite and total > 0) @floatCast(1.0 / total) else 1.0;
        }
        if (logits) |out| {
            @memcpy(out, self.draft_logits.floats());
            for (out) |v| if (!std.math.isFinite(v)) return error.NonFiniteResult;
        }
    }

    /// `draftForward` with a host `h_prev`, staged through `draft_chain`.
    pub fn draftForwardHost(self: *Plan, h_prev: []const f32, token: u32, position: usize, greedy: ?*u32, pmax: ?*f32, logits: ?[]f32) !void {
        const hidden = self.binding.shape.hidden;
        if (h_prev.len != hidden) return error.InvalidShape;
        @memcpy(self.draft_chain.floats()[0..hidden], h_prev);
        try self.draftForward(self.draft_chain, token, position, greedy, pmax, logits);
    }

    /// Greedy candidates from the state after the last committed token, each
    /// chained through the block's own hidden; `out.len` bounds the count.
    /// `p_min > 0` stops after a position whose top candidate's probability
    /// (the device top-1 pass) is below it; the first position is always
    /// proposed.
    pub fn propose(self: *Plan, token: u32, out: []u32, p_min: f32) !usize {
        if (!self.mtp) return error.NoDraftBlock;
        if (!std.math.isFinite(p_min) or p_min < 0 or p_min > 1) return error.InvalidShape;
        try self.settle();
        const start = self.state.position;
        var h_prev = self.draft_pending_h;
        var next = token;
        var count: usize = 0;
        while (count < out.len) : (count += 1) {
            var greedy: u32 = 0;
            var pmax: f32 = 0;
            try self.draftForward(h_prev, next, start + count, &greedy, if (p_min > 0) &pmax else null, null);
            out[count] = greedy;
            if (p_min > 0 and pmax < p_min) {
                count += 1;
                break;
            }
            next = greedy;
            h_prev = self.draft_chain;
        }
        return count;
    }

    /// Advances the block over tokens the main model committed, whose target
    /// hidden rows are `h_rows` (`tokens.len * hidden`). The caller has
    /// already `recover`ed the session; the block's cache row index is the
    /// main token position, so the accepted prefix ends at `state.position`.
    /// A single token runs the standalone `draftForward`; a longer prefix runs
    /// one batched forward per `chunk` rows (the cache is filled from the
    /// target hidden, never the block's own).
    pub fn commit(self: *Plan, tokens: []const u32, h_rows: []const f32) !void {
        const hidden = self.binding.shape.hidden;
        if (!self.mtp) return error.NoDraftBlock;
        if (h_rows.len != tokens.len * hidden) return error.InvalidShape;
        if (tokens.len == 0) return;
        try self.settle();
        if (self.state.position < tokens.len) return error.InvalidShape;
        const start = self.state.position - tokens.len;
        if (tokens.len == 1) {
            try self.draftForward(self.draft_pending_h, tokens[0], start, null, null, null);
        } else {
            var base: usize = 0;
            while (base < tokens.len) {
                const count = @min(tokens.len - base, self.chunk);
                const h_prev: []const f32 = if (base == 0) self.draft_pending_h.floats()[0..hidden] else h_rows[(base - 1) * hidden ..][0..hidden];
                const next: []const f32 = if (count > 1) h_rows[base * hidden ..][0 .. (count - 1) * hidden] else &.{};
                try self.commitBatch(tokens[base..][0..count], h_prev, next, start + base);
                base += count;
            }
        }
        @memcpy(self.draft_pending_h.floats()[0..hidden], h_rows[h_rows.len - hidden ..][0..hidden]);
    }

    /// One command buffer over `count` committed rows at cache positions
    /// `start .. start + count`. Row `t` of the block pairs with the target
    /// hidden of the row before it: `h_prev` for row 0, `h_next_rows`
    /// (`(count − 1) × hidden`) for the rest, staged into `draft_hprev_c`.
    /// The head norm is skipped: a commit only fills the cache rows, and the
    /// caller sets `draft_pending_h` from the last target hidden.
    fn commitBatch(self: *Plan, tokens: []const u32, h_prev: []const f32, h_next_rows: []const f32, start: usize) !void {
        const hidden = self.binding.shape.hidden;
        const ffn = self.binding.shape.ffn;
        const count = tokens.len;
        std.debug.assert(count >= 1 and count <= self.chunk);
        std.debug.assert(h_prev.len == hidden and h_next_rows.len == (count - 1) * hidden);
        const block = self.binding.draft.?;
        const constants = self.draft_constants.?;
        const b = self.backend;
        const hp = self.draft_hprev_c.floats();
        @memcpy(hp[0..hidden], h_prev);
        if (count > 1) @memcpy(hp[hidden .. count * hidden], h_next_rows);
        try b.begin();
        errdefer b.discard();
        const embedding = try self.weight(self.binding.token_embedding);
        for (tokens, 0..) |token, t| try b.embed(embedding.buffer, embedding.matrix, token, self.x_c.slice(t * hidden * 4, hidden * 4));
        // enorm fills column 0 of the pair, hnorm column `hidden`; the 10240
        // stride leaves room for the other half without a per-row copy.
        const pair: metal.Backend.Norm = .{ .rows = count, .width = hidden, .in_stride = hidden, .out_stride = 2 * hidden };
        try b.rmsNorm(self.x_c, self.draft_enorm, self.draft_concat_c, pair);
        try b.rmsNorm(self.draft_hprev_c, self.draft_hnorm_w, self.draft_concat_c.slice(hidden * 4, self.draft_concat_c.len - hidden * 4), pair);
        try self.mmRows(block.eh_proj, self.draft_concat_c, 2 * hidden, self.x_c, hidden, count);
        try b.rmsNorm(self.x_c, constants.attention_norm, self.normalized_c, .{ .rows = count, .width = hidden, .in_stride = hidden, .out_stride = hidden });
        try self.attentionChunk(block.layer.mixer.full_attention, constants.mixer.full_attention, self.draft_layer, count, start);
        try b.addRmsNorm(self.x_c, self.projected_c, constants.post_attention_norm, self.normalized_c, .{ .rows = count, .width = hidden, .in_stride = hidden, .out_stride = hidden });
        try self.mmRows(block.layer.ffn_gate, self.normalized_c, hidden, self.gate_c, ffn, count);
        try self.mmRows(block.layer.ffn_up, self.normalized_c, hidden, self.up_c, ffn, count);
        try b.siluMul(self.gate_c, self.up_c, count * ffn);
        try self.mmRows(block.layer.ffn_down, self.gate_c, ffn, self.projected_c, hidden, count);
        try b.add(self.x_c, self.projected_c, count * hidden);
        try b.commit();
    }

    // --- the DFlash2 drafter --------------------------------------------

    /// Advances the DFlash2 drafter over committed tokens: each row's five
    /// residuals (`tokens.len × dflash2.hidden_width`) encode to one feature
    /// whose keys and values enter the drafter's caches at the token's
    /// position, one command buffer per `chunk` rows. The caller has already
    /// `recover`ed the session.
    pub fn flashCommit(self: *Plan, tokens: []const u32, h_rows: []const f32) !void {
        const d = &(self.flash orelse return error.NoDraftBlock);
        if (h_rows.len != tokens.len * dflash2.hidden_width) return error.InvalidShape;
        if (tokens.len == 0) return;
        try self.settle();
        if (self.state.position < tokens.len) return error.InvalidShape;
        const start = self.state.position - tokens.len;
        var offset: usize = 0;
        while (offset < tokens.len) {
            const count = @min(self.chunk, tokens.len - offset);
            try self.flashCommitRows(d, count, h_rows[offset * dflash2.hidden_width ..][0 .. count * dflash2.hidden_width], start + offset);
            offset += count;
        }
    }

    fn flashCommitRows(self: *Plan, d: *Flash, count: usize, h_rows: []const f32, start: usize) !void {
        const b = self.backend;
        @memcpy(d.stage.floats()[0 .. count * dflash2.hidden_width], h_rows);
        try b.begin();
        errdefer b.discard();
        try d.mmRows(b, d.binding.fc, d.stage, dflash2.hidden_width, d.encoded, dflash2.embedding, count);
        try b.rmsNorm(d.encoded, d.encoder_norm, d.encoded, flashNorm(count));
        for (d.binding.layers, d.constants, 0..) |layer, c, il| {
            try d.mmRows(b, layer.key, d.encoded, dflash2.embedding, d.inject_k, dflash2.kv_width, count);
            try d.mmRows(b, layer.value, d.encoded, dflash2.embedding, d.inject_v, dflash2.kv_width, count);
            try b.rmsNorm(d.inject_k, c.key_norm, d.inject_k, flashHeadNorm(count * dflash2.kv_heads));
            try b.ropeRows(d.inject_k, d.rope_table, dflash2.kv_heads, dflash2.head_size, dflash2.head_size, start, count, dflash2.kv_width, .split_half);
            const cache = self.state.layers[self.draft_layer + il].attention;
            try b.storeRows(&.{
                .{ .dst = self.stateSlice(cache.keys.range(start, count)), .src = d.inject_k, .count = count * dflash2.kv_width },
                .{ .dst = self.stateSlice(cache.values.range(start, count)), .src = d.inject_v, .count = count * dflash2.kv_width },
            }, cache.keys.precision);
        }
        try b.commit();
    }

    /// Runs the noise block `rows` rows from `token` at the session position
    /// (`dflash2_runtime.Draft.block`): every row's final normed hidden lands
    /// in `hidden`, and its keys and values in the drafter's caches, stale
    /// once the caller commits.
    fn flashBlock(self: *Plan, token: u32, rows: usize) !void {
        const d = &(self.flash orelse return error.NoDraftBlock);
        if (rows == 0 or rows > dflash2.block_size) return error.InvalidShape;
        const base = self.state.position;
        if (base + rows > self.state.capacity) return error.ContextFull;
        const last = base + rows - 1;
        const width = dflash2.embedding;
        const b = self.backend;
        try b.begin();
        errdefer b.discard();
        const embedding = try self.weight(self.binding.token_embedding);
        for (0..rows) |r| {
            const input = if (r == 0) token else dflash2.mask_token;
            try b.embed(embedding.buffer, embedding.matrix, input, d.x.slice(r * width * 4, width * 4));
        }
        for (d.binding.layers, d.constants, 0..) |layer, c, il| {
            // Attention, wrapped in its convolution.
            try b.rmsNorm(d.x, c.attention_norm, d.normalized, flashNorm(rows));
            try d.mmRows(b, layer.attention_conv.projection, d.normalized, width, d.taps, dflash2.conv_projection, rows);
            try b.dflashConv(c.attention_base, d.taps, d.normalized, d.prepared, rows, width, 0, dflash2.conv_group, dflash2.conv_projection);
            try d.mmRows(b, layer.query, d.prepared, width, d.q, dflash2.query_width, rows);
            try d.mmRows(b, layer.key, d.prepared, width, d.k, dflash2.kv_width, rows);
            try d.mmRows(b, layer.value, d.prepared, width, d.v, dflash2.kv_width, rows);
            try b.rmsNorm(d.q, c.query_norm, d.q, flashHeadNorm(rows * dflash2.heads));
            try b.ropeRows(d.q, d.rope_table, dflash2.heads, dflash2.head_size, dflash2.head_size, base, rows, dflash2.query_width, .split_half);
            try b.rmsNorm(d.k, c.key_norm, d.k, flashHeadNorm(rows * dflash2.kv_heads));
            try b.ropeRows(d.k, d.rope_table, dflash2.kv_heads, dflash2.head_size, dflash2.head_size, base, rows, dflash2.kv_width, .split_half);
            const cache = self.state.layers[self.draft_layer + il].attention;
            try b.storeRows(&.{
                .{ .dst = self.stateSlice(cache.keys.range(base, rows)), .src = d.k, .count = rows * dflash2.kv_width },
                .{ .dst = self.stateSlice(cache.values.range(base, rows)), .src = d.v, .count = rows * dflash2.kv_width },
            }, cache.keys.precision);
            // Non-causal inside the block: each row takes one decode over its
            // own visible slice, the prefix back to its window and the block.
            for (0..rows) |r| {
                const position = base + r;
                const first = if (position + 1 > dflash2.window) position + 1 - dflash2.window else 0;
                const visible = last - first + 1;
                try b.attentionDecode(
                    self.stateSlice(cache.keys.range(first, visible)),
                    self.stateSlice(cache.values.range(first, visible)),
                    d.q.slice(r * dflash2.query_width * 4, dflash2.query_width * 4),
                    d.partials,
                    d.mixed.slice(r * dflash2.query_width * 4, dflash2.query_width * 4),
                    .{ .query_heads = dflash2.heads, .kv_heads = dflash2.kv_heads, .key_width = dflash2.head_size, .value_width = dflash2.head_size, .visible = visible, .scale = 1.0 / @sqrt(@as(f32, dflash2.head_size)), .precision = cache.keys.precision },
                );
            }
            try d.mmRows(b, layer.output, d.mixed, dflash2.query_width, d.sub, width, rows);
            try b.dflashConv(c.attention_base, d.taps, d.sub, d.finished, rows, width, 1, dflash2.conv_group, dflash2.conv_projection);
            try b.add(d.x, d.finished, rows * width);
            // The FFN, wrapped the same way.
            try b.rmsNorm(d.x, c.ffn_norm, d.normalized, flashNorm(rows));
            try d.mmRows(b, layer.ffn_conv.projection, d.normalized, width, d.taps, dflash2.conv_projection, rows);
            try b.dflashConv(c.ffn_base, d.taps, d.normalized, d.prepared, rows, width, 0, dflash2.conv_group, dflash2.conv_projection);
            try d.mmRows(b, layer.ffn_gate, d.prepared, width, d.gate, dflash2.feed_forward, rows);
            try d.mmRows(b, layer.ffn_up, d.prepared, width, d.up, dflash2.feed_forward, rows);
            try b.siluMul(d.gate, d.up, rows * dflash2.feed_forward);
            try d.mmRows(b, layer.ffn_down, d.gate, dflash2.feed_forward, d.sub, width, rows);
            try b.dflashConv(c.ffn_base, d.taps, d.sub, d.finished, rows, width, 1, dflash2.conv_group, dflash2.conv_projection);
            try b.add(d.x, d.finished, rows * width);
        }
        try b.rmsNorm(d.x, d.output_norm, d.hidden, flashNorm(rows));
        try b.commit();
    }

    /// One noise block from `token`, then the selector chain over rows
    /// 1… on the host: each position's device top-16 logits, re-scored by the
    /// bilinear term of the chosen predecessor (`dflash2.selectorScores`).
    /// `p_min > 0` stops after a position whose choice's full-vocabulary
    /// softmax probability is below it; the first position is always proposed.
    pub fn flashPropose(self: *Plan, token: u32, out: []u32, p_min: f32) !usize {
        const d = &(self.flash orelse return error.NoDraftBlock);
        if (token >= vocabulary) return error.InvalidTokenId;
        if (!std.math.isFinite(p_min) or p_min < 0 or p_min > 1) return error.InvalidShape;
        const count_max: usize = @min(out.len, dflash2.block_size - 1);
        if (count_max == 0) return 0;
        try self.settle();
        const rows = count_max + 1;
        try self.flashBlock(token, rows);
        const b = self.backend;
        try b.begin();
        errdefer b.discard();
        const head = try self.weight(self.binding.output);
        try b.matmul(head.buffer, head.matrix, d.hidden, dflash2.embedding, d.logits, vocabulary, rows);
        for (1..rows) |r| try b.topk(d.logits.slice(r * vocabulary * 4, vocabulary * 4), vocabulary, dflash2.selector_top_k, 1.0, d.topk[r]);
        try d.mmRows(b, d.binding.selector_hidden, d.hidden, dflash2.embedding, d.projected, dflash2.selector_rank, rows);
        try b.commit();
        const view = d.binding.view;
        var previous = token;
        var count: usize = 0;
        while (count < count_max) {
            const scratch = d.topk[1 + count];
            const ids = @as([*]const u32, @ptrCast(@alignCast(scratch.indices.host)))[0..dflash2.selector_top_k];
            const values = scratch.values.floats()[0..dflash2.selector_top_k];
            for (ids) |id| if (id >= vocabulary) return error.NonFiniteResult;
            try view.row(d.binding.selector_predecessor, previous, d.predecessor.floats()[0..dflash2.selector_rank]);
            for (ids, 0..) |id, k| try view.row(d.binding.selector_successor, id, d.successors.floats()[k * dflash2.selector_rank ..][0..dflash2.selector_rank]);
            var scores: [dflash2.selector_top_k]f32 = undefined;
            dflash2.selectorScores(values, d.predecessor.floats()[0..dflash2.selector_rank], d.projected.floats()[(1 + count) * dflash2.selector_rank ..][0..dflash2.selector_rank], d.successors.floats()[0 .. dflash2.selector_top_k * dflash2.selector_rank], &scores);
            const choice = dflash2.selectorChoice(&scores);
            out[count] = ids[choice];
            previous = ids[choice];
            count += 1;
            if (p_min > 0) {
                // The partition sums are Σ exp(l − max); values[0] is the max.
                var total: f64 = 0;
                for (scratch.sums.floats()[0..metal.Backend.topk_partials]) |partial| total += partial;
                const p = @exp(@as(f64, values[choice]) - values[0]) / total;
                if (total > 0 and p < p_min) break;
            }
        }
        return count;
    }

    /// One noise block for `--draft-trace`: every row's final hidden into
    /// `hidden` (`rows × dflash2.embedding`).
    pub fn flashBlockTrace(self: *Plan, token: u32, rows: usize, hidden: []f32) !void {
        const d = &(self.flash orelse return error.NoDraftBlock);
        if (hidden.len != rows * dflash2.embedding) return error.InvalidShape;
        try self.settle();
        try self.flashBlock(token, rows);
        @memcpy(hidden, d.hidden.floats()[0..hidden.len]);
    }

    fn flashProposeFn(host: *anyopaque, token: u32, out: []u32, p_min: f32) anyerror!usize {
        const self: *Plan = @ptrCast(@alignCast(host));
        return self.flashPropose(token, out, p_min);
    }
    fn flashCommitFn(host: *anyopaque, tokens: []const u32, h_rows: []const f32) anyerror!void {
        const self: *Plan = @ptrCast(@alignCast(host));
        return self.flashCommit(tokens, h_rows);
    }
    fn flashResetFn(host: *anyopaque) void {
        // The drafter's state is its session caches, which `reset` clears.
        _ = host;
    }
    fn flashCarried(host: *anyopaque) []f32 {
        _ = host;
        return &.{};
    }
    fn flashBytes(host: *anyopaque) usize {
        const self: *Plan = @ptrCast(@alignCast(host));
        const d = &(self.flash orelse return 0);
        return d.bytes();
    }

    /// The contract value the engine holds, or null when no block is loaded.
    pub fn drafter(self: *Plan) ?Drafter {
        const hidden = self.binding.shape.hidden;
        if (self.flash != null) return .{ .host = self, .hidden = dflash2.hidden_width, .max_proposals = dflash2.block_size - 1, .propose_fn = flashProposeFn, .commit_fn = flashCommitFn, .reset_fn = flashResetFn, .carried_fn = flashCarried, .bytes_fn = flashBytes };
        if (!self.mtp) return null;
        return .{ .host = self, .hidden = hidden, .max_proposals = model.max_draft_proposals, .propose_fn = proposeFn, .commit_fn = commitFn, .reset_fn = resetDraftFn, .carried_fn = carriedDraft, .bytes_fn = draftBytes };
    }
    fn proposeFn(host: *anyopaque, token: u32, out: []u32, p_min: f32) anyerror!usize {
        const self: *Plan = @ptrCast(@alignCast(host));
        return self.propose(token, out, p_min);
    }
    fn commitFn(host: *anyopaque, tokens: []const u32, h_rows: []const f32) anyerror!void {
        const self: *Plan = @ptrCast(@alignCast(host));
        return self.commit(tokens, h_rows);
    }
    fn resetDraftFn(host: *anyopaque) void {
        const self: *Plan = @ptrCast(@alignCast(host));
        if (self.mtp) @memset(self.draft_pending_h.floats(), 0);
    }
    fn carriedDraft(host: *anyopaque) []f32 {
        const self: *Plan = @ptrCast(@alignCast(host));
        if (!self.mtp) return &.{};
        return self.draft_pending_h.floats();
    }
    fn draftBytes(host: *anyopaque) usize {
        const self: *Plan = @ptrCast(@alignCast(host));
        if (!self.mtp) return 0;
        return (self.draft_h.len + self.draft_hnorm.len + self.draft_chain.len + self.draft_pending_h.len + self.draft_concat.len + self.draft_logits.len);
    }

    /// One decode step's full-attention layer: the new row is written at
    /// cache row `row` and attends to rows `[0, row]`, rotated at
    /// `rope_position`. The two differ after an image span, whose rows
    /// advance the rotary position by less than their count.
    fn fullAttention(self: *Plan, attn: model.FullAttention, c: AttentionConstants, il: usize, row: usize, rope_position: usize) !void {
        const b = self.backend;
        const cache = self.state.layers[il].attention;
        const k_slot = self.stateSlice(cache.keys.range(row, 1));
        const v_slot = self.stateSlice(cache.values.range(row, 1));
        const precision = cache.keys.precision;
        // An F32 cache takes the projections directly; F16 and Q8_0 take
        // them through F32 scratch rows and one pack dispatch.
        const k_row = if (precision == .f32) k_slot else self.k;
        const v_row = if (precision == .f32) v_slot else self.v;
        const hidden = self.binding.shape.hidden;
        const heads = self.binding.shape.heads;
        try self.rotate(self.normalized, hidden, 1);
        try self.projections(&.{ attn.query_and_gate, attn.key, attn.value }, &.{ self.qg, k_row, v_row }, .plain);
        // Each projected head stores query then gate; the gate is applied after attention.
        try b.rmsNormRope(self.qg, c.query_norm, self.rope_table, self.q, .{ .rows = heads, .width = 256, .in_stride = 512, .out_stride = 256 }, 64, rope_position, .split_half);
        try b.rmsNormRope(k_row, c.key_norm, self.rope_table, k_row, .{ .rows = 4, .width = 256, .in_stride = 256, .out_stride = 256 }, 64, rope_position, .split_half);
        if (precision != .f32) try b.storeRows(&.{ .{ .dst = k_slot, .src = k_row, .count = 1024 }, .{ .dst = v_slot, .src = v_row, .count = 1024 } }, precision);
        const visible = row + 1;
        try b.attentionDecode(self.stateSlice(cache.keys.range(0, visible)), self.stateSlice(cache.values.range(0, visible)), self.q, self.partials, self.mixed_out, .{ .query_heads = heads, .kv_heads = 4, .key_width = 256, .value_width = 256, .visible = visible, .scale = 1.0 / 16.0, .precision = precision });
        try b.sigmoidGate(self.mixed_out, self.qg, heads, 256, 512, 256);
        try self.rotate(self.mixed_out, heads * 256, 1);
        try self.mm(attn.output, self.mixed_out, self.projected);
    }

    fn linearAttention(self: *Plan, linear: model.DeltaNet, c: DeltaConstants, il: usize) !void {
        const b = self.backend;
        const state = self.state.layers[il].recurrent;
        const s = self.binding.shape;
        const vh = s.value_heads;
        const qkv = s.qkvWidth();
        if (self.rotation == null) {
            try self.projections(&.{ linear.qkv, linear.gate, linear.beta, linear.alpha }, &.{ self.mixed, self.z, self.beta, self.alpha }, .plain);
        } else {
            // The gating projections read the residual before its transform,
            // so the merge splits in two around it.
            try self.projections(&.{ linear.beta, linear.alpha }, &.{ self.beta, self.alpha }, .plain);
            try self.rotate(self.normalized, s.hidden, 1);
            try self.projections(&.{ linear.qkv, linear.gate }, &.{ self.mixed, self.z }, .plain);
        }
        try b.convolution(self.stateFloats(state.history), self.mixed, c.convolution, self.convolved, qkv, 4);
        try b.silu(self.convolved, qkv);
        // Convolution output is [Q:16x128 | K:16x128 | V:value heads x 128]; Q and K are L2-normalized.
        try b.l2Norm(self.convolved, 32, 128, 128, 1e-6);
        try b.deltaGates(self.alpha, self.beta, c.a, c.time_bias, vh);
        try b.delta(self.stateFloats(state.matrix), self.convolved, self.alpha, self.beta, self.mixed_out, .{ .qheads = 16, .vheads = vh, .keys = 128, .values = 128, .scale = 1.0 / @sqrt(@as(f32, 128)) });
        try b.rmsNorm(self.mixed_out, c.norm, self.mixed_out, .{ .rows = vh, .width = 128, .in_stride = 128, .out_stride = 128, .silu_multiplier = .{ .buffer = self.z, .stride = 128 } });
        const out_input = try self.rotateGrouped(self.mixed_out, 1, false);
        try self.mm(linear.output, out_input, self.projected);
    }
};
