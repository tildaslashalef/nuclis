//! OmniVoice / VoiceTut-TTS masked-diffusion backbone on Metal.
//!
//! Shares all speed pieces from day one:
//! - Wraps mapped Q8_0 / F16 / F32 weights in place (`Backend.wrap`, matched
//!   `Backend.unwrap` on `deinit` and partial-init failure).
//! - Packs conditional (`S_cond`) and unconditional (`S_uncond`) CFG sequences
//!   into ONE packed `[S_cond + S_uncond]` forward pass per diffusion step via
//!   `Backend.attentionSegmentsGrouped` with per-segment `[begin, end)` bounds
//!   and per-segment NeoX RoPE positions `[0..S_cond)` and `[0..S_uncond)`.
//! - Routes all 28 layers + final norm + `audio_heads` (on target rows only)
//!   into ONE command buffer per diffusion step via `Backend.matmulTile`, with
//!   reusable scratch buffers grown on demand (zero allocations per step).
//! Facts: docs/models/omnivoice.md, docs/benchmarks/record/omnivoice.md.
const std = @import("std");
const metal = @import("../backends/metal/root.zig");
const cpu = @import("../backends/cpu/root.zig");
const model = @import("omnivoice.zig");
const runtime = @import("omnivoice_runtime.zig");
const hc = @import("../audio/higgs_codec.zig");

const Allocator = std.mem.Allocator;
const Backend = metal.Backend;
const Buffer = metal.Buffer;

const hidden = runtime.hidden;
const intermediate = runtime.intermediate;
const layer_count = runtime.layer_count;
const heads = runtime.heads;
const kv_heads = runtime.kv_heads;
const head_dim = runtime.head_dim;
const query_width = runtime.query_width;
const kv_width = runtime.kv_width;
const audio_table_rows = runtime.audio_table_rows;

/// Hard ceiling on packed `S_cond + S_uncond` rows.
pub const max_total_rows: usize = runtime.max_sequence_rows;

const Weight = struct { buffer: Buffer, matrix: cpu.Matrix };

const Layer = struct {
    input_norm: Buffer,
    post_norm: Buffer,
    q_norm: Buffer,
    k_norm: Buffer,
    q_proj: Weight,
    k_proj: Weight,
    v_proj: Weight,
    o_proj: Weight,
    gate_proj: Weight,
    up_proj: Weight,
    down_proj: Weight,
};

const Scratch = struct {
    capacity_rows: usize = 0,
    target_capacity: usize = 0,
    scores_capacity: usize = 0,
    x: ?Buffer = null,
    normed: ?Buffer = null,
    q: ?Buffer = null,
    k: ?Buffer = null,
    v: ?Buffer = null,
    attn_out: ?Buffer = null,
    proj: ?Buffer = null,
    gate: ?Buffer = null,
    up: ?Buffer = null,
    bounds: ?Buffer = null,
    tiles: ?Buffer = null,
    scores: ?Buffer = null,
    target_h: ?Buffer = null,
    target_logits: ?Buffer = null,

    fn release(self: *Scratch, backend: *Backend) void {
        const bufs = [_]?Buffer{
            self.x,    self.normed, self.q,      self.k,     self.v,      self.attn_out, self.proj,
            self.gate, self.up,     self.bounds, self.tiles, self.scores, self.target_h, self.target_logits,
        };
        for (bufs) |mb| if (mb) |b| backend.release(b) catch {};
        self.* = .{};
    }

    fn ensure(self: *Scratch, backend: *Backend, total_rows: usize, target_rows: usize, needed_scores: usize) !void {
        const padded = @max(Backend.matmulPadded(total_rows), Backend.attentionChunkRows(total_rows));
        if (padded > self.capacity_rows) {
            const old = [_]?Buffer{
                self.x, self.normed, self.q, self.k, self.v, self.attn_out, self.proj, self.gate, self.up, self.bounds, self.tiles,
            };
            for (old) |mb| if (mb) |b| backend.release(b) catch {};
            self.x = null;
            self.normed = null;
            self.q = null;
            self.k = null;
            self.v = null;
            self.attn_out = null;
            self.proj = null;
            self.gate = null;
            self.up = null;
            self.bounds = null;
            self.tiles = null;
            self.capacity_rows = 0;

            self.x = try backend.create(padded * hidden * 4);
            self.normed = try backend.create(padded * hidden * 4);
            self.q = try backend.create(padded * query_width * 4);
            self.k = try backend.create(padded * kv_width * 4);
            self.v = try backend.create(padded * kv_width * 4);
            self.attn_out = try backend.create(padded * query_width * 4);
            self.proj = try backend.create(padded * hidden * 4);
            self.gate = try backend.create(padded * intermediate * 4);
            self.up = try backend.create(padded * intermediate * 4);
            self.bounds = try backend.create(padded * 8);
            const max_tiles = @divCeil(padded, 8) + 2;
            self.tiles = try backend.create(max_tiles * @sizeOf(Backend.AttentionTile));
            self.capacity_rows = padded;
        }
        const target_padded = Backend.matmulPadded(target_rows);
        if (target_padded > self.target_capacity) {
            if (self.target_h) |b| backend.release(b) catch {};
            if (self.target_logits) |b| backend.release(b) catch {};
            self.target_h = null;
            self.target_logits = null;
            self.target_capacity = 0;

            self.target_h = try backend.create(target_padded * hidden * 4);
            self.target_logits = try backend.create(target_padded * audio_table_rows * 4);
            self.target_capacity = target_padded;
        }
        if (needed_scores > self.scores_capacity) {
            if (self.scores) |b| backend.release(b) catch {};
            self.scores = null;
            self.scores_capacity = 0;
            self.scores = try backend.create(needed_scores * 4);
            self.scores_capacity = needed_scores;
        }
    }
};

/// Metal execution plan for OmniVoice's 28-layer bidirectional Qwen3-0.6B
/// backbone. Borrows `backend` and `model` (both must outlive `Plan`).
pub const Plan = struct {
    gpa: Allocator,
    backend: *Backend,
    model: *const runtime.Model,
    wrapped: std.ArrayList([]const u8),
    created: std.ArrayList(Buffer),
    output_norm: Buffer,
    audio_heads: Weight,
    rope_table: Buffer,
    layers: [layer_count]Layer,
    scratch: Scratch = .{},

    pub fn init(gpa: Allocator, backend: *Backend, m: *const runtime.Model) !Plan {
        if (!metal.enabled) return error.MetalNotEnabled;

        var wrapped: std.ArrayList([]const u8) = .empty;
        errdefer {
            for (wrapped.items) |slice| backend.unwrap(slice);
            wrapped.deinit(gpa);
        }
        try wrapped.ensureTotalCapacity(gpa, 1 + layer_count * 7);

        var created: metal.Created = .init(gpa, backend);
        defer created.deinit();
        errdefer created.releaseAll();

        const output_norm = try uploadVector(&created, m.output_norm);
        const audio_heads = try wrapMatrix(backend, &wrapped, m.audio_heads);
        const rope_table = try created.create(max_total_rows * (head_dim / 2) * 8);
        try Backend.ropeTable(rope_table, max_total_rows, head_dim, model.rope_theta, null);

        var layers: [layer_count]Layer = undefined;
        for (&layers, m.layers) |*dst, src| {
            dst.* = .{
                .input_norm = try uploadVector(&created, src.input_norm),
                .post_norm = try uploadVector(&created, src.post_norm),
                .q_norm = try uploadVector(&created, src.q_norm),
                .k_norm = try uploadVector(&created, src.k_norm),
                .q_proj = try wrapMatrix(backend, &wrapped, src.q_proj),
                .k_proj = try wrapMatrix(backend, &wrapped, src.k_proj),
                .v_proj = try wrapMatrix(backend, &wrapped, src.v_proj),
                .o_proj = try wrapMatrix(backend, &wrapped, src.o_proj),
                .gate_proj = try wrapMatrix(backend, &wrapped, src.gate_proj),
                .up_proj = try wrapMatrix(backend, &wrapped, src.up_proj),
                .down_proj = try wrapMatrix(backend, &wrapped, src.down_proj),
            };
        }

        // Transfer ownership of created buffer handles to Plan.
        var owned_created: std.ArrayList(Buffer) = .empty;
        try owned_created.appendSlice(gpa, created.buffers.items);
        created.buffers.clearRetainingCapacity();

        return .{
            .gpa = gpa,
            .backend = backend,
            .model = m,
            .wrapped = wrapped,
            .created = owned_created,
            .output_norm = output_norm,
            .audio_heads = audio_heads,
            .rope_table = rope_table,
            .layers = layers,
        };
    }

    pub fn deinit(self: *Plan) void {
        self.scratch.release(self.backend);
        while (self.created.pop()) |b| self.backend.release(b) catch {};
        self.created.deinit(self.gpa);
        for (self.wrapped.items) |slice| self.backend.unwrap(slice);
        self.wrapped.deinit(self.gpa);
        self.* = undefined;
    }

    /// Runs ONE packed `[cond ‖ uncond]` bidirectional Qwen3-0.6B pass in ONE
    /// command buffer, writing `c_logits_out` (`[target_frames * 8200]`) and
    /// optional `u_logits_out` (`[target_frames * 8200]`).
    pub fn forwardPacked(
        self: *Plan,
        cond_rows: []const runtime.Row,
        uncond_rows: ?[]const runtime.Row,
        target_frames: usize,
        c_logits_out: []f32,
        u_logits_out: ?[]f32,
    ) !void {
        const s_cond = cond_rows.len;
        const s_uncond = if (uncond_rows) |ur| ur.len else 0;
        const total = s_cond + s_uncond;
        if (s_cond == 0 or total > max_total_rows or target_frames == 0 or target_frames > s_cond)
            return error.InvalidShape;
        if (s_uncond > 0 and target_frames > s_uncond) return error.InvalidShape;
        if (c_logits_out.len != target_frames * audio_table_rows) return error.InvalidShape;
        if (u_logits_out) |ul| if (s_uncond == 0 or ul.len != target_frames * audio_table_rows) return error.InvalidShape;

        const target_rows: usize = if (s_uncond > 0) target_frames * 2 else target_frames;
        const padded_attn = Backend.attentionChunkRows(total);
        // Upper bound on attention score buffer for one chunk covering all rows.
        const max_span = std.mem.alignForward(usize, total, 32);
        const max_head_stride = padded_attn * max_span;
        const needed_scores = heads * max_head_stride + heads * max_head_stride / 16;

        try self.scratch.ensure(self.backend, total, target_rows, needed_scores);

        const x_buf = self.scratch.x.?;
        const normed = self.scratch.normed.?;
        const q_buf = self.scratch.q.?;
        const k_buf = self.scratch.k.?;
        const v_buf = self.scratch.v.?;
        const attn_out = self.scratch.attn_out.?;
        const proj = self.scratch.proj.?;
        const gate = self.scratch.gate.?;
        const up = self.scratch.up.?;
        const bounds_buf = self.scratch.bounds.?;
        const tiles_buf = self.scratch.tiles.?;
        const scores_buf = self.scratch.scores.?;
        const target_h = self.scratch.target_h.?;
        const target_logits = self.scratch.target_logits.?;

        // 1. Fill embeddings on host into shared `x_buf`.
        const x_floats = x_buf.floats();
        try self.model.embedRows(cond_rows, x_floats[0 .. s_cond * hidden]);
        if (uncond_rows) |ur| {
            try self.model.embedRows(ur, x_floats[s_cond * hidden .. total * hidden]);
        }

        // 2. Fill per-row segment bounds `[begin, end)` and attention tiles.
        const bounds_u32: []u32 = @as([*]u32, @ptrCast(@alignCast(bounds_buf.host)))[0 .. total * 2];
        for (0..s_cond) |r| {
            bounds_u32[2 * r] = 0;
            bounds_u32[2 * r + 1] = @intCast(s_cond);
        }
        for (0..s_uncond) |r| {
            bounds_u32[2 * (s_cond + r)] = @intCast(s_cond);
            bounds_u32[2 * (s_cond + r) + 1] = @intCast(total);
        }

        const tiles_slice: []Backend.AttentionTile = @as([*]Backend.AttentionTile, @ptrCast(@alignCast(tiles_buf.host)))[0 .. tiles_buf.len / @sizeOf(Backend.AttentionTile)];
        var tile_count: usize = 0;
        try Backend.attentionTiles(bounds_u32, total, null, 0, padded_attn, tiles_slice, &tile_count);
        var chunks: [64]Backend.AttentionChunk = undefined;
        const chunk_count = try Backend.chunkTiles(tiles_slice[0..tile_count], heads, self.scratch.scores_capacity, &chunks);

        const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
        const b = self.backend;

        // 3. Record all 28 layers + final norm + target audio_heads in ONE command buffer.
        try b.begin();
        errdefer b.discard();

        for (self.layers) |layer| {
            try b.rmsNorm(x_buf, layer.input_norm, normed, .{ .rows = total, .width = hidden, .in_stride = hidden, .out_stride = hidden, .eps = model.rms_norm_eps });
            try b.matmulTile(layer.q_proj.buffer, layer.q_proj.matrix, normed, hidden, q_buf, query_width, total);
            try b.matmulTile(layer.k_proj.buffer, layer.k_proj.matrix, normed, hidden, k_buf, kv_width, total);
            try b.matmulTile(layer.v_proj.buffer, layer.v_proj.matrix, normed, hidden, v_buf, kv_width, total);

            // Per-head Q/K RMSNorm in place.
            try b.rmsNorm(q_buf, layer.q_norm, q_buf, .{ .rows = total * heads, .width = head_dim, .in_stride = head_dim, .out_stride = head_dim, .eps = model.rms_norm_eps });
            try b.rmsNorm(k_buf, layer.k_norm, k_buf, .{ .rows = total * kv_heads, .width = head_dim, .in_stride = head_dim, .out_stride = head_dim, .eps = model.rms_norm_eps });

            // Per-segment NeoX RoPE (positions 0..S_cond and 0..S_uncond).
            try b.ropeRows(q_buf, self.rope_table, heads, head_dim, head_dim, 0, s_cond, query_width, .split_half);
            try b.ropeRows(k_buf, self.rope_table, kv_heads, head_dim, head_dim, 0, s_cond, kv_width, .split_half);
            if (s_uncond > 0) {
                try b.ropeRows(q_buf.slice(s_cond * query_width * 4, s_uncond * query_width * 4), self.rope_table, heads, head_dim, head_dim, 0, s_uncond, query_width, .split_half);
                try b.ropeRows(k_buf.slice(s_cond * kv_width * 4, s_uncond * kv_width * 4), self.rope_table, kv_heads, head_dim, head_dim, 0, s_uncond, kv_width, .split_half);
            }

            // Materialized tiled bidirectional GQA over packed segments.
            for (chunks[0..chunk_count]) |chunk| {
                const chunk_tiles = tiles_buf.slice(chunk.first * @sizeOf(Backend.AttentionTile), chunk.count * @sizeOf(Backend.AttentionTile));
                try b.attentionSegmentsGrouped(q_buf, k_buf, v_buf, attn_out, bounds_buf, chunk_tiles, chunk, scores_buf, .{
                    .query_heads = heads,
                    .kv_heads = kv_heads,
                    .width = head_dim,
                    .rows = total,
                    .q_stride = query_width,
                    .kv_stride = kv_width,
                    .out_stride = query_width,
                    .scale = attn_scale,
                    .window = null,
                });
            }

            try b.matmulTile(layer.o_proj.buffer, layer.o_proj.matrix, attn_out, query_width, proj, hidden, total);
            try b.addRmsNorm(x_buf, proj, layer.post_norm, normed, .{ .rows = total, .width = hidden, .in_stride = hidden, .out_stride = hidden, .eps = model.rms_norm_eps });
            try b.matmulTile(layer.gate_proj.buffer, layer.gate_proj.matrix, normed, hidden, gate, intermediate, total);
            try b.matmulTile(layer.up_proj.buffer, layer.up_proj.matrix, normed, hidden, up, intermediate, total);
            try b.siluMul(gate, up, total * intermediate);
            try b.matmulTile(layer.down_proj.buffer, layer.down_proj.matrix, gate, intermediate, proj, hidden, total);
            try b.add(x_buf, proj, total * hidden);
        }

        // Final RMSNorm across all rows.
        try b.rmsNorm(x_buf, self.output_norm, x_buf, .{ .rows = total, .width = hidden, .in_stride = hidden, .out_stride = hidden, .eps = model.rms_norm_eps });

        // Pack target tail rows for audio_heads projection.
        const cond_tail_off = (s_cond - target_frames) * hidden * 4;
        try b.copy(target_h, x_buf.slice(cond_tail_off, target_frames * hidden * 4), target_frames * hidden);
        if (s_uncond > 0) {
            const uncond_tail_off = (s_cond + s_uncond - target_frames) * hidden * 4;
            try b.copy(target_h.slice(target_frames * hidden * 4, target_frames * hidden * 4), x_buf.slice(uncond_tail_off, target_frames * hidden * 4), target_frames * hidden);
        }

        try b.matmulTile(self.audio_heads.buffer, self.audio_heads.matrix, target_h, hidden, target_logits, audio_table_rows, target_rows);
        try b.commit();

        const out_floats = target_logits.floats();
        @memcpy(c_logits_out, out_floats[0 .. target_frames * audio_table_rows]);
        if (u_logits_out) |ul| {
            @memcpy(ul, out_floats[target_frames * audio_table_rows ..][0 .. target_frames * audio_table_rows]);
        }
    }

    /// Observed single-sequence forward pass for stage-by-stage trace validation
    /// against the CPU reference (`omnivoice-trace-metal`).
    pub fn forwardObserved(
        self: *Plan,
        rows: []const runtime.Row,
        target_tail: usize,
        logits_out: []f32,
        observer: ?runtime.Observer,
    ) !void {
        const s_len = rows.len;
        if (s_len == 0 or s_len > max_total_rows or target_tail == 0 or target_tail > s_len) return error.InvalidShape;
        if (logits_out.len != target_tail * audio_table_rows) return error.InvalidShape;

        const padded_attn = Backend.attentionChunkRows(s_len);
        const max_span = std.mem.alignForward(usize, s_len, 32);
        const head_stride = padded_attn * max_span;
        const needed_scores = heads * head_stride + heads * head_stride / 16;
        try self.scratch.ensure(self.backend, s_len, target_tail, needed_scores);

        const x_buf = self.scratch.x.?;
        const normed = self.scratch.normed.?;
        const q_buf = self.scratch.q.?;
        const k_buf = self.scratch.k.?;
        const v_buf = self.scratch.v.?;
        const attn_out = self.scratch.attn_out.?;
        const proj = self.scratch.proj.?;
        const gate = self.scratch.gate.?;
        const up = self.scratch.up.?;
        const bounds_buf = self.scratch.bounds.?;
        const tiles_buf = self.scratch.tiles.?;
        const scores_buf = self.scratch.scores.?;
        const target_h = self.scratch.target_h.?;
        const target_logits = self.scratch.target_logits.?;

        const x_floats = x_buf.floats();
        try self.model.embedRows(rows, x_floats[0 .. s_len * hidden]);
        if (observer) |o| o.record(o.context, .embed, null, x_floats[0 .. s_len * hidden]);

        const bounds_u32: []u32 = @as([*]u32, @ptrCast(@alignCast(bounds_buf.host)))[0 .. s_len * 2];
        for (0..s_len) |r| {
            bounds_u32[2 * r] = 0;
            bounds_u32[2 * r + 1] = @intCast(s_len);
        }
        const tiles_slice: []Backend.AttentionTile = @as([*]Backend.AttentionTile, @ptrCast(@alignCast(tiles_buf.host)))[0 .. tiles_buf.len / @sizeOf(Backend.AttentionTile)];
        var tile_count: usize = 0;
        try Backend.attentionTiles(bounds_u32, s_len, null, 0, padded_attn, tiles_slice, &tile_count);
        var chunks: [64]Backend.AttentionChunk = undefined;
        const chunk_count = try Backend.chunkTiles(tiles_slice[0..tile_count], heads, self.scratch.scores_capacity, &chunks);

        const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));
        const b = self.backend;

        for (self.layers, 0..) |layer, li| {
            try b.begin();
            errdefer b.discard();
            try b.rmsNorm(x_buf, layer.input_norm, normed, .{ .rows = s_len, .width = hidden, .in_stride = hidden, .out_stride = hidden, .eps = model.rms_norm_eps });
            try b.matmulTile(layer.q_proj.buffer, layer.q_proj.matrix, normed, hidden, q_buf, query_width, s_len);
            try b.matmulTile(layer.k_proj.buffer, layer.k_proj.matrix, normed, hidden, k_buf, kv_width, s_len);
            try b.matmulTile(layer.v_proj.buffer, layer.v_proj.matrix, normed, hidden, v_buf, kv_width, s_len);
            try b.rmsNorm(q_buf, layer.q_norm, q_buf, .{ .rows = s_len * heads, .width = head_dim, .in_stride = head_dim, .out_stride = head_dim, .eps = model.rms_norm_eps });
            try b.rmsNorm(k_buf, layer.k_norm, k_buf, .{ .rows = s_len * kv_heads, .width = head_dim, .in_stride = head_dim, .out_stride = head_dim, .eps = model.rms_norm_eps });
            try b.ropeRows(q_buf, self.rope_table, heads, head_dim, head_dim, 0, s_len, query_width, .split_half);
            try b.ropeRows(k_buf, self.rope_table, kv_heads, head_dim, head_dim, 0, s_len, kv_width, .split_half);
            for (chunks[0..chunk_count]) |chunk| {
                const chunk_tiles = tiles_buf.slice(chunk.first * @sizeOf(Backend.AttentionTile), chunk.count * @sizeOf(Backend.AttentionTile));
                try b.attentionSegmentsGrouped(q_buf, k_buf, v_buf, attn_out, bounds_buf, chunk_tiles, chunk, scores_buf, .{
                    .query_heads = heads,
                    .kv_heads = kv_heads,
                    .width = head_dim,
                    .rows = s_len,
                    .q_stride = query_width,
                    .kv_stride = kv_width,
                    .out_stride = query_width,
                    .scale = attn_scale,
                    .window = null,
                });
            }
            try b.matmulTile(layer.o_proj.buffer, layer.o_proj.matrix, attn_out, query_width, proj, hidden, s_len);
            try b.addRmsNorm(x_buf, proj, layer.post_norm, normed, .{ .rows = s_len, .width = hidden, .in_stride = hidden, .out_stride = hidden, .eps = model.rms_norm_eps });
            try b.matmulTile(layer.gate_proj.buffer, layer.gate_proj.matrix, normed, hidden, gate, intermediate, s_len);
            try b.matmulTile(layer.up_proj.buffer, layer.up_proj.matrix, normed, hidden, up, intermediate, s_len);
            try b.siluMul(gate, up, s_len * intermediate);
            try b.matmulTile(layer.down_proj.buffer, layer.down_proj.matrix, gate, intermediate, proj, hidden, s_len);
            try b.add(x_buf, proj, s_len * hidden);
            try b.commit();

            if (observer) |o| o.record(o.context, .layer, li, x_floats[0 .. s_len * hidden]);
        }

        try b.begin();
        errdefer b.discard();
        try b.rmsNorm(x_buf, self.output_norm, x_buf, .{ .rows = s_len, .width = hidden, .in_stride = hidden, .out_stride = hidden, .eps = model.rms_norm_eps });
        const tail_off = (s_len - target_tail) * hidden * 4;
        try b.copy(target_h, x_buf.slice(tail_off, target_tail * hidden * 4), target_tail * hidden);
        try b.matmulTile(self.audio_heads.buffer, self.audio_heads.matrix, target_h, hidden, target_logits, audio_table_rows, target_tail);
        try b.commit();

        if (observer) |o| o.record(o.context, .norm, null, x_floats[0 .. s_len * hidden]);
        @memcpy(logits_out, target_logits.floats()[0 .. target_tail * audio_table_rows]);
        if (observer) |o| o.record(o.context, .logits, null, logits_out);
    }

    /// Runs iterative masked-diffusion generation on Metal (one packed command
    /// buffer per diffusion step, cancellation checked between steps).
    pub fn generate(
        self: *Plan,
        gpa: Allocator,
        prompt_ids: []const u32,
        ref_codes: ?[]const hc.Frame,
        target_frames: usize,
        cfg: runtime.SamplingConfig,
        cancel: ?*const std.atomic.Value(bool),
    ) ![]hc.Frame {
        if (target_frames == 0 or cfg.num_step == 0) return error.InvalidShape;
        const ref_len = if (ref_codes) |rc| rc.len else 0;
        const c_len = prompt_ids.len + ref_len + target_frames;
        const u_len: usize = if (cfg.guidance_scale != 0.0) target_frames else 0;
        if (c_len + u_len > max_total_rows) return error.InvalidShape;

        const schedule = try runtime.buildSchedule(gpa, target_frames, cfg.num_step, cfg.t_shift);
        defer gpa.free(schedule);

        const cond_rows = try gpa.alloc(runtime.Row, c_len);
        defer gpa.free(cond_rows);
        for (prompt_ids, 0..) |id, i| cond_rows[i] = .{ .text = id };
        if (ref_codes) |rc| {
            for (rc, 0..) |frame, i| cond_rows[prompt_ids.len + i] = .{ .audio = frame };
        }
        const masked: hc.Frame = @splat(runtime.audio_mask_id);
        for (0..target_frames) |t| cond_rows[prompt_ids.len + ref_len + t] = .{ .audio = masked };

        const uncond_rows = if (u_len > 0) blk: {
            const ur = try gpa.alloc(runtime.Row, u_len);
            for (ur) |*r| r.* = .{ .audio = masked };
            break :blk ur;
        } else null;
        defer if (uncond_rows) |ur| gpa.free(ur);

        const tokens = try gpa.alloc(hc.Frame, target_frames);
        errdefer gpa.free(tokens);
        for (tokens) |*f| f.* = masked;

        const c_logits = try gpa.alloc(f32, target_frames * audio_table_rows);
        defer gpa.free(c_logits);
        const u_logits = if (u_len > 0) try gpa.alloc(f32, target_frames * audio_table_rows) else null;
        defer if (u_logits) |ul| gpa.free(ul);

        var prng = std.Random.DefaultPrng.init(cfg.seed);
        const rng = prng.random();

        for (schedule) |k| {
            if (cancel) |flag| if (flag.load(.acquire)) return error.Canceled;
            if (k == 0) continue;

            try self.forwardPacked(cond_rows, uncond_rows, target_frames, c_logits, u_logits);
            try runtime.unmaskStep(gpa, c_logits, u_logits, tokens, k, cfg, rng);

            for (0..target_frames) |t| {
                cond_rows[prompt_ids.len + ref_len + t] = .{ .audio = tokens[t] };
                if (uncond_rows) |ur| ur[t] = .{ .audio = tokens[t] };
            }
        }
        return tokens;
    }
};

fn uploadVector(created: *metal.Created, values: []const f32) !Buffer {
    const b = try created.create(values.len * 4);
    @memcpy(b.floats()[0..values.len], values);
    return b;
}

fn wrapMatrix(backend: *Backend, wrapped: *std.ArrayList([]const u8), m: cpu.Matrix) !Weight {
    const b = try backend.wrap(m.bytes);
    wrapped.appendAssumeCapacity(m.bytes);
    return .{ .buffer = b, .matrix = m };
}
