//! Laya on Metal: the CPU forward's schedule (modernbert_runtime.zig, then
//! laya.zig's head) over packed batches. Sequences sit back to back, at most
//! `max_rows` rows; attention reads each row's `[begin, end)` from a bounds
//! buffer and RoPE a per-row table of positions within the sequence, so no
//! row is padding. Matrices are the checkpoint mapping's F16 or F32 bytes
//! wrapped in place (read by the generic F32 matmul tile); vectors are
//! decoded to F32 buffers. The plan owns its `Backend`, so every device
//! buffer goes with `destroy`; the checkpoint must outlive it.
//! Contract: docs/reference/laya.md.
const std = @import("std");
const metal = @import("../backends/metal/root.zig");
const cpu = @import("../backends/cpu/root.zig");
const safetensors = @import("../formats/safetensors.zig");
const modernbert = @import("modernbert.zig");
const runtime = @import("modernbert_runtime.zig");
const laya = @import("laya.zig");

const Backend = metal.Backend;
const Buffer = metal.Buffer;

/// Rows one batch holds: activations are sized for it (about 56 KB a row
/// for ModernBERT-large), and it bounds one sequence's length.
pub const max_rows = 2048;
const head_eps = 1e-5;
const head_width = 64;

const Weight = struct { buffer: Buffer, matrix: cpu.Matrix };
const Layer = struct {
    attn_norm: ?Buffer,
    qkv: Weight,
    out: Weight,
    mlp_norm: Buffer,
    wi: Weight,
    wo: Weight,
};
const HeadLayer = struct {
    norm1_w: Buffer,
    norm1_b: Buffer,
    in_proj: Weight,
    in_proj_b: Buffer,
    out_proj: Weight,
    out_proj_b: Buffer,
    norm2_w: Buffer,
    norm2_b: Buffer,
    linear1: Weight,
    linear1_b: Buffer,
    linear2: Weight,
    linear2_b: Buffer,
};

pub const Plan = struct {
    gpa: std.mem.Allocator,
    backend: Backend,
    config: modernbert.Config,
    ff: usize,
    embeddings: safetensors.Ref,
    embed_norm: Buffer,
    layers: []Layer,
    final_norm: Buffer,
    head: []HeadLayer,
    type_emb: Buffer,
    scorer_norm_w: Buffer,
    scorer_norm_b: Buffer,
    scorer: Weight,
    scorer_b: Buffer,
    /// `scorer.3`, applied on the host to the marker rows.
    out_w: []f32,
    out_b: f32,
    /// ModernBERT's norms have no bias; LayerNorm reads this instead.
    zeros: Buffer,
    // Activations, `Backend.matmulPadded(max_rows)` rows each.
    x: Buffer,
    normed: Buffer,
    qkv: Buffer,
    attended: Buffer,
    projected: Buffer,
    /// Wi's output (2 · intermediate) or the head's feed-forward rows.
    wide: Buffer,
    gated: Buffer,
    /// Per row: `[begin, end)` of its sequence, as u32 pairs.
    bounds: Buffer,
    /// Per row, (cos, sin) pairs at its position: global and local theta.
    rope_global: Buffer,
    rope_local: Buffer,
    /// Host tables for positions 0..max_rows, copied per sequence into the rope buffers.
    table_global: []f32,
    table_local: []f32,

    /// Builds the plan: a new Metal backend, the matrices wrapped, the
    /// vectors decoded, the activations created. `weights` and `head` borrow
    /// the checkpoint, which must outlive the plan.
    pub fn create(gpa: std.mem.Allocator, config: modernbert.Config, weights: modernbert.Weights, head: laya.HeadRefs) !*Plan {
        const self = try gpa.create(Plan);
        errdefer gpa.destroy(self);
        var diagnostic: [512]u8 = undefined;
        self.backend = try Backend.init(gpa, &diagnostic);
        errdefer self.backend.deinit();
        self.gpa = gpa;
        self.config = config;
        self.ff = head.ff;
        self.embeddings = weights.embeddings;
        const d = config.hidden;
        const half = config.headDim() / 2;

        self.layers = try gpa.alloc(Layer, weights.layers.len);
        errdefer gpa.free(self.layers);
        for (self.layers, weights.layers) |*l, w| l.* = .{
            .attn_norm = if (w.attn_norm) |n| try self.vector(n) else null,
            .qkv = try self.matrix(w.qkv),
            .out = try self.matrix(w.out),
            .mlp_norm = try self.vector(w.mlp_norm),
            .wi = try self.matrix(w.wi),
            .wo = try self.matrix(w.wo),
        };
        self.embed_norm = try self.vector(weights.embed_norm);
        self.final_norm = try self.vector(weights.final_norm);
        self.head = try gpa.alloc(HeadLayer, head.layers);
        errdefer gpa.free(self.head);
        for (self.head, 0..) |*l, i| l.* = .{
            .norm1_w = try self.vector(head.layer(i, "norm1.weight")),
            .norm1_b = try self.vector(head.layer(i, "norm1.bias")),
            .in_proj = try self.matrix(head.layer(i, "self_attn.in_proj_weight")),
            .in_proj_b = try self.vector(head.layer(i, "self_attn.in_proj_bias")),
            .out_proj = try self.matrix(head.layer(i, "self_attn.out_proj.weight")),
            .out_proj_b = try self.vector(head.layer(i, "self_attn.out_proj.bias")),
            .norm2_w = try self.vector(head.layer(i, "norm2.weight")),
            .norm2_b = try self.vector(head.layer(i, "norm2.bias")),
            .linear1 = try self.matrix(head.layer(i, "linear1.weight")),
            .linear1_b = try self.vector(head.layer(i, "linear1.bias")),
            .linear2 = try self.matrix(head.layer(i, "linear2.weight")),
            .linear2_b = try self.vector(head.layer(i, "linear2.bias")),
        };
        self.type_emb = try self.vector(head.refs[0]);
        self.scorer_norm_w = try self.vector(head.scorer("scorer.0.weight"));
        self.scorer_norm_b = try self.vector(head.scorer("scorer.0.bias"));
        self.scorer = try self.matrix(head.scorer("scorer.1.weight"));
        self.scorer_b = try self.vector(head.scorer("scorer.1.bias"));
        self.out_w = try gpa.alloc(f32, d);
        errdefer gpa.free(self.out_w);
        try head.scorer("scorer.3.weight").decode(0, self.out_w);
        var out_b: [1]f32 = undefined;
        try head.scorer("scorer.3.bias").decode(0, &out_b);
        self.out_b = out_b[0];

        const rows = Backend.matmulPadded(max_rows);
        self.zeros = try self.created(d);
        self.x = try self.created(rows * d);
        self.normed = try self.created(rows * d);
        self.qkv = try self.created(rows * 3 * d);
        self.attended = try self.created(rows * d);
        self.projected = try self.created(rows * d);
        self.wide = try self.created(rows * @max(2 * config.intermediate, head.ff));
        self.gated = try self.created(rows * config.intermediate);
        self.bounds = try self.created(max_rows * 2);
        self.rope_global = try self.created(max_rows * half * 2);
        self.rope_local = try self.created(max_rows * half * 2);

        self.table_global = try gpa.alloc(f32, 2 * max_rows * half);
        errdefer gpa.free(self.table_global);
        self.table_local = try gpa.alloc(f32, 2 * max_rows * half);
        errdefer gpa.free(self.table_local);
        for ([_][]f32{ self.table_global, self.table_local }, [_]f32{ config.global_theta, config.local_theta }) |table, theta| {
            const cos = try gpa.alloc(f32, max_rows * half);
            defer gpa.free(cos);
            const sin = try gpa.alloc(f32, max_rows * half);
            defer gpa.free(sin);
            // The CPU forward's own tables, interleaved as the kernel's (cos, sin) pairs.
            _ = runtime.Rope.init(cos, sin, max_rows, config.headDim(), theta);
            for (cos, sin, 0..) |c, s, k| {
                table[2 * k] = c;
                table[2 * k + 1] = s;
            }
        }
        return self;
    }

    /// A zeroed F32 buffer of `floats` values, the backend's.
    fn created(self: *Plan, floats: usize) !Buffer {
        const buffer = try self.backend.create(floats * 4);
        @memset(buffer.floats()[0..floats], 0);
        return buffer;
    }
    fn vector(self: *Plan, ref: safetensors.Ref) !Buffer {
        const len: usize = @intCast(ref.tensor.elements);
        const buffer = try self.backend.create(len * 4);
        try ref.decode(0, buffer.floats()[0..len]);
        return buffer;
    }
    fn matrix(self: *Plan, ref: safetensors.Ref) !Weight {
        const encoding: u32 = switch (ref.tensor.dtype) {
            .f32 => 0,
            .f16 => 1,
            else => return error.UnsupportedDtype,
        };
        const shape = ref.tensor.shape;
        const m: cpu.Matrix = .{ .encoding = encoding, .rows = @intCast(shape[0]), .columns = @intCast(shape[1]), .bytes = ref.bytes };
        // The matmul's tile constraints; every Laya matrix meets them.
        if (m.rows % 8 != 0 or m.columns % 64 != 0) return error.UnsupportedConfig;
        return .{ .buffer = try self.backend.wrap(ref.bytes), .matrix = m };
    }

    pub fn destroy(self: *Plan) void {
        const gpa = self.gpa;
        // Device buffers belong to the backend and go with it.
        self.backend.deinit();
        gpa.free(self.layers);
        gpa.free(self.head);
        gpa.free(self.out_w);
        gpa.free(self.table_global);
        gpa.free(self.table_local);
        gpa.destroy(self);
    }

    /// The logits of each sequence into `outs[i]` (`markers.len` values):
    /// one command buffer for the whole batch, or, with a `trace`, one per
    /// stage so the host can read each stage's rows (the batch's, packed).
    pub fn run(self: *Plan, sequences: []const laya.Sequence, outs: []const []f32, trace: ?laya.Trace) !void {
        const c = self.config;
        const d = c.hidden;
        const hd = c.headDim();
        const half = hd / 2;
        if (sequences.len == 0 or sequences.len != outs.len) return error.InvalidShape;
        var n: usize = 0;
        for (sequences, outs) |s, out| {
            if (s.ids.len == 0 or out.len != s.markers.len) return error.InvalidShape;
            for (s.markers) |m| if (m >= s.ids.len) return error.InvalidMarker;
            for (s.ids) |id| if (id >= c.vocabulary) return error.InvalidTokenId;
            n += s.ids.len;
        }
        if (n > max_rows) return error.SequenceTooLong;

        // Host inputs: embedding rows, each row's sequence bounds, and its rotary rows.
        const x = self.x.floats();
        const pairs: []u32 = @as([*]u32, @ptrCast(@alignCast(self.bounds.host)))[0 .. 2 * n];
        var begin: usize = 0;
        for (sequences) |s| {
            const len = s.ids.len;
            for (s.ids, 0..) |id, t| try self.embeddings.decode(@as(u64, id) * d, x[(begin + t) * d ..][0..d]);
            for (begin..begin + len) |r| {
                pairs[2 * r] = @intCast(begin);
                pairs[2 * r + 1] = @intCast(begin + len);
            }
            @memcpy(self.rope_global.floats()[2 * begin * half ..][0 .. 2 * len * half], self.table_global[0 .. 2 * len * half]);
            @memcpy(self.rope_local.floats()[2 * begin * half ..][0 .. 2 * len * half], self.table_local[0 .. 2 * len * half]);
            begin += len;
        }

        const b = &self.backend;
        const norm: Backend.Norm = .{ .rows = n, .width = d, .in_stride = d, .out_stride = d, .eps = c.norm_eps };
        const head_norm: Backend.Norm = .{ .rows = n, .width = d, .in_stride = d, .out_stride = d, .eps = head_eps };
        const inter = c.intermediate;
        try b.begin();
        errdefer b.discard();
        try b.layerNorm(self.x, self.embed_norm, self.zeros, self.x, norm);
        for (self.layers, 0..) |l, i| {
            const global = c.global.isSet(i);
            const input = if (l.attn_norm) |w| blk: {
                try b.layerNorm(self.x, w, self.zeros, self.normed, norm);
                break :blk self.normed;
            } else self.x;
            try b.matmul(l.qkv.buffer, l.qkv.matrix, input, d, self.qkv, 3 * d, n);
            // q and k are the first 2·heads heads of each fused row.
            try b.ropeRows(self.qkv, if (global) self.rope_global else self.rope_local, 2 * c.heads, hd, hd, 0, n, 3 * d, .split_half);
            try self.attention(n, c.heads, hd, if (global) null else c.window);
            try b.matmul(l.out.buffer, l.out.matrix, self.attended, d, self.projected, d, n);
            try b.add(self.x, self.projected, n * d);
            try b.layerNorm(self.x, l.mlp_norm, self.zeros, self.normed, norm);
            try b.matmul(l.wi.buffer, l.wi.matrix, self.normed, d, self.wide, 2 * inter, n);
            try b.geluErfMulRows(self.wide, self.wide.slice(inter * 4, self.wide.len - inter * 4), self.gated, inter, n, 2 * inter, 2 * inter, inter);
            try b.matmul(l.wo.buffer, l.wo.matrix, self.gated, inter, self.projected, d, n);
            try b.add(self.x, self.projected, n * d);
            try self.stage(trace, .{ .encoder = i }, n);
        }
        try b.layerNorm(self.x, self.final_norm, self.zeros, self.x, norm);
        try self.stage(trace, .final, n);

        begin = 0;
        for (sequences) |s| {
            const rows = s.ids.len;
            try b.addBiasRows(self.x.slice(begin * d * 4, rows * d * 4), self.type_emb.slice(@as(usize, @backingInt(s.kind)) * d * 4, d * 4), d, rows, d);
            begin += rows;
        }
        for (self.head, 0..) |l, i| {
            try b.layerNorm(self.x, l.norm1_w, l.norm1_b, self.normed, head_norm);
            try b.matmul(l.in_proj.buffer, l.in_proj.matrix, self.normed, d, self.qkv, 3 * d, n);
            try b.addBiasRows(self.qkv, l.in_proj_b, 3 * d, n, 3 * d);
            try self.attention(n, d / head_width, head_width, null);
            try b.matmul(l.out_proj.buffer, l.out_proj.matrix, self.attended, d, self.projected, d, n);
            try b.addBiasRows(self.projected, l.out_proj_b, d, n, d);
            try b.add(self.x, self.projected, n * d);
            try b.layerNorm(self.x, l.norm2_w, l.norm2_b, self.normed, head_norm);
            try b.matmul(l.linear1.buffer, l.linear1.matrix, self.normed, d, self.wide, self.ff, n);
            try b.addBiasRows(self.wide, l.linear1_b, self.ff, n, self.ff);
            try b.clamp(self.wide, n * self.ff, 0, std.math.inf(f32));
            try b.matmul(l.linear2.buffer, l.linear2.matrix, self.wide, self.ff, self.projected, d, n);
            try b.addBiasRows(self.projected, l.linear2_b, d, n, d);
            try b.add(self.x, self.projected, n * d);
            try self.stage(trace, .{ .head = i }, n);
        }
        // The scorer on every row (the marker rows are a few of them): its
        // LayerNorm, Linear, and GELU here, the last Linear on the host.
        try b.layerNorm(self.x, self.scorer_norm_w, self.scorer_norm_b, self.normed, head_norm);
        try b.matmul(self.scorer.buffer, self.scorer.matrix, self.normed, d, self.projected, d, n);
        try b.addBiasRows(self.projected, self.scorer_b, d, n, d);
        try b.geluErf(self.projected, n * d);
        try b.commit();

        const scored = self.projected.floats();
        begin = 0;
        for (sequences, outs) |s, out| {
            for (s.markers, out) |m, *logit| {
                var sum: f64 = self.out_b;
                for (scored[(begin + m) * d ..][0..d], self.out_w) |v, w| sum += @as(f64, v) * w;
                logit.* = @floatCast(sum);
            }
            begin += s.ids.len;
        }
    }

    fn attention(self: *Plan, n: usize, heads: usize, width: usize, window: ?usize) !void {
        const d = self.config.hidden;
        try self.backend.attentionSegments(self.qkv, self.qkv.slice(d * 4, self.qkv.len - d * 4), self.qkv.slice(2 * d * 4, self.qkv.len - 2 * d * 4), self.attended, self.bounds, .{
            .heads = heads,
            .width = width,
            .rows = n,
            .q_stride = 3 * d,
            .kv_stride = 3 * d,
            .out_stride = d,
            .scale = 1 / @sqrt(@as(f32, @floatFromInt(width))),
            .window = window,
        });
    }

    /// With a trace: finishes the stage's command buffer, shows the host the
    /// residual rows, and starts the next.
    fn stage(self: *Plan, trace: ?laya.Trace, at: laya.Stage, n: usize) !void {
        const t = trace orelse return;
        try self.backend.commit();
        t.record(t.context, at, self.x.floats()[0 .. n * self.config.hidden]);
        try self.backend.begin();
    }
};
