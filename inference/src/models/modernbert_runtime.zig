//! F32 CPU forward of a ModernBERT encoder over one unpadded sequence:
//! `[len]` ids → `[len][hidden]` states after `final_norm`. `load` decodes
//! every weight but the embedding table to F32 once (about 1.4 GB for
//! ModernBERT-large); embedding rows are decoded from the checkpoint as ids
//! need them, so the checkpoint must outlive the Model. Kernels are
//! backends/cpu/dense.zig; RoPE angles are rounded to F32 before their cosine
//! and sine, as the reference computes them. Contract: docs/reference/laya.md.
const std = @import("std");
const safetensors = @import("../formats/safetensors.zig");
const dense = @import("../backends/cpu/dense.zig");
const vector = @import("../backends/cpu/vector.zig");
const modernbert = @import("modernbert.zig");

pub const Error = std.mem.Allocator.Error || dense.Error || error{ UnsupportedDtype, TensorOutOfBounds, InvalidTokenId, InvalidShape };

/// Observes the residual stream: after each layer (`layer` 0..layers−1,
/// before `final_norm`) and once more after it (`layer == null`).
pub const Trace = struct {
    context: *anyopaque,
    record: *const fn (context: *anyopaque, layer: ?usize, hidden: []const f32) void,
};

const Layer = struct {
    attn_norm: ?[]const f32,
    qkv: []const f32,
    out: []const f32,
    mlp_norm: []const f32,
    wi: []const f32,
    wo: []const f32,
};

pub const Model = struct {
    gpa: std.mem.Allocator,
    config: modernbert.Config,
    storage: []f32,
    layers: []Layer,
    embeddings: safetensors.Ref,
    embed_norm: []const f32,
    final_norm: []const f32,

    pub fn load(gpa: std.mem.Allocator, weights: modernbert.Weights, config: modernbert.Config) Error!Model {
        var total: usize = 2 * config.hidden;
        for (weights.layers) |l| {
            if (l.attn_norm) |n| total += n.tensor.elements;
            total += l.qkv.tensor.elements + l.out.tensor.elements + l.mlp_norm.tensor.elements + l.wi.tensor.elements + l.wo.tensor.elements;
        }
        const storage = try gpa.alloc(f32, total);
        errdefer gpa.free(storage);
        const layers = try gpa.alloc(Layer, weights.layers.len);
        errdefer gpa.free(layers);
        var cursor: Cursor = .{ .storage = storage };
        for (layers, weights.layers) |*l, w| l.* = .{
            .attn_norm = if (w.attn_norm) |n| try cursor.take(n) else null,
            .qkv = try cursor.take(w.qkv),
            .out = try cursor.take(w.out),
            .mlp_norm = try cursor.take(w.mlp_norm),
            .wi = try cursor.take(w.wi),
            .wo = try cursor.take(w.wo),
        };
        const embed_norm = try cursor.take(weights.embed_norm);
        const final_norm = try cursor.take(weights.final_norm);
        return .{ .gpa = gpa, .config = config, .storage = storage, .layers = layers, .embeddings = weights.embeddings, .embed_norm = embed_norm, .final_norm = final_norm };
    }

    pub fn deinit(self: *Model) void {
        self.gpa.free(self.layers);
        self.gpa.free(self.storage);
        self.* = undefined;
    }

    /// Writes the encoder's output for `ids` into `hidden` ([ids.len][hidden]).
    /// Scratch comes from `gpa` and is freed before returning.
    pub fn forward(self: *const Model, io: std.Io, gpa: std.mem.Allocator, ids: []const u32, hidden: []f32, trace: ?Trace) Error!void {
        const c = self.config;
        const n = ids.len;
        const h = c.hidden;
        const hd = c.headDim();
        if (n == 0 or hidden.len != n * h) return error.InvalidShape;
        const scratch = try Scratch.init(gpa, c, n);
        defer scratch.deinit(gpa);
        for (ids, 0..) |id, t| {
            if (id >= c.vocabulary) return error.InvalidTokenId;
            const row = hidden[t * h ..][0..h];
            try self.embeddings.decode(@as(u64, id) * h, row);
            try dense.layerNorm(row, self.embed_norm, null, c.norm_eps, row);
        }
        const global = Rope.init(scratch.cos_global, scratch.sin_global, n, hd, c.global_theta);
        const local = Rope.init(scratch.cos_local, scratch.sin_local, n, hd, c.local_theta);
        for (self.layers, 0..) |l, i| {
            const is_global = c.global.isSet(i);
            const input = if (l.attn_norm) |w| blk: {
                for (0..n) |t| try dense.layerNorm(hidden[t * h ..][0..h], w, null, c.norm_eps, scratch.normed[t * h ..][0..h]);
                break :blk scratch.normed;
            } else hidden;
            try dense.matmul(io, .{ .rows = n, .inner = h, .outputs = 3 * h, .x = input, .w = l.qkv }, scratch.qkv);
            (if (is_global) global else local).apply(scratch.qkv, n, c.heads, h);
            try dense.attention(io, .{
                .tokens = n,
                .heads = c.heads,
                .head_dim = hd,
                .window = if (is_global) null else c.window,
                .scale = 1 / @sqrt(@as(f32, @floatFromInt(hd))),
                .stride = 3 * h,
                .q = scratch.qkv,
                .k = scratch.qkv[h..],
                .v = scratch.qkv[2 * h ..],
            }, scratch.attended, scratch.scores);
            try dense.matmul(io, .{ .rows = n, .inner = h, .outputs = h, .x = scratch.attended, .w = l.out }, scratch.projected);
            for (hidden, scratch.projected) |*x, p| x.* += p;
            for (0..n) |t| try dense.layerNorm(hidden[t * h ..][0..h], l.mlp_norm, null, c.norm_eps, scratch.normed[t * h ..][0..h]);
            try dense.matmul(io, .{ .rows = n, .inner = h, .outputs = 2 * c.intermediate, .x = scratch.normed, .w = l.wi }, scratch.wide);
            for (0..n) |t| {
                const row = scratch.wide[t * 2 * c.intermediate ..][0 .. 2 * c.intermediate];
                const gated = scratch.gated[t * c.intermediate ..][0..c.intermediate];
                for (gated, row[0..c.intermediate], row[c.intermediate..]) |*g, a, gate| g.* = vector.geluErf(a) * gate;
            }
            try dense.matmul(io, .{ .rows = n, .inner = c.intermediate, .outputs = h, .x = scratch.gated, .w = l.wo }, scratch.projected);
            for (hidden, scratch.projected) |*x, p| x.* += p;
            if (trace) |tr| tr.record(tr.context, i, hidden);
        }
        for (0..n) |t| try dense.layerNorm(hidden[t * h ..][0..h], self.final_norm, null, c.norm_eps, hidden[t * h ..][0..h]);
        if (trace) |tr| tr.record(tr.context, null, hidden);
    }
};

/// Hands out consecutive ranges of one allocation, each decoded from a tensor.
const Cursor = struct {
    storage: []f32,
    used: usize = 0,

    fn take(self: *Cursor, ref: safetensors.Ref) Error![]const f32 {
        const len: usize = @intCast(ref.tensor.elements);
        const out = self.storage[self.used..][0..len];
        try ref.decode(0, out);
        self.used += len;
        return out;
    }
};

const Scratch = struct {
    block: []f32,
    normed: []f32,
    qkv: []f32,
    attended: []f32,
    projected: []f32,
    wide: []f32,
    gated: []f32,
    scores: []f32,
    cos_global: []f32,
    sin_global: []f32,
    cos_local: []f32,
    sin_local: []f32,

    fn init(gpa: std.mem.Allocator, c: modernbert.Config, n: usize) Error!Scratch {
        const h = c.hidden;
        const half = c.headDim() / 2;
        const sizes = [_]usize{ n * h, 3 * n * h, n * h, n * h, 2 * n * c.intermediate, n * c.intermediate, c.heads * n, n * half, n * half, n * half, n * half };
        var total: usize = 0;
        for (sizes) |s| total += s;
        const block = try gpa.alloc(f32, total);
        var parts: [sizes.len][]f32 = undefined;
        var at: usize = 0;
        for (&parts, sizes) |*p, s| {
            p.* = block[at..][0..s];
            at += s;
        }
        return .{ .block = block, .normed = parts[0], .qkv = parts[1], .attended = parts[2], .projected = parts[3], .wide = parts[4], .gated = parts[5], .scores = parts[6], .cos_global = parts[7], .sin_global = parts[8], .cos_local = parts[9], .sin_local = parts[10] };
    }

    fn deinit(self: Scratch, gpa: std.mem.Allocator) void {
        gpa.free(self.block);
    }
};

/// Split-half RoPE tables for positions 0..n: angle `pos · theta^(−2i/d)`
/// rounded to F32 (the reference's F32 product), then cosine and sine.
const Rope = struct {
    cos: []const f32,
    sin: []const f32,
    half: usize,

    fn init(cos: []f32, sin: []f32, n: usize, head_dim: usize, theta: f32) Rope {
        const half = head_dim / 2;
        for (0..half) |i| {
            const exponent = @as(f64, @floatFromInt(2 * i)) / @as(f64, @floatFromInt(head_dim));
            const inverse: f32 = 1.0 / @as(f32, @floatCast(std.math.pow(f64, theta, exponent)));
            for (0..n) |p| {
                const angle: f32 = inverse * @as(f32, @floatFromInt(p));
                cos[p * half + i] = @floatCast(@cos(@as(f64, angle)));
                sin[p * half + i] = @floatCast(@sin(@as(f64, angle)));
            }
        }
        return .{ .cos = cos[0 .. n * half], .sin = sin[0 .. n * half], .half = half };
    }

    /// Rotates the q and k thirds of each fused `[q | k | v]` row in place.
    fn apply(self: Rope, qkv: []f32, n: usize, heads: usize, hidden: usize) void {
        const d = 2 * self.half;
        for (0..n) |p| {
            const cos = self.cos[p * self.half ..][0..self.half];
            const sin = self.sin[p * self.half ..][0..self.half];
            for (0..2) |part| for (0..heads) |head| {
                const x = qkv[p * 3 * hidden + part * hidden + head * d ..][0..d];
                for (0..self.half) |i| {
                    const a = x[i];
                    const b = x[i + self.half];
                    x[i] = a * cos[i] - b * sin[i];
                    x[i + self.half] = b * cos[i] + a * sin[i];
                }
            };
        }
    }
};

test "rope tables rotate split-half pairs and leave position zero alone" {
    var cos: [3 * 4]f32 = undefined;
    var sin: [3 * 4]f32 = undefined;
    const rope = Rope.init(&cos, &sin, 3, 8, 10000);
    // One head, hidden 8: q = k = v = 1..8 at each of three positions.
    var qkv: [3 * 24]f32 = undefined;
    for (0..3) |p| for (0..24) |c| {
        qkv[p * 24 + c] = @floatFromInt(c % 8 + 1);
    };
    rope.apply(&qkv, 3, 1, 8);
    for (0..24) |c| try std.testing.expectEqual(@as(f32, @floatFromInt(c % 8 + 1)), qkv[c]);
    // Position 1, pair 0 (dims 0 and 4) turns by exactly 1 radian.
    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(1 * @cos(1.0) - 5 * @sin(1.0))), qkv[24], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(5 * @cos(1.0) + 1 * @sin(1.0))), qkv[28], 1e-6);
    // The value third is untouched.
    for (16..24) |c| try std.testing.expectEqual(@as(f32, @floatFromInt(c % 8 + 1)), qkv[24 + c]);
}
