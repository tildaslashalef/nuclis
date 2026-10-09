//! EmbeddingGemma 2 on the CPU: the numerical reference for the family,
//! written from docs/models/embeddinggemma.md § Forward pass. One input is one
//! bidirectional pass in F32 with no cache, from its rows to the pooled unit
//! vector. The binding and the mapped weights are borrowed; each `embed` call
//! allocates its own workspace (about 0.4 GB at 8192 rows) and frees it.
//!
//! Matrices are decoded from their encoded rows into one reused F32 scratch
//! right before each `dense.matmul`, so the model costs no resident F32 copy;
//! `token_embd` rows are decoded one gathered row at a time. Attention is
//! `dense.attention` with grouped-query heads: every key on a global layer,
//! |i − j| ≤ 512 on a sliding one.
const std = @import("std");
const model = @import("embeddinggemma.zig");
const weights = @import("../runtime/weights.zig");
const cpu = @import("../backends/cpu/root.zig");
const dense = cpu.dense;
const quant = @import("../quant/decode.zig");
const gguf = @import("../formats/gguf.zig");
const Span = @import("../vision/root.zig").Span;
const Tensor = gguf.Tensor;

const d = model.config.embedding;
const w = model.config.per_layer_input;
const ff = model.config.feed_forward;
const out_width = model.config.embedding_out;
/// The widest matrix, the scratch every decode reuses.
const max_matrix = ff * d;

/// The stages an observer sees, each as `[rows][width]` row-major.
pub const Stage = union(enum) {
    /// x0: token rows × √512, projector rows as given (`inp_scaled`).
    input,
    /// Layer l's per-layer input, `per_layer_input` wide (a slice of `inp_per_layer`).
    per_layer: usize,
    /// The residual after layer l (`l_out-l`).
    layer: usize,
    /// rmsnorm(x) · output_norm (`result_norm`).
    final_norm,
    /// The output projection of every row, `embedding_out` wide (`result_embd`).
    projected,
};
pub const Observer = struct {
    context: *anyopaque,
    record: *const fn (context: *anyopaque, stage: Stage, rows: []const f32) void,
};

/// One input's rows. A span's tokens are placeholders: its rows are the
/// matching `d`-wide rows of `features`, in span order.
pub const Input = struct {
    tokens: []const u32,
    spans: []const Span = &.{},
    features: []const f32 = &.{},
};

/// The F32 vectors read once at `init`: norms and per-layer scales.
const LayerConstants = struct {
    attention_norm: []f32,
    post_attention_norm: []f32,
    ffn_norm: []f32,
    post_ffn_norm: []f32,
    per_layer_post_norm: []f32,
    query_norm: []f32,
    key_norm: []f32,
    output_scale: f32,
};

pub const Runtime = struct {
    gpa: std.mem.Allocator,
    io: std.Io,
    view: weights.View,
    binding: *const model.Binding,
    /// Owns the constants below.
    arena: std.heap.ArenaAllocator,
    layers: [model.layer_count]LayerConstants,
    per_layer_norm: []f32,
    output_norm: []f32,
    /// One decoded matrix at a time, `max_matrix` values.
    matrix: []f32,

    /// `binding` and the mapping behind `view` must outlive the runtime.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, view: weights.View, binding: *const model.Binding) !Runtime {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        var layers: [model.layer_count]LayerConstants = undefined;
        for (&layers, binding.layers) |*c, layer| c.* = .{
            .attention_norm = try view.vector(a, layer.attention_norm),
            .post_attention_norm = try view.vector(a, layer.post_attention_norm),
            .ffn_norm = try view.vector(a, layer.ffn_norm),
            .post_ffn_norm = try view.vector(a, layer.post_ffn_norm),
            .per_layer_post_norm = try view.vector(a, layer.per_layer_post_norm),
            .query_norm = try view.vector(a, layer.query_norm),
            .key_norm = try view.vector(a, layer.key_norm),
            .output_scale = try view.scalar(layer.output_scale, 0),
        };
        return .{
            .gpa = gpa,
            .io = io,
            .view = view,
            .binding = binding,
            .layers = layers,
            .per_layer_norm = try view.vector(a, binding.per_layer_norm),
            .output_norm = try view.vector(a, binding.output_norm),
            .matrix = try a.alloc(f32, max_matrix),
            .arena = arena,
        };
    }

    pub fn deinit(self: *Runtime) void {
        self.arena.deinit();
        self.* = undefined;
    }

    /// Writes the unit vector (`embedding_out` values) of one input. Errors
    /// leave `vector` unspecified; the runtime stays usable.
    pub fn embed(self: *Runtime, input: Input, vector: []f32, observer: ?Observer) !void {
        const rows = input.tokens.len;
        if (rows == 0 or rows > model.max_tokens) return error.InvalidShape;
        if (vector.len != out_width) return error.InvalidShape;
        var arena: std.heap.ArenaAllocator = .init(self.gpa);
        defer arena.deinit();
        var ws = try Workspace.init(arena.allocator(), rows);
        try inputRows(self.view, self.binding, input, ws.x0);
        @memcpy(ws.x, ws.x0);
        if (observer) |o| o.record(o.context, .input, ws.x0);
        for (self.binding.layers, self.layers, 0..) |layer, constants, il| {
            try self.attention(&ws, layer, constants);
            try self.feedForward(&ws, layer, constants);
            try self.perLayer(&ws, layer, constants, il, observer);
            for (ws.x) |*v| v.* *= constants.output_scale;
            if (observer) |o| o.record(o.context, .{ .layer = il }, ws.x);
        }
        try normRows(ws.x, ws.h, self.output_norm);
        if (observer) |o| o.record(o.context, .final_norm, ws.h);
        try self.linear(self.binding.output, ws.h, rows, ws.projected);
        if (observer) |o| o.record(o.context, .projected, ws.projected);
        try pool(ws.projected, rows, vector);
    }

    fn attention(self: *Runtime, ws: *Workspace, layer: model.Layer, constants: LayerConstants) !void {
        const rows = ws.rows;
        const hd = layer.kind.headSize();
        const qw = layer.queryWidth();
        const kvw = layer.kvWidth();
        const q = ws.q[0 .. rows * qw];
        const k = ws.k[0 .. rows * kvw];
        const v = ws.v[0 .. rows * kvw];
        try normRows(ws.x, ws.h, constants.attention_norm);
        try self.linear(layer.query, ws.h, rows, q);
        try self.linear(layer.key, ws.h, rows, k);
        try self.linear(layer.value, ws.h, rows, v);
        for (0..rows) |r| {
            const rope: cpu.rope.Options = .{ .dimensions = hd, .base = layer.kind.ropeBase(), .position = @intCast(r) };
            for (0..qw / hd) |h| {
                const head = q[r * qw + h * hd ..][0..hd];
                try normRow(head, head, constants.query_norm);
                try cpu.rope.apply(head, head, rope);
            }
            for (0..kvw / hd) |h| {
                const key = k[r * kvw + h * hd ..][0..hd];
                try normRow(key, key, constants.key_norm);
                try cpu.rope.apply(key, key, rope);
                const value = v[r * kvw + h * hd ..][0..hd];
                try cpu.rmsNorm(value, value, model.rms_epsilon);
            }
        }
        const attended = ws.attended[0 .. rows * qw];
        try dense.attention(self.io, .{
            .tokens = rows,
            .heads = model.config.heads,
            .head_dim = hd,
            .window = if (layer.kind == .sliding) model.config.half_window else null,
            .scale = 1,
            .stride = qw,
            .kv_heads = layer.kind.kvHeads(),
            .kv_stride = kvw,
            .q = q,
            .k = k,
            .v = v,
        }, attended, ws.scores);
        try self.linear(layer.output, attended, rows, ws.h);
        try addNormed(ws.x, ws.h, constants.post_attention_norm);
    }

    fn feedForward(self: *Runtime, ws: *Workspace, layer: model.Layer, constants: LayerConstants) !void {
        try normRows(ws.x, ws.h, constants.ffn_norm);
        try self.linear(layer.ffn_gate, ws.h, ws.rows, ws.gate);
        try self.linear(layer.ffn_up, ws.h, ws.rows, ws.up);
        for (ws.gate, ws.up) |*g, u| g.* = cpu.gelu(g.*) * u;
        try self.linear(layer.ffn_down, ws.gate, ws.rows, ws.h);
        try addNormed(ws.x, ws.h, constants.post_ffn_norm);
    }

    /// x += rmsnorm(proj · (gelu(inp_gate · x) ⊙ ple[l])) · post_norm, where
    /// ple[l] = rmsnorm(P[l] · x0 / √d) · per_layer_norm.
    fn perLayer(self: *Runtime, ws: *Workspace, layer: model.Layer, constants: LayerConstants, il: usize, observer: ?Observer) !void {
        try self.linearRows(self.binding.per_layer_projection, il * w, w, ws.x0, ws.rows, ws.ple);
        const inv: f32 = 1 / @sqrt(@as(f32, d));
        for (ws.ple) |*v| v.* *= inv;
        try normRows(ws.ple, ws.ple, self.per_layer_norm);
        if (observer) |o| o.record(o.context, .{ .per_layer = il }, ws.ple);
        try self.linear(layer.per_layer_gate, ws.x, ws.rows, ws.pe);
        for (ws.pe, ws.ple) |*g, e| g.* = cpu.gelu(g.*) * e;
        try self.linear(layer.per_layer_projection, ws.pe, ws.rows, ws.h);
        try addNormed(ws.x, ws.h, constants.per_layer_post_norm);
    }

    /// y = x · Wᵀ over every output row of `tensor`.
    fn linear(self: *Runtime, tensor: *const Tensor, x: []const f32, rows: usize, y: []f32) !void {
        const m = try self.view.matrix(tensor);
        try self.linearRows(tensor, 0, m.rows, x, rows, y);
    }

    /// y = x · W[first .. first + count]ᵀ: decode those rows, then one matmul.
    fn linearRows(self: *Runtime, tensor: *const Tensor, first: usize, count: usize, x: []const f32, rows: usize, y: []f32) !void {
        const m = try self.view.matrix(tensor);
        if (first + count > m.rows or count * m.columns > self.matrix.len) return error.InvalidShape;
        const decoded = self.matrix[0 .. count * m.columns];
        try decodeRows(self.io, m, first, count, decoded);
        try dense.matmul(self.io, .{ .rows = rows, .inner = m.columns, .outputs = count, .x = x[0 .. rows * m.columns], .w = decoded }, y[0 .. rows * count]);
    }
};

/// x0 (`input.tokens.len` rows of `d`): each token row decoded from
/// `token_embd` and scaled by √d, each span row copied from `features`
/// unscaled. Shared with the Metal plan.
pub fn inputRows(view: weights.View, binding: *const model.Binding, input: Input, x0: []f32) !void {
    const scale: f32 = @sqrt(@as(f32, d));
    var span: usize = 0;
    var feature: usize = 0;
    var i: usize = 0;
    while (i < input.tokens.len) {
        if (span < input.spans.len and input.spans[span].start == i) {
            const s = input.spans[span];
            if (s.count == 0 or i + s.count > input.tokens.len or (feature + s.count) * d > input.features.len) return error.InvalidShape;
            @memcpy(x0[i * d ..][0 .. s.count * d], input.features[feature * d ..][0 .. s.count * d]);
            i += s.count;
            feature += s.count;
            span += 1;
            continue;
        }
        const token = input.tokens[i];
        if (token >= model.vocabulary) return error.InvalidToken;
        const row = x0[i * d ..][0..d];
        try view.row(binding.token_embedding, token, row);
        for (row) |*v| v.* *= scale;
        i += 1;
    }
    if (span != input.spans.len or feature * d != input.features.len) return error.InvalidShape;
}

/// The unit vector of `rows` projected rows (`embedding_out` wide): the mean
/// over every row, then unit length, as sentence-transformers' Pooling
/// (`include_prompt`) and Normalize modules do. Shared with the Metal plan.
pub fn pool(projected: []const f32, rows: usize, vector: []f32) !void {
    if (rows == 0 or projected.len < rows * out_width or vector.len != out_width) return error.InvalidShape;
    var sum: [out_width]f64 = @splat(0);
    for (0..rows) |r| for (&sum, projected[r * out_width ..][0..out_width]) |*s, v| {
        s.* += v;
    };
    var norm: f64 = 0;
    for (&sum) |*s| {
        s.* /= @floatFromInt(rows);
        norm += s.* * s.*;
    }
    norm = @sqrt(norm);
    if (!(norm > 0) or !std.math.isFinite(norm)) return error.NonFiniteOutput;
    for (vector, sum) |*v, s| v.* = @floatCast(s / norm);
}

/// Every buffer one pass needs, sized for its rows.
const Workspace = struct {
    rows: usize,
    x: []f32,
    x0: []f32,
    h: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    attended: []f32,
    scores: []f32,
    gate: []f32,
    up: []f32,
    ple: []f32,
    pe: []f32,
    projected: []f32,

    fn init(a: std.mem.Allocator, rows: usize) !Workspace {
        const q_max = model.config.heads * model.Kind.global.headSize();
        const kv_max = @max(model.Kind.sliding.kvHeads() * model.Kind.sliding.headSize(), model.Kind.global.kvHeads() * model.Kind.global.headSize());
        return .{
            .rows = rows,
            .x = try a.alloc(f32, rows * d),
            .x0 = try a.alloc(f32, rows * d),
            .h = try a.alloc(f32, rows * d),
            .q = try a.alloc(f32, rows * q_max),
            .k = try a.alloc(f32, rows * kv_max),
            .v = try a.alloc(f32, rows * kv_max),
            .attended = try a.alloc(f32, rows * q_max),
            .scores = try a.alloc(f32, rows * model.config.heads),
            .gate = try a.alloc(f32, rows * ff),
            .up = try a.alloc(f32, rows * ff),
            .ple = try a.alloc(f32, rows * w),
            .pe = try a.alloc(f32, rows * w),
            .projected = try a.alloc(f32, rows * out_width),
        };
    }
};

/// out = rmsnorm(x) · weight, row by row; `out` may alias `x`.
fn normRow(x: []const f32, out: []f32, weight: []const f32) !void {
    try cpu.rmsNorm(x, out, model.rms_epsilon);
    for (out, weight) |*o, g| o.* *= g;
}
fn normRows(x: []const f32, out: []f32, weight: []const f32) !void {
    const width = weight.len;
    for (0..x.len / width) |r| try normRow(x[r * width ..][0..width], out[r * width ..][0..width], weight);
}
/// x += rmsnorm(y) · weight, row by row; y is overwritten.
fn addNormed(x: []f32, y: []f32, weight: []const f32) !void {
    try normRows(y, y, weight);
    for (x, y) |*a, b| a.* += b;
}

/// Decodes rows `[first, first + count)` of `m` into `out`, split across `io`
/// tasks; the decode is exact per row, so the split changes no value.
fn decodeRows(io: std.Io, m: cpu.Matrix, first: usize, count: usize, out: []f32) !void {
    if (m.rows == 0 or m.bytes.len % m.rows != 0) return error.InvalidShape;
    const row_bytes = m.bytes.len / m.rows;
    const tasks = @max(1, @min(count / 64, std.Thread.getCpuCount() catch 1, max_decode_tasks));
    var failures: [max_decode_tasks]?quant.Error = @splat(null);
    var group: std.Io.Group = .init;
    for (0..tasks) |t| {
        const lo = count * t / tasks;
        const hi = count * (t + 1) / tasks;
        group.async(io, decodeRange, .{ m, row_bytes, first, lo, hi, out, &failures[t] });
    }
    try group.await(io);
    for (failures[0..tasks]) |failure| if (failure) |err| return err;
}
const max_decode_tasks = 16;

fn decodeRange(m: cpu.Matrix, row_bytes: usize, first: usize, lo: usize, hi: usize, out: []f32, failure: *?quant.Error) void {
    for (lo..hi) |r| {
        quant.row(m.encoding, m.bytes[(first + r) * row_bytes ..][0..row_bytes], out[r * m.columns ..][0..m.columns]) catch |err| {
            failure.* = err;
            return;
        };
    }
}

test "a sliding layer sees keys 512 positions away and not 513" {
    // Equal scores make each query's output the mean of its visible values;
    // one value row is 1, the rest 0, so the output says whether it was seen.
    const gpa = std.testing.allocator;
    const tokens = 2 * model.config.half_window + 2;
    const dim = 8;
    const q = try gpa.alloc(f32, tokens * dim);
    defer gpa.free(q);
    @memset(q, 0);
    const v = try gpa.alloc(f32, tokens * dim);
    defer gpa.free(v);
    const out = try gpa.alloc(f32, tokens * dim);
    defer gpa.free(out);
    var scores: [tokens]f32 = undefined;
    for ([_]usize{ model.config.half_window, model.config.half_window + 1 }) |distance| {
        @memset(v, 0);
        v[distance * dim] = 1;
        try dense.attention(std.testing.io, .{ .tokens = tokens, .heads = 1, .head_dim = dim, .window = model.config.half_window, .scale = 1, .stride = dim, .q = q, .k = q, .v = v }, out, &scores);
        // Query 0 sees keys 0 … half_window.
        const seen = out[0];
        if (distance <= model.config.half_window) {
            try std.testing.expectApproxEqAbs(1.0 / @as(f32, model.config.half_window + 1), seen, 1e-7);
        } else try std.testing.expectEqual(@as(f32, 0), seen);
    }
}
