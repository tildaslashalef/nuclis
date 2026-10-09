//! Gemma 4's audio encoder (`gemma4a`), the conformer in Gemma 4 E4B's
//! and EmbeddingGemma 2's companion files: log-mel frames through two
//! strided 3×3 convolutions (4× fewer rows), 12 conformer blocks (two
//! half-step FFNs, chunked local attention with relative positions and a
//! logit cap, a causal depthwise convolution), an output projection, and
//! the embedder's norm and projection to the language model's width. One
//! row per 40 ms. `bind` validates the companion file's `a.*` and `mm.a.*`
//! tensors; `Runtime` is the CPU reference, written from Google's
//! `Gemma4AudioModel` (docs/engine/audio.md); `subsample` and
//! `relativeKeys` are shared with the Metal plan (`gemma4a_metal.zig`).
const std = @import("std");
const gguf = @import("../formats/gguf.zig");
const weights = @import("../runtime/weights.zig");
const cpu = @import("../backends/cpu/root.zig");
const quant = @import("../quant/decode.zig");
const qwen3vl = @import("../vision/qwen3vl.zig");
const gemma4v = @import("../vision/gemma4.zig");
const mel = @import("mel.zig");

const Tensor = gguf.Tensor;
const dense = cpu.dense;
pub const Error = qwen3vl.Error;
pub const Clamp = gemma4v.Clamp;
pub const Bounds = gemma4v.Bounds;
/// The Metal plan, the same schedule on the device.
pub const Plan = @import("gemma4a_metal.zig").Plan;

pub const hidden = 1024;
pub const heads = 8;
pub const head_dim = hidden / heads;
pub const ffn = 4 * hidden;
pub const blocks = 12;
/// `a.pre_encode.out`'s width (Google's `output_proj_dims`).
pub const output_dims = 1536;
pub const channels = [2]usize{ 128, 32 };
/// Frequency columns after the two stride-2 convolutions.
pub const sub_columns = mel.filters / 4;
pub const sub_width = sub_columns * channels[1];
/// Chunked attention: queries in blocks of `chunk` with `left` rows of
/// context before each block (Google's `attention_context_left − 1`) and
/// none after. The mask is narrower: a query sees the keys at distance
/// 0 … `window − 1` before it (`sliding_window_mask_function`'s
/// `dist < left`), so it sees itself and 11 rows back.
pub const chunk = 12;
pub const left = 12;
pub const window = left;
pub const context = chunk + left;
/// Relative position rows: distances `left` down to 0 (distance `left`
/// is computed by the reference and always masked).
pub const relative = left + 1;
pub const logit_cap: f32 = 50;
pub const residual_weight: f32 = 0.5;
pub const conv_kernel = 5;
pub const epsilon: f32 = 1e-6;

/// Rows the encoder gives for `frames` mel frames: two stride-2 halvings.
pub fn tokensFor(frames: usize) usize {
    return (((frames + 1) / 2) + 1) / 2;
}

/// The matrices that clamp (`Clamp`), in `Block.clamps` order.
pub const Linear = enum { attn_q, attn_k, attn_v, attn_out, ffn_up, ffn_down, ffn_up_1, ffn_down_1, conv_pw1, conv_pw2 };

pub const Block = struct {
    ffn_norm: *const Tensor,
    ffn_up: *const Tensor,
    ffn_down: *const Tensor,
    ffn_post_norm: *const Tensor,
    attn_pre_norm: *const Tensor,
    attn_q: *const Tensor,
    attn_k: *const Tensor,
    attn_v: *const Tensor,
    attn_out: *const Tensor,
    /// Already `softplus(per_dim_scale)` in the file (`[head_dim]`).
    per_dim_scale: *const Tensor,
    attn_k_rel: *const Tensor,
    attn_post_norm: *const Tensor,
    /// The light convolution's norms. The file's names are the other way
    /// round: its `conv_norm` holds Google's `pre_layer_norm`, its
    /// `norm_conv` the norm after the convolution (checked by value).
    conv_pre_norm: *const Tensor,
    conv_pw1: *const Tensor,
    /// F32 `[hidden][conv_kernel]`.
    conv_dw: *const Tensor,
    conv_post_norm: *const Tensor,
    conv_pw2: *const Tensor,
    ffn_norm_1: *const Tensor,
    ffn_up_1: *const Tensor,
    ffn_down_1: *const Tensor,
    ffn_post_norm_1: *const Tensor,
    out_norm: *const Tensor,
    clamps: [@typeInfo(Linear).@"enum".field_names.len]Clamp,

    pub fn matrix(self: Block, which: Linear) *const Tensor {
        return switch (which) {
            inline else => |tag| @field(self, @tagName(tag)),
        };
    }
};

pub const Binding = struct {
    /// F32 `[channels[0]][1][3][3]` and `[channels[1]][channels[0]][3][3]`.
    conv: [2]*const Tensor,
    conv_norm: [2]*const Tensor,
    /// F32 `[hidden][sub_width]`.
    input_projection: *const Tensor,
    layers: [blocks]Block,
    output: *const Tensor,
    output_bias: *const Tensor,
    projection: *const Tensor,
    /// The language model width the projection writes.
    output_width: usize,
    tensors: u32,
    bytes: u64,
};

/// Whether `doc` carries a `gemma4a` audio encoder.
pub fn present(doc: *const gguf.Document) bool {
    const value = doc.get("clip.audio.projector_type") orelse return false;
    return value == .string and std.mem.eql(u8, value.string, "gemma4a");
}

fn expectUnsigned(doc: *const gguf.Document, key: []const u8, value: u64) Error!void {
    if (try qwen3vl.unsignedValue(doc, key) != value) return error.UnsupportedConfiguration;
}

fn takeClamp(binder: *qwen3vl.Binder, block: usize, name: []const u8) Error!Clamp {
    var result: Clamp = .{};
    inline for (.{ "input_min", "input_max", "output_min", "output_max" }) |field| {
        var buffer: [96]u8 = undefined;
        const key = std.fmt.bufPrint(&buffer, "a.blk.{d}.{s}.{s}", .{ block, name, field }) catch unreachable;
        if (binder.remaining.contains(key)) @field(result, field) = try binder.take(key, &.{1}, true);
    }
    return result;
}

/// doc must come from successful GGUF parsing; the returned tensor
/// references borrow it. Only `a.*` and `mm.a.*` are bound; a file's vision
/// tensors are another binding's. Allocations are freed before return.
pub fn bind(alloc: std.mem.Allocator, doc: *const gguf.Document) Error!Binding {
    if (!std.mem.eql(u8, try qwen3vl.stringValue(doc, "general.architecture"), "clip")) return error.UnsupportedArchitecture;
    if (!present(doc)) return error.UnsupportedProjector;
    switch (doc.get("clip.audio.attention.layer_norm_epsilon") orelse return error.MissingMetadata) {
        .float => |f| if (@abs(f - epsilon) > 1e-9) return error.UnsupportedConfiguration,
        else => return error.InvalidMetadata,
    }
    try expectUnsigned(doc, "clip.audio.embedding_length", hidden);
    try expectUnsigned(doc, "clip.audio.feed_forward_length", ffn);
    try expectUnsigned(doc, "clip.audio.block_count", blocks);
    try expectUnsigned(doc, "clip.audio.attention.head_count", heads);
    try expectUnsigned(doc, "clip.audio.num_mel_bins", mel.filters);
    const width = try qwen3vl.unsignedValue(doc, "clip.audio.projection_dim");
    if (width == 0 or width > 16384) return error.UnsupportedConfiguration;

    var binder: qwen3vl.Binder = .{ .remaining = .init(alloc) };
    defer binder.remaining.deinit();
    for (doc.tensors) |*tensor| {
        if (!std.mem.startsWith(u8, tensor.name, "a.") and !std.mem.startsWith(u8, tensor.name, "mm.a.")) continue;
        const slot = try binder.remaining.getOrPut(tensor.name);
        if (slot.found_existing) return error.DuplicateTensor;
        slot.value_ptr.* = tensor;
    }
    var result: Binding = undefined;
    result.output_width = @intCast(width);
    result.conv = .{
        try binder.take("a.conv1d.0.weight", &.{ 3, 3, 1, channels[0] }, true),
        try binder.take("a.conv1d.1.weight", &.{ 3, 3, channels[0], channels[1] }, true),
    };
    result.conv_norm = .{
        try binder.take("a.conv1d.0.norm.weight", &.{channels[0]}, true),
        try binder.take("a.conv1d.1.norm.weight", &.{channels[1]}, true),
    };
    result.input_projection = try binder.take("a.input_projection.weight", &.{ sub_width, hidden }, false);
    const h = hidden;
    for (&result.layers, 0..) |*layer, i| layer.* = .{
        .ffn_norm = try binder.named("a.blk.{d}.ffn_norm.weight", .{i}, &.{h}, true),
        .ffn_up = try binder.named("a.blk.{d}.ffn_up.weight", .{i}, &.{ h, ffn }, false),
        .ffn_down = try binder.named("a.blk.{d}.ffn_down.weight", .{i}, &.{ ffn, h }, false),
        .ffn_post_norm = try binder.named("a.blk.{d}.ffn_post_norm.weight", .{i}, &.{h}, true),
        .attn_pre_norm = try binder.named("a.blk.{d}.attn_pre_norm.weight", .{i}, &.{h}, true),
        .attn_q = try binder.named("a.blk.{d}.attn_q.weight", .{i}, &.{ h, h }, false),
        .attn_k = try binder.named("a.blk.{d}.attn_k.weight", .{i}, &.{ h, h }, false),
        .attn_v = try binder.named("a.blk.{d}.attn_v.weight", .{i}, &.{ h, h }, false),
        .attn_out = try binder.named("a.blk.{d}.attn_out.weight", .{i}, &.{ h, h }, false),
        .per_dim_scale = try binder.named("a.blk.{d}.per_dim_scale.weight", .{i}, &.{head_dim}, true),
        .attn_k_rel = try binder.named("a.blk.{d}.attn_k_rel.weight", .{i}, &.{ h, h }, false),
        .attn_post_norm = try binder.named("a.blk.{d}.attn_post_norm.weight", .{i}, &.{h}, true),
        .conv_pre_norm = try binder.named("a.blk.{d}.conv_norm.weight", .{i}, &.{h}, true),
        .conv_pw1 = try binder.named("a.blk.{d}.conv_pw1.weight", .{i}, &.{ h, 2 * h }, false),
        .conv_dw = try binder.named("a.blk.{d}.conv_dw.weight", .{i}, &.{ conv_kernel, h }, true),
        .conv_post_norm = try binder.named("a.blk.{d}.norm_conv.weight", .{i}, &.{h}, true),
        .conv_pw2 = try binder.named("a.blk.{d}.conv_pw2.weight", .{i}, &.{ h, h }, false),
        .ffn_norm_1 = try binder.named("a.blk.{d}.ffn_norm_1.weight", .{i}, &.{h}, true),
        .ffn_up_1 = try binder.named("a.blk.{d}.ffn_up_1.weight", .{i}, &.{ h, ffn }, false),
        .ffn_down_1 = try binder.named("a.blk.{d}.ffn_down_1.weight", .{i}, &.{ ffn, h }, false),
        .ffn_post_norm_1 = try binder.named("a.blk.{d}.ffn_post_norm_1.weight", .{i}, &.{h}, true),
        .out_norm = try binder.named("a.blk.{d}.ln2.weight", .{i}, &.{h}, true),
        .clamps = blk: {
            var clamps: [@typeInfo(Linear).@"enum".field_names.len]Clamp = undefined;
            inline for (@typeInfo(Linear).@"enum".field_names, 0..) |name, k| clamps[k] = try takeClamp(&binder, i, name);
            break :blk clamps;
        },
    };
    result.output = try binder.take("a.pre_encode.out.weight", &.{ h, output_dims }, false);
    result.output_bias = try binder.take("a.pre_encode.out.bias", &.{output_dims}, true);
    result.projection = try binder.take("mm.a.input_projection.weight", &.{ output_dims, width }, false);
    if (binder.remaining.count() != 0) return error.UnexpectedTensor;
    result.tensors = binder.tensors;
    result.bytes = binder.bytes;
    return result;
}

/// The relative-position rows every block's keys are offset by: the
/// sinusoid of distance `left − r` for row r (sines, then cosines, over
/// 512 timescales from 1 to 10⁴), Google's `Gemma4AudioRelPositionalEncoding`
/// in F32. `out` holds `relative × hidden`.
pub fn positionRows(out: []f32) !void {
    if (out.len != relative * hidden) return error.InvalidShape;
    const half = hidden / 2;
    const increment: f32 = @floatCast(@log(@as(f64, 1e4)) / @as(f64, half - 1));
    for (0..relative) |r| {
        const position: f32 = @floatFromInt(left - r);
        for (0..half) |i| {
            const inv = @exp(@as(f32, @floatFromInt(i)) * -increment);
            const t = position * inv;
            out[r * hidden + i] = @sin(t);
            out[r * hidden + half + i] = @cos(t);
        }
    }
}

/// Decodes all of an encoded matrix into `out` (`rows × columns` F32).
pub fn decodeMatrix(view: weights.View, tensor: *const Tensor, out: []f32) !cpu.Matrix {
    const m = try view.matrix(tensor);
    if (out.len < m.rows * m.columns or m.bytes.len % m.rows != 0) return error.InvalidShape;
    const row_bytes = m.bytes.len / m.rows;
    for (0..m.rows) |r| try quant.row(m.encoding, m.bytes[r * row_bytes ..][0..row_bytes], out[r * m.columns ..][0..m.columns]);
    return m;
}

/// Each block's relative keys, `attn_k_rel · positionRows`: `blocks ×
/// relative × hidden` into `out`. `scratch` holds one decoded matrix.
pub fn relativeKeys(io: std.Io, view: weights.View, binding: *const Binding, out: []f32, scratch: []f32) !void {
    if (out.len != blocks * relative * hidden or scratch.len < hidden * hidden) return error.InvalidShape;
    var rows: [relative * hidden]f32 = undefined;
    try positionRows(&rows);
    for (binding.layers, 0..) |layer, i| {
        _ = try decodeMatrix(view, layer.attn_k_rel, scratch);
        try dense.matmul(io, .{ .rows = relative, .inner = hidden, .outputs = hidden, .x = &rows, .w = scratch[0 .. hidden * hidden] }, out[i * relative * hidden ..][0 .. relative * hidden]);
    }
}

/// The subsampling front of the encoder over `frames` mel rows: per layer a
/// 3×3, stride-2, pad-1 convolution, a LayerNorm over its channels (weight,
/// no bias) and a ReLU. Writes `tokensFor(frames) × sub_width` rows, each
/// frequency-major (`f · channels[1] + c`), which `input_projection` reads.
/// `scratch` holds `((frames + 1) / 2) · (mel.filters / 2) · channels[0]`.
pub fn subsample(view: weights.View, binding: *const Binding, features: []const f32, frames: usize, out: []f32, scratch: []f32) !void {
    const t1 = (frames + 1) / 2;
    const t2 = (t1 + 1) / 2;
    const f1 = mel.filters / 2;
    const f2 = sub_columns;
    if (frames == 0 or features.len != frames * mel.filters or out.len != t2 * sub_width or scratch.len < t1 * f1 * channels[0]) return error.InvalidShape;
    const w0 = try floats(view, binding.conv[0]);
    const n0 = try floats(view, binding.conv_norm[0]);
    const w1 = try floats(view, binding.conv[1]);
    const n1 = try floats(view, binding.conv_norm[1]);
    // Layer 0: one input channel, `[t1][f1][c]`.
    const first = scratch[0 .. t1 * f1 * channels[0]];
    var sums: [channels[0]]f64 = undefined;
    var row: [channels[0]]f32 = undefined;
    for (0..t1) |t| for (0..f1) |f| {
        for (&sums, 0..) |*s, o| {
            var acc: f64 = 0;
            for (0..3) |kh| for (0..3) |kw| {
                const y = @as(i64, @intCast(2 * t + kh)) - 1;
                const x = @as(i64, @intCast(2 * f + kw)) - 1;
                if (y < 0 or y >= frames or x < 0 or x >= mel.filters) continue;
                acc += @as(f64, w0[(o * 3 + kh) * 3 + kw]) * features[@as(usize, @intCast(y)) * mel.filters + @as(usize, @intCast(x))];
            };
            s.* = acc;
        }
        for (&row, sums) |*r, s| r.* = @floatCast(s);
        try normRelu(&row, n0, first[(t * f1 + f) * channels[0] ..][0..channels[0]]);
    };
    // Layer 1: `[t2][f2][c]`, already the flattened row order.
    var sums1: [channels[1]]f64 = undefined;
    var row1: [channels[1]]f32 = undefined;
    for (0..t2) |t| for (0..f2) |f| {
        @memset(&sums1, 0);
        for (0..3) |kh| for (0..3) |kw| {
            const y = @as(i64, @intCast(2 * t + kh)) - 1;
            const x = @as(i64, @intCast(2 * f + kw)) - 1;
            if (y < 0 or y >= t1 or x < 0 or x >= f1) continue;
            const input = first[(@as(usize, @intCast(y)) * f1 + @as(usize, @intCast(x))) * channels[0] ..][0..channels[0]];
            for (&sums1, 0..) |*s, o| {
                var acc: f64 = 0;
                for (input, 0..) |v, c| acc += @as(f64, w1[((o * channels[0] + c) * 3 + kh) * 3 + kw]) * v;
                s.* += acc;
            }
        };
        for (&row1, sums1) |*r, s| r.* = @floatCast(s);
        try normRelu(&row1, n1, out[t * sub_width + f * channels[1] ..][0..channels[1]]);
    };
}

fn normRelu(row: []const f32, weight: []const f32, out: []f32) !void {
    try dense.layerNorm(row, weight, null, epsilon, out);
    for (out) |*v| v.* = @max(v.*, 0);
}

fn floats(view: weights.View, tensor: *const Tensor) ![]const f32 {
    const bytes = try view.bytes(tensor);
    if (bytes.len % 4 != 0) return error.InvalidShape;
    return @as([*]const f32, @ptrCast(@alignCast(bytes.ptr)))[0 .. bytes.len / 4];
}

/// `x · scale` with `scale` a query's per-dimension scale: Google's
/// `q · head_dim^−½ / ln 2 · softplus(per_dim_scale)`.
pub fn queryScale(per_dim_scale: []const f32, out: []f32) void {
    const base: f32 = @floatCast((1.0 / @sqrt(@as(f64, head_dim))) / @log(2.0));
    for (out, per_dim_scale) |*o, p| o.* = base * p;
}
/// Google's key scale, `ln(1 + e) / ln 2`.
pub const key_scale: f32 = @floatCast(@log(1.0 + std.math.e) / @log(2.0));

/// Chunked local attention of `rows` rows for one head's slices: query i
/// sees keys `i − window + 1 … i` within the clip, scored
/// `q·k + q·relk[distance]`, capped by `logit_cap` through tanh, then a
/// softmax over those keys. `q` is already scaled; `relk` is the block's
/// `relative × hidden` keys. All of `q`, `k`, `v`, `out` are `[rows][hidden]`.
pub fn attention(q: []const f32, k: []const f32, v: []const f32, relk: []const f32, rows: usize, out: []f32) void {
    for (0..heads) |h| for (0..rows) |i| {
        const qi = q[i * hidden + h * head_dim ..][0..head_dim];
        var scores: [window]f64 = undefined;
        const lowest = i -| (window - 1);
        var max: f64 = -std.math.inf(f64);
        for (lowest..i + 1, 0..) |j, s| {
            const kj = k[j * hidden + h * head_dim ..][0..head_dim];
            const rel = relk[(left - (i - j)) * hidden + h * head_dim ..][0..head_dim];
            var ac: f64 = 0;
            var bd: f64 = 0;
            for (qi, kj, rel) |a, b, r| {
                ac += @as(f64, a) * b;
                bd += @as(f64, a) * r;
            }
            const raw: f32 = @floatCast(ac + bd);
            const capped = std.math.tanh(raw / logit_cap) * logit_cap;
            scores[s] = capped;
            max = @max(max, capped);
        }
        const count = i + 1 - lowest;
        var total: f64 = 0;
        for (scores[0..count]) |*s| {
            s.* = @exp(s.* - max);
            total += s.*;
        }
        const o = out[i * hidden + h * head_dim ..][0..head_dim];
        for (o, 0..) |*value, d| {
            var sum: f64 = 0;
            for (lowest..i + 1, 0..) |j, s| sum += scores[s] * v[j * hidden + h * head_dim + d];
            value.* = @floatCast(sum / total);
        }
    };
}

/// The light convolution's middle: the GLU of `start` (`rows × 2·hidden`:
/// values, then gates) through a causal depthwise convolution of
/// `conv_kernel` taps (`dw` is `[hidden][conv_kernel]`), into `out`.
pub fn gluConv(start: []const f32, dw: []const f32, rows: usize, out: []f32) void {
    for (0..rows) |t| for (0..hidden) |c| {
        var acc: f64 = 0;
        for (0..conv_kernel) |tap| {
            const s = @as(i64, @intCast(t + tap)) - (conv_kernel - 1);
            if (s < 0) continue;
            const r = start[@as(usize, @intCast(s)) * 2 * hidden ..];
            const glu = r[c] * sigmoid(r[hidden + c]);
            acc += @as(f64, dw[c * conv_kernel + tap]) * glu;
        }
        out[t * hidden + c] = @floatCast(acc);
    };
}

fn sigmoid(x: f32) f32 {
    return 1 / (1 + @exp(-x));
}

/// Sees the reference's stages, each `[rows][width]`: `subsample` (after
/// the input projection), block 0's `l0-ff1`, `l0-attn` (the attention's
/// output projection) and `l0-lconv` (the residual after each), `layer-<i>`
/// (each block's output), `output` (after the output projection),
/// `projected` (the embedder's rows).
pub const Observer = struct {
    context: *anyopaque,
    record: *const fn (context: *anyopaque, stage: []const u8, rows: []const f32) void,

    fn see(self: ?Observer, stage: []const u8, rows: []const f32) void {
        if (self) |o| o.record(o.context, stage, rows);
    }
};

/// The CPU reference: every matrix decoded to F32 and multiplied by
/// `dense.matmul`, norms and attention statistics in F64.
pub const Runtime = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    view: weights.View,
    binding: *const Binding,
    /// `blocks × relative × hidden`, from `relativeKeys`.
    relk: []f32,
    /// One decoded matrix, `ffn × hidden` values.
    matrix: []f32,

    pub fn init(alloc: std.mem.Allocator, io: std.Io, view: weights.View, binding: *const Binding) !Runtime {
        const matrix = try alloc.alloc(f32, @max(ffn * hidden, output_dims * hidden));
        errdefer alloc.free(matrix);
        const relk = try alloc.alloc(f32, blocks * relative * hidden);
        errdefer alloc.free(relk);
        try relativeKeys(io, view, binding, relk, matrix);
        return .{ .alloc = alloc, .io = io, .view = view, .binding = binding, .relk = relk, .matrix = matrix };
    }
    pub fn deinit(self: *Runtime) void {
        self.alloc.free(self.matrix);
        self.alloc.free(self.relk);
        self.* = undefined;
    }

    /// y = clamp(clamp(x) · Wᵀ (+ bias)) over `rows` rows; the input clamp
    /// goes to a copy, since Q, K and V read one input with their own bounds.
    fn linear(self: *Runtime, tensor: *const Tensor, bounds: Bounds, bias: ?[]const f32, x: []const f32, rows: usize, y: []f32) !void {
        const m = try decodeMatrix(self.view, tensor, self.matrix);
        const open = bounds.isOpen();
        var source = x[0 .. rows * m.columns];
        var staged: []f32 = &.{};
        defer self.alloc.free(staged);
        if (!open) {
            staged = try self.alloc.alloc(f32, source.len);
            for (staged, source) |*s, v| s.* = std.math.clamp(v, bounds.input[0], bounds.input[1]);
            source = staged;
        }
        try dense.matmul(self.io, .{ .rows = rows, .inner = m.columns, .outputs = m.rows, .x = source, .w = self.matrix[0 .. m.rows * m.columns], .bias = bias }, y[0 .. rows * m.rows]);
        if (!open) for (y[0 .. rows * m.rows]) |*v| {
            v.* = std.math.clamp(v.*, bounds.output[0], bounds.output[1]);
        };
    }

    fn clamped(self: *Runtime, layer: Block, which: Linear, x: []const f32, rows: usize, y: []f32) !void {
        const bounds = try gemma4v.clampBounds(self.view, layer.clamps[@backingInt(which)]);
        try self.linear(layer.matrix(which), bounds, null, x, rows, y);
    }

    fn norm(self: *Runtime, x: []const f32, tensor: ?*const Tensor, width: usize, out: []f32) !void {
        const weight = if (tensor) |t| try floats(self.view, t) else null;
        for (0..x.len / width) |r| {
            const o = out[r * width ..][0..width];
            try cpu.rmsNorm(x[r * width ..][0..width], o, epsilon);
            if (weight) |w| for (o, w) |*a, b| {
                a.* *= b;
            };
        }
    }

    /// Encodes `frames` mel rows (`frames × mel.filters`) into `out`:
    /// `tokensFor(frames) × output_width` rows.
    pub fn encode(self: *Runtime, features: []const f32, frames: usize, out: []f32) !void {
        return self.encodeObserved(features, frames, out, null);
    }

    /// `encode`, with `observer` shown each stage.
    pub fn encodeObserved(self: *Runtime, features: []const f32, frames: usize, out: []f32, observer: ?Observer) !void {
        const rows = tokensFor(frames);
        const width = self.binding.output_width;
        if (frames == 0 or out.len != rows * width) return error.InvalidShape;
        var arena: std.heap.ArenaAllocator = .init(self.alloc);
        defer arena.deinit();
        const ws = arena.allocator();
        const sub = try ws.alloc(f32, rows * sub_width);
        const scratch = try ws.alloc(f32, ((frames + 1) / 2) * (mel.filters / 2) * channels[0]);
        try subsample(self.view, self.binding, features, frames, sub, scratch);
        const x = try ws.alloc(f32, rows * hidden);
        try self.linear(self.binding.input_projection, Bounds.open, null, sub, rows, x);
        Observer.see(observer, "subsample", x);

        const n = try ws.alloc(f32, rows * hidden);
        const q = try ws.alloc(f32, rows * hidden);
        const k = try ws.alloc(f32, rows * hidden);
        const v = try ws.alloc(f32, rows * hidden);
        const wide = try ws.alloc(f32, rows * ffn);
        const scale = try ws.alloc(f32, head_dim);
        for (self.binding.layers, 0..) |layer, il| {
            try self.halfFeedForward(layer, .{ layer.ffn_norm, layer.ffn_post_norm }, .{ .ffn_up, .ffn_down }, x, n, wide, rows);
            if (il == 0) Observer.see(observer, "l0-ff1", x);
            // Attention.
            try self.norm(x, layer.attn_pre_norm, hidden, n);
            try self.clamped(layer, .attn_q, n, rows, q);
            try self.clamped(layer, .attn_k, n, rows, k);
            try self.clamped(layer, .attn_v, n, rows, v);
            queryScale(try floats(self.view, layer.per_dim_scale), scale);
            for (0..rows * heads) |r| for (q[r * head_dim ..][0..head_dim], scale) |*value, s| {
                value.* *= s;
            };
            for (k) |*value| value.* *= key_scale;
            attention(q, k, v, self.relk[il * relative * hidden ..][0 .. relative * hidden], rows, n);
            try self.clamped(layer, .attn_out, n, rows, q);
            if (il == 0) Observer.see(observer, "l0-attn", q);
            try self.norm(q, layer.attn_post_norm, hidden, q);
            for (x, q) |*a, b| a.* += b;
            // The light convolution.
            try self.norm(x, layer.conv_pre_norm, hidden, n);
            try self.clamped(layer, .conv_pw1, n, rows, wide[0 .. rows * 2 * hidden]);
            gluConv(wide[0 .. rows * 2 * hidden], try floats(self.view, layer.conv_dw), rows, q);
            try self.norm(q, layer.conv_post_norm, hidden, q);
            for (q) |*value| value.* = cpu.silu(value.*);
            try self.clamped(layer, .conv_pw2, q, rows, n);
            for (x, n) |*a, b| a.* += b;
            if (il == 0) Observer.see(observer, "l0-lconv", x);
            try self.halfFeedForward(layer, .{ layer.ffn_norm_1, layer.ffn_post_norm_1 }, .{ .ffn_up_1, .ffn_down_1 }, x, n, wide, rows);
            try self.norm(x, layer.out_norm, hidden, x);
            var name: [16]u8 = undefined;
            Observer.see(observer, std.fmt.bufPrint(&name, "layer-{d}", .{il}) catch unreachable, x);
        }
        const projected = try ws.alloc(f32, rows * output_dims);
        try self.linear(self.binding.output, Bounds.open, try floats(self.view, self.binding.output_bias), x, rows, projected);
        Observer.see(observer, "output", projected);
        try self.norm(projected, null, output_dims, projected);
        try self.linear(self.binding.projection, Bounds.open, null, projected, rows, out);
        Observer.see(observer, "projected", out);
    }

    /// x += ½ · rms(W_down · silu(W_up · rms(x)·pre))·post.
    fn halfFeedForward(self: *Runtime, layer: Block, norms: [2]*const Tensor, which: [2]Linear, x: []f32, n: []f32, wide: []f32, rows: usize) !void {
        try self.norm(x, norms[0], hidden, n);
        try self.clamped(layer, which[0], n, rows, wide);
        for (wide) |*value| value.* = cpu.silu(value.*);
        try self.clamped(layer, which[1], wide, rows, n);
        try self.norm(n, norms[1], hidden, n);
        for (x, n) |*a, b| a.* += residual_weight * b;
    }
};

test "the row count halves twice, rounding up" {
    try std.testing.expectEqual(@as(usize, 120), tokensFor(479));
    try std.testing.expectEqual(@as(usize, 88), tokensFor(350));
    try std.testing.expectEqual(@as(usize, 1), tokensFor(1));
    try std.testing.expectEqual(@as(usize, 750), tokensFor(2999));
}

test "the relative rows: distance 12 first, 0 last (sin 0, cos 1)" {
    var rows: [relative * hidden]f32 = undefined;
    try positionRows(&rows);
    try std.testing.expectEqual(@as(f32, 0), rows[left * hidden]);
    try std.testing.expectEqual(@as(f32, 1), rows[left * hidden + hidden / 2]);
    try std.testing.expectApproxEqAbs(@as(f32, @sin(12.0)), rows[0], 1e-6);
}

test "attention sees 11 rows back and none ahead" {
    // One-hot keys per row and values equal to the row index: a query
    // matching every key equally averages over exactly its window.
    const rows = 30;
    const q = try std.testing.allocator.alloc(f32, rows * hidden);
    defer std.testing.allocator.free(q);
    const k = try std.testing.allocator.alloc(f32, rows * hidden);
    defer std.testing.allocator.free(k);
    const v = try std.testing.allocator.alloc(f32, rows * hidden);
    defer std.testing.allocator.free(v);
    const out = try std.testing.allocator.alloc(f32, rows * hidden);
    defer std.testing.allocator.free(out);
    const relk = try std.testing.allocator.alloc(f32, relative * hidden);
    defer std.testing.allocator.free(relk);
    @memset(q, 0);
    @memset(k, 0);
    @memset(relk, 0);
    for (0..rows) |r| @memset(v[r * hidden ..][0..hidden], @floatFromInt(r));
    attention(q, k, v, relk, rows, out);
    // Row 20 averages rows 9..20; row 3 rows 0..3.
    try std.testing.expectApproxEqAbs(@as(f32, 14.5), out[20 * hidden], 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, 1.5), out[3 * hidden + 5 * head_dim], 1e-5);
}

test "the light convolution is causal: a row sees itself and four before" {
    const rows = 8;
    var start: [rows * 2 * hidden]f32 = undefined;
    // Values 1 at row 2 only, gates large so sigmoid ≈ 1.
    @memset(&start, 0);
    for (0..rows) |t| @memset(start[t * 2 * hidden + hidden ..][0..hidden], 40);
    @memset(start[2 * 2 * hidden ..][0..hidden], 1);
    var dw: [hidden * conv_kernel]f32 = undefined;
    for (0..hidden) |c| for (0..conv_kernel) |tap| {
        dw[c * conv_kernel + tap] = @floatFromInt(tap + 1);
    };
    var out: [rows * hidden]f32 = undefined;
    gluConv(&start, &dw, rows, &out);
    // Row 2 reaches rows 2..6 with taps 5, 4, 3, 2, 1; rows 0, 1 and 7 see none.
    for ([_]usize{ 0, 1, 7 }) |t| try std.testing.expectEqual(@as(f32, 0), out[t * hidden]);
    for (2..7) |t| try std.testing.expectApproxEqAbs(@as(f32, @floatFromInt(conv_kernel - (t - 2))), out[t * hidden + 9], 1e-5);
}
