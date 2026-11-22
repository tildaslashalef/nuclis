//! Higgs Audio V2 codec CPU decoder (`HiggsAudioV2TokenizerModel`): turns
//! `[frames][8]` RVQ codes at 25 fps into 24 kHz mono F32 samples (960 samples
//! per frame).
//!
//! Resolves `audio_tokenizer_weights/*` tensors through `audiocpp.tensor_names`
//! (handling `_audiocpp.N` aliases), decodes quantized/F16 tensors to F32 at
//! load, and runs 8-codebook RVQ decode → `fc2` (1024 → 256) → 5-block DAC
//! decoder (`upsampling_ratios = [8, 5, 4, 2, 3]`, `output_padding = stride % 2`,
//! Snake1d activations, no final tanh).
//! Facts: docs/models/omnivoice.md, docs/benchmarks/record/omnivoice.md.
const std = @import("std");
const gguf = @import("../formats/gguf.zig");
const weights = @import("../runtime/weights.zig");
const quant = @import("../quant/decode.zig");
const cpu = @import("../backends/cpu/root.zig");

const Allocator = std.mem.Allocator;
const dense = cpu.dense;

pub const codebooks = 8;
pub const codebook_size = 1024;
pub const codebook_dim = 64;
pub const latent_dim = 1024;
pub const acoustic_dim = 256;
pub const decoder_dim = 1024;
pub const sample_rate = 24_000;
pub const samples_per_frame = 960;
pub const frame_rate = 25;
pub const rates = [5]usize{ 8, 5, 4, 2, 3 };
pub const dilations = [3]usize{ 1, 3, 9 };
pub const snake_eps: f32 = 1e-9;
pub const max_decode_frames: usize = 3000; // 120 s ceiling

pub const Frame = [codebooks]u16;

pub const Error = Allocator.Error || gguf.Error || dense.Error || quant.Error || error{
    MissingTensor,
    InvalidTensor,
    InvalidLength,
    InvalidCode,
    UnsupportedConfiguration,
};

/// Stages exposed to an optional `Observer`, each `[rows][channels]` (time-major).
pub const Stage = enum {
    quantized,
    fc2,
    conv1,
    block_0,
    block_1,
    block_2,
    block_3,
    block_4,
    samples,
};

pub const Observer = struct {
    context: *anyopaque,
    record: *const fn (context: *anyopaque, stage: Stage, rows: []const f32) void,

    fn see(self: ?Observer, stage: Stage, rows: []const f32) void {
        if (self) |o| o.record(o.context, stage, rows);
    }
};

pub const Linear = struct {
    /// `[out][in]` row-major.
    w: []const f32,
    b: ?[]const f32,
    in: usize,
    out: usize,
};

pub const Conv1d = struct {
    /// `[out][k * in]` tap-major (`reordered[o, k, i] = raw[o, i, k]`).
    w: []const f32,
    b: ?[]const f32,
    in: usize,
    out: usize,
    k: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
};

pub const ConvTranspose1d = struct {
    /// `[k * out][in]` (`reordered[k, o, i] = raw[i, o, k]`).
    w: []const f32,
    b: ?[]const f32,
    in: usize,
    out: usize,
    k: usize,
    stride: usize,
    padding: usize,
    output_padding: usize,
};

pub const ResUnit = struct {
    snake1: []const f32,
    conv1: Conv1d,
    snake2: []const f32,
    conv2: Conv1d,
};

pub const DecoderBlock = struct {
    snake1: []const f32,
    conv_t1: ConvTranspose1d,
    units: [3]ResUnit,
};

/// Resolves `audiocpp.tensor_names` once so callers can look up aliased
/// `_audiocpp.N` tensors by their full native names (`audio_tokenizer_weights/*`).
pub const NativeLoader = struct {
    arena: Allocator,
    gpa: Allocator,
    view: weights.View,
    map: std.StringHashMapUnmanaged(*const gguf.Tensor),

    pub fn init(gpa: Allocator, arena: Allocator, doc: gguf.Document, directory: []const u8, view: weights.View) !NativeLoader {
        const names_val = doc.get("audiocpp.tensor_names") orelse return error.UnsupportedConfiguration;
        const names_arr = switch (names_val) {
            .array => |a| a,
            else => return error.UnsupportedConfiguration,
        };
        if (names_arr.element_type != .string or names_arr.count != doc.tensors.len)
            return error.UnsupportedConfiguration;

        var reader = try gguf.ArrayReader.init(names_arr, directory, .{});
        var map: std.StringHashMapUnmanaged(*const gguf.Tensor) = .empty;
        errdefer map.deinit(gpa);
        try map.ensureTotalCapacity(gpa, @intCast(doc.tensors.len));
        for (doc.tensors) |*t| {
            const native = (try reader.nextString()) orelse return error.UnsupportedConfiguration;
            map.putAssumeCapacity(native, t);
        }
        return .{ .arena = arena, .gpa = gpa, .view = view, .map = map };
    }

    pub fn deinit(self: *NativeLoader) void {
        self.map.deinit(self.gpa);
        self.* = undefined;
    }

    /// Decodes a tensor (`F32`, `F16`, or `Q8_0`) to F32 in `self.arena`,
    /// checking exact element count and GGUF shape (or total elements when
    /// trailing singleton dimensions are elided in GGUF).
    pub fn floats(self: *const NativeLoader, name: []const u8, expected_elements: usize) ![]f32 {
        const t = self.map.get(name) orelse return error.MissingTensor;
        if (t.elements != expected_elements) return error.InvalidTensor;
        const raw = self.view.bytes(t) catch return error.InvalidTensor;
        const out = try self.arena.alloc(f32, expected_elements);
        switch (t.encoding_id) {
            0 => {
                if (raw.len != expected_elements * 4) return error.InvalidTensor;
                for (out, 0..) |*dst, i| dst.* = @bitCast(std.mem.readInt(u32, raw[i * 4 ..][0..4], .little));
            },
            1, 8 => {
                quant.row(t.encoding_id, raw, out) catch return error.InvalidTensor;
            },
            else => return error.InvalidTensor,
        }
        return out;
    }

    pub fn linear(self: *const NativeLoader, weight_name: []const u8, bias_name: ?[]const u8, in_dim: usize, out_dim: usize) !Linear {
        const w = try self.floats(weight_name, out_dim * in_dim);
        const b = if (bias_name) |bn| try self.floats(bn, out_dim) else null;
        return .{ .w = w, .b = b, .in = in_dim, .out = out_dim };
    }

    /// Loads PyTorch `[out][in][k]` Conv1d weight into tap-major `[out][k][in]`.
    pub fn conv1d(
        self: *const NativeLoader,
        weight_name: []const u8,
        bias_name: ?[]const u8,
        in_ch: usize,
        out_ch: usize,
        k: usize,
        stride: usize,
        padding: usize,
        dilation: usize,
    ) !Conv1d {
        const raw = try self.floats(weight_name, out_ch * in_ch * k);
        const w = if (k == 1) raw else blk: {
            const reordered = try self.arena.alloc(f32, out_ch * k * in_ch);
            for (0..out_ch) |o| {
                for (0..in_ch) |c| {
                    for (0..k) |tap| {
                        reordered[(o * k + tap) * in_ch + c] = raw[(o * in_ch + c) * k + tap];
                    }
                }
            }
            break :blk reordered;
        };
        const b = if (bias_name) |bn| try self.floats(bn, out_ch) else null;
        return .{ .w = w, .b = b, .in = in_ch, .out = out_ch, .k = k, .stride = stride, .padding = padding, .dilation = dilation };
    }

    /// Loads PyTorch `[in][out][k]` ConvTranspose1d weight into `[k * out][in]`.
    pub fn convTranspose1d(
        self: *const NativeLoader,
        weight_name: []const u8,
        bias_name: ?[]const u8,
        in_ch: usize,
        out_ch: usize,
        k: usize,
        stride: usize,
        padding: usize,
        output_padding: usize,
    ) !ConvTranspose1d {
        const raw = try self.floats(weight_name, in_ch * out_ch * k);
        const reordered = try self.arena.alloc(f32, k * out_ch * in_ch);
        for (0..in_ch) |i| {
            for (0..out_ch) |o| {
                for (0..k) |tap| {
                    reordered[(tap * out_ch + o) * in_ch + i] = raw[(i * out_ch + o) * k + tap];
                }
            }
        }
        const b = if (bias_name) |bn| try self.floats(bn, out_ch) else null;
        return .{ .w = reordered, .b = b, .in = in_ch, .out = out_ch, .k = k, .stride = stride, .padding = padding, .output_padding = output_padding };
    }

    pub fn resUnit(self: *const NativeLoader, prefix: []const u8, channels: usize, dilation: usize) !ResUnit {
        var b1: [160]u8 = undefined;
        var b2: [160]u8 = undefined;
        const pad = ((7 - 1) * dilation) / 2;
        return .{
            .snake1 = try self.floats(fmt(&b1, "{s}.snake1.alpha", .{prefix}), channels),
            .conv1 = try self.conv1d(
                fmt(&b1, "{s}.conv1.weight", .{prefix}),
                fmt(&b2, "{s}.conv1.bias", .{prefix}),
                channels,
                channels,
                7,
                1,
                pad,
                dilation,
            ),
            .snake2 = try self.floats(fmt(&b1, "{s}.snake2.alpha", .{prefix}), channels),
            .conv2 = try self.conv1d(
                fmt(&b1, "{s}.conv2.weight", .{prefix}),
                fmt(&b2, "{s}.conv2.bias", .{prefix}),
                channels,
                channels,
                1,
                1,
                0,
                1,
            ),
        };
    }
};

fn fmt(buf: *[160]u8, comptime format: []const u8, args: anytype) []const u8 {
    return std.fmt.bufPrint(buf, format, args) catch unreachable;
}

/// F32 decoder weights owned by `storage`; immutable once initialized.
pub const Runtime = struct {
    io: std.Io,
    storage: std.heap.ArenaAllocator,
    /// `[codebook][codebook_size * codebook_dim]` (`[1024][64]`).
    books: [codebooks][]const f32,
    project_out: [codebooks]Linear,
    fc2: Linear,
    conv1: Conv1d,
    blocks: [5]DecoderBlock,
    snake_out: []const f32,
    conv2: Conv1d,

    /// Loads and decodes all Higgs Audio V2 decoder & quantizer tensors from
    /// `doc` / `directory` / `view`, which may be released afterwards.
    pub fn init(gpa: Allocator, io: std.Io, doc: gguf.Document, directory: []const u8, view: weights.View) !Runtime {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        errdefer arena.deinit();
        var loader = try NativeLoader.init(gpa, arena.allocator(), doc, directory, view);
        defer loader.deinit();

        var b1: [160]u8 = undefined;
        var b2: [160]u8 = undefined;

        var books: [codebooks][]const f32 = undefined;
        var project_out: [codebooks]Linear = undefined;
        for (0..codebooks) |c| {
            books[c] = try loader.floats(fmt(&b1, "audio_tokenizer_weights/quantizer.quantizers.{d}.codebook.embed", .{c}), codebook_size * codebook_dim);
            project_out[c] = try loader.linear(
                fmt(&b1, "audio_tokenizer_weights/quantizer.quantizers.{d}.project_out.weight", .{c}),
                fmt(&b2, "audio_tokenizer_weights/quantizer.quantizers.{d}.project_out.bias", .{c}),
                codebook_dim,
                latent_dim,
            );
        }

        const fc2 = try loader.linear("audio_tokenizer_weights/fc2.weight", "audio_tokenizer_weights/fc2.bias", latent_dim, acoustic_dim);
        const conv1 = try loader.conv1d("audio_tokenizer_weights/acoustic_decoder.conv1.weight", "audio_tokenizer_weights/acoustic_decoder.conv1.bias", acoustic_dim, decoder_dim, 7, 1, 3, 1);

        var blocks: [5]DecoderBlock = undefined;
        var ch: usize = decoder_dim;
        for (&blocks, rates, 0..) |*blk, stride, i| {
            const out_ch = ch / 2;
            const k = 2 * stride;
            const pad = (stride + 1) / 2;
            const op = stride % 2;
            blk.snake1 = try loader.floats(fmt(&b1, "audio_tokenizer_weights/acoustic_decoder.block.{d}.snake1.alpha", .{i}), ch);
            blk.conv_t1 = try loader.convTranspose1d(
                fmt(&b1, "audio_tokenizer_weights/acoustic_decoder.block.{d}.conv_t1.weight", .{i}),
                fmt(&b2, "audio_tokenizer_weights/acoustic_decoder.block.{d}.conv_t1.bias", .{i}),
                ch,
                out_ch,
                k,
                stride,
                pad,
                op,
            );
            for (&blk.units, dilations, 0..) |*u, d, j| {
                var pbuf: [160]u8 = undefined;
                const prefix = fmt(&pbuf, "audio_tokenizer_weights/acoustic_decoder.block.{d}.res_unit{d}", .{ i, j + 1 });
                u.* = try loader.resUnit(prefix, out_ch, d);
            }
            ch = out_ch;
        }

        const snake_out = try loader.floats("audio_tokenizer_weights/acoustic_decoder.snake1.alpha", ch);
        const conv2 = try loader.conv1d("audio_tokenizer_weights/acoustic_decoder.conv2.weight", "audio_tokenizer_weights/acoustic_decoder.conv2.bias", ch, 1, 7, 1, 3, 1);

        return .{
            .io = io,
            .storage = arena,
            .books = books,
            .project_out = project_out,
            .fc2 = fc2,
            .conv1 = conv1,
            .blocks = blocks,
            .snake_out = snake_out,
            .conv2 = conv2,
        };
    }

    pub fn deinit(self: *Runtime) void {
        self.storage.deinit();
        self.* = undefined;
    }

    /// Decodes `frames` (`[T][8]u16`) into 24 kHz mono F32 samples (`T * 960`).
    /// Caller owns the returned slice.
    pub fn decode(self: *const Runtime, gpa: Allocator, frames: []const Frame) Error![]f32 {
        return self.decodeObserved(gpa, frames, null);
    }

    /// Decodes `frames` while reporting intermediate stages to `observer`.
    pub fn decodeObserved(self: *const Runtime, gpa: Allocator, frames: []const Frame, observer: ?Observer) Error![]f32 {
        if (frames.len == 0 or frames.len > max_decode_frames) return error.InvalidLength;
        for (frames) |frame| {
            for (frame) |code| if (code >= codebook_size) return error.InvalidCode;
        }

        var scratch: std.heap.ArenaAllocator = .init(gpa);
        defer scratch.deinit();
        const ws = scratch.allocator();
        const t_len = frames.len;

        // 1. RVQ dequantization: sum over 8 codebooks of project_out(embed[code]).
        const quantized = try ws.alloc(f32, t_len * latent_dim);
        @memset(quantized, 0);
        const gathered = try ws.alloc(f32, t_len * codebook_dim);
        const projected = try ws.alloc(f32, t_len * latent_dim);
        for (0..codebooks) |c| {
            const book = self.books[c];
            for (frames, 0..) |frame, t| {
                const src = book[@as(usize, frame[c]) * codebook_dim ..][0..codebook_dim];
                @memcpy(gathered[t * codebook_dim ..][0..codebook_dim], src);
            }
            try runLinear(self.io, self.project_out[c], gathered, t_len, projected);
            for (quantized, projected) |*acc, v| acc.* += v;
        }
        Observer.see(observer, .quantized, quantized);

        // 2. fc2: [T, 1024] -> [T, 256]
        const aco_in = try ws.alloc(f32, t_len * acoustic_dim);
        try runLinear(self.io, self.fc2, quantized, t_len, aco_in);
        Observer.see(observer, .fc2, aco_in);

        // 3. acoustic_decoder.conv1: [T, 256] -> [T, 1024]
        var cur = try runConv1d(self.io, ws, self.conv1, aco_in, t_len);
        var cur_t = t_len;
        Observer.see(observer, .conv1, cur);

        // 4. 5 DAC decoder blocks.
        const block_stages = [5]Stage{ .block_0, .block_1, .block_2, .block_3, .block_4 };
        for (self.blocks, block_stages) |blk, stage| {
            runSnake(cur, cur_t, blk.snake1);
            const up = try runConvTranspose1d(self.io, ws, blk.conv_t1, cur, cur_t);
            cur = up.data;
            cur_t = up.t;
            for (blk.units) |unit| {
                try runResUnit(self.io, ws, unit, cur, cur_t);
            }
            Observer.see(observer, stage, cur);
        }

        // 5. Final Snake1d + conv2 (no tanh per HiggsAudioV2TokenizerModel._adjust_dac_decoder).
        runSnake(cur, cur_t, self.snake_out);
        const out_ws = try runConv1d(self.io, ws, self.conv2, cur, cur_t);
        Observer.see(observer, .samples, out_ws);

        return gpa.dupe(f32, out_ws);
    }
};

// ---- shared math helpers (reused by higgs_codec_encoder.zig) ---------------

pub fn runLinear(io: std.Io, l: Linear, x: []const f32, rows: usize, y: []f32) dense.Error!void {
    try dense.matmul(io, .{ .rows = rows, .inner = l.in, .outputs = l.out, .x = x, .w = l.w, .bias = l.b }, y);
}

/// Computes `Conv1d` output length `(t + 2*pad - dilation*(k-1) - 1) / stride + 1`.
pub fn conv1dOutLen(t: usize, k: usize, stride: usize, padding: usize, dilation: usize) usize {
    const eff_k = dilation * (k - 1) + 1;
    const padded = t + 2 * padding;
    if (padded < eff_k) return 0;
    return (padded - eff_k) / stride + 1;
}

/// Runs explicit-padded `Conv1d` over time-major `x: [t_in][c.in]`, returning
/// arena-allocated `[t_out][c.out]`.
pub fn runConv1d(io: std.Io, ws: Allocator, c: Conv1d, x: []const f32, t_in: usize) (Allocator.Error || dense.Error)![]f32 {
    const t_out = conv1dOutLen(t_in, c.k, c.stride, c.padding, c.dilation);
    const y = try ws.alloc(f32, t_out * c.out);
    if (t_out == 0) return y;
    if (c.k == 1 and c.stride == 1 and c.padding == 0) {
        try dense.matmul(io, .{ .rows = t_out, .inner = c.in, .outputs = c.out, .x = x, .w = c.w, .bias = c.b }, y);
        return y;
    }
    const inner = c.k * c.in;
    const cols = try ws.alloc(f32, t_out * inner);
    @memset(cols, 0);
    const pad_i: isize = @intCast(c.padding);
    const t_in_i: isize = @intCast(t_in);
    for (0..t_out) |to| {
        const base_in: isize = @as(isize, @intCast(to * c.stride)) - pad_i;
        const dst_row = cols[to * inner ..][0..inner];
        for (0..c.k) |tap| {
            const ti = base_in + @as(isize, @intCast(tap * c.dilation));
            if (ti >= 0 and ti < t_in_i) {
                const src_row = x[@as(usize, @intCast(ti)) * c.in ..][0..c.in];
                @memcpy(dst_row[tap * c.in ..][0..c.in], src_row);
            }
        }
    }
    try dense.matmul(io, .{ .rows = t_out, .inner = inner, .outputs = c.out, .x = cols, .w = c.w, .bias = c.b }, y);
    return y;
}

/// Runs explicit-padded `ConvTranspose1d` over time-major `x: [t_in][c.in]`,
/// returning arena-allocated `[t_out][c.out]`.
pub fn runConvTranspose1d(
    io: std.Io,
    ws: Allocator,
    c: ConvTranspose1d,
    x: []const f32,
    t_in: usize,
) (Allocator.Error || dense.Error)!struct { data: []f32, t: usize } {
    const t_out = (t_in - 1) * c.stride + c.k + c.output_padding - 2 * c.padding;
    const tmp = try ws.alloc(f32, t_in * c.k * c.out);
    try dense.matmul(io, .{ .rows = t_in, .inner = c.in, .outputs = c.k * c.out, .x = x, .w = c.w, .bias = null }, tmp);

    const y = try ws.alloc(f32, t_out * c.out);
    if (c.b) |bias| {
        for (0..t_out) |to| @memcpy(y[to * c.out ..][0..c.out], bias);
    } else {
        @memset(y, 0);
    }

    const pad_i: isize = @intCast(c.padding);
    const t_out_i: isize = @intCast(t_out);
    for (0..t_in) |ti| {
        const base: isize = @as(isize, @intCast(ti * c.stride)) - pad_i;
        const src_frame = tmp[ti * c.k * c.out ..][0 .. c.k * c.out];
        for (0..c.k) |tap| {
            const pos = base + @as(isize, @intCast(tap));
            if (pos >= 0 and pos < t_out_i) {
                const dst = y[@as(usize, @intCast(pos)) * c.out ..][0..c.out];
                const src = src_frame[tap * c.out ..][0..c.out];
                for (dst, src) |*d, s| d.* += s;
            }
        }
    }
    return .{ .data = y, .t = t_out };
}

/// In-place `Snake1d`: `x += sin²(α * x) / (α + 1e-9)` over `[t][channels]`.
pub fn runSnake(x: []f32, t_len: usize, alpha: []const f32) void {
    const ch = alpha.len;
    for (0..t_len) |t| {
        const row = x[t * ch ..][0..ch];
        for (row, alpha) |*v, a| {
            const s = @sin(a * v.*);
            v.* += (s * s) / (a + snake_eps);
        }
    }
}

/// In-place `DacResidualUnit`: `x += conv2(snake2(conv1(snake1(x))))`.
pub fn runResUnit(io: std.Io, ws: Allocator, u: ResUnit, x: []f32, t_len: usize) (Allocator.Error || dense.Error)!void {
    const h = try ws.dupe(f32, x);
    runSnake(h, t_len, u.snake1);
    const h1 = try runConv1d(io, ws, u.conv1, h, t_len);
    runSnake(h1, t_len, u.snake2);
    const h2 = try runConv1d(io, ws, u.conv2, h1, t_len);
    for (x, h2) |*dst, delta| dst.* += delta;
}

// ---- tests -----------------------------------------------------------------

test "ConvTranspose1d output length matches HiggsAudioV2 upsampling ratios" {
    var t: usize = 5;
    for (rates) |s| {
        const k = 2 * s;
        const p = (s + 1) / 2;
        const op = s % 2;
        t = (t - 1) * s + k + op - 2 * p;
    }
    try std.testing.expectEqual(@as(usize, 5 * samples_per_frame), t);
}

test "Conv1d, ConvTranspose1d, and Snake1d match direct arithmetic on synthetic shapes" {
    var arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena.deinit();
    const ws = arena.allocator();
    const io = std.testing.io;

    // Conv1d: 1 in, 1 out, k=3, pad=1, stride=1, dilation=1, w=[1, 2, 3], b=0.5
    const conv: Conv1d = .{ .w = &.{ 1, 2, 3 }, .b = &.{0.5}, .in = 1, .out = 1, .k = 3, .stride = 1, .padding = 1, .dilation = 1 };
    const x = [_]f32{ 1, 2, 3, 4 };
    const y = try runConv1d(io, ws, conv, &x, 4);
    try std.testing.expectEqual(@as(usize, 4), y.len);
    // t=0: 0*1 + 1*2 + 2*3 + 0.5 = 8.5; t=1: 1*1 + 2*2 + 3*3 + 0.5 = 14.5
    try std.testing.expectApproxEqAbs(@as(f32, 8.5), y[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 14.5), y[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 20.5), y[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 11.5), y[3], 1e-6);

    // ConvTranspose1d with odd stride=3, k=6, pad=2, op=1: T=2 -> T_out=6.
    const ct: ConvTranspose1d = .{ .w = &.{ 1, 2, 3, 4, 5, 6 }, .b = &.{0.0}, .in = 1, .out = 1, .k = 6, .stride = 3, .padding = 2, .output_padding = 1 };
    const up = try runConvTranspose1d(io, ws, ct, &.{ 1.0, 2.0 }, 2);
    try std.testing.expectEqual(@as(usize, 6), up.t);
    // ti=0 writes taps [0..5] to pos [-2..3] -> pos 0:3, 1:4, 2:5, 3:6
    // ti=1 writes 2*taps to pos [1..6] -> pos 1:+2, 2:+4, 3:+6, 4:+8, 5:+10
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), up.data[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 6.0), up.data[1], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 9.0), up.data[2], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 12.0), up.data[3], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 8.0), up.data[4], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 10.0), up.data[5], 1e-6);

    // Snake1d: x=0 -> 0; x=pi/2, alpha=1 -> pi/2 + 1 / (1 + 1e-9).
    var s = [_]f32{ 0.0, std.math.pi / 2.0 };
    runSnake(&s, 2, &.{1.0});
    try std.testing.expectApproxEqAbs(@as(f32, 0.0), s[0], 1e-7);
    try std.testing.expectApproxEqAbs(@as(f32, std.math.pi / 2.0 + 1.0), s[1], 1e-6);
}
