//! Fish's modded DAC encoder (Drama 3's voice cloning): 44.1 kHz mono samples
//! become frames of 10 codes, one per 2,048 samples. Binds `codec.enc.*` and
//! the quantizer's `down`, `pre`, `in_proj`, codebook, and `out_proj`
//! tensors of the Drama 3 GGUF; whole clips on the CPU. The pipeline:
//! docs/models/drama3.md § The codec.
const std = @import("std");
const gguf = @import("../formats/gguf.zig");
const weights = @import("../runtime/weights.zig");
const cpu = @import("../backends/cpu/root.zig");
const q3 = @import("qwen3_codec.zig");
const fc = @import("fish_codec.zig");

const Allocator = std.mem.Allocator;
const dense = cpu.dense;

const encoder_dim = 64;
const strides = [4]usize{ 2, 4, 8, 8 };
const dilations = [3]usize{ 1, 3, 9 };
const encoder_window = 512;
const encoder_layers = 4;
const pre_layers = 8;

/// Most samples `encode` takes: 60 s at 44.1 kHz.
pub const max_samples = 60 * fc.sample_rate;

/// A causal convolution of kernel `2 · stride` (or `stride`) and stride
/// `stride`: output t reads input rows (t − 1)·stride … (t + 1)·stride − 1
/// (or t·stride … when the kernel is the stride), zeros before the start.
/// `w` is [out][taps · stride · in], groups oldest first, each tap-major.
const Strided = struct { w: []const f32, b: []const f32, in: usize, out: usize, stride: usize, taps: usize };
const Unit = struct { act1: q3.SnakeBeta, conv1: q3.CausalConv, act2: q3.SnakeBeta, conv2: q3.CausalConv };
const Block = struct { units: [3]Unit, act: q3.SnakeBeta, down: Strided };
const Down = struct { conv: Strided, next: q3.NeXt };

pub const Encoder = struct {
    io: std.Io,
    storage: std.heap.ArenaAllocator,
    in_conv: q3.CausalConv,
    blocks: [4]Block,
    transformer: fc.Transformer,
    act_out: q3.SnakeBeta,
    out_conv: q3.CausalConv,
    down: [2]Down,
    pre: fc.Transformer,
    in_proj: [fc.codebooks]q3.Linear,
    /// Each book's rows L2-normalized, for the nearest-neighbour search.
    normalized: [fc.codebooks][]const f32,
    books: [fc.codebooks][]const f32,
    out_proj: [fc.codebooks]q3.Linear,

    pub fn init(gpa: Allocator, io: std.Io, doc: gguf.Document, view: weights.View) !Encoder {
        var self: Encoder = undefined;
        self.io = io;
        self.storage = .init(gpa);
        errdefer self.storage.deinit();
        const l: fc.Loader = .{ .arena = self.storage.allocator(), .gpa = gpa, .doc = doc, .view = view };
        self.in_conv = try l.causal("codec.enc.in_conv", 1, encoder_dim, 7, 1);
        var channels: usize = encoder_dim;
        for (&self.blocks, strides, 0..) |*block, stride, i| {
            var name: fc.Name = .{};
            const base = name.of("codec.enc.blk.{d}", .{i});
            for (&block.units, dilations, 0..) |*u, d, j| {
                const ub = try std.fmt.allocPrint(l.arena, "{s}.res.{d}", .{ base, j });
                u.* = .{
                    .act1 = try l.snake(try l.sub(ub, "snake1"), channels),
                    .conv1 = try l.causal(try l.sub(ub, "conv1"), channels, channels, 7, d),
                    .act2 = try l.snake(try l.sub(ub, "snake2"), channels),
                    .conv2 = try l.causal(try l.sub(ub, "conv2"), channels, channels, 1, 1),
                };
            }
            block.act = try l.snake(try l.sub(base, "snake"), channels);
            block.down = try strided(l, try l.sub(base, "down"), channels, 2 * channels, stride, 2);
            channels *= 2;
        }
        self.transformer = try l.transformer("codec.enc.blk.3.tf", encoder_layers, encoder_window);
        self.act_out = try l.snake("codec.enc.out_snake", channels);
        self.out_conv = try l.causal("codec.enc.out_conv", channels, fc.latent, 3, 1);
        for (&self.down, 0..) |*d, i| {
            var name: fc.Name = .{};
            const base = name.of("codec.q.down.{d}", .{i});
            d.* = .{ .conv = try strided(l, try l.sub(base, "conv"), fc.latent, fc.latent, 2, 1), .next = try l.convNeXt(base) };
        }
        self.pre = try l.transformer("codec.q.pre", pre_layers, fc.window);
        for (0..fc.codebooks) |c| {
            const size: usize = if (c == 0) fc.semantic_size else fc.residual_size;
            const base = if (c == 0) "codec.q.semantic" else try std.fmt.allocPrint(l.arena, "codec.q.residual.{d}", .{c - 1});
            self.in_proj[c] = try l.linear(try l.sub(base, "in_proj.weight"), try l.sub(base, "in_proj.bias"), fc.latent, fc.codebook_dim);
            self.out_proj[c] = try l.linear(try l.sub(base, "out_proj.weight"), try l.sub(base, "out_proj.bias"), fc.codebook_dim, fc.latent);
            const book = try l.floats(try l.sub(base, "codebook"), &.{ fc.codebook_dim, size });
            self.books[c] = book;
            const unit = try l.arena.dupe(f32, book);
            for (0..size) |r| normalize(unit[r * fc.codebook_dim ..][0..fc.codebook_dim]);
            self.normalized[c] = unit;
        }
        return self;
    }

    pub fn deinit(self: *Encoder) void {
        self.storage.deinit();
        self.* = undefined;
    }

    /// The codes of `samples` (44.1 kHz mono), zero-padded to a whole
    /// frame (Fish's `DAC.encode`): `ceil(len / 2048)` frames, owned.
    pub fn encode(self: *const Encoder, gpa: Allocator, samples: []const f32) ![]fc.Frame {
        return self.encodeObserved(gpa, samples, null);
    }

    pub fn encodeObserved(self: *const Encoder, gpa: Allocator, samples: []const f32, observer: ?Observer) ![]fc.Frame {
        if (samples.len == 0 or samples.len > max_samples) return error.InvalidLength;
        const io = self.io;
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        const ws = arena.allocator();
        const frames = std.math.divCeil(usize, samples.len, fc.samples_per_frame) catch unreachable;
        var rows = frames * fc.samples_per_frame;
        const input = try ws.alloc(f32, rows);
        @memset(input, 0);
        @memcpy(input[0..samples.len], samples);

        var x = try ws.alloc(f32, rows * encoder_dim);
        try conv(ws, io, self.in_conv, input, rows, x);
        Observer.see(observer, .encoder_in, x);
        for (self.blocks, [4]Stage{ .block_0, .block_1, .block_2, .block_3 }) |block, label| {
            const a = try ws.alloc(f32, x.len);
            const b = try ws.alloc(f32, x.len);
            for (block.units) |unit| {
                q3.snakeBeta(unit.act1, x, a);
                try conv(ws, io, unit.conv1, a, rows, b);
                q3.snakeBeta(unit.act2, b, b);
                try conv(ws, io, unit.conv2, b, rows, a);
                for (x, a) |*r, v| r.* += v;
            }
            q3.snakeBeta(block.act, x, a);
            rows /= block.down.stride;
            const y = try ws.alloc(f32, rows * block.down.out);
            try stridedConv(ws, io, block.down, a, rows, y);
            x = y;
            Observer.see(observer, label, x);
        }
        try runWhole(ws, io, self.transformer, x, rows);
        Observer.see(observer, .transformer, x);
        const activated = try ws.alloc(f32, x.len);
        q3.snakeBeta(self.act_out, x, activated);
        const z = try ws.alloc(f32, rows * fc.latent);
        try conv(ws, io, self.out_conv, activated, rows, z);
        x = z;
        Observer.see(observer, .encoder_out, x);
        for (self.down, [2]Stage{ .down_0, .down_1 }) |d, label| {
            rows /= d.conv.stride;
            const y = try ws.alloc(f32, rows * fc.latent);
            try stridedConv(ws, io, d.conv, x, rows, y);
            const empty = try q3.history(ws, 6, fc.latent);
            @memset(empty.data, 0);
            var h = empty;
            try q3.convNeXt(ws, io, d.next, &h, y, rows);
            x = y;
            Observer.see(observer, label, x);
        }
        std.debug.assert(rows == frames);
        try runWhole(ws, io, self.pre, x, rows);
        Observer.see(observer, .pre, x);

        const codes = try gpa.alloc(fc.Frame, frames);
        errdefer gpa.free(codes);
        try self.quantize(ws, x, codes);
        return codes;
    }

    /// The semantic book on `z`, then the nine residual books in turn on what
    /// it leaves: each the nearest unit codebook row to the unit projection.
    fn quantize(self: *const Encoder, ws: Allocator, z: []f32, codes: []fc.Frame) !void {
        const n = codes.len;
        const e = try ws.alloc(f32, n * fc.codebook_dim);
        const picked = try ws.alloc(f32, n * fc.codebook_dim);
        const projected = try ws.alloc(f32, n * fc.latent);
        for (0..fc.codebooks) |c| {
            try q3.linear(self.io, self.in_proj[c], z, n, e);
            const size = self.books[c].len / fc.codebook_dim;
            for (0..n) |t| {
                const row = e[t * fc.codebook_dim ..][0..fc.codebook_dim];
                normalize(row);
                var best: usize = 0;
                var best_score = -std.math.inf(f64);
                for (0..size) |k| {
                    var dot: f64 = 0;
                    for (row, self.normalized[c][k * fc.codebook_dim ..][0..fc.codebook_dim]) |a, b| dot += @as(f64, a) * b;
                    if (dot > best_score) {
                        best_score = dot;
                        best = k;
                    }
                }
                codes[t][c] = @intCast(best);
                @memcpy(picked[t * fc.codebook_dim ..][0..fc.codebook_dim], self.books[c][best * fc.codebook_dim ..][0..fc.codebook_dim]);
            }
            try q3.linear(self.io, self.out_proj[c], picked, n, projected);
            for (z[0 .. n * fc.latent], projected) |*a, b| a.* -= b;
        }
    }
};

/// A stage the observer is shown, each `[rows][channels]`.
pub const Stage = enum { encoder_in, block_0, block_1, block_2, block_3, transformer, encoder_out, down_0, down_1, pre };

pub const Observer = struct {
    context: *anyopaque,
    record: *const fn (context: *anyopaque, stage: Stage, rows: []const f32) void,

    fn see(self: ?Observer, stage: Stage, rows: []const f32) void {
        if (self) |o| o.record(o.context, stage, rows);
    }
};

/// `F.normalize`: x / max(‖x‖, 1e-12).
fn normalize(x: []f32) void {
    var sum: f32 = 0;
    for (x) |v| sum += v * v;
    const norm = @max(@sqrt(sum), 1e-12);
    for (x) |*v| v.* /= norm;
}

/// A whole clip through a windowed transformer from position 0.
fn runWhole(ws: Allocator, io: std.Io, t: fc.Transformer, x: []f32, rows: usize) !void {
    const caches = try t.caches(ws);
    // Chunks of the window keep the attention buffers bounded.
    var at: usize = 0;
    while (at < rows) {
        const n = @min(t.window, rows - at);
        try t.run(io, ws, caches, at, x[at * fc.latent ..][0 .. n * fc.latent], n);
        at += n;
    }
}

/// A causal convolution over a whole clip from zero context.
fn conv(ws: Allocator, io: std.Io, c: q3.CausalConv, x: []const f32, rows: usize, y: []f32) !void {
    if (c.kernel == 1) return q3.causalConv(ws, io, c, null, x, rows, y);
    var h = try q3.causalHistory(ws, c);
    @memset(h.data, 0);
    try q3.causalConv(ws, io, c, &h, x, rows, y);
}

/// `base.first` and `base.second` (`[out][stride · in]` each, the kernel's
/// older and newer halves), or `base.weight` alone when the kernel is the
/// stride, and `base.bias`.
fn strided(l: fc.Loader, base: []const u8, in: usize, out: usize, stride: usize, taps: usize) !Strided {
    const inner = stride * in;
    const w = try l.arena.alloc(f32, out * taps * inner);
    const halves: []const []const u8 = if (taps == 2) &.{ "first", "second" } else &.{"weight"};
    for (halves, 0..) |leaf, half| {
        const raw = try l.decode(l.gpa, try l.sub(base, leaf), &.{ inner, out });
        defer l.gpa.free(raw);
        for (0..out) |o| @memcpy(w[(o * taps + half) * inner ..][0..inner], raw[o * inner ..][0..inner]);
    }
    return .{ .w = w, .b = try l.floats(try l.sub(base, "bias"), &.{out}), .in = in, .out = out, .stride = stride, .taps = taps };
}

/// `rows` outputs of `c` over `x` (`[rows · stride][in]`) into `y`.
fn stridedConv(ws: Allocator, io: std.Io, c: Strided, x: []const f32, rows: usize, y: []f32) !void {
    const group = c.stride * c.in;
    const inner = c.taps * group;
    const gathered = try ws.alloc(f32, rows * inner);
    for (0..rows) |t| for (0..c.taps) |j| {
        const dst = gathered[t * inner + j * group ..][0..group];
        // Group t − (taps − 1) + j; before the start it is zero.
        const g = @as(isize, @intCast(t + j)) - @as(isize, @intCast(c.taps - 1));
        if (g < 0) @memset(dst, 0) else @memcpy(dst, x[@as(usize, @intCast(g)) * group ..][0..group]);
    };
    try dense.matmul(io, .{ .rows = rows, .inner = inner, .outputs = c.out, .x = gathered, .w = c.w, .bias = c.b }, y);
}
