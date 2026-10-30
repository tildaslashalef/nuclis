//! Fish's modded DAC decoder (Drama 3's codec): frames of 10 codes become
//! 44.1 kHz mono samples, 2,048 per frame. `Runtime.init` binds the
//! `codec.q.*` and `codec.dec.*` tensors of the Drama 3 GGUF by name and
//! shape and decodes them to F32, the CPU reference; `Stream` decodes frames
//! as they arrive. Every stage is causal, so pieces of any size give one
//! `decode`'s samples. The convolutions reuse qwen3_codec.zig's matrices.
//! The pipeline: docs/models/drama3.md § The codec.
const std = @import("std");
const gguf = @import("../formats/gguf.zig");
const weights = @import("../runtime/weights.zig");
const quant = @import("../quant/decode.zig");
const cpu = @import("../backends/cpu/root.zig");
const q3 = @import("qwen3_codec.zig");

const Allocator = std.mem.Allocator;
const dense = cpu.dense;

pub const codebooks = 10;
pub const semantic_size = 4096;
pub const residual_size = 1024;
pub const codebook_dim = 8;
pub const latent = 1024;
pub const samples_per_frame = 2048;
pub const sample_rate = 44100;
pub const Frame = [codebooks]u16;

const layers = 8;
const heads = 16;
const head_dim = 64;
const window = 128;
const intermediate = 3072;
const rms_eps = 1e-5;
const rope_base = 1e4;
const decoder_dim = 1536;
const rates = [4]usize{ 8, 8, 4, 2 };
const dilations = [3]usize{ 1, 3, 9 };
const dwconv_kernel = 7;

pub const Error = error{ MissingTensor, InvalidTensor };

/// A stage the observer is shown, each `[rows][channels]`.
pub const Stage = enum { quantized, transformer, upsample_0, upsample_1, decoder_in, block_0, block_1, block_2, block_3, samples };

pub const Observer = struct {
    context: *anyopaque,
    record: *const fn (context: *anyopaque, stage: Stage, rows: []const f32) void,

    fn see(self: ?Observer, stage: Stage, rows: []const f32) void {
        if (self) |o| o.record(o.context, stage, rows);
    }
};

const Layer = struct {
    attn_norm: []const f32,
    ffn_norm: []const f32,
    q: q3.Linear,
    k: q3.Linear,
    v: q3.Linear,
    o: q3.Linear,
    gate: q3.Linear,
    up: q3.Linear,
    down: q3.Linear,
    attn_scale: []const f32,
    ffn_scale: []const f32,
};
const UpStage = struct { trans: q3.TransConv, next: q3.NeXt };
const Unit = struct { act1: q3.SnakeBeta, conv1: q3.CausalConv, act2: q3.SnakeBeta, conv2: q3.CausalConv };
const Block = struct { act: q3.SnakeBeta, trans: q3.TransConv, units: [3]Unit };

/// Frames `Runtime.decode` pushes at once, bounding its working memory.
const decode_frames = 16;

/// The decoder's weights in F32, owned by `storage`; read-only once built, so
/// streams may share it.
pub const Runtime = struct {
    io: std.Io,
    storage: std.heap.ArenaAllocator,
    /// [codebook][size][dim]: the semantic book, then the nine residual ones.
    books: [codebooks][]const f32,
    project: [codebooks]q3.Linear,
    layers: [layers]Layer,
    norm: []const f32,
    upsample: [2]UpStage,
    conv_in: q3.CausalConv,
    blocks: [4]Block,
    act_out: q3.SnakeBeta,
    conv_out: q3.CausalConv,

    /// Binds and decodes every decoder tensor of `doc` through `view`, which
    /// may be released afterwards.
    pub fn init(gpa: Allocator, io: std.Io, doc: gguf.Document, view: weights.View) !Runtime {
        var self: Runtime = undefined;
        self.io = io;
        self.storage = .init(gpa);
        errdefer self.storage.deinit();
        const l: Loader = .{ .arena = self.storage.allocator(), .gpa = gpa, .doc = doc, .view = view };
        for (0..codebooks) |c| {
            const size: usize = if (c == 0) semantic_size else residual_size;
            var name: Name = .{};
            const base = if (c == 0) name.of("codec.q.semantic", .{}) else name.of("codec.q.residual.{d}", .{c - 1});
            self.books[c] = try l.floats(try l.sub(base, "codebook"), &.{ codebook_dim, size });
            self.project[c] = try l.linear(try l.sub(base, "out_proj.weight"), try l.sub(base, "out_proj.bias"), codebook_dim, latent);
        }
        for (&self.layers, 0..) |*layer, i| {
            var name: Name = .{};
            const base = name.of("codec.q.post.blk.{d}", .{i});
            layer.* = .{
                .attn_norm = try l.floats(try l.sub(base, "attn_norm"), &.{latent}),
                .ffn_norm = try l.floats(try l.sub(base, "ffn_norm"), &.{latent}),
                .q = try l.linear(try l.sub(base, "attn_q"), null, latent, heads * head_dim),
                .k = try l.linear(try l.sub(base, "attn_k"), null, latent, heads * head_dim),
                .v = try l.linear(try l.sub(base, "attn_v"), null, latent, heads * head_dim),
                .o = try l.linear(try l.sub(base, "attn_o"), null, heads * head_dim, latent),
                .gate = try l.linear(try l.sub(base, "ffn_gate"), null, latent, intermediate),
                .up = try l.linear(try l.sub(base, "ffn_up"), null, latent, intermediate),
                .down = try l.linear(try l.sub(base, "ffn_down"), null, intermediate, latent),
                .attn_scale = try l.floats(try l.sub(base, "attn_scale"), &.{latent}),
                .ffn_scale = try l.floats(try l.sub(base, "ffn_scale"), &.{latent}),
            };
        }
        self.norm = try l.floats("codec.q.post.norm", &.{latent});
        for (&self.upsample, 0..) |*stage, i| {
            var name: Name = .{};
            const base = name.of("codec.q.up.{d}", .{i});
            stage.* = .{ .trans = try l.transposed(try l.sub(base, "conv"), latent, latent, 2), .next = try l.convNeXt(base) };
        }
        self.conv_in = try l.causal("codec.dec.in_conv", latent, decoder_dim, 7, 1);
        var channels: usize = decoder_dim;
        for (&self.blocks, rates, 0..) |*block, rate, i| {
            var name: Name = .{};
            const base = name.of("codec.dec.blk.{d}", .{i});
            block.act = try l.snake(try l.sub(base, "snake"), channels);
            block.trans = try l.transposed(try l.sub(base, "up"), channels, channels / 2, rate);
            channels /= 2;
            for (&block.units, dilations, 0..) |*u, d, j| {
                var unit_name: Name = .{};
                const ub = unit_name.of("{s}.res.{d}", .{ base, j });
                u.* = .{
                    .act1 = try l.snake(try l.sub(ub, "snake1"), channels),
                    .conv1 = try l.causal(try l.sub(ub, "conv1"), channels, channels, 7, d),
                    .act2 = try l.snake(try l.sub(ub, "snake2"), channels),
                    .conv2 = try l.causal(try l.sub(ub, "conv2"), channels, channels, 1, 1),
                };
            }
        }
        self.act_out = try l.snake("codec.dec.out_snake", channels);
        self.conv_out = try l.causal("codec.dec.out_conv", channels, 1, 7, 1);
        return self;
    }

    pub fn deinit(self: *Runtime) void {
        self.storage.deinit();
        self.* = undefined;
    }

    /// Decodes `codes` from a fresh state into `out` (`codes.len · 2048`
    /// samples in [−1, 1]).
    pub fn decode(self: *const Runtime, gpa: Allocator, codes: []const Frame, out: []f32) !void {
        return self.decodeObserved(gpa, codes, out, null);
    }

    pub fn decodeObserved(self: *const Runtime, gpa: Allocator, codes: []const Frame, out: []f32, observer: ?Observer) !void {
        if (out.len != codes.len * samples_per_frame) return error.InvalidShape;
        var stream = try Stream.init(gpa, self);
        defer stream.deinit();
        var at: usize = 0;
        while (at < codes.len) {
            const n: usize = @min(decode_frames, codes.len - at);
            try stream.pushObserved(codes[at..][0..n], out[at * samples_per_frame ..][0 .. n * samples_per_frame], observer);
            at += n;
        }
    }

    /// Fish's `DownsampleResidualVectorQuantize.decode` lookup: codes clamped
    /// to each book, each book's vector projected to the latent and summed.
    pub fn quantize(self: *const Runtime, ws: Allocator, codes: []const Frame, out: []f32) !void {
        const n = codes.len;
        const picked = try ws.alloc(f32, n * codebook_dim);
        const projected = try ws.alloc(f32, n * latent);
        @memset(out[0 .. n * latent], 0);
        for (0..codebooks) |c| {
            const size: usize = if (c == 0) semantic_size else residual_size;
            for (codes, 0..) |frame, t| {
                const code = @min(@as(usize, frame[c]), size - 1);
                @memcpy(picked[t * codebook_dim ..][0..codebook_dim], self.books[c][code * codebook_dim ..][0..codebook_dim]);
            }
            try q3.linear(self.io, self.project[c], picked, n, projected);
            for (out[0 .. n * latent], projected) |*a, b| a.* += b;
        }
    }
};

/// A tensor name built in place.
const Name = struct {
    buffer: [96]u8 = undefined,
    fn of(self: *Name, comptime format: []const u8, args: anytype) []const u8 {
        return std.fmt.bufPrint(&self.buffer, format, args) catch unreachable;
    }
};

const Loader = struct {
    arena: Allocator,
    gpa: Allocator,
    doc: gguf.Document,
    view: weights.View,

    /// `base.leaf` in the arena (names outlive the call that builds them).
    fn sub(self: Loader, base: []const u8, leaf: []const u8) ![]const u8 {
        return std.fmt.allocPrint(self.arena, "{s}.{s}", .{ base, leaf });
    }

    fn find(self: Loader, name: []const u8, dims: []const u64) !*const gguf.Tensor {
        for (self.doc.tensors) |*t| if (std.mem.eql(u8, t.name, name)) {
            if (!std.mem.eql(u64, t.dimensions, dims) or (t.encoding_id != 0 and t.encoding_id != 1)) return error.InvalidTensor;
            return t;
        };
        return error.MissingTensor;
    }

    /// The tensor whole, in F32, in its stored order (GGUF dimension 0 fastest).
    fn decode(self: Loader, alloc: Allocator, name: []const u8, dims: []const u64) ![]f32 {
        const t = try self.find(name, dims);
        const out = try alloc.alloc(f32, @intCast(t.elements));
        try quant.row(t.encoding_id, try self.view.bytes(t), out);
        return out;
    }

    fn floats(self: Loader, name: []const u8, dims: []const u64) ![]f32 {
        return self.decode(self.arena, name, dims);
    }

    /// Stored `[out][in]`.
    fn linear(self: Loader, weight: []const u8, bias: ?[]const u8, in: usize, out: usize) !q3.Linear {
        return .{
            .w = try self.floats(weight, &.{ in, out }),
            .b = if (bias) |b| try self.floats(b, &.{out}) else null,
            .in = in,
            .out = out,
        };
    }

    /// Stored `[kernel][out][in]`; the matrix wants `[out][kernel][in]`.
    fn causal(self: Loader, base: []const u8, in: usize, out: usize, kernel: usize, dilation: usize) !q3.CausalConv {
        const raw = try self.decode(self.gpa, try self.sub(base, "weight"), &.{ in, out, kernel });
        defer self.gpa.free(raw);
        const w = try self.arena.alloc(f32, raw.len);
        for (0..out) |o| for (0..kernel) |j| {
            @memcpy(w[(o * kernel + j) * in ..][0..in], raw[(j * out + o) * in ..][0..in]);
        };
        return .{ .w = w, .b = try self.floats(try self.sub(base, "bias"), &.{out}), .in = in, .out = out, .kernel = kernel, .dilation = dilation };
    }

    /// A causal transposed convolution of kernel `2 · stride` (or `stride`),
    /// stored `[kernel][out][in]`, as qwen3_codec's `TransConv`
    /// (`[stride][out][taps][in]`): output row m·stride + p reads tap
    /// p + q·stride of input row m − q.
    fn transposed(self: Loader, base: []const u8, in: usize, out: usize, stride: usize) !q3.TransConv {
        const name = try self.sub(base, "weight");
        const t = for (self.doc.tensors) |*t| {
            if (std.mem.eql(u8, t.name, name)) break t;
        } else return error.MissingTensor;
        if (t.dimensions.len != 3) return error.InvalidTensor;
        const kernel: usize = @intCast(t.dimensions[2]);
        if (kernel != stride and kernel != 2 * stride) return error.InvalidTensor;
        const taps = kernel / stride;
        const raw = try self.decode(self.gpa, name, &.{ in, out, kernel });
        defer self.gpa.free(raw);
        const w = try self.arena.alloc(f32, raw.len);
        for (0..stride) |p| for (0..out) |o| for (0..taps) |q| {
            @memcpy(w[((p * out + o) * taps + q) * in ..][0..in], raw[((p + q * stride) * out + o) * in ..][0..in]);
        };
        const bias = try self.decode(self.gpa, try self.sub(base, "bias"), &.{out});
        defer self.gpa.free(bias);
        const b = try self.arena.alloc(f32, stride * out);
        for (0..stride) |p| @memcpy(b[p * out ..][0..out], bias);
        return .{ .w = w, .b = b, .in = in, .out = out, .stride = stride, .taps = taps };
    }

    /// Snake `x + inv · sin²(α · x)`; the file stores α and 1/(α + 1e-9).
    fn snake(self: Loader, base: []const u8, channels: usize) !q3.SnakeBeta {
        return .{
            .scale = try self.floats(try self.sub(base, "alpha"), &.{channels}),
            .inv = try self.floats(try self.sub(base, "inv_alpha"), &.{channels}),
        };
    }

    /// The depthwise kernel is stored `[kernel][channel]`; `q3.depthwise`
    /// reads `[channel][kernel]`.
    fn convNeXt(self: Loader, base: []const u8) !q3.NeXt {
        const raw = try self.decode(self.gpa, try self.sub(base, "convnext.dwconv.weight"), &.{ latent, dwconv_kernel });
        defer self.gpa.free(raw);
        const dw = try self.arena.alloc(f32, raw.len);
        for (0..latent) |c| for (0..dwconv_kernel) |j| {
            dw[c * dwconv_kernel + j] = raw[j * latent + c];
        };
        return .{
            .dw = dw,
            .dw_b = try self.floats(try self.sub(base, "convnext.dwconv.bias"), &.{latent}),
            .ln_w = try self.floats(try self.sub(base, "convnext.norm.weight"), &.{latent}),
            .ln_b = try self.floats(try self.sub(base, "convnext.norm.bias"), &.{latent}),
            .gamma = try self.floats(try self.sub(base, "convnext.gamma"), &.{latent}),
            .pw1 = try self.linear(try self.sub(base, "convnext.pw1.weight"), try self.sub(base, "convnext.pw1.bias"), latent, 4 * latent),
            .pw2 = try self.linear(try self.sub(base, "convnext.pw2.weight"), try self.sub(base, "convnext.pw2.bias"), 4 * latent, latent),
        };
    }
};

/// Fish's codec RoPE: GPT-J (adjacent-pair) rotation whose cos and sin are
/// computed in F32 and then rounded to BF16 (`precompute_freqs_cis`'s
/// default dtype), applied in F32.
pub fn rope(x: []f32, position: usize, base: f32) void {
    const half = x.len / 2;
    for (0..half) |i| {
        const exponent: f32 = @as(f32, @floatFromInt(2 * i)) / @as(f32, @floatFromInt(x.len));
        const freq: f32 = 1.0 / std.math.pow(f32, base, exponent);
        const angle: f32 = @as(f32, @floatFromInt(position)) * freq;
        const c = bf16(@cos(angle));
        const s = bf16(@sin(angle));
        const a = x[2 * i];
        const b = x[2 * i + 1];
        x[2 * i] = a * c - b * s;
        x[2 * i + 1] = b * c + a * s;
    }
}

/// Round to the nearest BF16, ties to even.
pub fn bf16(v: f32) f32 {
    const bits: u32 = @bitCast(v);
    const rounded = bits +% (0x7FFF + ((bits >> 16) & 1));
    return @bitCast(rounded & 0xFFFF0000);
}

const Cache = struct { keys: []f32, values: []f32, count: usize };
const UpState = struct { trans: q3.History, dw: q3.History };
const BlockState = struct { trans: q3.History, units: [3]q3.History };

/// A decode in progress: push frames as they come and get their samples.
/// Borrows the runtime, which must outlive it.
pub const Stream = struct {
    runtime: *const Runtime,
    gpa: Allocator,
    memory: std.heap.ArenaAllocator,
    /// Frames pushed so far: the next frame's RoPE position.
    position: usize,
    caches: [layers]Cache,
    upsample: [2]UpState,
    conv_in: q3.History,
    blocks: [4]BlockState,
    conv_out: q3.History,

    pub fn init(gpa: Allocator, runtime: *const Runtime) !Stream {
        var self: Stream = undefined;
        self.runtime = runtime;
        self.gpa = gpa;
        self.memory = .init(gpa);
        errdefer self.memory.deinit();
        const a = self.memory.allocator();
        const width = heads * head_dim;
        for (&self.caches) |*cache| cache.* = .{
            .keys = try a.alloc(f32, (window - 1) * width),
            .values = try a.alloc(f32, (window - 1) * width),
            .count = 0,
        };
        for (&self.upsample, runtime.upsample) |*state, stage| state.* = .{
            .trans = try q3.transHistory(a, stage.trans),
            .dw = try q3.history(a, dwconv_kernel - 1, latent),
        };
        self.conv_in = try q3.causalHistory(a, runtime.conv_in);
        for (&self.blocks, runtime.blocks) |*state, block| {
            state.trans = try q3.transHistory(a, block.trans);
            for (&state.units, block.units) |*h, unit| h.* = try q3.causalHistory(a, unit.conv1);
        }
        self.conv_out = try q3.causalHistory(a, runtime.conv_out);
        self.reset();
        return self;
    }

    pub fn deinit(self: *Stream) void {
        self.memory.deinit();
        self.* = undefined;
    }

    /// Back to the start of a clip: zero context, position 0.
    pub fn reset(self: *Stream) void {
        self.position = 0;
        for (&self.caches) |*cache| cache.count = 0;
        for (&self.upsample) |*u| {
            @memset(u.trans.data, 0);
            @memset(u.dw.data, 0);
        }
        @memset(self.conv_in.data, 0);
        @memset(self.conv_out.data, 0);
        for (&self.blocks) |*b| {
            @memset(b.trans.data, 0);
            for (&b.units) |*h| @memset(h.data, 0);
        }
    }

    pub fn push(self: *Stream, codes: []const Frame, out: []f32) !void {
        return self.pushObserved(codes, out, null);
    }

    /// Decodes the next `codes` into `out` (`codes.len · 2048` samples).
    pub fn pushObserved(self: *Stream, codes: []const Frame, out: []f32, observer: ?Observer) !void {
        const rt = self.runtime;
        const io = rt.io;
        const n = codes.len;
        if (out.len != n * samples_per_frame) return error.InvalidShape;
        if (n == 0) return;
        var arena: std.heap.ArenaAllocator = .init(self.gpa);
        defer arena.deinit();
        const ws = arena.allocator();

        var x = try ws.alloc(f32, n * latent);
        try rt.quantize(ws, codes, x);
        Observer.see(observer, .quantized, x);
        try self.transformer(ws, x, n);
        Observer.see(observer, .transformer, x);

        var rows = n;
        for (rt.upsample, &self.upsample, [2]Stage{ .upsample_0, .upsample_1 }) |stage, *state, label| {
            const up = try ws.alloc(f32, rows * stage.trans.stride * latent);
            try q3.transConv(ws, io, stage.trans, &state.trans, x, rows, up);
            rows *= stage.trans.stride;
            try q3.convNeXt(ws, io, stage.next, &state.dw, up, rows);
            x = up;
            Observer.see(observer, label, x);
        }
        const d0 = try ws.alloc(f32, rows * decoder_dim);
        try q3.causalConv(ws, io, rt.conv_in, &self.conv_in, x, rows, d0);
        x = d0;
        Observer.see(observer, .decoder_in, x);
        for (rt.blocks, &self.blocks, [4]Stage{ .block_0, .block_1, .block_2, .block_3 }) |block, *state, label| {
            const activated = try ws.alloc(f32, x.len);
            q3.snakeBeta(block.act, x, activated);
            const y = try ws.alloc(f32, rows * block.trans.stride * block.trans.out);
            try q3.transConv(ws, io, block.trans, &state.trans, activated, rows, y);
            rows *= block.trans.stride;
            const a = try ws.alloc(f32, y.len);
            const b = try ws.alloc(f32, y.len);
            for (block.units, &state.units) |unit, *h| {
                q3.snakeBeta(unit.act1, y, a);
                try q3.causalConv(ws, io, unit.conv1, h, a, rows, b);
                q3.snakeBeta(unit.act2, b, b);
                try q3.causalConv(ws, io, unit.conv2, null, b, rows, a);
                for (y, a) |*r, v| r.* += v;
            }
            x = y;
            Observer.see(observer, label, x);
        }
        const activated = try ws.alloc(f32, x.len);
        q3.snakeBeta(rt.act_out, x, activated);
        try q3.causalConv(ws, io, rt.conv_out, &self.conv_out, activated, rows, out);
        for (out) |*v| v.* = std.math.tanh(v.*);
        Observer.see(observer, .samples, out);
        self.position += n;
    }

    /// The post-quantizer transformer over `n` new rows of `x` (`[n][latent]`),
    /// in place: pre-norm blocks with layer scales, causal attention over the
    /// last `window` rows.
    fn transformer(self: *Stream, ws: Allocator, x: []f32, n: usize) !void {
        const io = self.runtime.io;
        const width = heads * head_dim;
        const h = try ws.alloc(f32, n * latent);
        const q = try ws.alloc(f32, n * width);
        const k = try ws.alloc(f32, n * width);
        const v = try ws.alloc(f32, n * width);
        const attended = try ws.alloc(f32, n * width);
        const o = try ws.alloc(f32, n * latent);
        const gate = try ws.alloc(f32, n * intermediate);
        const up = try ws.alloc(f32, n * intermediate);
        const keys = try ws.alloc(f32, (window - 1 + n) * width);
        const values = try ws.alloc(f32, (window - 1 + n) * width);
        const scores = try ws.alloc(f64, window);
        for (self.runtime.layers, &self.caches) |layer, *cache| {
            try q3.rmsRows(x, layer.attn_norm, rms_eps, h);
            try q3.linear(io, layer.q, h, n, q);
            try q3.linear(io, layer.k, h, n, k);
            try q3.linear(io, layer.v, h, n, v);
            for (0..n) |t| for (0..heads) |head| {
                const at = t * width + head * head_dim;
                rope(q[at..][0..head_dim], self.position + t, rope_base);
                rope(k[at..][0..head_dim], self.position + t, rope_base);
            };
            const past = cache.count;
            const total = past + n;
            @memcpy(keys[0 .. past * width], cache.keys[0 .. past * width]);
            @memcpy(keys[past * width ..][0 .. n * width], k);
            @memcpy(values[0 .. past * width], cache.values[0 .. past * width]);
            @memcpy(values[past * width ..][0 .. n * width], v);
            attention(q, keys, values, n, past, scores, attended);
            try q3.linear(io, layer.o, attended, n, o);
            q3.addScaled(x, layer.attn_scale, o);
            try q3.rmsRows(x, layer.ffn_norm, rms_eps, h);
            try q3.linear(io, layer.gate, h, n, gate);
            try q3.linear(io, layer.up, h, n, up);
            for (gate, up) |*g, u| g.* = cpu.silu(g.*) * u;
            try q3.linear(io, layer.down, gate, n, o);
            q3.addScaled(x, layer.ffn_scale, o);
            const kept: usize = @min(window - 1, total);
            @memcpy(cache.keys[0 .. kept * width], keys[(total - kept) * width ..][0 .. kept * width]);
            @memcpy(cache.values[0 .. kept * width], values[(total - kept) * width ..][0 .. kept * width]);
            cache.count = kept;
        }
        try q3.rmsRows(x, self.runtime.norm, rms_eps, h);
        @memcpy(x, h);
    }
};

/// Sliding-window attention of `n` new queries: query t (row `past + t` of
/// `keys`) sees rows `past + t − window + 1 … past + t`. Scores and sums in F64.
fn attention(q: []const f32, keys: []const f32, values: []const f32, n: usize, past: usize, scores: []f64, out: []f32) void {
    const width = heads * head_dim;
    const scale = 1 / @sqrt(@as(f64, head_dim));
    for (0..n) |t| for (0..heads) |head| {
        const i = past + t;
        const lowest = (i + 1) -| window;
        const qi = q[t * width + head * head_dim ..][0..head_dim];
        var max: f64 = -std.math.inf(f64);
        for (lowest..i + 1, 0..) |j, s| {
            var dot: f64 = 0;
            for (qi, keys[j * width + head * head_dim ..][0..head_dim]) |a, b| dot += @as(f64, a) * b;
            scores[s] = dot * scale;
            max = @max(max, scores[s]);
        }
        const count = i + 1 - lowest;
        var total: f64 = 0;
        for (scores[0..count]) |*s| {
            s.* = @exp(s.* - max);
            total += s.*;
        }
        const o = out[t * width + head * head_dim ..][0..head_dim];
        for (o, 0..) |*value, d| {
            var sum: f64 = 0;
            for (lowest..i + 1, 0..) |j, s| sum += scores[s] * values[j * width + head * head_dim + d];
            value.* = @floatCast(sum / total);
        }
    };
}

test "bf16 rounding is to nearest, ties to even" {
    try std.testing.expectEqual(@as(f32, 1.0), bf16(1.0));
    try std.testing.expectEqual(@as(f32, 1.0078125), bf16(1.0078125));
    // 1 + 2^-8 is halfway between 1 and 1 + 2^-7: ties to even (1).
    try std.testing.expectEqual(@as(f32, 1.0), bf16(1.00390625));
    try std.testing.expectEqual(@as(f32, 1.0078125), bf16(1.005));
}

test "rope rotates adjacent pairs and leaves position 0 unchanged" {
    var x = [_]f32{ 1, 2, 3, 4 };
    rope(&x, 0, 1e4);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3, 4 }, &x);
    var y = [_]f32{ 1, 0, 1, 0 };
    rope(&y, 1, 1e4);
    try std.testing.expectApproxEqAbs(bf16(@cos(@as(f32, 1))), y[0], 1e-7);
    try std.testing.expectApproxEqAbs(bf16(@sin(@as(f32, 1))), y[1], 1e-7);
}
