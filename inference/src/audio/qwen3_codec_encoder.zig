//! Qwen3-TTS's codec encoder on the CPU: 24 kHz mono samples to frames of
//! 16 codes at 12.5 Hz, the Base's reference codes. A Mimi encoder: the
//! SEANet convolutions (causal, zero-padded), an 8-layer causal transformer,
//! a stride-2 downsample, and the split quantizer's encode, whose codebooks
//! and projections are the decoder's (`qwen3_codec.Binding`; the file's
//! encoder copies are checked equal and not read). It runs once per saved
//! voice, so the F32 CPU path is the only one. Rows are time-major.
//! docs/models/qwen3-tts.md § The codec encoder.
const std = @import("std");
const safetensors = @import("../formats/safetensors.zig");
const cpu = @import("../backends/cpu/root.zig");
const codec = @import("qwen3_codec.zig");

const Allocator = std.mem.Allocator;
const Ref = safetensors.Ref;
const dense = cpu.dense;
const groups = codec.groups;
pub const Frame = codec.Frame;

/// `speech_tokenizer/config.json`'s `encoder_config`; the defaults are the
/// published file's.
pub const Config = struct {
    filters: usize = 64,
    /// Each stage's stride, in the order applied (the file's
    /// `upsampling_ratios` reversed).
    ratios: [4]usize = .{ 4, 5, 6, 8 },
    kernel: usize = 7,
    last_kernel: usize = 3,
    residual_kernel: usize = 3,
    /// A residual unit's hidden width is its channels over this.
    compress: usize = 2,
    hidden: usize = 512,
    layers: usize = 8,
    heads: usize = 8,
    head_dim: usize = 64,
    intermediate: usize = 2048,
    rope_theta: f32 = 10000,
    norm_eps: f32 = 1e-5,
    /// The 25 → 12.5 Hz downsample: kernel 4, stride 2, replicate padding.
    downsample: usize = 2,

    /// Samples a frame of codes spans: 1,920.
    pub fn samplesPerFrame(self: Config) usize {
        var n: usize = self.downsample;
        for (self.ratios) |r| n *= r;
        return n;
    }

    /// Channels into SEANet stage `s` (0–3) and, at `s = 4`, into the last convolution.
    fn channels(self: Config, s: usize) usize {
        return self.filters << @intCast(s);
    }
};

pub const Conv = struct { weight: Ref, bias: ?Ref };
pub const Stage = struct { conv1: Conv, conv2: Conv, down: Conv };
pub const Layer = struct {
    input_norm: Conv,
    post_norm: Conv,
    q: Ref,
    k: Ref,
    v: Ref,
    o: Ref,
    fc1: Ref,
    fc2: Ref,
    attn_scale: Ref,
    mlp_scale: Ref,
};

/// The encoder's own tensors, borrowing the checkpoint; `layers` is the
/// allocator's (`deinit`). The quantizer is the decoder binding's.
pub const Binding = struct {
    config: Config,
    first: Conv,
    stages: [4]Stage,
    last: Conv,
    layers: []Layer,
    downsample: Ref,
    tensors: u32,
    bytes: u64,

    pub fn deinit(self: *Binding, gpa: Allocator) void {
        gpa.free(self.layers);
        self.* = undefined;
    }
};

/// Binds the `encoder.*` tensors of `checkpoint` (the file `decoder` was
/// bound from) by name and shape. The encoder's quantizer copies must equal
/// the decoder's tensors byte for byte (`UnsupportedConfiguration`
/// otherwise); any other unknown `encoder.*` tensor is `UnexpectedTensor`.
/// Each is named in `diagnostic`.
pub fn bind(gpa: Allocator, checkpoint: *const safetensors.Checkpoint, decoder: *const codec.Binding, config: Config, diagnostic: ?*codec.Diagnostic) codec.Error!Binding {
    var binder: codec.Binder = .{ .gpa = gpa, .checkpoint = checkpoint, .diagnostic = diagnostic };
    defer binder.deinit();
    var binding = try bindWith(&binder, config);
    errdefer binding.deinit(gpa);
    for (checkpoint.shards) |shard| for (shard.document.tensors) |t| {
        if (!std.mem.startsWith(u8, t.name, "encoder.") or binder.bound.contains(t.name)) continue;
        if (!std.mem.startsWith(u8, t.name, "encoder.quantizer.")) {
            setDiagnostic(diagnostic, t.name);
            return error.UnexpectedTensor;
        }
    };
    try sameQuantizer(checkpoint, decoder, diagnostic);
    return binding;
}

fn setDiagnostic(d: ?*codec.Diagnostic, name: []const u8) void {
    const out = d orelse return;
    out.len = @min(name.len, out.buffer.len);
    @memcpy(out.buffer[0..out.len], name[0..out.len]);
}

/// The encoder's codebooks and projections, where the file has them, must
/// be the decoder's: the encode reads the decoder's.
fn sameQuantizer(checkpoint: *const safetensors.Checkpoint, decoder: *const codec.Binding, diagnostic: ?*codec.Diagnostic) codec.Error!void {
    var buffer: [160]u8 = undefined;
    for (0..groups) |g| {
        const at = if (g == 0) "semantic" else "acoustic";
        const layer = if (g == 0) 0 else g - 1;
        const pairs = [_]struct { []const u8, Ref }{ .{ "embed_sum", decoder.embedding_sum[g] }, .{ "cluster_usage", decoder.cluster_usage[g] } };
        for (pairs) |pair| {
            const name = std.fmt.bufPrint(&buffer, "encoder.quantizer.{s}_residual_vector_quantizer.layers.{d}.codebook.{s}", .{ at, layer, pair[0] }) catch unreachable;
            try sameBytes(checkpoint, name, pair[1], diagnostic);
        }
    }
    for ([2][]const u8{ "semantic", "acoustic" }, decoder.input_proj) |at, ref| {
        const name = std.fmt.bufPrint(&buffer, "encoder.quantizer.{s}_residual_vector_quantizer.input_proj.weight", .{at}) catch unreachable;
        try sameBytes(checkpoint, name, ref, diagnostic);
    }
}

fn sameBytes(checkpoint: *const safetensors.Checkpoint, name: []const u8, want: Ref, diagnostic: ?*codec.Diagnostic) codec.Error!void {
    const got = checkpoint.get(name) orelse return;
    if (got.tensor.dtype != want.tensor.dtype or !std.mem.eql(u64, got.tensor.shape, want.tensor.shape) or !std.mem.eql(u8, got.bytes, want.bytes)) {
        setDiagnostic(diagnostic, name);
        return error.UnsupportedConfiguration;
    }
}

fn bindWith(b: *codec.Binder, c: Config) codec.Error!Binding {
    if (c.filters == 0 or c.compress == 0 or c.channels(0) % c.compress != 0 or c.heads * c.head_dim == 0 or c.head_dim % 2 != 0 or
        c.downsample == 0 or c.kernel == 0 or c.last_kernel == 0 or c.residual_kernel == 0) return error.UnsupportedConfiguration;
    for (c.ratios) |r| if (r == 0) return error.UnsupportedConfiguration;
    var r: Binding = undefined;
    r.config = c;
    const conv = struct {
        fn get(binder: *codec.Binder, comptime format: []const u8, args: anytype, out: u64, in: u64, kernel: u64, bias: bool) codec.Error!Conv {
            return .{
                .weight = try binder.at("encoder." ++ format ++ ".weight", args, &.{ out, in, kernel }),
                .bias = if (bias) try binder.at("encoder." ++ format ++ ".bias", args, &.{out}) else null,
            };
        }
    }.get;
    r.first = try conv(b, "encoder.layers.0.conv", .{}, c.filters, 1, c.kernel, true);
    for (&r.stages, c.ratios, 0..) |*stage, ratio, s| {
        const width: u64 = c.channels(s);
        const hidden = width / c.compress;
        // `layers` index: the residual unit at 1 + 3s, its ELU, then the strided convolution.
        stage.* = .{
            .conv1 = try conv(b, "encoder.layers.{d}.block.1.conv", .{1 + 3 * s}, hidden, width, c.residual_kernel, true),
            .conv2 = try conv(b, "encoder.layers.{d}.block.3.conv", .{1 + 3 * s}, width, hidden, 1, true),
            .down = try conv(b, "encoder.layers.{d}.conv", .{3 + 3 * s}, 2 * width, width, 2 * ratio, true),
        };
    }
    r.last = try conv(b, "encoder.layers.{d}.conv", .{2 + 3 * c.ratios.len}, c.hidden, c.channels(c.ratios.len), c.last_kernel, true);
    const h: u64 = c.hidden;
    const width: u64 = c.heads * c.head_dim;
    const f: u64 = c.intermediate;
    r.layers = try b.gpa.alloc(Layer, c.layers);
    errdefer b.gpa.free(r.layers);
    for (r.layers, 0..) |*l, i| {
        const p = "encoder.encoder_transformer.layers.{d}.";
        l.* = .{
            .input_norm = .{ .weight = try b.at(p ++ "input_layernorm.weight", .{i}, &.{h}), .bias = try b.at(p ++ "input_layernorm.bias", .{i}, &.{h}) },
            .post_norm = .{ .weight = try b.at(p ++ "post_attention_layernorm.weight", .{i}, &.{h}), .bias = try b.at(p ++ "post_attention_layernorm.bias", .{i}, &.{h}) },
            .q = try b.at(p ++ "self_attn.q_proj.weight", .{i}, &.{ width, h }),
            .k = try b.at(p ++ "self_attn.k_proj.weight", .{i}, &.{ width, h }),
            .v = try b.at(p ++ "self_attn.v_proj.weight", .{i}, &.{ width, h }),
            .o = try b.at(p ++ "self_attn.o_proj.weight", .{i}, &.{ h, width }),
            .fc1 = try b.at(p ++ "mlp.fc1.weight", .{i}, &.{ f, h }),
            .fc2 = try b.at(p ++ "mlp.fc2.weight", .{i}, &.{ h, f }),
            .attn_scale = try b.at(p ++ "self_attn_layer_scale.scale", .{i}, &.{h}),
            .mlp_scale = try b.at(p ++ "mlp_layer_scale.scale", .{i}, &.{h}),
        };
    }
    r.downsample = try b.at("encoder.downsample.conv.weight", .{}, &.{ h, h, 2 * c.downsample });
    r.tensors = b.tensors;
    r.bytes = b.bytes;
    return r;
}

// ---- the forward -----------------------------------------------------------

/// A causal convolution: `w` is `[out][in][kernel]`; output row t reads
/// input rows `t · stride + j − (kernel − stride)`, j < kernel, and gives
/// ⌈rows / stride⌉ rows. Rows before and after the input read as `pad` says.
const StridedConv = struct { w: []const f32, b: ?[]const f32, in: usize, out: usize, kernel: usize, stride: usize };
const Pad = enum { zeros, replicate };
const Linear = struct { w: []const f32, in: usize, out: usize };
const AttnLayer = struct {
    input_norm: [2][]const f32,
    post_norm: [2][]const f32,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    fc1: Linear,
    fc2: Linear,
    attn_scale: []const f32,
    mlp_scale: []const f32,
};
const Unit = struct { conv1: StridedConv, conv2: StridedConv, down: StridedConv };

/// Floats of gathered rows per matrix multiply.
const gather_budget = 1 << 20;

/// Sees each stage's rows: `seanet` and `transformer` at 25 Hz,
/// `downsample` at 12.5 Hz.
pub const Observer = struct {
    context: *anyopaque,
    record: *const fn (context: *anyopaque, label: []const u8, rows: []const f32) void,

    fn see(self: ?Observer, label: []const u8, rows: []const f32) void {
        if (self) |o| o.record(o.context, label, rows);
    }
};

/// The encoder's weights and the decoder's quantizer, decoded to F32 and
/// owned by `storage`; read-only once built.
pub const Encoder = struct {
    io: std.Io,
    config: Config,
    codebook_size: usize,
    codebook_dim: usize,
    storage: std.heap.ArenaAllocator,
    first: StridedConv,
    units: [4]Unit,
    last: StridedConv,
    layers: []const AttnLayer,
    downsample: StridedConv,
    /// `rvq_first`'s and `rvq_rest`'s projections from `hidden` to a codebook vector.
    project: [2]Linear,
    /// `[groups][codebook_size][codebook_dim]`, divided by the usage.
    codebooks: []const f32,
    /// F32 bytes held.
    bytes: usize,

    /// Decodes `binding` and `decoder`'s quantizer; both may be released afterwards.
    pub fn init(gpa: Allocator, io: std.Io, decoder: *const codec.Binding, binding: *const Binding) !Encoder {
        const c = binding.config;
        const d = decoder.config;
        if (d.quantized != c.hidden) return error.UnsupportedConfiguration;
        var storage: std.heap.ArenaAllocator = .init(gpa);
        errdefer storage.deinit();
        var l: Loader = .{ .arena = storage.allocator() };
        var self: Encoder = undefined;
        self.io = io;
        self.config = c;
        self.codebook_size = d.codebook_size;
        self.codebook_dim = d.codebook_dim;
        self.first = try l.conv(binding.first, 1);
        for (&self.units, binding.stages, c.ratios) |*u, s, ratio| u.* = .{
            .conv1 = try l.conv(s.conv1, 1),
            .conv2 = try l.conv(s.conv2, 1),
            .down = try l.conv(s.down, ratio),
        };
        self.last = try l.conv(binding.last, 1);
        const layers = try l.arena.alloc(AttnLayer, binding.layers.len);
        for (layers, binding.layers) |*out, in| out.* = .{
            .input_norm = .{ try l.floats(in.input_norm.weight), try l.floats(in.input_norm.bias.?) },
            .post_norm = .{ try l.floats(in.post_norm.weight), try l.floats(in.post_norm.bias.?) },
            .q = try l.linear(in.q),
            .k = try l.linear(in.k),
            .v = try l.linear(in.v),
            .o = try l.linear(in.o),
            .fc1 = try l.linear(in.fc1),
            .fc2 = try l.linear(in.fc2),
            .attn_scale = try l.floats(in.attn_scale),
            .mlp_scale = try l.floats(in.mlp_scale),
        };
        self.layers = layers;
        self.downsample = try l.conv(.{ .weight = binding.downsample, .bias = null }, c.downsample);
        for (&self.project, decoder.input_proj) |*p, ref| p.* = try l.linear(ref);
        const books = try l.arena.alloc(f32, groups * d.codebook_size * d.codebook_dim);
        try codec.decodeCodebooks(gpa, decoder, books);
        l.bytes += books.len * 4;
        self.codebooks = books;
        self.bytes = l.bytes;
        self.storage = storage;
        return self;
    }

    pub fn deinit(self: *Encoder) void {
        self.storage.deinit();
        self.* = undefined;
    }

    /// The frames `samples` samples give: ⌈samples / 1920⌉.
    pub fn frameCount(self: *const Encoder, samples: usize) usize {
        return std.math.divCeil(usize, samples, self.config.samplesPerFrame()) catch unreachable;
    }

    /// Encodes `samples` (24 kHz mono, at least one) into `out`
    /// (`frameCount(samples.len)` frames). Scratch comes from `gpa` and is
    /// freed before it returns.
    pub fn encode(self: *const Encoder, gpa: Allocator, samples: []const f32, out: []Frame, observer: ?Observer) !void {
        if (samples.len == 0) return error.EmptyClip;
        std.debug.assert(out.len == self.frameCount(samples.len));
        var arena_state: std.heap.ArenaAllocator = .init(gpa);
        defer arena_state.deinit();
        const ws = arena_state.allocator();
        const io = self.io;
        const c = self.config;

        var rows = samples.len;
        var x = try ws.alloc(f32, rows * c.filters);
        try convolve(ws, io, self.first, samples, rows, .zeros, x);
        for (self.units) |unit| {
            // The residual unit: x + conv2(ELU(conv1(ELU(x)))).
            const width = unit.conv1.in;
            const activated = try ws.alloc(f32, rows * width);
            for (activated, x) |*a, v| a.* = elu(v);
            const hidden = try ws.alloc(f32, rows * unit.conv1.out);
            try convolve(ws, io, unit.conv1, activated, rows, .zeros, hidden);
            for (hidden) |*v| v.* = elu(v.*);
            try convolve(ws, io, unit.conv2, hidden, rows, .zeros, activated);
            for (x, activated) |*v, r| v.* += r;
            for (x) |*v| v.* = elu(v.*);
            const next = std.math.divCeil(usize, rows, unit.down.stride) catch unreachable;
            const y = try ws.alloc(f32, next * unit.down.out);
            try convolve(ws, io, unit.down, x, rows, .zeros, y);
            x = y;
            rows = next;
        }
        for (x) |*v| v.* = elu(v.*);
        const latent = try ws.alloc(f32, rows * c.hidden);
        try convolve(ws, io, self.last, x, rows, .zeros, latent);
        Observer.see(observer, "seanet", latent);

        try self.transformer(ws, latent, rows);
        Observer.see(observer, "transformer", latent);

        const frames = std.math.divCeil(usize, rows, c.downsample) catch unreachable;
        std.debug.assert(frames == out.len);
        const reduced = try ws.alloc(f32, frames * c.hidden);
        try convolve(ws, io, self.downsample, latent, rows, .replicate, reduced);
        Observer.see(observer, "downsample", reduced);

        try self.quantize(ws, reduced, out);
    }

    /// The causal transformer in place over `x` (`[rows][hidden]`): every
    /// query sees every earlier row, as the authors' SDPA mask is (the
    /// configured sliding window is not applied there).
    fn transformer(self: *const Encoder, ws: Allocator, x: []f32, rows: usize) !void {
        const c = self.config;
        const io = self.io;
        const width = c.heads * c.head_dim;
        const h = try ws.alloc(f32, rows * c.hidden);
        const q = try ws.alloc(f32, rows * width);
        const k = try ws.alloc(f32, rows * width);
        const v = try ws.alloc(f32, rows * width);
        const attended = try ws.alloc(f32, rows * width);
        const o = try ws.alloc(f32, rows * c.hidden);
        const wide = try ws.alloc(f32, rows * c.intermediate);
        const scores = try ws.alloc(f64, rows);
        const shape: codec.Config = .{ .heads = c.heads, .head_dim = c.head_dim, .window = rows };
        for (self.layers) |layer| {
            try layerNorm(x, layer.input_norm, c.norm_eps, h);
            try linear(io, layer.q, h, rows, q);
            try linear(io, layer.k, h, rows, k);
            try linear(io, layer.v, h, rows, v);
            for (0..rows) |t| for (0..c.heads) |head| {
                const options: cpu.rope.Options = .{ .dimensions = c.head_dim, .base = c.rope_theta, .position = @intCast(t) };
                const at = t * width + head * c.head_dim;
                try cpu.rope.apply(q[at..][0..c.head_dim], q[at..][0..c.head_dim], options);
                try cpu.rope.apply(k[at..][0..c.head_dim], k[at..][0..c.head_dim], options);
            };
            codec.attention(shape, q, k, v, rows, 0, scores, attended);
            try linear(io, layer.o, attended, rows, o);
            codec.addScaled(x, layer.attn_scale, o);
            try layerNorm(x, layer.post_norm, c.norm_eps, h);
            try linear(io, layer.fc1, h, rows, wide);
            for (wide) |*w| w.* = cpu.geluErf(w.*);
            try linear(io, layer.fc2, wide, rows, o);
            codec.addScaled(x, layer.mlp_scale, o);
        }
    }

    /// The split quantizer's encode: group 0 the nearest of `rvq_first`'s
    /// codewords to its projection; groups 1–15 residually over `rvq_rest`'s
    /// projection of the same rows, each subtracting its codeword.
    fn quantize(self: *const Encoder, ws: Allocator, x: []const f32, out: []Frame) !void {
        const frames = out.len;
        const dim = self.codebook_dim;
        const semantic = try ws.alloc(f32, frames * dim);
        const acoustic = try ws.alloc(f32, frames * dim);
        try linear(self.io, self.project[0], x, frames, semantic);
        try linear(self.io, self.project[1], x, frames, acoustic);
        for (out, 0..) |*frame, t| {
            frame[0] = self.nearest(0, semantic[t * dim ..][0..dim]);
            const residual = acoustic[t * dim ..][0..dim];
            for (1..groups) |g| {
                const code = self.nearest(g, residual);
                frame[g] = code;
                for (residual, self.codeword(g, code)) |*r, e| r.* -= e;
            }
        }
    }

    fn codeword(self: *const Encoder, group: usize, code: u16) []const f32 {
        return self.codebooks[(group * self.codebook_size + code) * self.codebook_dim ..][0..self.codebook_dim];
    }

    /// The code whose codeword is nearest `x` in group `group`; the lowest on a tie, as `argmin`.
    fn nearest(self: *const Encoder, group: usize, x: []const f32) u16 {
        var best: u16 = 0;
        var least = std.math.inf(f64);
        for (0..self.codebook_size) |code| {
            var sum: f64 = 0;
            for (x, self.codeword(group, @intCast(code))) |a, e| {
                const diff: f64 = @as(f64, a) - e;
                sum += diff * diff;
            }
            if (sum < least) {
                least = sum;
                best = @intCast(code);
            }
        }
        return best;
    }
};

fn elu(x: f32) f32 {
    return if (x > 0) x else @exp(x) - 1;
}

fn layerNorm(x: []const f32, norm: [2][]const f32, eps: f32, out: []f32) !void {
    const width = norm[0].len;
    for (0..x.len / width) |r| try dense.layerNorm(x[r * width ..][0..width], norm[0], norm[1], eps, out[r * width ..][0..width]);
}

fn linear(io: std.Io, m: Linear, x: []const f32, rows: usize, y: []f32) !void {
    try dense.matmul(io, .{ .rows = rows, .inner = m.in, .outputs = m.out, .x = x[0 .. rows * m.in], .w = m.w }, y[0 .. rows * m.out]);
}

/// `y = conv(x)` over `rows` input rows (`[rows][in]`), `y` holding
/// ⌈rows / stride⌉ rows: causal, `kernel − stride` padded rows before the
/// input and as many after as make whole strides.
fn convolve(ws: Allocator, io: std.Io, c: StridedConv, x: []const f32, rows: usize, pad: Pad, y: []f32) !void {
    const out_rows = std.math.divCeil(usize, rows, c.stride) catch unreachable;
    std.debug.assert(x.len == rows * c.in and y.len == out_rows * c.out and c.kernel >= c.stride);
    if (c.kernel == 1) return dense.matmul(io, .{ .rows = rows, .inner = c.in, .outputs = c.out, .x = x, .w = c.w, .bias = c.b }, y);
    const left = c.kernel - c.stride;
    const inner = c.in * c.kernel;
    const tile = @max(1, @min(out_rows, gather_budget / inner));
    const gathered = try ws.alloc(f32, tile * inner);
    var first: usize = 0;
    while (first < out_rows) : (first += tile) {
        const n = @min(tile, out_rows - first);
        for (0..n) |r| for (0..c.kernel) |j| {
            const source = @as(i64, @intCast((first + r) * c.stride + j)) - @as(i64, @intCast(left));
            const row: ?usize = if (source >= 0 and source < rows) @intCast(source) else switch (pad) {
                .zeros => null,
                .replicate => if (source < 0) 0 else rows - 1,
            };
            for (0..c.in) |i| gathered[r * inner + i * c.kernel + j] = if (row) |s| x[s * c.in + i] else 0;
        };
        try dense.matmul(io, .{ .rows = n, .inner = inner, .outputs = c.out, .x = gathered[0 .. n * inner], .w = c.w, .bias = c.b }, y[first * c.out ..][0 .. n * c.out]);
    }
}

/// Decodes binding refs into the encoder's arena.
const Loader = struct {
    arena: Allocator,
    bytes: usize = 0,

    fn floats(self: *Loader, ref: Ref) ![]f32 {
        const out = try self.arena.alloc(f32, @intCast(ref.tensor.elements));
        try ref.decode(0, out);
        self.bytes += out.len * 4;
        return out;
    }

    /// A [out][in] or [out][in][1] weight.
    fn linear(self: *Loader, ref: Ref) !Linear {
        const shape = ref.tensor.shape;
        return .{ .w = try self.floats(ref), .out = @intCast(shape[0]), .in = @intCast(shape[1]) };
    }

    fn conv(self: *Loader, c: Conv, stride: usize) !StridedConv {
        const shape = c.weight.tensor.shape;
        return .{
            .w = try self.floats(c.weight),
            .b = if (c.bias) |b| try self.floats(b) else null,
            .out = @intCast(shape[0]),
            .in = @intCast(shape[1]),
            .kernel = @intCast(shape[2]),
            .stride = stride,
        };
    }
};

// ---- tests -----------------------------------------------------------------

/// A tiny encoder over `codec.test_config` (hidden 6 = its `quantized`):
/// 2·2·2·2 · 2 = 32 samples a frame.
const tiny: Config = .{ .filters = 2, .ratios = .{ 2, 2, 2, 2 }, .hidden = 6, .layers = 2, .heads = 2, .head_dim = 4, .intermediate = 12 };

fn specs(gpa: Allocator, config: Config) !std.ArrayList(codec.Spec) {
    var list: std.ArrayList(codec.Spec) = .empty;
    errdefer {
        for (list.items) |s| {
            gpa.free(s.name);
            gpa.free(s.shape);
        }
        list.deinit(gpa);
    }
    var binder: codec.Binder = .{ .gpa = gpa, .checkpoint = null, .record = &list, .diagnostic = null };
    defer binder.deinit();
    var binding = try bindWith(&binder, config);
    binding.deinit(gpa);
    return list;
}

const TinyFile = struct {
    tmp: std.testing.TmpDir,
    checkpoint: safetensors.Checkpoint,
    list: std.ArrayList(codec.Spec),

    fn open(io: std.Io, more: []const codec.Spec, change: codec.Change) !TinyFile {
        const gpa = std.testing.allocator;
        var list = try specs(gpa, tiny);
        errdefer freeList(gpa, &list);
        const all = try std.mem.concat(gpa, codec.Spec, &.{ list.items, more });
        defer gpa.free(all);
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        try codec.writeTestCheckpoint(gpa, io, tmp.dir, all, change);
        const path = try tmp.dir.realPathFileAlloc(io, ".", gpa);
        defer gpa.free(path);
        return .{ .tmp = tmp, .checkpoint = try safetensors.Checkpoint.open(gpa, io, path, .{}), .list = list };
    }

    fn freeList(gpa: Allocator, list: *std.ArrayList(codec.Spec)) void {
        for (list.items) |s| {
            gpa.free(s.name);
            gpa.free(s.shape);
        }
        list.deinit(gpa);
    }

    fn close(self: *TinyFile, io: std.Io) void {
        freeList(std.testing.allocator, &self.list);
        self.checkpoint.deinit(io);
        self.tmp.cleanup();
    }
};

test "the published encoder binds 125 tensors of its own and spans 1,920 samples a frame" {
    var list = try specs(std.testing.allocator, .{});
    defer TinyFile.freeList(std.testing.allocator, &list);
    try std.testing.expectEqual(@as(usize, 125), list.items.len);
    try std.testing.expectEqual(@as(usize, 1920), (Config{}).samplesPerFrame());
}

test "bind refuses a missing or unknown encoder tensor and a quantizer copy that differs, by name" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    {
        // A copy the decoder's tensor is not: refused.
        const copy: codec.Spec = .{ .name = "encoder.quantizer.acoustic_residual_vector_quantizer.layers.3.codebook.embed_sum", .shape = &.{ codec.test_config.codebook_size, codec.test_config.codebook_dim } };
        var file = try TinyFile.open(io, &.{copy}, .{});
        defer file.close(io);
        var decoder = try codec.bind(gpa, &file.checkpoint, codec.test_config, null);
        defer decoder.deinit(gpa);
        var diagnostic: codec.Diagnostic = .{};
        try std.testing.expectError(error.UnsupportedConfiguration, bind(gpa, &file.checkpoint, &decoder, tiny, &diagnostic));
        try std.testing.expectEqualStrings(copy.name, diagnostic.name());
    }
    const Case = struct { change: codec.Change, err: codec.Error, name: []const u8 };
    const cases = [_]Case{
        .{ .change = .{ .skip = "encoder.encoder.layers.7.block.1.conv.bias" }, .err = error.MissingTensor, .name = "encoder.encoder.layers.7.block.1.conv.bias" },
        .{ .change = .{ .extra = .{ .name = "encoder.decoder.layers.0.conv.weight", .shape = &.{3} } }, .err = error.UnexpectedTensor, .name = "encoder.decoder.layers.0.conv.weight" },
    };
    for (cases) |case| {
        var file = try TinyFile.open(io, &.{}, case.change);
        defer file.close(io);
        var decoder = try codec.bind(gpa, &file.checkpoint, codec.test_config, null);
        defer decoder.deinit(gpa);
        var diagnostic: codec.Diagnostic = .{};
        try std.testing.expectError(case.err, bind(gpa, &file.checkpoint, &decoder, tiny, &diagnostic));
        try std.testing.expectEqualStrings(case.name, diagnostic.name());
    }
}

test "a causal convolution pads zeros or the edge row on the left and fills the last stride" {
    const gpa = std.testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    // One channel, kernel 4, stride 2: output t reads inputs 2t − 2 … 2t + 1.
    const x = [_]f32{ 1, 2, 3, 4, 5 };
    const c: StridedConv = .{ .w = &.{ 1, 10, 100, 1000 }, .b = null, .in = 1, .out = 1, .kernel = 4, .stride = 2 };
    var y: [3]f32 = undefined;
    try convolve(arena_state.allocator(), std.testing.io, c, &x, 5, .zeros, &y);
    try std.testing.expectEqualSlices(f32, &.{ 100 * 1 + 1000 * 2, 1 + 10 * 2 + 100 * 3 + 1000 * 4, 3 + 10 * 4 + 100 * 5 }, &y);
    try convolve(arena_state.allocator(), std.testing.io, c, &x, 5, .replicate, &y);
    try std.testing.expectEqualSlices(f32, &.{ 1 + 10 * 1 + 100 * 1 + 1000 * 2, 1 + 10 * 2 + 100 * 3 + 1000 * 4, 3 + 10 * 4 + 100 * 5 + 1000 * 5 }, &y);
}

test "the tiny encoder: ⌈L / 32⌉ frames of codes in range, each stage once, the same codes twice" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var file = try TinyFile.open(io, &.{}, .{});
    defer file.close(io);
    var decoder = try codec.bind(gpa, &file.checkpoint, codec.test_config, null);
    defer decoder.deinit(gpa);
    var binding = try bind(gpa, &file.checkpoint, &decoder, tiny, null);
    defer binding.deinit(gpa);
    var encoder = try Encoder.init(gpa, io, &decoder, &binding);
    defer encoder.deinit();

    var samples: [100]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(9);
    for (&samples) |*s| s.* = prng.random().float(f32) - 0.5;
    try std.testing.expectEqual(@as(usize, 4), encoder.frameCount(samples.len));
    const Seen = struct {
        sizes: [3]usize = .{ 0, 0, 0 },
        fn record(context: *anyopaque, label: []const u8, rows: []const f32) void {
            const self: *@This() = @ptrCast(@alignCast(context));
            const i: usize = if (std.mem.eql(u8, label, "seanet")) 0 else if (std.mem.eql(u8, label, "transformer")) 1 else 2;
            self.sizes[i] += rows.len;
        }
    };
    var seen: Seen = .{};
    var codes: [4]Frame = undefined;
    try encoder.encode(gpa, &samples, &codes, .{ .context = &seen, .record = Seen.record });
    // 100 samples: ⌈100/16⌉ = 7 rows at 25 Hz (the tiny one's), 4 after the downsample.
    try std.testing.expectEqualSlices(usize, &.{ 7 * 6, 7 * 6, 4 * 6 }, &seen.sizes);
    for (codes) |frame| for (frame) |code| try std.testing.expect(code < codec.test_config.codebook_size);
    var again: [4]Frame = undefined;
    try encoder.encode(gpa, &samples, &again, null);
    try std.testing.expectEqualSlices(Frame, &codes, &again);
    try std.testing.expectError(error.EmptyClip, encoder.encode(gpa, &.{}, codes[0..0], null));
}

test "a code is the nearest codeword; the residual groups subtract theirs" {
    // Group 0 looks at the semantic projection alone; group g ≥ 1 at what
    // the groups before it left of the acoustic one.
    const gpa = std.testing.allocator;
    const books = try gpa.alloc(f32, groups * 3 * 2);
    defer gpa.free(books);
    @memset(books, 100);
    // Group 0: codewords (0,0), (1,0), (0,1); group 1: (0.5,0), (0,0), …; group 2: (0, 0.25) at code 2.
    @memcpy(books[0..6], &[_]f32{ 0, 0, 1, 0, 0, 1 });
    @memcpy(books[6..12], &[_]f32{ 0.5, 0, 0, 0, 100, 100 });
    @memcpy(books[12..18], &[_]f32{ 100, 100, 100, 100, 0, 0.25 });
    var encoder: Encoder = undefined;
    encoder.codebook_size = 3;
    encoder.codebook_dim = 2;
    encoder.codebooks = books;
    try std.testing.expectEqual(@as(u16, 2), encoder.nearest(0, &.{ 0.1, 0.9 }));
    var residual = [_]f32{ 0.5, 0.25 };
    const first = encoder.nearest(1, &residual);
    try std.testing.expectEqual(@as(u16, 0), first);
    for (&residual, encoder.codeword(1, first)) |*r, e| r.* -= e;
    try std.testing.expectEqual(@as(u16, 2), encoder.nearest(2, &residual));
    // A tie goes to the lower code.
    try std.testing.expectEqual(@as(u16, 0), encoder.nearest(0, &.{ 0.5, 0 }));
}
