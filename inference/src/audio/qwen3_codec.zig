//! Qwen3-TTS's codec decoder: frames of 16 codes become 24 kHz mono
//! samples, 1,920 per frame. `bind` checks the `decoder.*` tensors of
//! `speech_tokenizer/model.safetensors` by name and shape (the `encoder.*`
//! ones are skipped by name); `Runtime` holds them decoded to F32, the CPU
//! reference; `Stream` decodes frames as they arrive, carrying every
//! convolution's left context and the attention window's keys and values,
//! so pieces of any size give one `decode`'s samples bit for bit (every
//! output element is summed in an order independent of the piece).
//! The pipeline and its constants: docs/models/qwen3-tts.md § The codec decoder.
const std = @import("std");
const safetensors = @import("../formats/safetensors.zig");
const cpu = @import("../backends/cpu/root.zig");
const alloc_check = @import("../alloc_check.zig");

const Allocator = std.mem.Allocator;
const Ref = safetensors.Ref;
const dense = cpu.dense;

/// Code groups per frame: the talker's group 0 and the predictor's 1–15.
pub const groups = 16;
pub const Frame = [groups]u16;
/// The residual units' dilations in each decoder block.
const dilations = [3]usize{ 1, 3, 9 };
const residual_kernel = 7;
const dwconv_kernel = 7;
const pre_conv_kernel = 3;
const layer_norm_eps = 1e-6;
const codebook_eps = 1e-5;
const snake_eps = 1e-9;

/// `speech_tokenizer/config.json`'s `decoder_config`; the defaults are the
/// published file's. Upsampling and block counts are the architecture's.
pub const Config = struct {
    codebook_size: usize = 2048,
    /// One codebook vector (the file config's `codebook_dim / 2`).
    codebook_dim: usize = 256,
    /// The quantizer's output width (the file config's `codebook_dim`).
    quantized: usize = 512,
    latent: usize = 1024,
    hidden: usize = 512,
    intermediate: usize = 1024,
    layers: usize = 8,
    heads: usize = 16,
    head_dim: usize = 64,
    /// Keys visible to a query, itself included.
    window: usize = 72,
    rope_theta: f32 = 10000,
    rms_eps: f32 = 1e-5,
    decoder_dim: usize = 1536,
    upsampling_ratios: [2]usize = .{ 2, 2 },
    upsample_rates: [4]usize = .{ 8, 5, 4, 3 },

    pub fn samplesPerFrame(self: Config) usize {
        var n: usize = 1;
        for (self.upsampling_ratios) |r| n *= r;
        for (self.upsample_rates) |r| n *= r;
        return n;
    }
    /// Channels into decoder block `i` (0–3) and, at `i = 4`, into the output.
    pub fn blockChannels(self: Config, i: usize) usize {
        return self.decoder_dim >> @intCast(i);
    }
};

pub const Error = Allocator.Error || error{ MissingTensor, TensorShapeMismatch, UnsupportedDtype, UnexpectedTensor, UnsupportedConfiguration };

/// Names the tensor behind a `bind` error.
pub const Diagnostic = struct {
    buffer: [160]u8 = undefined,
    len: usize = 0,

    pub fn name(self: *const Diagnostic) []const u8 {
        return self.buffer[0..self.len];
    }
    fn set(self: ?*Diagnostic, text: []const u8) void {
        const d = self orelse return;
        d.len = @min(text.len, d.buffer.len);
        @memcpy(d.buffer[0..d.len], text[0..d.len]);
    }
};

pub const Conv = struct { weight: Ref, bias: Ref };
pub const Snake = struct { alpha: Ref, beta: Ref };
pub const Layer = struct {
    input_norm: Ref,
    post_norm: Ref,
    q: Ref,
    k: Ref,
    v: Ref,
    o: Ref,
    gate: Ref,
    up: Ref,
    down: Ref,
    attn_scale: Ref,
    mlp_scale: Ref,
};
pub const ConvNeXt = struct { dwconv: Conv, norm: Conv, gamma: Ref, pw1: Conv, pw2: Conv };
pub const Upsample = struct { trans: Conv, block: ConvNeXt };
pub const Residual = struct { act1: Snake, conv1: Conv, act2: Snake, conv2: Conv };
pub const Block = struct { act: Snake, trans: Conv, units: [3]Residual };

/// The decoder's tensors; the refs borrow the checkpoint, `layers` is the
/// allocator's (`deinit`).
pub const Binding = struct {
    config: Config,
    embedding_sum: [groups]Ref,
    cluster_usage: [groups]Ref,
    /// `rvq_first`'s and `rvq_rest`'s 1×1 projections to `quantized`.
    output_proj: [2]Ref,
    pre_conv: Conv,
    transformer_in: Conv,
    layers: []Layer,
    norm: Ref,
    transformer_out: Conv,
    upsample: [2]Upsample,
    conv_in: Conv,
    blocks: [4]Block,
    act_out: Snake,
    conv_out: Conv,
    /// Decoder tensors bound (the encode side's two `input_proj`s included).
    tensors: u32,
    bytes: u64,

    pub fn deinit(self: *Binding, gpa: Allocator) void {
        gpa.free(self.layers);
        self.* = undefined;
    }
};

/// Looks tensors up by name and shape, or (with `record`) lists the names
/// and shapes asked for instead, which is how the tests build a file.
const Binder = struct {
    gpa: Allocator,
    checkpoint: ?*const safetensors.Checkpoint,
    bound: std.StringHashMapUnmanaged(void) = .empty,
    record: ?*std.ArrayList(Spec) = null,
    diagnostic: ?*Diagnostic,
    tensors: u32 = 0,
    bytes: u64 = 0,
    placeholder: safetensors.Tensor = .{ .name = "", .dtype = .f32, .shape = &.{}, .offset = 0, .bytes = 0, .elements = 0 },

    fn deinit(self: *Binder) void {
        self.bound.deinit(self.gpa);
    }

    fn get(self: *Binder, comptime format: []const u8, args: anytype, shape: []const u64) Error!Ref {
        var buffer: [160]u8 = undefined;
        const name = std.fmt.bufPrint(&buffer, "decoder." ++ format, args) catch unreachable;
        if (self.record) |list| {
            try list.ensureUnusedCapacity(self.gpa, 1);
            const owned = try self.gpa.dupe(u8, name);
            errdefer self.gpa.free(owned);
            const dims = try self.gpa.dupe(u64, shape);
            list.appendAssumeCapacity(.{ .name = owned, .shape = dims });
            return .{ .tensor = &self.placeholder, .bytes = &.{} };
        }
        const ref = self.checkpoint.?.get(name) orelse {
            Diagnostic.set(self.diagnostic, name);
            return error.MissingTensor;
        };
        if (!std.mem.eql(u64, ref.tensor.shape, shape)) {
            Diagnostic.set(self.diagnostic, name);
            return error.TensorShapeMismatch;
        }
        switch (ref.tensor.dtype) {
            .f32, .f16, .bf16 => {},
            else => {
                Diagnostic.set(self.diagnostic, name);
                return error.UnsupportedDtype;
            },
        }
        // The key borrows the checkpoint's own name.
        try self.bound.put(self.gpa, ref.tensor.name, {});
        self.tensors += 1;
        self.bytes += ref.tensor.bytes;
        return ref;
    }

    fn conv(self: *Binder, comptime prefix: []const u8, args: anytype, out: u64, in: u64, k: ?u64) Error!Conv {
        return .{
            .weight = try self.get(prefix ++ ".weight", args, if (k) |kernel| &.{ out, in, kernel } else &.{ out, in }),
            .bias = try self.get(prefix ++ ".bias", args, &.{out}),
        };
    }

    /// `ConvTranspose1d` stores [in][out][kernel].
    fn trans(self: *Binder, comptime prefix: []const u8, args: anytype, in: u64, out: u64, k: u64) Error!Conv {
        return .{
            .weight = try self.get(prefix ++ ".weight", args, &.{ in, out, k }),
            .bias = try self.get(prefix ++ ".bias", args, &.{out}),
        };
    }

    fn snake(self: *Binder, comptime prefix: []const u8, args: anytype, channels: u64) Error!Snake {
        return .{
            .alpha = try self.get(prefix ++ ".alpha", args, &.{channels}),
            .beta = try self.get(prefix ++ ".beta", args, &.{channels}),
        };
    }
};

/// A tensor the decoder reads, for building test files.
const Spec = struct { name: []const u8, shape: []const u64 };

/// Binds the decoder's tensors of `checkpoint` (a `speech_tokenizer`
/// file) against `config`. A missing, misshapen, or unknown `decoder.*`
/// tensor, or a tensor outside `decoder.*` and `encoder.*`, is refused and
/// named in `diagnostic`. The refs borrow `checkpoint`.
pub fn bind(gpa: Allocator, checkpoint: *const safetensors.Checkpoint, config: Config, diagnostic: ?*Diagnostic) Error!Binding {
    var binder: Binder = .{ .gpa = gpa, .checkpoint = checkpoint, .diagnostic = diagnostic };
    defer binder.deinit();
    var binding = try bindWith(&binder, config);
    errdefer binding.deinit(gpa);
    for (checkpoint.shards) |shard| for (shard.document.tensors) |t| {
        if (std.mem.startsWith(u8, t.name, "encoder.")) continue;
        if (!std.mem.startsWith(u8, t.name, "decoder.") or !binder.bound.contains(t.name)) {
            Diagnostic.set(diagnostic, t.name);
            return error.UnexpectedTensor;
        }
    };
    return binding;
}

fn bindWith(b: *Binder, c: Config) Error!Binding {
    if (c.heads * c.head_dim == 0 or c.window == 0 or c.decoder_dim % 16 != 0 or c.decoder_dim == 0) return error.UnsupportedConfiguration;
    for (c.upsampling_ratios) |x| if (x == 0) return error.UnsupportedConfiguration;
    for (c.upsample_rates) |x| if (x == 0) return error.UnsupportedConfiguration;
    var r: Binding = undefined;
    r.config = c;
    const cs: u64 = c.codebook_size;
    const cd: u64 = c.codebook_dim;
    for (0..groups) |g| {
        const at = if (g == 0) "first" else "rest";
        const layer = if (g == 0) 0 else g - 1;
        r.embedding_sum[g] = try b.get("quantizer.rvq_{s}.vq.layers.{d}._codebook.embedding_sum", .{ at, layer }, &.{ cs, cd });
        r.cluster_usage[g] = try b.get("quantizer.rvq_{s}.vq.layers.{d}._codebook.cluster_usage", .{ at, layer }, &.{cs});
    }
    for ([2][]const u8{ "first", "rest" }, 0..) |at, i| {
        r.output_proj[i] = try b.get("quantizer.rvq_{s}.output_proj.weight", .{at}, &.{ c.quantized, cd, 1 });
        // The encode side's projection: bound so the file holds nothing unknown.
        _ = try b.get("quantizer.rvq_{s}.input_proj.weight", .{at}, &.{ cd, c.quantized, 1 });
    }
    const latent: u64 = c.latent;
    const hidden: u64 = c.hidden;
    const width: u64 = c.heads * c.head_dim;
    r.pre_conv = try b.conv("pre_conv.conv", .{}, latent, c.quantized, pre_conv_kernel);
    r.transformer_in = try b.conv("pre_transformer.input_proj", .{}, hidden, latent, null);
    r.transformer_out = try b.conv("pre_transformer.output_proj", .{}, latent, hidden, null);
    r.norm = try b.get("pre_transformer.norm.weight", .{}, &.{hidden});
    r.layers = try b.gpa.alloc(Layer, c.layers);
    errdefer b.gpa.free(r.layers);
    for (r.layers, 0..) |*l, i| {
        const inter: u64 = c.intermediate;
        l.* = .{
            .input_norm = try b.get("pre_transformer.layers.{d}.input_layernorm.weight", .{i}, &.{hidden}),
            .post_norm = try b.get("pre_transformer.layers.{d}.post_attention_layernorm.weight", .{i}, &.{hidden}),
            .q = try b.get("pre_transformer.layers.{d}.self_attn.q_proj.weight", .{i}, &.{ width, hidden }),
            .k = try b.get("pre_transformer.layers.{d}.self_attn.k_proj.weight", .{i}, &.{ width, hidden }),
            .v = try b.get("pre_transformer.layers.{d}.self_attn.v_proj.weight", .{i}, &.{ width, hidden }),
            .o = try b.get("pre_transformer.layers.{d}.self_attn.o_proj.weight", .{i}, &.{ hidden, width }),
            .gate = try b.get("pre_transformer.layers.{d}.mlp.gate_proj.weight", .{i}, &.{ inter, hidden }),
            .up = try b.get("pre_transformer.layers.{d}.mlp.up_proj.weight", .{i}, &.{ inter, hidden }),
            .down = try b.get("pre_transformer.layers.{d}.mlp.down_proj.weight", .{i}, &.{ hidden, inter }),
            .attn_scale = try b.get("pre_transformer.layers.{d}.self_attn_layer_scale.scale", .{i}, &.{hidden}),
            .mlp_scale = try b.get("pre_transformer.layers.{d}.mlp_layer_scale.scale", .{i}, &.{hidden}),
        };
    }
    for (&r.upsample, c.upsampling_ratios, 0..) |*u, ratio, i| u.* = .{
        .trans = try b.trans("upsample.{d}.0.conv", .{i}, latent, latent, ratio),
        .block = .{
            .dwconv = try b.conv("upsample.{d}.1.dwconv.conv", .{i}, latent, 1, dwconv_kernel),
            .norm = .{
                .weight = try b.get("upsample.{d}.1.norm.weight", .{i}, &.{latent}),
                .bias = try b.get("upsample.{d}.1.norm.bias", .{i}, &.{latent}),
            },
            .gamma = try b.get("upsample.{d}.1.gamma", .{i}, &.{latent}),
            .pw1 = try b.conv("upsample.{d}.1.pwconv1", .{i}, 4 * latent, latent, null),
            .pw2 = try b.conv("upsample.{d}.1.pwconv2", .{i}, latent, 4 * latent, null),
        },
    };
    r.conv_in = try b.conv("decoder.0.conv", .{}, c.decoder_dim, latent, residual_kernel);
    for (&r.blocks, c.upsample_rates, 0..) |*blk, rate, i| {
        const in: u64 = c.blockChannels(i);
        const out: u64 = c.blockChannels(i + 1);
        blk.act = try b.snake("decoder.{d}.block.0", .{i + 1}, in);
        blk.trans = try b.trans("decoder.{d}.block.1.conv", .{i + 1}, in, out, 2 * rate);
        for (&blk.units, 0..) |*unit, u| unit.* = .{
            .act1 = try b.snake("decoder.{d}.block.{d}.act1", .{ i + 1, u + 2 }, out),
            .conv1 = try b.conv("decoder.{d}.block.{d}.conv1.conv", .{ i + 1, u + 2 }, out, out, residual_kernel),
            .act2 = try b.snake("decoder.{d}.block.{d}.act2", .{ i + 1, u + 2 }, out),
            .conv2 = try b.conv("decoder.{d}.block.{d}.conv2.conv", .{ i + 1, u + 2 }, out, out, 1),
        };
    }
    const last: u64 = c.blockChannels(4);
    r.act_out = try b.snake("decoder.5", .{}, last);
    r.conv_out = try b.conv("decoder.6.conv", .{}, 1, last, residual_kernel);
    r.tensors = b.tensors;
    r.bytes = b.bytes;
    return r;
}

// ---- the CPU reference ----------------------------------------------------

/// `y = W x (+ b)` with `w` [out][in].
const Linear = struct { w: []const f32, b: ?[]const f32, in: usize, out: usize };
/// A causal convolution as one matrix over gathered rows: `w` is
/// [out][kernel · in], tap-major, so each output is one dot product.
const CausalConv = struct { w: []const f32, b: []const f32, in: usize, out: usize, kernel: usize, dilation: usize };
/// A transposed convolution of kernel `taps · stride` as one matrix: input
/// row m with the `taps − 1` before it gives the `stride` output rows
/// m·stride … m·stride + stride − 1. `w` is [stride · out][taps · in].
const TransConv = struct { w: []const f32, b: []const f32, in: usize, out: usize, stride: usize, taps: usize };
/// SnakeBeta with its logarithms already raised: x + inv · sin²(scale · x).
pub const SnakeBeta = struct { scale: []const f32, inv: []const f32 };
const NeXt = struct { dw: []const f32, dw_b: []const f32, ln_w: []const f32, ln_b: []const f32, gamma: []const f32, pw1: Linear, pw2: Linear };
const AttnLayer = struct { input_norm: []const f32, post_norm: []const f32, q: Linear, k: Linear, v: Linear, o: Linear, gate: Linear, up: Linear, down: Linear, attn_scale: []const f32, mlp_scale: []const f32 };
const UpStage = struct { trans: TransConv, next: NeXt };
const Unit = struct { act1: SnakeBeta, conv1: CausalConv, act2: SnakeBeta, conv2: CausalConv };
const DecBlock = struct { act: SnakeBeta, trans: TransConv, units: [3]Unit };

/// A stage the decoder's observer is shown, each `[rows][channels]`.
pub const Stage = enum {
    quantized,
    pre_conv,
    transformer,
    upsample_0,
    upsample_1,
    decoder_0,
    decoder_1,
    decoder_2,
    decoder_3,
    decoder_4,

    /// The trace fixture's name for the stage.
    pub fn label(self: Stage) []const u8 {
        return switch (self) {
            .upsample_0 => "upsample.0",
            .upsample_1 => "upsample.1",
            .decoder_0 => "decoder.0",
            .decoder_1 => "decoder.1",
            .decoder_2 => "decoder.2",
            .decoder_3 => "decoder.3",
            .decoder_4 => "decoder.4",
            else => @tagName(self),
        };
    }
};

/// Sees each stage's rows of each push, in order.
pub const Observer = struct {
    context: *anyopaque,
    record: *const fn (context: *anyopaque, stage: Stage, rows: []const f32) void,

    fn see(self: ?Observer, stage: Stage, rows: []const f32) void {
        if (self) |o| o.record(o.context, stage, rows);
    }
};

/// Frames `Runtime.decode` pushes at once, bounding its working memory.
const decode_frames = 32;
/// Floats of gathered rows per matrix multiply.
const gather_budget = 1 << 20;

/// The decoder's weights decoded to F32 (about 460 MB for the published
/// file), owned by `storage`; read-only once built, so streams may share it.
pub const Runtime = struct {
    io: std.Io,
    config: Config,
    storage: std.heap.ArenaAllocator,
    /// [groups][codebook_size][codebook_dim], already divided by the usage.
    codebooks: []const f32,
    project: [2]Linear,
    pre_conv: CausalConv,
    input: Linear,
    layers: []const AttnLayer,
    norm: []const f32,
    output: Linear,
    upsample: [2]UpStage,
    conv_in: CausalConv,
    blocks: [4]DecBlock,
    act_out: SnakeBeta,
    conv_out: CausalConv,

    /// Decodes every bound tensor; `binding` may be released afterwards.
    pub fn init(gpa: Allocator, io: std.Io, binding: *const Binding) !Runtime {
        var self: Runtime = undefined;
        self.io = io;
        self.config = binding.config;
        self.storage = .init(gpa);
        errdefer self.storage.deinit();
        const l: Loader = .{ .arena = self.storage.allocator(), .gpa = gpa };
        const c = binding.config;
        const codebooks = try l.arena.alloc(f32, groups * c.codebook_size * c.codebook_dim);
        const usage = try gpa.alloc(f32, c.codebook_size);
        defer gpa.free(usage);
        for (0..groups) |g| {
            const table = codebooks[g * c.codebook_size * c.codebook_dim ..][0 .. c.codebook_size * c.codebook_dim];
            try binding.embedding_sum[g].decode(0, table);
            try binding.cluster_usage[g].decode(0, usage);
            for (0..c.codebook_size) |code| for (table[code * c.codebook_dim ..][0..c.codebook_dim]) |*x| {
                x.* /= @max(usage[code], codebook_eps);
            };
        }
        self.codebooks = codebooks;
        for (&self.project, binding.output_proj) |*p, ref| p.* = .{ .w = try l.floats(ref), .b = null, .in = c.codebook_dim, .out = c.quantized };
        self.pre_conv = try l.causal(binding.pre_conv, 1);
        self.input = try l.linear(binding.transformer_in);
        const layers = try l.arena.alloc(AttnLayer, c.layers);
        for (layers, binding.layers) |*out, in| out.* = .{
            .input_norm = try l.floats(in.input_norm),
            .post_norm = try l.floats(in.post_norm),
            .q = try l.matrix(in.q, null),
            .k = try l.matrix(in.k, null),
            .v = try l.matrix(in.v, null),
            .o = try l.matrix(in.o, null),
            .gate = try l.matrix(in.gate, null),
            .up = try l.matrix(in.up, null),
            .down = try l.matrix(in.down, null),
            .attn_scale = try l.floats(in.attn_scale),
            .mlp_scale = try l.floats(in.mlp_scale),
        };
        self.layers = layers;
        self.norm = try l.floats(binding.norm);
        self.output = try l.linear(binding.transformer_out);
        for (&self.upsample, binding.upsample, c.upsampling_ratios) |*out, in, ratio| out.* = .{
            .trans = try l.transposed(in.trans, ratio),
            .next = .{
                .dw = try l.floats(in.block.dwconv.weight),
                .dw_b = try l.floats(in.block.dwconv.bias),
                .ln_w = try l.floats(in.block.norm.weight),
                .ln_b = try l.floats(in.block.norm.bias),
                .gamma = try l.floats(in.block.gamma),
                .pw1 = try l.linear(in.block.pw1),
                .pw2 = try l.linear(in.block.pw2),
            },
        };
        self.conv_in = try l.causal(binding.conv_in, 1);
        for (&self.blocks, binding.blocks, c.upsample_rates) |*out, in, rate| {
            out.act = try l.snake(in.act);
            out.trans = try l.transposed(in.trans, rate);
            for (&out.units, in.units, dilations) |*u, unit, d| u.* = .{
                .act1 = try l.snake(unit.act1),
                .conv1 = try l.causal(unit.conv1, d),
                .act2 = try l.snake(unit.act2),
                .conv2 = try l.causal(unit.conv2, 1),
            };
        }
        self.act_out = try l.snake(binding.act_out);
        self.conv_out = try l.causal(binding.conv_out, 1);
        return self;
    }

    pub fn deinit(self: *Runtime) void {
        self.storage.deinit();
        self.* = undefined;
    }

    /// Decodes `codes` from a fresh state into `out`
    /// (`codes.len · samplesPerFrame()` samples in [−1, 1]).
    pub fn decode(self: *const Runtime, gpa: Allocator, codes: []const Frame, out: []f32) !void {
        return self.decodeObserved(gpa, codes, out, null);
    }

    /// `decode`, with `observer` shown each stage.
    pub fn decodeObserved(self: *const Runtime, gpa: Allocator, codes: []const Frame, out: []f32, observer: ?Observer) !void {
        const per = self.config.samplesPerFrame();
        if (out.len != codes.len * per) return error.InvalidShape;
        var stream = try Stream.init(gpa, self);
        defer stream.deinit();
        var at: usize = 0;
        while (at < codes.len) {
            const n = @min(decode_frames, codes.len - at);
            try stream.pushObserved(codes[at..][0..n], out[at * per ..][0 .. n * per], observer);
            at += n;
        }
    }

    /// Group 0 through `rvq_first`, groups 1–15 summed and then through
    /// `rvq_rest`, as the authors' split quantizer: [n][quantized].
    fn quantize(self: *const Runtime, ws: Allocator, codes: []const Frame, out: []f32) !void {
        const c = self.config;
        const n = codes.len;
        const first = try ws.alloc(f32, n * c.codebook_dim);
        const rest = try ws.alloc(f32, n * c.codebook_dim);
        self.lookup(codes, first, rest);
        const projected = try ws.alloc(f32, n * c.quantized);
        try linear(self.io, self.project[0], first, n, out);
        try linear(self.io, self.project[1], rest, n, projected);
        for (out, projected) |*a, b| a.* += b;
    }

    fn lookup(self: *const Runtime, codes: []const Frame, first: []f32, rest: []f32) void {
        sumCodebooks(self.codebooks, self.config.codebook_size, self.config.codebook_dim, codes, first, rest);
    }
};

/// Decodes `Binding` refs into the runtime's arena, re-laying convolution
/// kernels for `CausalConv` and `TransConv`.
const Loader = struct {
    arena: Allocator,
    gpa: Allocator,

    fn floats(self: Loader, ref: Ref) ![]f32 {
        const out = try self.arena.alloc(f32, @intCast(ref.tensor.elements));
        try ref.decode(0, out);
        return out;
    }

    fn temporary(self: Loader, ref: Ref) ![]f32 {
        const out = try self.gpa.alloc(f32, @intCast(ref.tensor.elements));
        errdefer self.gpa.free(out);
        try ref.decode(0, out);
        return out;
    }

    fn matrix(self: Loader, weight: Ref, bias: ?Ref) !Linear {
        const shape = weight.tensor.shape;
        return .{ .w = try self.floats(weight), .b = if (bias) |b| try self.floats(b) else null, .in = @intCast(shape[1]), .out = @intCast(shape[0]) };
    }

    fn linear(self: Loader, conv: Conv) !Linear {
        return self.matrix(conv.weight, conv.bias);
    }

    /// [out][in][k] to [out][k][in].
    fn causal(self: Loader, conv: Conv, dilation: usize) !CausalConv {
        const shape = conv.weight.tensor.shape;
        const out: usize = @intCast(shape[0]);
        const in: usize = @intCast(shape[1]);
        const k: usize = @intCast(shape[2]);
        const raw = try self.temporary(conv.weight);
        defer self.gpa.free(raw);
        const w = try self.arena.alloc(f32, raw.len);
        for (0..out) |o| for (0..in) |i| for (0..k) |j| {
            w[(o * k + j) * in + i] = raw[(o * in + i) * k + j];
        };
        return .{ .w = w, .b = try self.floats(conv.bias), .in = in, .out = out, .kernel = k, .dilation = dilation };
    }

    /// `ConvTranspose1d`'s [in][out][taps · stride] to [stride][out][taps][in]:
    /// output row m·stride + p reads tap p + q·stride of input row m − q.
    fn transposed(self: Loader, conv: Conv, stride: usize) !TransConv {
        const shape = conv.weight.tensor.shape;
        const in: usize = @intCast(shape[0]);
        const out: usize = @intCast(shape[1]);
        const kernel: usize = @intCast(shape[2]);
        const taps = kernel / stride;
        const raw = try self.temporary(conv.weight);
        defer self.gpa.free(raw);
        const w = try self.arena.alloc(f32, raw.len);
        for (0..stride) |p| for (0..out) |o| for (0..taps) |q| for (0..in) |i| {
            w[((p * out + o) * taps + q) * in + i] = raw[(i * out + o) * kernel + p + q * stride];
        };
        const bias = try self.temporary(conv.bias);
        defer self.gpa.free(bias);
        const b = try self.arena.alloc(f32, stride * out);
        for (0..stride) |p| @memcpy(b[p * out ..][0..out], bias);
        return .{ .w = w, .b = b, .in = in, .out = out, .stride = stride, .taps = taps };
    }

    /// The file stores log α and log β; raised in F32, as the authors do.
    fn snake(self: Loader, s: Snake) !SnakeBeta {
        const scale = try self.floats(s.alpha);
        for (scale) |*x| x.* = @exp(x.*);
        const inv = try self.floats(s.beta);
        for (inv) |*x| x.* = 1 / (@exp(x.*) + snake_eps);
        return .{ .scale = scale, .inv = inv };
    }
};

fn linear(io: std.Io, m: Linear, x: []const f32, rows: usize, y: []f32) !void {
    try dense.matmul(io, .{ .rows = rows, .inner = m.in, .outputs = m.out, .x = x[0 .. rows * m.in], .w = m.w, .bias = m.b }, y[0 .. rows * m.out]);
}

/// The split quantizer's lookup: of `books` (`[groups][size][dim]`), group
/// 0's vectors at `codes` into `first`, groups 1–15's summed in group order
/// into `rest` (`[codes.len][dim]` each).
pub fn sumCodebooks(books: []const f32, size: usize, dim: usize, codes: []const Frame, first: []f32, rest: []f32) void {
    for (codes, 0..) |frame, t| {
        @memcpy(first[t * dim ..][0..dim], books[@as(usize, frame[0]) * dim ..][0..dim]);
        const sum = rest[t * dim ..][0..dim];
        @memcpy(sum, books[(size + frame[1]) * dim ..][0..dim]);
        for (2..groups) |g| for (sum, books[(g * size + frame[g]) * dim ..][0..dim]) |*a, b| {
            a.* += b;
        };
    }
}

/// `y` may alias `x`; rows of `s.scale.len` channels.
pub fn snakeBeta(s: SnakeBeta, x: []const f32, y: []f32) void {
    const channels = s.scale.len;
    for (x, y, 0..) |v, *out, i| {
        const sine = @sin(v * s.scale[i % channels]);
        out.* = v + s.inv[i % channels] * (sine * sine);
    }
}

/// `x / rms(x) · w`, per row of `w.len`.
fn rmsRows(x: []const f32, w: []const f32, eps: f32, out: []f32) !void {
    const width = w.len;
    for (0..x.len / width) |r| {
        const o = out[r * width ..][0..width];
        try cpu.rmsNorm(x[r * width ..][0..width], o, eps);
        for (o, w) |*a, b| a.* *= b;
    }
}

/// A convolution's left context: its last `rows` input rows, zero at the
/// start (the authors' causal zero padding).
const History = struct {
    rows: usize,
    data: []f32,

    /// The history then `x`, in a new buffer.
    fn join(self: History, ws: Allocator, x: []const f32) ![]f32 {
        const joined = try ws.alloc(f32, self.data.len + x.len);
        @memcpy(joined[0..self.data.len], self.data);
        @memcpy(joined[self.data.len..], x);
        return joined;
    }

    /// Keeps the last rows of `joined`.
    fn keep(self: History, joined: []const f32) void {
        @memcpy(self.data, joined[joined.len - self.data.len ..]);
    }
};

/// One layer's keys (after RoPE) and values of the rows before the next
/// push: the last `window − 1`, all a later query can see.
const Cache = struct { keys: []f32, values: []f32, count: usize };
const UpState = struct { trans: History, dw: History };
const BlockState = struct { trans: History, units: [3]History };

/// A decode in progress: push frames as they come and get their samples.
/// Borrows the runtime, which must outlive it.
pub const Stream = struct {
    runtime: *const Runtime,
    gpa: Allocator,
    memory: std.heap.ArenaAllocator,
    /// Frames pushed so far: the next frame's RoPE position.
    position: usize,
    pre_conv: History,
    caches: []Cache,
    upsample: [2]UpState,
    conv_in: History,
    blocks: [4]BlockState,
    conv_out: History,

    pub fn init(gpa: Allocator, runtime: *const Runtime) !Stream {
        var self: Stream = undefined;
        self.runtime = runtime;
        self.gpa = gpa;
        self.memory = .init(gpa);
        errdefer self.memory.deinit();
        const a = self.memory.allocator();
        const c = runtime.config;
        self.pre_conv = try causalHistory(a, runtime.pre_conv);
        const caches = try a.alloc(Cache, c.layers);
        const width = c.heads * c.head_dim;
        for (caches) |*cache| cache.* = .{
            .keys = try a.alloc(f32, (c.window - 1) * width),
            .values = try a.alloc(f32, (c.window - 1) * width),
            .count = 0,
        };
        self.caches = caches;
        for (&self.upsample, runtime.upsample) |*state, stage| state.* = .{
            .trans = try transHistory(a, stage.trans),
            .dw = try history(a, dwconv_kernel - 1, c.latent),
        };
        self.conv_in = try causalHistory(a, runtime.conv_in);
        for (&self.blocks, runtime.blocks) |*state, block| {
            state.trans = try transHistory(a, block.trans);
            for (&state.units, block.units) |*h, unit| h.* = try causalHistory(a, unit.conv1);
        }
        self.conv_out = try causalHistory(a, runtime.conv_out);
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
        for (self.caches) |*cache| cache.count = 0;
        const all = [_]*History{ &self.pre_conv, &self.upsample[0].trans, &self.upsample[0].dw, &self.upsample[1].trans, &self.upsample[1].dw, &self.conv_in, &self.conv_out };
        for (all) |h| @memset(h.data, 0);
        for (&self.blocks) |*b| {
            @memset(b.trans.data, 0);
            for (&b.units) |*h| @memset(h.data, 0);
        }
    }

    pub fn push(self: *Stream, codes: []const Frame, out: []f32) !void {
        return self.pushObserved(codes, out, null);
    }

    /// Decodes the next `codes` into `out` (`codes.len · samplesPerFrame()`).
    /// A code past the codebook is `error.InvalidCode`, before any state changes.
    pub fn pushObserved(self: *Stream, codes: []const Frame, out: []f32, observer: ?Observer) !void {
        const rt = self.runtime;
        const c = rt.config;
        const n = codes.len;
        if (out.len != n * c.samplesPerFrame()) return error.InvalidShape;
        if (n == 0) return;
        for (codes) |frame| for (frame) |code| if (code >= c.codebook_size) return error.InvalidCode;
        if (self.position + n > std.math.maxInt(i32)) return error.StreamTooLong;
        var arena: std.heap.ArenaAllocator = .init(self.gpa);
        defer arena.deinit();
        const ws = arena.allocator();

        const quantized = try ws.alloc(f32, n * c.quantized);
        try rt.quantize(ws, codes, quantized);
        Observer.see(observer, .quantized, quantized);
        const pre = try ws.alloc(f32, n * c.latent);
        try causalConv(ws, rt.io, rt.pre_conv, &self.pre_conv, quantized, n, pre);
        Observer.see(observer, .pre_conv, pre);
        var x = try ws.alloc(f32, n * c.latent);
        try self.transformer(ws, pre, n, x);
        Observer.see(observer, .transformer, x);

        var rows = n;
        for (rt.upsample, &self.upsample, [2]Stage{ .upsample_0, .upsample_1 }) |stage, *state, label| {
            const up = try ws.alloc(f32, rows * stage.trans.stride * c.latent);
            try transConv(ws, rt.io, stage.trans, &state.trans, x, rows, up);
            rows *= stage.trans.stride;
            try convNeXt(ws, rt.io, stage.next, &state.dw, up, rows);
            x = up;
            Observer.see(observer, label, x);
        }
        const d0 = try ws.alloc(f32, rows * c.decoder_dim);
        try causalConv(ws, rt.io, rt.conv_in, &self.conv_in, x, rows, d0);
        x = d0;
        Observer.see(observer, .decoder_0, x);
        for (rt.blocks, &self.blocks, [4]Stage{ .decoder_1, .decoder_2, .decoder_3, .decoder_4 }) |block, *state, label| {
            const activated = try ws.alloc(f32, x.len);
            snakeBeta(block.act, x, activated);
            const y = try ws.alloc(f32, rows * block.trans.stride * block.trans.out);
            try transConv(ws, rt.io, block.trans, &state.trans, activated, rows, y);
            rows *= block.trans.stride;
            const a = try ws.alloc(f32, y.len);
            const b = try ws.alloc(f32, y.len);
            for (block.units, &state.units) |unit, *h| {
                snakeBeta(unit.act1, y, a);
                try causalConv(ws, rt.io, unit.conv1, h, a, rows, b);
                snakeBeta(unit.act2, b, b);
                try causalConv(ws, rt.io, unit.conv2, null, b, rows, a);
                for (y, a) |*r, v| r.* += v;
            }
            x = y;
            Observer.see(observer, label, x);
        }
        const activated = try ws.alloc(f32, x.len);
        snakeBeta(rt.act_out, x, activated);
        try causalConv(ws, rt.io, rt.conv_out, &self.conv_out, activated, rows, out);
        for (out) |*v| v.* = std.math.clamp(v.*, -1, 1);
        self.position += n;
    }

    /// The pre-transformer over `n` new rows of `y` ([n][latent]) into `out`.
    fn transformer(self: *Stream, ws: Allocator, y: []const f32, n: usize, out: []f32) !void {
        const rt = self.runtime;
        const c = rt.config;
        const io = rt.io;
        const width = c.heads * c.head_dim;
        const x = try ws.alloc(f32, n * c.hidden);
        try linear(io, rt.input, y, n, x);
        const h = try ws.alloc(f32, n * c.hidden);
        const q = try ws.alloc(f32, n * width);
        const k = try ws.alloc(f32, n * width);
        const v = try ws.alloc(f32, n * width);
        const attended = try ws.alloc(f32, n * width);
        const o = try ws.alloc(f32, n * c.hidden);
        const gate = try ws.alloc(f32, n * c.intermediate);
        const up = try ws.alloc(f32, n * c.intermediate);
        const keys = try ws.alloc(f32, (c.window - 1 + n) * width);
        const values = try ws.alloc(f32, (c.window - 1 + n) * width);
        const scores = try ws.alloc(f64, c.window);
        for (rt.layers, self.caches) |layer, *cache| {
            try rmsRows(x, layer.input_norm, c.rms_eps, h);
            try linear(io, layer.q, h, n, q);
            try linear(io, layer.k, h, n, k);
            try linear(io, layer.v, h, n, v);
            for (0..n) |t| for (0..c.heads) |head| {
                const options: cpu.rope.Options = .{ .dimensions = c.head_dim, .base = c.rope_theta, .position = @intCast(self.position + t) };
                const at = t * width + head * c.head_dim;
                try cpu.rope.apply(q[at..][0..c.head_dim], q[at..][0..c.head_dim], options);
                try cpu.rope.apply(k[at..][0..c.head_dim], k[at..][0..c.head_dim], options);
            };
            const past = cache.count;
            const total = past + n;
            @memcpy(keys[0 .. past * width], cache.keys[0 .. past * width]);
            @memcpy(keys[past * width ..][0 .. n * width], k);
            @memcpy(values[0 .. past * width], cache.values[0 .. past * width]);
            @memcpy(values[past * width ..][0 .. n * width], v);
            attention(c, q, keys, values, n, past, scores, attended);
            try linear(io, layer.o, attended, n, o);
            addScaled(x, layer.attn_scale, o);
            try rmsRows(x, layer.post_norm, c.rms_eps, h);
            try linear(io, layer.gate, h, n, gate);
            try linear(io, layer.up, h, n, up);
            for (gate, up) |*g, u| g.* = cpu.silu(g.*) * u;
            try linear(io, layer.down, gate, n, o);
            addScaled(x, layer.mlp_scale, o);
            const kept = @min(c.window - 1, total);
            @memcpy(cache.keys[0 .. kept * width], keys[(total - kept) * width ..][0 .. kept * width]);
            @memcpy(cache.values[0 .. kept * width], values[(total - kept) * width ..][0 .. kept * width]);
            cache.count = kept;
        }
        try rmsRows(x, rt.norm, c.rms_eps, h);
        try linear(io, rt.output, h, n, out);
    }
};

fn history(a: Allocator, rows: usize, width: usize) !History {
    return .{ .rows = rows, .data = try a.alloc(f32, rows * width) };
}
fn causalHistory(a: Allocator, conv: CausalConv) !History {
    return history(a, (conv.kernel - 1) * conv.dilation, conv.in);
}
fn transHistory(a: Allocator, conv: TransConv) !History {
    return history(a, conv.taps - 1, conv.in);
}

/// `x[i] += scale[i mod scale.len] · y[i]`: the layer scales and ConvNeXt's γ.
pub fn addScaled(x: []f32, scale: []const f32, y: []const f32) void {
    for (x, y, 0..) |*r, value, i| r.* += scale[i % scale.len] * value;
}

/// Sliding-window attention of `n` new queries: query t (row `past + t` of
/// `keys`) sees rows `past + t − window + 1 … past + t`. Scores and sums in F64.
pub fn attention(c: Config, q: []const f32, keys: []const f32, values: []const f32, n: usize, past: usize, scores: []f64, out: []f32) void {
    const width = c.heads * c.head_dim;
    const scale = 1 / @sqrt(@as(f64, @floatFromInt(c.head_dim)));
    for (0..n) |t| for (0..c.heads) |head| {
        const i = past + t;
        const lowest = (i + 1) -| c.window;
        const qi = q[t * width + head * c.head_dim ..][0..c.head_dim];
        var max: f64 = -std.math.inf(f64);
        for (lowest..i + 1, 0..) |j, s| {
            var dot: f64 = 0;
            for (qi, keys[j * width + head * c.head_dim ..][0..c.head_dim]) |a, b| dot += @as(f64, a) * b;
            scores[s] = dot * scale;
            max = @max(max, scores[s]);
        }
        const count = i + 1 - lowest;
        var total: f64 = 0;
        for (scores[0..count]) |*s| {
            s.* = @exp(s.* - max);
            total += s.*;
        }
        const o = out[t * width + head * c.head_dim ..][0..c.head_dim];
        for (o, 0..) |*value, d| {
            var sum: f64 = 0;
            for (lowest..i + 1, 0..) |j, s| sum += scores[s] * values[j * width + head * c.head_dim + d];
            value.* = @floatCast(sum / total);
        }
    };
}

/// A convolution's input rows tap-major: `out[r][j·width …]` is row
/// `first + r + base + j·step` of `joined` (`[rows][taps · width]`). A causal
/// convolution reads base 0 and its dilation as the step; a transposed one
/// its history length and step −1.
pub fn gather(joined: []const f32, width: usize, taps: usize, base: usize, step: isize, first: usize, rows: usize, out: []f32) void {
    const inner = taps * width;
    for (0..rows) |r| for (0..taps) |j| {
        const row: usize = @intCast(@as(isize, @intCast(first + r + base)) + @as(isize, @intCast(j)) * step);
        @memcpy(out[r * inner + j * width ..][0..width], joined[row * width ..][0..width]);
    };
}

/// A causal convolution of `n` new rows (`[n][in]`) into `y` (`[n][out]`)
/// through `h`'s left context, which it then advances. A kernel of 1 has
/// no context (`h` null).
fn causalConv(ws: Allocator, io: std.Io, conv: CausalConv, h: ?*History, x: []const f32, n: usize, y: []f32) !void {
    if (conv.kernel == 1) return dense.matmul(io, .{ .rows = n, .inner = conv.in, .outputs = conv.out, .x = x, .w = conv.w, .bias = conv.b }, y);
    const joined = try h.?.join(ws, x);
    const inner = conv.kernel * conv.in;
    const tile = @max(1, @min(n, gather_budget / inner));
    const gathered = try ws.alloc(f32, tile * inner);
    var first: usize = 0;
    while (first < n) {
        const rows = @min(tile, n - first);
        gather(joined, conv.in, conv.kernel, 0, @intCast(conv.dilation), first, rows, gathered);
        try dense.matmul(io, .{ .rows = rows, .inner = inner, .outputs = conv.out, .x = gathered[0 .. rows * inner], .w = conv.w, .bias = conv.b }, y[first * conv.out ..][0 .. rows * conv.out]);
        first += rows;
    }
    h.?.keep(joined);
}

/// A transposed convolution of `n` input rows into `n · stride` output
/// rows, the authors' right trim already applied.
fn transConv(ws: Allocator, io: std.Io, conv: TransConv, h: *History, x: []const f32, n: usize, y: []f32) !void {
    const outputs = conv.stride * conv.out;
    if (conv.taps == 1) return dense.matmul(io, .{ .rows = n, .inner = conv.in, .outputs = outputs, .x = x, .w = conv.w, .bias = conv.b }, y);
    const joined = try h.join(ws, x);
    const inner = conv.taps * conv.in;
    const tile = @max(1, @min(n, gather_budget / inner));
    const gathered = try ws.alloc(f32, tile * inner);
    var first: usize = 0;
    while (first < n) {
        const rows = @min(tile, n - first);
        gather(joined, conv.in, conv.taps, h.rows, -1, first, rows, gathered);
        try dense.matmul(io, .{ .rows = rows, .inner = inner, .outputs = outputs, .x = gathered[0 .. rows * inner], .w = conv.w, .bias = conv.b }, y[first * outputs ..][0 .. rows * outputs]);
        first += rows;
    }
    h.keep(joined);
}

/// The ConvNeXt block in place over `x` (`[rows][latent]`):
/// x + γ ⊙ pw2(gelu(pw1(LayerNorm(dwconv(x))))).
fn convNeXt(ws: Allocator, io: std.Io, next: NeXt, h: *History, x: []f32, rows: usize) !void {
    const channels = next.gamma.len;
    const joined = try h.join(ws, x);
    const d = try ws.alloc(f32, rows * channels);
    depthwise(next.dw, next.dw_b, joined, rows, d);
    h.keep(joined);
    for (0..rows) |t| {
        const row = d[t * channels ..][0..channels];
        try dense.layerNorm(row, next.ln_w, next.ln_b, layer_norm_eps, row);
    }
    const wide = try ws.alloc(f32, rows * next.pw1.out);
    try linear(io, next.pw1, d, rows, wide);
    for (wide) |*v| v.* = cpu.geluErf(v.*);
    try linear(io, next.pw2, wide, rows, d);
    addScaled(x, next.gamma, d);
}

/// ConvNeXt's causal depthwise convolution: `out[t][c] = Σ_j weight[c][j] ·
/// joined[t + j][c] + bias[c]` over `rows` rows of `bias.len` channels;
/// `joined` holds the kernel's history rows, then the input.
pub fn depthwise(weight: []const f32, bias: []const f32, joined: []const f32, rows: usize, out: []f32) void {
    const channels = bias.len;
    const taps = weight.len / channels;
    for (0..rows) |t| for (0..channels) |ch| {
        var acc: f32 = 0;
        for (0..taps) |j| acc += weight[ch * taps + j] * joined[(t + j) * channels + ch];
        out[t * channels + ch] = acc + bias[ch];
    };
}

// ---- tests ------------------------------------------------------------------

const tiny: Config = .{
    .codebook_size = 8,
    .codebook_dim = 4,
    .quantized = 6,
    .latent = 8,
    .hidden = 8,
    .intermediate = 12,
    .layers = 2,
    .heads = 2,
    .head_dim = 4,
    .window = 5,
    .decoder_dim = 16,
    .upsample_rates = .{ 2, 3, 2, 2 },
};

/// The names and shapes `bind` asks for under `config`; the caller frees
/// them with `freeSpecs`.
fn specs(gpa: Allocator, config: Config) !std.ArrayList(Spec) {
    var list: std.ArrayList(Spec) = .empty;
    errdefer freeSpecs(gpa, &list);
    var binder: Binder = .{ .gpa = gpa, .checkpoint = null, .record = &list, .diagnostic = null };
    defer binder.deinit();
    var binding = try bindWith(&binder, config);
    binding.deinit(gpa);
    return list;
}

fn freeSpecs(gpa: Allocator, list: *std.ArrayList(Spec)) void {
    for (list.items) |s| {
        gpa.free(s.name);
        gpa.free(s.shape);
    }
    list.deinit(gpa);
}

const Change = struct { skip: ?[]const u8 = null, reshape: ?Spec = null, extra: ?Spec = null };

/// Writes `model.safetensors` with the tiny decoder's tensors (values in
/// [−0.5, 0.5), usages in [0.5, 1.5)), one encoder tensor, and `change`.
fn writeTiny(gpa: Allocator, io: std.Io, dir: std.Io.Dir, change: Change) !void {
    var list = try specs(gpa, tiny);
    defer freeSpecs(gpa, &list);
    var extras: [2]Spec = .{ .{ .name = "encoder.quantizer.weight", .shape = &.{ 2, 3 } }, change.extra orelse .{ .name = "", .shape = &.{} } };
    var header: std.Io.Writer.Allocating = .init(gpa);
    defer header.deinit();
    var data: std.ArrayList(u8) = .empty;
    defer data.deinit(gpa);
    var prng = std.Random.DefaultPrng.init(11);
    try header.writer.writeByte('{');
    var first = true;
    for ([2][]Spec{ list.items, &extras }) |part| for (part) |s| {
        if (s.name.len == 0) continue;
        if (change.skip) |name| if (std.mem.eql(u8, s.name, name)) continue;
        var shape = s.shape;
        if (change.reshape) |r| if (std.mem.eql(u8, s.name, r.name)) {
            shape = r.shape;
        };
        var elements: u64 = 1;
        for (shape) |x| elements *= x;
        const start = data.items.len;
        const usage = std.mem.endsWith(u8, s.name, "cluster_usage");
        for (0..elements) |_| {
            const value: f32 = prng.random().float(f32) + if (usage) @as(f32, 0.5) else -0.5;
            try data.appendSlice(gpa, std.mem.asBytes(&value));
        }
        if (!first) try header.writer.writeByte(',');
        first = false;
        try header.writer.print("\"{s}\":{{\"dtype\":\"F32\",\"shape\":[", .{s.name});
        for (shape, 0..) |x, k| try header.writer.print("{s}{d}", .{ if (k == 0) "" else ",", x });
        try header.writer.print("],\"data_offsets\":[{d},{d}]}}", .{ start, data.items.len });
    };
    try header.writer.writeByte('}');
    const file = try dir.createFile(io, "model.safetensors", .{});
    defer file.close(io);
    var length: [8]u8 = undefined;
    std.mem.writeInt(u64, &length, header.written().len, .little);
    try file.writeStreamingAll(io, &length);
    try file.writeStreamingAll(io, header.written());
    try file.writeStreamingAll(io, data.items);
}

const TinyFile = struct {
    tmp: std.testing.TmpDir,
    checkpoint: safetensors.Checkpoint,

    fn open(io: std.Io, change: Change) !TinyFile {
        const gpa = std.testing.allocator;
        var tmp = std.testing.tmpDir(.{});
        errdefer tmp.cleanup();
        try writeTiny(gpa, io, tmp.dir, change);
        const path = try tmp.dir.realPathFileAlloc(io, ".", gpa);
        defer gpa.free(path);
        return .{ .tmp = tmp, .checkpoint = try safetensors.Checkpoint.open(gpa, io, path, .{}) };
    }

    fn close(self: *TinyFile, io: std.Io) void {
        self.checkpoint.deinit(io);
        self.tmp.cleanup();
    }
};

fn tinyCodes(frames: []Frame) void {
    var prng = std.Random.DefaultPrng.init(5);
    for (frames) |*frame| for (frame) |*code| {
        code.* = prng.random().uintLessThan(u16, @intCast(tiny.codebook_size));
    };
}

test "the published decoder has 271 tensors and 1,920 samples a frame" {
    var list = try specs(std.testing.allocator, .{});
    defer freeSpecs(std.testing.allocator, &list);
    try std.testing.expectEqual(@as(usize, 271), list.items.len);
    try std.testing.expectEqual(@as(usize, 1920), (Config{}).samplesPerFrame());
}

test "streaming in pieces of 1, 3, and 7 frames gives one decode's samples bit for bit" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var file = try TinyFile.open(io, .{});
    defer file.close(io);
    var binding = try bind(gpa, &file.checkpoint, tiny, null);
    defer binding.deinit(gpa);
    var runtime = try Runtime.init(gpa, io, &binding);
    defer runtime.deinit();
    const per = 96;
    try std.testing.expectEqual(@as(usize, per), tiny.samplesPerFrame());
    var codes: [13]Frame = undefined;
    tinyCodes(&codes);
    var whole: [codes.len * per]f32 = undefined;
    try runtime.decode(gpa, &codes, &whole);
    var nonzero: usize = 0;
    for (whole) |v| {
        try std.testing.expect(v >= -1 and v <= 1);
        if (v != 0) nonzero += 1;
    }
    try std.testing.expect(nonzero > whole.len / 2);

    var stream = try Stream.init(gpa, &runtime);
    defer stream.deinit();
    for ([_]usize{ 1, 3, 7 }) |piece| {
        stream.reset();
        var pieces: [whole.len]f32 = undefined;
        var at: usize = 0;
        while (at < codes.len) : (at += piece) {
            const n = @min(piece, codes.len - at);
            try stream.push(codes[at..][0..n], pieces[at * per ..][0 .. n * per]);
        }
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(&whole), std.mem.sliceAsBytes(&pieces));
    }
    // A prefix decodes as the whole's start: the decoder is causal.
    var prefix: [5 * per]f32 = undefined;
    try runtime.decode(gpa, codes[0..5], &prefix);
    try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(whole[0..prefix.len]), std.mem.sliceAsBytes(&prefix));

    var bad = codes[0..1].*;
    bad[0][3] = @intCast(tiny.codebook_size);
    try std.testing.expectError(error.InvalidCode, stream.push(&bad, prefix[0..per]));
    try std.testing.expectError(error.InvalidShape, stream.push(codes[0..2], prefix[0..per]));
}

test "bind refuses a missing, misshapen, or unknown tensor by name and skips the encoder's" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    {
        var file = try TinyFile.open(io, .{});
        defer file.close(io);
        var binding = try bind(gpa, &file.checkpoint, tiny, null);
        defer binding.deinit(gpa);
        try std.testing.expectEqual(file.checkpoint.tensorCount() - 1, binding.tensors);
    }
    const Case = struct { change: Change, err: Error, name: []const u8 };
    const cases = [_]Case{
        .{ .change = .{ .reshape = .{ .name = "decoder.pre_conv.conv.weight", .shape = &.{ 8, 6, 2 } } }, .err = error.TensorShapeMismatch, .name = "decoder.pre_conv.conv.weight" },
        .{ .change = .{ .reshape = .{ .name = "decoder.decoder.2.block.1.conv.weight", .shape = &.{ 4, 8, 6 } } }, .err = error.TensorShapeMismatch, .name = "decoder.decoder.2.block.1.conv.weight" },
        .{ .change = .{ .skip = "decoder.decoder.6.conv.bias" }, .err = error.MissingTensor, .name = "decoder.decoder.6.conv.bias" },
        .{ .change = .{ .extra = .{ .name = "decoder.pre_transformer.layers.2.mlp.up_proj.weight", .shape = &.{ 12, 8 } } }, .err = error.UnexpectedTensor, .name = "decoder.pre_transformer.layers.2.mlp.up_proj.weight" },
        .{ .change = .{ .extra = .{ .name = "speaker_encoder.weight", .shape = &.{4} } }, .err = error.UnexpectedTensor, .name = "speaker_encoder.weight" },
    };
    for (cases) |case| {
        var file = try TinyFile.open(io, case.change);
        defer file.close(io);
        var diagnostic: Diagnostic = .{};
        try std.testing.expectError(case.err, bind(gpa, &file.checkpoint, tiny, &diagnostic));
        try std.testing.expectEqualStrings(case.name, diagnostic.name());
    }
}

/// An F32 tensor over a few values, for the convolution tests.
const TestTensor = struct {
    tensor: safetensors.Tensor = undefined,
    bytes: [64]u8 = undefined,

    fn ref(self: *TestTensor, shape: []const u64, values: []const f32) Ref {
        const bytes = std.mem.sliceAsBytes(values);
        @memcpy(self.bytes[0..bytes.len], bytes);
        self.tensor = .{ .name = "t", .dtype = .f32, .shape = shape, .offset = 0, .bytes = bytes.len, .elements = values.len };
        return .{ .tensor = &self.tensor, .bytes = self.bytes[0..bytes.len] };
    }
};

test "the convolutions are the authors': causal with dilation, transposed with the right trim" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const l: Loader = .{ .arena = a, .gpa = gpa };
    var w: TestTensor = .{};
    var b: TestTensor = .{};
    // Kernel [1, 2, 3], dilation 2: an impulse reaches t, t + 2, t + 4, across pushes.
    const causal = try l.causal(.{ .weight = w.ref(&.{ 1, 1, 3 }, &.{ 1, 2, 3 }), .bias = b.ref(&.{1}, &.{0}) }, 2);
    var h = try causalHistory(a, causal);
    @memset(h.data, 0);
    var y: [6]f32 = undefined;
    try causalConv(a, io, causal, &h, &.{ 1, 0, 0 }, 3, y[0..3]);
    try causalConv(a, io, causal, &h, &.{ 0, 0, 0 }, 3, y[3..6]);
    try std.testing.expectEqualSlices(f32, &.{ 3, 0, 2, 0, 1, 0 }, &y);
    // Kernel 4, stride 2, bias 0.5: an impulse gives the kernel; the tail is trimmed.
    const trans = try l.transposed(.{ .weight = w.ref(&.{ 1, 1, 4 }, &.{ 1, 2, 3, 4 }), .bias = b.ref(&.{1}, &.{0.5}) }, 2);
    var th = try transHistory(a, trans);
    @memset(th.data, 0);
    var z: [6]f32 = undefined;
    try transConv(a, io, trans, &th, &.{ 1, 0, 0 }, 3, &z);
    try std.testing.expectEqualSlices(f32, &.{ 1.5, 2.5, 3.5, 4.5, 0.5, 0.5 }, &z);
}

test "bind, init, and a push release everything on allocation failure" {
    const io = std.testing.io;
    var file = try TinyFile.open(io, .{});
    defer file.close(io);
    try alloc_check.checkAll(std.testing.allocator, struct {
        fn run(gpa: Allocator, checkpoint: *const safetensors.Checkpoint) !void {
            var binding = try bind(gpa, checkpoint, tiny, null);
            defer binding.deinit(gpa);
            var runtime = try Runtime.init(gpa, std.testing.io, &binding);
            defer runtime.deinit();
            var stream = try Stream.init(gpa, &runtime);
            defer stream.deinit();
            var codes: [2]Frame = undefined;
            tinyCodes(&codes);
            var out: [2 * 96]f32 = undefined;
            try stream.push(&codes, &out);
        }
    }.run, .{&file.checkpoint});
}
