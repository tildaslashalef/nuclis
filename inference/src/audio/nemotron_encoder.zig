//! Nemotron 3.5 ASR's cache-aware FastConformer encoder and its language
//! prompt block, the CPU reference (docs/models/nemotron-asr.md § The
//! encoder, § Cache-aware streaming): log-mel frames through three causal
//! stride-2 convolutions (8× fewer rows, one per 80 ms), 24 conformer blocks
//! whose attention sees its own chunk of `r + 1` rows and `left` rows
//! before it, then the prompt block. `bind` validates the safetensors
//! checkpoint and views its F32 tensors; `Encoder` holds what every stream
//! shares; `Stream` turns samples or frames into rows chunk by chunk, and
//! `finish` ends as a whole-clip pass ends, so a whole clip is one stream
//! fed at once. Written from transformers' `Nemotron3_5AsrForRNNT`.
const std = @import("std");
const builtin = @import("builtin");
const safetensors = @import("../formats/safetensors.zig");
const cpu = @import("../backends/cpu/root.zig");
const mel = @import("mel.zig");

const dense = cpu.dense;

/// The lookaheads the checkpoint was trained with (`att_context_size`).
pub const lookaheads = [_]usize{ 0, 3, 6, 13 };
pub const max_lookahead = 13;
/// Every LayerNorm's ε (`nn.LayerNorm`'s default).
pub const epsilon: f32 = 1e-5;
/// The subsampling convolutions: 3×3, stride 2 on both axes.
const taps = 3;

pub const Config = struct {
    hidden: usize,
    heads: usize,
    ffn: usize,
    blocks: usize,
    /// Subsampling channels.
    channels: usize,
    mels: usize,
    /// The depthwise convolution's taps.
    kernel: usize,
    /// Rows before its chunk a query sees (`sliding_window − 1`).
    left: usize,
    prompts: usize,
    prompt_hidden: usize,

    pub const nemotron: Config = .{ .hidden = 1024, .heads = 8, .ffn = 4096, .blocks = 24, .channels = 256, .mels = 128, .kernel = 9, .left = 56, .prompts = 128, .prompt_hidden = 2048 };

    pub fn headDim(self: Config) usize {
        return self.hidden / self.heads;
    }

    /// Frequency columns into subsampling stage `i` (3 is its output).
    pub fn columns(self: Config, i: usize) usize {
        var c = self.mels;
        for (0..i) |_| c = halve(c);
        return c;
    }

    /// The flattened subsampling row, `channels · columns(3)`.
    pub fn subWidth(self: Config) usize {
        return self.channels * self.columns(3);
    }

    /// Relative-position rows: distances `−max_lookahead … left + max_lookahead`.
    pub fn positions(self: Config) usize {
        return self.left + 2 * max_lookahead + 1;
    }
};

/// A stride-2 causal layer's output count: `⌊n / 2⌋ + 1`, two zeros before
/// and one after.
fn halve(n: usize) usize {
    return n / 2 + 1;
}

/// Encoder rows for `frames` mel frames.
pub fn rowsFor(frames: usize) usize {
    if (frames == 0) return 0;
    return halve(halve(halve(frames)));
}

pub const ConfigError = error{ InvalidConfig, UnsupportedConfig, OutOfMemory };

/// Reads `config.json` (`Nemotron3_5AsrForRNNT`): the encoder's widths and
/// the prompt block's, refusing what this encoder does not implement.
pub fn parseConfig(gpa: std.mem.Allocator, text: []const u8) ConfigError!Config {
    const EncoderJson = struct {
        hidden_size: u32,
        intermediate_size: u32,
        num_hidden_layers: u32,
        num_attention_heads: u32,
        num_key_value_heads: u32,
        hidden_act: []const u8,
        attention_bias: bool,
        convolution_bias: bool,
        conv_kernel_size: u32,
        subsampling_factor: u32,
        subsampling_conv_channels: u32,
        subsampling_conv_kernel_size: u32,
        subsampling_conv_stride: u32,
        num_mel_bins: u32,
        scale_input: bool,
        sliding_window: u32,
        supported_num_lookahead_tokens: []const u32,
    };
    const Json = struct { encoder_config: EncoderJson, num_prompts: u32, prompt_intermediate_size: u32 };
    const parsed = std.json.parseFromSlice(Json, gpa, text, .{ .ignore_unknown_fields = true }) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidConfig,
    };
    defer parsed.deinit();
    const j = parsed.value;
    const e = j.encoder_config;
    if (!std.mem.eql(u8, e.hidden_act, "silu") or e.attention_bias or e.convolution_bias or e.scale_input or
        e.subsampling_factor != 8 or e.subsampling_conv_kernel_size != taps or e.subsampling_conv_stride != 2 or
        e.num_key_value_heads != e.num_attention_heads) return error.UnsupportedConfig;
    for (e.supported_num_lookahead_tokens) |r| if (std.mem.indexOfScalar(usize, &lookaheads, r) == null) return error.UnsupportedConfig;
    if (e.num_attention_heads == 0 or e.hidden_size % e.num_attention_heads != 0 or e.hidden_size > 16384 or
        e.intermediate_size == 0 or e.intermediate_size > 65536 or e.num_hidden_layers == 0 or e.num_hidden_layers > 256 or
        e.subsampling_conv_channels == 0 or e.subsampling_conv_channels > 4096 or e.num_mel_bins == 0 or e.num_mel_bins > 1024 or
        e.conv_kernel_size < 2 or e.conv_kernel_size > 64 or e.sliding_window == 0 or e.sliding_window > 4096 or
        j.num_prompts == 0 or j.num_prompts > 4096 or j.prompt_intermediate_size == 0 or j.prompt_intermediate_size > 65536) return error.InvalidConfig;
    return .{
        .hidden = e.hidden_size,
        .heads = e.num_attention_heads,
        .ffn = e.intermediate_size,
        .blocks = e.num_hidden_layers,
        .channels = e.subsampling_conv_channels,
        .mels = e.num_mel_bins,
        .kernel = e.conv_kernel_size,
        .left = e.sliding_window - 1,
        .prompts = j.num_prompts,
        .prompt_hidden = j.prompt_intermediate_size,
    };
}

pub const Linear = struct {
    /// `[outputs][inputs]`, PyTorch's layout.
    weight: []const f32,
    bias: ?[]const f32 = null,
};

pub const Norm = struct { weight: []const f32, bias: []const f32 };

pub const Block = struct {
    norm_ff1: Norm,
    ff1_in: []const f32,
    ff1_out: []const f32,
    norm_attention: Norm,
    q: []const f32,
    k: []const f32,
    v: []const f32,
    o: []const f32,
    /// `relative_k_proj`, applied to the positional rows.
    relative: []const f32,
    /// `[heads][head_dim]`: added to the query against keys (u) and
    /// against positions (v).
    bias_u: []const f32,
    bias_v: []const f32,
    norm_conv: Norm,
    /// `[2 · hidden][hidden]`: the GLU's values, then its gates.
    pointwise1: []const f32,
    /// `[hidden][kernel]`.
    depthwise: []const f32,
    conv_norm: Norm,
    pointwise2: []const f32,
    norm_ff2: Norm,
    ff2_in: []const f32,
    ff2_out: []const f32,
    norm_out: Norm,
};

/// Every tensor the encoder and the prompt block read, as F32 views.
pub const Weights = struct {
    config: Config,
    /// `[channels][1][3][3]` and its bias.
    conv_in: Linear,
    /// `[channels][1][3][3]` each.
    depthwise: [2]Linear,
    /// `[channels][channels]` each.
    pointwise: [2]Linear,
    /// `subWidth → hidden`, reading the channel-major flatten.
    linear: Linear,
    blocks: []const Block,
    /// `hidden + prompts → prompt_hidden → hidden`.
    prompt_in: Linear,
    prompt_out: Linear,
};

/// The bound checkpoint: views into its mapping, or copies where a tensor
/// is unaligned. Owns `blocks` and the copies; borrows the checkpoint, which
/// must outlive it.
pub const Binding = struct {
    arena: std.heap.ArenaAllocator,
    weights: Weights,

    pub fn deinit(self: *Binding) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

pub const BindError = error{ MissingTensor, TensorShapeMismatch, UnsupportedDtype, OutOfMemory, TensorOutOfBounds };

/// Binds the encoder's and the prompt block's tensors of `checkpoint`
/// (`encoder.*`, `prompt_projector.*`) at `config`'s shapes, F32 only: the
/// file's weights run as published. The transducer's tensors are left to it.
pub fn bind(gpa: std.mem.Allocator, checkpoint: *const safetensors.Checkpoint, config: Config) BindError!Binding {
    comptime std.debug.assert(builtin.cpu.arch.endian() == .little);
    var binding: Binding = .{ .arena = .init(gpa), .weights = undefined };
    errdefer binding.arena.deinit();
    const arena = binding.arena.allocator();
    var b: Binder = .{ .checkpoint = checkpoint, .arena = arena };
    const c = config;
    const h = c.hidden;
    var w: Weights = undefined;
    w.config = c;
    w.conv_in = .{ .weight = try b.get("encoder.subsampling.conv_in.weight", .{}, &.{ c.channels, 1, taps, taps }), .bias = try b.get("encoder.subsampling.conv_in.bias", .{}, &.{c.channels}) };
    for (0..2) |i| {
        w.depthwise[i] = .{ .weight = try b.get("encoder.subsampling.layers.{d}.depthwise_conv.weight", .{i}, &.{ c.channels, 1, taps, taps }), .bias = try b.get("encoder.subsampling.layers.{d}.depthwise_conv.bias", .{i}, &.{c.channels}) };
        w.pointwise[i] = .{ .weight = try b.get("encoder.subsampling.layers.{d}.pointwise_conv.weight", .{i}, &.{ c.channels, c.channels, 1, 1 }), .bias = try b.get("encoder.subsampling.layers.{d}.pointwise_conv.bias", .{i}, &.{c.channels}) };
    }
    w.linear = .{ .weight = try b.get("encoder.subsampling.linear.weight", .{}, &.{ h, c.subWidth() }), .bias = try b.get("encoder.subsampling.linear.bias", .{}, &.{h}) };
    const blocks = try arena.alloc(Block, c.blocks);
    for (blocks, 0..) |*blk, i| blk.* = .{
        .norm_ff1 = try b.norm("norm_feed_forward1", i, h),
        .ff1_in = try b.get("encoder.layers.{d}.feed_forward1.linear1.weight", .{i}, &.{ c.ffn, h }),
        .ff1_out = try b.get("encoder.layers.{d}.feed_forward1.linear2.weight", .{i}, &.{ h, c.ffn }),
        .norm_attention = try b.norm("norm_self_att", i, h),
        .q = try b.get("encoder.layers.{d}.self_attn.q_proj.weight", .{i}, &.{ h, h }),
        .k = try b.get("encoder.layers.{d}.self_attn.k_proj.weight", .{i}, &.{ h, h }),
        .v = try b.get("encoder.layers.{d}.self_attn.v_proj.weight", .{i}, &.{ h, h }),
        .o = try b.get("encoder.layers.{d}.self_attn.o_proj.weight", .{i}, &.{ h, h }),
        .relative = try b.get("encoder.layers.{d}.self_attn.relative_k_proj.weight", .{i}, &.{ h, h }),
        .bias_u = try b.get("encoder.layers.{d}.self_attn.bias_u", .{i}, &.{ c.heads, c.headDim() }),
        .bias_v = try b.get("encoder.layers.{d}.self_attn.bias_v", .{i}, &.{ c.heads, c.headDim() }),
        .norm_conv = try b.norm("norm_conv", i, h),
        .pointwise1 = try b.get("encoder.layers.{d}.conv.pointwise_conv1.weight", .{i}, &.{ 2 * h, h, 1 }),
        .depthwise = try b.get("encoder.layers.{d}.conv.depthwise_conv.weight", .{i}, &.{ h, 1, c.kernel }),
        .conv_norm = try b.norm("conv.norm", i, h),
        .pointwise2 = try b.get("encoder.layers.{d}.conv.pointwise_conv2.weight", .{i}, &.{ h, h, 1 }),
        .norm_ff2 = try b.norm("norm_feed_forward2", i, h),
        .ff2_in = try b.get("encoder.layers.{d}.feed_forward2.linear1.weight", .{i}, &.{ c.ffn, h }),
        .ff2_out = try b.get("encoder.layers.{d}.feed_forward2.linear2.weight", .{i}, &.{ h, c.ffn }),
        .norm_out = try b.norm("norm_out", i, h),
    };
    w.blocks = blocks;
    w.prompt_in = .{ .weight = try b.get("prompt_projector.linear_1.weight", .{}, &.{ c.prompt_hidden, h + c.prompts }), .bias = try b.get("prompt_projector.linear_1.bias", .{}, &.{c.prompt_hidden}) };
    w.prompt_out = .{ .weight = try b.get("prompt_projector.linear_2.weight", .{}, &.{ h, c.prompt_hidden }), .bias = try b.get("prompt_projector.linear_2.bias", .{}, &.{h}) };
    binding.weights = w;
    return binding;
}

const Binder = struct {
    checkpoint: *const safetensors.Checkpoint,
    arena: std.mem.Allocator,

    fn get(self: *Binder, comptime format: []const u8, args: anytype, shape: []const usize) BindError![]const f32 {
        var buffer: [128]u8 = undefined;
        const name = std.fmt.bufPrint(&buffer, format, args) catch return error.MissingTensor;
        const ref = self.checkpoint.get(name) orelse return error.MissingTensor;
        if (ref.tensor.shape.len != shape.len) return error.TensorShapeMismatch;
        for (ref.tensor.shape, shape) |a, want| if (a != want) return error.TensorShapeMismatch;
        if (ref.tensor.dtype != .f32) return error.UnsupportedDtype;
        const count: usize = @intCast(ref.tensor.elements);
        if (@intFromPtr(ref.bytes.ptr) % @alignOf(f32) == 0)
            return @as([*]const f32, @ptrCast(@alignCast(ref.bytes.ptr)))[0..count];
        const copy = try self.arena.alloc(f32, count);
        ref.decode(0, copy) catch |e| return switch (e) {
            error.UnsupportedDtype => error.UnsupportedDtype,
            error.TensorOutOfBounds => error.TensorOutOfBounds,
        };
        return copy;
    }

    fn norm(self: *Binder, comptime name: []const u8, block: usize, width: usize) BindError!Norm {
        return .{
            .weight = try self.get("encoder.layers.{d}." ++ name ++ ".weight", .{block}, &.{width}),
            .bias = try self.get("encoder.layers.{d}." ++ name ++ ".bias", .{block}, &.{width}),
        };
    }
};

/// What an observer is shown: each batch of rows as a stage finishes them,
/// the first at encoder row `first`.
pub const Stage = union(enum) {
    /// After the subsampling's linear.
    subsampling,
    /// After block i's `norm_out`.
    block: usize,
    /// After the prompt block: the encoder's output.
    prompt,
};

pub const Observer = struct {
    context: *anyopaque,
    record: *const fn (context: *anyopaque, stage: Stage, first: usize, rows: []const f32) void,

    fn see(self: ?Observer, stage: Stage, first: usize, rows: []const f32) void {
        if (self) |o| o.record(o.context, stage, first, rows);
    }
};

/// What every stream of one checkpoint shares: the weights, the relative
/// keys, and the front end's bank.
pub const Encoder = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    weights: *const Weights,
    /// `blocks × positions × hidden`: `relative_k_proj · pe(d)`, row
    /// `d + max_lookahead`.
    relk: []f32,
    /// Built when the encoder reads 128 mel bins; null otherwise (frames only).
    bank: ?*mel.Bank,

    pub fn init(alloc: std.mem.Allocator, io: std.Io, weights: *const Weights) !Encoder {
        const c = weights.config;
        const relk = try alloc.alloc(f32, c.blocks * c.positions() * c.hidden);
        errdefer alloc.free(relk);
        const table = try alloc.alloc(f32, c.positions() * c.hidden);
        defer alloc.free(table);
        positionRows(c, table);
        for (weights.blocks, 0..) |blk, i| {
            try dense.matmul(io, .{ .rows = c.positions(), .inner = c.hidden, .outputs = c.hidden, .x = table, .w = blk.relative }, relk[i * c.positions() * c.hidden ..][0 .. c.positions() * c.hidden]);
        }
        var bank: ?*mel.Bank = null;
        if (c.mels == mel.filters) {
            bank = try alloc.create(mel.Bank);
            bank.?.init(.nemotron);
        }
        return .{ .alloc = alloc, .io = io, .weights = weights, .relk = relk, .bank = bank };
    }

    pub fn deinit(self: *Encoder) void {
        if (self.bank) |b| self.alloc.destroy(b);
        self.alloc.free(self.relk);
        self.* = undefined;
    }

    /// The whole-clip pass over `features` (`frames × mels`): one stream fed
    /// at once. Returns `rowsFor(frames) × hidden` rows, owned by `alloc`.
    pub fn encode(self: *const Encoder, alloc: std.mem.Allocator, features: []const f32, lookahead: usize, prompt: usize, observer: ?Observer) ![]f32 {
        var stream = try Stream.init(alloc, self, lookahead, prompt);
        defer stream.deinit();
        stream.observer = observer;
        var out: std.ArrayList(f32) = .empty;
        errdefer out.deinit(alloc);
        try stream.pushFeatures(features, &out);
        try stream.finish(&out);
        return out.toOwnedSlice(alloc);
    }
};

/// `pe(d)` for d = −max_lookahead … left + max_lookahead: sine and cosine
/// interleaved, `ω_m = 10000^(−2m / hidden)`, in F32 from a float position
/// (`NemotronAsrStreamingEncoderRelPositionalEncoding`).
pub fn positionRows(config: Config, out: []f32) void {
    const h = config.hidden;
    for (0..config.positions()) |row| {
        const d: f32 = @floatFromInt(@as(i64, @intCast(row)) - max_lookahead);
        for (0..h / 2) |m| {
            const exponent = @as(f32, @floatFromInt(2 * m)) / @as(f32, @floatFromInt(h));
            const inv = 1.0 / std.math.pow(f32, 10000, exponent);
            const t = d * inv;
            out[row * h + 2 * m] = @sin(t);
            out[row * h + 2 * m + 1] = @cos(t);
        }
    }
}

/// One subsampling convolution's pending input: rows `first …` of
/// `width · channels` values each, `[column][channel]`.
const Queue = struct {
    rows: std.ArrayList(f32) = .empty,
    first: usize = 0,
    /// Rows received, the right pad row excluded.
    count: usize = 0,
    /// The next output row.
    next: usize = 0,
};

/// One stream's state (nemotron-asr.md § Cache-aware streaming). Output
/// rows are appended to the caller's list with the stream's allocator.
pub const Stream = struct {
    alloc: std.mem.Allocator,
    encoder: *const Encoder,
    lookahead: usize,
    /// The language prompt; may change between pushes.
    prompt: usize,
    observer: ?Observer = null,
    // The front end: samples from `origin`, the one before it, and the next frame.
    samples: std.ArrayList(f32) = .empty,
    origin: usize = 0,
    previous: f32 = 0,
    pushed: usize = 0,
    frame: usize = 0,
    // The subsampling's three convolutions.
    queues: [3]Queue = .{ .{}, .{}, .{} },
    /// Encoder rows from the linear awaiting a whole chunk, from row `row`.
    pending: std.ArrayList(f32) = .empty,
    row: usize = 0,
    /// Per block: the keys and values of the last `left` rows, and the last
    /// `kernel − 1` GLU rows.
    keys: []std.ArrayList(f32),
    values: []std.ArrayList(f32),
    history: []f32,
    finished: bool = false,

    pub fn init(alloc: std.mem.Allocator, encoder: *const Encoder, lookahead: usize, prompt: usize) !Stream {
        const c = encoder.weights.config;
        if (std.mem.indexOfScalar(usize, &lookaheads, lookahead) == null) return error.UnsupportedLookahead;
        if (prompt >= c.prompts) return error.UnsupportedPrompt;
        const keys = try alloc.alloc(std.ArrayList(f32), c.blocks);
        errdefer alloc.free(keys);
        const values = try alloc.alloc(std.ArrayList(f32), c.blocks);
        errdefer alloc.free(values);
        @memset(keys, .empty);
        @memset(values, .empty);
        const history = try alloc.alloc(f32, c.blocks * (c.kernel - 1) * c.hidden);
        @memset(history, 0);
        return .{ .alloc = alloc, .encoder = encoder, .lookahead = lookahead, .prompt = prompt, .keys = keys, .values = values, .history = history };
    }

    pub fn deinit(self: *Stream) void {
        for (self.keys, self.values) |*k, *v| {
            k.deinit(self.alloc);
            v.deinit(self.alloc);
        }
        self.alloc.free(self.keys);
        self.alloc.free(self.values);
        self.alloc.free(self.history);
        for (&self.queues) |*q| q.rows.deinit(self.alloc);
        self.pending.deinit(self.alloc);
        self.samples.deinit(self.alloc);
        self.* = undefined;
    }

    fn config(self: *const Stream) Config {
        return self.encoder.weights.config;
    }

    /// Mel frames per batch through the pipeline: bounds the working memory
    /// whatever a push holds.
    const batch_frames = 512;

    /// Appends 16 kHz samples; the rows whose chunk they complete go to `out`.
    pub fn push(self: *Stream, pcm: []const f32, out: *std.ArrayList(f32)) !void {
        if (self.finished) return error.StreamFinished;
        const bank = self.encoder.bank orelse return error.NoFrontEnd;
        try self.samples.appendSlice(self.alloc, pcm);
        self.pushed += pcm.len;
        const spec = bank.spec;
        var ready = self.frame;
        while (spec.frameEnd(ready) <= self.pushed) ready += 1;
        try self.frames(bank, ready, out);
    }

    /// Computes frames up to `end` (exclusive) from the held samples and
    /// feeds them on, then drops the samples no later frame reads.
    fn frames(self: *Stream, bank: *const mel.Bank, end: usize, out: *std.ArrayList(f32)) !void {
        const features = try self.alloc.alloc(f32, batch_frames * mel.filters);
        defer self.alloc.free(features);
        while (self.frame < end) {
            const n: usize = @min(end - self.frame, batch_frames);
            const span: mel.Span = .{ .values = self.samples.items, .origin = self.origin, .previous = self.previous };
            bank.frames(span, self.frame, features[0 .. n * mel.filters]);
            self.frame += n;
            try self.pushFeatures(features[0 .. n * mel.filters], out);
        }
        const keep = bank.spec.frameStart(self.frame);
        if (keep > self.origin) {
            const drop = @min(keep - self.origin, self.samples.items.len);
            if (drop > 0) {
                self.previous = self.samples.items[drop - 1];
                std.mem.copyForwards(f32, self.samples.items, self.samples.items[drop..]);
                self.samples.shrinkRetainingCapacity(self.samples.items.len - drop);
                self.origin += drop;
            }
        }
    }

    /// Appends mel frames (`n × mels`); the rows whose chunk they complete
    /// go to `out`. Not to be mixed with `push` on one stream.
    pub fn pushFeatures(self: *Stream, features: []const f32, out: *std.ArrayList(f32)) !void {
        if (self.finished) return error.StreamFinished;
        const mels = self.config().mels;
        if (features.len % mels != 0) return error.InvalidShape;
        var at: usize = 0;
        while (at < features.len) {
            const n = @min(features.len - at, batch_frames * mels);
            try self.queues[0].rows.appendSlice(self.alloc, features[at..][0..n]);
            self.queues[0].count += n / mels;
            at += n;
            try self.advance(false, out);
        }
    }

    /// Ends the stream as the whole-clip pass ends: the last frames with
    /// zeros past the samples, one zero row of right padding per
    /// convolution, and a final chunk of the rows left.
    pub fn finish(self: *Stream, out: *std.ArrayList(f32)) !void {
        if (self.finished) return error.StreamFinished;
        if (self.encoder.bank) |bank| if (self.pushed > 0) try self.frames(bank, bank.spec.frameCount(self.pushed), out);
        try self.advance(true, out);
        self.finished = true;
    }

    /// Runs the subsampling as far as its inputs allow (at the end, with the
    /// right pad), then every whole chunk (at the end, the rest) through
    /// the blocks.
    fn advance(self: *Stream, last: bool, out: *std.ArrayList(f32)) !void {
        const c = self.config();
        var arena: std.heap.ArenaAllocator = .init(self.alloc);
        defer arena.deinit();
        const ws = arena.allocator();
        for (0..3) |stage| {
            const q = &self.queues[stage];
            const width_in = c.columns(stage);
            const width_out = c.columns(stage + 1);
            const in_row = width_in * (if (stage == 0) 1 else c.channels);
            const out_row = width_out * c.channels;
            // Output t reads input rows 2t − 2 … 2t; the pad row is index `count`.
            if (q.count == 0) continue;
            const limit = if (last) q.count + 1 else q.count;
            const end = (limit - 1) / 2 + 1;
            if (end <= q.next) continue;
            const produced = try ws.alloc(f32, (end - q.next) * out_row);
            try self.convolve(stage, q, end, in_row, width_in, width_out, produced, ws);
            // Rows before 2·next − 2 are read by no later output.
            const keep_from = (2 * end) -| 2;
            if (keep_from > q.first) {
                const drop = @min(keep_from - q.first, q.rows.items.len / in_row);
                std.mem.copyForwards(f32, q.rows.items, q.rows.items[drop * in_row ..]);
                q.rows.shrinkRetainingCapacity(q.rows.items.len - drop * in_row);
                q.first += drop;
            }
            const first_out = q.next;
            q.next = end;
            if (stage < 2) {
                try self.queues[stage + 1].rows.appendSlice(self.alloc, produced);
                self.queues[stage + 1].count += end - first_out;
            } else try self.project(produced, end - first_out, first_out, ws);
        }
        const chunk = self.lookahead + 1;
        const available = self.pending.items.len / c.hidden;
        const whole = if (last) available else available / chunk * chunk;
        // Bounded batches of whole chunks: the stream's rows do not depend on them.
        const batch = @max(chunk, 256 / chunk * chunk);
        var done: usize = 0;
        while (done < whole) {
            const n = @min(whole - done, batch);
            try self.blocks(self.pending.items[done * c.hidden ..][0 .. n * c.hidden], self.row + done, n, out);
            done += n;
        }
        if (done > 0) {
            std.mem.copyForwards(f32, self.pending.items, self.pending.items[done * c.hidden ..]);
            self.pending.shrinkRetainingCapacity(self.pending.items.len - done * c.hidden);
            self.row += done;
        }
    }

    /// Outputs `q.next … end − 1` of subsampling stage `stage` into `out`
    /// (`[t][column][channel]`), ReLU applied.
    fn convolve(self: *Stream, stage: usize, q: *const Queue, end: usize, in_row: usize, width_in: usize, width_out: usize, out: []f32, ws: std.mem.Allocator) !void {
        const c = self.config();
        const w = self.encoder.weights;
        const ch = c.channels;
        const count = end - q.next;
        const zero = try ws.alloc(f32, in_row);
        @memset(zero, 0);
        for (0..count) |i| {
            const t = q.next + i;
            var inputs: [taps][]const f32 = undefined;
            for (0..taps) |kh| {
                const y = @as(i64, @intCast(2 * t + kh)) - 2;
                inputs[kh] = if (y < 0 or y >= q.count) zero else q.rows.items[(@as(usize, @intCast(y)) - q.first) * in_row ..][0..in_row];
            }
            const o = out[i * width_out * ch ..][0 .. width_out * ch];
            for (0..width_out) |f| for (0..ch) |channel| {
                const filter = if (stage == 0) w.conv_in else w.depthwise[stage - 1];
                var acc: f64 = 0;
                for (0..taps) |kh| for (0..taps) |kw| {
                    const x = @as(i64, @intCast(2 * f + kw)) - 2;
                    if (x < 0 or x >= width_in) continue;
                    const v = if (stage == 0) inputs[kh][@intCast(x)] else inputs[kh][@as(usize, @intCast(x)) * ch + channel];
                    acc += @as(f64, filter.weight[(channel * taps + kh) * taps + kw]) * v;
                };
                o[f * ch + channel] = @floatCast(acc + filter.bias.?[channel]);
            };
        }
        if (stage > 0) {
            const mixed = try ws.alloc(f32, out.len);
            const p = w.pointwise[stage - 1];
            try dense.matmul(self.encoder.io, .{ .rows = count * width_out, .inner = ch, .outputs = ch, .x = out, .w = p.weight, .bias = p.bias }, mixed);
            @memcpy(out, mixed);
        }
        for (out) |*v| v.* = @max(v.*, 0);
    }

    /// The channel-major flatten of `count` stage-3 rows, the linear, and
    /// the rows onto `pending`.
    fn project(self: *Stream, produced: []const f32, count: usize, first: usize, ws: std.mem.Allocator) !void {
        const c = self.config();
        const w = self.encoder.weights;
        const cols = c.columns(3);
        const flat = try ws.alloc(f32, count * c.subWidth());
        for (0..count) |t| for (0..cols) |f| for (0..c.channels) |channel| {
            flat[t * c.subWidth() + channel * cols + f] = produced[(t * cols + f) * c.channels + channel];
        };
        const rows = try ws.alloc(f32, count * c.hidden);
        try dense.matmul(self.encoder.io, .{ .rows = count, .inner = c.subWidth(), .outputs = c.hidden, .x = flat, .w = w.linear.weight, .bias = w.linear.bias }, rows);
        Observer.see(self.observer, .subsampling, first, rows);
        try self.pending.appendSlice(self.alloc, rows);
    }

    /// Rows `first … first + n − 1` through every block and the prompt
    /// block, appended to `out`; the caches move past them.
    fn blocks(self: *Stream, input: []const f32, first: usize, n: usize, out: *std.ArrayList(f32)) !void {
        const c = self.config();
        const w = self.encoder.weights;
        const io = self.encoder.io;
        const h = c.hidden;
        var arena: std.heap.ArenaAllocator = .init(self.alloc);
        defer arena.deinit();
        const ws = arena.allocator();
        const x = try ws.dupe(f32, input);
        const t = try ws.alloc(f32, n * h);
        const wide = try ws.alloc(f32, n * @max(c.ffn, 2 * h));
        const q = try ws.alloc(f32, n * h);
        const k = try ws.alloc(f32, n * h);
        const v = try ws.alloc(f32, n * h);
        const glu = try ws.alloc(f32, (c.kernel - 1 + n) * h);
        const scores = try ws.alloc(f64, c.left + 2 * max_lookahead + 1);
        for (w.blocks, 0..) |blk, il| {
            try self.feedForward(blk.norm_ff1, blk.ff1_in, blk.ff1_out, x, t, wide, n);
            // Attention over the cached rows and these.
            try norm(blk.norm_attention, x, t, h);
            try dense.matmul(io, .{ .rows = n, .inner = h, .outputs = h, .x = t, .w = blk.q }, q);
            try dense.matmul(io, .{ .rows = n, .inner = h, .outputs = h, .x = t, .w = blk.k }, k);
            try dense.matmul(io, .{ .rows = n, .inner = h, .outputs = h, .x = t, .w = blk.v }, v);
            const keys = &self.keys[il];
            const values = &self.values[il];
            try keys.appendSlice(self.alloc, k);
            try values.appendSlice(self.alloc, v);
            const held = keys.items.len / h;
            self.attention(blk, il, q, keys.items, values.items, first, first + n - held, n, t, scores);
            try dense.matmul(io, .{ .rows = n, .inner = h, .outputs = h, .x = t, .w = blk.o }, q);
            for (x, q) |*a, b| a.* += b;
            trim(keys, c.left * h);
            trim(values, c.left * h);
            // The convolution module: GLU, causal depthwise over the held rows, norm, SiLU.
            try norm(blk.norm_conv, x, t, h);
            try dense.matmul(io, .{ .rows = n, .inner = h, .outputs = 2 * h, .x = t, .w = blk.pointwise1 }, wide[0 .. n * 2 * h]);
            const held_glu = (c.kernel - 1) * h;
            const history = self.history[il * held_glu ..][0..held_glu];
            @memcpy(glu[0..held_glu], history);
            for (0..n) |r| for (0..h) |ch| {
                const a = wide[r * 2 * h + ch];
                const g = wide[r * 2 * h + h + ch];
                glu[held_glu + r * h + ch] = a * cpu.sigmoid(g);
            };
            for (0..n) |r| for (0..h) |ch| {
                var acc: f64 = 0;
                for (0..c.kernel) |tap| acc += @as(f64, blk.depthwise[ch * c.kernel + tap]) * glu[(r + tap) * h + ch];
                q[r * h + ch] = @floatCast(acc);
            };
            @memcpy(history, glu[n * h ..][0..held_glu]);
            try norm(blk.conv_norm, q, q, h);
            for (q) |*value| value.* = cpu.silu(value.*);
            try dense.matmul(io, .{ .rows = n, .inner = h, .outputs = h, .x = q, .w = blk.pointwise2 }, t);
            for (x, t) |*a, b| a.* += b;
            try self.feedForward(blk.norm_ff2, blk.ff2_in, blk.ff2_out, x, t, wide, n);
            try norm(blk.norm_out, x, x, h);
            Observer.see(self.observer, .{ .block = il }, first, x);
        }
        // The prompt block: [row ; one-hot] → prompt_hidden → hidden.
        const joined = try ws.alloc(f32, n * (h + c.prompts));
        @memset(joined, 0);
        for (0..n) |r| {
            @memcpy(joined[r * (h + c.prompts) ..][0..h], x[r * h ..][0..h]);
            joined[r * (h + c.prompts) + h + self.prompt] = 1;
        }
        const inner = try ws.alloc(f32, n * c.prompt_hidden);
        try dense.matmul(io, .{ .rows = n, .inner = h + c.prompts, .outputs = c.prompt_hidden, .x = joined, .w = w.prompt_in.weight, .bias = w.prompt_in.bias }, inner);
        for (inner) |*value| value.* = @max(value.*, 0);
        const rows = try out.addManyAsSlice(self.alloc, n * h);
        try dense.matmul(io, .{ .rows = n, .inner = c.prompt_hidden, .outputs = h, .x = inner, .w = w.prompt_out.weight, .bias = w.prompt_out.bias }, rows);
        Observer.see(self.observer, .prompt, first, rows);
    }

    /// x += ½ · W_out silu(W_in LN(x)).
    fn feedForward(self: *Stream, n_: Norm, w_in: []const f32, w_out: []const f32, x: []f32, t: []f32, wide: []f32, n: usize) !void {
        const c = self.config();
        const io = self.encoder.io;
        try norm(n_, x, t, c.hidden);
        const up = wide[0 .. n * c.ffn];
        try dense.matmul(io, .{ .rows = n, .inner = c.hidden, .outputs = c.ffn, .x = t, .w = w_in }, up);
        for (up) |*value| value.* = cpu.silu(value.*);
        try dense.matmul(io, .{ .rows = n, .inner = c.ffn, .outputs = c.hidden, .x = up, .w = w_out }, t);
        for (x, t) |*a, b| a.* += 0.5 * b;
    }

    /// Relative attention of `n` queries (rows `first …`) over the held
    /// keys (rows `key_first …`), chunk-masked; per head
    /// `((q + u)·k + (q + v)·p(i − j)) / √head_dim`, softmax, values.
    fn attention(self: *Stream, blk: Block, layer: usize, q: []const f32, keys: []const f32, values: []const f32, first: usize, key_first: usize, n: usize, out: []f32, scores: []f64) void {
        const c = self.config();
        const h = c.hidden;
        const d = c.headDim();
        const chunk = self.lookahead + 1;
        const reach = c.left / chunk;
        const relk = self.encoder.relk[layer * c.positions() * h ..][0 .. c.positions() * h];
        const held = keys.len / h;
        const scale = 1.0 / @sqrt(@as(f64, @floatFromInt(d)));
        for (0..c.heads) |head| for (0..n) |i| {
            const a = first + i;
            const qi = q[i * h + head * d ..][0..d];
            const u = blk.bias_u[head * d ..][0..d];
            const pv = blk.bias_v[head * d ..][0..d];
            // Visible keys: chunks ⌊a/chunk⌋ − reach … ⌊a/chunk⌋.
            const lo_row = @max(key_first, (a / chunk -| reach) * chunk);
            const hi_row = @min(key_first + held, (a / chunk + 1) * chunk);
            var max: f64 = -std.math.inf(f64);
            for (lo_row..hi_row, 0..) |b, s| {
                const kj = keys[(b - key_first) * h + head * d ..][0..d];
                const p = relk[(a + max_lookahead - b) * h + head * d ..][0..d];
                var ac: f64 = 0;
                var bd: f64 = 0;
                for (qi, u, pv, kj, p) |qv, uv, vv, kv, pp| {
                    ac += @as(f64, qv + uv) * kv;
                    bd += @as(f64, qv + vv) * pp;
                }
                scores[s] = (ac + bd) * scale;
                max = @max(max, scores[s]);
            }
            const count = hi_row - lo_row;
            var total: f64 = 0;
            for (scores[0..count]) |*s| {
                s.* = @exp(s.* - max);
                total += s.*;
            }
            const o = out[i * h + head * d ..][0..d];
            for (o, 0..) |*value, e| {
                var sum: f64 = 0;
                for (lo_row..hi_row, 0..) |b, s| sum += scores[s] * values[(b - key_first) * h + head * d + e];
                value.* = @floatCast(sum / total);
            }
        };
    }
};

fn norm(n: Norm, x: []const f32, out: []f32, width: usize) !void {
    for (0..x.len / width) |r| try dense.layerNorm(x[r * width ..][0..width], n.weight, n.bias, epsilon, out[r * width ..][0..width]);
}

/// Keeps the last `keep` values of `list`.
fn trim(list: *std.ArrayList(f32), keep: usize) void {
    if (list.items.len <= keep) return;
    const drop = list.items.len - keep;
    std.mem.copyForwards(f32, list.items, list.items[drop..]);
    list.shrinkRetainingCapacity(keep);
}

// ---- tests: synthetic weights at a small width, no model file ----

const tiny: Config = .{ .hidden = 16, .heads = 2, .ffn = 24, .blocks = 2, .channels = 3, .mels = mel.filters, .kernel = 4, .left = 8, .prompts = 5, .prompt_hidden = 12 };

/// Random weights for `config`, all in one allocation the caller frees.
const Synthetic = struct {
    storage: []f32,
    blocks: []Block,
    weights: Weights,

    fn init(gpa: std.mem.Allocator, config: Config, seed: u64) !Synthetic {
        var prng = std.Random.DefaultPrng.init(seed);
        const random = prng.random();
        const c = config;
        const h = c.hidden;
        const per_block = 4 * c.ffn * h + 8 * h * h + h * c.kernel + 14 * h;
        const total = 2 * c.channels * taps * taps + 5 * c.channels + 2 * c.channels * c.channels + c.channels * taps * taps +
            h * c.subWidth() + h + c.blocks * per_block + c.prompt_hidden * (h + c.prompts + 1) + h * (c.prompt_hidden + 1);
        const storage = try gpa.alloc(f32, total);
        errdefer gpa.free(storage);
        const blocks = try gpa.alloc(Block, c.blocks);
        var at: usize = 0;
        const S = struct {
            fn take(s: []f32, cursor: *usize, r: std.Random, n: usize, scale: f32, offset: f32) []const f32 {
                const out = s[cursor.*..][0..n];
                cursor.* += n;
                for (out) |*v| v.* = offset + scale * (r.float(f32) * 2 - 1);
                return out;
            }
        };
        const t = S.take;
        var w: Weights = undefined;
        w.config = c;
        w.conv_in = .{ .weight = t(storage, &at, random, c.channels * taps * taps, 0.5, 0), .bias = t(storage, &at, random, c.channels, 0.1, 0) };
        for (0..2) |i| {
            w.depthwise[i] = .{ .weight = t(storage, &at, random, c.channels * taps * taps, 0.5, 0), .bias = t(storage, &at, random, c.channels, 0.1, 0) };
            w.pointwise[i] = .{ .weight = t(storage, &at, random, c.channels * c.channels, 0.5, 0), .bias = t(storage, &at, random, c.channels, 0.1, 0) };
        }
        w.linear = .{ .weight = t(storage, &at, random, h * c.subWidth(), 0.2, 0), .bias = t(storage, &at, random, h, 0.1, 0) };
        const scale = 1 / @sqrt(@as(f32, @floatFromInt(h)));
        for (blocks) |*b| b.* = .{
            .norm_ff1 = .{ .weight = t(storage, &at, random, h, 0.2, 1), .bias = t(storage, &at, random, h, 0.1, 0) },
            .ff1_in = t(storage, &at, random, c.ffn * h, scale, 0),
            .ff1_out = t(storage, &at, random, h * c.ffn, scale, 0),
            .norm_attention = .{ .weight = t(storage, &at, random, h, 0.2, 1), .bias = t(storage, &at, random, h, 0.1, 0) },
            .q = t(storage, &at, random, h * h, scale, 0),
            .k = t(storage, &at, random, h * h, scale, 0),
            .v = t(storage, &at, random, h * h, scale, 0),
            .o = t(storage, &at, random, h * h, scale, 0),
            .relative = t(storage, &at, random, h * h, scale, 0),
            .bias_u = t(storage, &at, random, h, 0.2, 0),
            .bias_v = t(storage, &at, random, h, 0.2, 0),
            .norm_conv = .{ .weight = t(storage, &at, random, h, 0.2, 1), .bias = t(storage, &at, random, h, 0.1, 0) },
            .pointwise1 = t(storage, &at, random, 2 * h * h, scale, 0),
            .depthwise = t(storage, &at, random, h * c.kernel, 0.5, 0),
            .conv_norm = .{ .weight = t(storage, &at, random, h, 0.2, 1), .bias = t(storage, &at, random, h, 0.1, 0) },
            .pointwise2 = t(storage, &at, random, h * h, scale, 0),
            .norm_ff2 = .{ .weight = t(storage, &at, random, h, 0.2, 1), .bias = t(storage, &at, random, h, 0.1, 0) },
            .ff2_in = t(storage, &at, random, c.ffn * h, scale, 0),
            .ff2_out = t(storage, &at, random, h * c.ffn, scale, 0),
            .norm_out = .{ .weight = t(storage, &at, random, h, 0.2, 1), .bias = t(storage, &at, random, h, 0.1, 0) },
        };
        w.blocks = blocks;
        w.prompt_in = .{ .weight = t(storage, &at, random, c.prompt_hidden * (h + c.prompts), 0.3, 0), .bias = t(storage, &at, random, c.prompt_hidden, 0.1, 0) };
        w.prompt_out = .{ .weight = t(storage, &at, random, h * c.prompt_hidden, 0.3, 0), .bias = t(storage, &at, random, h, 0.1, 0) };
        std.debug.assert(at == total);
        return .{ .storage = storage, .blocks = blocks, .weights = w };
    }

    fn deinit(self: *Synthetic, gpa: std.mem.Allocator) void {
        gpa.free(self.blocks);
        gpa.free(self.storage);
    }
};

fn randomFeatures(gpa: std.mem.Allocator, frames: usize, mels: usize, seed: u64) ![]f32 {
    var prng = std.Random.DefaultPrng.init(seed);
    const out = try gpa.alloc(f32, frames * mels);
    for (out) |*v| v.* = prng.random().float(f32) * 6 - 4;
    return out;
}

test "rows: three halvings, two zeros before and one after" {
    try std.testing.expectEqual(@as(usize, 64), rowsFor(505));
    try std.testing.expectEqual(@as(usize, 1), rowsFor(1));
    try std.testing.expectEqual(@as(usize, 4), rowsFor(25));
    try std.testing.expectEqual(@as(usize, 0), rowsFor(0));
    try std.testing.expectEqual(@as(usize, 17), Config.nemotron.columns(3));
    try std.testing.expectEqual(@as(usize, 4352), Config.nemotron.subWidth());
}

test "positions: distance 0 is sin 0, cos 1 interleaved; the first row is distance −13" {
    var rows: [(tiny.left + 2 * max_lookahead + 1) * tiny.hidden]f32 = undefined;
    positionRows(tiny, &rows);
    const zero = rows[max_lookahead * tiny.hidden ..][0..tiny.hidden];
    for (0..tiny.hidden / 2) |m| {
        try std.testing.expectEqual(@as(f32, 0), zero[2 * m]);
        try std.testing.expectEqual(@as(f32, 1), zero[2 * m + 1]);
    }
    try std.testing.expectApproxEqAbs(@as(f32, @sin(-13.0)), rows[0], 1e-6);
}

test "streamed in chunks equals the whole clip at r = 0 and r = 3, N = 1 + 8r + 8(r + 1)m" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var synthetic = try Synthetic.init(gpa, tiny, 1);
    defer synthetic.deinit(gpa);
    var encoder = try Encoder.init(gpa, io, &synthetic.weights);
    defer encoder.deinit();
    for ([_][2]usize{ .{ 0, 20 }, .{ 3, 4 } }) |case| {
        const r = case[0];
        const m = case[1];
        const frames = 1 + 8 * r + 8 * (r + 1) * m;
        const features = try randomFeatures(gpa, frames, tiny.mels, 2 + r);
        defer gpa.free(features);
        const whole = try encoder.encode(gpa, features, r, 3, null);
        defer gpa.free(whole);
        try std.testing.expectEqual((m + 1) * (r + 1), rowsFor(frames));
        try std.testing.expectEqual(rowsFor(frames) * tiny.hidden, whole.len);
        // The authors' chunks: 1 + 8r mel frames, then 8(r + 1); each gives r + 1 rows.
        var stream = try Stream.init(gpa, &encoder, r, 3);
        defer stream.deinit();
        var rows: std.ArrayList(f32) = .empty;
        defer rows.deinit(gpa);
        var at: usize = 0;
        var chunk: usize = 1 + 8 * r;
        while (at < frames) : (chunk = 8 * (r + 1)) {
            try stream.pushFeatures(features[at * tiny.mels ..][0 .. chunk * tiny.mels], &rows);
            at += chunk;
            try std.testing.expectEqual(rowsFor(at) * tiny.hidden, rows.items.len);
        }
        try stream.finish(&rows);
        try std.testing.expectEqualSlices(f32, whole, rows.items);
    }
}

test "any pieces, any length: a stream ends as the whole clip does" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var synthetic = try Synthetic.init(gpa, tiny, 5);
    defer synthetic.deinit(gpa);
    var encoder = try Encoder.init(gpa, io, &synthetic.weights);
    defer encoder.deinit();
    const frames = 150;
    const features = try randomFeatures(gpa, frames, tiny.mels, 6);
    defer gpa.free(features);
    const whole = try encoder.encode(gpa, features, 3, 0, null);
    defer gpa.free(whole);
    try std.testing.expectEqual(rowsFor(frames) * tiny.hidden, whole.len);
    var stream = try Stream.init(gpa, &encoder, 3, 0);
    defer stream.deinit();
    var rows: std.ArrayList(f32) = .empty;
    defer rows.deinit(gpa);
    var at: usize = 0;
    while (at < frames) {
        const n: usize = @min(frames - at, 7);
        try stream.pushFeatures(features[at * tiny.mels ..][0 .. n * tiny.mels], &rows);
        at += n;
        // Only whole chunks of four rows leave before the end.
        try std.testing.expectEqual(@as(usize, 0), rows.items.len / tiny.hidden % 4);
    }
    try stream.finish(&rows);
    try std.testing.expectEqualSlices(f32, whole, rows.items);
    try std.testing.expectError(error.StreamFinished, stream.finish(&rows));
    try std.testing.expectError(error.UnsupportedLookahead, Stream.init(gpa, &encoder, 2, 0));
    try std.testing.expectError(error.UnsupportedPrompt, Stream.init(gpa, &encoder, 3, tiny.prompts));
}

test "samples in 80 ms pieces give the rows of the clip's features" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var synthetic = try Synthetic.init(gpa, tiny, 9);
    defer synthetic.deinit(gpa);
    var encoder = try Encoder.init(gpa, io, &synthetic.weights);
    defer encoder.deinit();
    // 1.03 s of a chirp: 103 frames, the last one reading 40 zeros past the end.
    const pcm = try gpa.alloc(f32, 16480);
    defer gpa.free(pcm);
    for (pcm, 0..) |*s, n| {
        const time = @as(f64, @floatFromInt(n)) / mel.sample_rate;
        s.* = @floatCast(0.3 * @sin(2 * std.math.pi * (200 + 900 * time) * time));
    }
    const frames = mel.Spec.nemotron.frameCount(pcm.len);
    const features = try gpa.alloc(f32, frames * mel.filters);
    defer gpa.free(features);
    try encoder.bank.?.features(pcm, features);
    const whole = try encoder.encode(gpa, features, 0, 1, null);
    defer gpa.free(whole);
    var stream = try Stream.init(gpa, &encoder, 0, 1);
    defer stream.deinit();
    var rows: std.ArrayList(f32) = .empty;
    defer rows.deinit(gpa);
    var at: usize = 0;
    while (at < pcm.len) : (at += 1280) {
        try stream.push(pcm[at..@min(pcm.len, at + 1280)], &rows);
        // Only the samples of frames still open are held.
        try std.testing.expect(stream.samples.items.len < 1280 + 400);
    }
    try stream.finish(&rows);
    try std.testing.expectEqualSlices(f32, whole, rows.items);
}

/// A safetensors file of `config`'s encoder tensors, all F32 zeros, with
/// `skew`'s first dimension one larger.
fn writeCheckpoint(gpa: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, config: Config, skew: ?[]const u8) !void {
    const c = config;
    const h = c.hidden;
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const a = arena.allocator();
    const Entry = struct { name: []const u8, shape: []const usize };
    var entries: std.ArrayList(Entry) = .empty;
    const List = struct {
        fn add(al: std.mem.Allocator, list: *std.ArrayList(Entry), comptime format: []const u8, args: anytype, shape: []const usize) !void {
            try list.append(al, .{ .name = try std.fmt.allocPrint(al, format, args), .shape = try al.dupe(usize, shape) });
        }
    };
    const add = List.add;
    try add(a, &entries, "encoder.subsampling.conv_in.weight", .{}, &.{ c.channels, 1, 3, 3 });
    try add(a, &entries, "encoder.subsampling.conv_in.bias", .{}, &.{c.channels});
    for (0..2) |i| {
        try add(a, &entries, "encoder.subsampling.layers.{d}.depthwise_conv.weight", .{i}, &.{ c.channels, 1, 3, 3 });
        try add(a, &entries, "encoder.subsampling.layers.{d}.depthwise_conv.bias", .{i}, &.{c.channels});
        try add(a, &entries, "encoder.subsampling.layers.{d}.pointwise_conv.weight", .{i}, &.{ c.channels, c.channels, 1, 1 });
        try add(a, &entries, "encoder.subsampling.layers.{d}.pointwise_conv.bias", .{i}, &.{c.channels});
    }
    try add(a, &entries, "encoder.subsampling.linear.weight", .{}, &.{ h, c.subWidth() });
    try add(a, &entries, "encoder.subsampling.linear.bias", .{}, &.{h});
    for (0..c.blocks) |i| {
        for ([_][]const u8{ "norm_feed_forward1", "norm_self_att", "norm_conv", "conv.norm", "norm_feed_forward2", "norm_out" }) |n| {
            try add(a, &entries, "encoder.layers.{d}.{s}.weight", .{ i, n }, &.{h});
            try add(a, &entries, "encoder.layers.{d}.{s}.bias", .{ i, n }, &.{h});
        }
        for ([_][]const u8{ "feed_forward1", "feed_forward2" }) |n| {
            try add(a, &entries, "encoder.layers.{d}.{s}.linear1.weight", .{ i, n }, &.{ c.ffn, h });
            try add(a, &entries, "encoder.layers.{d}.{s}.linear2.weight", .{ i, n }, &.{ h, c.ffn });
        }
        for ([_][]const u8{ "q_proj", "k_proj", "v_proj", "o_proj", "relative_k_proj" }) |n|
            try add(a, &entries, "encoder.layers.{d}.self_attn.{s}.weight", .{ i, n }, &.{ h, h });
        try add(a, &entries, "encoder.layers.{d}.self_attn.bias_u", .{i}, &.{ c.heads, c.headDim() });
        try add(a, &entries, "encoder.layers.{d}.self_attn.bias_v", .{i}, &.{ c.heads, c.headDim() });
        try add(a, &entries, "encoder.layers.{d}.conv.pointwise_conv1.weight", .{i}, &.{ 2 * h, h, 1 });
        try add(a, &entries, "encoder.layers.{d}.conv.depthwise_conv.weight", .{i}, &.{ h, 1, c.kernel });
        try add(a, &entries, "encoder.layers.{d}.conv.pointwise_conv2.weight", .{i}, &.{ h, h, 1 });
    }
    try add(a, &entries, "prompt_projector.linear_1.weight", .{}, &.{ c.prompt_hidden, h + c.prompts });
    try add(a, &entries, "prompt_projector.linear_1.bias", .{}, &.{c.prompt_hidden});
    try add(a, &entries, "prompt_projector.linear_2.weight", .{}, &.{ h, c.prompt_hidden });
    try add(a, &entries, "prompt_projector.linear_2.bias", .{}, &.{h});

    var header: std.Io.Writer.Allocating = .init(a);
    var offset: usize = 0;
    try header.writer.writeAll("{");
    for (entries.items, 0..) |e, i| {
        const shape = try a.dupe(usize, e.shape);
        if (skew) |name| if (std.mem.eql(u8, name, e.name)) {
            shape[0] += 1;
        };
        var elements: usize = 1;
        for (shape) |d| elements *= d;
        if (i > 0) try header.writer.writeAll(",");
        try header.writer.print("\"{s}\":{{\"dtype\":\"F32\",\"shape\":[", .{e.name});
        for (shape, 0..) |d, k| try header.writer.print("{s}{d}", .{ if (k > 0) "," else "", d });
        try header.writer.print("],\"data_offsets\":[{d},{d}]}}", .{ offset, offset + 4 * elements });
        offset += 4 * elements;
    }
    try header.writer.writeAll("}");
    // Pad the header so the data starts aligned, as writers do.
    while ((8 + header.written().len) % 8 != 0) try header.writer.writeAll(" ");
    const json = header.written();
    const bytes = try a.alloc(u8, 8 + json.len + offset);
    std.mem.writeInt(u64, bytes[0..8], json.len, .little);
    @memcpy(bytes[8..][0..json.len], json);
    @memset(bytes[8 + json.len ..], 0);
    try dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = bytes });
}

test "bind views every tensor at the config's shapes and refuses a wrong one" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    for ([_]?[]const u8{ null, "encoder.layers.1.self_attn.relative_k_proj.weight", "prompt_projector.linear_1.weight" }) |skew| {
        var tmp = std.testing.tmpDir(.{});
        defer tmp.cleanup();
        try writeCheckpoint(gpa, io, tmp.dir, tiny, skew);
        const path = try tmp.dir.realPathFileAlloc(io, "model.safetensors", gpa);
        defer gpa.free(path);
        var checkpoint = try safetensors.Checkpoint.open(gpa, io, path, .{});
        defer checkpoint.deinit(io);
        if (skew != null) {
            try std.testing.expectError(error.TensorShapeMismatch, bind(gpa, &checkpoint, tiny));
            continue;
        }
        var binding = try bind(gpa, &checkpoint, tiny);
        defer binding.deinit();
        try std.testing.expectEqual(tiny.blocks, binding.weights.blocks.len);
        try std.testing.expectEqual(tiny.hidden * tiny.kernel, binding.weights.blocks[1].depthwise.len);
        // Another width finds the tensors but not their shapes; another depth misses one.
        var wider = tiny;
        wider.ffn += 1;
        try std.testing.expectError(error.TensorShapeMismatch, bind(gpa, &checkpoint, wider));
        var deeper = tiny;
        deeper.blocks += 1;
        try std.testing.expectError(error.MissingTensor, bind(gpa, &checkpoint, deeper));
    }
}

test "config.json: the checkpoint's values, and what is refused" {
    const gpa = std.testing.allocator;
    const text =
        \\{"encoder_config":{"hidden_size":1024,"intermediate_size":4096,"num_hidden_layers":24,"num_attention_heads":8,
        \\ "num_key_value_heads":8,"hidden_act":"silu","attention_bias":false,"convolution_bias":false,"conv_kernel_size":9,
        \\ "subsampling_factor":8,"subsampling_conv_channels":256,"subsampling_conv_kernel_size":3,"subsampling_conv_stride":2,
        \\ "num_mel_bins":128,"scale_input":false,"sliding_window":57,"supported_num_lookahead_tokens":[3,0,6,13]},
        \\ "num_prompts":128,"prompt_intermediate_size":2048,"vocab_size":13088}
    ;
    try std.testing.expectEqual(Config.nemotron, try parseConfig(gpa, text));
    const cases = [_]struct { []const u8, []const u8, anyerror }{
        .{ "\"silu\"", "\"gelu\"", error.UnsupportedConfig },
        .{ "\"scale_input\":false", "\"scale_input\":true", error.UnsupportedConfig },
        .{ "[3,0,6,13]", "[3,0,1,13]", error.UnsupportedConfig },
        .{ "\"num_key_value_heads\":8", "\"num_key_value_heads\":4", error.UnsupportedConfig },
        .{ "\"hidden_size\":1024", "\"hidden_size\":1020", error.InvalidConfig },
        .{ "\"sliding_window\":57", "\"sliding_window\":0", error.InvalidConfig },
        .{ "\"num_prompts\":128", "\"num_prompts\":\"x\"", error.InvalidConfig },
    };
    for (cases) |case| {
        const bad = try std.mem.replaceOwned(u8, gpa, text, case[0], case[1]);
        defer gpa.free(bad);
        try std.testing.expectError(case[2], parseConfig(gpa, bad));
    }
}
