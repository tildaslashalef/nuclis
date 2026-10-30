//! Nemotron 3.5 ASR on Metal: the encoder's chunk step (the subsampling, the
//! 24 blocks, the prompt block), the projector and the joint on the device;
//! the prediction network's LSTM on the CPU (`nemotron_rnnt.Decoder.predict`).
//! A `Plan` holds the device weights and the scratch its streams share; a
//! `Stream` is one utterance: the front end on the host, its caches on the
//! device, and the CPU `Stream`'s and `Decoder`'s semantics (push, finish,
//! reset; rows leave in whole chunks and are decoded as they arrive).
//! Matrices are the checkpoint mapping's F32 bytes wrapped in place, read by
//! the generic F32 tile. The plan owns its `Backend`, so every device buffer
//! goes with `destroy`; the checkpoint and the CPU encoder must outlive it.
//! One stream records at a time. docs/models/nemotron-asr.md § On Metal.
const std = @import("std");
const metal = @import("../backends/metal/root.zig");
const cpu = @import("../backends/cpu/root.zig");
const mel = @import("mel.zig");
const encoder = @import("nemotron_encoder.zig");
const rnnt = @import("nemotron_rnnt.zig");

const Backend = metal.Backend;
const Buffer = metal.Buffer;

/// Mel frames fed to the subsampling per command buffer, as the CPU stream.
const batch_frames = encoder.Stream.batch_frames;
/// Rows one subsampling stage can emit per batch: half its input, plus the
/// two rows a stage keeps and the end's pad row.
fn stageOut(rows_in: usize) usize {
    return rows_in / 2 + 2;
}
const out_rows = [3]usize{ stageOut(batch_frames), stageOut(stageOut(batch_frames)), stageOut(stageOut(stageOut(batch_frames))) };
/// Encoder rows through the blocks at once: a batch's rows and the
/// unfinished chunk before them.
pub const max_rows = 128;
comptime {
    std.debug.assert(out_rows[2] + encoder.max_lookahead <= max_rows);
}
/// Rows of joint logits per speculative batch: one prediction `g` scores
/// them all, valid until the next emitted token changes it.
pub const joint_rows = 16;
/// Prompts one stream decodes at once: its own and `max_prompts − 1`
/// alternates, each with its prompt block, projector rows and decoder.
pub const max_prompts = 4;

const Weight = struct { buffer: Buffer, matrix: cpu.Matrix };
const Norm = struct { weight: Buffer, bias: Buffer };
const Block = struct {
    norm_ff1: Norm,
    ff1_in: Weight,
    ff1_out: Weight,
    norm_attention: Norm,
    q: Weight,
    k: Weight,
    v: Weight,
    o: Weight,
    bias_u: Buffer,
    bias_v: Buffer,
    norm_conv: Norm,
    pointwise1: Weight,
    depthwise: Buffer,
    conv_norm: Norm,
    pointwise2: Weight,
    norm_ff2: Norm,
    ff2_in: Weight,
    ff2_out: Weight,
    norm_out: Norm,
};

/// Where the time of a stream goes, accumulated since its start (wall time).
pub const Stats = struct {
    /// Command buffers that ran the encoder, and the rows they produced.
    chunks: usize = 0,
    rows: usize = 0,
    /// Recording, running and waiting for the encoder's command buffers,
    /// each with the first joint batch of its rows.
    encode_ns: u64 = 0,
    /// The later joint batches (a token changed `g`) and the LSTM steps.
    joint_batches: usize = 0,
    joint_ns: u64 = 0,
    lstm_steps: usize = 0,
    lstm_ns: u64 = 0,
};

pub const Plan = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    backend: Backend,
    encoder: *const encoder.Encoder,
    transducer: *const rnnt.Weights,
    conv_in: Norm,
    depthwise: [2]Norm,
    pointwise: [2]Weight,
    pointwise_bias: [2]Buffer,
    /// The subsampling's linear with its columns in `[column][channel]`
    /// order, so it reads the last convolution's rows as they are.
    linear: Weight,
    linear_bias: Buffer,
    blocks: []Block,
    /// `blocks × positions × hidden`: the CPU encoder's relative keys.
    relk: Buffer,
    /// `prompt_projector.linear_1`'s first `hidden` columns; the one-hot's
    /// column joins its bias per stream (`Stream.prompt_bias`).
    prompt_in: Weight,
    prompt_out: Weight,
    prompt_out_bias: Buffer,
    projector: Weight,
    projector_bias: Buffer,
    head: Weight,
    head_bias: Buffer,
    // Scratch the streams share, `matmulPadded(max_rows)` rows each.
    x: Buffer,
    t: Buffer,
    q: Buffer,
    attn: Buffer,
    wide: Buffer,
    inner: Buffer,
    prompt_bias: Buffer,
    enc: Buffer,
    /// The depthwise subsampling convolutions' output, the pointwise input.
    dw: Buffer,
    /// The last subsampling convolution's rows, the linear's input.
    sub: Buffer,
    e: Buffer,
    g: Buffer,
    jin: Buffer,
    logits: Buffer,
    /// Copies of each stage's rows for an observer: subsampling, blocks, prompt.
    taps: ?Buffer = null,

    /// Builds the plan on a new Metal backend over `enc`'s and
    /// `transducer`'s weights, which (with their checkpoint) must outlive it.
    pub fn create(alloc: std.mem.Allocator, io: std.Io, enc: *const encoder.Encoder, transducer: *const rnnt.Weights) !*Plan {
        const w = enc.weights;
        const c = w.config;
        const t = transducer.config;
        if (c.headDim() != 128 or c.hidden % 128 != 0 or c.channels % 64 != 0 or c.hidden % 64 != 0 or t.hidden % 64 != 0 or t.vocab % 8 != 0 or c.mels != mel.filters or t.encoder != c.hidden) return error.UnsupportedConfig;
        const self = try alloc.create(Plan);
        errdefer alloc.destroy(self);
        var diagnostic: [512]u8 = @splat(0);
        self.backend = try Backend.init(alloc, &diagnostic);
        errdefer self.backend.deinit();
        self.alloc = alloc;
        self.io = io;
        self.encoder = enc;
        self.transducer = transducer;
        self.taps = null;
        const h = c.hidden;

        self.conv_in = .{ .weight = try self.vector(w.conv_in.weight), .bias = try self.vector(w.conv_in.bias.?) };
        for (0..2) |i| {
            self.depthwise[i] = .{ .weight = try self.vector(w.depthwise[i].weight), .bias = try self.vector(w.depthwise[i].bias.?) };
            self.pointwise[i] = try self.matrix(w.pointwise[i].weight, c.channels, c.channels);
            self.pointwise_bias[i] = try self.vector(w.pointwise[i].bias.?);
        }
        const cols = c.columns(3);
        const linear = try self.created(h * c.subWidth());
        for (0..h) |o| for (0..cols) |f| for (0..c.channels) |ch| {
            linear.floats()[o * c.subWidth() + f * c.channels + ch] = w.linear.weight[o * c.subWidth() + ch * cols + f];
        };
        self.linear = .{ .buffer = linear, .matrix = .{ .encoding = 0, .rows = h, .columns = c.subWidth(), .bytes = linear.host[0..linear.len] } };
        self.linear_bias = try self.vector(w.linear.bias.?);

        self.blocks = try alloc.alloc(Block, c.blocks);
        errdefer alloc.free(self.blocks);
        for (self.blocks, w.blocks) |*out, blk| out.* = .{
            .norm_ff1 = try self.norm(blk.norm_ff1),
            .ff1_in = try self.matrix(blk.ff1_in, c.ffn, h),
            .ff1_out = try self.matrix(blk.ff1_out, h, c.ffn),
            .norm_attention = try self.norm(blk.norm_attention),
            .q = try self.matrix(blk.q, h, h),
            .k = try self.matrix(blk.k, h, h),
            .v = try self.matrix(blk.v, h, h),
            .o = try self.matrix(blk.o, h, h),
            .bias_u = try self.vector(blk.bias_u),
            .bias_v = try self.vector(blk.bias_v),
            .norm_conv = try self.norm(blk.norm_conv),
            .pointwise1 = try self.matrix(blk.pointwise1, 2 * h, h),
            .depthwise = try self.vector(blk.depthwise),
            .conv_norm = try self.norm(blk.conv_norm),
            .pointwise2 = try self.matrix(blk.pointwise2, h, h),
            .norm_ff2 = try self.norm(blk.norm_ff2),
            .ff2_in = try self.matrix(blk.ff2_in, c.ffn, h),
            .ff2_out = try self.matrix(blk.ff2_out, h, c.ffn),
            .norm_out = try self.norm(blk.norm_out),
        };
        self.relk = try self.created(enc.relk.len);
        @memcpy(self.relk.floats()[0..enc.relk.len], enc.relk);
        const prompt_in = try self.created(c.prompt_hidden * h);
        for (0..c.prompt_hidden) |o| @memcpy(prompt_in.floats()[o * h ..][0..h], w.prompt_in.weight[o * (h + c.prompts) ..][0..h]);
        self.prompt_in = .{ .buffer = prompt_in, .matrix = .{ .encoding = 0, .rows = c.prompt_hidden, .columns = h, .bytes = prompt_in.host[0..prompt_in.len] } };
        self.prompt_out = try self.matrix(w.prompt_out.weight, h, c.prompt_hidden);
        self.prompt_out_bias = try self.vector(w.prompt_out.bias.?);
        self.projector = try self.matrix(transducer.projector.weight, t.hidden, t.encoder);
        self.projector_bias = try self.vector(transducer.projector.bias);
        self.head = try self.matrix(transducer.head.weight, t.vocab, t.hidden);
        self.head_bias = try self.vector(transducer.head.bias);

        const rows = Backend.matmulPadded(max_rows);
        self.x = try self.created(rows * h);
        self.t = try self.created(rows * h);
        self.q = try self.created(rows * h);
        self.attn = try self.created(rows * h);
        self.wide = try self.created(rows * @max(c.ffn, 2 * h));
        self.inner = try self.created(rows * c.prompt_hidden);
        self.prompt_bias = try self.created(max_prompts * c.prompt_hidden);
        self.enc = try self.created(rows * h);
        self.dw = try self.created((out_rows[1] * c.columns(2) + Backend.matmul_tile) * c.channels);
        self.sub = try self.created(@max((out_rows[2] * cols + Backend.matmul_tile) * c.channels, Backend.matmulPadded(out_rows[2]) * c.subWidth()));
        self.e = try self.created(max_prompts * rows * t.hidden);
        self.g = try self.created(t.hidden);
        self.jin = try self.created(Backend.matmulPadded(joint_rows) * t.hidden);
        self.logits = try self.created(Backend.matmulPadded(joint_rows) * t.vocab);
        return self;
    }

    pub fn destroy(self: *Plan) void {
        const alloc = self.alloc;
        // Device buffers belong to the backend and go with it.
        self.backend.deinit();
        alloc.free(self.blocks);
        alloc.destroy(self);
    }

    /// Prompt `which`'s projected rows: one of `e`'s `max_prompts` parts.
    fn rowsOf(self: *const Plan, which: usize) Buffer {
        const part = Backend.matmulPadded(max_rows) * self.transducer.config.hidden * 4;
        return self.e.slice(which * part, part);
    }

    fn config(self: *const Plan) encoder.Config {
        return self.encoder.weights.config;
    }

    /// A zeroed F32 buffer of `floats` values.
    fn created(self: *Plan, floats: usize) !Buffer {
        return self.backend.create(floats * 4);
    }
    fn vector(self: *Plan, values: []const f32) !Buffer {
        return self.backend.wrap(std.mem.sliceAsBytes(values));
    }
    fn norm(self: *Plan, n: encoder.Norm) !Norm {
        return .{ .weight = try self.vector(n.weight), .bias = try self.vector(n.bias) };
    }
    fn matrix(self: *Plan, values: []const f32, rows: usize, columns: usize) !Weight {
        if (values.len != rows * columns or rows % 8 != 0 or columns % 64 != 0) return error.UnsupportedConfig;
        const bytes = std.mem.sliceAsBytes(values);
        return .{ .buffer = try self.backend.wrap(bytes), .matrix = .{ .encoding = 0, .rows = rows, .columns = columns, .bytes = bytes } };
    }

    /// `output = W · input` over `n` rows, then silu when `silu`: a chunk's
    /// few rows through the rows kernel (each weight row read once), a
    /// batch's through the tile.
    fn product(self: *Plan, w: Weight, input: Buffer, in_stride: usize, output: Buffer, out_stride: usize, n: usize, silu: bool) !void {
        const b = &self.backend;
        if (n <= Backend.asr_matmul_rows_max) return b.asrMatmulRows(w.matrix, &.{.{ .weights = w.buffer, .output = output }}, input, in_stride, out_stride, n, if (silu) .silu else .store);
        try b.matmulTile(w.buffer, w.matrix, input, in_stride, output, out_stride, n);
        if (silu) try b.silu(output, n * out_stride);
    }

    /// `x += factor · W · input` over `n` rows (the tile's product goes
    /// through `t`).
    fn residual(self: *Plan, w: Weight, input: Buffer, in_stride: usize, n: usize, factor: f32) !void {
        const b = &self.backend;
        const h = self.config().hidden;
        if (n <= Backend.asr_matmul_rows_max) return b.asrMatmulRows(w.matrix, &.{.{ .weights = w.buffer, .output = self.x }}, input, in_stride, h, n, .{ .accumulate = factor });
        try b.matmulTile(w.buffer, w.matrix, input, in_stride, self.t, h, n);
        if (factor == 1) try b.add(self.x, self.t, n * h) else try b.asrAddScaled(self.x, self.t, n * h, factor);
    }

    /// The tap buffer, created on first use: `(blocks + 2) × max_rows × hidden`.
    fn tapBuffer(self: *Plan) !Buffer {
        if (self.taps) |b| return b;
        const c = self.config();
        self.taps = try self.created((c.blocks + 2) * max_rows * c.hidden);
        return self.taps.?;
    }
};

/// One subsampling convolution's input on the device: rows `first …
/// count − 1` at the buffer's start (the pad row past them is implied), and
/// the next output row to compute.
const Queue = struct {
    buffer: Buffer,
    first: usize = 0,
    count: usize = 0,
    next: usize = 0,
};

/// One utterance stream on a `Plan` (nemotron-asr.md § Cache-aware
/// streaming): samples or frames in, tokens out as each chunk of rows
/// completes, with the CPU `Stream`'s and `Decoder`'s results. The host
/// moves its counters while recording, so after a failed push the stream
/// needs a `reset`.
pub const Stream = struct {
    alloc: std.mem.Allocator,
    plan: *Plan,
    lookahead: usize,
    prompt: usize,
    /// Shown each stage's rows after the command buffer that made them.
    observer: ?encoder.Observer = null,
    /// When set, the encoder's output rows (after the prompt block) are
    /// appended here.
    rows_out: ?*std.ArrayList(f32) = null,
    stats: Stats = .{},
    front: encoder.FrontEnd = .{},
    queues: [3]Queue,
    /// Subsampled rows awaiting a whole chunk, from row `row`.
    pending: Buffer,
    pending_rows: usize = 0,
    row: usize = 0,
    /// Per block, keys then values: `cache_rows` rows each, the `held`
    /// cached ones first.
    cache: Buffer,
    held: usize = 0,
    /// Per block, the last `kernel − 1` GLU rows, then a batch's.
    history: Buffer,
    /// `linear_1`'s bias plus the column of the prompt's one-hot.
    prompt_bias: []f32,
    /// The CPU prediction network: its state, `g`, and the rows decoded.
    decoder: rnnt.Decoder,
    /// Other prompts decoded over the same encoder rows, at most
    /// `max_prompts − 1`; their tokens stay in each `Alternate`.
    alternates: std.ArrayList(Alternate) = .empty,
    finished: bool = false,

    pub const Alternate = struct {
        prompt: usize,
        /// As `Stream.prompt_bias`, for this prompt.
        bias: []f32,
        decoder: rnnt.Decoder,
        tokens: std.ArrayList(rnnt.Token) = .empty,

        fn deinit(self: *Alternate, alloc: std.mem.Allocator) void {
            alloc.free(self.bias);
            self.decoder.deinit();
            self.tokens.deinit(alloc);
        }
    };

    /// The most rows before its chunk a query sees that the cache holds.
    const max_left = 56;
    const cache_rows = max_left + Backend.matmulPadded(max_rows);

    fn historyRows(c: encoder.Config) usize {
        return c.kernel - 1 + max_rows;
    }

    pub fn init(alloc: std.mem.Allocator, plan: *Plan, lookahead: usize, prompt: usize) !Stream {
        const c = plan.config();
        if (std.mem.indexOfScalar(usize, &encoder.lookaheads, lookahead) == null) return error.UnsupportedLookahead;
        if (prompt >= c.prompts) return error.UnsupportedPrompt;
        if (c.left > max_left) return error.UnsupportedConfig;
        const b = &plan.backend;
        var self: Stream = .{
            .alloc = alloc,
            .plan = plan,
            .lookahead = lookahead,
            .prompt = prompt,
            .queues = undefined,
            .pending = undefined,
            .cache = undefined,
            .history = undefined,
            .prompt_bias = undefined,
            .decoder = undefined,
        };
        var made: std.ArrayList(Buffer) = .empty;
        defer made.deinit(alloc);
        errdefer for (made.items) |buffer| b.release(buffer) catch {};
        const sizes = [_]usize{
            (2 + batch_frames) * c.columns(0),
            (2 + out_rows[0]) * c.columns(1) * c.channels,
            2 * c.columns(2) * c.channels + (out_rows[1] * c.columns(2) + Backend.matmul_tile) * c.channels,
            (encoder.max_lookahead + out_rows[2] + Backend.matmul_tile) * c.hidden,
            c.blocks * 2 * cache_rows * c.hidden,
            c.blocks * historyRows(c) * c.hidden,
        };
        var buffers: [sizes.len]Buffer = undefined;
        for (sizes, &buffers) |floats, *out| {
            out.* = try b.create(floats * 4);
            try made.append(alloc, out.*);
        }
        for (&self.queues, buffers[0..3]) |*q, buffer| q.* = .{ .buffer = buffer };
        self.pending = buffers[3];
        self.cache = buffers[4];
        self.history = buffers[5];
        self.prompt_bias = try alloc.alloc(f32, c.prompt_hidden);
        errdefer alloc.free(self.prompt_bias);
        try self.setPrompt(prompt);
        self.decoder = try rnnt.Decoder.init(alloc, plan.io, plan.transducer);
        made.clearRetainingCapacity();
        return self;
    }

    pub fn deinit(self: *Stream) void {
        const b = &self.plan.backend;
        for ([_]Buffer{ self.queues[0].buffer, self.queues[1].buffer, self.queues[2].buffer, self.pending, self.cache, self.history }) |buffer| b.release(buffer) catch {};
        self.alloc.free(self.prompt_bias);
        self.decoder.deinit();
        for (self.alternates.items) |*a| a.deinit(self.alloc);
        self.alternates.deinit(self.alloc);
        self.front.deinit(self.alloc);
        self.* = undefined;
    }

    /// The language prompt from the next chunk on; touches no encoder state.
    pub fn setPrompt(self: *Stream, prompt: usize) !void {
        if (prompt >= self.plan.config().prompts) return error.UnsupportedPrompt;
        self.prompt = prompt;
        self.promptBias(prompt, self.prompt_bias);
    }

    fn promptBias(self: *const Stream, prompt: usize, out: []f32) void {
        const c = self.plan.config();
        const w = self.plan.encoder.weights.prompt_in;
        for (out, 0..) |*v, o| v.* = w.bias.?[o] + w.weight[o * (c.hidden + c.prompts) + c.hidden + prompt];
    }

    /// The alternate prompts from the next chunk on, replacing any before;
    /// their decoders start at this stream's row with no tokens.
    pub fn setAlternates(self: *Stream, prompts: []const usize) !void {
        const c = self.plan.config();
        if (prompts.len >= max_prompts) return error.TooManyPrompts;
        for (prompts) |p| if (p >= c.prompts) return error.UnsupportedPrompt;
        for (self.alternates.items) |*a| a.deinit(self.alloc);
        self.alternates.clearRetainingCapacity();
        for (prompts) |p| {
            const bias = try self.alloc.alloc(f32, c.prompt_hidden);
            errdefer self.alloc.free(bias);
            self.promptBias(p, bias);
            var decoder = try rnnt.Decoder.init(self.alloc, self.plan.io, self.plan.transducer);
            errdefer decoder.deinit();
            decoder.row = self.decoder.row;
            try self.alternates.append(self.alloc, .{ .prompt = p, .bias = bias, .decoder = decoder });
        }
    }

    /// A new utterance: empty caches, the decoder from its start. The
    /// lookahead and prompt stay.
    pub fn reset(self: *Stream) !void {
        const c = self.plan.config();
        self.front.deinit(self.alloc);
        self.front = .{};
        for (&self.queues) |*q| q.* = .{ .buffer = q.buffer };
        self.pending_rows = 0;
        self.row = 0;
        self.held = 0;
        @memset(self.history.floats()[0 .. c.blocks * historyRows(c) * c.hidden], 0);
        self.finished = false;
        self.stats = .{};
        try self.decoder.reset();
        for (self.alternates.items) |*a| {
            try a.decoder.reset();
            a.tokens.clearRetainingCapacity();
        }
    }

    /// Appends 16 kHz samples; the tokens of the rows whose chunk they
    /// complete go to `tokens`.
    pub fn push(self: *Stream, pcm: []const f32, tokens: *std.ArrayList(rnnt.Token)) !void {
        if (self.finished) return error.StreamFinished;
        const bank = self.plan.encoder.bank orelse return error.NoFrontEnd;
        try self.frames(bank, try self.front.append(self.alloc, bank.spec, pcm), tokens);
    }

    /// Appends mel frames (`n × mels`). Not to be mixed with `push`.
    pub fn pushFeatures(self: *Stream, features: []const f32, tokens: *std.ArrayList(rnnt.Token)) !void {
        if (self.finished) return error.StreamFinished;
        const mels = self.plan.config().mels;
        if (features.len % mels != 0) return error.InvalidShape;
        var at: usize = 0;
        while (at < features.len) {
            const n = @min(features.len - at, batch_frames * mels);
            @memcpy(self.frameSlot(n / mels), features[at..][0..n]);
            self.queues[0].count += n / mels;
            at += n;
            try self.advance(false, tokens);
        }
    }

    /// Ends the stream as the whole-clip pass ends (nemotron-asr.md §
    /// Decisions: end of stream).
    pub fn finish(self: *Stream, tokens: *std.ArrayList(rnnt.Token)) !void {
        if (self.finished) return error.StreamFinished;
        if (self.plan.encoder.bank) |bank| try self.frames(bank, self.front.lastFrame(bank.spec), tokens);
        try self.advance(true, tokens);
        self.finished = true;
    }

    /// Where the next `n` mel frames go in the first queue (host memory,
    /// written while the device is idle).
    fn frameSlot(self: *Stream, n: usize) []f32 {
        const q = &self.queues[0];
        const mels = self.plan.config().mels;
        return q.buffer.floats()[(q.count - q.first) * mels ..][0 .. n * mels];
    }

    fn frames(self: *Stream, bank: *const mel.Bank, end: usize, tokens: *std.ArrayList(rnnt.Token)) !void {
        while (self.front.frame < end) {
            const n: usize = @min(end - self.front.frame, batch_frames);
            self.front.compute(bank, self.frameSlot(n));
            self.queues[0].count += n;
            try self.advance(false, tokens);
        }
        self.front.trim(bank.spec);
    }

    /// One command buffer: the subsampling as far as its inputs allow (at
    /// the end, with the pad row), every whole chunk (at the end, the rest)
    /// through the blocks, the prompt block, the projector and the first
    /// joint batch; then the greedy decoding of those rows.
    fn advance(self: *Stream, last: bool, tokens: *std.ArrayList(rnnt.Token)) !void {
        const plan = self.plan;
        const b = &plan.backend;
        const c = plan.config();
        const h = c.hidden;
        const started = std.Io.Clock.awake.now(plan.io);
        const taps: ?Buffer = if (self.observer != null) try plan.tapBuffer() else null;
        var sub_first: usize = 0;
        var sub_rows: usize = 0;
        var recorded = false;

        const chunk = self.lookahead + 1;
        // Bookkeeping first, so the command buffer is all recorded at once.
        @memcpy(plan.prompt_bias.floats()[0..c.prompt_hidden], self.prompt_bias);
        for (self.alternates.items, 1..) |a, i| @memcpy(plan.prompt_bias.floats()[i * c.prompt_hidden ..][0..c.prompt_hidden], a.bias);
        @memcpy(plan.g.floats()[0..self.decoder.g.len], self.decoder.g);
        try b.begin();
        errdefer b.discard();
        for (0..3) |stage| {
            const q = &self.queues[stage];
            if (q.count == 0) continue;
            const limit = if (last) q.count + 1 else q.count;
            const end = (limit - 1) / 2 + 1;
            if (end <= q.next) continue;
            const rows = end - q.next;
            recorded = true;
            const width_in = c.columns(stage);
            const width_out = c.columns(stage + 1);
            const in_channels: usize = if (stage == 0) 1 else c.channels;
            const in_row = width_in * in_channels;
            const out_row = width_out * c.channels;
            const conv: Backend.AsrConv = .{ .rows = rows, .next = q.next, .first = q.first, .count = q.count, .width_in = width_in, .width_out = width_out, .channels = c.channels, .in_channels = in_channels, .relu = stage == 0 };
            // Stage 0 writes the next queue directly; the others through their pointwise matmul.
            const target: Buffer = if (stage < 2) blk: {
                const nq = &self.queues[stage + 1];
                const at = (nq.count - nq.first) * out_row * 4;
                break :blk nq.buffer.slice(at, nq.buffer.len - at);
            } else plan.sub;
            if (stage == 0) {
                try b.asrConv2d(q.buffer, plan.conv_in.weight, plan.conv_in.bias, target, conv);
            } else {
                try b.asrConv2d(q.buffer, plan.depthwise[stage - 1].weight, plan.depthwise[stage - 1].bias, plan.dw, conv);
                const p = plan.pointwise[stage - 1];
                try b.matmulTile(p.buffer, p.matrix, plan.dw, c.channels, target, c.channels, rows * width_out);
                try b.asrBiasRelu(target, plan.pointwise_bias[stage - 1], c.channels, rows * width_out);
            }
            // Rows before 2·end − 2 are read by no later output.
            const keep_from = (2 * end) -| 2;
            if (keep_from > q.first) {
                const drop = @min(keep_from - q.first, q.count - q.first);
                try b.asrShift(q.buffer, in_row, q.count - q.first - drop, drop, 1, 0);
                q.first += drop;
            }
            q.next = end;
            if (stage < 2) {
                self.queues[stage + 1].count += rows;
            } else {
                const at = self.pending_rows * h * 4;
                const out = self.pending.slice(at, self.pending.len - at);
                try plan.product(plan.linear, plan.sub, c.subWidth(), out, h, rows, false);
                try b.addBiasRows(out, plan.linear_bias, h, rows, h);
                if (taps) |tap| try b.copy(tap, out, rows * h);
                sub_first = self.row + self.pending_rows;
                sub_rows = rows;
                self.pending_rows += rows;
            }
        }
        const whole = if (last) self.pending_rows else self.pending_rows / chunk * chunk;
        if (whole > max_rows) return error.InvalidShape;
        const first = self.row;
        if (whole > 0) {
            try b.copy(plan.x, self.pending, whole * h);
            try b.asrShift(self.pending, h, self.pending_rows - whole, whole, 1, 0);
            try self.recordBlocks(whole, taps);
            try self.recordJoint(0, 0, @min(joint_rows, whole));
            recorded = true;
        }
        if (!recorded) return b.discard();
        try b.commit();
        self.pending_rows -= whole;
        self.row += whole;
        self.held = @min(c.left, self.held + whole);
        if (whole == 0 and sub_rows == 0) return;
        self.stats.encode_ns += lapse(plan.io, started);
        if (whole > 0) {
            self.stats.chunks += 1;
            self.stats.rows += whole;
        }

        if (self.observer) |o| {
            const tap = taps.?;
            const stride = max_rows * h;
            if (sub_rows > 0) o.record(o.context, .subsampling, sub_first, tap.floats()[0 .. sub_rows * h]);
            if (whole > 0) {
                for (0..c.blocks) |il| o.record(o.context, .{ .block = il }, first, tap.floats()[(1 + il) * stride ..][0 .. whole * h]);
                o.record(o.context, .prompt, first, tap.floats()[(1 + c.blocks) * stride ..][0 .. whole * h]);
            }
        }
        if (whole == 0) return;
        if (self.rows_out) |list| try list.appendSlice(self.alloc, plan.enc.floats()[0 .. whole * h]);
        try self.decode(0, &self.decoder, whole, @min(joint_rows, whole), tokens);
        for (self.alternates.items, 1..) |*a, i| try self.decode(i, &a.decoder, whole, 0, &a.tokens);
    }

    /// `n` rows from `plan.x` through every block, the prompt block and the
    /// projector into `plan.e`; the caches move past them.
    fn recordBlocks(self: *Stream, n: usize, taps: ?Buffer) !void {
        const plan = self.plan;
        const b = &plan.backend;
        const c = plan.config();
        const h = c.hidden;
        const t = plan.transducer.config;
        const spec: Backend.Norm = .{ .rows = n, .width = h, .in_stride = h, .out_stride = h, .eps = encoder.epsilon };
        const chunk = self.lookahead + 1;
        const plane = cache_rows * h * 4;
        const history = historyRows(c) * h * 4;
        const relk_rows = c.positions() * h * 4;
        for (plan.blocks, 0..) |blk, il| {
            try self.feedForward(blk.norm_ff1, blk.ff1_in, blk.ff1_out, n);
            // Attention: this chunk's keys and values after the cached ones.
            try b.layerNorm(plan.x, blk.norm_attention.weight, blk.norm_attention.bias, plan.t, spec);
            const keys = self.cache.slice(2 * il * plane, plane);
            const values = self.cache.slice((2 * il + 1) * plane, plane);
            const held = self.held * h * 4;
            const new_keys = keys.slice(held, plane - held);
            const new_values = values.slice(held, plane - held);
            if (n <= Backend.asr_matmul_rows_max) {
                try b.asrMatmulRows(blk.q.matrix, &.{ .{ .weights = blk.q.buffer, .output = plan.q }, .{ .weights = blk.k.buffer, .output = new_keys }, .{ .weights = blk.v.buffer, .output = new_values } }, plan.t, h, h, n, .store);
            } else {
                try plan.product(blk.q, plan.t, h, plan.q, h, n, false);
                try plan.product(blk.k, plan.t, h, new_keys, h, n, false);
                try plan.product(blk.v, plan.t, h, new_values, h, n, false);
            }
            try b.asrAttention(plan.q, keys, values, plan.relk.slice(il * relk_rows, relk_rows), blk.bias_u, blk.bias_v, plan.attn, .{
                .rows = n,
                .held = self.held,
                .first = self.row,
                .chunk = chunk,
                .reach = c.left / chunk,
                .width = h,
                .offset = encoder.max_lookahead,
                .positions = c.positions(),
            });
            try plan.residual(blk.o, plan.attn, h, n, 1);
            // The convolution module over the held GLU rows and these.
            try b.layerNorm(plan.x, blk.norm_conv.weight, blk.norm_conv.bias, plan.t, spec);
            const gated = self.history.slice(il * history, history);
            const new_gated = gated.slice((c.kernel - 1) * h * 4, history - (c.kernel - 1) * h * 4);
            if (n <= Backend.asr_matmul_rows_max) {
                try b.asrMatmulRows(blk.pointwise1.matrix, &.{.{ .weights = blk.pointwise1.buffer, .output = new_gated }}, plan.t, h, h, n, .glu);
            } else {
                try plan.product(blk.pointwise1, plan.t, h, plan.wide, 2 * h, n, false);
                try b.asrGlu(plan.wide, new_gated, h, n);
            }
            try b.asrDepthwise(gated, blk.depthwise, plan.q, h, n, c.kernel);
            try b.layerNorm(plan.q, blk.conv_norm.weight, blk.conv_norm.bias, plan.q, spec);
            try b.silu(plan.q, n * h);
            try plan.residual(blk.pointwise2, plan.q, h, n, 1);
            try self.feedForward(blk.norm_ff2, blk.ff2_in, blk.ff2_out, n);
            try b.layerNorm(plan.x, blk.norm_out.weight, blk.norm_out.bias, plan.x, spec);
            if (taps) |tap| try b.copy(tap.slice((1 + il) * max_rows * h * 4, max_rows * h * 4), plan.x, n * h);
        }
        // Keys and values of the last `left` rows stay for the next chunk.
        const total = self.held + n;
        const keep = @min(c.left, total);
        try b.asrShift(self.cache, h, keep, total - keep, 2 * c.blocks, cache_rows * h);
        // And the last `kernel − 1` GLU rows.
        try b.asrShift(self.history, h, c.kernel - 1, n, c.blocks, historyRows(c) * h);
        // The prompt block per prompt, the alternates' first so that
        // `plan.enc` ends holding this stream's own rows.
        var i = self.alternates.items.len + 1;
        while (i > 0) {
            i -= 1;
            // linear_1 over the row, its one-hot column in the bias.
            const bias = plan.prompt_bias.slice(i * c.prompt_hidden * 4, c.prompt_hidden * 4);
            try plan.product(plan.prompt_in, plan.x, h, plan.inner, c.prompt_hidden, n, false);
            try b.asrBiasRelu(plan.inner, bias, c.prompt_hidden, n);
            try plan.product(plan.prompt_out, plan.inner, c.prompt_hidden, plan.enc, h, n, false);
            try b.addBiasRows(plan.enc, plan.prompt_out_bias, h, n, h);
            const e = plan.rowsOf(i);
            try plan.product(plan.projector, plan.enc, h, e, t.hidden, n, false);
            try b.addBiasRows(e, plan.projector_bias, t.hidden, n, t.hidden);
        }
        if (taps) |tap| try b.copy(tap.slice((1 + c.blocks) * max_rows * h * 4, max_rows * h * 4), plan.enc, n * h);
    }

    /// x += ½ · W_out silu(W_in LN(x)) over `n` rows.
    fn feedForward(self: *Stream, n_: Norm, w_in: Weight, w_out: Weight, n: usize) !void {
        const plan = self.plan;
        const b = &plan.backend;
        const c = plan.config();
        const h = c.hidden;
        try b.layerNorm(plan.x, n_.weight, n_.bias, plan.t, .{ .rows = n, .width = h, .in_stride = h, .out_stride = h, .eps = encoder.epsilon });
        try plan.product(w_in, plan.t, h, plan.wide, c.ffn, n, true);
        try plan.residual(w_out, plan.wide, c.ffn, n, 0.5);
    }

    /// Records the joint of rows `first … first + n − 1` of prompt
    /// `which`'s projected rows under the `g` in `plan.g`: `n` rows of
    /// logits in `plan.logits`.
    fn recordJoint(self: *Stream, which: usize, first: usize, n: usize) !void {
        const plan = self.plan;
        const b = &plan.backend;
        const t = plan.transducer.config;
        try b.asrJoint(plan.rowsOf(which).slice(first * t.hidden * 4, n * t.hidden * 4), plan.g, plan.jin, t.hidden, n);
        try plan.product(plan.head, plan.jin, t.hidden, plan.logits, t.vocab, n, false);
        try b.addBiasRows(plan.logits, plan.head_bias, t.vocab, n, t.vocab);
    }

    /// Greedy decoding by `decoder` of prompt `which`'s `n` projected rows,
    /// whose first `ready` rows of logits are already in `plan.logits`
    /// (`Decoder.decodeRow`'s loop): the logits of one `g` serve every row
    /// until a token changes it.
    fn decode(self: *Stream, which: usize, decoder: *rnnt.Decoder, n: usize, ready: usize, tokens: *std.ArrayList(rnnt.Token)) !void {
        const plan = self.plan;
        const t = plan.transducer.config;
        var lo: usize = 0;
        var count = ready;
        var i: usize = 0;
        var symbols: usize = 0;
        while (i < n) {
            if (i < lo or i >= lo + count) {
                const started = std.Io.Clock.awake.now(plan.io);
                lo = i;
                count = @min(joint_rows, n - i);
                @memcpy(plan.g.floats()[0..t.hidden], decoder.g);
                try plan.backend.begin();
                errdefer plan.backend.discard();
                try self.recordJoint(which, lo, count);
                try plan.backend.commit();
                self.stats.joint_batches += 1;
                self.stats.joint_ns += lapse(plan.io, started);
            }
            const logits = plan.logits.floats()[(i - lo) * t.vocab ..][0..t.vocab];
            const k = decoder.pick(logits);
            if (decoder.observer) |o| o.record(o.context, .{ .row = decoder.row, .logits = logits, .token = k });
            if (k == t.blank) {
                i += 1;
                decoder.row += 1;
                symbols = 0;
                continue;
            }
            try tokens.append(self.alloc, .{ .id = k, .row = @intCast(decoder.row) });
            const started = std.Io.Clock.awake.now(plan.io);
            try decoder.predict(k);
            self.stats.lstm_steps += 1;
            self.stats.lstm_ns += lapse(plan.io, started);
            // A new `g`: the logits from this row on are stale.
            count = 0;
            symbols += 1;
            if (symbols == t.max_symbols) {
                i += 1;
                decoder.row += 1;
                symbols = 0;
            }
        }
    }
};

fn lapse(io: std.Io, since: std.Io.Timestamp) u64 {
    return @intCast(since.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds());
}
