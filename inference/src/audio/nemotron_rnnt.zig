//! Nemotron 3.5 ASR's RNN-T transducer, the CPU reference
//! (docs/models/nemotron-asr.md § RNN-T decoding): `encoder_projector` takes
//! each encoder row (after the prompt block) to the joint's width; the
//! prediction network (embedding, two LSTM layers, a projection) turns the
//! last emitted token into `g`; the joint scores `head(relu(e + g))`, and
//! greedy decoding emits the argmax until the blank or the per-row symbol
//! cap moves it to the next row. A row is decoded in full when it arrives,
//! so a `Decoder` fed rows chunk by chunk emits the whole clip's tokens.
//! Written from transformers' `Nemotron3_5AsrForRNNT` and
//! `ParakeetRNNTGenerationMixin`.
const std = @import("std");
const builtin = @import("builtin");
const safetensors = @import("../formats/safetensors.zig");
const cpu = @import("../backends/cpu/root.zig");
const mel = @import("mel.zig");
const boost_mod = @import("boost.zig");

const dense = cpu.dense;

/// Seconds per encoder row: the hop times the subsampling, 160 / 16000 · 8.
pub const seconds_per_row: f64 = @as(f64, mel.hop) / mel.sample_rate * 8;

pub const Config = struct {
    /// The encoder rows' width.
    encoder: usize,
    /// The prediction network's and the joint's width.
    hidden: usize,
    /// LSTM layers.
    layers: usize,
    /// Joint outputs, the blank included.
    vocab: usize,
    blank: u32,
    /// Symbols one row may emit before decoding moves on.
    max_symbols: usize,

    pub const nemotron: Config = .{ .encoder = 1024, .hidden = 640, .layers = 2, .vocab = 13088, .blank = 13087, .max_symbols = 10 };
};

pub const ConfigError = error{ InvalidConfig, UnsupportedConfig, OutOfMemory };

/// Reads the transducer's part of `config.json` (`Nemotron3_5AsrForRNNT`).
pub fn parseConfig(gpa: std.mem.Allocator, text: []const u8) ConfigError!Config {
    const Json = struct {
        encoder_config: struct { hidden_size: u32 },
        decoder_hidden_size: u32,
        num_decoder_layers: u32,
        vocab_size: u32,
        blank_token_id: u32,
        max_symbols_per_step: u32,
        hidden_act: []const u8,
    };
    const parsed = std.json.parseFromSlice(Json, gpa, text, .{ .ignore_unknown_fields = true }) catch |e| return switch (e) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidConfig,
    };
    defer parsed.deinit();
    const j = parsed.value;
    if (!std.mem.eql(u8, j.hidden_act, "relu")) return error.UnsupportedConfig;
    if (j.encoder_config.hidden_size == 0 or j.encoder_config.hidden_size > 16384 or j.decoder_hidden_size == 0 or
        j.decoder_hidden_size > 16384 or j.num_decoder_layers == 0 or j.num_decoder_layers > 16 or j.vocab_size < 2 or
        j.vocab_size > 1 << 20 or j.blank_token_id >= j.vocab_size or j.max_symbols_per_step == 0 or j.max_symbols_per_step > 1000)
        return error.InvalidConfig;
    return .{
        .encoder = j.encoder_config.hidden_size,
        .hidden = j.decoder_hidden_size,
        .layers = j.num_decoder_layers,
        .vocab = j.vocab_size,
        .blank = j.blank_token_id,
        .max_symbols = j.max_symbols_per_step,
    };
}

pub const Linear = struct {
    /// `[outputs][inputs]`, PyTorch's layout.
    weight: []const f32,
    bias: []const f32,
};

/// One `nn.LSTM` layer: `[4 · hidden][hidden]` matrices, gates stacked
/// `[i | f | g | o]`.
pub const Layer = struct {
    input: Linear,
    recurrent: Linear,
};

pub const Weights = struct {
    config: Config,
    /// `encoder_projector`: encoder width → hidden.
    projector: Linear,
    /// `[vocab][hidden]`; the blank's row is the padding row.
    embedding: []const f32,
    layers: []const Layer,
    /// `decoder_projector`: hidden → hidden.
    output: Linear,
    /// `joint.head`: hidden → vocab.
    head: Linear,
};

/// The bound tensors: views into the checkpoint's mapping, or copies where
/// one is unaligned. Owns `layers` and the copies; borrows the checkpoint.
pub const Binding = struct {
    arena: std.heap.ArenaAllocator,
    weights: Weights,

    pub fn deinit(self: *Binding) void {
        self.arena.deinit();
        self.* = undefined;
    }
};

pub const BindError = error{ MissingTensor, TensorShapeMismatch, UnsupportedDtype, OutOfMemory, TensorOutOfBounds };

/// Binds `encoder_projector.*`, `decoder.*` and `joint.*` at `config`'s
/// shapes, F32 only, as published.
pub fn bind(gpa: std.mem.Allocator, checkpoint: *const safetensors.Checkpoint, config: Config) BindError!Binding {
    comptime std.debug.assert(builtin.cpu.arch.endian() == .little);
    var binding: Binding = .{ .arena = .init(gpa), .weights = undefined };
    errdefer binding.arena.deinit();
    const arena = binding.arena.allocator();
    const c = config;
    const h = c.hidden;
    var w: Weights = undefined;
    w.config = c;
    w.projector = .{ .weight = try get(checkpoint, arena, "encoder_projector.weight", .{}, &.{ h, c.encoder }), .bias = try get(checkpoint, arena, "encoder_projector.bias", .{}, &.{h}) };
    w.embedding = try get(checkpoint, arena, "decoder.embedding.weight", .{}, &.{ c.vocab, h });
    const layers = try arena.alloc(Layer, c.layers);
    for (layers, 0..) |*l, i| l.* = .{
        .input = .{ .weight = try get(checkpoint, arena, "decoder.lstm.weight_ih_l{d}", .{i}, &.{ 4 * h, h }), .bias = try get(checkpoint, arena, "decoder.lstm.bias_ih_l{d}", .{i}, &.{4 * h}) },
        .recurrent = .{ .weight = try get(checkpoint, arena, "decoder.lstm.weight_hh_l{d}", .{i}, &.{ 4 * h, h }), .bias = try get(checkpoint, arena, "decoder.lstm.bias_hh_l{d}", .{i}, &.{4 * h}) },
    };
    w.layers = layers;
    w.output = .{ .weight = try get(checkpoint, arena, "decoder.decoder_projector.weight", .{}, &.{ h, h }), .bias = try get(checkpoint, arena, "decoder.decoder_projector.bias", .{}, &.{h}) };
    w.head = .{ .weight = try get(checkpoint, arena, "joint.head.weight", .{}, &.{ c.vocab, h }), .bias = try get(checkpoint, arena, "joint.head.bias", .{}, &.{c.vocab}) };
    binding.weights = w;
    return binding;
}

fn get(checkpoint: *const safetensors.Checkpoint, arena: std.mem.Allocator, comptime format: []const u8, args: anytype, shape: []const usize) BindError![]const f32 {
    var buffer: [128]u8 = undefined;
    const name = std.fmt.bufPrint(&buffer, format, args) catch return error.MissingTensor;
    const ref = checkpoint.get(name) orelse return error.MissingTensor;
    if (ref.tensor.shape.len != shape.len) return error.TensorShapeMismatch;
    for (ref.tensor.shape, shape) |a, want| if (a != want) return error.TensorShapeMismatch;
    if (ref.tensor.dtype != .f32) return error.UnsupportedDtype;
    const count: usize = @intCast(ref.tensor.elements);
    if (@intFromPtr(ref.bytes.ptr) % @alignOf(f32) == 0)
        return @as([*]const f32, @ptrCast(@alignCast(ref.bytes.ptr)))[0..count];
    const copy = try arena.alloc(f32, count);
    ref.decode(0, copy) catch |e| return switch (e) {
        error.UnsupportedDtype => error.UnsupportedDtype,
        error.TensorOutOfBounds => error.TensorOutOfBounds,
    };
    return copy;
}

/// An emitted token and the encoder row it was emitted at (start time
/// `row · seconds_per_row`).
pub const Token = struct { id: u32, row: u32 };

/// One greedy step as an observer sees it: the row, the joint's logits,
/// and the token chosen (the blank included).
pub const Step = struct { row: usize, logits: []const f32, token: u32 };

pub const Observer = struct {
    context: *anyopaque,
    record: *const fn (context: *anyopaque, step: Step) void,
};

/// Greedy decoding state: the LSTM's `h` and `c`, the prediction `g` of the
/// last emitted token (or of the start blank), and the rows consumed.
/// Nothing resets between pushes; `reset` starts a new utterance.
pub const Decoder = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    weights: *const Weights,
    observer: ?Observer = null,
    /// Phrases to lean toward (borrowed; outlives the decoder's use of it),
    /// and where the emitted pieces stand in them.
    boost: ?*const boost_mod.Boost = null,
    boosting: boost_mod.State = .{},
    /// `layers × hidden` each.
    h: []f32,
    c: []f32,
    g: []f32,
    /// Encoder rows decoded so far.
    row: usize = 0,
    /// The greedy path's log probability since the last reset: every step's
    /// chosen output, blanks included, under the plain logits.
    score: f64 = 0,
    // Scratch, one step's worth.
    gates: []f32,
    recurrent: []f32,
    joint: []f32,
    logits: []f32,
    /// Every slice above, in one allocation.
    storage: []f32,

    pub fn init(alloc: std.mem.Allocator, io: std.Io, weights: *const Weights) !Decoder {
        const c = weights.config;
        const storage = try alloc.alloc(f32, 2 * c.layers * c.hidden + 10 * c.hidden + c.vocab);
        errdefer alloc.free(storage);
        var at: usize = 0;
        var self: Decoder = .{
            .alloc = alloc,
            .io = io,
            .weights = weights,
            .h = take(storage, &at, c.layers * c.hidden),
            .c = take(storage, &at, c.layers * c.hidden),
            .g = take(storage, &at, c.hidden),
            .gates = take(storage, &at, 4 * c.hidden),
            .recurrent = take(storage, &at, 4 * c.hidden),
            .joint = take(storage, &at, c.hidden),
            .logits = take(storage, &at, c.vocab),
            .storage = storage,
        };
        std.debug.assert(at == storage.len);
        try self.reset();
        return self;
    }

    pub fn deinit(self: *Decoder) void {
        self.alloc.free(self.storage);
        self.* = undefined;
    }

    /// A new utterance: zero state, then the start blank through the
    /// prediction network, as `generate`'s first step runs it.
    pub fn reset(self: *Decoder) !void {
        @memset(self.h, 0);
        @memset(self.c, 0);
        self.row = 0;
        self.score = 0;
        self.boosting.reset();
        try self.predict(self.weights.config.blank);
    }

    /// Decodes `rows` (`n × encoder`, the encoder's output after the prompt
    /// block) in order, appending each emitted token to `out`.
    pub fn push(self: *Decoder, rows: []const f32, out: *std.ArrayList(Token)) !void {
        const c = self.weights.config;
        if (rows.len % c.encoder != 0) return error.InvalidShape;
        const n = rows.len / c.encoder;
        if (n == 0) return;
        const e = try self.alloc.alloc(f32, n * c.hidden);
        defer self.alloc.free(e);
        const p = self.weights.projector;
        try dense.matmul(self.io, .{ .rows = n, .inner = c.encoder, .outputs = c.hidden, .x = rows, .w = p.weight, .bias = p.bias }, e);
        for (0..n) |i| try self.decodeRow(e[i * c.hidden ..][0..c.hidden], out);
    }

    /// Greedy steps on one projected row until the blank or the cap.
    fn decodeRow(self: *Decoder, e: []const f32, out: *std.ArrayList(Token)) !void {
        const c = self.weights.config;
        var symbols: usize = 0;
        while (symbols < c.max_symbols) {
            const k = try self.argmaxJoint(e);
            if (k == c.blank) break;
            try out.append(self.alloc, .{ .id = k, .row = @intCast(self.row) });
            try self.predict(k);
            symbols += 1;
        }
        self.row += 1;
    }

    /// `head · relu(e + g) + b`; the first largest logit's index.
    fn argmaxJoint(self: *Decoder, e: []const f32) !u32 {
        const c = self.weights.config;
        jointInput(e, self.g, self.joint);
        try dense.matmul(self.io, .{ .rows = 1, .inner = c.hidden, .outputs = c.vocab, .x = self.joint, .w = self.weights.head.weight, .bias = self.weights.head.bias }, self.logits);
        const k = self.pick(self.logits);
        if (self.observer) |o| o.record(o.context, .{ .row = self.row, .logits = self.logits, .token = k });
        return k;
    }

    /// The greedy choice from one row's joint logits: the argmax, or a
    /// phrase's piece within the bonus of it.
    pub fn choose(self: *const Decoder, logits: []const f32) u32 {
        const k: u32 = @intCast(argmax(logits));
        const b = self.boost orelse return k;
        return self.boosting.choose(b, logits, k);
    }

    /// `choose`, with the choice's log probability added to `score`.
    pub fn pick(self: *Decoder, logits: []const f32) u32 {
        const k = self.choose(logits);
        self.score += logProbability(logits, k);
        return k;
    }

    /// The prediction network on `token`: its embedding through every LSTM
    /// layer (state updated), then `decoder_projector` into `g`. An emitted
    /// token also moves the phrase state.
    pub fn predict(self: *Decoder, token: u32) !void {
        const c = self.weights.config;
        if (token != c.blank) if (self.boost) |b| self.boosting.advance(b, token);
        const hd = c.hidden;
        var x: []const f32 = self.weights.embedding[@as(usize, token) * hd ..][0..hd];
        for (self.weights.layers, 0..) |layer, l| {
            const h = self.h[l * hd ..][0..hd];
            const cell = self.c[l * hd ..][0..hd];
            try dense.matmul(self.io, .{ .rows = 1, .inner = hd, .outputs = 4 * hd, .x = x, .w = layer.input.weight, .bias = layer.input.bias }, self.gates);
            try dense.matmul(self.io, .{ .rows = 1, .inner = hd, .outputs = 4 * hd, .x = h, .w = layer.recurrent.weight, .bias = layer.recurrent.bias }, self.recurrent);
            lstmCell(self.gates, self.recurrent, h, cell);
            x = h;
        }
        try dense.matmul(self.io, .{ .rows = 1, .inner = hd, .outputs = hd, .x = x, .w = self.weights.output.weight, .bias = self.weights.output.bias }, self.g);
    }
};

/// `log softmax(logits)[k]`.
pub fn logProbability(logits: []const f32, k: usize) f64 {
    var top: f32 = logits[0];
    for (logits) |v| top = @max(top, v);
    var sum: f64 = 0;
    for (logits) |v| sum += @exp(@as(f64, v - top));
    return @as(f64, logits[k] - top) - @log(sum);
}

/// The joint's input, `relu(e + g)`, one row.
pub fn jointInput(e: []const f32, g: []const f32, out: []f32) void {
    for (out, e, g) |*j, a, b| j.* = @max(a + b, 0);
}

/// One LSTM step from the two pre-activations `[i | f | g | o]`:
/// `c ← σ(f)·c + σ(i)·tanh(g)`, `h ← σ(o)·tanh(c)`.
pub fn lstmCell(input: []const f32, recurrent: []const f32, h: []f32, c: []f32) void {
    const n = h.len;
    for (0..n) |j| {
        const i = cpu.sigmoid(input[j] + recurrent[j]);
        const f = cpu.sigmoid(input[n + j] + recurrent[n + j]);
        const g = std.math.tanh(input[2 * n + j] + recurrent[2 * n + j]);
        const o = cpu.sigmoid(input[3 * n + j] + recurrent[3 * n + j]);
        c[j] = f * c[j] + i * g;
        h[j] = o * std.math.tanh(c[j]);
    }
}

fn take(storage: []f32, cursor: *usize, n: usize) []f32 {
    defer cursor.* += n;
    return storage[cursor.*..][0..n];
}

/// The first index of the largest value, as `torch.argmax` picks.
pub fn argmax(values: []const f32) usize {
    var best: usize = 0;
    for (values, 0..) |v, i| if (v > values[best]) {
        best = i;
    };
    return best;
}

/// The locale of a language tag piece such as `<de-DE>`, or null.
pub fn languageTag(piece: []const u8) ?[]const u8 {
    if (piece.len != 7 or piece[0] != '<' or piece[3] != '-' or piece[6] != '>') return null;
    for ([_]usize{ 1, 2 }) |i| if (!std.ascii.isLower(piece[i])) return null;
    for ([_]usize{ 4, 5 }) |i| if (!std.ascii.isUpper(piece[i])) return null;
    return piece[1..6];
}

// ---- tests: synthetic weights at a small width, no model file ----

const tiny: Config = .{ .encoder = 6, .hidden = 4, .layers = 2, .vocab = 7, .blank = 6, .max_symbols = 10 };

const Synthetic = struct {
    storage: []f32,
    layers: []Layer,
    weights: Weights,

    fn init(gpa: std.mem.Allocator, c: Config, seed: u64) !Synthetic {
        var prng = std.Random.DefaultPrng.init(seed);
        const r = prng.random();
        const h = c.hidden;
        const total = h * (c.encoder + 1) + c.vocab * h + c.layers * 2 * (4 * h * h + 4 * h) + h * (h + 1) + c.vocab * (h + 1);
        const storage = try gpa.alloc(f32, total);
        errdefer gpa.free(storage);
        for (storage) |*v| v.* = r.float(f32) * 2 - 1;
        const layers = try gpa.alloc(Layer, c.layers);
        var at: usize = 0;
        var w: Weights = undefined;
        w.config = c;
        w.projector = .{ .weight = take(storage, &at, h * c.encoder), .bias = take(storage, &at, h) };
        const embedding = take(storage, &at, c.vocab * h);
        @memset(embedding[c.blank * h ..][0..h], 0);
        w.embedding = embedding;
        for (layers) |*l| l.* = .{
            .input = .{ .weight = take(storage, &at, 4 * h * h), .bias = take(storage, &at, 4 * h) },
            .recurrent = .{ .weight = take(storage, &at, 4 * h * h), .bias = take(storage, &at, 4 * h) },
        };
        w.layers = layers;
        w.output = .{ .weight = take(storage, &at, h * h), .bias = take(storage, &at, h) };
        w.head = .{ .weight = take(storage, &at, c.vocab * h), .bias = take(storage, &at, c.vocab) };
        std.debug.assert(at == total);
        return .{ .storage = storage, .layers = layers, .weights = w };
    }

    fn deinit(self: *Synthetic, gpa: std.mem.Allocator) void {
        gpa.free(self.layers);
        gpa.free(self.storage);
    }
};

test "LSTM gates are stacked [i | f | g | o]" {
    // Saturated gates make each one's role visible in one unit.
    var h = [_]f32{0};
    var c = [_]f32{0.5};
    const zero = [_]f32{ 0, 0, 0, 0 };
    // i open, f shut, g = tanh(20) ≈ 1, o open: c = 1, h = tanh(1).
    lstmCell(&.{ 20, -20, 20, 20 }, &zero, &h, &c);
    try std.testing.expectApproxEqAbs(@as(f32, 1), c[0], 1e-6);
    try std.testing.expectApproxEqAbs(std.math.tanh(@as(f32, 1)), h[0], 1e-6);
    // f open, i shut: c kept; o shut: h = 0.
    lstmCell(&.{ -20, 20, 20, -20 }, &zero, &h, &c);
    try std.testing.expectApproxEqAbs(@as(f32, 1), c[0], 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, 0), h[0], 1e-6);
    // The two pre-activations add: g = tanh(−10 + −10) ≈ −1 with i open, f shut.
    lstmCell(&.{ 10, -10, -10, 20 }, &.{ 10, -10, -10, 0 }, &h, &c);
    try std.testing.expectApproxEqAbs(@as(f32, -1), c[0], 1e-6);
}

/// Head weights that make the joint's argmax `token` whatever `e + g` is:
/// that row's bias alone dominates.
fn force(w: *Weights, storage_head: []f32, bias: []f32, token: u32) void {
    @memset(storage_head, 0);
    @memset(bias, 0);
    bias[token] = 1;
    w.head = .{ .weight = storage_head, .bias = bias };
}

test "greedy decoding: the blank moves on, ten symbols at most per row" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var synthetic = try Synthetic.init(gpa, tiny, 3);
    defer synthetic.deinit(gpa);
    var head: [tiny.vocab * tiny.hidden]f32 = undefined;
    var bias: [tiny.vocab]f32 = undefined;
    const rows: [3 * tiny.encoder]f32 = @splat(0.5);
    var tokens: std.ArrayList(Token) = .empty;
    defer tokens.deinit(gpa);

    // Always the blank: no tokens, every row consumed.
    force(&synthetic.weights, &head, &bias, tiny.blank);
    var decoder = try Decoder.init(gpa, io, &synthetic.weights);
    defer decoder.deinit();
    try decoder.push(&rows, &tokens);
    try std.testing.expectEqual(@as(usize, 0), tokens.items.len);
    try std.testing.expectEqual(@as(usize, 3), decoder.row);

    // Never the blank: the cap emits `max_symbols` per row, then moves on.
    force(&synthetic.weights, &head, &bias, 2);
    try decoder.reset();
    try decoder.push(&rows, &tokens);
    try std.testing.expectEqual(@as(usize, 3 * tiny.max_symbols), tokens.items.len);
    for (tokens.items, 0..) |t, i| {
        try std.testing.expectEqual(@as(u32, 2), t.id);
        try std.testing.expectEqual(@as(u32, @intCast(i / tiny.max_symbols)), t.row);
    }
    try std.testing.expectError(error.InvalidShape, decoder.push(rows[0..5], &tokens));
}

test "a decoder fed rows in pieces emits what one push of them does; reset starts over" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var synthetic = try Synthetic.init(gpa, tiny, 11);
    defer synthetic.deinit(gpa);
    // Make the blank likely enough that both branches run.
    const bias: []f32 = @constCast(synthetic.weights.head.bias);
    bias[tiny.blank] += 1.5;
    var prng = std.Random.DefaultPrng.init(12);
    var rows: [40 * tiny.encoder]f32 = undefined;
    for (&rows) |*v| v.* = prng.random().float(f32) * 4 - 2;

    var whole: std.ArrayList(Token) = .empty;
    defer whole.deinit(gpa);
    var decoder = try Decoder.init(gpa, io, &synthetic.weights);
    defer decoder.deinit();
    try decoder.push(&rows, &whole);
    // Some rows emit and some end on the blank before the cap.
    try std.testing.expect(whole.items.len > 0 and whole.items.len < 40 * tiny.max_symbols);

    var pieces: std.ArrayList(Token) = .empty;
    defer pieces.deinit(gpa);
    try decoder.reset();
    var at: usize = 0;
    for ([_]usize{ 1, 3, 4, 7, 25 }) |n| {
        try decoder.push(rows[at * tiny.encoder ..][0 .. n * tiny.encoder], &pieces);
        at += n;
    }
    try std.testing.expectEqual(@as(usize, 40), at);
    try std.testing.expectEqualSlices(Token, whole.items, pieces.items);
}

test "the start state is the blank through the network from zeros" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var synthetic = try Synthetic.init(gpa, tiny, 5);
    defer synthetic.deinit(gpa);
    var decoder = try Decoder.init(gpa, io, &synthetic.weights);
    defer decoder.deinit();
    // By hand: x = 0, h = c = 0 per layer, so each layer's gates are its two biases.
    const w = synthetic.weights;
    const hd = tiny.hidden;
    var x: [hd]f32 = @splat(0);
    for (w.layers) |l| {
        var h: [hd]f32 = @splat(0);
        var c: [hd]f32 = @splat(0);
        var a: [4 * hd]f32 = undefined;
        var b: [4 * hd]f32 = undefined;
        for (0..4 * hd) |j| {
            var s: f32 = l.input.bias[j];
            for (0..hd) |k| s += l.input.weight[j * hd + k] * x[k];
            a[j] = s;
            b[j] = l.recurrent.bias[j];
        }
        lstmCell(&a, &b, &h, &c);
        x = h;
    }
    for (0..hd) |j| {
        var s: f32 = w.output.bias[j];
        for (0..hd) |k| s += w.output.weight[j * hd + k] * x[k];
        try std.testing.expectApproxEqAbs(s, decoder.g[j], 1e-5);
    }
}

test "language tags: <ll-CC> pieces only" {
    try std.testing.expectEqualStrings("de-DE", languageTag("<de-DE>").?);
    try std.testing.expectEqualStrings("en-US", languageTag("<en-US>").?);
    for ([_][]const u8{ "<unk>", "<pad>", "<blank>", "<DE-de>", "de-DE", "<de_DE>", "<deu-DE>" }) |p|
        try std.testing.expectEqual(@as(?[]const u8, null), languageTag(p));
}

test "config.json: the transducer's values" {
    const gpa = std.testing.allocator;
    const text =
        \\{"encoder_config":{"hidden_size":1024},"decoder_hidden_size":640,"num_decoder_layers":2,"vocab_size":13088,
        \\ "blank_token_id":13087,"max_symbols_per_step":10,"hidden_act":"relu","pad_token_id":0}
    ;
    try std.testing.expectEqual(Config.nemotron, try parseConfig(gpa, text));
    const relu_free = try std.mem.replaceOwned(u8, gpa, text, "\"relu\"", "\"tanh\"");
    defer gpa.free(relu_free);
    try std.testing.expectError(error.UnsupportedConfig, parseConfig(gpa, relu_free));
    const blank_out = try std.mem.replaceOwned(u8, gpa, text, "13087", "13088");
    defer gpa.free(blank_out);
    try std.testing.expectError(error.InvalidConfig, parseConfig(gpa, blank_out));
}
