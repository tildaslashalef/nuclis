//! The CPU reference of Qwen3-TTS's talker and code predictor, and the
//! `Synthesizer` that drives them, or the Metal plan (qwen3_tts_metal.zig),
//! into a codec decoder stream, a chunk of frames at a time; `Voice` opens a
//! checkpoint directory on either backend. `Model.init` decodes every bound
//! weight but the text embedding to F32 (about 2.4 GB for the 0.6B, 6.4 GB
//! for the 1.7B; text rows are decoded from the checkpoint as ids need them,
//! so it must outlive the model). Projections are dense.zig's F32 matmul;
//! norms, RoPE, and attention keep F64 statistics. The equations:
//! docs/models/qwen3-tts.md § Forward pass.
const std = @import("std");
const safetensors = @import("../formats/safetensors.zig");
const cpu = @import("../backends/cpu/root.zig");
const sampling = @import("../sampling/root.zig");
const codec = @import("../audio/qwen3_codec.zig");
const tts = @import("qwen3_tts.zig");
const hf_json = @import("../tokenizer/hf_json.zig");
const metal = @import("../backends/metal/root.zig");
const tts_metal = @import("qwen3_tts_metal.zig");
const codec_metal = @import("../audio/qwen3_codec_metal.zig");
const ecapa = @import("../audio/ecapa.zig");
const codec_encoder = @import("../audio/qwen3_codec_encoder.zig");

const Allocator = std.mem.Allocator;
const dense = cpu.dense;
const groups = tts.groups;
pub const Frame = codec.Frame;

const Linear = struct { w: []const f32, b: ?[]const f32, in: usize, out: usize };
const Block = struct {
    input_norm: []const f32,
    post_norm: []const f32,
    q_norm: []const f32,
    k_norm: []const f32,
    q: Linear,
    k: Linear,
    v: Linear,
    o: Linear,
    gate: Linear,
    up: Linear,
    down: Linear,
};

/// Sees the residual stream after each layer (`layer` 0…) and, with `layer`
/// null, the rows after the final norm; `rows` is `[n][hidden]`.
pub const Observer = struct {
    context: *anyopaque,
    record: *const fn (context: *anyopaque, layer: ?usize, rows: []const f32) void,

    fn see(self: ?Observer, layer: ?usize, rows: []const f32) void {
        if (self) |o| o.record(o.context, layer, rows);
    }
};

/// The weights, read-only once built (`storage` owns the F32 copies).
pub const Model = struct {
    io: std.Io,
    config: *const tts.Config,
    storage: std.heap.ArenaAllocator,
    text_embedding: safetensors.Ref,
    fc1: Linear,
    fc2: Linear,
    /// [codec_vocab][talker width]
    codec_embedding: []const f32,
    talker: []const Block,
    norm: []const f32,
    codec_head: Linear,
    projection: ?Linear,
    predictor: []const Block,
    /// Group g's table at g − 1: [2048][talker width].
    embeddings: [groups - 1][]const f32,
    heads: [groups - 1]Linear,
    predictor_norm: []const f32,
    /// F32 bytes held.
    bytes: usize,

    /// Decodes `binding`'s tensors; `config` and the checkpoint must outlive
    /// the model, the binding need not.
    pub fn init(gpa: Allocator, io: std.Io, binding: *const tts.Binding, config: *const tts.Config) !Model {
        var self: Model = undefined;
        self.io = io;
        self.config = config;
        self.storage = .init(gpa);
        errdefer self.storage.deinit();
        var l: Loader = .{ .arena = self.storage.allocator() };
        self.text_embedding = binding.text_embedding;
        self.fc1 = try l.linear(binding.text_fc1.weight, binding.text_fc1.bias);
        self.fc2 = try l.linear(binding.text_fc2.weight, binding.text_fc2.bias);
        self.codec_embedding = try l.floats(binding.codec_embedding);
        self.talker = try l.blocks(binding.talker);
        self.norm = try l.floats(binding.norm);
        self.codec_head = try l.linear(binding.codec_head, null);
        self.projection = if (binding.projection) |p| try l.linear(p.weight, p.bias) else null;
        self.predictor = try l.blocks(binding.predictor);
        for (&self.embeddings, &self.heads, binding.embeddings, binding.heads) |*e, *h, eref, href| {
            e.* = try l.floats(eref);
            h.* = try l.linear(href, null);
        }
        self.predictor_norm = try l.floats(binding.predictor_norm);
        self.bytes = l.bytes;
        return self;
    }

    pub fn deinit(self: *Model) void {
        self.storage.deinit();
        self.* = undefined;
    }

    fn width(self: *const Model) usize {
        return self.config.talker.hidden;
    }

    /// The talker's input rows for `rows` into `out` ([rows.len][width]):
    /// the projected text row plus the codec row, where present, plus a
    /// Base's voice term from `clone`.
    pub fn embed(self: *const Model, gpa: Allocator, rows: []const tts.Row, clone: ?tts.Clone, out: []f32) !void {
        const w = self.width();
        const n = rows.len;
        if (out.len != n * w) return error.InvalidShape;
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        const ws = arena.allocator();
        const th = self.config.text_hidden;
        const texts = try ws.alloc(f32, n * th);
        const wide = try ws.alloc(f32, n * th);
        var count: usize = 0;
        for (rows) |row| if (row.text) |id| {
            if (id >= self.config.text_vocab) return error.InvalidTokenId;
            try self.text_embedding.decode(@as(u64, id) * th, texts[count * th ..][0..th]);
            count += 1;
        };
        const projected = try ws.alloc(f32, @max(count, 1) * w);
        if (count > 0) {
            try linear(self.io, self.fc1, texts[0 .. count * th], count, wide[0 .. count * th]);
            for (wide[0 .. count * th]) |*x| x.* = cpu.silu(x.*);
            try linear(self.io, self.fc2, wide[0 .. count * th], count, projected[0 .. count * w]);
        }
        var t: usize = 0;
        for (rows, 0..) |row, i| {
            const o = out[i * w ..][0..w];
            if (row.text != null) {
                @memcpy(o, projected[t * w ..][0..w]);
                t += 1;
            } else @memset(o, 0);
            if (row.codec) |id| {
                if (id >= self.config.codec_vocab) return error.InvalidCode;
                for (o, self.codec_embedding[id * w ..][0..w]) |*a, b| a.* += b;
            }
            switch (row.voice) {
                .none => {},
                .speaker => for (o, (clone orelse return error.InvalidVoice).embedding) |*a, b| {
                    a.* += b;
                },
                .frame => |k| {
                    const frames = (clone orelse return error.InvalidVoice).frames;
                    if (k >= frames.len) return error.InvalidVoice;
                    // The groups summed first, then the text: the authors' order, and `nextRow`'s.
                    const sum = try ws.alloc(f32, w);
                    @memset(sum, 0);
                    self.addFrame(frames[k], sum);
                    for (o, sum) |*a, b| a.* += b;
                },
            }
        }
    }

    /// `out += ` the frame's 16 code embeddings, in group order.
    fn addFrame(self: *const Model, frame: Frame, out: []f32) void {
        const w = self.width();
        for (out, self.codec_embedding[@as(usize, frame[0]) * w ..][0..w]) |*a, b| a.* += b;
        for (self.embeddings, frame[1..]) |table, code| for (out, table[@as(usize, code) * w ..][0..w]) |*a, b| {
            a.* += b;
        };
    }

    /// The next talker row (§ (e)): the frame's 16 code embeddings summed,
    /// then the text track `text` (the projected `tts_pad` row).
    pub fn nextRow(self: *const Model, frame: Frame, text: []const f32, out: []f32) void {
        @memset(out, 0);
        self.addFrame(frame, out);
        for (out, text) |*a, b| a.* += b;
    }
};

/// Decodes refs into the model's arena.
const Loader = struct {
    arena: Allocator,
    bytes: usize = 0,

    fn floats(self: *Loader, ref: safetensors.Ref) ![]f32 {
        const out = try self.arena.alloc(f32, @intCast(ref.tensor.elements));
        try ref.decode(0, out);
        self.bytes += out.len * 4;
        return out;
    }

    fn linear(self: *Loader, weight: safetensors.Ref, bias: ?safetensors.Ref) !Linear {
        const shape = weight.tensor.shape;
        return .{ .w = try self.floats(weight), .b = if (bias) |b| try self.floats(b) else null, .in = @intCast(shape[1]), .out = @intCast(shape[0]) };
    }

    fn blocks(self: *Loader, layers: []const tts.Layer) ![]const Block {
        const out = try self.arena.alloc(Block, layers.len);
        for (out, layers) |*b, l| b.* = .{
            .input_norm = try self.floats(l.input_norm),
            .post_norm = try self.floats(l.post_norm),
            .q_norm = try self.floats(l.q_norm),
            .k_norm = try self.floats(l.k_norm),
            .q = try self.linear(l.q, null),
            .k = try self.linear(l.k, null),
            .v = try self.linear(l.v, null),
            .o = try self.linear(l.o, null),
            .gate = try self.linear(l.gate, null),
            .up = try self.linear(l.up, null),
            .down = try self.linear(l.down, null),
        };
        return out;
    }
};

fn linear(io: std.Io, m: Linear, x: []const f32, rows: usize, y: []f32) !void {
    try dense.matmul(io, .{ .rows = rows, .inner = m.in, .outputs = m.out, .x = x[0 .. rows * m.in], .w = m.w, .bias = m.b }, y[0 .. rows * m.out]);
}

/// `x / rms(x) · w` per row of `w.len`; `out` may alias `x`.
fn rmsRows(x: []const f32, w: []const f32, eps: f32, out: []f32) !void {
    const width = w.len;
    for (0..x.len / width) |r| {
        const o = out[r * width ..][0..width];
        try cpu.rmsNorm(x[r * width ..][0..width], o, eps);
        for (o, w) |*a, b| a.* *= b;
    }
}

/// One stack's keys (after RoPE) and values, every row so far, per layer.
const Cache = struct {
    keys: []std.ArrayList(f32),
    values: []std.ArrayList(f32),
    rows: usize = 0,

    fn init(gpa: Allocator, layers: usize) !Cache {
        const keys = try gpa.alloc(std.ArrayList(f32), layers);
        errdefer gpa.free(keys);
        const values = try gpa.alloc(std.ArrayList(f32), layers);
        @memset(keys, .empty);
        @memset(values, .empty);
        return .{ .keys = keys, .values = values };
    }

    fn deinit(self: *Cache, gpa: Allocator) void {
        for (self.keys, self.values) |*k, *v| {
            k.deinit(gpa);
            v.deinit(gpa);
        }
        gpa.free(self.keys);
        gpa.free(self.values);
        self.* = undefined;
    }

    fn clear(self: *Cache) void {
        for (self.keys, self.values) |*k, *v| {
            k.clearRetainingCapacity();
            v.clearRetainingCapacity();
        }
        self.rows = 0;
    }
};

/// `n` rows of `x` ([n][hidden], updated in place to the residual stream)
/// through every block of a stack at positions `cache.rows`…, appending to
/// the cache; `hidden` receives the rows after `final_norm`.
fn runStack(gpa: Allocator, io: std.Io, blocks: []const Block, s: tts.Stack, final_norm: []const f32, cache: *Cache, x: []f32, n: usize, hidden: []f32, observer: ?Observer) !void {
    var arena: std.heap.ArenaAllocator = .init(gpa);
    defer arena.deinit();
    const ws = arena.allocator();
    const w = s.hidden;
    const d = s.head_dim;
    const qw = s.heads * d;
    const kvw = s.kv_heads * d;
    const past = cache.rows;
    const h = try ws.alloc(f32, n * w);
    const q = try ws.alloc(f32, n * qw);
    const k = try ws.alloc(f32, n * kvw);
    const v = try ws.alloc(f32, n * kvw);
    const attended = try ws.alloc(f32, n * qw);
    const o = try ws.alloc(f32, n * w);
    const gate = try ws.alloc(f32, n * s.intermediate);
    const up = try ws.alloc(f32, n * s.intermediate);
    const scores = try ws.alloc(f64, past + n);
    for (blocks, 0..) |b, layer| {
        try rmsRows(x, b.input_norm, s.rms_eps, h);
        try linear(io, b.q, h, n, q);
        try linear(io, b.k, h, n, k);
        try linear(io, b.v, h, n, v);
        try rmsRows(q, b.q_norm, s.rms_eps, q);
        try rmsRows(k, b.k_norm, s.rms_eps, k);
        for (0..n) |t| {
            // One position per row: the talker's M-RoPE with three equal
            // components is this 1-D NEOX RoPE (docs/models/qwen3-tts.md § (b)).
            const options: cpu.rope.Options = .{ .dimensions = d, .base = s.rope_theta, .position = @intCast(past + t) };
            for (0..s.heads) |head| {
                const at = q[t * qw + head * d ..][0..d];
                try cpu.rope.apply(at, at, options);
            }
            for (0..s.kv_heads) |head| {
                const at = k[t * kvw + head * d ..][0..d];
                try cpu.rope.apply(at, at, options);
            }
        }
        try cache.keys[layer].appendSlice(gpa, k);
        try cache.values[layer].appendSlice(gpa, v);
        attention(s, q, cache.keys[layer].items, cache.values[layer].items, n, past, scores, attended);
        try linear(io, b.o, attended, n, o);
        for (x, o) |*r, value| r.* += value;
        try rmsRows(x, b.post_norm, s.rms_eps, h);
        try linear(io, b.gate, h, n, gate);
        try linear(io, b.up, h, n, up);
        for (gate, up) |*g, u| g.* = cpu.silu(g.*) * u;
        try linear(io, b.down, gate, n, o);
        for (x, o) |*r, value| r.* += value;
        Observer.see(observer, layer, x);
    }
    cache.rows += n;
    try rmsRows(x, final_norm, s.rms_eps, hidden);
    Observer.see(observer, null, hidden);
}

/// Causal grouped-query attention of `n` new queries (row `past + t` sees
/// rows 0…past + t); scores and sums in F64.
fn attention(s: tts.Stack, q: []const f32, keys: []const f32, values: []const f32, n: usize, past: usize, scores: []f64, out: []f32) void {
    const d = s.head_dim;
    const qw = s.heads * d;
    const kvw = s.kv_heads * d;
    const per_kv = s.heads / s.kv_heads;
    const scale = 1 / @sqrt(@as(f64, @floatFromInt(d)));
    for (0..n) |t| for (0..s.heads) |head| {
        const kv = head / per_kv;
        const count = past + t + 1;
        const qi = q[t * qw + head * d ..][0..d];
        var max: f64 = -std.math.inf(f64);
        for (0..count) |j| {
            var dot: f64 = 0;
            for (qi, keys[j * kvw + kv * d ..][0..d]) |a, b| dot += @as(f64, a) * b;
            scores[j] = dot * scale;
            max = @max(max, scores[j]);
        }
        var total: f64 = 0;
        for (scores[0..count]) |*x| {
            x.* = @exp(x.* - max);
            total += x.*;
        }
        const row = out[t * qw + head * d ..][0..d];
        for (row, 0..) |*value, e| {
            var sum: f64 = 0;
            for (0..count) |j| sum += scores[j] * values[j * kvw + kv * d + e];
            value.* = @floatCast(sum / total);
        }
    };
}

/// The talker's state over one utterance: its cache and the rows so far.
pub const Talker = struct {
    gpa: Allocator,
    model: *const Model,
    cache: Cache,

    pub fn init(gpa: Allocator, model: *const Model) !Talker {
        return .{ .gpa = gpa, .model = model, .cache = try .init(gpa, model.talker.len) };
    }

    pub fn deinit(self: *Talker) void {
        self.cache.deinit(self.gpa);
        self.* = undefined;
    }

    /// Rows fed so far: the next row's position.
    pub fn position(self: *const Talker) usize {
        return self.cache.rows;
    }

    /// Runs `n` input rows `x` (consumed) and writes their post-norm rows to
    /// `hidden` ([n][width]).
    pub fn forward(self: *Talker, x: []f32, n: usize, hidden: []f32, observer: ?Observer) !void {
        const m = self.model;
        try runStack(self.gpa, m.io, m.talker, m.config.talker, m.norm, &self.cache, x, n, hidden, observer);
    }

    /// The group-0 logits of a post-norm row `h` (`codec_head`, codec_vocab wide).
    pub fn logits(self: *const Talker, h: []const f32, out: []f32) !void {
        try linear(self.model.io, self.model.codec_head, h, 1, out);
    }
};

/// Chooses group `g`'s code (1…15) from its logits: the sampler, or a trace's code.
pub const Pick = struct {
    context: *anyopaque,
    pick: *const fn (context: *anyopaque, group: usize, logits: []const f32) anyerror!u16,
};

/// The code predictor (§ (d)): a fresh 16-row cache each frame.
pub const Predictor = struct {
    gpa: Allocator,
    model: *const Model,
    cache: Cache,
    input: []f32,
    x: []f32,
    hidden: []f32,
    logits: []f32,

    pub fn init(gpa: Allocator, model: *const Model) !Predictor {
        const p = model.config.predictor.hidden;
        const w = model.config.talker.hidden;
        var cache: Cache = try .init(gpa, model.predictor.len);
        errdefer cache.deinit(gpa);
        const input = try gpa.alloc(f32, 2 * w);
        errdefer gpa.free(input);
        const x = try gpa.alloc(f32, 2 * p);
        errdefer gpa.free(x);
        const hidden = try gpa.alloc(f32, 2 * p);
        errdefer gpa.free(hidden);
        return .{ .gpa = gpa, .model = model, .cache = cache, .input = input, .x = x, .hidden = hidden, .logits = try gpa.alloc(f32, tts.codebook_size) };
    }

    pub fn deinit(self: *Predictor) void {
        self.cache.deinit(self.gpa);
        for ([_][]f32{ self.input, self.x, self.hidden, self.logits }) |b| self.gpa.free(b);
        self.* = undefined;
    }

    /// Codes 1–15 of the frame whose group-0 code `c0` the talker's post-norm
    /// row `h` gave; `pick` sees each group's logits in turn.
    pub fn frame(self: *Predictor, h: []const f32, c0: u16, pick: Pick) !Frame {
        const m = self.model;
        const s = m.config.predictor;
        const w = m.config.talker.hidden;
        self.cache.clear();
        var codes: Frame = undefined;
        codes[0] = c0;
        @memcpy(self.input[0..w], h);
        @memcpy(self.input[w..][0..w], m.codec_embedding[@as(usize, c0) * w ..][0..w]);
        var rows: usize = 2;
        for (1..groups) |g| {
            if (g > 1) {
                rows = 1;
                @memcpy(self.input[0..w], m.embeddings[g - 2][@as(usize, codes[g - 1]) * w ..][0..w]);
            }
            const x = self.x[0 .. rows * s.hidden];
            if (m.projection) |proj| try linear(m.io, proj, self.input[0 .. rows * w], rows, x) else @memcpy(x, self.input[0 .. rows * w]);
            try runStack(self.gpa, m.io, m.predictor, s, m.predictor_norm, &self.cache, x, rows, self.hidden[0 .. rows * s.hidden], null);
            try linear(m.io, m.heads[g - 1], self.hidden[(rows - 1) * s.hidden ..][0..s.hidden], 1, self.logits);
            const code = try pick.pick(pick.context, g, self.logits);
            if (code >= tts.codebook_size) return error.InvalidCode;
            codes[g] = code;
        }
        return codes;
    }
};

/// § (c) and (d): group 0 under the repetition penalty over the group-0
/// codes sampled so far, the end code blocked until `min_new_tokens`
/// frames, the markers suppressed; then temperature and top-k (top-p) for
/// both; one seeded stream of draws (`sampling.Sampler`).
pub const Sampler = struct {
    inner: sampling.Sampler,
    talker: sampling.Options,
    predictor: sampling.Options,
    /// Group-0 codes generated, over the compact vocabulary (codes, then the end).
    history: sampling.History,
    candidates: []sampling.Candidate,
    /// The codebook's logits then, when allowed, the end code's.
    compact: []f32,
    eos: u32,
    min_new_tokens: usize,
    /// Group-0 samples drawn.
    sampled: usize = 0,

    pub fn init(gpa: Allocator, seed: u64, generation: tts.Generation, eos: u32) !Sampler {
        const talker: sampling.Options = .{ .temperature = generation.temperature, .top_k = generation.top_k, .top_p = generation.top_p, .repetition_penalty = generation.repetition_penalty };
        const predictor: sampling.Options = .{ .temperature = generation.predictor_temperature, .top_k = generation.predictor_top_k, .top_p = generation.predictor_top_p };
        var inner = try sampling.Sampler.init(seed, talker);
        try inner.setOptions(predictor);
        var history = try sampling.History.init(gpa, tts.codebook_size + 1);
        errdefer history.deinit();
        const candidates = try gpa.alloc(sampling.Candidate, tts.codebook_size + 1);
        errdefer gpa.free(candidates);
        return .{ .inner = inner, .talker = talker, .predictor = predictor, .history = history, .candidates = candidates, .compact = try gpa.alloc(f32, tts.codebook_size + 1), .eos = eos, .min_new_tokens = generation.min_new_tokens };
    }

    pub fn deinit(self: *Sampler, gpa: Allocator) void {
        self.history.deinit();
        gpa.free(self.candidates);
        gpa.free(self.compact);
        self.* = undefined;
    }

    /// Group 0 from the talker's logits (codec_vocab wide): a code, or `eos`.
    pub fn groupZero(self: *Sampler, logits: []const f32) !u32 {
        if (logits.len <= self.eos) return error.InvalidLogits;
        const n = tts.codebook_size;
        @memcpy(self.compact[0..n], logits[0..n]);
        const end_allowed = self.sampled >= self.min_new_tokens;
        self.compact[n] = logits[self.eos];
        try self.inner.setOptions(self.talker);
        const id = try self.inner.select(self.compact[0 .. n + @as(usize, @intFromBool(end_allowed))], self.candidates, &self.history);
        self.sampled += 1;
        if (id == n) return self.eos;
        try self.history.observe(id);
        return id;
    }

    /// Group g (1–15) from the predictor's logits (2048 wide).
    pub fn group(self: *Sampler, logits: []const f32) !u16 {
        try self.inner.setOptions(self.predictor);
        return @intCast(try self.inner.select(logits, self.candidates, null));
    }

    fn pickFn(context: *anyopaque, _: usize, logits: []const f32) anyerror!u16 {
        const self: *Sampler = @ptrCast(@alignCast(context));
        return self.group(logits);
    }

    pub fn pick(self: *Sampler) Pick {
        return .{ .context = self, .pick = pickFn };
    }
};

/// Where a synthesizer's talker and predictor run: the CPU reference's
/// `Model`, or the Metal plan. Borrowed.
pub const Engine = union(enum) {
    cpu: *const Model,
    metal: *tts_metal.Plan,
};

/// Where its codec decoder runs, if anywhere. Borrowed.
pub const Decoder = union(enum) {
    none,
    cpu: *const codec.Runtime,
    metal: *codec_metal.Decoder,
};

/// Frames `Synthesizer.read` decodes at once by default: 320 ms of audio.
pub const default_chunk = 4;
/// Frames the first `read` decodes by default: the first audio leaves after
/// one frame (80 ms), the rest in `chunk`s. The pieces change no sample.
pub const default_first_chunk = 1;
/// After the first, a chunk is at most the frames made before it (1, 1, 2,
/// 4, …): faster than real time, each chunk is made before the audio
/// before it has played.
/// The most frames one `read` decodes.
pub const max_chunk = 32;

/// Wall time by stage over an utterance, in nanoseconds.
pub const Timing = struct {
    /// The prompt's embedding and prefill, through the first logits.
    prefill: u64 = 0,
    /// Every frame's codes and the talker step after it, sampling included.
    frames: u64 = 0,
    /// Of `frames`, the talker steps (the next row, its layers, the group-0
    /// logits) and the predictor (15 groups): wall time on the CPU, device
    /// time on Metal (there only with `Synthesizer.split`).
    talker: u64 = 0,
    predictor: u64 = 0,
    /// Codec pushes.
    codec: u64 = 0,
};

/// One utterance: the prompt's prefill, then frame after frame of codes
/// (talker, sampler, predictor) until the end code or the length limit,
/// decoded through a codec stream in chunks of `chunk` frames. Borrows the
/// engine and the decoder.
pub const Synthesizer = struct {
    gpa: Allocator,
    io: std.Io,
    config: *const tts.Config,
    state: State,
    sampler: Sampler,
    stream: Stream,
    /// The next input row and the projected `tts_pad` row.
    row: []f32,
    pad: []f32,
    /// The prompt's trailing text, projected: frame k's next row adds row
    /// k, then `pad` (`[n][width]`).
    trailing: []f32,
    /// Frames `read` decodes per call at most (1 … `max_chunk`), and in its first call.
    chunk: usize = default_chunk,
    first_chunk: usize = default_first_chunk,
    frames: usize = 0,
    done: bool = false,
    timing: Timing = .{},
    /// On Metal, commit between the predictor and the talker step so that
    /// `timing` splits them (one more command buffer a frame).
    split: bool = false,
    /// On Metal, sample groups 1–15 on the host a command buffer at a time
    /// instead of on the device: the same codes, slower.
    host_sampling: bool = false,

    const State = union(enum) {
        cpu: struct { model: *const Model, talker: Talker, predictor: Predictor, hidden: []f32, logits: []f32 },
        metal: *tts_metal.Plan,
    };
    const Stream = union(enum) {
        none,
        cpu: codec.Stream,
        metal: codec_metal.Stream,
    };

    /// Embeds and prefills `prompt`; the first frame's logits are ready
    /// after. A cloned voice's reference frames go through the decoder
    /// first, their samples discarded, so the voice's audio begins where its
    /// reference ends (docs/models/qwen3-tts.md § The decode). The prompt's
    /// rows and trailing text are copied in; its clone is not read after.
    pub fn init(gpa: Allocator, io: std.Io, config: *const tts.Config, engine: Engine, decoder: Decoder, prompt: *const tts.Prompt, seed: u64) !Synthesizer {
        const w = config.talker.hidden;
        const start = std.Io.Clock.awake.now(io);
        var self: Synthesizer = undefined;
        self.gpa = gpa;
        self.io = io;
        self.config = config;
        self.chunk = default_chunk;
        self.first_chunk = default_first_chunk;
        self.frames = 0;
        self.done = false;
        self.timing = .{};
        self.split = false;
        self.host_sampling = false;
        self.sampler = try .init(gpa, seed, config.generation, config.codec.eos);
        errdefer self.sampler.deinit(gpa);
        self.row = try gpa.alloc(f32, w);
        errdefer gpa.free(self.row);
        self.pad = try gpa.alloc(f32, w);
        errdefer gpa.free(self.pad);
        self.trailing = try gpa.alloc(f32, prompt.trailing.len * w);
        errdefer gpa.free(self.trailing);
        self.stream = switch (decoder) {
            .none => .none,
            .cpu => |r| .{ .cpu = try codec.Stream.init(gpa, r) },
            .metal => |d| .{ .metal = try codec_metal.Stream.init(d) },
        };
        errdefer self.closeStream();
        try self.prime(prompt.primer());
        const pad_row = [_]tts.Row{.{ .text = config.text.tts_pad, .codec = null }};
        const trailing_rows = try gpa.alloc(tts.Row, prompt.trailing.len);
        defer gpa.free(trailing_rows);
        for (trailing_rows, prompt.trailing) |*r, id| r.* = .{ .text = id, .codec = null };
        const rows = prompt.rows;
        const x = try gpa.alloc(f32, rows.len * w);
        defer gpa.free(x);
        switch (engine) {
            .cpu => |model| {
                var talker = try Talker.init(gpa, model);
                errdefer talker.deinit();
                var predictor = try Predictor.init(gpa, model);
                errdefer predictor.deinit();
                const hidden = try gpa.alloc(f32, w);
                errdefer gpa.free(hidden);
                const logits = try gpa.alloc(f32, config.codec_vocab);
                errdefer gpa.free(logits);
                try model.embed(gpa, &pad_row, null, self.pad);
                if (trailing_rows.len > 0) try model.embed(gpa, trailing_rows, null, self.trailing);
                const states = try gpa.alloc(f32, rows.len * w);
                defer gpa.free(states);
                try model.embed(gpa, rows, prompt.clone, x);
                try talker.forward(x, rows.len, states, null);
                @memcpy(hidden, states[(rows.len - 1) * w ..][0..w]);
                try talker.logits(hidden, logits);
                self.state = .{ .cpu = .{ .model = model, .talker = talker, .predictor = predictor, .hidden = hidden, .logits = logits } };
            },
            .metal => |plan| {
                try plan.embed(&pad_row, null, self.pad);
                if (trailing_rows.len > 0) try plan.embed(trailing_rows, null, self.trailing);
                try plan.embed(rows, prompt.clone, x);
                try plan.prefill(x, rows.len);
                self.state = .{ .metal = plan };
            },
        }
        self.timing.prefill = elapsed(io, start);
        return self;
    }

    /// Decodes `frames` into the stream and drops their samples.
    fn prime(self: *Synthesizer, frames: []const Frame) !void {
        if (frames.len == 0 or self.stream == .none) return;
        const per = switch (self.stream) {
            .none => unreachable,
            .cpu => |s| s.runtime.config.samplesPerFrame(),
            .metal => |s| s.decoder.config.samplesPerFrame(),
        };
        const scratch = try self.gpa.alloc(f32, max_chunk * per);
        defer self.gpa.free(scratch);
        var at: usize = 0;
        while (at < frames.len) : (at += max_chunk) {
            const piece = frames[at..@min(frames.len, at + max_chunk)];
            switch (self.stream) {
                .none => unreachable,
                .cpu => |*s| try s.push(piece, scratch[0 .. piece.len * per]),
                .metal => |*s| try s.push(piece, scratch[0 .. piece.len * per]),
            }
        }
    }

    /// The text track frame `k`'s next row adds: trailing row k, then `tts_pad`.
    fn textRow(self: *const Synthesizer, k: usize) []const f32 {
        const w = self.pad.len;
        return if (k * w < self.trailing.len) self.trailing[k * w ..][0..w] else self.pad;
    }

    fn closeStream(self: *Synthesizer) void {
        switch (self.stream) {
            .none => {},
            .cpu => |*s| s.deinit(),
            .metal => |*s| s.deinit(),
        }
    }

    pub fn deinit(self: *Synthesizer) void {
        switch (self.state) {
            .cpu => |*s| {
                self.gpa.free(s.hidden);
                self.gpa.free(s.logits);
                s.predictor.deinit();
                s.talker.deinit();
            },
            .metal => {},
        }
        self.closeStream();
        self.gpa.free(self.row);
        self.gpa.free(self.pad);
        self.gpa.free(self.trailing);
        self.sampler.deinit(self.gpa);
        self.* = undefined;
    }

    /// The next frame's codes, or null at the end code or when the talker
    /// has sampled `max_new_tokens` codes (the last of which has no frame).
    pub fn nextFrame(self: *Synthesizer) !?Frame {
        if (self.done) return null;
        const logits = switch (self.state) {
            .cpu => |s| s.logits,
            .metal => |plan| plan.talkerLogits(),
        };
        const c0 = try self.sampler.groupZero(logits);
        if (c0 == self.config.codec.eos or self.sampler.sampled >= self.config.generation.max_new_tokens) {
            self.done = true;
            return null;
        }
        const start = std.Io.Clock.awake.now(self.io);
        const text = self.textRow(self.frames);
        const frame = switch (self.state) {
            .cpu => |*s| frame: {
                const frame = try s.predictor.frame(s.hidden, @intCast(c0), self.sampler.pick());
                const step = std.Io.Clock.awake.now(self.io);
                self.timing.predictor += @intCast(start.durationTo(step).toNanoseconds());
                s.model.nextRow(frame, text, self.row);
                try s.talker.forward(self.row, 1, s.hidden, null);
                try s.talker.logits(s.hidden, s.logits);
                self.timing.talker += elapsed(self.io, step);
                break :frame frame;
            },
            .metal => |plan| frame: {
                const o = self.sampler.predictor;
                if (self.host_sampling or o.min_p != 0) {
                    const frame = try plan.frame(@intCast(c0), self.sampler.pick());
                    try plan.nextRow(frame, text, self.row);
                    try plan.step(self.row);
                    break :frame frame;
                }
                // The device samples with the host sampler's next draws; the
                // host checks every choice (`Plan.frameOnDevice`).
                var draws: [groups - 1]f64 = undefined;
                var rng = self.sampler.inner.rng;
                for (&draws) |*d| d.* = rng.random().float(f64);
                plan.split_timing = self.split;
                const frame = try plan.frameOnDevice(@intCast(c0), &draws, .{ .temperature = o.temperature, .top_k = o.top_k, .top_p = o.top_p }, text, self.sampler.pick());
                if (self.split) {
                    self.timing.predictor += @intFromFloat(plan.predictor_gpu * 1e9);
                    self.timing.talker += @intFromFloat(plan.talker_gpu * 1e9);
                }
                break :frame frame;
            },
        };
        self.timing.frames += elapsed(self.io, start);
        self.frames += 1;
        return frame;
    }

    /// Generates up to `chunk` frames (`first_chunk` in the first call, no
    /// more than were made before, and no more than `out` holds)
    /// and decodes them into `out`; returns the samples written, 0 once the
    /// utterance ended. Needs a decoder.
    pub fn read(self: *Synthesizer, out: []f32) !usize {
        const per = switch (self.stream) {
            .none => return error.NoCodec,
            .cpu => |s| s.runtime.config.samplesPerFrame(),
            .metal => |s| s.decoder.config.samplesPerFrame(),
        };
        const wanted = if (self.frames == 0) self.first_chunk else @min(self.chunk, self.frames);
        const capacity = @min(out.len / per, wanted, max_chunk);
        if (capacity == 0) return error.InvalidShape;
        var frames: [max_chunk]Frame = undefined;
        var n: usize = 0;
        while (n < capacity) {
            frames[n] = (try self.nextFrame()) orelse break;
            n += 1;
        }
        if (n == 0) return 0;
        const start = std.Io.Clock.awake.now(self.io);
        switch (self.stream) {
            .none => unreachable,
            .cpu => |*s| try s.push(frames[0..n], out[0 .. n * per]),
            .metal => |*s| try s.push(frames[0..n], out[0 .. n * per]),
        }
        self.timing.codec += elapsed(self.io, start);
        return n * per;
    }
};

fn elapsed(io: std.Io, start: std.Io.Timestamp) u64 {
    return @intCast(start.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds());
}

/// Where a `Voice` runs.
pub const Backend = enum { cpu, metal };

pub const VoiceOptions = struct {
    backend: Backend = .cpu,
    /// The talker cache's precision on Metal (the CPU's is F32).
    kv: tts_metal.Precision = .f16,
};

/// A checkpoint directory opened whole: the config, the tokenizer, the
/// talker and predictor, and the codec decoder of `speech_tokenizer/`, on
/// the CPU (F32 copies of every weight) or on Metal (its own `Backend`, the
/// BF16 weights wrapped in place). Heap allocated so the borrows stay put
/// (`open`, `close`).
pub const Voice = struct {
    gpa: Allocator,
    io: std.Io,
    backend: Backend,
    config: tts.Config,
    tokenizer: hf_json.Tokenizer,
    checkpoint: safetensors.Checkpoint,
    binding: tts.Binding,
    codec_checkpoint: safetensors.Checkpoint,
    codec_binding: codec.Binding,
    /// The CPU engine and codec (`.cpu`).
    model: ?Model,
    codec: ?codec.Runtime,
    /// The Metal engine and codec (`.metal`), on `gpu`.
    gpu: ?metal.Backend,
    plan: ?tts_metal.Plan,
    decoder: ?codec_metal.Decoder,

    /// Opens `dir`; a binding error names its tensor in `diagnostic`.
    pub fn open(gpa: Allocator, io: std.Io, dir: []const u8, diagnostic: ?*tts.Diagnostic, options: VoiceOptions) !*Voice {
        const self = try gpa.create(Voice);
        errdefer gpa.destroy(self);
        self.gpa = gpa;
        self.io = io;
        self.backend = options.backend;
        self.model = null;
        self.codec = null;
        self.gpu = null;
        self.plan = null;
        self.decoder = null;
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const config_bytes = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "config.json" }), a, .limited(1 << 20));
        const generation_bytes: ?[]const u8 = std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "generation_config.json" }), a, .limited(1 << 20)) catch |err| switch (err) {
            error.FileNotFound => null,
            else => return err,
        };
        self.config = try tts.parseConfig(gpa, config_bytes, generation_bytes);
        errdefer self.config.deinit();
        self.tokenizer = try tts.loadTokenizer(gpa, io, dir);
        errdefer self.tokenizer.deinit();
        self.checkpoint = try .open(gpa, io, try std.fs.path.join(a, &.{ dir, "model.safetensors" }), .{});
        errdefer self.checkpoint.deinit(io);
        self.binding = try tts.bind(gpa, &self.checkpoint, &self.config, diagnostic);
        errdefer self.binding.deinit(gpa);
        self.codec_checkpoint = try .open(gpa, io, try std.fs.path.join(a, &.{ dir, "speech_tokenizer", "model.safetensors" }), .{});
        errdefer self.codec_checkpoint.deinit(io);
        var codec_diagnostic: codec.Diagnostic = .{};
        self.codec_binding = codec.bind(gpa, &self.codec_checkpoint, .{}, &codec_diagnostic) catch |err| {
            if (diagnostic) |d| tts.Diagnostic.set(d, codec_diagnostic.name());
            return err;
        };
        errdefer self.codec_binding.deinit(gpa);
        // The codec's F32 matrices are laid out once on the CPU; Metal copies them.
        self.codec = try codec.Runtime.init(gpa, io, &self.codec_binding);
        errdefer self.release();
        switch (options.backend) {
            .cpu => self.model = try Model.init(gpa, io, &self.binding, &self.config),
            .metal => {
                var message: [512]u8 = @splat(0);
                self.gpu = try metal.Backend.init(gpa, &message);
                self.plan = try tts_metal.Plan.init(gpa, &self.gpu.?, &self.binding, &self.config, options.kv);
                self.decoder = try codec_metal.Decoder.init(gpa, &self.gpu.?, &self.codec.?);
                self.codec.?.deinit();
                self.codec = null;
            },
        }
        return self;
    }

    /// The engines, in reverse order of creation.
    fn release(self: *Voice) void {
        if (self.decoder) |*d| d.deinit();
        if (self.plan) |*p| p.deinit();
        if (self.gpu) |*g| g.deinit();
        if (self.model) |*m| m.deinit();
        if (self.codec) |*c| c.deinit();
        self.decoder = null;
        self.plan = null;
        self.gpu = null;
        self.model = null;
        self.codec = null;
    }

    /// Device memory the Metal engine and codec created (converted weights,
    /// caches, scratch), beside the mapped checkpoints; 0 on the CPU.
    pub fn deviceBytes(self: *const Voice) u64 {
        return if (self.gpu) |g| g.created_bytes else 0;
    }

    pub fn close(self: *Voice, io: std.Io) void {
        self.release();
        self.codec_binding.deinit(self.gpa);
        self.codec_checkpoint.deinit(io);
        self.binding.deinit(self.gpa);
        self.checkpoint.deinit(io);
        self.tokenizer.deinit();
        self.config.deinit();
        self.gpa.destroy(self);
    }

    /// A Base's speaker embedding of `samples` (24 kHz mono) into `out`
    /// (the talker's width). The CPU encoder is built for the call and
    /// freed after; `NotABase` on a CustomVoice.
    pub fn speakerEmbedding(self: *const Voice, gpa: Allocator, samples: []const f32, out: []f32) !void {
        const config = self.config.speaker_encoder orelse return error.NotABase;
        var encoder = try ecapa.Encoder.init(gpa, &self.binding.speaker_encoder.?, config);
        defer encoder.deinit();
        try encoder.embed(gpa, self.io, samples, out, null);
    }

    /// The codec's codes of `samples` (24 kHz mono), `⌈samples / 1920⌉`
    /// frames owned by `gpa`: a Base's reference codes. The CPU encoder is
    /// built for the call (about 190 MB as F32) and freed after.
    pub fn referenceCodes(self: *const Voice, gpa: Allocator, samples: []const f32) ![]Frame {
        var diagnostic: codec.Diagnostic = .{};
        var binding = try codec_encoder.bind(gpa, &self.codec_checkpoint, &self.codec_binding, .{}, &diagnostic);
        defer binding.deinit(gpa);
        var encoder = try codec_encoder.Encoder.init(gpa, self.io, &self.codec_binding, &binding);
        defer encoder.deinit();
        const frames = try gpa.alloc(Frame, encoder.frameCount(samples.len));
        errdefer gpa.free(frames);
        try encoder.encode(gpa, samples, frames, null);
        return frames;
    }

    /// A Base's prompt for `text` in `voice`: ICL when `reference` (the
    /// clip's transcript) is given, the speaker embedding alone otherwise.
    /// The prompt borrows `voice`.
    pub fn clonePrompt(self: *const Voice, gpa: Allocator, text: []const u8, reference: ?[]const u8, language: []const u8, voice: tts.Clone) !tts.Prompt {
        var ids = try tts.encodeText(gpa, &self.tokenizer, text, null);
        defer ids.deinit(gpa);
        const reference_ids: ?[]u32 = if (reference) |r| try tts.encodeReference(gpa, &self.tokenizer, r) else null;
        defer if (reference_ids) |r| gpa.free(r);
        return tts.Prompt.buildClone(gpa, &self.config, .{ .role = ids.role, .text = ids.text, .reference = reference_ids, .language = language, .clone = voice });
    }

    /// The prompt for `text` spoken by `speaker` in `language`, under
    /// `instruct` (the 1.7B's only: `error.InstructionUnsupported` otherwise).
    pub fn prompt(self: *const Voice, gpa: Allocator, text: []const u8, speaker: ?[]const u8, language: []const u8, instruct: ?[]const u8) !tts.Prompt {
        if (instruct) |i| if (i.len > 0 and !self.config.takesInstruction()) return error.InstructionUnsupported;
        var ids = try tts.encodeText(gpa, &self.tokenizer, text, instruct);
        defer ids.deinit(gpa);
        return tts.Prompt.build(gpa, &self.config, .{ .role = ids.role, .text = ids.text, .instruct = ids.instruct, .speaker = speaker, .language = language });
    }

    /// The engine the voice opened.
    pub fn engine(self: *Voice) Engine {
        return if (self.plan) |*p| .{ .metal = p } else .{ .cpu = &self.model.? };
    }

    /// The codec decoder the voice opened.
    pub fn decoderOf(self: *Voice) Decoder {
        return if (self.decoder) |*d| .{ .metal = d } else .{ .cpu = &self.codec.? };
    }

    /// A synthesizer for `prompt` that decodes into samples. One at a time
    /// on Metal: the plan holds the talker's cache.
    pub fn synthesizer(self: *Voice, gpa: Allocator, request: *const tts.Prompt, seed: u64) !Synthesizer {
        return Synthesizer.init(gpa, self.io, &self.config, self.engine(), self.decoderOf(), request, seed);
    }
};

// ---- tests -----------------------------------------------------------------

test "group 0: penalty on generated codes only, the end blocked for two frames, markers suppressed" {
    const gpa = std.testing.allocator;
    const vocab = 3072;
    var logits: [vocab]f32 = @splat(-5);
    // A marker other than the end, and the end, outscore every code.
    logits[2500] = 9;
    logits[2150] = 8;
    logits[5] = 1.04;
    logits[6] = 1.0;
    // Greedy keeps the processors visible without a draw.
    var s = try Sampler.init(gpa, 1, .{ .temperature = 0, .repetition_penalty = 1.05 }, 2150);
    defer s.deinit(gpa);
    try std.testing.expectEqual(@as(u32, 5), try s.groupZero(&logits));
    // Code 5 is penalized now: 1.04 / 1.05 < 1.0.
    try std.testing.expectEqual(@as(u32, 6), try s.groupZero(&logits));
    // Two frames sampled: the end is allowed and wins; 2500 never does.
    try std.testing.expectEqual(@as(u32, 2150), try s.groupZero(&logits));
    try std.testing.expectEqual(@as(usize, 2), s.history.count());

    // A negative logit is multiplied: −1 becomes −1.05, below −1.04.
    var negative: [vocab]f32 = @splat(-5);
    negative[7] = -1;
    negative[8] = -1.04;
    var t = try Sampler.init(gpa, 1, .{ .temperature = 0, .repetition_penalty = 1.05 }, 2150);
    defer t.deinit(gpa);
    try std.testing.expectEqual(@as(u32, 7), try t.groupZero(&negative));
    try std.testing.expectEqual(@as(u32, 8), try t.groupZero(&negative));
}

test "top-k bounds every draw to the k best, for group 0 and the predictor" {
    const gpa = std.testing.allocator;
    var logits: [3072]f32 = @splat(0);
    logits[10] = 3;
    logits[11] = 2.9;
    logits[12] = 2.8;
    var codes: [2048]f32 = @splat(0);
    codes[100] = 3;
    codes[101] = 2.9;
    codes[102] = 2.8;
    var seen: [3]bool = @splat(false);
    for (0..64) |seed| {
        var s = try Sampler.init(gpa, seed, .{ .top_k = 2, .predictor_top_k = 2, .repetition_penalty = 1 }, 2150);
        defer s.deinit(gpa);
        const zero = try s.groupZero(&logits);
        try std.testing.expect(zero == 10 or zero == 11);
        seen[zero - 10] = true;
        const g = try s.group(&codes);
        try std.testing.expect(g == 100 or g == 101);
    }
    try std.testing.expect(seen[0] and seen[1] and !seen[2]);
}

test "a Base's voice rows: the speaker embedding added, a reference frame embedded as a generated one's next row" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const json = try tts.testConfigJson(gpa, "0b6", 8, "base");
    defer gpa.free(json);
    var config = try tts.parseConfig(gpa, json, null);
    defer config.deinit();
    var file = try tts.TestFile.open(io, &config, .{});
    defer file.close(io);
    var binding = try tts.bind(gpa, &file.checkpoint, &config, null);
    defer binding.deinit(gpa);
    var model = try Model.init(gpa, io, &binding, &config);
    defer model.deinit();
    const w = config.talker.hidden;
    var embedding: [8]f32 = undefined;
    for (&embedding, 0..) |*v, i| v.* = @floatFromInt(i);
    var frame: Frame = undefined;
    for (&frame, 0..) |*code, g| code.* = @intCast(100 + 37 * g);
    const clone: tts.Clone = .{ .embedding = &embedding, .frames = &.{frame} };
    const rows = [_]tts.Row{
        .{ .text = 22, .codec = null },
        .{ .text = 22, .codec = null, .voice = .speaker },
        .{ .text = 22, .codec = null, .voice = .{ .frame = 0 } },
    };
    var out: [3 * 8]f32 = undefined;
    try model.embed(gpa, &rows, clone, &out);
    for (out[w..][0..w], out[0..w], embedding) |got, text, e| try std.testing.expectEqual(text + e, got);
    var next: [8]f32 = undefined;
    model.nextRow(frame, out[0..w], &next);
    try std.testing.expectEqualSlices(f32, &next, out[2 * w ..][0..w]);
    // A voice row needs the clone, and its frame must exist.
    try std.testing.expectError(error.InvalidVoice, model.embed(gpa, rows[1..2], null, out[0..w]));
    const late = [_]tts.Row{.{ .text = 22, .codec = null, .voice = .{ .frame = 1 } }};
    try std.testing.expectError(error.InvalidVoice, model.embed(gpa, &late, clone, out[0..w]));
}

test "a tiny model synthesizes frames of valid codes and stops at the length limit" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const json = try std.fmt.allocPrint(gpa,
        \\{{"model_type":"qwen3_tts","tts_model_type":"custom_voice","tts_model_size":"1b7",
        \\ "im_start_token_id":20,"im_end_token_id":21,"assistant_token_id":7,"tts_pad_token_id":22,"tts_bos_token_id":23,"tts_eos_token_id":24,
        \\ "talker_config":{{"num_hidden_layers":2,"hidden_size":16,"intermediate_size":12,"num_attention_heads":4,"num_key_value_heads":2,
        \\  "head_dim":4,"rms_norm_eps":1e-6,"rope_theta":1000000,"hidden_act":"silu","vocab_size":3072,"num_code_groups":16,
        \\  "text_vocab_size":32,"text_hidden_size":6,"codec_pad_id":2148,"codec_bos_id":2149,"codec_eos_token_id":2150,
        \\  "codec_think_id":2154,"codec_nothink_id":2155,"codec_think_bos_id":2156,"codec_think_eos_id":2157,
        \\  "codec_language_id":{{"english":2050}},"spk_id":{{"ryan":3061}},"spk_is_dialect":{{"ryan":false}},
        \\  "code_predictor_config":{{"num_hidden_layers":1,"hidden_size":8,"intermediate_size":12,"num_attention_heads":4,
        \\   "num_key_value_heads":2,"head_dim":4,"rms_norm_eps":1e-6,"rope_theta":1000000,"hidden_act":"silu","vocab_size":2048,"num_code_groups":16}}}}}}
    , .{});
    defer gpa.free(json);
    var config = try tts.parseConfig(gpa, json, "{\"max_new_tokens\":6}");
    defer config.deinit();
    var file = try tts.TestFile.open(io, &config, .{});
    defer file.close(io);
    var binding = try tts.bind(gpa, &file.checkpoint, &config, null);
    defer binding.deinit(gpa);
    var model = try Model.init(gpa, io, &binding, &config);
    defer model.deinit();

    var prompt = try tts.Prompt.build(gpa, &config, .{ .role = .{ 20, 7, 1 }, .text = &.{ 30, 31 }, .instruct = &.{ 20, 9, 21 }, .speaker = "ryan", .language = "english" });
    defer prompt.deinit(gpa);
    // Prefilled at once or row by row, the talker's rows agree.
    const w = config.talker.hidden;
    const n = prompt.rows.len;
    const x = try gpa.alloc(f32, 2 * n * w);
    defer gpa.free(x);
    const hidden = try gpa.alloc(f32, 2 * n * w);
    defer gpa.free(hidden);
    try model.embed(gpa, prompt.rows, null, x[0 .. n * w]);
    @memcpy(x[n * w ..], x[0 .. n * w]);
    var whole = try Talker.init(gpa, &model);
    defer whole.deinit();
    try whole.forward(x[0 .. n * w], n, hidden[0 .. n * w], null);
    var stepped = try Talker.init(gpa, &model);
    defer stepped.deinit();
    for (0..n) |i| try stepped.forward(x[(n + i) * w ..][0..w], 1, hidden[(n + i) * w ..][0..w], null);
    for (hidden[0 .. n * w], hidden[n * w ..]) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-5);

    var frames: usize = 0;
    var s = try Synthesizer.init(gpa, io, &config, .{ .cpu = &model }, .none, &prompt, 7);
    defer s.deinit();
    while (try s.nextFrame()) |frame| {
        for (frame) |code| try std.testing.expect(code < tts.codebook_size);
        frames += 1;
    }
    // Six group-0 samples at most, the last without a frame; or the end code earlier.
    try std.testing.expect(frames <= 5);
    try std.testing.expect(s.sampler.sampled >= 2 or frames == 0);
}
