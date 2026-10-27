//! The CPU reference of Qwen3-TTS's talker and code predictor, and the
//! `Synthesizer` that drives them into the codec decoder. `Model.init`
//! decodes every bound weight but the text embedding to F32 (about 2.4 GB
//! for the 0.6B, 6.4 GB for the 1.7B; text rows are decoded from the
//! checkpoint as ids need them, so it must outlive the model). Projections
//! are dense.zig's F32 matmul; norms, RoPE, and attention keep F64
//! statistics. The equations: docs/models/qwen3-tts.md § Forward pass.
const std = @import("std");
const safetensors = @import("../formats/safetensors.zig");
const cpu = @import("../backends/cpu/root.zig");
const sampling = @import("../sampling/root.zig");
const codec = @import("../audio/qwen3_codec.zig");
const tts = @import("qwen3_tts.zig");
const hf_json = @import("../tokenizer/hf_json.zig");

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
    /// the projected text row plus the codec row, where present.
    pub fn embed(self: *const Model, gpa: Allocator, rows: []const tts.Row, out: []f32) !void {
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
        }
    }

    /// The next talker row (§ (e)): the frame's 16 code embeddings summed,
    /// then the text track `text` (the projected `tts_pad` row).
    pub fn nextRow(self: *const Model, frame: Frame, text: []const f32, out: []f32) void {
        const w = self.width();
        @memcpy(out, self.codec_embedding[@as(usize, frame[0]) * w ..][0..w]);
        for (self.embeddings, frame[1..]) |table, code| for (out, table[@as(usize, code) * w ..][0..w]) |*a, b| {
            a.* += b;
        };
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

/// One utterance: the prompt's prefill, then frame after frame of codes
/// (talker, sampler, predictor) until the end code or the length limit,
/// decoded through a codec `Stream` when given a codec runtime. Borrows the
/// model and the codec runtime.
pub const Synthesizer = struct {
    gpa: Allocator,
    model: *const Model,
    talker: Talker,
    predictor: Predictor,
    sampler: Sampler,
    stream: ?codec.Stream,
    /// The last post-norm talker row, its logits, the next input row, the
    /// projected `tts_pad` row.
    hidden: []f32,
    logits: []f32,
    row: []f32,
    pad: []f32,
    frames: usize = 0,
    done: bool = false,

    /// Embeds and prefills `prompt`; the first frame's logits are ready after.
    pub fn init(gpa: Allocator, model: *const Model, codec_runtime: ?*const codec.Runtime, prompt: []const tts.Row, seed: u64) !Synthesizer {
        const config = model.config;
        const w = config.talker.hidden;
        var self: Synthesizer = undefined;
        self.gpa = gpa;
        self.model = model;
        self.frames = 0;
        self.done = false;
        self.talker = try .init(gpa, model);
        errdefer self.talker.deinit();
        self.predictor = try .init(gpa, model);
        errdefer self.predictor.deinit();
        self.sampler = try .init(gpa, seed, config.generation, config.codec.eos);
        errdefer self.sampler.deinit(gpa);
        self.stream = if (codec_runtime) |r| try codec.Stream.init(gpa, r) else null;
        errdefer if (self.stream) |*s| s.deinit();
        self.hidden = try gpa.alloc(f32, w);
        errdefer gpa.free(self.hidden);
        self.logits = try gpa.alloc(f32, config.codec_vocab);
        errdefer gpa.free(self.logits);
        self.row = try gpa.alloc(f32, w);
        errdefer gpa.free(self.row);
        self.pad = try gpa.alloc(f32, w);
        errdefer gpa.free(self.pad);
        try model.embed(gpa, &.{.{ .text = config.text.tts_pad, .codec = null }}, self.pad);

        const x = try gpa.alloc(f32, prompt.len * w);
        defer gpa.free(x);
        const hidden = try gpa.alloc(f32, prompt.len * w);
        defer gpa.free(hidden);
        try model.embed(gpa, prompt, x);
        try self.talker.forward(x, prompt.len, hidden, null);
        @memcpy(self.hidden, hidden[(prompt.len - 1) * w ..][0..w]);
        try self.talker.logits(self.hidden, self.logits);
        return self;
    }

    pub fn deinit(self: *Synthesizer) void {
        for ([_][]f32{ self.hidden, self.logits, self.row, self.pad }) |b| self.gpa.free(b);
        if (self.stream) |*s| s.deinit();
        self.sampler.deinit(self.gpa);
        self.predictor.deinit();
        self.talker.deinit();
        self.* = undefined;
    }

    /// The next frame's codes, or null at the end code or when the talker
    /// has sampled `max_new_tokens` codes (the last of which has no frame).
    pub fn nextFrame(self: *Synthesizer) !?Frame {
        if (self.done) return null;
        const c0 = try self.sampler.groupZero(self.logits);
        if (c0 == self.model.config.codec.eos or self.sampler.sampled >= self.model.config.generation.max_new_tokens) {
            self.done = true;
            return null;
        }
        const frame = try self.predictor.frame(self.hidden, @intCast(c0), self.sampler.pick());
        self.model.nextRow(frame, self.pad, self.row);
        try self.talker.forward(self.row, 1, self.hidden, null);
        try self.talker.logits(self.hidden, self.logits);
        self.frames += 1;
        return frame;
    }

    /// Generates up to `out.len / samplesPerFrame` frames and decodes them
    /// into `out`; returns the samples written, 0 once the utterance ended.
    /// Needs the codec runtime.
    pub fn read(self: *Synthesizer, out: []f32) !usize {
        const stream = if (self.stream) |*s| s else return error.NoCodec;
        const per = stream.runtime.config.samplesPerFrame();
        const capacity = out.len / per;
        if (capacity == 0) return error.InvalidShape;
        var frames: [32]Frame = undefined;
        var n: usize = 0;
        while (n < @min(capacity, frames.len)) {
            frames[n] = (try self.nextFrame()) orelse break;
            n += 1;
        }
        if (n == 0) return 0;
        try stream.push(frames[0..n], out[0 .. n * per]);
        return n * per;
    }
};

/// A checkpoint directory opened whole: the config, the tokenizer, the
/// talker and predictor, and the codec decoder of `speech_tokenizer/`. Heap
/// allocated so the model's borrows stay put (`open`, `close`).
pub const Voice = struct {
    gpa: Allocator,
    config: tts.Config,
    tokenizer: hf_json.Tokenizer,
    checkpoint: safetensors.Checkpoint,
    binding: tts.Binding,
    model: Model,
    codec_checkpoint: safetensors.Checkpoint,
    codec_binding: codec.Binding,
    codec: codec.Runtime,

    /// Opens `dir`; a binding error names its tensor in `diagnostic`.
    pub fn open(gpa: Allocator, io: std.Io, dir: []const u8, diagnostic: ?*tts.Diagnostic) !*Voice {
        const self = try gpa.create(Voice);
        errdefer gpa.destroy(self);
        self.gpa = gpa;
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
        self.model = try .init(gpa, io, &self.binding, &self.config);
        errdefer self.model.deinit();
        self.codec_checkpoint = try .open(gpa, io, try std.fs.path.join(a, &.{ dir, "speech_tokenizer", "model.safetensors" }), .{});
        errdefer self.codec_checkpoint.deinit(io);
        var codec_diagnostic: codec.Diagnostic = .{};
        self.codec_binding = codec.bind(gpa, &self.codec_checkpoint, .{}, &codec_diagnostic) catch |err| {
            if (diagnostic) |d| tts.Diagnostic.set(d, codec_diagnostic.name());
            return err;
        };
        errdefer self.codec_binding.deinit(gpa);
        self.codec = try .init(gpa, io, &self.codec_binding);
        return self;
    }

    pub fn close(self: *Voice, io: std.Io) void {
        self.codec.deinit();
        self.codec_binding.deinit(self.gpa);
        self.codec_checkpoint.deinit(io);
        self.model.deinit();
        self.binding.deinit(self.gpa);
        self.checkpoint.deinit(io);
        self.tokenizer.deinit();
        self.config.deinit();
        self.gpa.destroy(self);
    }

    /// The prompt for `text` spoken by `speaker` in `language`, under
    /// `instruct` (the 1.7B's only: `error.InstructionUnsupported` otherwise).
    pub fn prompt(self: *const Voice, gpa: Allocator, text: []const u8, speaker: ?[]const u8, language: []const u8, instruct: ?[]const u8) !tts.Prompt {
        if (instruct) |i| if (i.len > 0 and !self.config.takesInstruction()) return error.InstructionUnsupported;
        var ids = try tts.encodeText(gpa, &self.tokenizer, text, instruct);
        defer ids.deinit(gpa);
        return tts.Prompt.build(gpa, &self.config, .{ .role = ids.role, .text = ids.text, .instruct = ids.instruct, .speaker = speaker, .language = language });
    }

    /// A synthesizer for `prompt` that decodes into samples.
    pub fn synthesizer(self: *const Voice, gpa: Allocator, prompt_rows: []const tts.Row, seed: u64) !Synthesizer {
        return Synthesizer.init(gpa, &self.model, &self.codec, prompt_rows, seed);
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
    try model.embed(gpa, prompt.rows, x[0 .. n * w]);
    @memcpy(x[n * w ..], x[0 .. n * w]);
    var whole = try Talker.init(gpa, &model);
    defer whole.deinit();
    try whole.forward(x[0 .. n * w], n, hidden[0 .. n * w], null);
    var stepped = try Talker.init(gpa, &model);
    defer stepped.deinit();
    for (0..n) |i| try stepped.forward(x[(n + i) * w ..][0..w], 1, hidden[(n + i) * w ..][0..w], null);
    for (hidden[0 .. n * w], hidden[n * w ..]) |a, b| try std.testing.expectApproxEqAbs(a, b, 1e-5);

    var frames: usize = 0;
    var s = try Synthesizer.init(gpa, &model, null, prompt.rows, 7);
    defer s.deinit();
    while (try s.nextFrame()) |frame| {
        for (frame) |code| try std.testing.expect(code < tts.codebook_size);
        frames += 1;
    }
    // Six group-0 samples at most, the last without a frame; or the end code earlier.
    try std.testing.expect(frames <= 5);
    try std.testing.expect(s.sampler.sampled >= 2 or frames == 0);
}
