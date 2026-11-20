//! clef-flash's joint schema head on the CPU: from the backbone's final
//! hidden rows (after `output_norm`) and each option's mean output-head row,
//! one logit per option of every question, as Cloudflare's
//! `JointSchemaHead.forward` computes them. The head's tensors live in the
//! clef GGUF beside the backbone (`decision.*`, `dec.blk.*`); they are
//! decoded to F32 once at `open` (about 0.5 GB) and the head owns them. Matmuls go
//! through `cpu.dense` (F32 SIMD lanes, split across `Io` tasks), so results
//! match an F32 reference to rounding. Contract: docs/models/clef-flash.md.
const std = @import("std");
const cpu = @import("../backends/cpu/root.zig");
const dense = cpu.dense;
const profile = @import("../profiles/clef.zig");
const qwen35 = @import("qwen35.zig");
const weights = @import("../runtime/weights.zig");
const gguf = @import("../formats/gguf.zig");
const quant = @import("../quant/decode.zig");
const vocabulary = @import("../tokenizer/vocabulary.zig");
const Encoder = @import("../tokenizer/encode.zig").Encoder;
const metal = @import("../backends/metal/root.zig");
const Backend = @import("laya.zig").Backend;
const vision = @import("../vision/root.zig");

/// The head's shape (`clef.decision.*`); only the released one is accepted.
pub const Config = struct {
    hidden_size: usize,
    width: usize,
    routing_layers: usize,
    layers: usize,
    heads: usize,
    feedforward: usize,
};
pub const released: Config = .{ .hidden_size = 4096, .width = 1024, .routing_layers = 2, .layers = 4, .heads = 16, .feedforward = 4096 };

/// PyTorch's LayerNorm default.
const norm_eps = 1e-5;
/// Queries per attention block, bounding the score scratch to
/// `block × tokens` values.
const query_block = 256;

const Linear = struct {
    w: []const f32,
    b: ?[]const f32,
    outputs: usize,
    inner: usize,

    fn apply(self: Linear, io: std.Io, x: []const f32, rows: usize, y: []f32) !void {
        try dense.matmul(io, .{ .rows = rows, .inner = self.inner, .outputs = self.outputs, .x = x, .w = self.w, .bias = self.b }, y);
    }
};

const Norm = struct {
    w: []const f32,
    b: []const f32,

    fn rows(self: Norm, x: []const f32, out: []f32) !void {
        const width = self.w.len;
        for (0..x.len / width) |r| try dense.layerNorm(x[r * width ..][0..width], self.w, self.b, norm_eps, out[r * width ..][0..width]);
    }
};

/// `nn.MultiheadAttention`: the fused input projection split into the query
/// rows and the key-value rows, so memory is projected once per layer.
const Attention = struct {
    query: Linear,
    key_value: Linear,
    out: Linear,
};

const Routing = struct { query_norm: Norm, memory_norm: Norm, attention: Attention, feedforward_norm: Norm, up: Linear, down: Linear };
const Decoder = struct { norm1: Norm, norm2: Norm, norm3: Norm, self_attention: Attention, cross_attention: Attention, up: Linear, down: Linear };

pub const Head = struct {
    storage: std.heap.ArenaAllocator,
    config: Config,
    hidden_norm: Norm,
    memory_projection: Linear,
    question_projection: Linear,
    option_question_projection: Linear,
    global_projection: Linear,
    option_context_projection: Linear,
    option_lexical_projection: Linear,
    type_embedding: []const f32,
    routing: []Routing,
    option_summary_norm: Norm,
    decoders: []Decoder,
    field_norm: Norm,
    option_norm: Norm,
    scorer_up: Linear,
    scorer_down: Linear,
    /// The forward's factors as stored: exp(prior), exp(joint), sigmoid(gate).
    prior_scale: f32,
    joint_scale: f32,
    residual_gate: f32,

    /// Reads the head from the clef GGUF: its shape from `clef.decision.*`
    /// (the released one only) and every `decision.*`, `dec.blk.*`, and
    /// `token_types` tensor decoded to F32, each bound with its shape.
    pub fn open(gpa: std.mem.Allocator, doc: *const gguf.Document, view: weights.View) !Head {
        var storage: std.heap.ArenaAllocator = .init(gpa);
        errdefer storage.deinit();
        const arena = storage.allocator();
        const config = try readConfig(doc);
        var loader: Loader = .{ .arena = arena, .doc = doc, .view = view };
        const h = config.hidden_size;
        const w = config.width;
        var head: Head = undefined;
        head.config = config;
        head.hidden_norm = try loader.norm("decision.hidden_norm", h);
        head.memory_projection = try loader.linear("decision.proj_memory", w, h, false);
        head.question_projection = try loader.linear("decision.proj_question", w, h, false);
        head.option_question_projection = try loader.linear("decision.proj_option_question", w, h, false);
        head.global_projection = try loader.linear("decision.proj_global", w, h, false);
        head.option_context_projection = try loader.linear("decision.proj_option_context", w, h, false);
        head.option_lexical_projection = try loader.linear("decision.proj_option_lexical", w, h, false);
        head.type_embedding = try loader.tensor("token_types.weight", &.{ 3, w });
        head.routing = try arena.alloc(Routing, config.routing_layers);
        for (head.routing, 0..) |*r, i| {
            const p = try arena.print("dec.blk.{d}.", .{i});
            r.* = .{
                .query_norm = try loader.norm(try cat(arena, p, "cross_attn_norm"), w),
                .memory_norm = try loader.norm(try cat(arena, p, "cross_attn_norm_kv"), w),
                .attention = try loader.attention(try cat(arena, p, "cross_attn"), w),
                .feedforward_norm = try loader.norm(try cat(arena, p, "ffn_norm"), w),
                .up = try loader.linear(try cat(arena, p, "ffn_up"), config.feedforward, w, true),
                .down = try loader.linear(try cat(arena, p, "ffn_down"), w, config.feedforward, true),
            };
        }
        head.option_summary_norm = try loader.norm("decision.option_summary_norm", w);
        head.decoders = try arena.alloc(Decoder, config.layers);
        for (head.decoders, 0..) |*d, i| {
            const p = try arena.print("dec.blk.{d}.", .{config.routing_layers + i});
            d.* = .{
                .norm1 = try loader.norm(try cat(arena, p, "attn_norm"), w),
                .norm2 = try loader.norm(try cat(arena, p, "cross_attn_norm"), w),
                .norm3 = try loader.norm(try cat(arena, p, "ffn_norm"), w),
                .self_attention = try loader.attention(try cat(arena, p, "attn"), w),
                .cross_attention = try loader.attention(try cat(arena, p, "cross_attn"), w),
                .up = try loader.linear(try cat(arena, p, "ffn_up"), config.feedforward, w, true),
                .down = try loader.linear(try cat(arena, p, "ffn_down"), w, config.feedforward, true),
            };
        }
        head.field_norm = try loader.norm("decision.field_norm", w);
        head.option_norm = try loader.norm("decision.option_norm", w);
        head.scorer_up = try loader.linear("decision.scorer", w, 4 * w, true);
        head.scorer_down = .{
            .w = try loader.tensor("decision.scorer_out.weight", &.{w}),
            .b = try loader.tensor("decision.scorer_out.bias", &.{1}),
            .outputs = 1,
            .inner = w,
        };
        // Stored as the forward uses them: exp(prior), exp(joint), sigmoid(gate).
        const scales = try loader.tensor("decision.scales", &.{3});
        head.prior_scale = scales[0];
        head.joint_scale = scales[1];
        head.residual_gate = scales[2];
        if (!(head.prior_scale > 0 and head.joint_scale > 0 and head.residual_gate >= 0 and head.residual_gate <= 1)) return error.InvalidHeadScales;
        if (loader.bound != headTensorCount(doc)) return error.UnexpectedHeadTensor;
        head.storage = storage;
        return head;
    }

    pub fn deinit(self: *Head) void {
        self.storage.deinit();
        self.* = undefined;
    }

    /// One sequence's logits: `hidden` is `[tokens][hidden_size]` after the
    /// backbone's `output_norm`, `lexical` one mean output-head row per
    /// option in field order, `logits[f]` one value per option of field `f`.
    /// `gpa` holds the scratch, freed before returning.
    pub fn forward(self: *const Head, gpa: std.mem.Allocator, io: std.Io, hidden: []const f32, sequence: profile.Sequence, lexical: []const f32, logits: []const []f32) !void {
        const h = self.config.hidden_size;
        const w = self.config.width;
        const tokens = sequence.ids.len;
        const fields = sequence.fields;
        if (tokens == 0 or fields.len == 0 or logits.len != fields.len) return error.InvalidShape;
        var options: usize = 0;
        for (fields, logits) |f, l| {
            if (l.len != f.options.len) return error.InvalidShape;
            options += f.options.len;
        }
        if (hidden.len != tokens * h or lexical.len != options * h) return error.InvalidShape;

        var scratch_state: std.heap.ArenaAllocator = .init(gpa);
        defer scratch_state.deinit();
        const scratch = scratch_state.allocator();

        const normalized = try scratch.alloc(f32, tokens * h);
        try self.hidden_norm.rows(hidden, normalized);
        const memory = try scratch.alloc(f32, tokens * w);
        try self.memory_projection.apply(io, normalized, tokens, memory);
        const global = normalized[(tokens - 1) * h ..][0..h];

        // Span means: one question vector per field, one context per option.
        const questions = try scratch.alloc(f32, fields.len * h);
        const contexts = try scratch.alloc(f32, options * h);
        const owner = try scratch.alloc(usize, options);
        var o: usize = 0;
        for (fields, 0..) |f, fi| {
            try spanMean(normalized, h, f.span, questions[fi * h ..][0..h]);
            for (f.options) |span| {
                try spanMean(normalized, h, span, contexts[o * h ..][0..h]);
                owner[o] = fi;
                o += 1;
            }
        }

        // Option queries: context, lexical, and their question's projection.
        const routed = try scratch.alloc(f32, options * w);
        const term = try scratch.alloc(f32, @max(options, fields.len) * w);
        try self.option_context_projection.apply(io, contexts, options, routed);
        try self.option_lexical_projection.apply(io, lexical, options, term[0 .. options * w]);
        addInto(routed, term[0 .. options * w]);
        const question_terms = try scratch.alloc(f32, fields.len * w);
        try self.option_question_projection.apply(io, questions, fields.len, question_terms);
        for (0..options) |i| addInto(routed[i * w ..][0..w], question_terms[owner[i] * w ..][0..w]);

        const work: Work = .{ .scratch = scratch, .io = io, .heads = self.config.heads, .width = w };
        const memory_normalized = try scratch.alloc(f32, tokens * w);
        for (self.routing) |*layer| {
            try layer.memory_norm.rows(memory, memory_normalized);
            try work.attend(layer.query_norm, layer.attention, routed, options, memory_normalized, tokens);
            try work.feedForward(layer.feedforward_norm, layer.up, layer.down, routed, options);
        }

        // Fields: their projection, a softmax-weighted summary of their
        // routed options, the global vector, and the type embedding.
        const state = try scratch.alloc(f32, fields.len * w);
        try self.question_projection.apply(io, questions, fields.len, state);
        const summaries = try scratch.alloc(f32, fields.len * w);
        @memset(summaries, 0);
        o = 0;
        const inv_sqrt_width: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(w)));
        for (fields, 0..) |f, fi| {
            const base = state[fi * w ..][0..w];
            const routing = try scratch.alloc(f32, f.options.len);
            for (routing, 0..) |*x, k| x.* = dot(routed[(o + k) * w ..][0..w], base) * inv_sqrt_width;
            try cpu.softmax(routing, routing);
            const summary = summaries[fi * w ..][0..w];
            for (routing, 0..) |x, k| for (summary, routed[(o + k) * w ..][0..w]) |*s, r| {
                s.* += x * r;
            };
            o += f.options.len;
        }
        try self.option_summary_norm.rows(summaries, summaries);
        addInto(state, summaries);
        const global_term = try scratch.alloc(f32, w);
        try self.global_projection.apply(io, global, 1, global_term);
        for (fields, 0..) |f, fi| {
            addInto(state[fi * w ..][0..w], global_term);
            addInto(state[fi * w ..][0..w], self.type_embedding[profile.typeId(f.kind) * w ..][0..w]);
        }
        for (self.decoders) |*layer| {
            try work.attend(layer.norm1, layer.self_attention, state, fields.len, null, fields.len);
            try work.attend(layer.norm2, layer.cross_attention, state, fields.len, memory, tokens);
            try work.feedForward(layer.norm3, layer.up, layer.down, state, fields.len);
        }
        try self.field_norm.rows(state, state);

        // Scores: a lexical prior against the question's anchor, plus a gated
        // joint term (cosine and a residual MLP of the field-option pair).
        const prior_scale = self.prior_scale;
        const joint_scale = self.joint_scale;
        const gate = self.residual_gate;
        const anchor = try scratch.alloc(f32, h);
        const option = try scratch.alloc(f32, w);
        const features = try scratch.alloc(f32, 4 * w);
        const hidden_scorer = try scratch.alloc(f32, w);
        o = 0;
        for (logits, 0..) |out, fi| {
            for (anchor, questions[fi * h ..][0..h], global) |*a, q, g| a.* = q + g;
            normalize(anchor);
            const field = state[fi * w ..][0..w];
            for (out) |*logit| {
                const lex = lexical[o * h ..][0..h];
                const prior = prior_scale * dot(lex, anchor) / @max(norm2(lex), 1e-12);
                try self.option_norm.rows(routed[o * w ..][0..w], option);
                const cosine = dot(field, option) / @max(norm2(field) * norm2(option), 1e-8);
                for (0..w) |k| {
                    features[k] = field[k];
                    features[w + k] = option[k];
                    features[2 * w + k] = field[k] * option[k];
                    features[3 * w + k] = @abs(field[k] - option[k]);
                }
                try self.scorer_up.apply(io, features, 1, hidden_scorer);
                for (hidden_scorer) |*x| x.* = cpu.geluErf(x.*);
                var residual: [1]f32 = undefined;
                try self.scorer_down.apply(io, hidden_scorer, 1, &residual);
                logit.* = prior + gate * (joint_scale * cosine + residual[0]);
                o += 1;
            }
        }
    }
};

const Work = struct {
    scratch: std.mem.Allocator,
    io: std.Io,
    heads: usize,
    width: usize,

    /// x += attention(norm(x) as queries, memory as keys and values);
    /// `memory` null attends x to its own normalized rows (self-attention).
    fn attend(self: Work, norm: Norm, a: Attention, x: []f32, rows: usize, memory: ?[]const f32, memory_rows: usize) !void {
        const w = self.width;
        const normalized = try self.scratch.alloc(f32, rows * w);
        try norm.rows(x[0 .. rows * w], normalized);
        const queries = try self.scratch.alloc(f32, rows * w);
        try a.query.apply(self.io, normalized, rows, queries);
        const kv = try self.scratch.alloc(f32, memory_rows * 2 * w);
        try a.key_value.apply(self.io, memory orelse normalized, memory_rows, kv);
        const mixed = try self.scratch.alloc(f32, rows * w);
        try self.multiHead(queries, rows, kv, memory_rows, mixed);
        const out = try self.scratch.alloc(f32, rows * w);
        try a.out.apply(self.io, mixed, rows, out);
        addInto(x[0 .. rows * w], out);
    }

    /// Per head: scores = Q·Kᵀ/√d as one matmul, a row softmax, then
    /// scores·V as another, with K and Vᵀ laid out per head first.
    fn multiHead(self: Work, queries: []const f32, rows: usize, kv: []const f32, keys: usize, out: []f32) !void {
        const w = self.width;
        const d = w / self.heads;
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));
        const k_head = try self.scratch.alloc(f32, keys * d);
        const vt_head = try self.scratch.alloc(f32, d * keys);
        const block = @min(rows, query_block);
        const q_head = try self.scratch.alloc(f32, block * d);
        const scores = try self.scratch.alloc(f32, block * keys);
        const mixed = try self.scratch.alloc(f32, block * d);
        for (0..self.heads) |head| {
            for (0..keys) |t| {
                @memcpy(k_head[t * d ..][0..d], kv[t * 2 * w + head * d ..][0..d]);
                for (0..d) |c| vt_head[c * keys + t] = kv[t * 2 * w + w + head * d + c];
            }
            var first: usize = 0;
            while (first < rows) : (first += block) {
                const n = @min(block, rows - first);
                for (0..n) |r| for (q_head[r * d ..][0..d], queries[(first + r) * w + head * d ..][0..d]) |*q, v| {
                    q.* = v * scale;
                };
                try dense.matmul(self.io, .{ .rows = n, .inner = d, .outputs = keys, .x = q_head[0 .. n * d], .w = k_head }, scores[0 .. n * keys]);
                for (0..n) |r| try cpu.softmax(scores[r * keys ..][0..keys], scores[r * keys ..][0..keys]);
                try dense.matmul(self.io, .{ .rows = n, .inner = keys, .outputs = d, .x = scores[0 .. n * keys], .w = vt_head }, mixed[0 .. n * d]);
                for (0..n) |r| @memcpy(out[(first + r) * w + head * d ..][0..d], mixed[r * d ..][0..d]);
            }
        }
    }

    /// x += down(GELU(up(norm(x)))).
    fn feedForward(self: Work, norm: Norm, up: Linear, down: Linear, x: []f32, rows: usize) !void {
        const w = self.width;
        const normalized = try self.scratch.alloc(f32, rows * w);
        try norm.rows(x[0 .. rows * w], normalized);
        const inner = try self.scratch.alloc(f32, rows * up.outputs);
        try up.apply(self.io, normalized, rows, inner);
        for (inner) |*v| v.* = cpu.geluErf(v.*);
        const out = try self.scratch.alloc(f32, rows * w);
        try down.apply(self.io, inner, rows, out);
        addInto(x[0 .. rows * w], out);
    }
};

/// Decodes the head's tensors from the GGUF to F32 and counts them.
const Loader = struct {
    arena: std.mem.Allocator,
    doc: *const gguf.Document,
    view: weights.View,
    bound: usize = 0,

    /// Tensor `name` as F32, row-major in `shape` (outputs first: the
    /// reverse of GGUF's dimension order).
    fn tensor(self: *Loader, name: []const u8, shape: []const usize) ![]f32 {
        var count: usize = 1;
        for (shape) |d| count *= d;
        const out = try self.arena.alloc(f32, count);
        try self.into(name, shape, out);
        return out;
    }

    fn into(self: *Loader, name: []const u8, shape: []const usize, out: []f32) !void {
        const t = findTensor(self.doc, name) orelse return error.MissingHeadTensor;
        if (t.dimensions.len != shape.len) return error.InvalidHeadTensorShape;
        for (t.dimensions, 0..) |d, i| if (d != shape[shape.len - 1 - i]) return error.InvalidHeadTensorShape;
        if (t.elements != out.len) return error.InvalidHeadTensorShape;
        const bytes = try self.view.bytes(t);
        const columns = shape[shape.len - 1];
        const rows = out.len / columns;
        if (bytes.len % rows != 0) return error.InvalidHeadTensorShape;
        const width = bytes.len / rows;
        for (0..rows) |r| try quant.row(t.encoding_id, bytes[r * width ..][0..width], out[r * columns ..][0..columns]);
        self.bound += 1;
    }

    fn norm(self: *Loader, prefix: []const u8, width: usize) !Norm {
        return .{ .w = try self.tensor(try cat(self.arena, prefix, ".weight"), &.{width}), .b = try self.tensor(try cat(self.arena, prefix, ".bias"), &.{width}) };
    }

    fn linear(self: *Loader, prefix: []const u8, outputs: usize, inner: usize, bias: bool) !Linear {
        return .{
            .w = try self.tensor(try cat(self.arena, prefix, ".weight"), &.{ outputs, inner }),
            .b = if (bias) try self.tensor(try cat(self.arena, prefix, ".bias"), &.{outputs}) else null,
            .outputs = outputs,
            .inner = inner,
        };
    }

    /// `<prefix>_{q,k,v,o}`; the key and value rows go into one buffer, so
    /// memory is projected once per layer.
    fn attention(self: *Loader, prefix: []const u8, w: usize) !Attention {
        const kv_w = try self.arena.alloc(f32, 2 * w * w);
        const kv_b = try self.arena.alloc(f32, 2 * w);
        try self.into(try cat(self.arena, prefix, "_k.weight"), &.{ w, w }, kv_w[0 .. w * w]);
        try self.into(try cat(self.arena, prefix, "_v.weight"), &.{ w, w }, kv_w[w * w ..]);
        try self.into(try cat(self.arena, prefix, "_k.bias"), &.{w}, kv_b[0..w]);
        try self.into(try cat(self.arena, prefix, "_v.bias"), &.{w}, kv_b[w..]);
        return .{
            .query = try self.linear(try cat(self.arena, prefix, "_q"), w, w, true),
            .key_value = .{ .w = kv_w, .b = kv_b, .outputs = 2 * w, .inner = w },
            .out = try self.linear(try cat(self.arena, prefix, "_o"), w, w, true),
        };
    }
};

fn findTensor(doc: *const gguf.Document, name: []const u8) ?*const gguf.Tensor {
    for (doc.tensors) |*t| if (std.mem.eql(u8, t.name, name)) return t;
    return null;
}

/// The head's tensors in the file, for the all-bound check.
fn headTensorCount(doc: *const gguf.Document) usize {
    var n: usize = 0;
    for (doc.tensors) |t| for (qwen35.decision_prefixes) |p| if (std.mem.startsWith(u8, t.name, p)) {
        n += 1;
        break;
    };
    return n;
}

/// `clef.decision.*`: the released shape only; widths follow the tensors.
fn readConfig(doc: *const gguf.Document) !Config {
    const kind = doc.string("clef.decision.type") orelse return error.MissingHeadConfig;
    if (!std.mem.eql(u8, kind, "clef")) return error.UnsupportedHeadConfig;
    const config: Config = .{
        .hidden_size = try unsigned(doc, "clef.embedding_length"),
        .width = released.width,
        .routing_layers = try unsigned(doc, "clef.decision.routing_block_count"),
        .layers = try unsigned(doc, "clef.decision.block_count"),
        .heads = try unsigned(doc, "clef.decision.head_count"),
        .feedforward = released.feedforward,
    };
    if (!std.meta.eql(config, released)) return error.UnsupportedHeadConfig;
    switch (doc.get("clef.attention.layer_norm_epsilon") orelse return error.MissingHeadConfig) {
        .float => |eps| if (@as(f32, @floatCast(eps)) != norm_eps) return error.UnsupportedHeadConfig,
        else => return error.InvalidHeadConfig,
    }
    return config;
}

fn unsigned(doc: *const gguf.Document, key: []const u8) !usize {
    return switch (doc.get(key) orelse return error.MissingHeadConfig) {
        .unsigned => |v| std.math.cast(usize, v) orelse error.InvalidHeadConfig,
        else => error.InvalidHeadConfig,
    };
}

fn cat(arena: std.mem.Allocator, a: []const u8, b: []const u8) ![]const u8 {
    return std.mem.concat(arena, u8, &.{ a, b });
}

fn spanMean(rows: []const f32, width: usize, span: profile.Span, out: []f32) !void {
    if (span.end <= span.start or span.end * width > rows.len) return error.InvalidShape;
    @memset(out, 0);
    for (span.start..span.end) |r| addInto(out, rows[r * width ..][0..width]);
    const inv = 1.0 / @as(f32, @floatFromInt(span.end - span.start));
    for (out) |*v| v.* *= inv;
}

fn addInto(x: []f32, y: []const f32) void {
    for (x, y) |*a, b| a.* += b;
}

fn dot(a: []const f32, b: []const f32) f32 {
    var sum: f64 = 0;
    for (a, b) |x, y| sum += @as(f64, x) * y;
    return @floatCast(sum);
}

fn norm2(x: []const f32) f32 {
    return @sqrt(dot(x, x));
}

/// `F.normalize`: x / max(‖x‖, 1e-12).
fn normalize(x: []f32) void {
    const n = @max(norm2(x), 1e-12);
    for (x) |*v| v.* /= n;
}

/// `lexical[o]` = the mean of `output.weight`'s rows over option `o`'s span
/// tokens, in field order: `row(id, out)` writes one row of `hidden` values.
pub fn lexicalRows(sequence: profile.Sequence, hidden: usize, context: anytype, comptime row: fn (@TypeOf(context), u32, []f32) anyerror!void, scratch: []f32, out: []f32) !void {
    if (scratch.len != hidden or out.len != optionCount(sequence) * hidden) return error.InvalidShape;
    for (sequence.fields) |f| for (f.options) |span| {
        if (span.end <= span.start or span.end > sequence.ids.len) return error.InvalidShape;
    };
    var o: usize = 0;
    for (sequence.fields) |f| for (f.options) |span| {
        const mean = out[o * hidden ..][0..hidden];
        @memset(mean, 0);
        for (sequence.ids[span.start..span.end]) |id| {
            try row(context, id, scratch);
            addInto(mean, scratch);
        }
        const inv = 1.0 / @as(f32, @floatFromInt(span.end - span.start));
        for (mean) |*v| v.* *= inv;
        o += 1;
    };
}

/// Options over every field of a sequence.
pub fn optionCount(sequence: profile.Sequence) usize {
    var n: usize = 0;
    for (sequence.fields) |f| n += f.options.len;
    return n;
}

/// The head's weights beside the support files; its presence marks a clef
/// checkpoint directory.
/// clef-flash on one backend from its one GGUF: the backbone (the qwen35
/// family under the `clef` id), the head, and the tokenizer; and when given,
/// the vision projector. Owns the mappings, the executors, and the head; one sequence
/// at a time.
pub const Model = struct {
    gpa: std.mem.Allocator,
    storage: std.heap.ArenaAllocator,
    mapped: *weights.Mapped,
    vocab: *vocabulary.Vocabulary,
    tokens: *Encoder,
    binding: qwen35.Binding,
    backbone: union(Backend) {
        cpu: *qwen35.family.Runtime,
        metal: struct { backend: *metal.Backend, plan: *qwen35.family.Plan },
    },
    head: Head,
    frame: profile.Frame,
    /// The projector, when the model was opened with one.
    projector: ?*Projector,

    const Projector = struct { mapped: weights.Mapped, projector: vision.Projector };

    /// `path` is the clef GGUF (backbone, head, and tokenizer); `mmproj`
    /// the projector, without which a request with images fails.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, path: []const u8, mmproj: ?[]const u8, backend: Backend) !Model {
        var storage: std.heap.ArenaAllocator = .init(gpa);
        errdefer storage.deinit();
        const arena = storage.allocator();
        const mapped = try arena.create(weights.Mapped);
        mapped.* = try weights.Mapped.open(gpa, io, path);
        errdefer mapped.deinit(io);
        const arch = mapped.document.string("general.architecture") orelse return error.MissingMetadata;
        if (!std.mem.eql(u8, arch, qwen35.clef_architecture)) return error.NotClef;
        const binding = try qwen35.bind(arena, &mapped.document);
        var head = try Head.open(gpa, &mapped.document, mapped.view());
        errdefer head.deinit();
        if (binding.shape.hidden != head.config.hidden_size) return error.HeadDoesNotFitBackbone;
        const vocab = try arena.create(vocabulary.Vocabulary);
        vocab.* = try vocabulary.load(gpa, mapped.document, mapped.mapping.memory[0..@intCast(mapped.document.directory_bytes)], .{});
        errdefer vocab.deinit();
        const tokens = try arena.create(Encoder);
        tokens.* = try Encoder.init(gpa, vocab);
        errdefer tokens.deinit();
        const frame = try profile.Frame.init(arena, tokens);
        const view = mapped.view();
        const capacity = profile.max_length;
        var model: Model = .{ .gpa = gpa, .storage = undefined, .mapped = mapped, .vocab = vocab, .tokens = tokens, .binding = binding, .backbone = undefined, .head = head, .frame = frame, .projector = null };
        switch (backend) {
            .cpu => {
                const runtime = try arena.create(qwen35.family.Runtime);
                runtime.* = try qwen35.family.Runtime.init(gpa, io, view, binding, capacity, false, false);
                model.backbone = .{ .cpu = runtime };
            },
            .metal => {
                const gpu = try arena.create(metal.Backend);
                var diagnostic: [8192]u8 = @splat(0);
                gpu.* = metal.Backend.init(gpa, &diagnostic) catch |err| {
                    const reason = std.mem.sliceTo(&diagnostic, 0);
                    if (reason.len > 0) std.log.err("{s}", .{reason});
                    return err;
                };
                errdefer gpu.deinit();
                const plan = try arena.create(qwen35.family.Plan);
                plan.* = try qwen35.family.Plan.init(gpa, gpu, view, binding, capacity, chunk, .f16, false, false);
                model.backbone = .{ .metal = .{ .backend = gpu, .plan = plan } };
            },
        }
        errdefer model.closeBackbone();
        if (mmproj) |p| {
            const v = try arena.create(Projector);
            v.mapped = try weights.Mapped.open(gpa, io, p);
            errdefer v.mapped.deinit(io);
            const gpu = switch (model.backbone) {
                .cpu => null,
                .metal => |m| m.backend,
            };
            try v.projector.init(gpa, io, &v.mapped.document, v.mapped.view(), gpu);
            if (v.projector.outputWidth() != binding.shape.hidden) {
                v.projector.deinit();
                return error.VisionSourceMismatch;
            }
            model.projector = v;
        }
        model.storage = storage;
        return model;
    }

    fn closeBackbone(self: *Model) void {
        switch (self.backbone) {
            .cpu => |r| r.deinit(),
            .metal => |m| {
                m.plan.deinit();
                m.backend.deinit();
            },
        }
    }

    pub fn deinit(self: *Model, io: std.Io) void {
        if (self.projector) |v| {
            v.projector.deinit();
            v.mapped.deinit(io);
        }
        self.closeBackbone();
        self.head.deinit();
        self.tokens.deinit();
        self.vocab.deinit();
        self.mapped.deinit(io);
        self.storage.deinit();
        self.* = undefined;
    }

    pub fn encoder(self: *const Model) *const Encoder {
        return self.tokens;
    }

    pub fn hiddenSize(self: *const Model) usize {
        return self.binding.shape.hidden;
    }

    /// The backbone's final hidden rows of `ids` (`ids.len × hidden`);
    /// `spans` place `features` rows (images) as the chat path does.
    pub fn hiddenRows(self: *const Model, ids: []const u32, spans: []const vision.Span, features: []const f32, out: []f32) !void {
        switch (self.backbone) {
            .cpu => |r| try r.hiddenRows(ids, spans, features, out),
            .metal => |m| try m.plan.prefillHidden(ids, spans, features, out),
        }
    }

    /// An image's token grid by clef's processor bounds: at least
    /// `profile.min_image_pixels`, at most the projector plan's 1,024 tokens
    /// (the reference allows 16,384; a larger image is scaled down further).
    pub fn imageGrid(self: *const Model, size: vision.preprocess.Size) !vision.Grid {
        _ = self.projector orelse return error.NoVision;
        const qwen3vl = vision.qwen3vl;
        const target = vision.preprocess.smartSize(size, .{ .align_size = qwen3vl.patch * qwen3vl.merge, .min_pixels = profile.min_image_pixels, .max_pixels = qwen3vl.max_tokens * qwen3vl.token_pixels });
        return .{ .width_tokens = target.width / (qwen3vl.patch * qwen3vl.merge), .height_tokens = target.height / (qwen3vl.patch * qwen3vl.merge) };
    }

    /// Projects `images` (decoded, with their grids) into `features`
    /// (`Σ tokens × hidden`) and returns their spans: image `i`'s tokens
    /// start at `starts[i]` in the sequence.
    pub fn projectImages(self: *const Model, arena: std.mem.Allocator, images: []const Image, starts: []const usize, features: []f32) ![]vision.Span {
        const v = self.projector orelse return error.NoVision;
        const h = self.hiddenSize();
        const spans = try arena.alloc(vision.Span, images.len);
        var row: usize = 0;
        for (images, starts, spans) |img, start, *span| {
            var patches = try v.projector.prepareStretched(self.gpa, img.pixels, img.grid);
            defer patches.deinit(self.gpa);
            const n = img.grid.tokens();
            try v.projector.encode(patches, img.grid, features[row * h ..][0 .. n * h]);
            span.* = .{ .start = start, .count = n, .width_tokens = img.grid.width_tokens, .height_tokens = img.grid.height_tokens };
            row += n;
        }
        return spans;
    }

    /// Each option's mean `output.weight` row, in field order.
    pub fn lexical(self: *const Model, sequence: profile.Sequence, scratch: []f32, out: []f32) !void {
        try lexicalRows(sequence, self.hiddenSize(), self, outputRow, scratch, out);
    }

    fn outputRow(self: *const Model, id: u32, out: []f32) anyerror!void {
        try self.mapped.view().row(self.binding.output, id, out);
    }

    /// One sequence through the backbone and the head; `out[f]` holds field
    /// `f`'s logits in the model's option order. `images` are the ones whose
    /// tokens follow the prefix. Scratch from `gpa`.
    pub fn logits(self: *const Model, io: std.Io, sequence: profile.Sequence, images: []const Image, out: []const []f32) !void {
        const h = self.hiddenSize();
        const hidden = try self.gpa.alloc(f32, sequence.ids.len * h);
        defer self.gpa.free(hidden);
        var arena_state: std.heap.ArenaAllocator = .init(self.gpa);
        defer arena_state.deinit();
        const arena = arena_state.allocator();
        var tokens: usize = 0;
        const starts = try arena.alloc(usize, images.len);
        // Each image's pads follow its `<|vision_start|>`.
        var at = self.frame.prefix.len;
        for (images, starts) |img, *start| {
            start.* = at + 1;
            at += img.grid.tokens() + 2;
            tokens += img.grid.tokens();
        }
        const features = try arena.alloc(f32, tokens * h);
        const spans = try self.projectImages(arena, images, starts, features);
        try self.hiddenRows(sequence.ids, spans, features, hidden);
        const lex = try self.gpa.alloc(f32, optionCount(sequence) * h);
        defer self.gpa.free(lex);
        const scratch = try self.gpa.alloc(f32, h);
        defer self.gpa.free(scratch);
        try self.lexical(sequence, scratch, lex);
        try self.head.forward(self.gpa, io, hidden, sequence, lex, out);
    }
};

/// A decoded image and the grid it is projected at (`Model.imageGrid`).
pub const Image = struct { pixels: vision.image.Rgb8, grid: vision.Grid };

/// Prompt tokens per Metal command buffer: 512 measured 291 tok/s on
/// clef against about 260 at the engine's 256.
const chunk = 512;

test "the head refuses a sequence it cannot score before reading a weight" {
    // Only `config` is read before the shape checks.
    var head: Head = undefined;
    head.config = released;
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const hidden = try gpa.alloc(f32, released.hidden_size);
    defer gpa.free(hidden);
    const span: profile.Span = .{ .start = 0, .end = 1 };
    const fields = [_]profile.Field{.{ .kind = .choice, .span = span, .options = &.{span} }};
    var one: [1]f32 = undefined;
    const logits = [_][]f32{&one};
    const empty: profile.Sequence = .{ .ids = &.{}, .fields = &fields, .state_offset = 0, .state_tokens = 0, .truncated = false };
    try std.testing.expectError(error.InvalidShape, head.forward(gpa, io, &.{}, empty, hidden, &logits));
    const sequence: profile.Sequence = .{ .ids = &.{7}, .fields = &fields, .state_offset = 0, .state_tokens = 1, .truncated = false };
    try std.testing.expectError(error.InvalidShape, head.forward(gpa, io, hidden, sequence, hidden, &.{}));
    try std.testing.expectError(error.InvalidShape, head.forward(gpa, io, hidden, sequence, hidden, &.{ &one, &one }));
    const no_fields: profile.Sequence = .{ .ids = &.{7}, .fields = &.{}, .state_offset = 0, .state_tokens = 1, .truncated = false };
    try std.testing.expectError(error.InvalidShape, head.forward(gpa, io, hidden, no_fields, &.{}, &.{}));
}

test "lexical rows refuse spans outside the sequence" {
    const Rows = struct {
        fn row(_: void, id: u32, out: []f32) anyerror!void {
            @memset(out, @floatFromInt(id));
        }
    };
    var scratch: [2]f32 = undefined;
    var out: [2]f32 = undefined;
    const inside = [_]profile.Field{.{ .kind = .choice, .span = .{ .start = 0, .end = 1 }, .options = &.{.{ .start = 0, .end = 2 }} }};
    try lexicalRows(.{ .ids = &.{ 2, 4 }, .fields = &inside, .state_offset = 0, .state_tokens = 2, .truncated = false }, 2, {}, Rows.row, &scratch, &out);
    try std.testing.expectEqual(@as(f32, 3), out[0]);
    try std.testing.expectError(error.InvalidShape, lexicalRows(.{ .ids = &.{2}, .fields = &inside, .state_offset = 0, .state_tokens = 1, .truncated = false }, 2, {}, Rows.row, &scratch, &out));
    const empty_span = [_]profile.Field{.{ .kind = .choice, .span = .{ .start = 0, .end = 1 }, .options = &.{.{ .start = 1, .end = 1 }} }};
    try std.testing.expectError(error.InvalidShape, lexicalRows(.{ .ids = &.{ 2, 4 }, .fields = &empty_span, .state_offset = 0, .state_tokens = 2, .truncated = false }, 2, {}, Rows.row, &scratch, &out));
    var short: [1]f32 = undefined;
    try std.testing.expectError(error.InvalidShape, lexicalRows(.{ .ids = &.{ 2, 4 }, .fields = &inside, .state_offset = 0, .state_tokens = 2, .truncated = false }, 2, {}, Rows.row, &scratch, &short));
}
