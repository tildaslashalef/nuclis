//! Laya: a ModernBERT encoder with a typed-decision head, from one checkpoint
//! directory (`encoder/config.json`, `model.safetensors`). `logits` scores one
//! already built sequence: the encoder, the question type's embedding on every
//! row, two pre-norm transformer layers, and at each option's marker a scorer
//! giving that option's logit. Sequence building and calibration are the
//! profile's (profiles/laya.zig). F32 on the CPU; the head's weights are
//! decoded at open like the encoder's. Contract: docs/models/laya.md.
const std = @import("std");
const alloc_check = @import("../alloc_check.zig");
const safetensors = @import("../formats/safetensors.zig");
const dense = @import("../backends/cpu/dense.zig");
const vector = @import("../backends/cpu/vector.zig");
const modernbert = @import("modernbert.zig");
const runtime = @import("modernbert_runtime.zig");
const metal_plan = @import("laya_metal.zig");

pub const Error = runtime.Error || modernbert.Error || error{ UnexpectedTensor, InvalidMarker, InvalidQuestionType };

/// PyTorch's `LayerNorm` default, which the head and scorer keep.
const head_eps = 1e-5;
const head_width = 64;

/// The question type; its value indexes `type_emb`.
pub const Kind = @import("../profiles/laya.zig").Kind;

/// A stage of the forward, for checks against a reference.
pub const Stage = union(enum) {
    /// After encoder layer i, before `final_norm`.
    encoder: usize,
    /// The encoder's output, after `final_norm`.
    final,
    /// After head layer i, every row.
    head: usize,
};

pub const Trace = struct {
    context: *anyopaque,
    record: *const fn (context: *anyopaque, stage: Stage, hidden: []const f32) void,
};

const HeadLayer = struct {
    norm1_w: []const f32,
    norm1_b: []const f32,
    in_proj_w: []const f32,
    in_proj_b: []const f32,
    out_proj_w: []const f32,
    out_proj_b: []const f32,
    norm2_w: []const f32,
    norm2_b: []const f32,
    linear1_w: []const f32,
    linear1_b: []const f32,
    linear2_w: []const f32,
    linear2_b: []const f32,
};

/// The head's tensors, bound by name and shape: `refs` holds `type_emb`,
/// then `layer_tensors.len` per layer in that order, then `scorer_tensors`.
/// Borrows the checkpoint; `refs` is `gpa`'s.
pub const HeadRefs = struct {
    refs: []safetensors.Ref,
    layers: usize,
    ff: usize,
    /// Checkpoint tensors bound.
    count: usize,

    pub const layer_tensors = [_][]const u8{
        "norm1.weight", "norm1.bias", "self_attn.in_proj_weight", "self_attn.in_proj_bias", "self_attn.out_proj.weight", "self_attn.out_proj.bias",
        "norm2.weight", "norm2.bias", "linear1.weight",           "linear1.bias",           "linear2.weight",            "linear2.bias",
    };
    pub const scorer_tensors = [_][]const u8{ "scorer.0.weight", "scorer.0.bias", "scorer.1.weight", "scorer.1.bias", "scorer.3.weight", "scorer.3.bias" };

    /// Binds `type_emb`, `head.layers.N.*` (as many as are present), and `scorer.*`.
    pub fn bind(gpa: std.mem.Allocator, checkpoint: *const safetensors.Checkpoint, hidden: usize) Error!HeadRefs {
        const d = hidden;
        var binder: modernbert.Binder = .{ .checkpoint = checkpoint, .prefix = "" };
        var layer_count: usize = 0;
        while (layer_count < 16) : (layer_count += 1) {
            var buffer: [64]u8 = undefined;
            const name = std.fmt.bufPrint(&buffer, "head.layers.{d}.norm1.weight", .{layer_count}) catch unreachable;
            if (checkpoint.get(name) == null) break;
        }
        if (layer_count == 0) return error.MissingTensor;
        const linear1 = checkpoint.get("head.layers.0.linear1.weight") orelse return error.MissingTensor;
        if (linear1.tensor.shape.len != 2) return error.TensorShapeMismatch;
        const ff: usize = @intCast(linear1.tensor.shape[0]);

        const refs = try gpa.alloc(safetensors.Ref, 1 + layer_tensors.len * layer_count + scorer_tensors.len);
        errdefer gpa.free(refs);
        refs[0] = try binder.get("", "type_emb.weight", &.{ 3, d });
        const layer_shapes = [layer_tensors.len][]const u64{ &.{d}, &.{d}, &.{ 3 * d, d }, &.{3 * d}, &.{ d, d }, &.{d}, &.{d}, &.{d}, &.{ ff, d }, &.{ff}, &.{ d, ff }, &.{d} };
        for (0..layer_count) |i| {
            var buffer: [64]u8 = undefined;
            const at = std.fmt.bufPrint(&buffer, "head.layers.{d}.", .{i}) catch unreachable;
            for (layer_tensors, layer_shapes, 0..) |name, shape, k| refs[1 + layer_tensors.len * i + k] = try binder.get(at, name, shape);
        }
        const scorer_shapes = [scorer_tensors.len][]const u64{ &.{d}, &.{d}, &.{ d, d }, &.{d}, &.{ 1, d }, &.{1} };
        for (scorer_tensors, scorer_shapes, 0..) |name, shape, k| refs[1 + layer_tensors.len * layer_count + k] = try binder.get("", name, shape);
        return .{ .refs = refs, .layers = layer_count, .ff = ff, .count = binder.count };
    }

    pub fn deinit(self: *HeadRefs, gpa: std.mem.Allocator) void {
        gpa.free(self.refs);
        self.* = undefined;
    }

    /// Layer `i`'s tensor named `layer_tensors[k]`.
    pub fn layer(self: HeadRefs, i: usize, comptime name: []const u8) safetensors.Ref {
        return self.refs[1 + layer_tensors.len * i + comptime index(&layer_tensors, name)];
    }
    pub fn scorer(self: HeadRefs, comptime name: []const u8) safetensors.Ref {
        return self.refs[1 + layer_tensors.len * self.layers + comptime index(&scorer_tensors, name)];
    }
    fn index(comptime names: []const []const u8, comptime name: []const u8) usize {
        for (names, 0..) |n, i| if (std.mem.eql(u8, n, name)) return i;
        @compileError("no head tensor " ++ name);
    }
};

/// The head in F32 on the CPU.
pub const Head = struct {
    gpa: std.mem.Allocator,
    storage: []f32,
    hidden: usize,
    ff: usize,
    type_emb: []const f32,
    layers: []HeadLayer,
    scorer_norm_w: []const f32,
    scorer_norm_b: []const f32,
    scorer_w: []const f32,
    scorer_b: []const f32,
    out_w: []const f32,
    out_b: f32,

    /// Decodes every bound tensor to F32.
    pub fn load(gpa: std.mem.Allocator, bound: HeadRefs, hidden: usize) Error!Head {
        const refs = bound.refs;
        var total: usize = 0;
        for (refs) |r| total += @intCast(r.tensor.elements);
        const storage = try gpa.alloc(f32, total);
        errdefer gpa.free(storage);
        var slices = try gpa.alloc([]const f32, refs.len);
        defer gpa.free(slices);
        var at: usize = 0;
        for (refs, slices) |r, *s| {
            const len: usize = @intCast(r.tensor.elements);
            try r.decode(0, storage[at..][0..len]);
            s.* = storage[at..][0..len];
            at += len;
        }
        const layers = try gpa.alloc(HeadLayer, bound.layers);
        for (layers, 0..) |*l, i| {
            const s = slices[1 + HeadRefs.layer_tensors.len * i ..][0..HeadRefs.layer_tensors.len];
            l.* = .{ .norm1_w = s[0], .norm1_b = s[1], .in_proj_w = s[2], .in_proj_b = s[3], .out_proj_w = s[4], .out_proj_b = s[5], .norm2_w = s[6], .norm2_b = s[7], .linear1_w = s[8], .linear1_b = s[9], .linear2_w = s[10], .linear2_b = s[11] };
        }
        const scorer = slices[1 + HeadRefs.layer_tensors.len * bound.layers ..];
        return .{
            .gpa = gpa,
            .storage = storage,
            .hidden = hidden,
            .ff = bound.ff,
            .type_emb = slices[0],
            .layers = layers,
            .scorer_norm_w = scorer[0],
            .scorer_norm_b = scorer[1],
            .scorer_w = scorer[2],
            .scorer_b = scorer[3],
            .out_w = scorer[4],
            .out_b = scorer[5][0],
        };
    }

    pub fn deinit(self: *Head) void {
        self.gpa.free(self.layers);
        self.gpa.free(self.storage);
        self.* = undefined;
    }

    /// One logit per marker from the encoder's output `hidden` ([n][d]),
    /// which this overwrites with the head's residual stream.
    pub fn logits(self: *const Head, io: std.Io, gpa: std.mem.Allocator, hidden: []f32, kind: Kind, markers: []const usize, out: []f32, trace: ?Trace) Error!void {
        const d = self.hidden;
        if (hidden.len == 0 or hidden.len % d != 0 or out.len != markers.len) return error.InvalidShape;
        const n = hidden.len / d;
        for (markers) |m| if (m >= n) return error.InvalidMarker;
        const heads = d / head_width;
        const block = try gpa.alloc(f32, n * (d + 3 * d + d + d + self.ff) + heads * n);
        defer gpa.free(block);
        const normed = block[0 .. n * d];
        const qkv = block[n * d ..][0 .. 3 * n * d];
        const attended = block[4 * n * d ..][0 .. n * d];
        const projected = block[5 * n * d ..][0 .. n * d];
        const wide = block[6 * n * d ..][0 .. n * self.ff];
        const scores = block[6 * n * d + n * self.ff ..][0 .. heads * n];

        const type_row = self.type_emb[@backingInt(kind) * d ..][0..d];
        for (0..n) |t| for (hidden[t * d ..][0..d], type_row) |*x, e| {
            x.* += e;
        };
        for (self.layers, 0..) |l, i| {
            for (0..n) |t| try dense.layerNorm(hidden[t * d ..][0..d], l.norm1_w, l.norm1_b, head_eps, normed[t * d ..][0..d]);
            try dense.matmul(io, .{ .rows = n, .inner = d, .outputs = 3 * d, .x = normed, .w = l.in_proj_w, .bias = l.in_proj_b }, qkv);
            try dense.attention(io, .{ .tokens = n, .heads = heads, .head_dim = head_width, .scale = 1.0 / @sqrt(@as(f32, head_width)), .stride = 3 * d, .q = qkv, .k = qkv[d..], .v = qkv[2 * d ..] }, attended, scores);
            try dense.matmul(io, .{ .rows = n, .inner = d, .outputs = d, .x = attended, .w = l.out_proj_w, .bias = l.out_proj_b }, projected);
            for (hidden, projected) |*x, p| x.* += p;
            for (0..n) |t| try dense.layerNorm(hidden[t * d ..][0..d], l.norm2_w, l.norm2_b, head_eps, normed[t * d ..][0..d]);
            try dense.matmul(io, .{ .rows = n, .inner = d, .outputs = self.ff, .x = normed, .w = l.linear1_w, .bias = l.linear1_b }, wide);
            for (wide) |*w| w.* = @max(w.*, 0);
            try dense.matmul(io, .{ .rows = n, .inner = self.ff, .outputs = d, .x = wide, .w = l.linear2_w, .bias = l.linear2_b }, projected);
            for (hidden, projected) |*x, p| x.* += p;
            if (trace) |tr| tr.record(tr.context, .{ .head = i }, hidden);
        }
        // The scorer on the marker rows only: LayerNorm, Linear, GELU, Linear to one value.
        const k = markers.len;
        if (k == 0) return;
        for (markers, 0..) |m, r| try dense.layerNorm(hidden[m * d ..][0..d], self.scorer_norm_w, self.scorer_norm_b, head_eps, normed[r * d ..][0..d]);
        try dense.matmul(io, .{ .rows = k, .inner = d, .outputs = d, .x = normed[0 .. k * d], .w = self.scorer_w, .bias = self.scorer_b }, projected[0 .. k * d]);
        for (projected[0 .. k * d]) |*p| p.* = vector.geluErf(p.*);
        try dense.matmul(io, .{ .rows = k, .inner = d, .outputs = 1, .x = projected[0 .. k * d], .w = self.out_w, .bias = &.{self.out_b} }, out);
    }
};

/// Where the forward runs: F32 on the CPU, or the Metal plan (F16 weights
/// read in place, F32 accumulation; laya_metal.zig).
pub const Backend = enum { cpu, metal };

/// One built sequence: `markers` are the positions of its `[MASK]` tokens,
/// `kind` its question type.
pub const Sequence = struct {
    ids: []const u32,
    markers: []const usize,
    kind: Kind,
};

pub const Laya = struct {
    checkpoint: safetensors.Checkpoint,
    config: modernbert.Config,
    head_layers: usize,
    engine: union(Backend) {
        cpu: struct { encoder: runtime.Model, head: Head },
        /// Owns its Metal backend and every device buffer; `gpa`'s.
        metal: *metal_plan.Plan,
    },

    /// Opens `dir`: parses `encoder/config.json` (at most 1 MiB), maps
    /// `model.safetensors`, binds every tensor, rejects any it does not know
    /// (`act_head.*` and `temperature` are known and unused), and builds the
    /// `backend`'s forward: the CPU decodes every weight to F32, the Metal
    /// plan wraps the mapped matrices.
    pub fn open(gpa: std.mem.Allocator, io: std.Io, dir: []const u8, backend: Backend) !Laya {
        const config_path = try std.fs.path.join(gpa, &.{ dir, "encoder", "config.json" });
        defer gpa.free(config_path);
        const config_bytes = try std.Io.Dir.cwd().readFileAlloc(io, config_path, gpa, .limited(1024 * 1024));
        defer gpa.free(config_bytes);
        const config = try modernbert.parseConfig(gpa, config_bytes);
        if (config.hidden % head_width != 0) return error.UnsupportedConfig;
        const weights_path = try std.fs.path.join(gpa, &.{ dir, "model.safetensors" });
        defer gpa.free(weights_path);
        var checkpoint = try safetensors.Checkpoint.open(gpa, io, weights_path, .{});
        errdefer checkpoint.deinit(io);
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        const weights = try modernbert.bind(arena.allocator(), &checkpoint, config, "encoder.");
        var head_refs = try HeadRefs.bind(gpa, &checkpoint, config.hidden);
        defer head_refs.deinit(gpa);
        var unused: usize = 0;
        var names = checkpoint.locations.keyIterator();
        while (names.next()) |name| {
            if (std.mem.startsWith(u8, name.*, "act_head.") or std.mem.eql(u8, name.*, "temperature")) unused += 1;
        }
        if (weights.count + head_refs.count + unused != checkpoint.tensorCount()) return error.UnexpectedTensor;
        var self: Laya = .{ .checkpoint = checkpoint, .config = config, .head_layers = head_refs.layers, .engine = undefined };
        switch (backend) {
            .cpu => {
                var head = try Head.load(gpa, head_refs, config.hidden);
                errdefer head.deinit();
                self.engine = .{ .cpu = .{ .encoder = try runtime.Model.load(gpa, weights, config), .head = head } };
            },
            .metal => self.engine = .{ .metal = try metal_plan.Plan.create(gpa, config, weights, head_refs) },
        }
        return self;
    }

    pub fn deinit(self: *Laya, io: std.Io) void {
        switch (self.engine) {
            .cpu => |*c| {
                c.encoder.deinit();
                c.head.deinit();
            },
            .metal => |plan| plan.destroy(),
        }
        self.checkpoint.deinit(io);
        self.* = undefined;
    }

    /// The raw logit of each option of one built sequence into `out`
    /// (`markers.len` values); `trace` sees each stage's rows.
    pub fn logits(self: *const Laya, io: std.Io, gpa: std.mem.Allocator, ids: []const u32, markers: []const usize, kind: Kind, out: []f32, trace: ?Trace) !void {
        switch (self.engine) {
            .cpu => |*c| {
                const hidden = try gpa.alloc(f32, ids.len * self.config.hidden);
                defer gpa.free(hidden);
                const Adapter = struct {
                    fn record(context: *anyopaque, layer: ?usize, states: []const f32) void {
                        const t: *const Trace = @ptrCast(@alignCast(context));
                        t.record(t.context, if (layer) |i| .{ .encoder = i } else .final, states);
                    }
                };
                var outer = trace;
                const inner: ?runtime.Trace = if (outer) |*t| .{ .context = @ptrCast(t), .record = Adapter.record } else null;
                try c.encoder.forward(io, gpa, ids, hidden, inner);
                try c.head.logits(io, gpa, hidden, kind, markers, out, trace);
            },
            .metal => |plan| try plan.run(&.{.{ .ids = ids, .markers = markers, .kind = kind }}, &.{out}, trace),
        }
    }

    /// `logits` of every sequence, `outs[i]` for `sequences[i]`. The Metal
    /// plan packs consecutive sequences into batches of at most
    /// `metal_plan.max_rows` rows; the CPU runs them one at a time.
    pub fn logitsBatch(self: *const Laya, io: std.Io, gpa: std.mem.Allocator, sequences: []const Sequence, outs: []const []f32) !void {
        if (sequences.len != outs.len) return error.InvalidShape;
        switch (self.engine) {
            .cpu => for (sequences, outs) |s, out| try self.logits(io, gpa, s.ids, s.markers, s.kind, out, null),
            .metal => |plan| {
                var first: usize = 0;
                while (first < sequences.len) {
                    var last = first;
                    var rows: usize = 0;
                    while (last < sequences.len and rows + sequences[last].ids.len <= metal_plan.max_rows) : (last += 1) rows += sequences[last].ids.len;
                    if (last == first) return error.SequenceTooLong;
                    try plan.run(sequences[first..last], outs[first..last], null);
                    first = last;
                }
            },
        }
    }
};

// A tiny checkpoint in Laya's layout: hidden 64 (one head of 64 in the head),
// two encoder layers of eight heads of eight, and the tensors it ignores;
// every matrix a multiple of 8 rows and 64 columns, as the Metal matmul needs.
const tiny_config =
    \\{"model_type":"modernbert","hidden_size":64,"num_attention_heads":8,"num_hidden_layers":2,
    \\ "intermediate_size":64,"vocab_size":20,"local_attention":4,
    \\ "layer_types":["full_attention","sliding_attention"],
    \\ "rope_parameters":{"full_attention":{"rope_theta":160000.0},"sliding_attention":{"rope_theta":10000.0}}}
;

pub const TinyTensor = struct { name: []const u8, shape: []const u64 };

const tiny_tensors = blk: {
    const d = 64;
    const layer = [_]TinyTensor{
        .{ .name = "attn.Wqkv.weight", .shape = &.{ 3 * d, d } },
        .{ .name = "attn.Wo.weight", .shape = &.{ d, d } },
        .{ .name = "mlp_norm.weight", .shape = &.{d} },
        .{ .name = "mlp.Wi.weight", .shape = &.{ 128, d } },
        .{ .name = "mlp.Wo.weight", .shape = &.{ d, 64 } },
    };
    var layers: [2 * layer.len]TinyTensor = undefined;
    for (0..2) |i| for (layer, 0..) |t, j| {
        layers[i * layer.len + j] = .{ .name = std.fmt.comptimePrint("encoder.layers.{d}.", .{i}) ++ t.name, .shape = t.shape };
    };
    break :blk [_]TinyTensor{
        .{ .name = "encoder.embeddings.tok_embeddings.weight", .shape = &.{ 20, d } },
        .{ .name = "encoder.embeddings.norm.weight", .shape = &.{d} },
        .{ .name = "encoder.layers.1.attn_norm.weight", .shape = &.{d} },
        .{ .name = "encoder.final_norm.weight", .shape = &.{d} },
    } ++ layers ++ [_]TinyTensor{
        .{ .name = "type_emb.weight", .shape = &.{ 3, d } },
        .{ .name = "head.layers.0.norm1.weight", .shape = &.{d} },
        .{ .name = "head.layers.0.norm1.bias", .shape = &.{d} },
        .{ .name = "head.layers.0.self_attn.in_proj_weight", .shape = &.{ 3 * d, d } },
        .{ .name = "head.layers.0.self_attn.in_proj_bias", .shape = &.{3 * d} },
        .{ .name = "head.layers.0.self_attn.out_proj.weight", .shape = &.{ d, d } },
        .{ .name = "head.layers.0.self_attn.out_proj.bias", .shape = &.{d} },
        .{ .name = "head.layers.0.norm2.weight", .shape = &.{d} },
        .{ .name = "head.layers.0.norm2.bias", .shape = &.{d} },
        .{ .name = "head.layers.0.linear1.weight", .shape = &.{ 64, d } },
        .{ .name = "head.layers.0.linear1.bias", .shape = &.{64} },
        .{ .name = "head.layers.0.linear2.weight", .shape = &.{ d, 64 } },
        .{ .name = "head.layers.0.linear2.bias", .shape = &.{d} },
        .{ .name = "scorer.0.weight", .shape = &.{d} },
        .{ .name = "scorer.0.bias", .shape = &.{d} },
        .{ .name = "scorer.1.weight", .shape = &.{ d, d } },
        .{ .name = "scorer.1.bias", .shape = &.{d} },
        .{ .name = "scorer.3.weight", .shape = &.{ 1, d } },
        .{ .name = "scorer.3.bias", .shape = &.{1} },
        .{ .name = "act_head.0.weight", .shape = &.{ 4, d + 4 } },
        .{ .name = "temperature", .shape = &.{3} },
    };
};

/// Writes the tiny checkpoint into `dir`, a fixture for the tests and the
/// check tool: F32 values in [−0.1, 0.1), norms all ones, except `scorer.3`
/// (weight zero, bias 0.25) so every logit is 0.25 whatever the forward
/// computes. `skip` drops a tensor; `extra` adds one.
pub fn writeTiny(gpa: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, skip: ?[]const u8, extra: ?TinyTensor) !void {
    try dir.createDirPath(io, "encoder");
    try dir.writeFile(io, .{ .sub_path = "encoder/config.json", .data = tiny_config });
    var header: std.Io.Writer.Allocating = .init(gpa);
    defer header.deinit();
    var data: std.ArrayList(u8) = .empty;
    defer data.deinit(gpa);
    var prng = std.Random.DefaultPrng.init(7);
    try header.writer.writeByte('{');
    var first = true;
    for (tiny_tensors ++ [_]TinyTensor{extra orelse .{ .name = "", .shape = &.{} }}, 0..) |t, i| {
        if (t.name.len == 0 or (i < tiny_tensors.len and skip != null and std.mem.eql(u8, t.name, skip.?))) continue;
        var elements: u64 = 1;
        for (t.shape) |x| elements *= x;
        const start = data.items.len;
        for (0..elements) |_| {
            const value: f32 = if (std.mem.endsWith(u8, t.name, "norm.weight") or std.mem.endsWith(u8, t.name, "norm1.weight") or std.mem.endsWith(u8, t.name, "norm2.weight"))
                1
            else if (std.mem.eql(u8, t.name, "scorer.3.weight"))
                0
            else if (std.mem.eql(u8, t.name, "scorer.3.bias"))
                0.25
            else
                prng.random().float(f32) * 0.2 - 0.1;
            try data.appendSlice(gpa, std.mem.asBytes(&value));
        }
        if (!first) try header.writer.writeByte(',');
        first = false;
        try header.writer.print("\"{s}\":{{\"dtype\":\"F32\",\"shape\":[", .{t.name});
        for (t.shape, 0..) |x, k| try header.writer.print("{s}{d}", .{ if (k == 0) "" else ",", x });
        try header.writer.print("],\"data_offsets\":[{d},{d}]}}", .{ start, data.items.len });
    }
    try header.writer.writeByte('}');
    const file = try dir.createFile(io, "model.safetensors", .{});
    defer file.close(io);
    var length: [8]u8 = undefined;
    std.mem.writeInt(u64, &length, header.written().len, .little);
    try file.writeStreamingAll(io, &length);
    try file.writeStreamingAll(io, header.written());
    try file.writeStreamingAll(io, data.items);
}

fn openTiny(gpa: std.mem.Allocator, io: std.Io, skip: ?[]const u8, extra: ?TinyTensor) !Laya {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try writeTiny(std.testing.allocator, io, tmp.dir, skip, extra);
    const path = try tmp.dir.realPathFileAlloc(io, ".", std.testing.allocator);
    defer std.testing.allocator.free(path);
    return Laya.open(gpa, io, path, .cpu);
}

test "a tiny checkpoint opens, traces every stage, and scores each marker" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var model = try openTiny(gpa, io, null, null);
    defer model.deinit(io);
    try std.testing.expectEqual(@as(usize, 1), model.head_layers);
    const Stages = struct {
        seen: [4]Stage = undefined,
        count: usize = 0,
        fn record(context: *anyopaque, stage: Stage, hidden: []const f32) void {
            const self: *@This() = @ptrCast(@alignCast(context));
            for (hidden) |x| std.debug.assert(std.math.isFinite(x));
            self.seen[self.count] = stage;
            self.count += 1;
        }
    };
    var stages: Stages = .{};
    var out: [2]f32 = undefined;
    try model.logits(io, gpa, &.{ 1, 5, 19, 2, 7 }, &.{ 1, 3 }, .noul, &out, .{ .context = &stages, .record = Stages.record });
    try std.testing.expectEqualSlices(f32, &.{ 0.25, 0.25 }, &out);
    try std.testing.expectEqual(@as(usize, 4), stages.count);
    try std.testing.expectEqualDeep([4]Stage{ .{ .encoder = 0 }, .{ .encoder = 1 }, .final, .{ .head = 0 } }, stages.seen);
    try std.testing.expectError(error.InvalidTokenId, model.logits(io, gpa, &.{20}, &.{0}, .choice, out[0..1], null));
    try std.testing.expectError(error.InvalidMarker, model.logits(io, gpa, &.{ 1, 2 }, &.{2}, .choice, out[0..1], null));
}

test "unknown, missing, and misshapen tensors are rejected" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    try std.testing.expectError(error.UnexpectedTensor, openTiny(gpa, io, null, .{ .name = "pooler.weight", .shape = &.{1} }));
    try std.testing.expectError(error.MissingTensor, openTiny(gpa, io, "encoder.layers.1.mlp.Wo.weight", null));
    try std.testing.expectError(error.MissingTensor, openTiny(gpa, io, "scorer.1.bias", null));
    // Layer 0 has no attention norm; one there is not a tensor Laya knows.
    try std.testing.expectError(error.UnexpectedTensor, openTiny(gpa, io, null, .{ .name = "encoder.layers.0.attn_norm.weight", .shape = &.{64} }));
    try std.testing.expectError(error.TensorShapeMismatch, openTiny(gpa, io, "type_emb.weight", .{ .name = "type_emb.weight", .shape = &.{ 2, 64 } }));
}

test "open releases everything on allocation failure" {
    try alloc_check.checkAll(std.testing.allocator, struct {
        fn run(gpa: std.mem.Allocator) !void {
            var model = try openTiny(gpa, std.testing.io, null, null);
            model.deinit(std.testing.io);
        }
    }.run, .{});
}
