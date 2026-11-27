//! The Inkling 2 projector on Metal: the CPU reference's schedule
//! (`inkling2v.Runtime.encode`) over the batched matmul tiles and the
//! LayerNorm, bias, layer-scale, RoPE, and erf-GELU kernels, with the
//! materialized-score attention (`attention_metal`) over every row. The
//! host writes the patch rows, the class and position rows, and the rotary
//! table; the 2×2 merge runs on the host between two command buffers.
//! Weights are the mapped file's bytes wrapped in place, except the patch
//! kernel, widened to F32 and zero-padded from 1,200 to the tile's 1,216
//! columns. Activations are sized for the largest image (16,385 rows).
const std = @import("std");
const metal = @import("../backends/metal/root.zig");
const weights = @import("../runtime/weights.zig");
const preprocess = @import("preprocess.zig");
const model = @import("inkling2v.zig");
const attention = @import("attention_metal.zig");
const plan = @import("plan_metal.zig");

const Backend = metal.Backend;
const Buffer = metal.Buffer;

const hidden = model.hidden;
const ffn = model.ffn;
const merged = model.merged_width;
const out_width = model.output_width;
/// The patch row padded to the matmul tile's column multiple.
pub const patch_columns: usize = @divCeil(model.patch_values, 64) * 64;
/// Activation rows: the largest image's, padded for the tiles and the
/// attention's query tiles.
const capacity_rows: usize = @max(Backend.matmulPadded(model.max_rows), Backend.attentionChunkRows(model.max_rows));

const Weight = plan.Weight;
const Layer = struct {
    n1w: Buffer,
    n1b: Buffer,
    query: Weight,
    qb: Buffer,
    key: Weight,
    kb: Buffer,
    value: Weight,
    vb: Buffer,
    output: Weight,
    ob: Buffer,
    s1: Buffer,
    n2w: Buffer,
    n2b: Buffer,
    up: Weight,
    ub: Buffer,
    down: Weight,
    db: Buffer,
    s2: Buffer,
};

pub const Plan = struct {
    alloc: std.mem.Allocator,
    backend: *Backend,
    view: weights.View,
    binding: *const model.Binding,
    kernel: Weight,
    pre_w: Buffer,
    pre_b: Buffer,
    layers: [model.blocks]Layer,
    adapter: [2]Weight,
    // Activations for `capacity_rows` rows.
    input: Buffer, // rows × patch_columns
    x: Buffer,
    h: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    attn: Buffer,
    /// The FFN rows; after the blocks, the merged rows and the adapter's
    /// hidden layer.
    up: Buffer,
    rope: Buffer,
    out: Buffer, // token rows × output_width
    spans: attention.Spans,
    scores: Buffer, // attention.score_capacity floats
    scratch: []f32,
    /// Bytes of device memory the plan created (the widened kernel and the
    /// activations); the wrapped file bytes are not counted.
    created_bytes: usize,

    pub fn init(alloc: std.mem.Allocator, backend: *Backend, view: weights.View, binding: *const model.Binding) !Plan {
        var self: Plan = undefined;
        self.alloc = alloc;
        self.backend = backend;
        self.view = view;
        self.binding = binding;
        self.created_bytes = 0;
        const kernel = try model.patchKernel(alloc, view, binding, patch_columns);
        defer alloc.free(kernel);
        const kernel_buffer = try plan.created(&self, kernel.len * 4);
        @memcpy(kernel_buffer.floats()[0..kernel.len], kernel);
        self.kernel = .{ .buffer = kernel_buffer, .matrix = .{ .encoding = 0, .rows = hidden, .columns = patch_columns, .bytes = kernel_buffer.host[0 .. kernel.len * 4] } };
        self.pre_w = try plan.wrapped(&self, binding.pre_norm.weight);
        self.pre_b = try plan.wrapped(&self, binding.pre_norm.bias);
        for (&self.layers, binding.layers) |*layer, l| layer.* = .{
            .n1w = try plan.wrapped(&self, l.norm1.weight),
            .n1b = try plan.wrapped(&self, l.norm1.bias),
            .query = try plan.weight(&self, l.query.weight),
            .qb = try plan.wrapped(&self, l.query.bias),
            .key = try plan.weight(&self, l.key.weight),
            .kb = try plan.wrapped(&self, l.key.bias),
            .value = try plan.weight(&self, l.value.weight),
            .vb = try plan.wrapped(&self, l.value.bias),
            .output = try plan.weight(&self, l.output.weight),
            .ob = try plan.wrapped(&self, l.output.bias),
            .s1 = try plan.wrapped(&self, l.scale1),
            .n2w = try plan.wrapped(&self, l.norm2.weight),
            .n2b = try plan.wrapped(&self, l.norm2.bias),
            .up = try plan.weight(&self, l.up.weight),
            .ub = try plan.wrapped(&self, l.up.bias),
            .down = try plan.weight(&self, l.down.weight),
            .db = try plan.wrapped(&self, l.down.bias),
            .s2 = try plan.wrapped(&self, l.scale2),
        };
        for (&self.adapter, binding.adapter) |*a, t| a.* = try plan.weight(&self, t);
        const tokens = Backend.matmulPadded(model.max_tokens);
        self.input = try plan.created(&self, capacity_rows * patch_columns * 4);
        inline for (.{ "x", "h", "q", "k", "v", "attn" }) |field| @field(self, field) = try plan.created(&self, capacity_rows * hidden * 4);
        self.up = try plan.created(&self, @max(capacity_rows * ffn, tokens * 2 * merged) * 4);
        self.rope = try plan.created(&self, model.max_rows * model.rope_pairs * 8);
        self.out = try plan.created(&self, tokens * out_width * 4);
        self.scores = try plan.created(&self, attention.score_capacity * 4);
        self.spans = try attention.Spans.init(alloc, backend, model.max_rows, &self.created_bytes);
        errdefer self.spans.deinit(alloc);
        self.scratch = try alloc.alloc(f32, 4 * hidden);
        return self;
    }
    pub fn deinit(self: *Plan) void {
        // Device buffers belong to the backend and go with it.
        self.alloc.free(self.scratch);
        self.spans.deinit(self.alloc);
        self.* = undefined;
    }

    /// Encodes `patches` (raster rows from `model.patchRows`) of an image of
    /// `grid` into `out` (`grid.tokens() × output_width`); commits and
    /// waits before the copy out.
    pub fn encode(self: *Plan, patches: preprocess.Patches, grid: model.Grid, out: []f32) !void {
        if (out.len < grid.tokens() * out_width) return error.InvalidShape;
        return self.run(patches, grid, model.blocks, .{ .features = out });
    }
    /// The residual rows (`grid.rows() × hidden`, the class row first)
    /// after the first `count` blocks, for the checks' traces.
    pub fn encodeResidual(self: *Plan, patches: preprocess.Patches, grid: model.Grid, count: usize, rows: []f32) !void {
        if (count > model.blocks or rows.len < grid.rows() * hidden) return error.InvalidShape;
        return self.run(patches, grid, count, .{ .residual = rows });
    }

    fn run(self: *Plan, patches: preprocess.Patches, grid: model.Grid, block_count: usize, result: model.Result) !void {
        const p = grid.patches();
        const tokens = grid.tokens();
        if (p == 0 or p > model.max_patches or patches.row != model.patch_values or patches.count() != p or patches.width_patches != grid.width_patches) return error.InvalidShape;
        const n = p + 1;
        const rows = @max(Backend.matmulPadded(n), Backend.attentionChunkRows(n));

        // Host inputs: row 0 (the class token) and the padded columns and
        // rows zero, so the tiles read finite values; the patch rows after
        // it; the class and position rows in `attn`; the rotary table.
        const input = self.input.floats()[0 .. rows * patch_columns];
        @memset(input, 0);
        for (0..p) |r| @memcpy(input[(r + 1) * patch_columns ..][0..model.patch_values], patches.values[r * model.patch_values ..][0..model.patch_values]);
        try model.positionRows(self.view, self.binding, grid, self.attn.floats(), self.scratch);
        try model.ropeTable(self.rope.floats(), grid);
        for ([_]Buffer{ self.x, self.h, self.attn, self.q, self.k, self.v }) |buffer| @memset(buffer.floats()[n * hidden .. rows * hidden], 0);
        @memset(self.up.floats()[n * ffn .. rows * ffn], 0);
        try self.spans.plan(&[_]attention.Window{.{ .begin = 0, .count = @intCast(n) }}, n, model.heads);

        const b = self.backend;
        const norm: Backend.Norm = .{ .rows = n, .width = hidden, .in_stride = hidden, .out_stride = hidden, .eps = model.norm_epsilon };
        const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, model.head_dim)));
        try b.begin();
        errdefer b.discard();
        try b.matmul(self.kernel.buffer, self.kernel.matrix, self.input, patch_columns, self.h, hidden, n);
        try b.add(self.h, self.attn, n * hidden);
        try b.layerNorm(self.h, self.pre_w, self.pre_b, self.x, norm);
        for (self.layers[0..block_count]) |l| {
            try b.layerNorm(self.x, l.n1w, l.n1b, self.h, norm);
            inline for (.{ .{ "query", "qb", "q" }, .{ "key", "kb", "k" }, .{ "value", "vb", "v" } }) |names| {
                const w = @field(l, names[0]);
                const dst = @field(self, names[2]);
                try b.matmul(w.buffer, w.matrix, self.h, hidden, dst, hidden, n);
                try b.addBiasRows(dst, @field(l, names[1]), hidden, n, hidden);
            }
            try b.ropeRows(self.q, self.rope, model.heads, model.head_dim, model.head_dim, 0, n, hidden, .adjacent);
            try b.ropeRows(self.k, self.rope, model.heads, model.head_dim, model.head_dim, 0, n, hidden, .adjacent);
            try self.spans.attend(b, self.q, self.k, self.v, self.attn, self.scores, .{ .width = model.head_dim, .q_stride = hidden, .kv_stride = hidden, .out_stride = hidden, .scale = scale });
            try b.matmul(l.output.buffer, l.output.matrix, self.attn, hidden, self.h, hidden, n);
            try b.addBiasRows(self.h, l.ob, hidden, n, hidden);
            try b.mulRows(self.h, l.s1, hidden, n, hidden);
            try b.add(self.x, self.h, n * hidden);
            try b.layerNorm(self.x, l.n2w, l.n2b, self.h, norm);
            try b.matmul(l.up.buffer, l.up.matrix, self.h, hidden, self.up, ffn, n);
            try b.addBiasRows(self.up, l.ub, ffn, n, ffn);
            try b.geluErf(self.up, n * ffn);
            try b.matmul(l.down.buffer, l.down.matrix, self.up, ffn, self.h, hidden, n);
            try b.addBiasRows(self.h, l.db, hidden, n, hidden);
            try b.mulRows(self.h, l.s2, hidden, n, hidden);
            try b.add(self.x, self.h, n * hidden);
        }
        try b.commit();
        const out = switch (result) {
            .residual => |dest| {
                @memcpy(dest[0 .. n * hidden], self.x.floats()[0 .. n * hidden]);
                return;
            },
            .features => |f| f,
        };

        // The merge of the patch rows (the class row dropped), then the
        // adapter over them.
        const padded_tokens = Backend.matmulPadded(tokens);
        const merged_rows = self.up.slice(0, padded_tokens * merged * 4);
        const mid = self.up.slice(padded_tokens * merged * 4, padded_tokens * merged * 4);
        const host = merged_rows.floats();
        try model.merge2x2(self.x.floats()[hidden .. n * hidden], grid, host);
        @memset(host[tokens * merged .. padded_tokens * merged], 0);
        try b.begin();
        try b.matmul(self.adapter[0].buffer, self.adapter[0].matrix, merged_rows, merged, mid, merged, tokens);
        try b.geluErf(mid, tokens * merged);
        try b.matmul(self.adapter[1].buffer, self.adapter[1].matrix, mid, merged, self.out, out_width, tokens);
        try b.commit();
        @memcpy(out[0 .. tokens * out_width], self.out.floats()[0 .. tokens * out_width]);
    }
};
