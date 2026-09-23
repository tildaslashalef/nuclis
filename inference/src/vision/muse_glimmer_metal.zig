//! The Muse Glimmer projector on Metal: the CPU reference's schedule
//! (`muse_glimmer.Runtime.encode`) over the batched matmul tiles and the
//! LayerNorm, bias, RoPE, and erf-GELU kernels. Attention is the chunk
//! kernel with a bidirectional span: once per window on its row slice in a
//! windowed block (in window order, since each dispatch stores up to seven
//! padding rows into the next window's output, which that window then
//! overwrites), once over the whole patch set in a global block. The pixel
//! shuffle runs on the host between two command buffers.
//! Weights are the mapped file's bytes wrapped in place, except the patch
//! kernel, zero-padded from 588 to the tile's 640 columns. Activations are
//! sized for the largest image (16,384 patches).
const std = @import("std");
const metal = @import("../backends/metal/root.zig");
const cpu = @import("../backends/cpu/root.zig");
const weights = @import("../runtime/weights.zig");
const preprocess = @import("preprocess.zig");
const model = @import("muse_glimmer.zig");

const Backend = metal.Backend;
const Buffer = metal.Buffer;
const Tensor = @import("../formats/gguf.zig").Tensor;

const hidden = model.hidden;
const ffn = model.ffn;
const merged = model.merged_width;
const adapter = model.adapter_width;
const out_width = model.output_width;
/// The patch row padded to the matmul tile's column multiple.
pub const patch_columns = (model.patch_values + 63) / 64 * 64;

const Weight = struct { buffer: Buffer, matrix: cpu.Matrix };
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
    n2w: Buffer,
    n2b: Buffer,
    up: Weight,
    ub: Buffer,
    down: Weight,
    db: Buffer,
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
    post_w: Buffer,
    post_b: Buffer,
    adapter: [3]Weight,
    // Activations for `max_patches` rows (padded to the matmul tile).
    input: Buffer, // rows × patch_columns
    x: Buffer,
    h: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    attn: Buffer,
    /// The FFN rows; after the blocks, the merged rows and the adapter's two
    /// hidden layers.
    up: Buffer,
    rope: Buffer,
    out: Buffer, // token rows × output_width
    scratch: []f32,
    /// Bytes of device memory the plan created (the padded kernel and the
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
        const kernel_buffer = try self.created(kernel.len * 4);
        @memcpy(kernel_buffer.floats()[0..kernel.len], kernel);
        self.kernel = .{ .buffer = kernel_buffer, .matrix = .{ .encoding = 0, .rows = hidden, .columns = patch_columns, .bytes = kernel_buffer.host[0 .. kernel.len * 4] } };
        self.pre_w = try self.wrapped(binding.pre_norm.weight);
        self.pre_b = try self.wrapped(binding.pre_norm.bias);
        for (&self.layers, binding.layers) |*layer, l| layer.* = .{
            .n1w = try self.wrapped(l.norm1.weight),
            .n1b = try self.wrapped(l.norm1.bias),
            .query = try self.weight(l.query.weight),
            .qb = try self.wrapped(l.query.bias),
            .key = try self.weight(l.key.weight),
            .kb = try self.wrapped(l.key.bias),
            .value = try self.weight(l.value.weight),
            .vb = try self.wrapped(l.value.bias),
            .output = try self.weight(l.output.weight),
            .ob = try self.wrapped(l.output.bias),
            .n2w = try self.wrapped(l.norm2.weight),
            .n2b = try self.wrapped(l.norm2.bias),
            .up = try self.weight(l.up.weight),
            .ub = try self.wrapped(l.up.bias),
            .down = try self.weight(l.down.weight),
            .db = try self.wrapped(l.down.bias),
        };
        self.post_w = try self.wrapped(binding.post_norm.weight);
        self.post_b = try self.wrapped(binding.post_norm.bias);
        for (&self.adapter, binding.adapter) |*a, t| a.* = try self.weight(t);
        const rows = Backend.matmulPadded(model.max_patches);
        const tokens = Backend.matmulPadded(model.max_tokens);
        self.input = try self.created(rows * patch_columns * 4);
        inline for (.{ "x", "h", "q", "k", "v", "attn" }) |field| @field(self, field) = try self.created(rows * hidden * 4);
        self.up = try self.created(@max(rows * ffn, tokens * (merged + 2 * adapter)) * 4);
        self.rope = try self.created(model.max_patches * model.rope_pairs * 8);
        self.out = try self.created(tokens * out_width * 4);
        self.scratch = try alloc.alloc(f32, 4 * hidden);
        return self;
    }
    fn created(self: *Plan, len: usize) !Buffer {
        const buffer = try self.backend.create(len);
        self.created_bytes += len;
        return buffer;
    }
    fn wrapped(self: *Plan, tensor: *const Tensor) !Buffer {
        return self.backend.wrap(try self.view.bytes(tensor));
    }
    fn weight(self: *Plan, tensor: *const Tensor) !Weight {
        const matrix = try self.view.matrix(tensor);
        return .{ .buffer = try self.backend.wrap(matrix.bytes), .matrix = matrix };
    }
    pub fn deinit(self: *Plan) void {
        // Device buffers belong to the backend and go with it.
        self.alloc.free(self.scratch);
        self.* = undefined;
    }

    /// Encodes `patches` (raster rows from `model.patchOptions`) of an image
    /// of `grid` into `out` (`grid.tokens() × output_width`); commits and
    /// waits before the copy out.
    pub fn encode(self: *Plan, patches: preprocess.Patches, grid: model.Grid, out: []f32) !void {
        if (out.len < grid.tokens() * out_width) return error.InvalidShape;
        return self.run(patches, grid, model.blocks, .{ .features = out });
    }
    /// The residual rows (window order, `grid.patches() × hidden`) after the
    /// first `count` blocks, for the checks' traces.
    pub fn encodeResidual(self: *Plan, patches: preprocess.Patches, grid: model.Grid, count: usize, rows: []f32) !void {
        if (count > model.blocks or rows.len < grid.patches() * hidden) return error.InvalidShape;
        return self.run(patches, grid, count, .{ .residual = rows });
    }

    fn run(self: *Plan, patches: preprocess.Patches, grid: model.Grid, block_count: usize, result: model.Result) !void {
        const n = grid.patches();
        const tokens = grid.tokens();
        if (n == 0 or n > model.max_patches or patches.row != model.patch_values or patches.count() != n or patches.width_patches != grid.width_patches) return error.InvalidShape;
        var layout = try model.Layout.init(self.alloc, grid);
        defer layout.deinit(self.alloc);
        const rows = Backend.matmulPadded(n);

        // Host inputs: the patch rows in window order (padded columns and
        // rows zero, so the tiles read finite values), their positions in
        // `attn`, and the rotary table.
        const input = self.input.floats()[0 .. rows * patch_columns];
        @memset(input, 0);
        for (0..n) |r| @memcpy(input[r * patch_columns ..][0..model.patch_values], patches.values[@as(usize, layout.order[r]) * model.patch_values ..][0..model.patch_values]);
        try model.positionRows(self.view, self.binding.position_embedding, grid, layout, self.attn.floats(), self.scratch);
        try model.ropeTable(self.rope.floats(), grid, layout);
        for ([_]Buffer{ self.x, self.h, self.attn, self.q, self.k, self.v }) |buffer| @memset(buffer.floats()[n * hidden .. rows * hidden], 0);
        @memset(self.up.floats()[n * ffn .. rows * ffn], 0);

        const b = self.backend;
        const norm: Backend.Norm = .{ .rows = n, .width = hidden, .in_stride = hidden, .out_stride = hidden, .eps = model.norm_epsilon };
        const scale: f32 = @floatCast(1.0 / @sqrt(@as(f64, model.head_dim)));
        try b.begin();
        errdefer if (b.recording) b.commit() catch {};
        try b.matmul(self.kernel.buffer, self.kernel.matrix, self.input, patch_columns, self.h, hidden, n);
        try b.add(self.h, self.attn, n * hidden);
        try b.layerNorm(self.h, self.pre_w, self.pre_b, self.x, norm);
        for (self.layers[0..block_count], 0..) |l, il| {
            try b.layerNorm(self.x, l.n1w, l.n1b, self.h, norm);
            inline for (.{ .{ "query", "qb", "q" }, .{ "key", "kb", "k" }, .{ "value", "vb", "v" } }) |names| {
                const w = @field(l, names[0]);
                const dst = @field(self, names[2]);
                try b.matmul(w.buffer, w.matrix, self.h, hidden, dst, hidden, n);
                try b.addBiasRows(dst, @field(l, names[1]), hidden, n, hidden);
            }
            try b.ropeRows(self.q, self.rope, model.heads, model.head_dim, model.head_dim, 0, n, hidden, .adjacent);
            try b.ropeRows(self.k, self.rope, model.heads, model.head_dim, model.head_dim, 0, n, hidden, .adjacent);
            const whole = [_]model.Window{.{ .begin = 0, .count = @intCast(n) }};
            for (if (model.isGlobal(il)) &whole else layout.windows) |w| {
                const offset = @as(usize, w.begin) * hidden * 4;
                const keys = @as(usize, w.count) * hidden * 4;
                const rows_len = Backend.attentionChunkRows(w.count) * hidden * 4;
                try b.attentionChunk(self.k.slice(offset, keys), self.v.slice(offset, keys), self.q.slice(offset, rows_len), self.attn.slice(offset, rows_len), .{ .query_heads = model.heads, .kv_heads = model.heads, .key_width = model.head_dim, .value_width = model.head_dim, .position = 0, .count = w.count, .q_stride = hidden, .out_stride = hidden, .scale = scale, .span = .{ .begin = 0, .end = w.count } });
            }
            try b.matmul(l.output.buffer, l.output.matrix, self.attn, hidden, self.h, hidden, n);
            try b.addBiasRows(self.h, l.ob, hidden, n, hidden);
            try b.add(self.x, self.h, n * hidden);
            try b.layerNorm(self.x, l.n2w, l.n2b, self.h, norm);
            try b.matmul(l.up.buffer, l.up.matrix, self.h, hidden, self.up, ffn, n);
            try b.addBiasRows(self.up, l.ub, ffn, n, ffn);
            try b.geluErf(self.up, n * ffn);
            try b.matmul(l.down.buffer, l.down.matrix, self.up, ffn, self.h, hidden, n);
            try b.addBiasRows(self.h, l.db, hidden, n, hidden);
            try b.add(self.x, self.h, n * hidden);
        }
        const out = switch (result) {
            .residual => |dest| {
                try b.commit();
                @memcpy(dest[0 .. n * hidden], self.x.floats()[0 .. n * hidden]);
                return;
            },
            .features => |f| f,
        };
        try b.layerNorm(self.x, self.post_w, self.post_b, self.h, norm);
        try b.commit();

        // The shuffle back to raster order and into merged rows, then the
        // adapter over them.
        const padded_tokens = Backend.matmulPadded(tokens);
        const merged_rows = self.up.slice(0, padded_tokens * merged * 4);
        const mid = self.up.slice(padded_tokens * merged * 4, padded_tokens * adapter * 4);
        const mid2 = self.up.slice(padded_tokens * (merged + adapter) * 4, padded_tokens * adapter * 4);
        const host = merged_rows.floats();
        try model.shuffle(self.h.floats()[0 .. n * hidden], grid, layout, host);
        @memset(host[tokens * merged .. padded_tokens * merged], 0);
        try b.begin();
        try b.matmul(self.adapter[0].buffer, self.adapter[0].matrix, merged_rows, merged, mid, adapter, tokens);
        try b.geluErf(mid, tokens * adapter);
        try b.matmul(self.adapter[1].buffer, self.adapter[1].matrix, mid, adapter, mid2, adapter, tokens);
        try b.geluErf(mid2, tokens * adapter);
        try b.matmul(self.adapter[2].buffer, self.adapter[2].matrix, mid2, adapter, self.out, out_width, tokens);
        try b.commit();
        @memcpy(out[0 .. tokens * out_width], self.out.floats()[0 .. tokens * out_width]);
    }
};
