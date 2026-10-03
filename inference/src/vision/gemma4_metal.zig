//! Gemma 4's projectors on Metal: the CPU reference's schedules
//! (`gemma4.Runtime`) over the batched matmul tiles and the norm, RoPE, and
//! attention kernels. The SigLIP encoder's attention is the chunk attention
//! kernel with the whole patch set as one bidirectional span; its 3×3 pool
//! and standardization run on the host between two command buffers. Weights
//! are the mapped file's bytes wrapped in place, except the FFN down
//! matrices, whose columns (4,304 on the 26B-A4B) are zero-padded to the
//! tile's 64. A clipped linear's bounds (the E4B) are `clamp` dispatches
//! around its matmul. Activations are sized for the largest image (1,120
//! tokens).
const std = @import("std");
const metal = @import("../backends/metal/root.zig");
const cpu = @import("../backends/cpu/root.zig");
const weights = @import("../runtime/weights.zig");
const preprocess = @import("preprocess.zig");
const model = @import("gemma4.zig");

const Backend = metal.Backend;
const Buffer = metal.Buffer;
const Tensor = @import("../formats/gguf.zig").Tensor;

const siglip = model.siglip;
const unified = model.unified;
/// A SigLIP FFN width rounded up to the matmul tile's column multiple.
pub fn ffnPadded(g: *const siglip.Geometry) usize {
    return @divCeil(g.ffn, 64) * 64;
}

const Weight = struct { buffer: Buffer, matrix: cpu.Matrix };

const Unified = struct {
    n1w: Buffer,
    n1b: Buffer,
    kernel: Weight,
    bias: Buffer,
    n2w: Buffer,
    n2b: Buffer,
    n3w: Buffer,
    n3b: Buffer,
    input: Buffer, // rows × values
    normed: Buffer, // rows × values
    x: Buffer, // rows × hidden
    h: Buffer,
    pos: Buffer,
};

const Layer = struct {
    ln1: Buffer,
    query: Weight,
    key: Weight,
    value: Weight,
    output: Weight,
    query_norm: Buffer,
    key_norm: Buffer,
    post_attention_norm: Buffer,
    ln2: Buffer,
    gate: Weight,
    up: Weight,
    /// The padded copy, `hidden × ffnPadded` in the file's encoding.
    down: Weight,
    post_ffn_norm: Buffer,
    bounds: [7]model.Bounds,
};

const Siglip = struct {
    kernel: Weight,
    layers: [siglip.max_blocks]Layer,
    /// Empty when the file has no standardization.
    std_bias: []f32,
    std_scale: []f32,
    input: Buffer, // patch rows × values
    x: Buffer, // patch rows × hidden
    h: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    attn: Buffer,
    gate: Buffer, // patch rows × ffnPadded
    up: Buffer,
    rope_x: Buffer,
    rope_y: Buffer,
    pooled: Buffer, // token rows × hidden
};

pub const Plan = struct {
    alloc: std.mem.Allocator,
    backend: *Backend,
    view: weights.View,
    binding: *const model.Binding,
    net: union(model.Kind) { unified: Unified, siglip: Siglip },
    projection: Weight,
    /// A row of ones: the weightless RMS norms.
    ones: Buffer,
    out: Buffer, // token rows × output_width
    /// Host rows: position tables, then the pooled SigLIP rows.
    scratch: []f32,
    /// Bytes of device memory the plan created (copies and activations);
    /// wrapped file bytes are not counted.
    created_bytes: usize,

    pub fn init(alloc: std.mem.Allocator, backend: *Backend, view: weights.View, binding: *const model.Binding) !Plan {
        var self: Plan = undefined;
        self.alloc = alloc;
        self.backend = backend;
        self.view = view;
        self.binding = binding;
        self.created_bytes = 0;
        const tokens = Backend.matmulPadded(model.max_tokens);
        self.ones = try self.created(@max(unified.hidden, siglip.max_hidden) * 4);
        @memset(self.ones.floats(), 1);
        self.out = try self.created(tokens * binding.output_width * 4);
        switch (binding.net) {
            .unified => |*net| {
                self.projection = try self.weight(net.projection);
                self.net = .{ .unified = .{
                    .n1w = try self.wrapped(net.patch_norm_1.weight),
                    .n1b = try self.wrapped(net.patch_norm_1.bias),
                    .kernel = try self.weight(net.patch_embedding),
                    .bias = try self.wrapped(net.patch_bias),
                    .n2w = try self.wrapped(net.patch_norm_2.weight),
                    .n2b = try self.wrapped(net.patch_norm_2.bias),
                    .n3w = try self.wrapped(net.patch_norm_3.weight),
                    .n3b = try self.wrapped(net.patch_norm_3.bias),
                    .input = try self.created(tokens * unified.values * 4),
                    .normed = try self.created(tokens * unified.values * 4),
                    .x = try self.created(tokens * unified.hidden * 4),
                    .h = try self.created(tokens * unified.hidden * 4),
                    .pos = try self.created(tokens * unified.hidden * 4),
                } };
                self.scratch = try alloc.alloc(f32, model.max_tokens * unified.hidden + unified.hidden);
            },
            .siglip => |*net| {
                self.projection = try self.weight(net.projection);
                const rows = Backend.matmulPadded(siglip.max_patches);
                const g = net.geometry;
                const h = g.hidden;
                var s: Siglip = undefined;
                const kernel = try model.siglipKernel(view, net);
                s.kernel = .{ .buffer = try backend.wrap(kernel.bytes), .matrix = kernel };
                for (s.layers[0..g.blocks], net.active()) |*layer, l| {
                    var bounds: [7]model.Bounds = undefined;
                    for (&bounds, l.clamps) |*b, c| b.* = try model.clampBounds(view, c);
                    layer.* = .{
                        .ln1 = try self.wrapped(l.ln1),
                        .query = try self.weight(l.query),
                        .key = try self.weight(l.key),
                        .value = try self.weight(l.value),
                        .output = try self.weight(l.output),
                        .query_norm = try self.wrapped(l.query_norm),
                        .key_norm = try self.wrapped(l.key_norm),
                        .post_attention_norm = try self.wrapped(l.post_attention_norm),
                        .ln2 = try self.wrapped(l.ln2),
                        .gate = try self.weight(l.gate),
                        .up = try self.weight(l.up),
                        .down = try self.paddedDown(l.down, g),
                        .post_ffn_norm = try self.wrapped(l.post_ffn_norm),
                        .bounds = bounds,
                    };
                }
                s.std_bias = if (net.std_bias) |t| try view.vector(alloc, t) else &.{};
                errdefer alloc.free(s.std_bias);
                s.std_scale = if (net.std_scale) |t| try view.vector(alloc, t) else &.{};
                errdefer alloc.free(s.std_scale);
                s.input = try self.created(rows * siglip.values * 4);
                // The attention reads queries and writes its output on the
                // kernel's own row padding (a multiple of 8, within `rows`).
                inline for (.{ "x", "h", "q", "k", "v", "attn" }) |field| @field(s, field) = try self.created(rows * h * 4);
                s.gate = try self.created(rows * ffnPadded(g) * 4);
                s.up = try self.created(rows * ffnPadded(g) * 4);
                s.rope_x = try self.created(siglip.max_patches * g.ropePairs() * 8);
                s.rope_y = try self.created(siglip.max_patches * g.ropePairs() * 8);
                s.pooled = try self.created(tokens * h * 4);
                self.net = .{ .siglip = s };
                self.scratch = try alloc.alloc(f32, siglip.max_patches * h + h);
            },
        }
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
    fn paddedDown(self: *Plan, tensor: *const Tensor, g: *const siglip.Geometry) !Weight {
        const down = try self.view.matrix(tensor);
        const element: usize = if (down.encoding == 0) 4 else 2;
        const padded = ffnPadded(g);
        if (down.bytes.len != g.hidden * g.ffn * element) return error.InvalidShape;
        const buffer = try self.created(g.hidden * padded * element);
        const host = buffer.host[0 .. g.hidden * padded * element];
        for (0..g.hidden) |r| @memcpy(host[r * padded * element ..][0 .. g.ffn * element], down.bytes[r * g.ffn * element ..][0 .. g.ffn * element]);
        return .{ .buffer = buffer, .matrix = .{ .encoding = down.encoding, .rows = g.hidden, .columns = padded, .bytes = host } };
    }
    /// `output = clamp(W · clamp(input))` over `n` rows; an input clamp
    /// rewrites `input` in place, which every caller's next use tolerates
    /// (the reference clamps the same tensor it then multiplies).
    fn clipped(self: *Plan, w: Weight, bounds: model.Bounds, input: Buffer, in_stride: usize, output: Buffer, out_stride: usize, n: usize) !void {
        const b = self.backend;
        if (!bounds.isOpen()) try b.clamp(input, n * in_stride, bounds.input[0], bounds.input[1]);
        try b.matmul(w.buffer, w.matrix, input, in_stride, output, out_stride, n);
        if (!bounds.isOpen()) try b.clamp(output, n * out_stride, bounds.output[0], bounds.output[1]);
    }
    pub fn deinit(self: *Plan) void {
        // Device buffers belong to the backend and go with it.
        self.alloc.free(self.scratch);
        switch (self.net) {
            .unified => {},
            .siglip => |s| {
                if (s.std_bias.len != 0) self.alloc.free(s.std_bias);
                if (s.std_scale.len != 0) self.alloc.free(s.std_scale);
            },
        }
        self.* = undefined;
    }

    /// Encodes `patches` of an image of `grid` into `out`
    /// (`grid.tokens() × output_width`); commits and waits before the copy out.
    pub fn encode(self: *Plan, patches: preprocess.Patches, grid: model.Grid, out: []f32) !void {
        const tokens = grid.tokens();
        const width = self.binding.output_width;
        if (tokens == 0 or tokens > model.max_tokens or out.len < tokens * width) return error.InvalidShape;
        switch (self.net) {
            .unified => |*u| try self.encodeUnified(u, patches, grid),
            .siglip => |*s| try self.encodeSiglip(s, patches, grid),
        }
        @memcpy(out[0 .. tokens * width], self.out.floats()[0 .. tokens * width]);
    }

    fn encodeUnified(self: *Plan, u: *Unified, patches: preprocess.Patches, grid: model.Grid) !void {
        const n = grid.tokens();
        const h = unified.hidden;
        const net = &self.binding.net.unified;
        if (patches.row != unified.values or patches.count() != n or patches.width_patches != grid.width_tokens) return error.InvalidShape;
        @memcpy(u.input.floats()[0 .. n * unified.values], patches.values);
        try model.positionRows(self.view, net.position, h, unified.table, grid.width_tokens, n, u.pos.floats(), self.scratch[n * h ..]);
        const b = self.backend;
        const eps = unified.layer_norm_epsilon;
        try b.begin();
        errdefer b.discard();
        try b.layerNorm(u.input, u.n1w, u.n1b, u.normed, .{ .rows = n, .width = unified.values, .in_stride = unified.values, .out_stride = unified.values, .eps = eps });
        try b.matmul(u.kernel.buffer, u.kernel.matrix, u.normed, unified.values, u.x, h, n);
        try b.addBiasRows(u.x, u.bias, h, n, h);
        const norm: Backend.Norm = .{ .rows = n, .width = h, .in_stride = h, .out_stride = h, .eps = eps };
        try b.layerNorm(u.x, u.n2w, u.n2b, u.h, norm);
        try b.add(u.h, u.pos, n * h);
        try b.layerNorm(u.h, u.n3w, u.n3b, u.x, norm);
        try b.rmsNorm(u.x, self.ones, u.h, .{ .rows = n, .width = h, .in_stride = h, .out_stride = h, .eps = model.rms_epsilon });
        try b.matmul(self.projection.buffer, self.projection.matrix, u.h, h, self.out, self.binding.output_width, n);
        try b.commit();
    }

    /// Q, K, and V from the normed rows in `s.h`. Each projection clamps
    /// its own input, so a clamped copy goes through `s.attn` (free until
    /// the attention writes it) and `s.h` stays unclamped for the next.
    fn clippedShared(self: *Plan, l: Layer, s: *Siglip, n: usize) !void {
        const b = self.backend;
        const h = self.binding.net.siglip.geometry.hidden;
        const targets = [_]struct { w: Weight, which: model.Linear, out: Buffer }{
            .{ .w = l.query, .which = .query, .out = s.q },
            .{ .w = l.key, .which = .key, .out = s.k },
            .{ .w = l.value, .which = .value, .out = s.v },
        };
        for (targets) |t| {
            const bounds = l.bounds[@backingInt(t.which)];
            var input = s.h;
            if (!bounds.isOpen()) {
                try b.copy(s.attn, s.h, n * h);
                input = s.attn;
            }
            try self.clipped(t.w, bounds, input, h, t.out, h, n);
        }
    }
    /// The gate and up projections of the normed rows in `s.h`, each with
    /// its own input clamp (a copy through `s.attn`, as `clippedShared`).
    fn clippedPair(self: *Plan, l: Layer, s: *Siglip, n: usize, ffn_padded: usize) !void {
        const b = self.backend;
        const h = self.binding.net.siglip.geometry.hidden;
        for ([_]struct { w: Weight, which: model.Linear, out: Buffer }{
            .{ .w = l.gate, .which = .gate, .out = s.gate },
            .{ .w = l.up, .which = .up, .out = s.up },
        }) |t| {
            const bounds = l.bounds[@backingInt(t.which)];
            var input = s.h;
            if (!bounds.isOpen()) {
                try b.copy(s.attn, s.h, n * h);
                input = s.attn;
            }
            try self.clipped(t.w, bounds, input, h, t.out, ffn_padded, n);
        }
    }

    fn encodeSiglip(self: *Plan, s: *Siglip, patches: preprocess.Patches, grid: model.Grid) !void {
        const wp: usize = grid.widthPatches();
        const n = wp * grid.heightPatches();
        const tokens = grid.tokens();
        const net = &self.binding.net.siglip;
        const g = net.geometry;
        const h = g.hidden;
        const d = g.headDim();
        const heads = g.heads;
        const ffn_padded = ffnPadded(g);
        if (patches.row != siglip.values or patches.count() != n or patches.width_patches != wp) return error.InvalidShape;
        @memcpy(s.input.floats()[0 .. n * siglip.values], patches.values);
        try model.positionRows(self.view, net.position, h, siglip.table, wp, n, self.scratch, self.scratch[n * h ..]);
        @memcpy(s.h.floats()[0 .. n * h], self.scratch[0 .. n * h]);
        try model.ropeTables(s.rope_x.floats(), s.rope_y.floats(), g.ropePairs(), wp, n);
        const b = self.backend;
        try b.begin();
        errdefer b.discard();
        try b.matmul(s.kernel.buffer, s.kernel.matrix, s.input, siglip.values, s.x, h, n);
        try b.add(s.x, s.h, n * h);
        const rows: Backend.Norm = .{ .rows = n, .width = h, .in_stride = h, .out_stride = h };
        const per_head: Backend.Norm = .{ .rows = n * heads, .width = d, .in_stride = d, .out_stride = d };
        const half = d / 2;
        for (s.layers[0..g.blocks]) |l| {
            try b.rmsNorm(s.x, l.ln1, s.h, rows);
            // The three input clamps apply to the same normed rows; each
            // clamps a copy only when its bounds differ from the one before.
            try self.clippedShared(l, s, n);
            try b.rmsNorm(s.q, l.query_norm, s.q, per_head);
            try b.rmsNorm(s.k, l.key_norm, s.k, per_head);
            // The first half of each head turns with the patch's x, the second with its y.
            for ([_]Buffer{ s.q, s.k }) |data| {
                try b.ropeRows(data, s.rope_x, heads, d, half, 0, n, h, .split_half);
                try b.ropeRows(data.slice(half * 4, data.len - half * 4), s.rope_y, heads, d, half, 0, n, h, .split_half);
            }
            try b.rmsNorm(s.v, self.ones, s.v, per_head);
            try b.attentionChunk(s.k, s.v, s.q, s.attn, .{ .query_heads = heads, .kv_heads = heads, .key_width = d, .value_width = d, .position = 0, .count = n, .q_stride = h, .out_stride = h, .scale = 1.0, .span = .{ .begin = 0, .end = n } });
            try self.clipped(l.output, l.bounds[@backingInt(model.Linear.output)], s.attn, h, s.h, h, n);
            try b.rmsNorm(s.h, l.post_attention_norm, s.h, rows);
            try b.add(s.x, s.h, n * h);
            try b.rmsNorm(s.x, l.ln2, s.h, rows);
            try self.clippedPair(l, s, n, ffn_padded);
            // The padding columns were never written: gelu_quick(0)·0 keeps them zero.
            try b.geluQuickMul(s.gate, s.up, n * ffn_padded);
            try self.clipped(l.down, l.bounds[@backingInt(model.Linear.down)], s.gate, ffn_padded, s.h, h, n);
            try b.rmsNorm(s.h, l.post_ffn_norm, s.h, rows);
            try b.add(s.x, s.h, n * h);
        }
        try b.commit();
        const pooled = self.scratch[0 .. tokens * h];
        try model.poolRows(s.x.floats()[0 .. n * h], grid, h, pooled);
        if (s.std_bias.len != 0) for (0..tokens) |t| model.standardize(pooled[t * h ..][0..h], s.std_bias, s.std_scale);
        @memcpy(s.pooled.floats()[0 .. tokens * h], pooled);
        try b.begin();
        try b.rmsNorm(s.pooled, self.ones, s.pooled, .{ .rows = tokens, .width = h, .in_stride = h, .out_stride = h, .eps = model.rms_epsilon });
        try b.matmul(self.projection.buffer, self.projection.matrix, s.pooled, h, self.out, self.binding.output_width, tokens);
        try b.commit();
    }
};
