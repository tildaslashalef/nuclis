//! Gemma 4's audio encoder on Metal: `gemma4a.Runtime`'s schedule after the
//! subsampling convolutions, which run on the host (`gemma4a.subsample`,
//! about 0.2 GFLOP for 30 s). Matrices are the mapped file's bytes wrapped
//! in place and read by the generic F32 matmul tile; activations are F32.
//! A clipped linear's input bound is applied to a copy when another matrix
//! reads the same rows. Every buffer is sized for the longest clip
//! (`max_rows`). The plan borrows its `Backend`, which must outlive it.
const std = @import("std");
const metal = @import("../backends/metal/root.zig");
const cpu = @import("../backends/cpu/root.zig");
const weights = @import("../runtime/weights.zig");
const gguf = @import("../formats/gguf.zig");
const model = @import("gemma4a.zig");
const mel = @import("mel.zig");
const gemma4v = @import("../vision/gemma4.zig");

const Backend = metal.Backend;
const Buffer = metal.Buffer;
const hidden = model.hidden;
const ffn = model.ffn;

/// Rows of the longest clip the embedder accepts: 30 s.
pub const max_rows = 750;

const Weight = struct { buffer: Buffer, matrix: cpu.Matrix, bounds: model.Bounds = .open };
const Layer = struct {
    ffn_norm: Buffer,
    ffn_up: Weight,
    ffn_down: Weight,
    /// The post-norm weights × ½: the half step folded into the norm.
    ffn_post_half: Buffer,
    attn_pre_norm: Buffer,
    attn_q: Weight,
    attn_k: Weight,
    attn_v: Weight,
    attn_out: Weight,
    /// `queryScale` of the block, 128 values.
    query_scale: Buffer,
    attn_post_norm: Buffer,
    conv_pre_norm: Buffer,
    conv_pw1: Weight,
    conv_dw: Buffer,
    conv_post_norm: Buffer,
    conv_pw2: Weight,
    ffn_norm_1: Buffer,
    ffn_up_1: Weight,
    ffn_down_1: Weight,
    ffn_post_half_1: Buffer,
    out_norm: Buffer,
};

pub const Plan = struct {
    alloc: std.mem.Allocator,
    io: std.Io,
    backend: *Backend,
    view: weights.View,
    binding: *const model.Binding,
    layers: [model.blocks]Layer,
    input_projection: Weight,
    output: Weight,
    output_bias: Buffer,
    projection: Weight,
    /// `blocks × relative × hidden`, uploaded once.
    relk: Buffer,
    ones: Buffer,
    sub: Buffer,
    x: Buffer,
    n: Buffer,
    staged: Buffer,
    q: Buffer,
    k: Buffer,
    v: Buffer,
    attn: Buffer,
    wide: Buffer,
    projected: Buffer,
    out: Buffer,
    /// The host's subsampling scratch.
    scratch: []f32,
    /// Bytes of device memory the plan created; wrapped file bytes are not counted.
    created_bytes: usize,

    pub fn init(alloc: std.mem.Allocator, io: std.Io, backend: *Backend, view: weights.View, binding: *const model.Binding) !Plan {
        var self: Plan = undefined;
        self.alloc = alloc;
        self.io = io;
        self.backend = backend;
        self.view = view;
        self.binding = binding;
        self.created_bytes = 0;
        const rows = Backend.matmulPadded(max_rows);
        for (&self.layers, binding.layers) |*out, l| {
            out.* = .{
                .ffn_norm = try self.wrapped(l.ffn_norm),
                .ffn_up = try self.weight(l, .ffn_up),
                .ffn_down = try self.weight(l, .ffn_down),
                .ffn_post_half = try self.halved(l.ffn_post_norm),
                .attn_pre_norm = try self.wrapped(l.attn_pre_norm),
                .attn_q = try self.weight(l, .attn_q),
                .attn_k = try self.weight(l, .attn_k),
                .attn_v = try self.weight(l, .attn_v),
                .attn_out = try self.weight(l, .attn_out),
                .query_scale = try self.created(model.head_dim * 4),
                .attn_post_norm = try self.wrapped(l.attn_post_norm),
                .conv_pre_norm = try self.wrapped(l.conv_pre_norm),
                .conv_pw1 = try self.weight(l, .conv_pw1),
                .conv_dw = try self.wrapped(l.conv_dw),
                .conv_post_norm = try self.wrapped(l.conv_post_norm),
                .conv_pw2 = try self.weight(l, .conv_pw2),
                .ffn_norm_1 = try self.wrapped(l.ffn_norm_1),
                .ffn_up_1 = try self.weight(l, .ffn_up_1),
                .ffn_down_1 = try self.weight(l, .ffn_down_1),
                .ffn_post_half_1 = try self.halved(l.ffn_post_norm_1),
                .out_norm = try self.wrapped(l.out_norm),
            };
            const pds = try view.bytes(l.per_dim_scale);
            model.queryScale(@as([*]const f32, @ptrCast(@alignCast(pds.ptr)))[0..model.head_dim], out.query_scale.floats()[0..model.head_dim]);
        }
        self.input_projection = try self.plain(binding.input_projection);
        self.output = try self.plain(binding.output);
        self.output_bias = try self.wrapped(binding.output_bias);
        self.projection = try self.plain(binding.projection);
        self.relk = try self.created(model.blocks * model.relative * hidden * 4);
        self.scratch = try alloc.alloc(f32, @max(ffn * hidden, (2 * max_rows) * (mel.filters / 2) * model.channels[0]));
        errdefer alloc.free(self.scratch);
        try model.relativeKeys(io, view, binding, self.relk.floats()[0 .. model.blocks * model.relative * hidden], self.scratch);
        self.ones = try self.created(model.output_dims * 4);
        @memset(self.ones.floats()[0..model.output_dims], 1);
        self.sub = try self.created(rows * model.sub_width * 4);
        inline for (.{ "x", "n", "staged", "q", "k", "v", "attn" }) |field| @field(self, field) = try self.created(rows * hidden * 4);
        self.wide = try self.created(rows * ffn * 4);
        self.projected = try self.created(rows * model.output_dims * 4);
        self.out = try self.created(rows * binding.output_width * 4);
        return self;
    }
    pub fn deinit(self: *Plan) void {
        // Device buffers belong to the backend and go with it.
        self.alloc.free(self.scratch);
        self.* = undefined;
    }

    fn created(self: *Plan, len: usize) !Buffer {
        const buffer = try self.backend.create(len);
        self.created_bytes += len;
        return buffer;
    }
    fn wrapped(self: *Plan, tensor: *const gguf.Tensor) !Buffer {
        return self.backend.wrap(try self.view.bytes(tensor));
    }
    fn halved(self: *Plan, tensor: *const gguf.Tensor) !Buffer {
        const bytes = try self.view.bytes(tensor);
        const buffer = try self.created(bytes.len);
        const source = @as([*]const f32, @ptrCast(@alignCast(bytes.ptr)))[0 .. bytes.len / 4];
        for (buffer.floats()[0..source.len], source) |*o, w| o.* = 0.5 * w;
        return buffer;
    }
    fn plain(self: *Plan, tensor: *const gguf.Tensor) !Weight {
        const m = try self.view.matrix(tensor);
        if (m.rows % 8 != 0 or m.columns % 64 != 0) return error.UnsupportedConfig;
        return .{ .buffer = try self.backend.wrap(m.bytes), .matrix = m };
    }
    fn weight(self: *Plan, layer: model.Block, which: model.Linear) !Weight {
        var w = try self.plain(layer.matrix(which));
        w.bounds = try gemma4v.clampBounds(self.view, layer.clamps[@backingInt(which)]);
        return w;
    }

    /// `output = clamp(W · clamp(input))` over `n` rows. With `shared`, the
    /// input clamp goes to a copy in `staged`, leaving `input` for the next
    /// matrix that reads it.
    fn clipped(self: *Plan, w: Weight, input: Buffer, in_width: usize, output: Buffer, n: usize, shared: bool) !void {
        const b = self.backend;
        var source = input;
        if (!w.bounds.isOpen()) {
            if (shared) {
                try b.copy(self.staged, input, n * in_width);
                source = self.staged;
            }
            try b.clamp(source, n * in_width, w.bounds.input[0], w.bounds.input[1]);
        }
        try b.matmulTile(w.buffer, w.matrix, source, in_width, output, w.matrix.rows, n);
        if (!w.bounds.isOpen()) try b.clamp(output, n * w.matrix.rows, w.bounds.output[0], w.bounds.output[1]);
    }

    /// x += ½ · rms(W_down · silu(W_up · rms(x)·pre))·post.
    fn halfFeedForward(self: *Plan, pre: Buffer, up: Weight, down: Weight, post_half: Buffer, n: usize) !void {
        const b = self.backend;
        const norm: Backend.Norm = .{ .rows = n, .width = hidden, .in_stride = hidden, .out_stride = hidden, .eps = model.epsilon };
        try b.rmsNorm(self.x, pre, self.n, norm);
        try self.clipped(up, self.n, hidden, self.wide, n, false);
        try b.silu(self.wide, n * ffn);
        try self.clipped(down, self.wide, ffn, self.n, n, false);
        try b.rmsNormAdd(self.x, self.n, post_half, 1, norm);
    }

    /// Encodes `frames` mel rows into `out` (`tokensFor(frames) ×
    /// output_width`); commits and waits before the copy out.
    pub fn encode(self: *Plan, features: []const f32, frames: usize, out: []f32) !void {
        const n = model.tokensFor(frames);
        const width = self.binding.output_width;
        if (frames == 0 or n > max_rows or out.len != n * width) return error.InvalidShape;
        try model.subsample(self.view, self.binding, features, frames, self.sub.floats()[0 .. n * model.sub_width], self.scratch);
        const b = self.backend;
        const norm: Backend.Norm = .{ .rows = n, .width = hidden, .in_stride = hidden, .out_stride = hidden, .eps = model.epsilon };
        try b.begin();
        errdefer b.discard();
        try b.matmulTile(self.input_projection.buffer, self.input_projection.matrix, self.sub, model.sub_width, self.x, hidden, n);
        for (self.layers, 0..) |l, il| {
            try self.halfFeedForward(l.ffn_norm, l.ffn_up, l.ffn_down, l.ffn_post_half, n);
            try b.rmsNorm(self.x, l.attn_pre_norm, self.n, norm);
            try self.clipped(l.attn_q, self.n, hidden, self.q, n, true);
            try self.clipped(l.attn_k, self.n, hidden, self.k, n, true);
            try self.clipped(l.attn_v, self.n, hidden, self.v, n, true);
            const relk = self.relk.slice(il * model.relative * hidden * 4, model.relative * hidden * 4);
            try b.audioAttention(self.q, self.k, self.v, relk, l.query_scale, self.attn, n, model.key_scale, model.logit_cap);
            try self.clipped(l.attn_out, self.attn, hidden, self.n, n, false);
            try b.rmsNormAdd(self.x, self.n, l.attn_post_norm, 1, norm);
            try b.rmsNorm(self.x, l.conv_pre_norm, self.n, norm);
            try self.clipped(l.conv_pw1, self.n, hidden, self.wide, n, false);
            try b.gluConv(self.wide, l.conv_dw, self.attn, n, hidden);
            try b.rmsNorm(self.attn, l.conv_post_norm, self.attn, norm);
            try b.silu(self.attn, n * hidden);
            try self.clipped(l.conv_pw2, self.attn, hidden, self.n, n, false);
            try b.add(self.x, self.n, n * hidden);
            try self.halfFeedForward(l.ffn_norm_1, l.ffn_up_1, l.ffn_down_1, l.ffn_post_half_1, n);
            try b.rmsNorm(self.x, l.out_norm, self.x, norm);
        }
        try b.matmulTile(self.output.buffer, self.output.matrix, self.x, hidden, self.projected, model.output_dims, n);
        try b.addBiasRows(self.projected, self.output_bias, model.output_dims, n, model.output_dims);
        try b.rmsNorm(self.projected, self.ones, self.projected, .{ .rows = n, .width = model.output_dims, .in_stride = model.output_dims, .out_stride = model.output_dims, .eps = model.epsilon });
        try b.matmulTile(self.projection.buffer, self.projection.matrix, self.projected, model.output_dims, self.out, width, n);
        try b.commit();
        @memcpy(out, self.out.floats()[0 .. n * width]);
    }
};
