//! The Qwen3-VL projector on Metal: the CPU reference's schedule
//! (`qwen3vl.Runtime.encode`) as one command buffer over the batched
//! matmul tiles, the LayerNorm, bias, GELU, and bidirectional attention
//! kernels, and the per-row RoPE table. Weights are the mapped file's
//! bytes wrapped in place, except the summed patch kernel and the FFN down
//! matrices (F32, F16, or BF16), whose 4,304 columns are zero-padded to the
//! tile's 64.
const std = @import("std");
const metal = @import("../backends/metal/root.zig");
const cpu = @import("../backends/cpu/root.zig");
const weights = @import("../runtime/weights.zig");
const preprocess = @import("preprocess.zig");
const model = @import("qwen3vl.zig");

const Backend = metal.Backend;
const Buffer = metal.Buffer;

const hidden = model.hidden;
const ffn = model.ffn;
/// The FFN width rounded up to the matmul tile's column multiple.
pub const ffn_padded = (ffn + 63) / 64 * 64;
const qkv_width = model.qkv_width;
const merged = model.merged_width;
const out_width = model.output_width;
const max_patches = model.max_patches;
const max_tokens = model.max_tokens;

const Layer = struct {
    n1w: Buffer,
    n1b: Buffer,
    qkv: Buffer,
    qkv_m: cpu.Matrix,
    qkv_b: Buffer,
    out: Buffer,
    out_m: cpu.Matrix,
    out_b: Buffer,
    n2w: Buffer,
    n2b: Buffer,
    up: Buffer,
    up_m: cpu.Matrix,
    up_b: Buffer,
    /// The padded copy, `hidden × ffn_padded` in the file's element encoding.
    down: Buffer,
    down_m: cpu.Matrix,
    down_b: Buffer,
};

pub const Plan = struct {
    alloc: std.mem.Allocator,
    backend: *Backend,
    view: weights.View,
    binding: *const model.Binding,
    kernel: Buffer,
    kernel_m: cpu.Matrix,
    patch_bias: Buffer,
    layers: [model.blocks]Layer,
    post_w: Buffer,
    post_b: Buffer,
    m0: Buffer,
    m0_m: cpu.Matrix,
    m0_b: Buffer,
    m2: Buffer,
    m2_m: cpu.Matrix,
    m2_b: Buffer,
    // Activations for `max_patches` rows.
    input: Buffer,
    x: Buffer,
    h: Buffer,
    qkv: Buffer,
    attn: Buffer,
    up: Buffer,
    pos: Buffer,
    rope: Buffer,
    mid: Buffer,
    out: Buffer,
    scratch: []f32,
    /// Bytes of device memory the plan created (weights copied or padded,
    /// and the activations); the wrapped file bytes are not counted.
    created_bytes: usize,

    pub fn init(alloc: std.mem.Allocator, backend: *Backend, view: weights.View, binding: *const model.Binding) !Plan {
        var self: Plan = undefined;
        self.alloc = alloc;
        self.backend = backend;
        self.view = view;
        self.binding = binding;
        self.created_bytes = 0;
        const kernel = try model.patchKernel(alloc, view, binding);
        defer alloc.free(kernel);
        self.kernel = try self.created(hidden * model.patch_values * 4);
        @memcpy(self.kernel.floats()[0 .. hidden * model.patch_values], kernel);
        self.kernel_m = .{ .encoding = 0, .rows = hidden, .columns = model.patch_values, .bytes = self.kernel.host[0 .. hidden * model.patch_values * 4] };
        self.patch_bias = try self.wrapped(binding.patch_bias);
        for (&self.layers, binding.layers) |*layer, l| {
            layer.n1w = try self.wrapped(l.norm1.weight);
            layer.n1b = try self.wrapped(l.norm1.bias);
            layer.qkv_m = try view.matrix(l.qkv.weight);
            layer.qkv = try backend.wrap(layer.qkv_m.bytes);
            layer.qkv_b = try self.wrapped(l.qkv.bias);
            layer.out_m = try view.matrix(l.output.weight);
            layer.out = try backend.wrap(layer.out_m.bytes);
            layer.out_b = try self.wrapped(l.output.bias);
            layer.n2w = try self.wrapped(l.norm2.weight);
            layer.n2b = try self.wrapped(l.norm2.bias);
            layer.up_m = try view.matrix(l.up.weight);
            layer.up = try backend.wrap(layer.up_m.bytes);
            layer.up_b = try self.wrapped(l.up.bias);
            const down = try view.matrix(l.down.weight);
            const element: usize = if (down.encoding == 0) 4 else 2;
            if (down.bytes.len != hidden * ffn * element) return error.InvalidShape;
            layer.down = try self.created(hidden * ffn_padded * element);
            const host = layer.down.host[0 .. hidden * ffn_padded * element];
            for (0..hidden) |r| @memcpy(host[r * ffn_padded * element ..][0 .. ffn * element], down.bytes[r * ffn * element ..][0 .. ffn * element]);
            layer.down_m = .{ .encoding = down.encoding, .rows = hidden, .columns = ffn_padded, .bytes = host };
            layer.down_b = try self.wrapped(l.down.bias);
        }
        self.post_w = try self.wrapped(binding.post_norm.weight);
        self.post_b = try self.wrapped(binding.post_norm.bias);
        self.m0_m = try view.matrix(binding.merger_0.weight);
        self.m0 = try backend.wrap(self.m0_m.bytes);
        self.m0_b = try self.wrapped(binding.merger_0.bias);
        self.m2_m = try view.matrix(binding.merger_2.weight);
        self.m2 = try backend.wrap(self.m2_m.bytes);
        self.m2_b = try self.wrapped(binding.merger_2.bias);
        self.input = try self.created(max_patches * model.patch_values * 4);
        self.x = try self.created(max_patches * hidden * 4);
        self.h = try self.created(max_patches * hidden * 4);
        self.qkv = try self.created(max_patches * qkv_width * 4);
        self.attn = try self.created(max_patches * hidden * 4);
        self.up = try self.created(max_patches * ffn_padded * 4);
        self.pos = try self.created(max_patches * hidden * 4);
        self.rope = try self.created(max_patches * model.rope_pairs * 8);
        self.mid = try self.created(max_tokens * merged * 4);
        self.out = try self.created(max_tokens * out_width * 4);
        self.scratch = try alloc.alloc(f32, 4 * hidden);
        return self;
    }
    fn created(self: *Plan, len: usize) !Buffer {
        const buffer = try self.backend.create(len);
        self.created_bytes += len;
        return buffer;
    }
    fn wrapped(self: *Plan, tensor: anytype) !Buffer {
        return self.backend.wrap(try self.view.bytes(tensor));
    }
    pub fn deinit(self: *Plan) void {
        // Device buffers belong to the backend and go with it.
        self.alloc.free(self.scratch);
        self.* = undefined;
    }

    /// Encodes `patches` into `out` (`grid.tokens() × output_width` rows);
    /// one command buffer, committed and waited for before the copy out.
    pub fn encode(self: *Plan, patches: preprocess.Patches, out: []f32) !void {
        const grid: model.Grid = .{ .width_patches = patches.width_patches, .height_patches = patches.height_patches };
        const n = grid.patches();
        const tokens = grid.tokens();
        if (n == 0 or n > max_patches or patches.row != model.patch_values or out.len < tokens * out_width) return error.InvalidShape;
        const b = self.backend;
        @memcpy(self.input.floats()[0 .. n * model.patch_values], patches.values);
        try model.positionRows(self.view, self.binding.position_embedding, grid, self.pos.floats(), self.scratch);
        try model.ropeTable(self.rope.floats(), grid);
        // The tile reads the padded rows past `n`; leave them finite.
        @memset(self.x.floats(), 0);
        @memset(self.h.floats(), 0);

        try b.begin();
        errdefer b.commit() catch {};
        try b.matmul(self.kernel, self.kernel_m, self.input, model.patch_values, self.x, hidden, n);
        try b.addBiasRows(self.x, self.patch_bias, hidden, n, hidden);
        try b.add(self.x, self.pos, n * hidden);
        const keys = self.qkv.slice(hidden * 4, self.qkv.len - hidden * 4);
        const values = self.qkv.slice(2 * hidden * 4, self.qkv.len - 2 * hidden * 4);
        for (self.layers) |layer| {
            try b.layerNorm(self.x, layer.n1w, layer.n1b, self.h, .{ .rows = n, .width = hidden, .in_stride = hidden, .out_stride = hidden, .eps = model.norm_epsilon });
            try b.matmul(layer.qkv, layer.qkv_m, self.h, hidden, self.qkv, qkv_width, n);
            try b.addBiasRows(self.qkv, layer.qkv_b, qkv_width, n, qkv_width);
            try b.ropeRows(self.qkv, self.rope, model.heads, model.head_dim, model.head_dim, 0, n, qkv_width, .split_half);
            try b.ropeRows(keys, self.rope, model.heads, model.head_dim, model.head_dim, 0, n, qkv_width, .split_half);
            try b.attentionFull(self.qkv, keys, values, self.attn, .{ .heads = model.heads, .width = model.head_dim, .rows = n, .q_stride = qkv_width, .kv_stride = qkv_width, .out_stride = hidden, .scale = 1.0 / @sqrt(@as(f32, model.head_dim)) });
            try b.matmul(layer.out, layer.out_m, self.attn, hidden, self.h, hidden, n);
            try b.addBiasRows(self.h, layer.out_b, hidden, n, hidden);
            try b.add(self.x, self.h, n * hidden);
            try b.layerNorm(self.x, layer.n2w, layer.n2b, self.h, .{ .rows = n, .width = hidden, .in_stride = hidden, .out_stride = hidden, .eps = model.norm_epsilon });
            try b.matmul(layer.up, layer.up_m, self.h, hidden, self.up, ffn_padded, n);
            try b.addBiasRows(self.up, layer.up_b, ffn, n, ffn_padded);
            try b.gelu(self.up, n * ffn_padded);
            try b.matmul(layer.down, layer.down_m, self.up, ffn_padded, self.h, hidden, n);
            try b.addBiasRows(self.h, layer.down_b, hidden, n, hidden);
            try b.add(self.x, self.h, n * hidden);
        }
        try b.layerNorm(self.x, self.post_w, self.post_b, self.h, .{ .rows = n, .width = hidden, .in_stride = hidden, .out_stride = hidden, .eps = model.norm_epsilon });
        try b.matmul(self.m0, self.m0_m, self.h, merged, self.mid, merged, tokens);
        try b.addBiasRows(self.mid, self.m0_b, merged, tokens, merged);
        try b.gelu(self.mid, tokens * merged);
        try b.matmul(self.m2, self.m2_m, self.mid, merged, self.out, out_width, tokens);
        try b.addBiasRows(self.out, self.m2_b, out_width, tokens, out_width);
        try b.commit();
        @memcpy(out[0 .. tokens * out_width], self.out.floats()[0 .. tokens * out_width]);
    }
};
