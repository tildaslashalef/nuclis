//! CPU execution of the pinned text-only Qwen schedule. Immutable weight views
//! and bindings borrow the loaded model; this runtime owns session and workspace.
//! A failed step poisons the session. No MTP, multimodal positions, or rewinding.
//! This is the numerical reference; `qwen35_metal.zig` runs the same schedule
//! on the GPU and is compared against it. On a rotated (Bonsai) file every
//! projection except `ssm_alpha` / `ssm_beta` reads its input through the
//! Hadamard transform and the embedding row is un-rotated after lookup.
const std = @import("std");
const model = @import("qwen35.zig");
const weights = @import("../runtime/weights.zig");
const session = @import("../runtime/session.zig");
const cpu = @import("../backends/cpu/root.zig");
const Tensor = @import("../formats/gguf.zig").Tensor;

/// The shared layer-boundary callbacks (`runtime/observer.zig`); re-exported
/// so the runtime's own signatures and `generation-check` name one type.
pub const Observer = @import("../runtime/observer.zig").Observer;

/// Small F32 weights decoded once per layer at init. Reading them through the
/// mapping per element was a measurable CPU hotspot on every backend.
const LayerConstants = struct {
    attention_norm: []f32,
    post_attention_norm: []f32,
    mixer: union(enum) {
        full_attention: struct { query_norm: []f32, key_norm: []f32 },
        delta_net: struct { convolution: []f32, a: []f32, time_bias: []f32, norm: []f32 },
    },
};

/// The binding's rotation with its sign vectors decoded once to F32,
/// indexed as `model.Rotation.widths`.
const Rotation = struct {
    block: usize,
    signs: [3][]f32,
    value_grouped: bool,

    fn signsFor(self: Rotation, width: usize) ![]const f32 {
        for (model.Rotation.widths, self.signs) |w, signs| if (w == width) return signs;
        return error.InvalidShape;
    }
};

pub const Runtime = struct {
    storage: std.heap.ArenaAllocator,
    state: session.Session,
    view: weights.View,
    binding: model.Binding,
    rotation: ?Rotation,
    constants: []LayerConstants,
    output_norm: []f32,
    x: []f32,
    normalized: []f32,
    projected: []f32,
    gate: []f32,
    up: []f32,
    row: []f32,
    qg: []f32,
    q: []f32,
    k: []f32,
    v: []f32,
    mixed: []f32,
    convolved: []f32,
    z: []f32,
    alpha: []f32,
    beta: []f32,
    mixed_out: []f32,
    rotated: []f32,
    attention_scratch: []f64,
    delta_scratch: []f64,

    pub fn init(gpa: std.mem.Allocator, view: weights.View, binding: model.Binding, capacity: usize, checkpoint: bool) !Runtime {
        var layouts: [64]session.Layout = undefined;
        for (binding.layers, &layouts) |layer, *layout| layout.* = switch (layer.mixer) {
            .full_attention => .{ .attention = .{ .key_row = 1024, .value_row = 1024 } },
            .delta_net => .{ .recurrent = .{ .history = 10240 * 3, .matrix = 48 * 128 * 128 } },
        };
        var state = try session.Session.init(gpa, &layouts, capacity, checkpoint);
        errdefer state.deinit();
        var storage: std.heap.ArenaAllocator = .init(gpa);
        errdefer storage.deinit();
        const a = storage.allocator();
        // Finish allocations before transferring the arena, whose internal
        // linked-list head can change on each allocation.
        var result: Runtime = undefined;
        inline for (.{ "x", "normalized", "projected" }) |field| @field(result, field) = try a.alloc(f32, 5120);
        inline for (.{ "gate", "up", "row" }) |field| @field(result, field) = try a.alloc(f32, 17408);
        result.qg = try a.alloc(f32, 12288);
        result.q = try a.alloc(f32, 6144);
        result.k = try a.alloc(f32, 1024);
        result.v = try a.alloc(f32, 1024);
        result.mixed = try a.alloc(f32, 10240);
        result.convolved = try a.alloc(f32, 10240);
        result.z = try a.alloc(f32, 6144);
        result.mixed_out = try a.alloc(f32, 6144);
        result.alpha = try a.alloc(f32, 48);
        result.beta = try a.alloc(f32, 48);
        result.rotated = try a.alloc(f32, 17408);
        result.rotation = null;
        if (binding.rotation) |rotation| {
            var signs: [3][]f32 = undefined;
            for (&signs, rotation.signs) |*decoded, source| {
                decoded.* = try a.alloc(f32, source.values.len);
                for (decoded.*, source.values) |*out, value| out.* = @floatFromInt(value.signed);
            }
            result.rotation = .{ .block = rotation.block, .signs = signs, .value_grouped = rotation.value_grouped };
        }
        result.attention_scratch = try a.alloc(f64, capacity);
        result.delta_scratch = try a.alloc(f64, 128 * 129);
        result.output_norm = try view.vector(a, binding.output_norm);
        result.constants = try a.alloc(LayerConstants, binding.layers.len);
        for (binding.layers, result.constants) |layer, *constants| {
            constants.* = .{
                .attention_norm = try view.vector(a, layer.attention_norm),
                .post_attention_norm = try view.vector(a, layer.post_attention_norm),
                .mixer = switch (layer.mixer) {
                    .full_attention => |attn| .{ .full_attention = .{ .query_norm = try view.vector(a, attn.query_norm), .key_norm = try view.vector(a, attn.key_norm) } },
                    .delta_net => |linear| .{ .delta_net = .{ .convolution = try view.vector(a, linear.convolution), .a = try view.vector(a, linear.a), .time_bias = try view.vector(a, linear.time_bias), .norm = try view.vector(a, linear.norm) } },
                },
            };
        }
        result.storage = storage;
        result.state = state;
        result.view = view;
        result.binding = binding;
        return result;
    }
    pub fn deinit(self: *Runtime) void {
        self.state.deinit();
        self.storage.deinit();
        self.* = undefined;
    }
    pub fn reset(self: *Runtime) void {
        self.state.reset();
    }

    fn mm(self: *Runtime, tensor: *const Tensor, input: []const f32, output: []f32) !void {
        try cpu.matvec(try self.view.matrix(tensor), input, output, self.row);
    }
    /// The activation a rotated projection reads: on a rotated file, `input`
    /// sign-flipped and transformed into the `rotated` scratch (so callers
    /// that also feed an unrotated projection keep `input`); else `input`.
    fn rotate(self: *Runtime, input: []const f32) ![]const f32 {
        const rotation = self.rotation orelse return input;
        const out = self.rotated[0..input.len];
        @memcpy(out, input);
        try cpu.hadamard.forward(out, try rotation.signsFor(input.len), rotation.block);
        return out;
    }
    /// `ssm_out`'s input: the fold was computed with the 48 value heads in
    /// group order (`nk * 3 + rep`) while the mixer emits them tiled
    /// (`rep * 16 + nk`), so the heads are regathered before the transform.
    fn rotateGrouped(self: *Runtime, input: []const f32) ![]const f32 {
        const rotation = self.rotation orelse return input;
        if (!rotation.value_grouped) return self.rotate(input);
        if (input.len != 6144) return error.InvalidShape;
        const out = self.rotated[0..6144];
        for (0..3) |rep| for (0..16) |nk| {
            @memcpy(out[(nk * 3 + rep) * 128 ..][0..128], input[(rep * 16 + nk) * 128 ..][0..128]);
        };
        try cpu.hadamard.forward(out, try rotation.signsFor(6144), rotation.block);
        return out;
    }
    fn norm(input: []const f32, output: []f32, weight: []const f32) !void {
        if (weight.len != output.len) return error.InvalidShape;
        try cpu.rmsNorm(input, output, 1e-6);
        for (output, weight) |*x, w| x.* *= w;
    }
    /// logits may be null for non-final prefill tokens, avoiding the LM head.
    /// It must otherwise hold exactly the vocabulary size. A successful step
    /// commits one token; a callback/error after begin requires reset.
    pub fn step(self: *Runtime, token: u32, logits: ?[]f32, observer: ?Observer) !void {
        if (token >= 248320) return error.InvalidTokenId;
        if (logits) |out| if (out.len != 248320) return error.InvalidShape;
        try self.state.begin();
        errdefer self.state.fail();
        try self.view.row(self.binding.token_embedding, token, self.x);
        if (self.rotation) |rotation| try cpu.hadamard.inverse(self.x, try rotation.signsFor(5120), rotation.block);
        for (self.binding.layers, self.constants, 0..) |layer, constants, il| {
            try norm(self.x, self.normalized, constants.attention_norm);
            switch (layer.mixer) {
                .full_attention => |attn| try self.fullAttention(attn, constants.mixer.full_attention, il),
                .delta_net => |linear| try self.linearAttention(linear, constants.mixer.delta_net, il),
            }
            for (self.x, self.projected) |*x, contribution| x.* += contribution;
            try norm(self.x, self.normalized, constants.post_attention_norm);
            const ffn_input = try self.rotate(self.normalized);
            try self.mm(layer.ffn_gate, ffn_input, self.gate);
            try self.mm(layer.ffn_up, ffn_input, self.up);
            for (self.gate, self.up) |*g, u| g.* = cpu.silu(g.*) * u;
            try self.mm(layer.ffn_down, try self.rotate(self.gate), self.projected);
            for (self.x, self.projected) |*x, contribution| {
                x.* += contribution;
                if (!std.math.isFinite(x.*)) return error.NonFiniteResult;
            }
            if (observer) |o| {
                if (o.check) |check| try check(o.context);
                if (o.layer) |report| try report(o.context, il, self.x);
            }
        }
        if (logits) |out| {
            try norm(self.x, self.normalized, self.output_norm);
            try self.mm(self.binding.output, try self.rotate(self.normalized), out);
            for (out) |x| if (!std.math.isFinite(x)) return error.NonFiniteResult;
        }
        try self.state.commit();
    }

    fn fullAttention(self: *Runtime, attn: model.FullAttention, constants: anytype, il: usize) !void {
        const input = try self.rotate(self.normalized);
        try self.mm(attn.query_and_gate, input, self.qg);
        try self.mm(attn.key, input, self.k);
        try self.mm(attn.value, input, self.v);
        for (0..24) |h| {
            const q = self.q[h * 256 ..][0..256];
            // Each projected head stores query then gate, not all queries then
            // all gates. The gate remains untouched until after attention.
            try norm(self.qg[h * 512 ..][0..256], q, constants.query_norm);
            try cpu.rope.apply(q, q, .{ .dimensions = 64, .base = 1e7, .position = @intCast(self.state.position) });
        }
        for (0..4) |h| {
            const k = self.k[h * 256 ..][0..256];
            try norm(k, k, constants.key_norm);
            try cpu.rope.apply(k, k, .{ .dimensions = 64, .base = 1e7, .position = @intCast(self.state.position) });
        }
        const cache = self.state.layers[il].attention;
        const position = self.state.position;
        // The reference cache is F32 by decision: the views assert it.
        @memcpy(cache.keys.floats(position, 1), self.k);
        @memcpy(cache.values.floats(position, 1), self.v);
        try cpu.attention.apply(.{ .query_heads = 24, .kv_heads = 4, .key_width = 256, .value_width = 256, .tokens = position + 1, .visible_tokens = position + 1, .scale = 1.0 / 16.0, .queries = self.q, .keys = cache.keys.floats(0, position + 1), .values = cache.values.floats(0, position + 1) }, self.mixed_out, self.attention_scratch);
        for (0..24) |h| for (0..256) |i| {
            self.mixed_out[h * 256 + i] *= cpu.sigmoid(self.qg[h * 512 + 256 + i]);
        };
        try self.mm(attn.output, try self.rotate(self.mixed_out), self.projected);
    }

    fn linearAttention(self: *Runtime, linear: model.DeltaNet, constants: anytype, il: usize) !void {
        const input = try self.rotate(self.normalized);
        try self.mm(linear.qkv, input, self.mixed);
        try self.mm(linear.gate, input, self.z);
        // The gating projections were left in the original basis.
        try self.mm(linear.beta, self.normalized, self.beta);
        try self.mm(linear.alpha, self.normalized, self.alpha);
        const state = self.state.layers[il].recurrent;
        try cpu.recurrent.convolution(self.mixed, constants.convolution, state.history, self.convolved, 4);
        for (self.convolved) |*x| x.* = cpu.silu(x.*);
        // Convolution output is [Q:16x128 | K:16x128 | V:48x128].
        for (0..32) |h| {
            const row = self.convolved[h * 128 ..][0..128];
            try cpu.l2Norm(row, row, 1e-6);
        }
        if (constants.a.len != 48 or constants.time_bias.len != 48) return error.InvalidShape;
        for (0..48) |h| {
            self.alpha[h] = constants.a[h] * cpu.softplus(self.alpha[h] + constants.time_bias[h]);
            self.beta[h] = cpu.sigmoid(self.beta[h]);
        }
        for (0..48) |h| {
            const kh = h % 16; // tiled Q/K broadcast, unlike attention's GQA
            const matrix = state.matrix[h * 128 * 128 ..][0 .. 128 * 128];
            const out = self.mixed_out[h * 128 ..][0..128];
            try cpu.recurrent.delta(.{ .query = self.convolved[kh * 128 ..][0..128], .key = self.convolved[2048 + kh * 128 ..][0..128], .value = self.convolved[4096 + h * 128 ..][0..128], .log_decay = self.alpha[h], .beta = self.beta[h], .scale = 1.0 / @sqrt(@as(f32, 128)) }, matrix, matrix, out, self.delta_scratch);
        }
        for (0..48) |h| {
            const out = self.mixed_out[h * 128 ..][0..128];
            try norm(out, out, constants.norm);
            for (out, self.z[h * 128 ..][0..128]) |*x, z| x.* *= cpu.silu(z);
        }
        try self.mm(linear.output, try self.rotateGrouped(self.mixed_out), self.projected);
    }
};

test "runtime workspace cleanup and invalid steps preserve session admission" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, struct {
        fn check(alloc: std.mem.Allocator) !void {
            // The binding only supplies layer kinds during preparation. A tiny
            // empty tensor deliberately fails embedding access after admission.
            const tensor: Tensor = .{ .name = "empty", .dimensions = &.{0}, .encoding_id = 0, .offset = 0, .elements = 0, .bytes = 0 };
            const attention: model.FullAttention = .{ .query_and_gate = &tensor, .key = &tensor, .value = &tensor, .output = &tensor, .query_norm = &tensor, .key_norm = &tensor };
            const layer: model.Layer = .{ .attention_norm = &tensor, .post_attention_norm = &tensor, .ffn_gate = &tensor, .ffn_up = &tensor, .ffn_down = &tensor, .mixer = .{ .full_attention = attention } };
            const binding: model.Binding = .{ .token_embedding = &tensor, .output_norm = &tensor, .output = &tensor, .layers = @splat(layer), .summary = .{ .profile = "test", .decoder_layers = 64, .layer_kinds = &.{}, .text_tensors = 0, .auxiliary_tensors = 0, .text_tensor_bytes = 0, .auxiliary_tensor_bytes = 0 } };
            var runtime = try Runtime.init(alloc, .{ .file = &.{}, .data_offset = 0 }, binding, 1, false);
            defer runtime.deinit();
            try std.testing.expectError(error.InvalidTokenId, runtime.step(248320, null, null));
            try std.testing.expectEqual(.ready, runtime.state.status);
            try std.testing.expectError(error.InvalidShape, runtime.step(0, &.{}, null));
            try std.testing.expectEqual(.ready, runtime.state.status);
            if (runtime.step(0, null, null)) |_| return error.ExpectedInvalidEmbedding else |_| {}
            try std.testing.expectEqual(.failed, runtime.state.status);
            try std.testing.expectError(error.SessionNotReady, runtime.step(0, null, null));
            runtime.reset();
            try std.testing.expectEqual(.ready, runtime.state.status);
            try std.testing.expectEqual(@as(usize, 0), runtime.state.position);
        }
    }.check, .{});
}
