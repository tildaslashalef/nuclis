//! Sampling on the device: penalties, top-k, argmax: the host side of `kernels/sampling.metal`. Methods of `Backend`,
//! which `root.zig` aliases so callers write `backend.topkBuffers(…)`.
const std = @import("std");
const root = @import("../root.zig");
const cpu = @import("../../cpu/root.zig");
const quant = @import("../../../quant/decode.zig");
const Backend = root.Backend;
const Buffer = root.Buffer;
const Kernel = root.Kernel;
const Precision = root.Precision;
const enabled = root.enabled;
const Shape = @import("../profile.zig").Shape;
const perElement = Backend.perElement;

pub const TopKParams = extern struct { count: u32, partials: u32, k: u32, temperature: f32 };
pub const topk_partials = 64;
/// Values each partial-pass thread keeps in registers (`NU_TOPK_LOCAL`);
/// bounds the vocabulary size the kernels accept.
const topk_local = 16;
pub const topk_max_count = topk_partials * 256 * topk_local;
/// Scratch and result buffers for `topk`, sized by `topkBuffers`.
pub const TopKBuffers = struct {
    partial_values: Buffer, // partials * k floats
    partial_indices: Buffer, // partials * k u32
    values: Buffer, // k floats, sorted (value desc, index asc)
    indices: Buffer, // k u32
    sums: Buffer, // partials floats: F32 partial sums of exp((l - max) / T)
    flags: Buffer, // partials u32: nonzero when the partition saw a non-finite logit
};
pub fn topkBuffers(self: *Backend, k: usize) !TopKBuffers {
    return .{
        .partial_values = try self.create(topk_partials * k * 4),
        .partial_indices = try self.create(topk_partials * k * 4),
        .values = try self.create(k * 4),
        .indices = try self.create(k * 4),
        .sums = try self.create(topk_partials * 4),
        .flags = try self.create(topk_partials * 4),
    };
}
/// The `k` best of `count` logits by (value desc, index asc) into
/// `scratch.values`/`indices`, and Σ exp((l − max) / temperature) as
/// `topk_partials` F32 partial sums with per-partition non-finite flags.
/// The CPU adds the partials in F64 (see `sampling.TopK`). Three dispatches.
pub fn topk(self: *Backend, logits: Buffer, count: usize, k: usize, temperature: f32, scratch: TopKBuffers) !void {
    if (!enabled) return error.MetalNotEnabled;
    if (count == 0 or count > topk_max_count or k == 0 or k > 256 or logits.len < count * 4) return error.InvalidShape;
    if (!std.math.isFinite(temperature) or temperature <= 0) return error.InvalidShape;
    if (scratch.partial_values.len < topk_partials * k * 4 or scratch.partial_indices.len < topk_partials * k * 4 or
        scratch.values.len < k * 4 or scratch.indices.len < k * 4 or scratch.sums.len < topk_partials * 4 or scratch.flags.len < topk_partials * 4) return error.InvalidShape;
    const p: TopKParams = .{ .count = @intCast(count), .partials = topk_partials, .k = @intCast(k), .temperature = temperature };
    try self.dispatch(.topk_partial, &.{ logits, scratch.partial_values, scratch.partial_indices }, p, topk_partials, 256, .{});
    try self.dispatch(.topk_final, &.{ scratch.partial_values, scratch.partial_indices, scratch.values, scratch.indices }, p, 1, 256, .{});
    try self.dispatch(.expsum_partial, &.{ logits, scratch.values, scratch.sums, scratch.flags }, p, topk_partials, 256, .{});
}

pub const PenalizeParams = extern struct { count: u32, history_words: u32, repetition: f32, presence: f32 };
/// Applies the history penalties to `logits[0..count]` in place:
/// `l / r` for positive and `l · r` for negative logits of every token in
/// `history` (bit `id` in word `id / 32`), then `l − presence`. One thread
/// per logit; `repetition` must be positive and both values finite. The
/// readback taken after this is the penalized logits, which is what the
/// CPU sampler sorts (`sampling.Sampler.penalize`).
pub fn penalize(self: *Backend, logits: Buffer, count: usize, history: Buffer, repetition: f32, presence: f32) !void {
    if (count == 0 or count > std.math.maxInt(u32) or logits.len < count * 4) return error.InvalidShape;
    if (!std.math.isFinite(repetition) or repetition <= 0 or !std.math.isFinite(presence)) return error.InvalidShape;
    const words = @divCeil(count, 32);
    if (history.len < words * 4) return error.InvalidShape;
    const p: PenalizeParams = .{ .count = @intCast(count), .history_words = @intCast(words), .repetition = repetition, .presence = presence };
    try self.dispatch(.penalize, &.{ logits, history }, p, perElement(count), 256, .{});
}

pub const ArgmaxParams = extern struct { count: u32, partials: u32 };
pub const argmax_partials = 64;
/// Two-pass greedy selection with lowest-index tie breaking. `values` and
/// `indices` hold `argmax_partials` scratch entries; `result` receives one u32.
pub fn argmax(self: *Backend, logits: Buffer, count: usize, values: Buffer, indices: Buffer, result: Buffer) !void {
    if (count == 0 or logits.len < count * 4 or values.len < argmax_partials * 4 or indices.len < argmax_partials * 4 or result.len < 4) return error.InvalidShape;
    const p: ArgmaxParams = .{ .count = @intCast(count), .partials = argmax_partials };
    try self.dispatch(.argmax_partial, &.{ logits, values, indices }, p, argmax_partials, 256, .{});
    try self.dispatch(.argmax_final, &.{ values, indices, result }, p, 1, 32, .{});
}
