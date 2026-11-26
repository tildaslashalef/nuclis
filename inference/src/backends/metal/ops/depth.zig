//! Attention over the depth: the host side of `kernels/depth.metal`, the
//! Metal counterparts of `cpu.depth` (`mix`, and `control`'s norms and
//! blend; its three products run on `matvec`). Methods of `Backend`, which
//! `root.zig` aliases so callers write `backend.depthMix(…)`.
const std = @import("std");
const root = @import("../root.zig");
const Backend = root.Backend;
const Buffer = root.Buffer;
const perElement = Backend.perElement;

/// The most states one `depthMix` row reads (the kernel's `NU_DEPTH_MAX`).
pub const depth_max_states = 32;
pub const DepthMixParams = extern struct { width: u32, count: u32, queries: u32, state_stride: u32, out_stride: u32, eps: f32 };
/// `cpu.depth.mix` for each of `tokens` tokens: its `count` states are rows
/// of `width` from `states + token · state_stride`, the `queries` rows of
/// `queries_buffer` are shared, and its outputs are `queries` rows of
/// `width` from `output + token · out_stride` (floats). The output must not
/// overlap the inputs.
pub fn depthMix(self: *Backend, states: Buffer, queries_buffer: Buffer, output: Buffer, width: usize, count: usize, queries: usize, tokens: usize, state_stride: usize, out_stride: usize, eps: f32) !void {
    if (width == 0 or count == 0 or count > depth_max_states or queries == 0 or tokens == 0 or !(eps > 0) or !std.math.isFinite(eps)) return error.InvalidShape;
    if (state_stride < count * width or out_stride < queries * width) return error.InvalidShape;
    const states_len = ((tokens - 1) * state_stride + count * width) * 4;
    const out_len = ((tokens - 1) * out_stride + queries * width) * 4;
    if (states.len < states_len or queries_buffer.len < queries * width * 4 or output.len < out_len) return error.InvalidShape;
    if (Buffer.overlaps(output, out_len, states, states_len) or Buffer.overlaps(output, out_len, queries_buffer, queries * width * 4)) return error.InvalidShape;
    const p: DepthMixParams = .{ .width = @intCast(width), .count = @intCast(count), .queries = @intCast(queries), .state_stride = @intCast(state_stride), .out_stride = @intCast(out_stride), .eps = eps };
    try self.dispatch(.depth_mix, &.{ states, queries_buffer, output }, p, @intCast(tokens * queries), 256, .{});
}

pub const ControlJoinParams = extern struct { width: u32, rows: u32, current_stride: u32, message_stride: u32, joined_stride: u32, eps: f32 };
/// `cpu.depth.control`'s product input: `joined[row] = [rms(current[row]);
/// rms(message[row])]` (`2 · width` floats, no gain). `joined` must not
/// overlap the inputs.
pub fn controlJoin(self: *Backend, current: Buffer, message: Buffer, joined: Buffer, width: usize, rows: usize, current_stride: usize, message_stride: usize, joined_stride: usize, eps: f32) !void {
    if (width == 0 or rows == 0 or current_stride < width or message_stride < width or joined_stride < 2 * width or !(eps > 0) or !std.math.isFinite(eps)) return error.InvalidShape;
    const joined_len = ((rows - 1) * joined_stride + 2 * width) * 4;
    const current_len = ((rows - 1) * current_stride + width) * 4;
    const message_len = ((rows - 1) * message_stride + width) * 4;
    if (current.len < current_len or message.len < message_len or joined.len < joined_len) return error.InvalidShape;
    if (Buffer.overlaps(joined, joined_len, current, current_len) or Buffer.overlaps(joined, joined_len, message, message_len)) return error.InvalidShape;
    const p: ControlJoinParams = .{ .width = @intCast(width), .rows = @intCast(rows), .current_stride = @intCast(current_stride), .message_stride = @intCast(message_stride), .joined_stride = @intCast(joined_stride), .eps = eps };
    try self.dispatch(.control_join, &.{ current, message, joined }, p, @intCast(2 * rows), 256, .{});
}

pub const ControlBlendParams = extern struct { width: u32, rows: u32, current_stride: u32, message_stride: u32, gate_stride: u32, out_stride: u32 };
/// `output = (1 − tanh(forget)) ⊙ current + tanh(gain) ⊙ message` per row,
/// `gain` and `forget` rows `gate_stride` apart. `output` may alias
/// `current` exactly (same offset and stride); it must not overlap the
/// other inputs.
pub fn controlBlend(self: *Backend, current: Buffer, message: Buffer, gain: Buffer, forget: Buffer, output: Buffer, width: usize, rows: usize, current_stride: usize, message_stride: usize, gate_stride: usize, out_stride: usize) !void {
    if (width == 0 or rows == 0 or current_stride < width or message_stride < width or gate_stride < width or out_stride < width) return error.InvalidShape;
    const len = struct {
        fn of(r: usize, stride: usize, w: usize) usize {
            return ((r - 1) * stride + w) * 4;
        }
    }.of;
    const out_len = len(rows, out_stride, width);
    if (current.len < len(rows, current_stride, width) or message.len < len(rows, message_stride, width) or gain.len < len(rows, gate_stride, width) or forget.len < len(rows, gate_stride, width) or output.len < out_len) return error.InvalidShape;
    const aliased = output.id == current.id and output.offset == current.offset and out_stride == current_stride;
    if (!aliased and Buffer.overlaps(output, out_len, current, len(rows, current_stride, width))) return error.InvalidShape;
    if (Buffer.overlaps(output, out_len, message, len(rows, message_stride, width)) or Buffer.overlaps(output, out_len, gain, len(rows, gate_stride, width)) or Buffer.overlaps(output, out_len, forget, len(rows, gate_stride, width))) return error.InvalidShape;
    const p: ControlBlendParams = .{ .width = @intCast(width), .rows = @intCast(rows), .current_stride = @intCast(current_stride), .message_stride = @intCast(message_stride), .gate_stride = @intCast(gate_stride), .out_stride = @intCast(out_stride) };
    try self.dispatch(.control_blend, &.{ current, message, gain, forget, output }, p, perElement(width * rows), 256, .{});
}
