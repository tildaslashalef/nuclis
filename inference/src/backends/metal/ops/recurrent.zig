//! Recurrent layers: DeltaNet steps and the short convolutions: the host side of `kernels/recurrent.metal`. Methods of `Backend`,
//! which `root.zig` aliases so callers write `backend.delta(…)`.
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

pub const DeltaParams = extern struct { qheads: u32, vheads: u32, keys: u32, values: u32, scale: f32, group: u32 };
/// `grouped`: value head h reads Q/K head h / (vheads / qheads) (Fish's
/// `repeat_interleave`) instead of h % qheads (llama.cpp's tiled order).
pub const DeltaShape = struct { qheads: usize, vheads: usize, keys: usize, values: usize, scale: f32, grouped: bool = false };
/// Nontransactional in-place state update: a failed command buffer leaves
/// partially updated state; the owning session must be reset.
pub fn delta(self: *Backend, state: Buffer, qkv: Buffer, decay_log: Buffer, beta: Buffer, output: Buffer, s: DeltaShape) !void {
    if (s.qheads == 0 or s.vheads == 0 or s.keys == 0 or s.values == 0 or s.vheads % s.qheads != 0 or !std.math.isFinite(s.scale) or s.scale <= 0) return error.InvalidShape;
    if (state.len < s.vheads * s.values * s.keys * 4 or qkv.len < (2 * s.qheads * s.keys + s.vheads * s.values) * 4 or decay_log.len < s.vheads * 4 or beta.len < s.vheads * 4 or output.len < s.vheads * s.values * 4) return error.InvalidShape;
    const p: DeltaParams = .{ .qheads = @intCast(s.qheads), .vheads = @intCast(s.vheads), .keys = @intCast(s.keys), .values = @intCast(s.values), .scale = s.scale, .group = if (s.grouped) @intCast(s.vheads / s.qheads) else 0 };
    try self.dispatch(.delta, &.{ state, qkv, decay_log, beta, output }, p, @intCast(s.vheads * s.values), 32, .{});
}
pub const DeltaChunkParams = extern struct { qheads: u32, vheads: u32, keys: u32, values: u32, count: u32, in_stride: u32, gate_stride: u32, out_stride: u32, scale: f32, group: u32 };
/// `count` tokens of one layer: `qkv` rows `[token][in_stride]` laid out
/// as `nu_delta`'s input (q, k, then v per head), `decay_log`/`beta` rows
/// `[token][gate_stride]`, `output` rows `[token][out_stride]` at
/// `head * values`. State `[head][value][key]` is updated in place.
/// `grouped` as in `DeltaShape`.
pub const DeltaChunkShape = struct { qheads: usize, vheads: usize, keys: usize, values: usize, count: usize, in_stride: usize, gate_stride: usize, out_stride: usize, scale: f32, grouped: bool = false };
/// Rows the `qkv` buffer must hold for `count` tokens (the kernel walks
/// 32-token sub-chunks; rows past `count` are masked, never read).
pub fn deltaChunkRows(count: usize) usize {
    return @divCeil(count, 32) * 32;
}
/// Chunkwise DeltaNet: one dispatch per layer for a prefill chunk,
/// threadgroup per (value head, 32 value rows), 32-token sub-chunks.
/// Nontransactional like `delta`: a failed command buffer leaves partial
/// state and the session must be reset.
pub fn deltaChunk(self: *Backend, state: Buffer, qkv: Buffer, decay_log: Buffer, beta: Buffer, output: Buffer, s: DeltaChunkShape) !void {
    if (s.qheads == 0 or s.vheads == 0 or s.keys == 0 or s.values == 0 or s.vheads % s.qheads != 0 or !std.math.isFinite(s.scale) or s.scale <= 0) return error.InvalidShape;
    if (s.keys % 8 != 0 or s.values % 32 != 0 or s.count == 0 or s.count > 4096) return error.InvalidShape;
    const qkv_row = 2 * s.qheads * s.keys + s.vheads * s.values;
    if (s.in_stride < qkv_row or s.gate_stride < s.vheads or s.out_stride < s.vheads * s.values) return error.InvalidShape;
    const rows = deltaChunkRows(s.count);
    if (state.len < s.vheads * s.values * s.keys * 4) return error.InvalidShape;
    if (qkv.len < rows * s.in_stride * 4 or decay_log.len < s.count * s.gate_stride * 4 or beta.len < s.count * s.gate_stride * 4 or output.len < s.count * s.out_stride * 4) return error.InvalidShape;
    const p: DeltaChunkParams = .{ .qheads = @intCast(s.qheads), .vheads = @intCast(s.vheads), .keys = @intCast(s.keys), .values = @intCast(s.values), .count = @intCast(s.count), .in_stride = @intCast(s.in_stride), .gate_stride = @intCast(s.gate_stride), .out_stride = @intCast(s.out_stride), .scale = s.scale, .group = if (s.grouped) @intCast(s.vheads / s.qheads) else 0 };
    try self.dispatch(.delta_chunk, &.{ state, qkv, decay_log, beta, output }, p, @intCast(s.vheads * (s.values / 32)), 128, .{});
}
/// A verify batch's rows for `deltaRows` and the tape `deltaReplay` reads:
/// `count` rows written, `capacity` the rows the tape holds. The other
/// fields are `deltaChunk`'s.
pub const DeltaRowsShape = struct { qheads: usize, vheads: usize, keys: usize, values: usize, count: usize, capacity: usize, in_stride: usize = 0, gate_stride: usize = 0, out_stride: usize = 0, scale: f32 = 1 };
pub const DeltaRowsParams = extern struct { qheads: u32, vheads: u32, keys: u32, values: u32, count: u32, in_stride: u32, gate_stride: u32, out_stride: u32, key_stride: u32, corr_offset: u32, scale: f32 };
pub const DeltaReplayParams = extern struct { qheads: u32, vheads: u32, keys: u32, values: u32, count: u32, key_stride: u32, corr_offset: u32 };
/// Floats a one-layer tape of `capacity` rows takes: per row the
/// normalized keys and the decays, then per row the corrections.
pub fn deltaTapeFloats(s: DeltaRowsShape) usize {
    return s.capacity * (s.qheads * s.keys + s.vheads + s.vheads * s.values);
}
fn deltaTapeShape(s: DeltaRowsShape, tape: Buffer) !struct { key_stride: u32, corr_offset: u32 } {
    if (s.qheads == 0 or s.vheads == 0 or s.keys == 0 or s.values == 0 or s.vheads % s.qheads != 0) return error.InvalidShape;
    if (s.keys % 32 != 0 or s.keys > 128 or s.count > s.capacity or tape.len < deltaTapeFloats(s) * 4) return error.InvalidShape;
    const key_stride = s.qheads * s.keys + s.vheads;
    return .{ .key_stride = @intCast(key_stride), .corr_offset = @intCast(s.capacity * key_stride) };
}
/// Verify rows token by token without writing `state`: `nu_delta`'s
/// arithmetic per row, outputs as `deltaChunk` lays them out, and the
/// rows' tape for `deltaReplay`. One to `capacity` rows.
pub fn deltaRows(self: *Backend, state: Buffer, qkv: Buffer, decay_log: Buffer, beta: Buffer, output: Buffer, tape: Buffer, s: DeltaRowsShape) !void {
    const t = try deltaTapeShape(s, tape);
    if (s.count == 0 or !std.math.isFinite(s.scale) or s.scale <= 0) return error.InvalidShape;
    if (s.in_stride < 2 * s.qheads * s.keys + s.vheads * s.values or s.gate_stride < s.vheads or s.out_stride < s.vheads * s.values) return error.InvalidShape;
    if (state.len < s.vheads * s.values * s.keys * 4 or qkv.len < s.count * s.in_stride * 4 or decay_log.len < s.count * s.gate_stride * 4 or beta.len < s.count * s.gate_stride * 4 or output.len < s.count * s.out_stride * 4) return error.InvalidShape;
    const p: DeltaRowsParams = .{ .qheads = @intCast(s.qheads), .vheads = @intCast(s.vheads), .keys = @intCast(s.keys), .values = @intCast(s.values), .count = @intCast(s.count), .in_stride = @intCast(s.in_stride), .gate_stride = @intCast(s.gate_stride), .out_stride = @intCast(s.out_stride), .key_stride = t.key_stride, .corr_offset = t.corr_offset, .scale = s.scale };
    try self.dispatch(.delta_rows, &.{ state, qkv, decay_log, beta, output, tape }, p, @intCast(s.vheads * s.values), 32, .{});
}
/// Applies a `deltaRows` tape's first `count` rows to `state` in place:
/// the state after `count` decode steps. Nontransactional like `delta`.
pub fn deltaReplay(self: *Backend, state: Buffer, tape: Buffer, s: DeltaRowsShape) !void {
    const t = try deltaTapeShape(s, tape);
    if (s.count == 0 or state.len < s.vheads * s.values * s.keys * 4) return error.InvalidShape;
    const p: DeltaReplayParams = .{ .qheads = @intCast(s.qheads), .vheads = @intCast(s.vheads), .keys = @intCast(s.keys), .values = @intCast(s.values), .count = @intCast(s.count), .key_stride = t.key_stride, .corr_offset = t.corr_offset };
    try self.dispatch(.delta_replay, &.{ state, tape }, p, @intCast(s.vheads * s.values), 32, .{});
}
pub const ConvParams = extern struct { channels: u32, taps: u32 };
pub fn convolution(self: *Backend, history: Buffer, input: Buffer, weights: Buffer, output: Buffer, channels: usize, taps: usize) !void {
    if (channels == 0 or taps < 2 or taps > 32 or history.len < channels * (taps - 1) * 4 or input.len < channels * 4 or weights.len < channels * taps * 4 or output.len < channels * 4) return error.InvalidShape;
    try self.dispatch(.convolution, &.{ history, input, weights, output }, ConvParams{ .channels = @intCast(channels), .taps = @intCast(taps) }, perElement(channels), 256, .{});
}
pub const ConvRowsParams = extern struct { channels: u32, taps: u32, rows: u32, stride: u32 };
/// Causal convolution over `rows` token rows (stride `stride`) reading the
/// pre-chunk inputs from `history`; bit-identical to `rows` sequential
/// `convolution` calls. Does not touch `history`: call
/// `convolutionHistory` afterwards.
pub fn convolutionRows(self: *Backend, history: Buffer, input: Buffer, weights: Buffer, output: Buffer, channels: usize, taps: usize, rows: usize, stride: usize) !void {
    if (channels == 0 or taps < 2 or taps > 32 or rows == 0 or stride < channels) return error.InvalidShape;
    if (history.len < channels * (taps - 1) * 4 or input.len < ((rows - 1) * stride + channels) * 4 or weights.len < channels * taps * 4 or output.len < ((rows - 1) * stride + channels) * 4) return error.InvalidShape;
    const p: ConvRowsParams = .{ .channels = @intCast(channels), .taps = @intCast(taps), .rows = @intCast(rows), .stride = @intCast(stride) };
    try self.dispatch(.convolution_rows, &.{ history, input, weights, output }, p, perElement(rows * channels), 256, .{});
}
/// Shifts the last `taps - 1` inputs of the chunk into `history`.
pub fn convolutionHistory(self: *Backend, history: Buffer, input: Buffer, channels: usize, taps: usize, rows: usize, stride: usize) !void {
    if (channels == 0 or taps < 2 or taps > 32 or rows == 0 or stride < channels) return error.InvalidShape;
    if (history.len < channels * (taps - 1) * 4 or input.len < ((rows - 1) * stride + channels) * 4) return error.InvalidShape;
    const p: ConvRowsParams = .{ .channels = @intCast(channels), .taps = @intCast(taps), .rows = @intCast(rows), .stride = @intCast(stride) };
    try self.dispatch(.convolution_history, &.{ history, input }, p, perElement(channels), 256, .{});
}
