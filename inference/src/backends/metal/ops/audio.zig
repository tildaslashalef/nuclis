//! Gemma 4's audio encoder kernels: the host side of `kernels/audio.metal`. Methods of `Backend`,
//! which `root.zig` aliases so callers write `backend.audioAttention(…)`.
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

pub const AudioAttentionParams = extern struct { rows: u32, width: u32, key_scale: f32, cap: f32 };
/// Gemma 4's chunked audio attention over `rows` rows of 8 heads of 128
/// (`gemma4a.attention`): q scaled by `qscale` (128 values), k by
/// `key_scale`, each query seeing the 11 rows before it and itself.
pub fn audioAttention(self: *Backend, q: Buffer, k: Buffer, v: Buffer, relk: Buffer, qscale: Buffer, output: Buffer, rows: usize, key_scale: f32, cap: f32) !void {
    const width = 1024;
    if (rows == 0 or q.len < rows * width * 4 or k.len < rows * width * 4 or v.len < rows * width * 4 or output.len < rows * width * 4 or relk.len < 13 * width * 4 or qscale.len < 128 * 4) return error.InvalidShape;
    try self.dispatch(.audio_attention, &.{ q, k, v, relk, qscale, output }, AudioAttentionParams{ .rows = @intCast(rows), .width = width, .key_scale = key_scale, .cap = cap }, @intCast(rows * 8), 32, .{});
}
pub const GluConvParams = extern struct { rows: u32, width: u32 };
/// `out[t][c] = Σ_tap dw[c][tap] · glu(start[t − 4 + tap])[c]` over
/// `rows` rows of `width`; `start` rows are `2·width` (values, gates).
pub fn gluConv(self: *Backend, start: Buffer, dw: Buffer, output: Buffer, rows: usize, width: usize) !void {
    if (rows == 0 or start.len < rows * 2 * width * 4 or dw.len < width * 5 * 4 or output.len < rows * width * 4) return error.InvalidShape;
    try self.dispatch(.glu_conv, &.{ start, dw, output }, GluConvParams{ .rows = @intCast(rows), .width = @intCast(width) }, perElement(rows * width), 256, .{});
}
