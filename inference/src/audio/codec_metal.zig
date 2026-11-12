//! The helper layer the codec decoders on Metal share (fish_codec_metal.zig,
//! qwen3_codec_metal.zig): weights copied as F32 matrices with their
//! columns padded to the matmul tile's 64, and the recording of a causal or
//! transposed convolution through its carried history as a gather and one
//! matrix product. `Device` owns every buffer it creates and borrows its
//! `Backend`, which must outlive it.
const std = @import("std");
const metal = @import("../backends/metal/root.zig");
const cpu = @import("../backends/cpu/root.zig");

const Allocator = std.mem.Allocator;
const Backend = metal.Backend;
const Buffer = metal.Buffer;

/// A matrix in a created buffer, `columns` padded to a multiple of 64.
pub const Mat = struct { buffer: Buffer, matrix: cpu.Matrix, bias: ?Buffer };
/// A causal convolution (`taps` 1 with `dilation` 1 is a plain matrix).
pub const Conv = struct { mat: Mat, in: usize, taps: usize, dilation: usize };
/// A transposed convolution: `stride` output rows per input row.
pub const Trans = struct { mat: Mat, in: usize, out: usize, stride: usize, taps: usize };
pub const Snake = struct { scale: Buffer, inv: Buffer };

pub fn padded(columns: usize) usize {
    return (columns + 63) / 64 * 64;
}
pub fn tile(rows: usize) usize {
    return Backend.matmulPadded(rows);
}

/// A decoder's buffers on the device and the convolution recording over
/// them. `gathered` and `joined` are the convolutions' scratch, which the
/// decoder sizes and creates (`floats`) before recording any.
pub const Device = struct {
    gpa: Allocator,
    backend: *Backend,
    buffers: std.ArrayList(Buffer) = .empty,
    /// Bytes of device memory created.
    bytes: usize = 0,
    gathered: Buffer = undefined,
    joined: Buffer = undefined,

    /// Releases every buffer `floats` created.
    pub fn releaseAll(self: *Device) void {
        for (self.buffers.items) |b| self.backend.release(b) catch {};
        self.buffers.deinit(self.gpa);
    }

    pub fn floats(self: *Device, count: usize) !Buffer {
        try self.buffers.ensureUnusedCapacity(self.gpa, 1);
        const b = try self.backend.create(count * 4);
        self.buffers.appendAssumeCapacity(b);
        self.bytes += count * 4;
        return b;
    }
    pub fn upload(self: *Device, values: []const f32) !Buffer {
        const b = try self.floats(values.len);
        @memcpy(b.floats()[0..values.len], values);
        return b;
    }
    /// `w` is `[rows][columns]`, copied with the columns padded to 64 (the
    /// buffer is created zeroed).
    pub fn linear(self: *Device, w: []const f32, bias: ?[]const f32, rows: usize, columns: usize) !Mat {
        const stride = padded(columns);
        const b = try self.floats(rows * stride);
        const dst = b.floats();
        for (0..rows) |r| @memcpy(dst[r * stride ..][0..columns], w[r * columns ..][0..columns]);
        return .{
            .buffer = b,
            .matrix = .{ .encoding = 0, .rows = rows, .columns = stride, .bytes = std.mem.sliceAsBytes(dst[0 .. rows * stride]) },
            .bias = if (bias) |v| try self.upload(v) else null,
        };
    }

    /// `output = m · input (+ bias)` over `rows` rows of `in_stride`.
    pub fn mm(self: *Device, m: Mat, input: Buffer, in_stride: usize, output: Buffer, rows: usize) !void {
        const b = self.backend;
        try b.matmulTile(m.buffer, m.matrix, input, in_stride, output, m.matrix.rows, rows);
        if (m.bias) |bias| try b.addBiasRows(output, bias, m.matrix.rows, rows, m.matrix.rows);
    }

    /// `joined = history ‖ x` (`history` has `history_rows` rows of `width`).
    pub fn join(self: *Device, history: ?Buffer, history_rows: usize, x: Buffer, rows: usize, width: usize) !void {
        const b = self.backend;
        if (history_rows > 0) try b.copy(self.joined, history.?, history_rows * width);
        try b.copy(self.joined.slice(history_rows * width * 4, rows * width * 4), x, rows * width);
    }
    /// `history = the last rows of joined` after a `join` of `rows` new rows.
    pub fn keep(self: *Device, history: ?Buffer, history_rows: usize, rows: usize, width: usize) !void {
        if (history_rows > 0) try self.backend.copy(history.?, self.joined.slice(rows * width * 4, history_rows * width * 4), history_rows * width);
    }

    /// A causal convolution of `rows` rows of `x` into `y` through `history`.
    pub fn convolve(self: *Device, conv: Conv, history: ?Buffer, x: Buffer, rows: usize, y: Buffer) !void {
        const h = (conv.taps - 1) * conv.dilation;
        const columns = conv.mat.matrix.columns;
        if (h == 0 and columns == conv.in) return self.mm(conv.mat, x, conv.in, y, rows);
        var source = x;
        if (h > 0) {
            try self.join(history, h, x, rows, conv.in);
            source = self.joined;
        }
        try self.backend.ttsGather(source, self.gathered, rows, conv.in, conv.taps, columns, 0, @intCast(conv.dilation));
        try self.mm(conv.mat, self.gathered, columns, y, rows);
        try self.keep(history, h, rows, conv.in);
    }

    /// A transposed convolution of `rows` input rows into `rows · stride` rows of `y`.
    pub fn transpose(self: *Device, t: Trans, history: ?Buffer, x: Buffer, rows: usize, y: Buffer) !void {
        const h = t.taps - 1;
        if (h == 0 and t.mat.matrix.columns == t.in) return self.mm(t.mat, x, t.in, y, rows);
        try self.join(history, h, x, rows, t.in);
        try self.backend.ttsGather(self.joined, self.gathered, rows, t.in, t.taps, t.mat.matrix.columns, h, -1);
        try self.mm(t.mat, self.gathered, t.mat.matrix.columns, y, rows);
        try self.keep(history, h, rows, t.in);
    }
};
