//! Read-only mapped GGUF storage. Tensor views borrow the mapping; mappings and
//! parsed documents must outlive execution. The file must not change while open.
const std = @import("std");
const gguf = @import("../formats/gguf.zig");
const cpu = @import("../backends/cpu/root.zig");
const quant = @import("../quant/decode.zig");

pub const View = struct {
    file: []const u8,
    data_offset: u64,
    pub fn bytes(self: View, tensor: *const gguf.Tensor) ![]const u8 {
        const start = try std.math.add(u64, self.data_offset, tensor.offset);
        if (start > self.file.len or tensor.bytes > self.file.len - start) return error.TensorOutOfBounds;
        return self.file[@intCast(start)..][0..@intCast(tensor.bytes)];
    }
    pub fn matrix(self: View, tensor: *const gguf.Tensor) !cpu.Matrix {
        if (tensor.dimensions.len != 2) return error.InvalidShape;
        return .{ .columns = std.math.cast(usize, tensor.dimensions[0]) orelse return error.Overflow, .rows = std.math.cast(usize, tensor.dimensions[1]) orelse return error.Overflow, .encoding = tensor.encoding_id, .bytes = try self.bytes(tensor) };
    }
    /// A 3-D tensor `[experts][rows][columns]` (GGUF dimensions
    /// `[columns, rows, experts]`) as contiguous expert matrices.
    pub fn expertMatrix(self: View, tensor: *const gguf.Tensor) !cpu.ExpertMatrix {
        if (tensor.dimensions.len != 3) return error.InvalidShape;
        return .{
            .columns = std.math.cast(usize, tensor.dimensions[0]) orelse return error.Overflow,
            .rows = std.math.cast(usize, tensor.dimensions[1]) orelse return error.Overflow,
            .experts = std.math.cast(usize, tensor.dimensions[2]) orelse return error.Overflow,
            .encoding = tensor.encoding_id,
            .bytes = try self.bytes(tensor),
        };
    }
    pub fn row(self: View, tensor: *const gguf.Tensor, index: usize, output: []f32) !void {
        const m = try self.matrix(tensor);
        if (index >= m.rows or output.len != m.columns or m.rows == 0 or m.bytes.len % m.rows != 0) return error.InvalidShape;
        const width = m.bytes.len / m.rows;
        try quant.row(m.encoding, m.bytes[index * width ..][0..width], output);
    }
    /// Borrowed F32 storage, read explicitly as little endian to avoid alignment
    /// and host-endian assumptions. Used for small weights, not matrix decode.
    pub fn scalar(self: View, tensor: *const gguf.Tensor, index: usize) !f32 {
        if (tensor.encoding_id != 0 or index >= tensor.elements) return error.InvalidShape;
        const raw = try self.bytes(tensor);
        if (index >= raw.len / 4) return error.TensorOutOfBounds;
        return @bitCast(std.mem.readInt(u32, raw[index * 4 ..][0..4], .little));
    }
    /// Copies an entire F32 tensor into caller-owned memory, so hot loops read
    /// aligned host values instead of re-decoding the mapping per element.
    pub fn vector(self: View, alloc: std.mem.Allocator, tensor: *const gguf.Tensor) ![]f32 {
        if (tensor.encoding_id != 0) return error.InvalidShape;
        const raw = try self.bytes(tensor);
        if (raw.len != tensor.elements * 4) return error.InvalidShape;
        const out = try alloc.alloc(f32, @intCast(tensor.elements));
        for (out, 0..) |*x, i| x.* = @bitCast(std.mem.readInt(u32, raw[i * 4 ..][0..4], .little));
        return out;
    }
};

pub const Mapped = struct {
    file: std.Io.File,
    mapping: std.Io.File.MemoryMap,
    document: gguf.Document,

    pub fn open(alloc: std.mem.Allocator, io: std.Io, path: []const u8) !Mapped {
        const file = try std.Io.Dir.cwd().openFile(io, path, .{});
        errdefer file.close(io);
        const stat = try file.stat(io);
        if (stat.kind != .file) return error.NotRegularFile;
        if (stat.size == 0 or stat.size > 128 * @as(u64, 1024 * 1024 * 1024)) return error.LimitExceeded;
        var buffer: [64 * 1024]u8 = undefined;
        var reader = file.reader(io, &buffer);
        var doc = try gguf.parse(alloc, &reader.interface, stat.size, .{});
        errdefer doc.deinit();
        const map = try file.createMemoryMap(io, .{ .len = std.math.cast(usize, stat.size) orelse return error.LimitExceeded, .protection = .{ .read = true, .write = false }, .populate = false });
        return .{ .file = file, .mapping = map, .document = doc };
    }
    pub fn view(self: *const Mapped) View {
        return .{ .file = self.mapping.memory, .data_offset = self.document.data_offset };
    }
    pub fn deinit(self: *Mapped, io: std.Io) void {
        self.mapping.destroy(io);
        self.document.deinit();
        self.file.close(io);
        self.* = undefined;
    }
};

test "weight views validate offsets and decode selected rows" {
    const raw = "xxxx\x00\x00\x80\x3f\x00\x00\x00\x40";
    const view_: View = .{ .file = raw, .data_offset = 4 };
    var tensor: gguf.Tensor = .{ .name = "test", .dimensions = &.{ 1, 2 }, .encoding_id = 0, .offset = 0, .bytes = 8, .elements = 2 };
    var output: [1]f32 = undefined;
    try view_.row(&tensor, 1, &output);
    try std.testing.expectEqual(@as(f32, 2), output[0]);
    try std.testing.expectEqual(@as(f32, 1), try view_.scalar(&tensor, 0));
    const all = try view_.vector(std.testing.allocator, &tensor);
    defer std.testing.allocator.free(all);
    try std.testing.expectEqualSlices(f32, &.{ 1, 2 }, all);
    try std.testing.expectError(error.InvalidShape, view_.row(&tensor, 2, &output));
    tensor.offset = 1;
    try std.testing.expectError(error.TensorOutOfBounds, view_.bytes(&tensor));
    tensor.offset = std.math.maxInt(u64);
    try std.testing.expectError(error.Overflow, view_.bytes(&tensor));
}
