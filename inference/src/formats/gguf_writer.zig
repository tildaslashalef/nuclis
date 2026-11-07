//! GGUF v3 writer for generated test models (`models/tiny.zig`): the inverse
//! of `gguf.zig`'s reader, from the same wire contract. Test support, never
//! reached from the application. Everything is built in one owned image; a
//! tiny model's file is a few megabytes, not a checkpoint.
const std = @import("std");
const gguf = @import("gguf.zig");
const encoding = @import("../tensor/encoding.zig");
const Allocator = std.mem.Allocator;

pub const Value = union(enum) {
    uint32: u32,
    float32: f32,
    boolean: bool,
    string: []const u8,
    int32s: []const i32,
    booleans: []const bool,
    strings: []const []const u8,
};

pub const Entry = struct { key: []const u8, value: Value };

/// `dimensions` is in GGUF order: dimension 0 is the contiguous row.
pub const Tensor = struct { name: []const u8, dimensions: []const u64, encoding: u32 };

const DirectoryError = error{ InvalidShape, Overflow, UnsupportedTensorType };
pub const Error = Allocator.Error || DirectoryError;

const alignment = 32;

fn writeString(w: *std.Io.Writer, text: []const u8) std.Io.Writer.Error!void {
    try w.writeInt(u64, text.len, .little);
    try w.writeAll(text);
}

fn writeTag(w: *std.Io.Writer, tag: gguf.Type) std.Io.Writer.Error!void {
    try w.writeInt(u32, @backingInt(tag), .little);
}

fn writeArrayHeader(w: *std.Io.Writer, element: gguf.Type, count: usize) std.Io.Writer.Error!void {
    try writeTag(w, .array);
    try writeTag(w, element);
    try w.writeInt(u64, count, .little);
}

fn writeValue(w: *std.Io.Writer, value: Value) std.Io.Writer.Error!void {
    switch (value) {
        .uint32 => |n| {
            try writeTag(w, .uint32);
            try w.writeInt(u32, n, .little);
        },
        .float32 => |x| {
            try writeTag(w, .float32);
            try w.writeInt(u32, @bitCast(x), .little);
        },
        .boolean => |b| {
            try writeTag(w, .boolean);
            try w.writeByte(@intFromBool(b));
        },
        .string => |s| {
            try writeTag(w, .string);
            try writeString(w, s);
        },
        .int32s => |values| {
            try writeArrayHeader(w, .int32, values.len);
            for (values) |n| try w.writeInt(i32, n, .little);
        },
        .booleans => |values| {
            try writeArrayHeader(w, .boolean, values.len);
            for (values) |b| try w.writeByte(@intFromBool(b));
        },
        .strings => |values| {
            try writeArrayHeader(w, .string, values.len);
            for (values) |s| try writeString(w, s);
        },
    }
}

/// The header, metadata, and tensor descriptors; returns the data section's
/// size. Offsets are relative to the data section, each 32-aligned.
fn writeDirectory(w: *std.Io.Writer, metadata: []const Entry, tensors: []const Tensor, sizes: []u64) (std.Io.Writer.Error || DirectoryError)!u64 {
    try w.writeAll("GGUF");
    try w.writeInt(u32, 3, .little);
    try w.writeInt(u64, tensors.len, .little);
    try w.writeInt(u64, metadata.len, .little);
    for (metadata) |entry| {
        try writeString(w, entry.key);
        try writeValue(w, entry.value);
    }
    var offset: u64 = 0;
    for (tensors, sizes) |tensor, *size| {
        const layout = encoding.layout(tensor.encoding) orelse return error.UnsupportedTensorType;
        size.* = try layout.byteCount(tensor.dimensions);
        try writeString(w, tensor.name);
        try w.writeInt(u32, @intCast(tensor.dimensions.len), .little);
        for (tensor.dimensions) |dimension| try w.writeInt(u64, dimension, .little);
        try w.writeInt(u32, tensor.encoding, .little);
        try w.writeInt(u64, offset, .little);
        offset = std.mem.alignForward(u64, offset + size.*, alignment);
    }
    return offset;
}

/// The complete file for `metadata` and `tensors`, tensor data in directory
/// order at the default 32-byte alignment. `fill` writes one tensor's bytes,
/// `index` into `tensors`; the region arrives zeroed. Caller owns the image.
pub fn image(
    gpa: Allocator,
    metadata: []const Entry,
    tensors: []const Tensor,
    context: anytype,
    comptime fill: fn (@TypeOf(context), index: usize, bytes: []u8) void,
) Error![]u8 {
    const sizes = try gpa.alloc(u64, tensors.len);
    defer gpa.free(sizes);
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    // The allocating writer's only failure is its allocator's.
    const data_bytes = writeDirectory(&out.writer, metadata, tensors, sizes) catch |err| switch (err) {
        error.WriteFailed => return error.OutOfMemory,
        else => |e| return e,
    };
    const head = out.written();
    const data_offset = std.mem.alignForward(usize, head.len, alignment);
    const bytes = try gpa.alloc(u8, data_offset + @as(usize, @intCast(data_bytes)));
    errdefer gpa.free(bytes);
    @memcpy(bytes[0..head.len], head);
    @memset(bytes[head.len..], 0);
    var at: usize = data_offset;
    for (sizes, 0..) |size, index| {
        const end = at + @as(usize, @intCast(size));
        fill(context, index, bytes[at..end]);
        at = data_offset + std.mem.alignForward(usize, end - data_offset, alignment);
    }
    return bytes;
}

/// The directory alone, for a check of the parser and validators that needs
/// no weights: `gguf.parse` it with the returned `file_bytes`. Caller owns `bytes`.
pub fn directory(gpa: Allocator, metadata: []const Entry, tensors: []const Tensor) Error!struct { bytes: []u8, file_bytes: u64 } {
    const sizes = try gpa.alloc(u64, tensors.len);
    defer gpa.free(sizes);
    var out: std.Io.Writer.Allocating = .init(gpa);
    defer out.deinit();
    const data_bytes = writeDirectory(&out.writer, metadata, tensors, sizes) catch |err| switch (err) {
        error.WriteFailed => return error.OutOfMemory,
        else => |e| return e,
    };
    const data_offset = std.mem.alignForward(u64, out.written().len, alignment);
    return .{ .bytes = try gpa.dupe(u8, out.written()), .file_bytes = data_offset + data_bytes };
}

fn parsed(bytes: []const u8) !gguf.Document {
    var reader: std.Io.Reader = .fixed(bytes);
    return gguf.parse(std.testing.allocator, &reader, bytes.len, .{});
}

fn counting(_: void, index: usize, bytes: []u8) void {
    @memset(bytes, @intCast(index + 1));
}

test "an image reads back through the strict parser, values and spans intact" {
    const gpa = std.testing.allocator;
    const bytes = try image(gpa, &.{
        .{ .key = "general.architecture", .value = .{ .string = "x" } },
        .{ .key = "n", .value = .{ .uint32 = 7 } },
        .{ .key = "x", .value = .{ .float32 = 0.5 } },
        .{ .key = "b", .value = .{ .boolean = true } },
        .{ .key = "i", .value = .{ .int32s = &.{ -1, 2 } } },
        .{ .key = "f", .value = .{ .booleans = &.{ true, false } } },
        .{ .key = "s", .value = .{ .strings = &.{ "ab", "" } } },
    }, &.{
        .{ .name = "a", .dimensions = &.{ 3, 2 }, .encoding = 0 },
        .{ .name = "b", .dimensions = &.{ 256, 2 }, .encoding = 12 },
        .{ .name = "c", .dimensions = &.{32}, .encoding = 8 },
    }, {}, counting);
    defer gpa.free(bytes);
    var doc = try parsed(bytes);
    defer doc.deinit();
    try std.testing.expectEqualStrings("x", doc.string("general.architecture").?);
    try std.testing.expectEqual(@as(u64, 7), doc.get("n").?.unsigned);
    try std.testing.expectEqual(@as(f64, 0.5), doc.get("x").?.float);
    try std.testing.expect(doc.get("b").?.boolean);
    const ints = doc.get("i").?.array;
    try std.testing.expectEqual(@as(i64, -1), ints.values.?[0].signed);
    try std.testing.expect(!doc.get("f").?.array.values.?[1].boolean);
    var strings: gguf.ArrayReader = try .init(doc.get("s").?.array, bytes[0..@intCast(doc.directory_bytes)], .{});
    try std.testing.expectEqualStrings("ab", (try strings.nextString()).?);
    try std.testing.expectEqualStrings("", (try strings.nextString()).?);
    try std.testing.expectEqual(@as(usize, 3), doc.tensors.len);
    try std.testing.expectEqual(@as(u64, 288), doc.tensors[1].bytes);
    for (doc.tensors, [_]u8{ 1, 2, 3 }) |tensor, mark| {
        const start: usize = @intCast(doc.data_offset + tensor.offset);
        try std.testing.expectEqual(@as(u64, 0), tensor.offset % 32);
        for (bytes[start..][0..@intCast(tensor.bytes)]) |byte| try std.testing.expectEqual(mark, byte);
    }
}

test "a directory parses to the same descriptors as its image" {
    const gpa = std.testing.allocator;
    const metadata = [_]Entry{.{ .key = "k", .value = .{ .uint32 = 1 } }};
    const tensors = [_]Tensor{ .{ .name = "a", .dimensions = &.{ 3, 2 }, .encoding = 0 }, .{ .name = "b", .dimensions = &.{ 256, 2 }, .encoding = 12 } };
    const whole = try image(gpa, &metadata, &tensors, {}, counting);
    defer gpa.free(whole);
    const head = try directory(gpa, &metadata, &tensors);
    defer gpa.free(head.bytes);
    try std.testing.expectEqual(@as(u64, whole.len), head.file_bytes);
    try std.testing.expectEqualSlices(u8, head.bytes, whole[0..head.bytes.len]);
    var reader: std.Io.Reader = .fixed(head.bytes);
    var doc = try gguf.parse(gpa, &reader, head.file_bytes, .{});
    defer doc.deinit();
    try std.testing.expectEqual(@as(u64, 288), doc.tensors[1].bytes);
}

test "an image with no tensors and a refused layout" {
    const gpa = std.testing.allocator;
    const bytes = try image(gpa, &.{.{ .key = "k", .value = .{ .uint32 = 1 } }}, &.{}, {}, counting);
    defer gpa.free(bytes);
    var doc = try parsed(bytes);
    defer doc.deinit();
    try std.testing.expectEqual(@as(usize, 0), doc.tensors.len);
    try std.testing.expectError(error.UnsupportedTensorType, image(gpa, &.{}, &.{.{ .name = "a", .dimensions = &.{4}, .encoding = 999 }}, {}, counting));
    try std.testing.expectError(error.InvalidShape, image(gpa, &.{}, &.{.{ .name = "a", .dimensions = &.{128}, .encoding = 12 }}, {}, counting));
}

test "an image survives allocation failure without leaks" {
    const alloc_check = @import("../alloc_check.zig");
    try alloc_check.checkAll(std.testing.allocator, struct {
        fn check(alloc: Allocator) !void {
            const bytes = try image(alloc, &.{.{ .key = "k", .value = .{ .string = "v" } }}, &.{.{ .name = "a", .dimensions = &.{ 4, 4 }, .encoding = 0 }}, {}, counting);
            alloc.free(bytes);
        }
    }.check, .{});
}
