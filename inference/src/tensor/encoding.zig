//! GGML storage layouts, independent of any model architecture.
//! These describe on-disk block sizes, not available numerical kernels.
//! Source: ggml-org/llama.cpp, ggml/src/ggml-common.h (see docs/gguf-inspection.md).
const std = @import("std");

pub const Layout = struct {
    name: []const u8,
    elements_per_block: u32,
    bytes_per_block: u32,

    /// GGUF dimension 0 is the contiguous row. Every row must contain complete
    /// quantization blocks; divisibility of the entire tensor is not sufficient.
    pub fn byteCount(self: Layout, dimensions: []const u64) error{ InvalidShape, Overflow }!u64 {
        if (dimensions.len == 0 or dimensions[0] == 0 or dimensions[0] % self.elements_per_block != 0)
            return error.InvalidShape;
        var bytes = try std.math.mul(u64, dimensions[0] / self.elements_per_block, self.bytes_per_block);
        for (dimensions[1..]) |dim| {
            if (dim == 0) return error.InvalidShape;
            bytes = try std.math.mul(u64, bytes, dim);
        }
        return bytes;
    }
};

/// Only layouts that we have explicitly checked are recognized. Unknown IDs
/// cannot be sized safely and must be rejected before exposing tensor spans.
pub fn layout(id: u32) ?Layout {
    return switch (id) {
        0 => .{ .name = "F32", .elements_per_block = 1, .bytes_per_block = 4 },
        1 => .{ .name = "F16", .elements_per_block = 1, .bytes_per_block = 2 },
        2 => .{ .name = "Q4_0", .elements_per_block = 32, .bytes_per_block = 18 },
        8 => .{ .name = "Q8_0", .elements_per_block = 32, .bytes_per_block = 34 },
        11 => .{ .name = "Q3_K", .elements_per_block = 256, .bytes_per_block = 110 },
        12 => .{ .name = "Q4_K", .elements_per_block = 256, .bytes_per_block = 144 },
        13 => .{ .name = "Q5_K", .elements_per_block = 256, .bytes_per_block = 176 },
        14 => .{ .name = "Q6_K", .elements_per_block = 256, .bytes_per_block = 210 },
        20 => .{ .name = "IQ4_NL", .elements_per_block = 32, .bytes_per_block = 18 },
        21 => .{ .name = "IQ3_S", .elements_per_block = 256, .bytes_per_block = 110 },
        23 => .{ .name = "IQ4_XS", .elements_per_block = 256, .bytes_per_block = 136 },
        30 => .{ .name = "BF16", .elements_per_block = 1, .bytes_per_block = 2 },
        else => null,
    };
}

test "quantized rows must contain whole blocks" {
    try std.testing.expectEqual(@as(u64, 288), try layout(12).?.byteCount(&.{ 256, 2 }));
    // 128 * 2 is divisible by 256, but neither individual row is.
    try std.testing.expectError(error.InvalidShape, layout(12).?.byteCount(&.{ 128, 2 }));
    try std.testing.expectEqual(@as(u64, 18), try layout(20).?.byteCount(&.{32}));
    try std.testing.expectError(error.InvalidShape, layout(0).?.byteCount(&.{ 4, 0 }));
    try std.testing.expectError(error.Overflow, layout(0).?.byteCount(&.{std.math.maxInt(u64)}));
    try std.testing.expect(layout(999) == null);
}
