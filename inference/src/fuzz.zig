//! Test support for fuzz tests of parsers that read outside bytes
//! (docs/guide/code.md § Hostile input). A test body draws one byte string
//! from the `Smith` with `bytes`; its seed corpus is plain inputs encoded by
//! `corpus`, which `zig build test` runs as ordinary cases and `make fuzz`
//! hands the fuzzer as a starting point. Never reached from the application.
const std = @import("std");

pub const Smith = std.testing.Smith;

/// The largest input a fuzz test draws: enough for every header and frame
/// the parsers bound, small enough to keep a corpus run fast.
pub const max_input = 64 * 1024;

/// One byte string from the fuzzer (or a corpus entry), in `buf`.
pub fn bytes(smith: *Smith, buf: []u8) []u8 {
    const n = smith.sliceWithHash(buf, 0x6e75636c);
    return buf[0..n];
}

/// A second, independent draw, for a test that needs two strings.
pub fn moreBytes(smith: *Smith, buf: []u8) []u8 {
    const n = smith.sliceWithHash(buf, 0x69733021);
    return buf[0..n];
}

/// `inputs` as corpus entries: each a little-endian `u32` length and the
/// bytes, the form `bytes` reads back.
pub fn corpus(comptime inputs: []const []const u8) []const []const u8 {
    return comptime blk: {
        var out: [inputs.len][]const u8 = undefined;
        for (inputs, &out) |input, *entry| {
            entry.* = std.mem.toBytes(std.mem.nativeToLittle(u32, input.len)) ++ input;
        }
        const final = out;
        break :blk &final;
    };
}

test "a corpus entry reads back as the input it encodes" {
    const Case = struct {
        fn one(_: void, smith: *Smith) !void {
            var buf: [64]u8 = undefined;
            const got = bytes(smith, &buf);
            try std.testing.expect(got.len == 0 or std.mem.eql(u8, got, "hello"));
        }
    };
    try std.testing.fuzz({}, Case.one, .{ .corpus = corpus(&.{"hello"}) });
}
