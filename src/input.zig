//! Bounded reads of what a command names on its line: a file, or standard
//! input for `-`. The bound is the caller's; a failure names the source in
//! the diagnostic.
const std = @import("std");
const config = @import("config.zig");

/// Reads a whole file, or standard input for `-`, up to `limit` bytes.
pub fn readBounded(arena: std.mem.Allocator, io: std.Io, path: []const u8, limit: usize, diag: *config.Diagnostic) ![]u8 {
    if (std.mem.eql(u8, path, "-")) {
        var buffer: [4096]u8 = undefined;
        var reader = std.Io.File.stdin().readerStreaming(io, &buffer);
        return reader.interface.allocRemaining(arena, .limited(limit)) catch |err| {
            diag.set("standard input: {s} (at most {d} bytes)", .{ @errorName(err), limit });
            return err;
        };
    }
    return std.Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(limit)) catch |err| {
        diag.set("{s}: {s}", .{ path, @errorName(err) });
        return err;
    };
}
