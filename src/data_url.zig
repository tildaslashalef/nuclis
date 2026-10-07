//! An image sent inline in a JSON request: a base64 data URL
//! (`data:image/png;base64,…`) or bare base64. Shared by the decision and
//! chat wire formats; neither fetches a URL.
const std = @import("std");

/// The base64 text of `text`: after the comma of a data URL, or all of it.
/// Null for a data URL that is not base64.
pub fn payload(text: []const u8) ?[]const u8 {
    if (!std.mem.startsWith(u8, text, "data:")) return text;
    const comma = std.mem.indexOfScalar(u8, text, ',') orelse return null;
    if (!std.mem.endsWith(u8, text[0..comma], ";base64")) return null;
    return text[comma + 1 ..];
}

/// The bytes `encoded` (base64) stands for, in `arena`; at most `max_bytes`.
pub fn decode(arena: std.mem.Allocator, encoded: []const u8, max_bytes: usize) error{ NotBase64, TooLarge, OutOfMemory }![]u8 {
    const decoder = std.base64.standard.Decoder;
    const size = decoder.calcSizeForSlice(encoded) catch return error.NotBase64;
    if (size > max_bytes) return error.TooLarge;
    const decoded = try arena.alloc(u8, size);
    decoder.decode(decoded, encoded) catch return error.NotBase64;
    return decoded;
}

test "a base64 data URL or bare base64; another data URL is refused" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    try std.testing.expectEqualStrings("aGk=", payload("data:image/png;base64,aGk=").?);
    try std.testing.expectEqualStrings("aGk=", payload("aGk=").?);
    try std.testing.expect(payload("data:image/png,raw") == null);
    try std.testing.expect(payload("data:image/png;base64") == null);
    try std.testing.expectEqualStrings("hi", try decode(arena, "aGk=", 2));
    try std.testing.expectError(error.TooLarge, decode(arena, "aGk=", 1));
    try std.testing.expectError(error.NotBase64, decode(arena, "a!k=", 8));
}
