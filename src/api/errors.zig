//! The API's error body, shared by every service and the transport:
//! `{"error": {"code", "message"}}` with the HTTP status that fits. Codes are
//! stable identifiers clients branch on; messages are for people.
//! docs/reference/api.md § Errors.
const std = @import("std");

pub const ApiError = struct {
    status: std.http.Status,
    code: []const u8,
    message: []const u8,

    pub fn init(status: std.http.Status, code: []const u8, message: []const u8) ApiError {
        return .{ .status = status, .code = code, .message = message };
    }
};

/// The JSON body of `err`, in `arena`. A body that cannot be allocated is a
/// fixed one, so an error always has a body.
pub fn body(arena: std.mem.Allocator, err: ApiError) []const u8 {
    var out: std.Io.Writer.Allocating = .init(arena);
    write(&out.writer, err) catch return "{\"error\":{\"code\":\"internal\",\"message\":\"out of memory\"}}\n";
    return out.written();
}

pub fn write(out: *std.Io.Writer, err: ApiError) std.Io.Writer.Error!void {
    var s: std.json.Stringify = .{ .writer = out };
    try s.beginObject();
    try s.objectField("error");
    try s.beginObject();
    try s.objectField("code");
    try s.write(err.code);
    try s.objectField("message");
    try s.write(err.message);
    try s.endObject();
    try s.endObject();
    try out.writeByte('\n');
}

test "the error body escapes its message" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const text = body(arena_state.allocator(), .init(.bad_request, "invalid_json", "a \"quote\""));
    try std.testing.expectEqualStrings("{\"error\":{\"code\":\"invalid_json\",\"message\":\"a \\\"quote\\\"\"}}\n", text);
}
