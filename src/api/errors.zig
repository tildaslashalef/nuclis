//! The API's error body, shared by every service and the transport:
//! `{"error": {"code", "message", "type", "param"}}` with the HTTP status that
//! fits. Codes are stable identifiers clients branch on; messages are for
//! people; `type` and `param` are the fields OpenAI's SDKs read.
//! docs/guide/api.md § Errors.
const std = @import("std");

pub const ApiError = struct {
    status: std.http.Status,
    code: []const u8,
    message: []const u8,
    /// The request field at fault, when one is.
    param: ?[]const u8 = null,

    pub fn init(status: std.http.Status, code: []const u8, message: []const u8) ApiError {
        return .{ .status = status, .code = code, .message = message };
    }

    /// OpenAI's error class: the client's request or the server.
    pub fn kind(self: ApiError) []const u8 {
        return if (@backingInt(self.status) < 500) "invalid_request_error" else "server_error";
    }
};

/// The JSON body of `err`, in `arena`. A body that cannot be allocated is a
/// fixed one, so an error always has a body.
pub fn body(arena: std.mem.Allocator, err: ApiError) []const u8 {
    var out: std.Io.Writer.Allocating = .init(arena);
    write(&out.writer, err) catch return "{\"error\":{\"code\":\"internal\",\"message\":\"out of memory\",\"type\":\"server_error\",\"param\":null}}\n";
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
    try s.objectField("type");
    try s.write(err.kind());
    try s.objectField("param");
    try s.write(err.param);
    try s.endObject();
    try s.endObject();
    try out.writeByte('\n');
}

test "the error body escapes its message" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const text = body(arena_state.allocator(), .init(.bad_request, "invalid_json", "a \"quote\""));
    try std.testing.expectEqualStrings("{\"error\":{\"code\":\"invalid_json\",\"message\":\"a \\\"quote\\\"\",\"type\":\"invalid_request_error\",\"param\":null}}\n", text);
    var with_param: ApiError = .init(http_overloaded, "busy", "overloaded");
    with_param.param = "model";
    try std.testing.expectEqualStrings("{\"error\":{\"code\":\"busy\",\"message\":\"overloaded\",\"type\":\"server_error\",\"param\":\"model\"}}\n", body(arena_state.allocator(), with_param));
}

const http_overloaded: std.http.Status = @fromBackingInt(529);
