//! A bounded `multipart/form-data` reader (RFC 7578 over RFC 2046's
//! delimiters) for a body the transport already holds whole: every part is
//! a slice of the body, nothing is copied, so a large upload costs its
//! bytes once. Part count and each part's header block are bounded, at most
//! one part is a file, and every refusal is a typed error. A delimiter is
//! CRLF, `--`, the boundary, then `--` (the last) or optional spaces and
//! CRLF; anything else after the boundary is content.
const std = @import("std");
const repeat = @import("../text.zig").repeat;

pub const Limits = struct {
    parts: usize = 32,
    /// One part's header lines, the blank line included.
    header_bytes: usize = 1024,
};

pub const Error = error{
    /// The content type is not `multipart/form-data`.
    NotMultipart,
    /// It names no boundary, or one longer than RFC 2046's 70 characters.
    InvalidBoundary,
    /// No delimiter where one belongs, a truncated body, a malformed header.
    Malformed,
    TooManyParts,
    HeaderTooLarge,
    /// A part without `content-disposition: form-data; name=…`.
    MissingName,
    /// A second part with a `filename`.
    TooManyFiles,
    OutOfMemory,
};

pub const Part = struct {
    name: []const u8,
    /// Set for a file part (its disposition has a `filename`), maybe empty.
    filename: ?[]const u8 = null,
    content_type: ?[]const u8 = null,
    data: []const u8,
};

pub const Form = struct {
    /// In the parse's arena; their strings are slices of the body.
    parts: []const Part,

    /// The first part named `name` that is not a file.
    pub fn field(self: Form, name: []const u8) ?[]const u8 {
        for (self.parts) |p| if (p.filename == null and std.mem.eql(u8, p.name, name)) return p.data;
        return null;
    }

    /// The file part, if one was sent.
    pub fn file(self: Form) ?Part {
        for (self.parts) |p| if (p.filename != null) return p;
        return null;
    }
};

/// The boundary a `multipart/form-data` content type names.
pub fn boundary(content_type: []const u8) Error![]const u8 {
    var params = std.mem.splitScalar(u8, content_type, ';');
    const media = std.mem.trim(u8, params.first(), " \t");
    if (!std.ascii.eqlIgnoreCase(media, "multipart/form-data")) return error.NotMultipart;
    while (params.next()) |param| {
        const p = std.mem.trim(u8, param, " \t");
        const eq = std.mem.indexOfScalar(u8, p, '=') orelse continue;
        if (!std.ascii.eqlIgnoreCase(std.mem.trim(u8, p[0..eq], " \t"), "boundary")) continue;
        var value = std.mem.trim(u8, p[eq + 1 ..], " \t");
        if (value.len >= 2 and value[0] == '"' and value[value.len - 1] == '"') value = value[1 .. value.len - 1];
        if (value.len == 0 or value.len > 70) return error.InvalidBoundary;
        return value;
    }
    return error.InvalidBoundary;
}

/// The parts of `body` under `content_type`'s boundary; `arena` holds the
/// list. The preamble and the epilogue are ignored.
pub fn parse(arena: std.mem.Allocator, body: []const u8, content_type: []const u8, limits: Limits) Error!Form {
    const b = try boundary(content_type);
    const delimiter = try std.mem.concat(arena, u8, &.{ "\r\n--", b });
    var parts: std.ArrayList(Part) = .empty;
    // The first delimiter may open the body, without its CRLF.
    const opens = std.mem.startsWith(u8, body, delimiter[2..]) and ends(body, delimiter.len - 2);
    var at = if (opens) delimiter.len - 2 else (try next(body, 0, delimiter) orelse return error.Malformed).after;
    var files: usize = 0;
    while (true) {
        // After a delimiter: `--` ends the body, else spaces and CRLF open a part.
        if (std.mem.startsWith(u8, body[at..], "--")) break;
        at = lineEnd(body, at) orelse return error.Malformed;
        if (parts.items.len == limits.parts) return error.TooManyParts;
        if (std.mem.startsWith(u8, body[at..], "\r\n")) return error.MissingName;
        const head_end = std.mem.indexOfPos(u8, body[0..@min(body.len, at + limits.header_bytes)], at, "\r\n\r\n") orelse {
            if (body.len - at > limits.header_bytes) return error.HeaderTooLarge;
            return error.Malformed;
        };
        var part = try headers(body[at .. head_end + 2]);
        const found = try next(body, head_end + 4, delimiter) orelse return error.Malformed;
        part.data = body[head_end + 4 .. found.start];
        if (part.filename != null) {
            files += 1;
            if (files > 1) return error.TooManyFiles;
        }
        try parts.append(arena, part);
        at = found.after;
    }
    return .{ .parts = parts.items };
}

const Found = struct { start: usize, after: usize };

/// The next delimiter at or after `from`: where it starts and the index
/// past the boundary. A match the boundary does not end is content.
fn next(body: []const u8, from: usize, delimiter: []const u8) Error!?Found {
    var at = from;
    while (std.mem.indexOfPos(u8, body, at, delimiter)) |start| {
        const after = start + delimiter.len;
        if (ends(body, after)) return .{ .start = start, .after = after };
        at = start + 1;
    }
    return null;
}

/// Whether a boundary that ends at `at` ends a delimiter.
fn ends(body: []const u8, at: usize) bool {
    return std.mem.startsWith(u8, body[at..], "--") or lineEnd(body, at) != null;
}

/// Past optional spaces or tabs and a CRLF at `at`, or null.
fn lineEnd(body: []const u8, at: usize) ?usize {
    var i = at;
    while (i < body.len and (body[i] == ' ' or body[i] == '\t')) i += 1;
    return if (std.mem.startsWith(u8, body[i..], "\r\n")) i + 2 else null;
}

/// A part's header lines, each ending in CRLF.
fn headers(block: []const u8) Error!Part {
    var part: Part = .{ .name = "", .data = "" };
    var named = false;
    var lines = std.mem.splitSequence(u8, block[0 .. block.len - 2], "\r\n");
    while (lines.next()) |line| {
        const colon = std.mem.indexOfScalar(u8, line, ':') orelse return error.Malformed;
        const key = line[0..colon];
        if (key.len == 0 or std.mem.indexOfAny(u8, key, " \t") != null) return error.Malformed;
        const value = std.mem.trim(u8, line[colon + 1 ..], " \t");
        if (std.ascii.eqlIgnoreCase(key, "content-type")) {
            part.content_type = value;
        } else if (std.ascii.eqlIgnoreCase(key, "content-disposition")) {
            var params = Params{ .rest = value };
            const kind = params.token() orelse return error.Malformed;
            if (!std.ascii.eqlIgnoreCase(kind, "form-data")) return error.MissingName;
            while (try params.next()) |p| {
                if (std.ascii.eqlIgnoreCase(p.key, "name")) {
                    part.name = p.value;
                    named = true;
                } else if (std.ascii.eqlIgnoreCase(p.key, "filename") or (std.ascii.eqlIgnoreCase(p.key, "filename*") and part.filename == null)) {
                    part.filename = p.value;
                }
            }
        }
    }
    if (!named) return error.MissingName;
    return part;
}

/// `; key=value` pairs, a value a token or a quoted string (its escapes
/// kept as sent: a name never needs them).
const Params = struct {
    rest: []const u8,

    fn token(self: *Params) ?[]const u8 {
        const end = std.mem.indexOfScalar(u8, self.rest, ';') orelse self.rest.len;
        const t = std.mem.trim(u8, self.rest[0..end], " \t");
        self.rest = self.rest[end..];
        return if (t.len > 0) t else null;
    }

    fn next(self: *Params) Error!?struct { key: []const u8, value: []const u8 } {
        self.rest = std.mem.trimStart(u8, self.rest, " \t");
        if (self.rest.len == 0) return null;
        if (self.rest[0] != ';') return error.Malformed;
        self.rest = std.mem.trimStart(u8, self.rest[1..], " \t");
        const eq = std.mem.indexOfScalar(u8, self.rest, '=') orelse return error.Malformed;
        const key = std.mem.trim(u8, self.rest[0..eq], " \t");
        self.rest = std.mem.trimStart(u8, self.rest[eq + 1 ..], " \t");
        if (self.rest.len > 0 and self.rest[0] == '"') {
            var i: usize = 1;
            while (i < self.rest.len and self.rest[i] != '"') : (i += 1) {
                if (self.rest[i] == '\\') i += 1;
            }
            if (i >= self.rest.len) return error.Malformed;
            const value = self.rest[1..i];
            self.rest = self.rest[i + 1 ..];
            return .{ .key = key, .value = value };
        }
        const end = std.mem.indexOfScalar(u8, self.rest, ';') orelse self.rest.len;
        const value = std.mem.trim(u8, self.rest[0..end], " \t");
        self.rest = self.rest[end..];
        return .{ .key = key, .value = value };
    }
};

// ----- tests -----

const test_type = "multipart/form-data; boundary=XyZ";

fn expectParts(body: []const u8, expected: []const Part) !void {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const form = try parse(arena_state.allocator(), body, test_type, .{});
    try std.testing.expectEqual(expected.len, form.parts.len);
    for (expected, form.parts) |e, p| {
        try std.testing.expectEqualStrings(e.name, p.name);
        try std.testing.expectEqualStrings(e.data, p.data);
        try std.testing.expectEqual(e.filename == null, p.filename == null);
        if (e.filename) |f| try std.testing.expectEqualStrings(f, p.filename.?);
        try std.testing.expectEqual(e.content_type == null, p.content_type == null);
        if (e.content_type) |c| try std.testing.expectEqualStrings(c, p.content_type.?);
    }
}

fn expectRefused(body: []const u8, err: Error, limits: Limits) !void {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    try std.testing.expectError(err, parse(arena_state.allocator(), body, test_type, limits));
}

test "the boundary from the content type: quoted or not, any case, bounded" {
    try std.testing.expectEqualStrings("XyZ", try boundary(test_type));
    try std.testing.expectEqualStrings("a b", try boundary("Multipart/Form-Data ; charset=utf-8; BOUNDARY=\"a b\""));
    try std.testing.expectEqualStrings("----WebKitFormBoundary7MA4YWxkTrZu0gW", try boundary("multipart/form-data; boundary=----WebKitFormBoundary7MA4YWxkTrZu0gW"));
    try std.testing.expectError(error.NotMultipart, boundary("application/json"));
    try std.testing.expectError(error.NotMultipart, boundary("multipart/mixed; boundary=x"));
    try std.testing.expectError(error.InvalidBoundary, boundary("multipart/form-data"));
    try std.testing.expectError(error.InvalidBoundary, boundary("multipart/form-data; boundary="));
    try std.testing.expectError(error.InvalidBoundary, boundary("multipart/form-data; boundary=" ++ repeat("b", 71)));
}

test "fields and one file, as curl and the SDKs send them" {
    const body = "--XyZ\r\n" ++
        "Content-Disposition: form-data; name=\"model\"\r\n\r\n" ++
        "nemotron-asr-0.6b\r\n" ++
        "--XyZ\r\n" ++
        "content-disposition: form-data; name=\"file\"; filename=\"memo one.wav\"\r\n" ++
        "Content-Type: audio/wav\r\n\r\n" ++
        "RIFF\x00\r\n\r\n--XyZ-not-a-delimiter\r\n--XyZ\r\n" ++
        "Content-Disposition: form-data; name=\"timestamp_granularities[]\"\r\n\r\n" ++
        "word\r\n" ++
        "--XyZ--\r\n";
    try expectParts(body, &.{
        .{ .name = "model", .data = "nemotron-asr-0.6b" },
        .{ .name = "file", .filename = "memo one.wav", .content_type = "audio/wav", .data = "RIFF\x00\r\n\r\n--XyZ-not-a-delimiter" },
        .{ .name = "timestamp_granularities[]", .data = "word" },
    });
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const form = try parse(arena_state.allocator(), body, test_type, .{});
    try std.testing.expectEqualStrings("nemotron-asr-0.6b", form.field("model").?);
    try std.testing.expect(form.field("file") == null);
    try std.testing.expectEqualStrings("memo one.wav", form.file().?.filename.?);
    // The parts are the body's own bytes.
    try std.testing.expect(form.file().?.data.ptr == body.ptr + std.mem.indexOf(u8, body, "RIFF").?);
}

test "preamble, epilogue, padding after the boundary, empty values and files" {
    try expectParts("ignored preamble\r\n--XyZ \t\r\nContent-Disposition: form-data; name=a\r\n\r\n\r\n--XyZ\r\n" ++
        "Content-Disposition: form-data; name=\"f\"; filename=\"\"\r\n\r\n\r\n--XyZ--epilogue, also ignored", &.{
        .{ .name = "a", .data = "" },
        .{ .name = "f", .filename = "", .data = "" },
    });
    try expectParts("--XyZ--", &.{});
    // A quoted name may hold a semicolon and an escaped quote.
    try expectParts("--XyZ\r\nContent-Disposition: form-data; name=\"a;\\\"b\"\r\n\r\nv\r\n--XyZ--", &.{.{ .name = "a;\\\"b", .data = "v" }});
}

test "content ending in every prefix of a delimiter is content" {
    // A streaming reader meets these split across reads; whole, each must
    // still end only at the real delimiter.
    const tails = [_][]const u8{ "", "\r", "\r\n", "\r\n-", "\r\n--", "\r\n--X", "\r\n--Xy", "\r\n--XyZz", "\r\n--XyZ-", "-", "--XyZ" };
    for (tails) |tail| {
        var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
        defer arena_state.deinit();
        const arena = arena_state.allocator();
        for (0..3) |pad| {
            // CRLF then the boundary would be a delimiter inside the file.
            if (pad == 2 and std.mem.eql(u8, tail, "--XyZ")) continue;
            const data = try std.mem.concat(arena, u8, &.{ "xxx", "\r\n"[0..@min(pad, 2)], tail });
            const body = try std.mem.concat(arena, u8, &.{ "--XyZ\r\nContent-Disposition: form-data; name=\"f\"; filename=\"a\"\r\n\r\n", data, "\r\n--XyZ--\r\n" });
            errdefer std.debug.print("tail {any}, pad {d}\n", .{ tail, pad });
            const form = try parse(arena, body, test_type, .{});
            try std.testing.expectEqualStrings(data, form.file().?.data);
        }
    }
}

test "refusals: no delimiter, truncated, bad headers, too many parts or files, oversize headers" {
    const part = "--XyZ\r\nContent-Disposition: form-data; name=\"a\"\r\n\r\nv\r\n";
    try expectRefused("no delimiter at all", error.Malformed, .{});
    try expectRefused(part, error.Malformed, .{});
    try expectRefused("--XyZ\r\nContent-Disposition: form-data; name=\"a\"\r\n", error.Malformed, .{});
    // A first line the boundary does not end is preamble.
    try expectParts("--XyZgarbage\r\n" ++ part ++ "--XyZ--", &.{.{ .name = "a", .data = "v" }});
    try expectRefused("--XyZ\r\n\r\nv\r\n--XyZ--", error.MissingName, .{});
    try expectRefused("--XyZ\r\nno colon here\r\n\r\nv\r\n--XyZ--", error.Malformed, .{});
    try expectRefused("--XyZ\r\nContent-Disposition: form-data; name=\"a\r\n\r\nv\r\n--XyZ--", error.Malformed, .{});
    try expectRefused("--XyZ\r\nContent-Type: text/plain\r\n\r\nv\r\n--XyZ--", error.MissingName, .{});
    try expectRefused("--XyZ\r\nContent-Disposition: attachment; name=\"a\"\r\n\r\nv\r\n--XyZ--", error.MissingName, .{});
    try expectRefused(part ++ part ++ part ++ "--XyZ--", error.TooManyParts, .{ .parts = 2 });
    try expectParts(part ++ part ++ "--XyZ--", &.{ .{ .name = "a", .data = "v" }, .{ .name = "a", .data = "v" } });
    const file = "--XyZ\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\n\r\nRIFF\r\n";
    try expectRefused(file ++ file ++ "--XyZ--", error.TooManyFiles, .{});
    const long = "--XyZ\r\nContent-Disposition: form-data; name=\"a\"\r\nX-Pad: " ++ repeat("p", 100) ++ "\r\n\r\nv\r\n--XyZ--";
    try expectRefused(long, error.HeaderTooLarge, .{ .header_bytes = 64 });
    try expectParts(long, &.{.{ .name = "a", .data = "v" }});
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    try std.testing.expectError(error.NotMultipart, parse(arena_state.allocator(), part, "application/x-www-form-urlencoded", .{}));
}
