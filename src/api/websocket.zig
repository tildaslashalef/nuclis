//! RFC 6455 on the server's side, over any reader and writer: the opening
//! handshake's checks and accept value, client frames read into whole
//! messages (masked, fragmented, control frames between fragments, bounded),
//! and server frames written (unmasked, never fragmented). No extensions and
//! no subprotocols are negotiated. Knows no route; `http.zig` hands a
//! connection over after the `101`. docs/guide/api.md § Live transcription.
const std = @import("std");
const errors = @import("errors.zig");
const repeat = @import("../text.zig").repeat;

pub const Opcode = enum(u4) { continuation = 0, text = 1, binary = 2, close = 8, ping = 9, pong = 10, _ };

/// The close codes the server sends (RFC 6455 § 7.4.1).
pub const Code = struct {
    pub const normal: u16 = 1000;
    pub const going_away: u16 = 1001;
    pub const protocol_error: u16 = 1002;
    pub const invalid_data: u16 = 1007;
    pub const policy: u16 = 1008;
    pub const too_big: u16 = 1009;
    pub const internal: u16 = 1011;
    pub const try_again: u16 = 1013;
};

/// What the request's headers say about an upgrade, read by the transport
/// when an `upgrade` header names `websocket`. Slices live in the
/// request's arena.
pub const Handshake = struct {
    /// `connection` lists the `upgrade` token.
    connection_upgrade: bool = false,
    key: []const u8 = "",
    version: []const u8 = "",
};

/// The handshake in `headers`, or null when no `upgrade` header names
/// `websocket`. Values are copied into `arena`: the head's strings die
/// with the next read.
pub fn readHandshake(arena: std.mem.Allocator, headers: *std.http.HeaderIterator) !?Handshake {
    var websocket = false;
    var h: Handshake = .{};
    while (headers.next()) |header| {
        if (std.ascii.eqlIgnoreCase(header.name, "upgrade")) {
            websocket = websocket or hasToken(header.value, "websocket");
        } else if (std.ascii.eqlIgnoreCase(header.name, "connection")) {
            h.connection_upgrade = h.connection_upgrade or hasToken(header.value, "upgrade");
        } else if (std.ascii.eqlIgnoreCase(header.name, "sec-websocket-key")) {
            h.key = try arena.dupe(u8, std.mem.trim(u8, header.value, " \t"));
        } else if (std.ascii.eqlIgnoreCase(header.name, "sec-websocket-version")) {
            h.version = try arena.dupe(u8, std.mem.trim(u8, header.value, " \t"));
        }
    }
    return if (websocket) h else null;
}

/// Whether the comma-separated `list` holds `token`, ignoring case.
fn hasToken(list: []const u8, token: []const u8) bool {
    var it = std.mem.splitScalar(u8, list, ',');
    while (it.next()) |item| if (std.ascii.eqlIgnoreCase(std.mem.trim(u8, item, " \t"), token)) return true;
    return false;
}

/// Why the opening handshake (§ 4.2.1) is refused, or null when it is
/// valid: `426` for a version other than 13 (the response must name 13 in
/// `sec-websocket-version`), `400` for the rest.
pub fn refusal(h: Handshake, method: std.http.Method) ?errors.ApiError {
    if (method != .GET) return .init(.bad_request, "bad_handshake", "a WebSocket opening handshake is a GET");
    if (!h.connection_upgrade) return .init(.bad_request, "bad_handshake", "the handshake's connection header must list upgrade");
    if (!std.mem.eql(u8, h.version, "13")) return .init(.upgrade_required, "unsupported_version", "this server speaks WebSocket version 13 (sec-websocket-version: 13)");
    var nonce: [16]u8 = undefined;
    const decoder = std.base64.standard.Decoder;
    const size = decoder.calcSizeForSlice(h.key) catch 0;
    if (size != nonce.len or (if (decoder.decode(&nonce, h.key)) |_| false else |_| true))
        return .init(.bad_request, "bad_handshake", "sec-websocket-key must be 16 bytes in base64");
    return null;
}

/// `sec-websocket-accept` for `key`: base64 of SHA-1 of the key and the
/// protocol's GUID (§ 4.2.2).
pub fn accept(key: []const u8) [28]u8 {
    var sha1: std.crypto.hash.Sha1 = .init(.{});
    sha1.update(key);
    sha1.update("258EAFA5-E914-47DA-95CA-C5AB0DC85B11");
    var digest: [std.crypto.hash.Sha1.digest_length]u8 = undefined;
    sha1.final(&digest);
    var out: [28]u8 = undefined;
    _ = std.base64.standard.Encoder.encode(&out, &digest);
    return out;
}

pub const Message = union(enum) {
    text: []const u8,
    binary: []const u8,
    ping: []const u8,
    pong: []const u8,
    /// `code` is null when the frame carried none (then the reply carries none either).
    close: struct { code: ?u16, reason: []const u8 },
};

pub const ReadError = error{
    /// A frame the protocol forbids: unmasked, reserved bits or opcodes,
    /// a fragmented or oversized control frame, a continuation out of place.
    ProtocolError,
    /// A message past the reader's limit; nothing of it was read past the header.
    TooBig,
    /// A text message or close reason that is not UTF-8, or an invalid close code.
    InvalidData,
    /// The peer closed without a close frame.
    EndOfStream,
    ReadFailed,
    OutOfMemory,
};

/// The close code a read failure is answered with; null when the
/// connection is gone and nothing can be sent.
pub fn closeCode(err: ReadError) ?u16 {
    return switch (err) {
        error.ProtocolError => Code.protocol_error,
        error.TooBig => Code.too_big,
        error.InvalidData => Code.invalid_data,
        error.OutOfMemory => Code.internal,
        error.EndOfStream, error.ReadFailed => null,
    };
}

/// Reads client messages. A data message is returned whole, after its last
/// fragment; control frames between fragments are returned as they come.
pub const Reader = struct {
    in: *std.Io.Reader,
    /// The largest text or binary message, its fragments together.
    max_message: usize,
    /// The data message being assembled; owned, at most `max_message` bytes.
    message: std.ArrayList(u8) = .empty,
    /// The opcode of the fragmented message in progress.
    partial: ?Opcode = null,
    control: [125]u8 = undefined,

    pub fn deinit(self: *Reader, gpa: std.mem.Allocator) void {
        self.message.deinit(gpa);
    }

    /// The next message or control frame. Its bytes live in the reader until
    /// the next call.
    pub fn next(self: *Reader, gpa: std.mem.Allocator) ReadError!Message {
        while (true) {
            const b0 = try self.in.takeByte();
            const b1 = try self.in.takeByte();
            const fin = b0 & 0x80 != 0;
            if (b0 & 0x70 != 0) return error.ProtocolError;
            // § 5.1: a server closes on a frame its client did not mask.
            if (b1 & 0x80 == 0) return error.ProtocolError;
            const opcode: Opcode = @fromBackingInt(@intCast(@as(u4, @truncate(b0))));
            var len: u64 = b1 & 0x7f;
            if (len == 126) {
                len = try self.in.takeInt(u16, .big);
            } else if (len == 127) {
                len = try self.in.takeInt(u64, .big);
                if (len >> 63 != 0) return error.ProtocolError;
            }
            const key = (try self.in.takeArray(4)).*;
            switch (opcode) {
                .ping, .pong, .close => {
                    if (!fin or len > self.control.len) return error.ProtocolError;
                    const payload = self.control[0..@intCast(len)];
                    try self.in.readSliceAll(payload);
                    unmask(payload, key);
                    return switch (opcode) {
                        .ping => .{ .ping = payload },
                        .pong => .{ .pong = payload },
                        else => try parseClose(payload),
                    };
                },
                .text, .binary => {
                    if (self.partial != null) return error.ProtocolError;
                    self.message.clearRetainingCapacity();
                    self.partial = opcode;
                    try self.appendPayload(gpa, len, key);
                    if (fin) return self.complete();
                },
                .continuation => {
                    if (self.partial == null) return error.ProtocolError;
                    try self.appendPayload(gpa, len, key);
                    if (fin) return self.complete();
                },
                _ => return error.ProtocolError,
            }
        }
    }

    fn appendPayload(self: *Reader, gpa: std.mem.Allocator, len: u64, key: [4]u8) ReadError!void {
        if (len > self.max_message - self.message.items.len) return error.TooBig;
        const n: usize = @intCast(len);
        const start = self.message.items.len;
        try self.message.ensureUnusedCapacity(gpa, n);
        const payload = self.message.unusedCapacitySlice()[0..n];
        try self.in.readSliceAll(payload);
        // Each frame has its own key, starting over at its first byte.
        unmask(payload, key);
        self.message.items.len = start + n;
    }

    fn complete(self: *Reader) ReadError!Message {
        const opcode = self.partial.?;
        self.partial = null;
        const bytes = self.message.items;
        if (opcode == .text) {
            if (!std.unicode.utf8ValidateSlice(bytes)) return error.InvalidData;
            return .{ .text = bytes };
        }
        return .{ .binary = bytes };
    }
};

fn unmask(payload: []u8, key: [4]u8) void {
    for (payload, 0..) |*b, i| b.* ^= key[i % 4];
}

/// A close frame's body (§ 5.5.1): empty, or a code and a UTF-8 reason.
fn parseClose(payload: []const u8) ReadError!Message {
    if (payload.len == 0) return .{ .close = .{ .code = null, .reason = "" } };
    if (payload.len == 1) return error.ProtocolError;
    const code = std.mem.readInt(u16, payload[0..2], .big);
    // § 7.4: codes an endpoint may send; 1004–1006 and 1015 are reserved.
    const valid = (code >= 1000 and code <= 1003) or (code >= 1007 and code <= 1014) or (code >= 3000 and code <= 4999);
    if (!valid) return error.ProtocolError;
    const reason = payload[2..];
    if (!std.unicode.utf8ValidateSlice(reason)) return error.InvalidData;
    return .{ .close = .{ .code = code, .reason = reason } };
}

/// One unfragmented, unmasked server frame.
pub fn writeFrame(out: *std.Io.Writer, opcode: Opcode, payload: []const u8) std.Io.Writer.Error!void {
    var head: [10]u8 = undefined;
    head[0] = 0x80 | @as(u8, @backingInt(opcode));
    var n: usize = 2;
    if (payload.len < 126) {
        head[1] = @intCast(payload.len);
    } else if (payload.len <= std.math.maxInt(u16)) {
        head[1] = 126;
        std.mem.writeInt(u16, head[2..4], @intCast(payload.len), .big);
        n = 4;
    } else {
        head[1] = 127;
        std.mem.writeInt(u64, head[2..10], payload.len, .big);
        n = 10;
    }
    try out.writeAll(head[0..n]);
    try out.writeAll(payload);
}

/// A close frame with `code` and as much of the ASCII `reason` as fits.
pub fn writeClose(out: *std.Io.Writer, code: ?u16, reason: []const u8) std.Io.Writer.Error!void {
    var payload: [125]u8 = undefined;
    const c = code orelse return writeFrame(out, .close, "");
    std.mem.writeInt(u16, payload[0..2], c, .big);
    const n = @min(reason.len, payload.len - 2);
    @memcpy(payload[2..][0..n], reason[0..n]);
    return writeFrame(out, .close, payload[0 .. 2 + n]);
}

// ----- tests: frames as a client sends them, through a byte stream -----

/// A masked client frame.
fn clientFrame(buffer: *std.ArrayList(u8), fin: bool, opcode: u4, payload: []const u8, key: [4]u8) !void {
    const gpa = std.testing.allocator;
    try buffer.append(gpa, (if (fin) @as(u8, 0x80) else 0) | opcode);
    if (payload.len < 126) {
        try buffer.append(gpa, 0x80 | @as(u8, @intCast(payload.len)));
    } else if (payload.len <= 0xffff) {
        try buffer.append(gpa, 0x80 | 126);
        try buffer.appendSlice(gpa, &std.mem.toBytes(std.mem.nativeToBig(u16, @intCast(payload.len))));
    } else {
        try buffer.append(gpa, 0x80 | 127);
        try buffer.appendSlice(gpa, &std.mem.toBytes(std.mem.nativeToBig(u64, payload.len)));
    }
    try buffer.appendSlice(gpa, &key);
    for (payload, 0..) |b, i| try buffer.append(gpa, b ^ key[i % 4]);
}

/// Reads every message of `bytes` through a small buffer, in pieces of at
/// most 5 bytes, as a socket would deliver them; the error that ended it.
fn readAll(bytes: []const u8, max_message: usize, out: *std.ArrayList(u8)) ReadError {
    var read_buffer: [16]u8 = undefined;
    var in: std.testing.Reader = .init(&read_buffer, &.{.{ .buffer = bytes }});
    in.artificial_limit = .limited(5);
    var reader: Reader = .{ .in = &in.interface, .max_message = max_message };
    defer reader.deinit(std.testing.allocator);
    const gpa = std.testing.allocator;
    while (true) {
        const message = reader.next(gpa) catch |err| return err;
        const tag, const data = switch (message) {
            .text => |t| .{ "text", t },
            .binary => |b| .{ "binary", b },
            .ping => |p| .{ "ping", p },
            .pong => |p| .{ "pong", p },
            .close => |c| .{ "close", c.reason },
        };
        out.print(gpa, "{s}:{s};", .{ tag, data }) catch return error.OutOfMemory;
        if (message == .close) out.print(gpa, "code={?d};", .{message.close.code}) catch return error.OutOfMemory;
    }
}

test "the handshake's accept value is RFC 6455's example" {
    try std.testing.expectEqualStrings("s3pPLMBiTxaQ9kYGzzhZRbK+xOo=", &accept("dGhlIHNhbXBsZSBub25jZQ=="));
}

test "a handshake is a GET with connection: upgrade, version 13, and a 16-byte key" {
    const good: Handshake = .{ .connection_upgrade = true, .key = "dGhlIHNhbXBsZSBub25jZQ==", .version = "13" };
    try std.testing.expect(refusal(good, .GET) == null);
    try std.testing.expectEqual(std.http.Status.bad_request, refusal(good, .POST).?.status);
    var v = good;
    v.version = "8";
    try std.testing.expectEqual(std.http.Status.upgrade_required, refusal(v, .GET).?.status);
    var k = good;
    k.key = "c2hvcnQ=";
    try std.testing.expectEqualStrings("bad_handshake", refusal(k, .GET).?.code);
    k.key = "not base64 at all!!!!!==";
    try std.testing.expectEqualStrings("bad_handshake", refusal(k, .GET).?.code);
    var c = good;
    c.connection_upgrade = false;
    try std.testing.expectEqualStrings("bad_handshake", refusal(c, .GET).?.code);

    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    var it: std.http.HeaderIterator = .init("GET / HTTP/1.1\r\nHost: x\r\nUpgrade: WebSocket\r\nConnection: keep-alive, Upgrade\r\nSec-WebSocket-Key:  dGhlIHNhbXBsZSBub25jZQ== \r\nSec-WebSocket-Version: 13\r\n\r\n");
    const read = (try readHandshake(arena_state.allocator(), &it)).?;
    try std.testing.expect(refusal(read, .GET) == null);
    var plain: std.http.HeaderIterator = .init("GET / HTTP/1.1\r\nConnection: upgrade\r\nUpgrade: h2c\r\n\r\n");
    try std.testing.expect(try readHandshake(arena_state.allocator(), &plain) == null);
}

test "fragments join into one message, each frame with its own mask; a ping between them comes first" {
    const gpa = std.testing.allocator;
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(gpa);
    try clientFrame(&bytes, false, 1, "Hel", .{ 1, 2, 3, 4 });
    try clientFrame(&bytes, true, 9, "are you there", .{ 9, 8, 7, 6 });
    try clientFrame(&bytes, false, 0, "lo, ", .{ 0xaa, 0xbb, 0xcc, 0xdd });
    try clientFrame(&bytes, true, 0, "wörld", .{ 5, 6, 7, 8 });
    const long = repeat("x", 300);
    try clientFrame(&bytes, true, 2, long, .{ 1, 1, 1, 1 });
    try clientFrame(&bytes, true, 10, "", .{ 0, 0, 0, 0 });
    try clientFrame(&bytes, true, 8, "\x03\xe8bye", .{ 4, 3, 2, 1 });
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try std.testing.expectEqual(error.EndOfStream, readAll(bytes.items, 1024, &out));
    try std.testing.expectEqualStrings("ping:are you there;text:Hello, wörld;binary:" ++ long ++ ";pong:;close:bye;code=1000;", out.items);
}

test "a 64-bit length, an empty close" {
    const gpa = std.testing.allocator;
    var bytes: std.ArrayList(u8) = .empty;
    defer bytes.deinit(gpa);
    const big = try gpa.alloc(u8, 70_000);
    defer gpa.free(big);
    @memset(big, 'y');
    try clientFrame(&bytes, true, 2, big, .{ 7, 7, 7, 7 });
    try clientFrame(&bytes, true, 8, "", .{ 0, 0, 0, 0 });
    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(gpa);
    try std.testing.expectEqual(error.EndOfStream, readAll(bytes.items, 1 << 20, &out));
    try std.testing.expectEqualStrings("binary:", out.items[0..7]);
    try std.testing.expectEqualSlices(u8, big, out.items[7..][0..big.len]);
    try std.testing.expectEqualStrings(";close:;code=null;", out.items[7 + big.len ..]);
}

test "what a server must refuse" {
    const gpa = std.testing.allocator;
    const Case = struct { frames: []const struct { fin: bool, opcode: u4, payload: []const u8 }, err: ReadError, before: []const u8 = "" };
    const cases = [_]Case{
        // A message past the limit: refused at its header, fragments counted together.
        .{ .frames = &.{.{ .fin = true, .opcode = 2, .payload = "0123456789abcdefg" }}, .err = error.TooBig },
        .{ .frames = &.{ .{ .fin = false, .opcode = 2, .payload = "01234567" }, .{ .fin = true, .opcode = 0, .payload = "89abcdefg" } }, .err = error.TooBig },
        // A continuation with nothing to continue, a new message inside one.
        .{ .frames = &.{.{ .fin = true, .opcode = 0, .payload = "x" }}, .err = error.ProtocolError },
        .{ .frames = &.{ .{ .fin = false, .opcode = 1, .payload = "a" }, .{ .fin = true, .opcode = 1, .payload = "b" } }, .err = error.ProtocolError },
        // Control frames: fragmented, too long; reserved opcodes.
        .{ .frames = &.{.{ .fin = false, .opcode = 9, .payload = "x" }}, .err = error.ProtocolError },
        .{ .frames = &.{.{ .fin = true, .opcode = 9, .payload = repeat("p", 126) }}, .err = error.ProtocolError },
        .{ .frames = &.{.{ .fin = true, .opcode = 3, .payload = "" }}, .err = error.ProtocolError },
        .{ .frames = &.{.{ .fin = true, .opcode = 11, .payload = "" }}, .err = error.ProtocolError },
        // Text that is not UTF-8, split across fragments or not; a bad close.
        .{ .frames = &.{.{ .fin = true, .opcode = 1, .payload = "\xff" }}, .err = error.InvalidData },
        .{ .frames = &.{ .{ .fin = false, .opcode = 1, .payload = "\xc3" }, .{ .fin = true, .opcode = 0, .payload = "\xa9" }, .{ .fin = true, .opcode = 1, .payload = "\xc3" } }, .err = error.InvalidData, .before = "text:é;" },
        .{ .frames = &.{.{ .fin = true, .opcode = 8, .payload = "\x03" }}, .err = error.ProtocolError },
        .{ .frames = &.{.{ .fin = true, .opcode = 8, .payload = "\x03\xed" }}, .err = error.ProtocolError },
        .{ .frames = &.{.{ .fin = true, .opcode = 8, .payload = "\x03\xe8\xff" }}, .err = error.InvalidData },
        // The client closes the connection mid-message, mid-header.
        .{ .frames = &.{.{ .fin = false, .opcode = 2, .payload = "abc" }}, .err = error.EndOfStream },
    };
    for (cases, 0..) |case, i| {
        errdefer std.debug.print("case {d}\n", .{i});
        var bytes: std.ArrayList(u8) = .empty;
        defer bytes.deinit(gpa);
        for (case.frames) |f| try clientFrame(&bytes, f.fin, f.opcode, f.payload, .{ 0x12, 0x34, 0x56, 0x78 });
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(gpa);
        try std.testing.expectEqual(case.err, readAll(bytes.items, 16, &out));
        try std.testing.expectEqualStrings(case.before, out.items);
    }
    // Unmasked, reserved bits, a length with its top bit set, a truncated header.
    const raw = [_]struct { bytes: []const u8, err: ReadError }{
        .{ .bytes = "\x82\x01x", .err = error.ProtocolError },
        .{ .bytes = "\xc2\x80\x00\x00\x00\x00", .err = error.ProtocolError },
        .{ .bytes = "\x82\xff\x80\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00", .err = error.ProtocolError },
        .{ .bytes = "\x82\x85\x00", .err = error.EndOfStream },
    };
    for (raw) |case| {
        var out: std.ArrayList(u8) = .empty;
        defer out.deinit(gpa);
        try std.testing.expectEqual(case.err, readAll(case.bytes, 16, &out));
    }
    try std.testing.expectEqual(@as(?u16, 1009), closeCode(error.TooBig));
    try std.testing.expectEqual(@as(?u16, null), closeCode(error.EndOfStream));
}

test "server frames: the three length forms, unmasked, and a close with its code" {
    var buffer: [70_000]u8 = undefined;
    var out: std.Io.Writer = .fixed(&buffer);
    try writeFrame(&out, .text, "hi");
    try std.testing.expectEqualSlices(u8, "\x81\x02hi", out.buffered());
    out.end = 0;
    try writeFrame(&out, .binary, repeat("\x00", 200));
    try std.testing.expectEqualSlices(u8, "\x82\x7e\x00\xc8", out.buffered()[0..4]);
    out.end = 0;
    const zeros = try std.testing.allocator.alloc(u8, 65_536);
    defer std.testing.allocator.free(zeros);
    @memset(zeros, 0);
    try writeFrame(&out, .binary, zeros);
    try std.testing.expectEqualSlices(u8, "\x82\x7f\x00\x00\x00\x00\x00\x01\x00\x00", out.buffered()[0..10]);
    out.end = 0;
    try writeClose(&out, Code.policy, "idle");
    try std.testing.expectEqualSlices(u8, "\x88\x06\x03\xf0idle", out.buffered());
    out.end = 0;
    try writeClose(&out, null, "");
    try std.testing.expectEqualSlices(u8, "\x88\x00", out.buffered());
}
