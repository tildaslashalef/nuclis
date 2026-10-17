//! Test-only: a loopback HTTP server that plays the search service from
//! canned answers, for the client's, the agent's, and the API's tests.
const std = @import("std");
const Io = std.Io;

/// A one-connection-per-answer HTTP server on loopback for the tests: it
/// records each request's target and `User-Agent` and replies with the next
/// canned answer, optionally late. `stop` drains the answers no request
/// took by connecting itself (a blocked `accept` does not wake on
/// cancellation), so a failing test ends instead of hanging.
pub const Fake = struct {
    /// `cache`, when set, is sent as `X-Search-Cache`.
    pub const Answer = struct { status: u16, body: []const u8, delay_ms: u64 = 0, cache: ?[]const u8 = null };

    server: std.Io.net.Server,
    answers: []const Answer,
    targets: [max_answers][256]u8 = undefined,
    target_lens: [max_answers]usize = @splat(0),
    agent: [64]u8 = undefined,
    agent_len: usize = 0,
    served: std.atomic.Value(usize) = .init(0),

    pub const max_answers = 12;

    pub fn init(io: Io, answers: []const Answer) !Fake {
        std.debug.assert(answers.len <= max_answers);
        const address = try std.Io.net.IpAddress.parseIp4("127.0.0.1", 0);
        return .{ .server = try address.listen(io, .{ .reuse_address = true }), .answers = answers };
    }

    pub fn base(self: *const Fake, buffer: []u8) []const u8 {
        return std.fmt.bufPrint(buffer, "http://127.0.0.1:{d}", .{self.server.socket.address.getPort()}) catch unreachable;
    }

    pub fn target(self: *const Fake, i: usize) []const u8 {
        return self.targets[i][0..self.target_lens[i]];
    }

    pub fn serve(self: *Fake, io: Io) void {
        for (self.answers, 0..) |answer, i| {
            self.answerOne(io, i, answer) catch {};
            _ = self.served.fetchAdd(1, .release);
        }
    }

    pub fn stop(self: *Fake, io: Io, task: *Io.Future(void)) void {
        while (self.served.load(.acquire) < self.answers.len) {
            const address = self.server.socket.address;
            if (address.connect(io, .{ .mode = .stream })) |stream| stream.close(io) else |_| {}
            io.sleep(.fromMilliseconds(10), .awake) catch {};
        }
        task.await(io);
        self.server.deinit(io);
    }

    fn answerOne(self: *Fake, io: Io, i: usize, answer: Answer) !void {
        const stream = try self.server.accept(io);
        defer stream.close(io);
        var read_buffer: [4096]u8 = undefined;
        var reader = std.Io.net.Stream.Reader.init(stream, io, &read_buffer);
        var first = true;
        while (true) {
            const line = std.mem.trimEnd(u8, try reader.interface.takeDelimiterInclusive('\n'), "\r\n");
            if (line.len == 0) break;
            if (first) {
                var words = std.mem.splitScalar(u8, line, ' ');
                _ = words.next();
                const t = words.next() orelse "";
                const n = @min(t.len, self.targets[i].len);
                @memcpy(self.targets[i][0..n], t[0..n]);
                self.target_lens[i] = n;
                first = false;
            } else if (std.ascii.startsWithIgnoreCase(line, "user-agent: ")) {
                const v = line["user-agent: ".len..];
                const n = @min(v.len, self.agent.len);
                @memcpy(self.agent[0..n], v[0..n]);
                self.agent_len = n;
            }
        }
        if (answer.delay_ms > 0) try io.sleep(.fromMilliseconds(@intCast(answer.delay_ms)), .awake);
        var write_buffer: [1024]u8 = undefined;
        var writer = std.Io.net.Stream.Writer.init(stream, io, &write_buffer);
        try writer.interface.print("HTTP/1.1 {d} Answer\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: close\r\n", .{ answer.status, answer.body.len });
        if (answer.cache) |cache| try writer.interface.print("X-Search-Cache: {s}\r\n", .{cache});
        try writer.interface.print("\r\n{s}", .{answer.body});
        try writer.interface.flush();
    }
};
