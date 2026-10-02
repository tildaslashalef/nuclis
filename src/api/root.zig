//! `nuclis serve`: the nuclis API. Composes the transport (`http.zig`), the
//! router, the GPU executor, and the services (decisions now), then accepts
//! connections until the process ends, each connection its own task. The
//! layers stay apart: the transport and the router know no model, a service
//! never touches a socket, and only the executor's worker runs a model.
//! docs/reference/api.md.
const std = @import("std");
const inference = @import("inference");
const config = @import("../config.zig");
const style = @import("../tui/style.zig");
const http = @import("http.zig");
const errors = @import("errors.zig");
const gpu = @import("gpu.zig");
const models = @import("models.zig");
const router_mod = @import("router.zig");
const decisions = @import("decisions/service.zig");
const batcher = @import("decisions/batcher.zig");

pub const default_port: u16 = 8735;

/// Host constants; a client never chooses them.
pub const limits = struct {
    pub const transport: http.Limits = .{ .max_head = 16 * 1024, .max_body = 4 * 1024 * 1024 };
    pub const connections = 64;
    pub const queued_jobs = 64;
    pub const write_buffer = 16 * 1024;
};

pub const Options = struct {
    host: []const u8 = "127.0.0.1",
    port: u16 = default_port,
    /// Opened at start, in order; others open on first use.
    models: std.ArrayList([]const u8) = .empty,
    backend: ?inference.decide.Backend = null,
};

/// Parses the words after `serve`; `arena` owns the list.
pub fn parseArgs(arena: std.mem.Allocator, args: []const []const u8, diag: *config.Diagnostic) !Options {
    var o: Options = .{};
    var seen: struct { host: bool = false, port: bool = false } = .{};
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const flag = args[i];
        const known = for ([_][]const u8{ "--host", "--port", "--model", "--backend" }) |k| {
            if (std.mem.eql(u8, flag, k)) break true;
        } else false;
        if (!known) {
            diag.set("{s} is not an option of serve (`nuclis serve --help`)", .{flag});
            return error.UnknownOption;
        }
        i += 1;
        if (i == args.len) {
            diag.set("{s} needs a value", .{flag});
            return error.MissingOptionValue;
        }
        const value = args[i];
        if (std.mem.eql(u8, flag, "--host")) {
            if (seen.host) return error.DuplicateOption;
            seen.host = true;
            o.host = value;
        } else if (std.mem.eql(u8, flag, "--port")) {
            if (seen.port) return error.DuplicateOption;
            seen.port = true;
            o.port = std.fmt.parseInt(u16, value, 10) catch {
                diag.set("--port takes a number from 0 to 65535, not {s}", .{value});
                return error.InvalidOptionValue;
            };
        } else if (std.mem.eql(u8, flag, "--model")) {
            try o.models.append(arena, value);
        } else {
            if (o.backend != null) return error.DuplicateOption;
            o.backend = std.meta.stringToEnum(inference.decide.Backend, value) orelse {
                diag.set("--backend takes cpu or metal, not {s}", .{value});
                return error.InvalidOptionValue;
            };
        }
    }
    if (o.models.items.len > @import("decisions/pool.zig").capacity) {
        diag.set("at most {d} --model (the server keeps {d} decision models open)", .{ @import("decisions/pool.zig").capacity, @import("decisions/pool.zig").capacity });
        return error.InvalidOptionValue;
    }
    return o;
}

/// The address `--host` names: an IP literal, or `localhost`.
fn address(host: []const u8, port: u16, diag: *config.Diagnostic) !std.Io.net.IpAddress {
    if (std.mem.eql(u8, host, "localhost")) return .{ .ip4 = .loopback(port) };
    return std.Io.net.IpAddress.parse(host, port) catch {
        diag.set("--host takes an IP address (127.0.0.1, ::1, 0.0.0.0) or localhost, not {s}", .{host});
        return error.InvalidOptionValue;
    };
}

fn isLoopback(a: std.Io.net.IpAddress) bool {
    return switch (a) {
        .ip4 => |v| v.bytes[0] == 127,
        .ip6 => |v| v.isLoopBack(),
    };
}

/// Everything a connection task reaches; lives for the whole serve.
const Server = struct {
    gpa: std.mem.Allocator,
    router: router_mod.Router = .{},
    listing: models.Models = .{},
    executor: gpu.Executor = .init(limits.queued_jobs),
    decisions: decisions.Service,
    backend: inference.decide.Backend,
    version: []const u8,
    active: std.atomic.Value(u32) = .init(0),

    fn register(self: *Server) !void {
        try self.decisions.register(self.gpa, &self.router, &self.listing);
        try self.listing.register(self.gpa, &self.router);
        try self.router.add(self.gpa, .GET, router_mod.prefix ++ "/health", .{ .context = self, .handle = health });
    }

    fn deinit(self: *Server) void {
        self.router.deinit(self.gpa);
        self.listing.deinit(self.gpa);
    }

    fn health(context: *anyopaque, arena: std.mem.Allocator, io: std.Io, request: http.Request) http.Response {
        _ = request;
        const self: *Server = @ptrCast(@alignCast(context));
        const stats = self.executor.stats(io);
        const batching = self.decisions.batcher.stats(io);
        const open = self.decisions.pool.openNames(io, arena) catch &.{};
        var out: std.Io.Writer.Allocating = .init(arena);
        writeHealth(&out.writer, self.version, self.backend, open, stats, batching, self.active.load(.monotonic)) catch
            return .fromError(arena, .init(.internal_server_error, "internal", "out of memory"));
        return .{ .body = out.written() };
    }

    /// One connection's task: serve it, then close it and free its slot.
    fn connection(self: *Server, io: std.Io, stream: std.Io.net.Stream) void {
        defer {
            stream.close(io);
            _ = self.active.fetchSub(1, .monotonic);
        }
        noDelay(stream);
        var read_buffer: [limits.transport.max_head]u8 = undefined;
        var write_buffer: [limits.write_buffer]u8 = undefined;
        var reader = stream.reader(io, &read_buffer);
        var writer = stream.writer(io, &write_buffer);
        http.serve(self.gpa, io, &reader.interface, &writer.interface, self.router.handler(), limits.transport);
    }
};

fn writeHealth(out: *std.Io.Writer, version: []const u8, backend: inference.decide.Backend, open: []const []const u8, stats: gpu.Stats, batching: batcher.Stats, connections: u32) !void {
    var s: std.json.Stringify = .{ .writer = out, .options = .{ .whitespace = .indent_2 } };
    try s.beginObject();
    try s.objectField("status");
    try s.write("ok");
    try s.objectField("version");
    try s.write(version);
    try s.objectField("backend");
    try s.write(@tagName(backend));
    try s.objectField("loaded");
    try s.write(open);
    try s.objectField("queue");
    try s.beginObject();
    try s.objectField("queued");
    try s.write(stats.queued);
    try s.objectField("running");
    try s.write(stats.running);
    try s.objectField("completed");
    try s.write(stats.completed);
    try s.endObject();
    try s.objectField("decisions");
    try s.beginObject();
    try s.objectField("waiting");
    try s.write(batching.waiting);
    try s.objectField("batches");
    try s.write(batching.batches);
    try s.objectField("requests");
    try s.write(batching.jobs);
    try s.endObject();
    try s.objectField("connections");
    try s.write(connections);
    try s.endObject();
    try out.writeByte('\n');
}

/// Small responses go out at once rather than waiting on the client's ack.
fn noDelay(stream: std.Io.net.Stream) void {
    const on: c_int = 1;
    std.posix.setsockopt(stream.socket.handle, std.posix.IPPROTO.TCP, std.posix.TCP.NODELAY, std.mem.asBytes(&on)) catch {};
}

pub const Context = struct {
    root: ?[]const u8,
    registry: config.Models,
    default_model: []const u8,
    version: []const u8,
};

/// Serves until the process ends or accepting fails. Prints where it
/// listens and what it opened to `out`.
pub fn serve(gpa: std.mem.Allocator, io: std.Io, context: Context, options: Options, out: *std.Io.Writer, sty: style.Style, diag: *config.Diagnostic) !void {
    const backend = options.backend orelse inference.decide.default_backend;
    const listen_address = try address(options.host, options.port, diag);
    var server: Server = .{
        .gpa = gpa,
        .decisions = .init(gpa, undefined, backend, context.root, context.registry, context.default_model),
        .backend = backend,
        .version = context.version,
    };
    server.decisions.executor = &server.executor;
    server.decisions.bind();
    defer server.deinit();
    try server.register();

    var worker = try io.concurrent(gpu.Executor.run, .{ &server.executor, io });
    defer {
        server.executor.stop(io);
        worker.await(io);
        server.decisions.deinit(io);
    }

    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    for (options.models.items) |name| {
        const started = std.Io.Clock.awake.now(io);
        try server.decisions.preload(arena_state.allocator(), io, name, diag);
        const ms = @as(f64, @floatFromInt(started.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / std.time.ns_per_ms;
        try out.print("{s}opened{s} {s} {s}({d:.0} ms){s}\n", .{ sty.on(.success), sty.off(), name, sty.on(.dim), ms, sty.off() });
    }

    var listener = listen_address.listen(io, .{ .reuse_address = true, .kernel_backlog = 128 }) catch |err| {
        diag.set("cannot listen on {f}: {s}", .{ listen_address, @errorName(err) });
        return err;
    };
    defer listener.deinit(io);
    if (!isLoopback(listen_address))
        try out.print("{s}warning:{s} listening beyond this machine ({s}); the API has no authentication\n", .{ sty.on(.warning), sty.off(), options.host });
    try out.print("{s}nuclis serve{s} listening on {s}http://{f}/v1{s} {s}(backend {s}, default model {s}; Ctrl-C stops){s}\n", .{ sty.on(.header), sty.off(), sty.on(.code), listen_address, sty.off(), sty.on(.dim), @tagName(backend), context.default_model, sty.off() });
    try out.flush();

    var group: std.Io.Group = .init;
    defer group.cancel(io);
    while (true) {
        const stream = listener.accept(io) catch |err| switch (err) {
            error.ConnectionAborted, error.ProtocolFailure, error.BlockedByFirewall => continue,
            // Out of descriptors or memory: refuse this one, keep serving.
            error.ProcessFdQuotaExceeded, error.SystemFdQuotaExceeded, error.SystemResources => {
                try io.sleep(.fromMilliseconds(10), .awake);
                continue;
            },
            else => {
                diag.set("accepting connections failed: {s}", .{@errorName(err)});
                return err;
            },
        };
        if (server.active.fetchAdd(1, .monotonic) >= limits.connections) {
            refuse(io, stream, &server.active);
            continue;
        }
        group.concurrent(io, Server.connection, .{ &server, io, stream }) catch refuse(io, stream, &server.active);
    }
}

/// Answers `busy` and closes: over the connection limit, or no task to run it.
fn refuse(io: std.Io, stream: std.Io.net.Stream, active: *std.atomic.Value(u32)) void {
    defer {
        stream.close(io);
        _ = active.fetchSub(1, .monotonic);
    }
    var buffer: [512]u8 = undefined;
    var writer = stream.writer(io, &buffer);
    var body_buffer: [256]u8 = undefined;
    var body: std.Io.Writer = .fixed(&body_buffer);
    errors.write(&body, .init(http.overloaded, "busy", "too many connections; retry shortly")) catch return;
    http.writeClosing(&writer.interface, http.overloaded, body.buffered()) catch {};
}

test "serve arguments" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diag: config.Diagnostic = .{};
    const o = try parseArgs(arena, &.{ "--port", "9000", "--model", "laya", "--model", "laya-multilingual", "--backend", "cpu", "--host", "0.0.0.0" }, &diag);
    try std.testing.expectEqual(@as(u16, 9000), o.port);
    try std.testing.expectEqual(@as(usize, 2), o.models.items.len);
    try std.testing.expectEqual(inference.decide.Backend.cpu, o.backend.?);
    try std.testing.expectEqualStrings("0.0.0.0", o.host);
    try std.testing.expectEqual(default_port, (try parseArgs(arena, &.{}, &diag)).port);
    const cases = .{
        .{ &[_][]const u8{ "--port", "70000" }, error.InvalidOptionValue },
        .{ &[_][]const u8{ "--backend", "tpu" }, error.InvalidOptionValue },
        .{ &[_][]const u8{ "--model", "a", "--model", "b", "--model", "c" }, error.InvalidOptionValue },
        .{ &[_][]const u8{ "--port", "1", "--port", "2" }, error.DuplicateOption },
        .{ &[_][]const u8{"--json"}, error.UnknownOption },
        .{ &[_][]const u8{"--host"}, error.MissingOptionValue },
    };
    inline for (cases) |case| try std.testing.expectError(case[1], parseArgs(arena, case[0], &diag));
    try std.testing.expect(isLoopback(try address("localhost", 1, &diag)));
    try std.testing.expect(isLoopback(try address("::1", 1, &diag)));
    try std.testing.expect(!isLoopback(try address("0.0.0.0", 1, &diag)));
    try std.testing.expectError(error.InvalidOptionValue, address("example.com", 1, &diag));
}

test "health reports the queue and the open models" {
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    try writeHealth(&out.writer, "0.1.0-dev", .cpu, &.{"laya"}, .{ .queued = 2, .running = true, .completed = 7 }, .{ .waiting = 5, .batches = 3, .jobs = 9 }, 3);
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, out.written(), .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings("cpu", parsed.value.object.get("backend").?.string);
    try std.testing.expectEqual(@as(i64, 2), parsed.value.object.get("queue").?.object.get("queued").?.integer);
    try std.testing.expectEqualStrings("laya", parsed.value.object.get("loaded").?.array.items[0].string);
    try std.testing.expectEqual(@as(i64, 5), parsed.value.object.get("decisions").?.object.get("waiting").?.integer);
}
