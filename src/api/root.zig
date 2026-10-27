//! `nuclis serve`: the nuclis API. Composes the transport (`http.zig`), the
//! router, the GPU executor, and the services (decisions, chat, embeddings,
//! speech, and the transcription models),
//! then accepts connections, each its own task, until Ctrl-C or a failed
//! accept; Ctrl-C drains the queued decisions and embeddings before the
//! connections close. The
//! layers stay apart: the transport and the router know no model, a service
//! never touches a socket, and only the executor's worker runs a model.
//! docs/guide/api.md.
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
const chat = @import("chat/service.zig");
const embeddings = @import("embeddings/service.zig");
const embedding_batcher = @import("embeddings/batcher.zig");
const speech = @import("speech/service.zig");
const transcription = @import("transcription/service.zig");
const memory = @import("memory.zig");
const log_mod = @import("log.zig");
const batcher = @import("decisions/batcher.zig");
const interrupt = @import("../interrupt.zig");
const web = @import("../web/root.zig");
const catalog = @import("../catalog.zig");

/// Host constants; a client never chooses them.
pub const limits = struct {
    pub const transport: http.Limits = .{ .max_head = 16 * 1024, .max_body = 4 * 1024 * 1024 };
    pub const connections = 64;
    pub const queued_jobs = 64;
    pub const write_buffer = 16 * 1024;
    /// How long Ctrl-C waits for queued decisions and embeddings before
    /// closing connections.
    pub const drain_seconds = 10;
};

/// Defaults come from the configuration's `serve` section; the flags
/// override them.
pub const Options = struct {
    host: []const u8,
    port: u16,
    /// A line per request on stdout.
    log: bool,
    /// Seconds a decision request may wait for the GPU, at least 1.
    timeout: u32,
    /// Opened at start, by kind, each by its service (`byKind`); every
    /// other opens on its first request.
    models: std.ArrayList([]const u8) = .empty,
    /// Opened at start too, and never closed to make room: `--pin`, else
    /// `serve.pin`.
    pins: std.ArrayList([]const u8) = .empty,
    backend: ?inference.decide.Backend = null,
    /// What every open model may hold together; null is the default.
    memory_bytes: ?u64 = null,
};

/// Parses the words after `serve` over `serve`'s configured `host` and
/// `port`; `arena` owns the list.
pub fn parseArgs(arena: std.mem.Allocator, args: []const []const u8, configured: config.Config.Serve, diag: *config.Diagnostic) !Options {
    var o: Options = .{ .host = configured.host, .port = configured.port, .log = configured.log, .timeout = configured.timeout, .memory_bytes = configured.memory_bytes };
    var seen: struct { host: bool = false, port: bool = false, timeout: bool = false, memory: bool = false } = .{};
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const flag = args[i];
        if (std.mem.eql(u8, flag, "--quiet")) {
            o.log = false;
            continue;
        }
        const known = for ([_][]const u8{ "--host", "--port", "--model", "--pin", "--backend", "--timeout", "--memory" }) |k| {
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
        } else if (std.mem.eql(u8, flag, "--timeout")) {
            if (seen.timeout) return error.DuplicateOption;
            seen.timeout = true;
            o.timeout = std.fmt.parseInt(u32, value, 10) catch {
                diag.set("--timeout takes seconds, a whole number, not {s}", .{value});
                return error.InvalidOptionValue;
            };
        } else if (std.mem.eql(u8, flag, "--model")) {
            try o.models.append(arena, value);
        } else if (std.mem.eql(u8, flag, "--pin")) {
            if (o.pins.items.len == config.Pins.max) {
                diag.set("at most {d} models with --pin", .{config.Pins.max});
                return error.InvalidOptionValue;
            }
            try o.pins.append(arena, value);
        } else if (std.mem.eql(u8, flag, "--memory")) {
            if (seen.memory) return error.DuplicateOption;
            seen.memory = true;
            const gib = std.fmt.parseInt(u64, value, 10) catch 0;
            if (gib == 0 or gib > 4096) {
                diag.set("--memory takes whole GiB, 1 to 4096, not {s}", .{value});
                return error.InvalidOptionValue;
            }
            o.memory_bytes = gib << 30;
        } else {
            if (o.backend != null) return error.DuplicateOption;
            o.backend = std.meta.stringToEnum(inference.decide.Backend, value) orelse {
                diag.set("--backend takes cpu or metal, not {s}", .{value});
                return error.InvalidOptionValue;
            };
        }
    }
    if (o.timeout == 0) {
        diag.set("the wait limit (--timeout or serve.timeout) is at least 1 s", .{});
        return error.InvalidOptionValue;
    }
    if (o.pins.items.len == 0) try o.pins.appendSlice(arena, configured.pin.names);
    return o;
}

/// What opens at start: the `--model` names, then each pin not among them.
pub fn opened(arena: std.mem.Allocator, options: Options) ![]const []const u8 {
    var all: std.ArrayList([]const u8) = .empty;
    try all.appendSlice(arena, options.models.items);
    for (options.pins.items) |pin| {
        for (all.items) |name| {
            if (std.mem.eql(u8, name, pin)) break;
        } else try all.append(arena, pin);
    }
    return all.items;
}

/// The `--model` names sorted by the service that opens them, in the order
/// given; `arena` owns the lists. More of a kind than its service keeps
/// open is `InvalidOptionValue`.
pub const Named = struct {
    decision: std.ArrayList([]const u8) = .empty,
    language: ?[]const u8 = null,
    embedding: ?[]const u8 = null,
    speech: ?[]const u8 = null,
    transcription: ?[]const u8 = null,
};

pub fn byKind(arena: std.mem.Allocator, io: std.Io, registry: config.Models, names: []const []const u8, diag: *config.Diagnostic) !Named {
    const capacity = @import("decisions/pool.zig").capacity;
    var named: Named = .{};
    for (names) |name| switch (servedKind(io, registry, name)) {
        .decision => {
            if (named.decision.items.len == capacity) {
                diag.set("at most {d} decision models with --model (the server keeps {d} open)", .{ capacity, capacity });
                return error.InvalidOptionValue;
            }
            try named.decision.append(arena, name);
        },
        .generation => {
            if (named.language) |first| {
                diag.set("{s} and {s} are both language models; the server keeps one open", .{ first, name });
                return error.InvalidOptionValue;
            }
            named.language = name;
        },
        .embedding => {
            if (named.embedding) |first| {
                diag.set("{s} and {s} are both embedding models; the server keeps one open", .{ first, name });
                return error.InvalidOptionValue;
            }
            named.embedding = name;
        },
        .speech => {
            if (named.speech) |first| {
                diag.set("{s} and {s} are both speech models; the server keeps one open", .{ first, name });
                return error.InvalidOptionValue;
            }
            named.speech = name;
        },
        .transcription => {
            if (named.transcription) |first| {
                diag.set("{s} and {s} are both transcription models; the server keeps one open", .{ first, name });
                return error.InvalidOptionValue;
            }
            named.transcription = name;
        },
    };
    return named;
}

fn printOpened(out: *std.Io.Writer, sty: style.Style, io: std.Io, name: []const u8, kind: []const u8, started: std.Io.Timestamp) !void {
    const ms = @as(f64, @floatFromInt(started.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / std.time.ns_per_ms;
    try out.print("{s}opened{s} {s} {s}({s}, {d:.0} ms){s}\n", .{ sty.on(.success), sty.off(), name, sty.on(.dim), kind, ms, sty.off() });
}

/// `catalog.kindOf`, and a directory is a decision checkpoint, as
/// `nuclis decide --model` takes one.
fn servedKind(io: std.Io, registry: config.Models, name: []const u8) catalog.ModelKind {
    const kind = catalog.kindOf(registry, name);
    if (kind != .generation or catalog.find(name) != null) return kind;
    const stat = std.Io.Dir.cwd().statFile(io, name, .{}) catch return kind;
    return if (stat.kind == .directory) .decision else kind;
}

/// The address `--host` names: an IP literal, or `localhost`.
fn address(host: []const u8, port: u16, diag: *config.Diagnostic) !std.Io.net.IpAddress {
    if (std.mem.eql(u8, host, "localhost")) return .{ .ip4 = .loopback(port) };
    return std.Io.net.IpAddress.parse(host, port) catch {
        diag.set("the host (--host or serve.host) is an IP address (127.0.0.1, ::1, 0.0.0.0) or localhost, not {s}", .{host});
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
    /// Every open model, decision and language, counts against it.
    memory: memory.Budget,
    decisions: decisions.Service,
    chat: chat.Service,
    embeddings: embeddings.Service,
    speech: speech.Service,
    transcription: transcription.Service,
    backend: inference.decide.Backend,
    version: []const u8,
    active: std.atomic.Value(u32) = .init(0),
    log: ?*log_mod.Log = null,
    /// Set when the accept task returns, so the main task stops waiting.
    accept_ended: std.atomic.Value(bool) = .init(false),

    fn register(self: *Server) !void {
        try self.decisions.register(self.gpa, &self.router, &self.listing);
        try self.chat.register(self.gpa, &self.router, &self.listing);
        try self.embeddings.register(self.gpa, &self.router, &self.listing);
        try self.speech.register(self.gpa, &self.router, &self.listing);
        try self.transcription.register(self.gpa, &self.router, &self.listing);
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
        const language = self.chat.language.openName(io, arena) catch null;
        const embedding = self.embeddings.pool.openName(io, arena) catch null;
        const voice = self.speech.slot.openName(io, arena) catch null;
        const ears = self.transcription.slot.openName(io, arena) catch null;
        const resident = self.memory.snapshot(io, arena) catch &.{};
        const web_state = webState(self.chat.web, arena);
        var out: std.Io.Writer.Allocating = .init(arena);
        writeHealth(&out.writer, self.version, self.backend, open, .{ .language = language, .embedding = embedding, .speech = voice, .transcription = ears }, web_state, .{ .limit = self.memory.limit, .models = resident }, stats, batching, self.embeddings.batcher.stats(io), self.speech.stats(io), self.active.load(.monotonic)) catch
            return .fromError(arena, .init(.internal_server_error, "internal", "out of memory"));
        return .{ .body = out.written() };
    }

    /// Decision, embedding, or speech work queued or running on the GPU.
    fn busy(self: *Server, io: std.Io) bool {
        const queue = self.executor.stats(io);
        const voice = self.speech.stats(io);
        return queue.queued > 0 or queue.running or self.decisions.batcher.stats(io).waiting > 0 or self.embeddings.batcher.stats(io).waiting > 0 or voice.waiting > 0 or voice.speaking;
    }

    /// The accept task: one connection task per accepted stream until the
    /// task is canceled (Ctrl-C) or accepting fails.
    fn accept(self: *Server, io: std.Io, listener: *std.Io.net.Server, group: *std.Io.Group, diag: *config.Diagnostic) !void {
        defer self.accept_ended.store(true, .release);
        while (true) {
            const stream = listener.accept(io) catch |err| switch (err) {
                error.Canceled => return err,
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
            if (self.active.fetchAdd(1, .monotonic) >= limits.connections) {
                refuse(io, stream, &self.active, self.log);
                continue;
            }
            group.concurrent(io, connection, .{ self, io, stream }) catch refuse(io, stream, &self.active, self.log);
        }
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
        http.serve(self.gpa, io, &reader.interface, &writer.interface, self.router.handler(), limits.transport, self.log, .{ .handle = stream.socket.handle });
    }
};

const Memory = struct { limit: u64, models: []const memory.Budget.Model };
/// The open model of each kind the server keeps one of.
const Open = struct { language: ?[]const u8 = null, embedding: ?[]const u8 = null, speech: ?[]const u8 = null, transcription: ?[]const u8 = null };

/// The hosted web tools as health reports them: `off` without `serve.web`,
/// else `ok` when the search service answers and any provider is `ok`,
/// `down` otherwise (its own health call, at most 5 s).
fn webState(service: ?*web.Service, arena: std.mem.Allocator) []const u8 {
    const s = service orelse return "off";
    var reply = s.health(arena, null) catch return "down";
    defer reply.deinit();
    return switch (reply.outcome) {
        .ok => |h| if (h.ok()) "ok" else "down",
        .failed => "down",
    };
}

fn writeHealth(out: *std.Io.Writer, version: []const u8, backend: inference.decide.Backend, open: []const []const u8, singles: Open, web_state: []const u8, budget: Memory, stats: gpu.Stats, batching: batcher.Stats, embedding_batching: embedding_batcher.Stats, speaking: speech.Stats, connections: u32) !void {
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
    try s.objectField("language");
    try s.write(singles.language);
    try s.objectField("embedding");
    try s.write(singles.embedding);
    try s.objectField("speech");
    try s.write(singles.speech);
    try s.objectField("transcription");
    try s.write(singles.transcription);
    try s.objectField("web");
    try s.write(web_state);
    try s.objectField("memory");
    try s.beginObject();
    try s.objectField("limit");
    try s.write(budget.limit);
    var resident: u64 = 0;
    for (budget.models) |m| resident += m.bytes;
    try s.objectField("resident");
    try s.write(resident);
    try s.objectField("models");
    try s.beginArray();
    for (budget.models) |m| {
        try s.beginObject();
        try s.objectField("name");
        try s.write(m.name);
        try s.objectField("kind");
        try s.write(@tagName(m.kind));
        try s.objectField("bytes");
        try s.write(m.bytes);
        try s.objectField("pinned");
        try s.write(m.pinned);
        try s.endObject();
    }
    try s.endArray();
    try s.endObject();
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
    try s.objectField("embeddings");
    try s.beginObject();
    try s.objectField("waiting");
    try s.write(embedding_batching.waiting);
    try s.objectField("passes");
    try s.write(embedding_batching.passes);
    try s.objectField("requests");
    try s.write(embedding_batching.jobs);
    try s.endObject();
    try s.objectField("speech_requests");
    try s.beginObject();
    try s.objectField("waiting");
    try s.write(speaking.waiting);
    try s.objectField("speaking");
    try s.write(speaking.speaking);
    try s.objectField("answered");
    try s.write(speaking.requests);
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
    /// The configuration, borrowed for the serve: the chat service resolves
    /// a language model's settings from it as `nuclis chat` does.
    loaded: *const config.Loaded,
    registry: config.Models,
    default_model: []const u8,
    version: []const u8,
};

/// Serves until Ctrl-C or a failed accept. Prints where it listens, what it
/// opened, and the shutdown to `out`. After Ctrl-C it accepts nothing new,
/// waits up to `limits.drain_seconds` for queued decisions, closes the
/// connections, and returns; a second Ctrl-C ends the process at once.
pub fn serve(gpa: std.mem.Allocator, io: std.Io, context: Context, options: Options, out: *std.Io.Writer, sty: style.Style, diag: *config.Diagnostic) !void {
    const backend = options.backend orelse inference.decide.default_backend;
    const listen_address = try address(options.host, options.port, diag);
    var server: Server = .{
        .gpa = gpa,
        .decisions = .init(gpa, undefined, backend, context.root, context.registry, context.default_model),
        .chat = .init(gpa, undefined, context.loaded, context.root, @as(u64, options.timeout) * std.time.ns_per_s),
        .embeddings = .init(gpa, undefined, switch (backend) {
            .cpu => .cpu,
            .metal => .metal,
        }, context.root, context.registry, context.loaded.config.embed.model, @as(u64, options.timeout) * std.time.ns_per_s),
        .speech = .init(gpa, undefined, backend == .metal, context.root, context.registry, context.loaded.config.speech, @as(u64, options.timeout) * std.time.ns_per_s),
        .transcription = .init(gpa, undefined, backend == .metal, context.root, context.registry, context.loaded.config.transcribe.model),
        .memory = .init(gpa, options.memory_bytes orelse memory.defaultLimit(io)),
        .backend = backend,
        .version = context.version,
    };
    server.decisions.executor = &server.executor;
    server.chat.executor = &server.executor;
    server.embeddings.executor = &server.executor;
    server.speech.executor = &server.executor;
    server.transcription.executor = &server.executor;
    server.embeddings.pool.budget = &server.memory;
    server.decisions.pool.budget = &server.memory;
    server.chat.language.budget = &server.memory;
    server.speech.slot.budget = &server.memory;
    server.transcription.slot.budget = &server.memory;
    try server.memory.keep(options.pins.items);
    server.decisions.timeout_ns = @as(u64, options.timeout) * std.time.ns_per_s;
    server.decisions.bind();
    server.embeddings.bind();
    // The hosted web tools exist only when the configuration opts in (`serve.web`).
    var web_service: ?web.Service = if (context.loaded.config.serve.web) .init(gpa, io, context.loaded.config.web.url.?, web.user_agent) else null;
    defer if (web_service) |*service| service.deinit();
    if (web_service) |*service| {
        // For the start line only: requests may ask for web_context either
        // way, and a service without it answers `not_available`.
        try service.probe(gpa);
        server.chat.web = service;
    }
    defer server.deinit();
    try server.register();

    var worker = try io.concurrent(gpu.Executor.run, .{ &server.executor, io });
    defer {
        server.executor.stop(io);
        worker.await(io);
        server.decisions.deinit(io);
        server.chat.deinit(io);
        server.embeddings.deinit(io);
        server.speech.deinit(io);
        server.transcription.deinit(io);
        server.memory.deinit();
    }

    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    // Only what --model and the pins name opens here; every other model
    // opens on its first request.
    const named = try byKind(arena_state.allocator(), io, context.registry, try opened(arena_state.allocator(), options), diag);
    for (named.decision.items) |name| {
        const started = std.Io.Clock.awake.now(io);
        try server.decisions.preload(arena_state.allocator(), io, name, diag);
        try printOpened(out, sty, io, name, "decision", started);
    }
    if (named.language) |name| {
        const started = std.Io.Clock.awake.now(io);
        server.chat.preload(io, name) catch |err| {
            diag.set("the language model {s} did not open: {s}", .{ name, @errorName(err) });
            return err;
        };
        try printOpened(out, sty, io, name, "chat", started);
    }
    if (named.embedding) |name| {
        const started = std.Io.Clock.awake.now(io);
        try server.embeddings.preload(arena_state.allocator(), io, name, diag);
        try printOpened(out, sty, io, name, "embedding", started);
    }
    if (named.speech) |name| {
        const started = std.Io.Clock.awake.now(io);
        try server.speech.preload(arena_state.allocator(), io, name, diag);
        try printOpened(out, sty, io, name, "speech", started);
    }
    if (named.transcription) |name| {
        const started = std.Io.Clock.awake.now(io);
        try server.transcription.preload(arena_state.allocator(), io, name, diag);
        try printOpened(out, sty, io, name, "transcription", started);
    }
    if (options.pins.items.len > 0) {
        const pins = try std.mem.join(arena_state.allocator(), ", ", options.pins.items);
        try out.print("{s}pinned:{s} {s} {s}(never closed to make room){s}\n", .{ sty.on(.label), sty.off(), pins, sty.on(.dim), sty.off() });
    }

    var log: log_mod.Log = .init(out, sty, io);
    if (options.log) server.log = &log;

    var listener = listen_address.listen(io, .{ .reuse_address = true, .kernel_backlog = 128 }) catch |err| {
        diag.set("cannot listen on {f}: {s}", .{ listen_address, @errorName(err) });
        return err;
    };
    defer listener.deinit(io);
    if (!isLoopback(listen_address))
        try out.print("{s}warning:{s} listening beyond this machine ({s}); the API has no authentication\n", .{ sty.on(.warning), sty.off(), options.host });
    if (web_service) |*service| try out.print("{s}web tools:{s} {s} run here for requests that ask, through {s}\n", .{ sty.on(.label), sty.off(), if (service.context_offered) "web_search, web_fetch, and web_context" else "web_search and web_fetch", service.base });
    try out.print("{s}nuclis serve{s} listening on {s}http://{f}/v1{s} {s}(backend {s}, default decision model {s}, models within {d:.1} GiB{s}; Ctrl-C stops){s}\n", .{ sty.on(.header), sty.off(), sty.on(.code), listen_address, sty.off(), sty.on(.dim), @tagName(backend), context.default_model, @as(f64, @floatFromInt(server.memory.limit)) / (1 << 30), if (options.log) "" else ", quiet", sty.off() });
    try out.flush();

    var group: std.Io.Group = .init;
    defer group.cancel(io);
    interrupt.install();
    var accepting = try io.concurrent(Server.accept, .{ &server, io, &listener, &group, diag });
    // The signal handler can only set a flag, so the main task polls it.
    while (!interrupt.requested() and !server.accept_ended.load(.acquire)) {
        io.sleep(.fromMilliseconds(100), .awake) catch break;
    }
    accepting.cancel(io) catch |err| switch (err) {
        error.Canceled => {},
        else => return err,
    };

    try out.print("{s}stopping:{s} no new connections; finishing queued decisions and embeddings {s}(up to {d} s; Ctrl-C again ends now){s}\n", .{ sty.on(.warning), sty.off(), sty.on(.dim), limits.drain_seconds, sty.off() });
    try out.flush();
    const drain_until = std.Io.Clock.awake.now(io).addDuration(.fromSeconds(limits.drain_seconds));
    while (server.busy(io) and std.Io.Clock.awake.now(io).compare(.lt, drain_until)) {
        io.sleep(.fromMilliseconds(50), .awake) catch break;
    }
    const abandoned = server.busy(io);
    // Wakes idle keep-alive connections; a request already taken by a batch
    // finishes first (the batcher waits through one cancelation).
    group.cancel(io);
    try out.print("{s}stopped{s}{s}\n", .{ sty.on(.success), sty.off(), if (abandoned) " (queued work abandoned after the drain limit)" else "" });
    try out.flush();
}

/// Answers `busy` and closes: over the connection limit, or no task to run it.
fn refuse(io: std.Io, stream: std.Io.net.Stream, active: *std.atomic.Value(u32), log: ?*log_mod.Log) void {
    defer {
        stream.close(io);
        _ = active.fetchSub(1, .monotonic);
    }
    var buffer: [512]u8 = undefined;
    var writer = stream.writer(io, &buffer);
    var body_buffer: [256]u8 = undefined;
    var body: std.Io.Writer = .fixed(&body_buffer);
    const refusal: errors.ApiError = .init(http.overloaded, "busy", "overloaded: too many connections; retry shortly");
    errors.write(&body, refusal) catch return;
    http.writeClosing(&writer.interface, http.overloaded, body.buffered()) catch {};
    if (log) |l| l.request(io, .{ .method = null, .path = "-", .status = http.overloaded, .duration_ns = 0, .bytes_out = body.buffered().len, .note = "busy: too many connections" });
}

test {
    // The streamed routes' worker-to-connection queue, tested on its own.
    _ = @import("pipe.zig");
    _ = @import("memory.zig");
    _ = chat;
    _ = embeddings;
    _ = embedding_batcher;
    _ = @import("embeddings/pool.zig");
    _ = @import("slot.zig");
    _ = @import("speech/wire.zig");
    _ = speech;
    _ = transcription;
}

test "--model names sort by kind, each within what its service keeps open" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const io = std.testing.io;
    var diag: config.Diagnostic = .{};
    const named = try byKind(arena, io, .{}, &.{ "embeddinggemma-2", "laya", "qwen3.8-27b", "clef-flash" }, &diag);
    try std.testing.expectEqual(@as(usize, 2), named.decision.items.len);
    try std.testing.expectEqualStrings("clef-flash", named.decision.items[1]);
    try std.testing.expectEqualStrings("qwen3.8-27b", named.language.?);
    try std.testing.expectEqualStrings("embeddinggemma-2", named.embedding.?);
    try std.testing.expectEqual(Named{}, try byKind(arena, io, .{}, &.{}, &diag));
    try std.testing.expectError(error.InvalidOptionValue, byKind(arena, io, .{}, &.{ "laya", "laya-multilingual", "clef-flash" }, &diag));
    try std.testing.expectError(error.InvalidOptionValue, byKind(arena, io, .{}, &.{ "qwen3.8-27b", "other.gguf" }, &diag));
    try std.testing.expectError(error.InvalidOptionValue, byKind(arena, io, .{}, &.{ "embeddinggemma-2", "embeddinggemma-2" }, &diag));
    // One of each kind together: the voice assistant's set.
    const voice = try byKind(arena, io, .{}, &.{ "gemma-4-e4b-qat", "qwen3-tts-1.7b", "nemotron-asr-0.6b" }, &diag);
    try std.testing.expectEqualStrings("gemma-4-e4b-qat", voice.language.?);
    try std.testing.expectEqualStrings("qwen3-tts-1.7b", voice.speech.?);
    try std.testing.expectEqualStrings("nemotron-asr-0.6b", voice.transcription.?);
    try std.testing.expectError(error.InvalidOptionValue, byKind(arena, io, .{}, &.{ "qwen3-tts-1.7b", "qwen3-tts-0.6b" }, &diag));
    try std.testing.expect(std.mem.indexOf(u8, diag.message(), "both speech models") != null);
}

test "pins open at start beside --model, once each; --pin replaces serve.pin" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diag: config.Diagnostic = .{};
    const configured: config.Config.Serve = .{ .pin = .{ .names = &.{ "gemma-4-e4b-qat", "qwen3-tts-1.7b" } } };
    const from_file = try parseArgs(arena, &.{ "--model", "qwen3-tts-1.7b", "--model", "laya" }, configured, &diag);
    try std.testing.expectEqual(@as(usize, 2), from_file.pins.items.len);
    const all = try opened(arena, from_file);
    try std.testing.expectEqual(@as(usize, 3), all.len);
    try std.testing.expectEqualStrings("gemma-4-e4b-qat", all[2]);
    const flagged = try parseArgs(arena, &.{ "--pin", "nemotron-asr-0.6b" }, configured, &diag);
    try std.testing.expectEqual(@as(usize, 1), flagged.pins.items.len);
    try std.testing.expectEqualStrings("nemotron-asr-0.6b", (try opened(arena, flagged))[0]);
    var many: [2 * (config.Pins.max + 1)][]const u8 = undefined;
    for (&many, 0..) |*word, i| word.* = if (i % 2 == 0) "--pin" else "a";
    try std.testing.expectError(error.InvalidOptionValue, parseArgs(arena, &many, .{}, &diag));
}

test "serve arguments" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diag: config.Diagnostic = .{};
    const configured: config.Config.Serve = .{};
    const o = try parseArgs(arena, &.{ "--port", "9000", "--model", "laya", "--model", "qwen3.8-27b", "--backend", "cpu", "--host", "0.0.0.0", "--quiet" }, configured, &diag);
    try std.testing.expectEqual(@as(?u64, 20 << 30), (try parseArgs(arena, &.{ "--memory", "20" }, configured, &diag)).memory_bytes);
    try std.testing.expectEqual(@as(?u64, 7), (try parseArgs(arena, &.{}, .{ .memory_bytes = 7 }, &diag)).memory_bytes);
    try std.testing.expect(!o.log);
    try std.testing.expectEqual(@as(u16, 9000), o.port);
    try std.testing.expectEqual(@as(usize, 2), o.models.items.len);
    try std.testing.expectEqual(inference.decide.Backend.cpu, o.backend.?);
    try std.testing.expectEqualStrings("0.0.0.0", o.host);
    const defaults = try parseArgs(arena, &.{}, configured, &diag);
    try std.testing.expectEqual(@as(u16, 9000), defaults.port);
    try std.testing.expectEqualStrings("127.0.0.1", defaults.host);
    try std.testing.expect(defaults.log);
    try std.testing.expectEqual(@as(u32, 300), defaults.timeout);
    try std.testing.expectEqual(@as(u32, 600), (try parseArgs(arena, &.{ "--timeout", "600" }, configured, &diag)).timeout);
    try std.testing.expectEqual(@as(u16, 9001), (try parseArgs(arena, &.{}, .{ .host = "::1", .port = 9001 }, &diag)).port);
    const cases = .{
        .{ &[_][]const u8{ "--port", "70000" }, error.InvalidOptionValue },
        .{ &[_][]const u8{ "--backend", "tpu" }, error.InvalidOptionValue },
        .{ &[_][]const u8{ "--port", "1", "--port", "2" }, error.DuplicateOption },
        .{ &[_][]const u8{"--json"}, error.UnknownOption },
        .{ &[_][]const u8{"--host"}, error.MissingOptionValue },
        .{ &[_][]const u8{ "--timeout", "0" }, error.InvalidOptionValue },
        .{ &[_][]const u8{ "--timeout", "1.5" }, error.InvalidOptionValue },
        .{ &[_][]const u8{ "--timeout", "5", "--timeout", "6" }, error.DuplicateOption },
        .{ &[_][]const u8{ "--chat-model", "a" }, error.UnknownOption },
        .{ &[_][]const u8{ "--memory", "0" }, error.InvalidOptionValue },
        .{ &[_][]const u8{ "--memory", "1.5" }, error.InvalidOptionValue },
    };
    inline for (cases) |case| try std.testing.expectError(case[1], parseArgs(arena, case[0], configured, &diag));
    try std.testing.expect(isLoopback(try address("localhost", 1, &diag)));
    try std.testing.expect(isLoopback(try address("::1", 1, &diag)));
    try std.testing.expect(!isLoopback(try address("0.0.0.0", 1, &diag)));
    try std.testing.expectError(error.InvalidOptionValue, address("example.com", 1, &diag));
}

test "health reports the queue and the open models" {
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    try writeHealth(&out.writer, "0.1.0-dev", .cpu, &.{"laya"}, .{ .language = "qwen3.8-27b", .embedding = "embeddinggemma-2", .speech = "qwen3-tts-1.7b" }, "off", .{ .limit = 100, .models = &.{ .{ .kind = .language, .name = "qwen3.8-27b", .bytes = 60, .pinned = true }, .{ .kind = .decision, .name = "laya", .bytes = 10 } } }, .{ .queued = 2, .running = true, .completed = 7 }, .{ .waiting = 5, .batches = 3, .jobs = 9 }, .{ .waiting = 1, .passes = 4, .jobs = 2 }, .{ .waiting = 2, .speaking = true, .requests = 11 }, 3);
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, out.written(), .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings("cpu", parsed.value.object.get("backend").?.string);
    try std.testing.expectEqual(@as(i64, 2), parsed.value.object.get("queue").?.object.get("queued").?.integer);
    try std.testing.expectEqualStrings("laya", parsed.value.object.get("loaded").?.array.items[0].string);
    try std.testing.expectEqualStrings("qwen3.8-27b", parsed.value.object.get("language").?.string);
    try std.testing.expectEqual(@as(i64, 70), parsed.value.object.get("memory").?.object.get("resident").?.integer);
    try std.testing.expectEqual(@as(i64, 5), parsed.value.object.get("decisions").?.object.get("waiting").?.integer);
    try std.testing.expectEqualStrings("embeddinggemma-2", parsed.value.object.get("embedding").?.string);
    try std.testing.expectEqualStrings("off", parsed.value.object.get("web").?.string);
    try std.testing.expectEqual(@as(i64, 4), parsed.value.object.get("embeddings").?.object.get("passes").?.integer);
    try std.testing.expectEqualStrings("qwen3-tts-1.7b", parsed.value.object.get("speech").?.string);
    try std.testing.expect(parsed.value.object.get("transcription").? == .null);
    const models_list = parsed.value.object.get("memory").?.object.get("models").?.array.items;
    try std.testing.expect(models_list[0].object.get("pinned").?.bool and !models_list[1].object.get("pinned").?.bool);
    const speaking = parsed.value.object.get("speech_requests").?.object;
    try std.testing.expectEqual(@as(i64, 2), speaking.get("waiting").?.integer);
    try std.testing.expect(speaking.get("speaking").?.bool);
    try std.testing.expectEqual(@as(i64, 11), speaking.get("answered").?.integer);
}
