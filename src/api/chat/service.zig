//! The chat service: `POST /v1/chat/completions` and the language models for
//! `GET /v1/models`. A handler reads and validates the request on its
//! connection's task (`wire.zig`), then submits one GPU item that opens the
//! model if needed and completes the conversation (`model.zig`). A whole
//! response waits for it; a streamed one drains its chunks from a `Pipe`.
//! Either way the connection watches its client, and a client that leaves
//! cancels the run. docs/guide/api.md § Chat Completions.
const std = @import("std");
const inference = @import("inference");
const config = @import("../../config.zig");
const catalog = @import("../../catalog.zig");
const model_ls = @import("../../model.zig");
const http = @import("../http.zig");
const errors = @import("../errors.zig");
const gpu = @import("../gpu.zig");
const models = @import("../models.zig");
const router_mod = @import("../router.zig");
const completer_mod = @import("../../completer.zig");
const Pipe = @import("../pipe.zig").Pipe;
const wire = @import("wire.zig");
const language_mod = @import("model.zig");

const ApiError = errors.ApiError;

/// The route's body limit: eight base64 images fit.
pub const max_body = 32 << 20;

pub const Service = struct {
    executor: *gpu.Executor,
    language: language_mod.Language,
    /// The wait for the GPU before a request starts; `serve.timeout`.
    timeout_ns: u64,

    /// `loaded` and `root` are borrowed for the service's lifetime.
    pub fn init(gpa: std.mem.Allocator, executor: *gpu.Executor, loaded: *const config.Loaded, root: ?[]const u8, timeout_ns: u64) Service {
        return .{ .executor = executor, .language = .{ .gpa = gpa, .loaded = loaded, .root = root }, .timeout_ns = timeout_ns };
    }

    /// After the executor's worker stopped.
    pub fn deinit(self: *Service, io: std.Io) void {
        self.language.close(io);
    }

    pub fn register(self: *Service, gpa: std.mem.Allocator, router: *router_mod.Router, listing: *models.Models) !void {
        try router.addLimited(gpa, .POST, router_mod.prefix ++ "/chat/completions", .{ .context = self, .handle = handle }, max_body);
        try listing.add(gpa, .{ .context = self, .list = list });
    }

    /// Opens `name` on the worker before the first request.
    pub fn preload(self: *Service, io: std.Io, name: []const u8) !void {
        var opening: Opening = .{ .language = &self.language, .name = name };
        try self.executor.submit(io, &opening.item);
        // The queue is empty at start: the item starts at once.
        try self.executor.wait(io, &opening.item, std.Io.Clock.Timestamp.fromNow(io, .{ .raw = .fromSeconds(3600), .clock = .awake }));
        if (opening.failed) |err| return err;
    }

    fn handle(context: *anyopaque, arena: std.mem.Allocator, io: std.Io, request: http.Request) http.Response {
        const self: *Service = @ptrCast(@alignCast(context));
        const deadline = std.Io.Clock.Timestamp.fromNow(io, .{ .raw = .fromNanoseconds(self.timeout_ns), .clock = .awake });
        var problem: wire.Problem = .{};
        const parsed = wire.parse(arena, request.body, &problem) catch |err| return switch (err) {
            error.Refused => refuse(arena, .bad_request, problem.code, problem.message, problem.param),
            error.OutOfMemory => refuse(arena, .internal_server_error, "internal", "out of memory", null),
        };
        const job = arena.create(Job) catch return refuse(arena, .internal_server_error, "internal", "out of memory", null);
        job.* = .{
            .service = self,
            .arena = arena,
            .request = parsed,
            .peer = request.peer,
            .deadline = deadline,
            .id = wire.newId(arena, io, "chatcmpl-") catch return refuse(arena, .internal_server_error, "internal", "out of memory", null),
            .created = std.Io.Timestamp.now(io, .real).toSeconds(),
            .collected = .{ .arena = arena },
        };
        if (parsed.stream) job.pipe = Pipe.init(self.language.gpa, pipe_capacity) catch
            return refuse(arena, .internal_server_error, "internal", "out of memory", null);
        self.executor.submit(io, &job.item) catch |err| {
            if (job.pipe) |*pipe| pipe.deinit(self.language.gpa);
            return switch (err) {
                error.Busy => refuse(arena, http.overloaded, "busy", "overloaded: too many requests wait for the GPU; retry shortly", null),
                error.Stopped => refuse(arena, .service_unavailable, "shutting_down", "the server is stopping", null),
            };
        };
        if (parsed.stream) return .{
            .body = "",
            .content_type = "text/event-stream",
            .stream = .{ .context = job, .write = Job.writeStream, .note = Job.streamNote },
        };
        switch (job.waitWhole(io)) {
            .done => {},
            .timed_out => return refuse(arena, http.overloaded, "timeout", timeoutMessage(arena, self.timeout_ns), null),
            .gone => return refuse(arena, .bad_request, "cancelled", "the client closed the connection", null),
        }
        const ran = switch (job.outcome) {
            .pending => unreachable,
            .failed => |e| return .fromError(arena, e),
            .done => |d| d,
        };
        return respond(arena, io, job, ran) catch refuse(arena, .internal_server_error, "internal", "out of memory", null);
    }

    fn list(context: *anyopaque, arena: std.mem.Allocator, io: std.Io, out: *std.ArrayList(models.Entry)) anyerror!void {
        const self: *Service = @ptrCast(@alignCast(context));
        const root = self.language.root orelse return;
        const loaded = self.language.loaded;
        const listing = try model_ls.list(arena, io, root, loaded.config.models);
        for (try model_ls.runnableModels(arena, listing, loaded.config.models)) |runnable| {
            const settings = config.resolve(loaded, runnable.name, .{}, .agent);
            const profile = settings.forced_profile orelse settings.profile;
            var efforts: std.json.Array = .init(arena);
            for (profile.efforts()) |e| try efforts.append(.{ .string = @tagName(e) });
            const projector = if (settings.entry) |entry| entry.mmproj != null else if (catalog.find(runnable.name)) |entry| entry.companion(.mmproj) != null else false;
            var details: std.json.ObjectMap = .empty;
            try details.put(arena, "kind", .{ .string = "language" });
            try details.put(arena, "present", .{ .bool = true });
            try details.put(arena, "loaded", .{ .bool = self.language.isOpen(io, runnable.name) });
            try details.put(arena, "default", .{ .bool = std.mem.eql(u8, runnable.name, loaded.config.engine.model) });
            try details.put(arena, "context_length", .{ .integer = @intCast(settings.ctx_size) });
            try details.put(arena, "images", .{ .bool = projector });
            try details.put(arena, "efforts", .{ .array = efforts });
            try details.put(arena, "detail", .{ .string = runnable.detail });
            try out.append(arena, .{ .id = runnable.name, .owned_by = "nuclis", .details = .{ .object = details } });
        }
    }
};

/// Bytes a streamed response may hold between the worker and the socket.
const pipe_capacity = 256 * 1024;
/// How often a waiting request looks at its client, and how long a stream
/// may stay silent before a keepalive comment.
const poll_ms = 1000;
const keepalive_ms = 10_000;

/// One request on the worker: open the model if needed, complete, and
/// deliver the events whole (`collected`) or as server-sent events
/// (`pipe`). It lives in the connection's arena, which the worker uses
/// while the connection task only waits or drains the pipe.
const Job = struct {
    item: gpu.Item = .{ .run = run },
    service: *Service,
    arena: std.mem.Allocator,
    request: wire.Request,
    peer: ?http.Peer,
    /// The latest start: a request still queued then is answered `timeout`.
    deadline: std.Io.Clock.Timestamp,
    id: []const u8,
    created: i64,
    collected: wire.Collector,
    pipe: ?Pipe = null,
    /// Set by the connection when its client is gone; the engine stops at
    /// the next layer boundary.
    cancel: std.atomic.Value(bool) = .init(false),
    started: std.atomic.Value(bool) = .init(false),
    /// The engine's progress, for keepalive comments: 0 none yet, 1
    /// prefill, 2 decode.
    phase: std.atomic.Value(u8) = .init(0),
    position: std.atomic.Value(usize) = .init(0),
    target: std.atomic.Value(usize) = .init(0),
    outcome: union(enum) { pending, done: language_mod.Language.Ran, failed: ApiError } = .pending,
    /// The model that ran, in `arena`.
    name: []const u8 = "",
    /// The streamed response's log line, written by the worker.
    note: []const u8 = "",
    /// Worker-only, while it runs: the chunk encoder and its scratch.
    io: std.Io = undefined,
    chunks: wire.ChunkWriter = undefined,
    scratch: std.heap.ArenaAllocator = undefined,

    fn run(item: *gpu.Item, io: std.Io) gpu.Item.After {
        const self: *Job = @alignCast(@fieldParentPtr("item", item));
        self.started.store(true, .release);
        self.io = io;
        self.scratch = .init(self.service.language.gpa);
        defer self.scratch.deinit();
        defer if (self.pipe) |*pipe| pipe.close(io);
        const language = &self.service.language;
        if (self.cancel.load(.acquire)) {
            self.outcome = .{ .failed = .init(.bad_request, "cancelled", "the client closed the connection") };
            return .done;
        }
        language.ensure(io, self.request.model) catch |err| {
            self.fail(openFailure(self.arena, self.request.model orelse language.loaded.config.engine.model, err));
            return .done;
        };
        self.name = self.arena.dupe(u8, language.name.?) catch "";
        var sink: completer_mod.Sink = if (self.pipe != null) .{ .context = self, .call = streamEvent } else .{ .context = self, .call = collect };
        if (self.pipe != null) {
            self.chunks = .{ .id = self.id, .created = self.created, .model = self.name, .include_usage = self.request.include_usage };
            self.emit(ChunkStart{});
        }
        const observer: inference.observer.Observer = .{ .context = self, .check = check, .progress = progress };
        const ran = language.run(io, self.arena, self.request, &sink, observer) catch |err| {
            self.fail(runFailure(self.arena, language, err));
            return .done;
        };
        const outcome = ran.reply.outcome;
        if (outcome.stop == .failure) {
            self.fail(.init(.internal_server_error, "model_failed", "the completion failed on the GPU"));
            return .done;
        }
        self.outcome = .{ .done = ran };
        const finish = wire.finishReason(outcome.stop, ran.calls);
        const usage = usageOf(ran);
        if (self.pipe != null and outcome.stop != .cancelled) self.emit(ChunkFinish{ .finish = finish, .usage = usage });
        self.note = self.arena.print("{s} {s}: {d} prompt ({d} reused), {d} generated, {s}{s}", .{
            self.name,
            @tagName(ran.effort),
            usage.prompt,
            usage.cached,
            usage.completion,
            if (outcome.stop == .cancelled) "cancelled" else @tagName(finish),
            if (self.pipe != null) ", streamed" else "",
        }) catch "";
        return .done;
    }

    /// A failure: the response's error, or in a stream an error line.
    fn fail(self: *Job, err: ApiError) void {
        self.outcome = .{ .failed = err };
        self.note = http.noteOf(self.arena, err);
        if (self.pipe != null) self.emit(err);
    }

    const ChunkStart = struct {};
    const ChunkFinish = struct { finish: wire.FinishReason, usage: wire.Usage };

    /// Encodes one piece of the stream and hands it to the connection. A
    /// consumer that has gone cancels the completion instead of failing
    /// it: the engine stops at its next layer boundary.
    fn emit(self: *Job, piece: anytype) void {
        if (self.cancel.load(.acquire)) return;
        defer _ = self.scratch.reset(.retain_capacity);
        const scratch = self.scratch.allocator();
        var out: std.Io.Writer.Allocating = .init(scratch);
        const encoded = switch (@TypeOf(piece)) {
            ChunkStart => self.chunks.start(&out.writer),
            ChunkFinish => self.chunks.finish(&out.writer, piece.finish, piece.usage),
            ApiError => wire.writeStreamError(&out.writer, piece),
            inference.events.Event => self.chunks.event(&out.writer, scratch, piece, if (piece == .tool_call) wire.newId(scratch, self.io, "call_") catch "call_0" else ""),
            else => @compileError("not a stream piece"),
        };
        encoded catch return;
        if (out.written().len == 0) return;
        self.pipe.?.write(self.io, out.written()) catch self.cancel.store(true, .release);
    }

    fn streamEvent(context: *anyopaque, event: inference.events.Event) anyerror!void {
        const self: *Job = @ptrCast(@alignCast(context));
        self.emit(event);
    }

    fn collect(context: *anyopaque, event: inference.events.Event) anyerror!void {
        const self: *Job = @ptrCast(@alignCast(context));
        try self.collected.send(event);
    }

    fn check(context: *anyopaque) anyerror!void {
        const self: *Job = @ptrCast(@alignCast(context));
        if (self.cancel.load(.acquire)) return error.Cancelled;
    }

    fn progress(context: *anyopaque, p: inference.observer.Progress) anyerror!void {
        const self: *Job = @ptrCast(@alignCast(context));
        self.position.store(p.position, .release);
        self.target.store(p.target, .release);
        self.phase.store(@as(u8, @backingInt(p.phase)) + 1, .release);
    }

    /// The connection's side of a whole response: wait, looking at the
    /// client every second. A client that has gone cancels the run (or
    /// withdraws it, not yet started); a request still queued at its
    /// deadline is withdrawn.
    fn waitWhole(self: *Job, io: std.Io) enum { done, timed_out, gone } {
        const executor = self.service.executor;
        while (!executor.poll(io, &self.item, fromNow(io, poll_ms))) {
            if (self.peer) |peer| if (peer.gone()) {
                self.cancel.store(true, .release);
                if (executor.withdraw(io, &self.item)) return .gone;
            };
            if (!self.started.load(.acquire) and pastDeadline(io, self.deadline) and executor.withdraw(io, &self.item)) return .timed_out;
        }
        return .done;
    }

    /// The connection's side of a stream: drain the pipe into the body,
    /// send a comment when it has been silent for `keepalive_ms`, and stop
    /// the run when the client goes. Returns once the worker has let go of
    /// the job.
    fn writeStream(context: *anyopaque, io: std.Io, body: *http.Body) bool {
        const self: *Job = @ptrCast(@alignCast(context));
        defer self.release(io);
        var buffer: [16 * 1024]u8 = undefined;
        var quiet_since = std.Io.Clock.awake.now(io);
        while (true) {
            if (self.peer) |peer| if (peer.gone()) return self.abandon(io);
            const taken = self.pipe.?.take(io, &buffer, fromNow(io, poll_ms)) catch return self.abandon(io);
            switch (taken) {
                .closed => return true,
                .data => |bytes| {
                    body.send(bytes) catch return self.abandon(io);
                    quiet_since = std.Io.Clock.awake.now(io);
                },
                .timeout => {
                    if (!self.started.load(.acquire) and pastDeadline(io, self.deadline) and self.service.executor.withdraw(io, &self.item)) {
                        // Never started: the error line is the connection's to write.
                        var line: [512]u8 = undefined;
                        var out: std.Io.Writer = .fixed(&line);
                        const err: ApiError = .init(http.overloaded, "timeout", timeoutMessage(self.arena, self.service.timeout_ns));
                        self.note = http.noteOf(self.arena, err);
                        wire.writeStreamError(&out, err) catch {};
                        body.send(out.buffered()) catch {};
                        return true;
                    }
                    if (quiet_since.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds() >= keepalive_ms) {
                        body.send(self.keepalive(&buffer)) catch return self.abandon(io);
                        quiet_since = std.Io.Clock.awake.now(io);
                    }
                },
            }
        }
    }

    /// An SSE comment saying where the request stands; parsers ignore it.
    fn keepalive(self: *Job, buffer: []u8) []const u8 {
        if (!self.started.load(.acquire)) return ": queued\n\n";
        return switch (self.phase.load(.acquire)) {
            1 => std.fmt.bufPrint(buffer, ": prefill {d}/{d}\n\n", .{ self.position.load(.acquire), self.target.load(.acquire) }) catch ": prefill\n\n",
            2 => std.fmt.bufPrint(buffer, ": generating {d}\n\n", .{self.position.load(.acquire)}) catch ": generating\n\n",
            else => ": starting\n\n",
        };
    }

    fn abandon(self: *Job, io: std.Io) bool {
        self.cancel.store(true, .release);
        self.pipe.?.abandon(io);
        if (self.note.len == 0) self.note = "cancelled: the client closed the connection";
        return false;
    }

    /// Waits until the worker can no longer touch the job (or takes it back
    /// unstarted), then frees the pipe.
    fn release(self: *Job, io: std.Io) void {
        const executor = self.service.executor;
        if (!executor.withdraw(io, &self.item)) {
            while (!executor.poll(io, &self.item, fromNow(io, poll_ms))) {}
        }
        self.pipe.?.deinit(self.service.language.gpa);
    }

    fn streamNote(context: *anyopaque) []const u8 {
        const self: *Job = @ptrCast(@alignCast(context));
        return self.note;
    }
};

fn fromNow(io: std.Io, ms: i64) std.Io.Clock.Timestamp {
    return std.Io.Clock.Timestamp.fromNow(io, .{ .raw = .fromMilliseconds(ms), .clock = .awake });
}

fn pastDeadline(io: std.Io, deadline: std.Io.Clock.Timestamp) bool {
    return std.Io.Clock.Timestamp.now(io, deadline.clock).compare(.gte, deadline);
}

fn timeoutMessage(arena: std.mem.Allocator, timeout_ns: u64) []const u8 {
    return arena.print("overloaded: waited {d} s for the GPU without starting (timeout)", .{timeout_ns / std.time.ns_per_s}) catch "timeout";
}

fn usageOf(ran: language_mod.Language.Ran) wire.Usage {
    const reply = ran.reply;
    return .{
        .prompt = reply.prompt_tokens,
        .cached = reply.reused_tokens,
        .completion = reply.outcome.timing.generated_tokens,
        .reasoning = reply.outcome.reasoning_tokens,
    };
}

const Opening = struct {
    item: gpu.Item = .{ .run = run },
    language: *language_mod.Language,
    name: []const u8,
    failed: ?anyerror = null,

    fn run(item: *gpu.Item, io: std.Io) gpu.Item.After {
        const self: *Opening = @alignCast(@fieldParentPtr("item", item));
        self.language.ensure(io, self.name) catch |err| {
            self.failed = err;
        };
        return .done;
    }
};

fn respond(arena: std.mem.Allocator, io: std.Io, job: *const Job, ran: language_mod.Language.Ran) !http.Response {
    const finish = wire.finishReason(ran.reply.outcome.stop, ran.calls);
    const call_ids = try arena.alloc([]const u8, job.collected.calls.items.len);
    for (call_ids) |*id| id.* = try wire.newId(arena, io, "call_");
    var out: std.Io.Writer.Allocating = .init(arena);
    try wire.writeResponse(&out.writer, arena, &job.collected, .{
        .id = job.id,
        .created = job.created,
        .model = job.name,
        .finish = finish,
        .usage = usageOf(ran),
        .call_ids = call_ids,
    });
    return .{ .body = out.written(), .note = job.note };
}

fn refuse(arena: std.mem.Allocator, status: std.http.Status, code: []const u8, message: []const u8, param: ?[]const u8) http.Response {
    var err: ApiError = .init(status, code, message);
    err.param = param;
    return .fromError(arena, err);
}

fn apiError(arena: std.mem.Allocator, status: std.http.Status, code: []const u8, param: ?[]const u8, comptime format: []const u8, args: anytype) ApiError {
    var err: ApiError = .init(status, code, arena.print(format, args) catch code);
    err.param = param;
    return err;
}

fn openFailure(arena: std.mem.Allocator, name: []const u8, err: anyerror) ApiError {
    return switch (err) {
        error.ModelNotFound => apiError(arena, .not_found, "model_not_found", "model", "{s} is not a pulled language model (`nuclis model ls` lists them, `GET /v1/models` the runnable ones)", .{name}),
        error.NotALanguageModel => apiError(arena, .bad_request, "not_a_language_model", "model", "{s} is a decision model; POST /v1/decisions runs it", .{name}),
        else => apiError(arena, .internal_server_error, "model_failed", "model", "{s} did not open: {s}", .{ name, @errorName(err) }),
    };
}

fn runFailure(arena: std.mem.Allocator, language: *language_mod.Language, err: anyerror) ApiError {
    return switch (err) {
        error.ContextFull => if (language.completer.overflow) |o|
            apiError(arena, .bad_request, "context_length_exceeded", "messages", "the conversation needs {d} tokens and the window holds {d}", .{ o.needed, o.capacity })
        else
            apiError(arena, .bad_request, "context_length_exceeded", "messages", "the conversation does not fit the window", .{}),
        error.ImagesUnsupported => apiError(arena, .bad_request, "images_unsupported", "messages", "this model has no image projector pulled; `nuclis model pull <name> --with mmproj` fetches it", .{}),
        error.InvalidSamplingOptions => apiError(arena, .bad_request, "invalid_request", null, "the sampling options are out of range for this model", .{}),
        error.ToolsUnsupported => apiError(arena, .bad_request, "invalid_request", "tools", "this model's prompt profile has no native tool calling", .{}),
        error.InvalidConversation, error.UnsupportedContent, error.LimitExceeded, error.InvalidUtf8 => apiError(arena, .bad_request, "invalid_request", "messages", "the model's template cannot render this conversation ({s})", .{@errorName(err)}),
        error.MalformedImage, error.UnsupportedImageFormat, error.ImageTooLarge => apiError(arena, .bad_request, "invalid_request", "messages", "an image could not be decoded ({s})", .{@errorName(err)}),
        error.OutOfMemory => apiError(arena, .internal_server_error, "internal", null, "out of memory", .{}),
        else => apiError(arena, .internal_server_error, "model_failed", null, "the completion failed: {s}", .{@errorName(err)}),
    };
}

test {
    _ = wire;
}
