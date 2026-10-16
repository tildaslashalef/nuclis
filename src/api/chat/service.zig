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
const paths = @import("../../paths.zig");
const http = @import("../http.zig");
const errors = @import("../errors.zig");
const gpu = @import("../gpu.zig");
const models = @import("../models.zig");
const router_mod = @import("../router.zig");
const completer_mod = @import("../../completer.zig");
const Pipe = @import("../pipe.zig").Pipe;
const wire = @import("wire.zig");
const language_mod = @import("model.zig");
const web = @import("../../web/root.zig");
const stash_mod = @import("stash.zig");

const ApiError = errors.ApiError;
const Profile = inference.profiles;

/// The route's body limit: eight base64 images fit.
pub const max_body = 32 << 20;
/// Completions one request may run to answer its hosted web calls; the last
/// runs without the web tools, so it answers from what it has.
pub const max_web_rounds = 6;

pub const Service = struct {
    executor: *gpu.Executor,
    language: language_mod.Language,
    /// The wait for the GPU before a request starts; `serve.timeout`.
    timeout_ns: u64,
    /// The search service behind the hosted web tools; null unless
    /// `serve.web`, and then requests may ask for them. Borrowed.
    web: ?*web.Service = null,
    /// The hosted exchanges kept for the clients' next requests; worker-only.
    stash: stash_mod.Stash,

    /// `loaded` and `root` are borrowed for the service's lifetime.
    pub fn init(gpa: std.mem.Allocator, executor: *gpu.Executor, loaded: *const config.Loaded, root: ?[]const u8, timeout_ns: u64) Service {
        return .{ .executor = executor, .language = .{ .gpa = gpa, .loaded = loaded, .root = root }, .timeout_ns = timeout_ns, .stash = .{ .gpa = gpa } };
    }

    /// After the executor's worker stopped.
    pub fn deinit(self: *Service, io: std.Io) void {
        self.language.close(io);
        self.stash.deinit();
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
        const parsed = wire.parseWith(arena, request.body, &problem, .{ .web = self.web != null }) catch |err| return switch (err) {
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
            const about = describe(arena, io, root, loaded.config.models, runnable.name);
            var details: std.json.ObjectMap = .empty;
            try details.put(arena, "kind", .{ .string = "language" });
            try details.put(arena, "name", .{ .string = about.name });
            try details.put(arena, "architecture", if (about.architecture) |a| .{ .string = a } else .null);
            try details.put(arena, "profile", .{ .string = @tagName(profile) });
            try details.put(arena, "quantization", if (about.quantization) |q| .{ .string = q } else .null);
            try details.put(arena, "size_bytes", .{ .integer = @intCast(about.size_bytes) });
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

/// What a client shows for a language model.
const About = struct {
    name: []const u8,
    architecture: ?[]const u8 = null,
    quantization: ?[]const u8 = null,
    size_bytes: u64 = 0,
};

/// A language model's name, architecture, quantization, and size: the
/// catalogue's when it pins the file, else its GGUF header's (the
/// quantization being the encoding that holds the most tensor bytes, the
/// name `general.name` or the file's).
fn describe(arena: std.mem.Allocator, io: std.Io, root: []const u8, registry: config.Models, name: []const u8) About {
    const path = paths.modelPath(arena, name, "", root, registry) catch return .{ .name = name };
    const size = if (std.Io.Dir.cwd().statFile(io, path, .{})) |stat| stat.size else |_| 0;
    if (pinned(arena, root, path) orelse catalog.find(name)) |entry| return .{
        .name = entry.title,
        .architecture = entry.architecture,
        .quantization = entry.quantization,
        .size_bytes = size,
    };
    const stem = std.fs.path.stem(path);
    var doc = inference.gguf.open(arena, io, path, .{}) catch return .{ .name = stem, .size_bytes = size };
    var best_id: ?u32 = null;
    var best_bytes: u64 = 0;
    var totals: std.AutoHashMapUnmanaged(u32, u64) = .empty;
    for (doc.tensors) |t| {
        const sum = totals.getOrPutValue(arena, t.encoding_id, 0) catch break;
        sum.value_ptr.* += t.bytes;
        if (sum.value_ptr.* > best_bytes) {
            best_bytes = sum.value_ptr.*;
            best_id = t.encoding_id;
        }
    }
    // A converter's leftover ("Hf") is no name: shorter than 4 is the file's.
    const general = doc.string("general.name");
    return .{
        .name = if (general) |g| if (g.len >= 4) g else stem else stem,
        .architecture = doc.string("general.architecture"),
        .quantization = if (best_id) |id| if (inference.encoding.layout(id)) |l| l.name else null else null,
        .size_bytes = size,
    };
}

/// The catalogue entry whose main file sits at `path`, read as
/// `<root>/models/<owner>/<repo>/<file>`.
fn pinned(arena: std.mem.Allocator, root: []const u8, path: []const u8) ?*const catalog.Entry {
    const dir = std.fs.path.join(arena, &.{ root, "models" }) catch return null;
    if (!std.mem.startsWith(u8, path, dir) or path.len <= dir.len + 1) return null;
    const relative = path[dir.len + 1 ..];
    const owner_end = std.mem.indexOfScalar(u8, relative, '/') orelse return null;
    const repo_end = owner_end + 1 + (std.mem.indexOfScalar(u8, relative[owner_end + 1 ..], '/') orelse return null);
    const match = catalog.findFile(relative[0..repo_end], relative[repo_end + 1 ..]) orelse return null;
    return if (match.role == .main) match.entry else null;
}

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
    /// Worker-only: this round's hosted calls (held back from the client),
    /// its text for the assistant message that carries them, every call in
    /// order, and the client's calls; the usage over every round; the
    /// hosted calls run; the text every round sent the client.
    hosted: std.ArrayList(Profile.ToolCall) = .empty,
    round_answer: std.ArrayList(u8) = .empty,
    round_reasoning: std.ArrayList(u8) = .empty,
    round_calls: std.ArrayList(RoundCall) = .empty,
    client_calls: usize = 0,
    usage: wire.Usage = .{ .prompt = 0, .cached = 0, .completion = 0, .reasoning = 0 },
    web_calls: usize = 0,
    answers: std.ArrayList(u8) = .empty,
    /// Worker-only, with a stash: the conversation as the client sent it
    /// and its digest, the exchanges put back into it, where this request's
    /// own hosted rounds start, and the results of hosted calls that ran
    /// beside the client's in the last round.
    client_messages: []Profile.Message = &.{},
    client_counts: []const usize = &.{},
    chain: stash_mod.Chain = .{},
    spliced: usize = 0,
    exchange_start: usize = 0,
    last_results: ?[]const []const u8 = null,

    const RoundCall = struct { hosted: bool, name: []const u8, arguments: []const u8 };

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
            self.fail(openFailure(self.arena, language, self.request.model orelse language.loaded.config.engine.model, err));
            return .done;
        };
        self.name = self.arena.dupe(u8, language.name.?) catch "";
        var sink: completer_mod.Sink = .{ .context = self, .call = route };
        if (self.pipe != null) {
            self.chunks = .{ .id = self.id, .created = self.created, .model = self.name, .include_usage = self.request.include_usage };
            self.emit(ChunkStart{});
        }
        const observer: inference.observer.Observer = .{ .context = self, .check = check, .progress = progress };
        if (self.service.web != null) self.restore() catch {
            self.fail(.init(.internal_server_error, "internal", "out of memory"));
            return .done;
        };
        // The metered searches are allowed per request.
        if (self.service.web) |service| service.beginTurn();
        var round: usize = 1;
        const ran = while (true) : (round += 1) {
            self.hosted.clearRetainingCapacity();
            self.round_answer.clearRetainingCapacity();
            self.round_reasoning.clearRetainingCapacity();
            self.round_calls.clearRetainingCapacity();
            self.client_calls = 0;
            const ran = self.complete(&sink, observer, round) catch |err| {
                self.fail(runFailure(self.arena, language, err));
                return .done;
            };
            const r = usageOf(ran);
            self.usage = .{ .prompt = self.usage.prompt + r.prompt, .cached = self.usage.cached + r.cached, .completion = self.usage.completion + r.completion, .reasoning = self.usage.reasoning + r.reasoning };
            if (ran.reply.outcome.stop != .eos or self.hosted.items.len == 0 or round == max_web_rounds) break ran;
            const results = self.runHosted() catch {
                self.fail(.init(.internal_server_error, "internal", "out of memory"));
                return .done;
            };
            // The client's own calls end the request; the hosted results
            // reach the model again through the stash.
            if (self.client_calls > 0) {
                self.last_results = results;
                break ran;
            }
            self.continueWith(results, round + 1 == max_web_rounds) catch {
                self.fail(.init(.internal_server_error, "internal", "out of memory"));
                return .done;
            };
        };
        const outcome = ran.reply.outcome;
        if (outcome.stop == .failure) {
            self.fail(.init(.internal_server_error, "model_failed", "the completion failed on the GPU"));
            return .done;
        }
        self.outcome = .{ .done = ran };
        // Only a cache: a request whose exchange cannot be kept still answers.
        if (self.web_calls > 0 and outcome.stop != .cancelled) self.record() catch {};
        const finish = wire.finishReason(outcome.stop, self.client_calls);
        const usage = self.usage;
        if (self.pipe != null and outcome.stop != .cancelled) self.emit(ChunkFinish{ .finish = finish, .usage = usage });
        self.note = self.arena.print("{s} {s}: {d} prompt ({d} reused), {d} generated, {s}{s}{s}{s}", .{
            self.name,
            @tagName(ran.effort),
            usage.prompt,
            usage.cached,
            usage.completion,
            if (outcome.stop == .cancelled) "cancelled" else @tagName(finish),
            if (self.pipe != null) ", streamed" else "",
            if (self.web_calls > 0) self.arena.print(", {d} web call{s} in {d} round{s}", .{ self.web_calls, if (self.web_calls == 1) "" else "s", round, if (round == 1) "" else "s" }) catch "" else "",
            if (self.spliced > 0) self.arena.print(", {d} web exchange{s} put back", .{ self.spliced, if (self.spliced == 1) "" else "s" }) catch "" else "",
        }) catch "";
        return .done;
    }

    /// The conversation with the stashed exchanges its assistant messages
    /// match put back (stash.zig); the client's own is kept for `complete`.
    fn restore(self: *Job) std.mem.Allocator.Error!void {
        self.client_messages = self.request.messages;
        self.client_counts = self.request.image_counts;
        const restored = try stash_mod.restore(&self.service.stash, self.arena, &self.request, nowSeconds(self.io));
        self.chain = restored.chain;
        self.spliced = restored.spliced;
        self.exchange_start = self.request.messages.len;
    }

    /// One completion. A first round that does not fit with the exchanges
    /// put back runs again on the client's conversation alone: a splice
    /// never fails a request that would succeed without it.
    fn complete(self: *Job, sink: *completer_mod.Sink, observer: inference.observer.Observer, round: usize) !language_mod.Language.Ran {
        const language = &self.service.language;
        return language.run(self.io, self.arena, self.request, sink, observer) catch |err| {
            const e: anyerror = err;
            const overflowed = e == error.ContextFull or e == error.LimitExceeded or e == error.InvalidConversation;
            if (round > 1 or self.spliced == 0 or !overflowed) return err;
            self.request.messages = self.client_messages;
            self.request.image_counts = self.client_counts;
            self.spliced = 0;
            self.exchange_start = self.client_messages.len;
            return language.run(self.io, self.arena, self.request, sink, observer);
        };
    }

    /// Stashes what the model saw beyond the client's conversation, keyed by
    /// that conversation and the message the client received.
    fn record(self: *Job) std.mem.Allocator.Error!void {
        const kept = try keptExchange(self.arena, self.request.messages, self.exchange_start, .{
            .calls = self.round_calls.items,
            .answer = self.round_answer.items,
            .reasoning = self.round_reasoning.items,
            .results = self.last_results,
        });
        const reply: Profile.Message = .{ .role = .assistant, .content = try wire.validUtf8(self.arena, self.answers.items), .tool_calls = kept.client };
        try stash_mod.record(&self.service.stash, self.arena, self.chain, reply, kept.exchange, nowSeconds(self.io));
    }

    /// Runs this round's hosted calls on the worker, reporting each as it
    /// finishes. The wait lets queued short GPU items run and stops for a
    /// client that left. Returns the text each call's tool message carries.
    fn runHosted(self: *Job) std.mem.Allocator.Error![]const []const u8 {
        const service = self.service.web.?;
        const texts = try self.arena.alloc([]const u8, self.hosted.items.len);
        const beat: web.Beat = .{ .context = self, .call = webBeat };
        for (self.hosted.items, texts) |call, *text| {
            const rendered = (try web.tools.run(service, self.arena, call.name, call.arguments, self.request.web.context, beat)).?;
            text.* = rendered.text;
            const report: wire.Collector.WebCall = .{
                .name = call.name,
                .arguments = call.arguments,
                .summary = if (rendered.is_error) rendered.text else rendered.summary orelse "",
                .is_error = rendered.is_error,
            };
            if (self.pipe != null) self.emit(report) else try self.collected.web_calls.append(self.arena, report);
            self.web_calls += 1;
        }
        return texts;
    }

    /// The conversation for the next round (`extendForHosted`).
    fn continueWith(self: *Job, texts: []const []const u8, last: bool) std.mem.Allocator.Error!void {
        try extendForHosted(self.arena, &self.request, self.hosted.items, self.round_answer.items, self.round_reasoning.items, texts, last);
    }

    /// The beat of a hosted call's wait, on the worker: short GPU items
    /// queued meanwhile run now; a client that left cancels the call.
    fn webBeat(context: *anyopaque) bool {
        const self: *Job = @ptrCast(@alignCast(context));
        self.service.executor.runShort(self.io);
        return self.cancel.load(.acquire);
    }

    /// Every event of a round: a hosted call is held back to run here, any
    /// other goes to the client, whole or streamed.
    fn route(context: *anyopaque, event: inference.events.Event) anyerror!void {
        const self: *Job = @ptrCast(@alignCast(context));
        switch (event) {
            .tool_call => |call| {
                const hosted = self.request.web.runs(call.name);
                const name = try self.arena.dupe(u8, call.name);
                const arguments = try self.arena.dupe(u8, call.arguments);
                try self.round_calls.append(self.arena, .{ .hosted = hosted, .name = name, .arguments = arguments });
                if (hosted) {
                    try self.hosted.append(self.arena, .{ .id = 0, .name = name, .arguments = arguments });
                    return;
                }
                self.client_calls += 1;
            },
            .answer => |text| {
                try self.round_answer.appendSlice(self.arena, text);
                try self.answers.appendSlice(self.arena, text);
            },
            .thinking => |text| try self.round_reasoning.appendSlice(self.arena, text),
            else => {},
        }
        if (self.pipe != null) self.emit(event) else try self.collected.send(event);
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
            wire.Collector.WebCall => self.chunks.webCall(&out.writer, scratch, piece),
            inference.events.Event => self.chunks.event(&out.writer, scratch, piece, if (piece == .tool_call) wire.newId(scratch, self.io, "call_") catch "call_0" else ""),
            else => @compileError("not a stream piece"),
        };
        encoded catch return;
        if (out.written().len == 0) return;
        self.pipe.?.write(self.io, out.written()) catch self.cancel.store(true, .release);
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
        // A step boundary: decisions queued meanwhile run now rather than
        // after the whole generation. Each model has its own Metal queue
        // and buffers, so they touch nothing this generation holds.
        self.service.executor.runShort(self.io);
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

/// A request's last round, as `keptExchange` reads it.
const LastRound = struct {
    calls: []const Job.RoundCall,
    answer: []const u8,
    reasoning: []const u8,
    /// The hosted calls' results when they ran beside the client's calls;
    /// null when the round's hosted calls never ran.
    results: ?[]const []const u8,
};

/// What a request's completions saw beyond the client's conversation: the
/// hosted rounds `extendForHosted` appended to `messages` from `start` on,
/// then the last round's assistant message (the client's calls with id 0)
/// and the results of the hosted calls in it that ran. `client` is the
/// client's calls as the response sent them.
fn keptExchange(arena: std.mem.Allocator, messages: []const Profile.Message, start: usize, last: LastRound) std.mem.Allocator.Error!struct { exchange: []const Profile.Message, client: []const Profile.ToolCall } {
    var exchange: std.ArrayList(Profile.Message) = .empty;
    try exchange.appendSlice(arena, messages[start..]);
    var next_id = nextId(messages);
    var calls: std.ArrayList(Profile.ToolCall) = .empty;
    var client: std.ArrayList(Profile.ToolCall) = .empty;
    for (last.calls) |call| {
        if (call.hosted and last.results == null) continue;
        try calls.append(arena, .{ .id = if (call.hosted) next_id else 0, .name = call.name, .arguments = call.arguments });
        if (call.hosted) next_id += 1 else try client.append(arena, .{ .id = 0, .name = try wire.validUtf8(arena, call.name), .arguments = try wire.validUtf8(arena, call.arguments) });
    }
    try exchange.append(arena, .{ .role = .assistant, .content = last.answer, .reasoning_content = last.reasoning, .tool_calls = calls.items });
    if (last.results) |texts| {
        var k: usize = 0;
        for (calls.items) |call| if (call.id != 0) {
            try exchange.append(arena, .{ .role = .tool, .content = texts[k], .tool_call_id = call.id });
            k += 1;
        };
    }
    return .{ .exchange = exchange.items, .client = client.items };
}

/// The first call id past every one `messages` uses.
fn nextId(messages: []const Profile.Message) u32 {
    var next_id: u32 = 1;
    for (messages) |message| {
        for (message.tool_calls) |call| next_id = @max(next_id, call.id + 1);
        if (message.tool_call_id) |id| next_id = @max(next_id, id + 1);
    }
    return next_id;
}

fn nowSeconds(io: std.Io) i64 {
    return std.Io.Clock.awake.now(io).toSeconds();
}

/// Extends `request` for the round after hosted calls: the round's
/// assistant message carrying `hosted` (numbered past every id the
/// conversation uses), then one tool message per result in `texts`. A
/// `last` round runs without the hosted definitions.
fn extendForHosted(arena: std.mem.Allocator, request: *wire.Request, hosted: []const Profile.ToolCall, answer: []const u8, reasoning: []const u8, texts: []const []const u8, last: bool) std.mem.Allocator.Error!void {
    const old = request.messages;
    const next_id = nextId(old);
    const calls = try arena.alloc(Profile.ToolCall, hosted.len);
    for (hosted, calls, 0..) |call, *numbered, i| numbered.* = .{ .id = next_id + @as(u32, @intCast(i)), .name = call.name, .arguments = call.arguments };
    const messages = try arena.alloc(Profile.Message, old.len + 1 + calls.len);
    @memcpy(messages[0..old.len], old);
    messages[old.len] = .{ .role = .assistant, .content = try arena.dupe(u8, answer), .reasoning_content = try arena.dupe(u8, reasoning), .tool_calls = calls };
    for (calls, texts, messages[old.len + 1 ..]) |call, text, *message| message.* = .{ .role = .tool, .content = text, .tool_call_id = call.id };
    const counts = try arena.alloc(usize, messages.len);
    @memcpy(counts[0..old.len], request.image_counts);
    @memset(counts[old.len..], 0);
    request.messages = messages;
    request.image_counts = counts;
    if (last) {
        var kept: std.ArrayList(Profile.ToolDefinition) = .empty;
        for (request.tools) |tool| if (!request.web.runs(tool.name)) try kept.append(arena, tool);
        request.tools = kept.items;
    }
}

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
    const finish = wire.finishReason(ran.reply.outcome.stop, job.client_calls);
    const call_ids = try arena.alloc([]const u8, job.collected.calls.items.len);
    for (call_ids) |*id| id.* = try wire.newId(arena, io, "call_");
    var out: std.Io.Writer.Allocating = .init(arena);
    try wire.writeResponse(&out.writer, arena, &job.collected, .{
        .id = job.id,
        .created = job.created,
        .model = job.name,
        .finish = finish,
        .usage = job.usage,
        .call_ids = call_ids,
    });
    return .{ .body = out.written(), .note = job.note };
}

fn gib(bytes: u64) f64 {
    return @as(f64, @floatFromInt(bytes)) / (1 << 30);
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

fn isEmbedding(language: *const language_mod.Language, name: []const u8) bool {
    if (language.loaded.config.models.find(name)) |entry| return entry.kind == .embedding;
    return catalog.findEmbedding(name) != null;
}

fn openFailure(arena: std.mem.Allocator, language: *const language_mod.Language, name: []const u8, err: anyerror) ApiError {
    return switch (err) {
        error.ModelNotFound => apiError(arena, .not_found, "model_not_found", "model", "{s} is not a pulled language model (`nuclis model ls` lists them, `GET /v1/models` the runnable ones)", .{name}),
        error.NotALanguageModel => if (isEmbedding(language, name))
            apiError(arena, .bad_request, "not_a_language_model", "model", "{s} is an embedding model; it answers no chat", .{name})
        else
            apiError(arena, .bad_request, "not_a_language_model", "model", "{s} is a decision model; POST /v1/decisions runs it", .{name}),
        error.ModelTooLarge => apiError(arena, .bad_request, "model_too_large", "model", "{s} needs about {d:.1} GiB (its weights and cache.memory_bytes) and the server's budget is {d:.1} GiB (serve.memory_bytes, --memory)", .{ name, gib(language.needed), gib(if (language.budget) |b| b.limit else 0) }),
        error.Pinned => apiError(arena, .service_unavailable, "busy", "model", "overloaded: {s} needs memory a running request holds; retry shortly", .{name}),
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
        error.ModelTooLarge, error.Pinned => apiError(arena, .bad_request, "model_too_large", "messages", "this model and its image projector need more memory than the server's budget allows (serve.memory_bytes, --memory)", .{}),
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
    _ = stash_mod;
}

test "a file the catalogue pins is found by its place in the models directory" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const qwen = catalog.find("qwen3.8-27b").?;
    const main = try std.fs.path.join(arena, &.{ "/r/models", qwen.repo, qwen.file });
    try std.testing.expectEqual(qwen, pinned(arena, "/r", main).?);
    const companion = try std.fs.path.join(arena, &.{ "/r/models", qwen.repo, qwen.companion(.mmproj).?.file });
    try std.testing.expect(pinned(arena, "/r", companion) == null);
    try std.testing.expect(pinned(arena, "/r", "/elsewhere/x.gguf") == null);
    try std.testing.expect(pinned(arena, "/r", "/r/models/owner/repo/unknown.gguf") == null);
}

test "a hosted round extends the conversation with its calls and results, numbered past the client's" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var problem: wire.Problem = .{};
    var request = try wire.parseWith(arena,
        \\{"messages": [
        \\  {"role": "user", "content": "look"},
        \\  {"role": "assistant", "content": null, "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read", "arguments": "{}"}}]},
        \\  {"role": "tool", "tool_call_id": "c1", "content": "data"}],
        \\ "tools": [{"type": "web_search"}, {"type": "function", "function": {"name": "read"}}]}
    , &problem, .{ .web = true });
    const hosted = [_]Profile.ToolCall{.{ .id = 0, .name = "web_search", .arguments = "{\"query\":\"zig\"}" }};
    try extendForHosted(arena, &request, &hosted, "Searching.", "why", &.{"1. Zig — https://ziglang.org/"}, false);
    try std.testing.expectEqual(@as(usize, 5), request.messages.len);
    try std.testing.expectEqual(@as(usize, 5), request.image_counts.len);
    const assistant = request.messages[3];
    try std.testing.expectEqual(Profile.Role.assistant, assistant.role);
    try std.testing.expectEqualStrings("Searching.", assistant.content);
    try std.testing.expectEqualStrings("why", assistant.reasoning_content);
    try std.testing.expectEqual(@as(u32, 2), assistant.tool_calls[0].id);
    try std.testing.expectEqual(@as(?u32, 2), request.messages[4].tool_call_id);
    try std.testing.expectEqualStrings("1. Zig — https://ziglang.org/", request.messages[4].content);
    try std.testing.expectEqual(@as(usize, 2), request.tools.len);

    try extendForHosted(arena, &request, &hosted, "", "", &.{"again"}, true);
    try std.testing.expectEqual(@as(u32, 3), request.messages[5].tool_calls[0].id);
    try std.testing.expectEqual(@as(usize, 1), request.tools.len);
    try std.testing.expectEqualStrings("read", request.tools[0].name);
}

test "a request's hosted rounds reach the client's next requests through the stash, turn after turn" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var stash: stash_mod.Stash = .{ .gpa = std.testing.allocator };
    defer stash.deinit();
    var problem: wire.Problem = .{};
    const tools =
        \\ "tools": [{"type": "web_search"}, {"type": "function", "function": {"name": "save_note"}}]}
    ;

    // Turn 1: a search round, then an answer.
    var first = try wire.parseWith(arena, "{\"messages\": [{\"role\": \"user\", \"content\": \"What is Zig?\"}]," ++ tools, &problem, .{ .web = true });
    const one = try stash_mod.restore(&stash, arena, &first, 0);
    const start = first.messages.len;
    const search = [_]Profile.ToolCall{.{ .id = 0, .name = "web_search", .arguments = "{\"query\":\"zig\"}" }};
    try extendForHosted(arena, &first, &search, "Searching. ", "", &.{"1. Zig — https://ziglang.org/"}, false);
    const answered = try keptExchange(arena, first.messages, start, .{ .calls = &.{}, .answer = "A language.", .reasoning = "", .results = null });
    try std.testing.expectEqual(@as(usize, 3), answered.exchange.len);
    try stash_mod.record(&stash, arena, one.chain, .{ .role = .assistant, .content = "Searching. A language." }, answered.exchange, 0);

    // Turn 2 resends the answer: the search comes back. This round fetches
    // and calls the client's save_note; a search queued last never ran.
    var second = try wire.parseWith(arena,
        \\{"messages": [{"role": "user", "content": "What is Zig?"},
        \\  {"role": "assistant", "content": "Searching. A language."},
        \\  {"role": "user", "content": "Save its homepage."}],
    ++ tools, &problem, .{ .web = true });
    const two = try stash_mod.restore(&stash, arena, &second, 1);
    try std.testing.expectEqual(@as(usize, 1), two.spliced);
    try std.testing.expectEqual(@as(usize, 5), second.messages.len);
    const calls = [_]Job.RoundCall{
        .{ .hosted = true, .name = "web_fetch", .arguments = "{\"url\":\"https://ziglang.org/\"}" },
        .{ .hosted = false, .name = "save_note", .arguments = "{\"url\":\"https://ziglang.org/\"}" },
    };
    const mixed = try keptExchange(arena, second.messages, second.messages.len, .{ .calls = &calls, .answer = "", .reasoning = "", .results = &.{"# Zig"} });
    try std.testing.expectEqual(@as(usize, 2), mixed.exchange.len);
    try std.testing.expectEqual(@as(usize, 1), mixed.client.len);
    try stash_mod.record(&stash, arena, two.chain, .{ .role = .assistant, .content = "", .tool_calls = mixed.client }, mixed.exchange, 1);

    // Turn 3 resends both turns: both exchanges come back, ids apart.
    var third = try wire.parseWith(arena,
        \\{"messages": [{"role": "user", "content": "What is Zig?"},
        \\  {"role": "assistant", "content": "Searching. A language."},
        \\  {"role": "user", "content": "Save its homepage."},
        \\  {"role": "assistant", "content": null, "tool_calls": [{"id": "call_n", "type": "function", "function": {"name": "save_note", "arguments": "{\"url\":\"https://ziglang.org/\"}"}}]},
        \\  {"role": "tool", "tool_call_id": "call_n", "content": "saved"}],
    ++ tools, &problem, .{ .web = true });
    const three = try stash_mod.restore(&stash, arena, &third, 2);
    try std.testing.expectEqual(@as(usize, 2), three.spliced);
    const m = third.messages;
    try std.testing.expectEqual(@as(usize, 8), m.len);
    try std.testing.expectEqualStrings("web_search", m[1].tool_calls[0].name);
    try std.testing.expectEqualStrings("A language.", m[3].content);
    try std.testing.expectEqualStrings("web_fetch", m[5].tool_calls[0].name);
    try std.testing.expectEqualStrings("# Zig", m[6].content);
    try std.testing.expectEqualStrings("saved", m[7].content);
    try std.testing.expectEqual(m[5].tool_calls[1].id, m[7].tool_call_id.?);
    try Profile.validate(arena, m, third.tools, .{});

    // A last round whose hosted calls never ran keeps only the client's.
    const unran = try keptExchange(arena, second.messages, second.messages.len, .{ .calls = &calls, .answer = "", .reasoning = "", .results = null });
    try std.testing.expectEqual(@as(usize, 1), unran.exchange.len);
    try std.testing.expectEqual(@as(usize, 1), unran.exchange[0].tool_calls.len);
    try std.testing.expectEqual(@as(u32, 0), unran.exchange[0].tool_calls[0].id);
}
