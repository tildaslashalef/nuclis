//! The chat service: `POST /v1/chat/completions` and the language models for
//! `GET /v1/models`. A handler reads and validates the request on its
//! connection's task (`wire.zig`), then waits for one GPU item that opens the
//! model if needed and completes the conversation (`model.zig`); it never
//! touches a model or a socket itself. docs/guide/api.md § Chat Completions.
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
        var job: Job = .{ .language = &self.language, .arena = arena, .request = parsed, .collected = .{ .arena = arena } };
        self.executor.submit(io, &job.item) catch |err| return switch (err) {
            error.Busy => refuse(arena, http.overloaded, "busy", "overloaded: too many requests wait for the GPU; retry shortly", null),
            error.Stopped => refuse(arena, .service_unavailable, "shutting_down", "the server is stopping", null),
        };
        self.executor.wait(io, &job.item, deadline) catch
            return refuse(arena, http.overloaded, "timeout", arena.print("overloaded: waited {d} s for the GPU without starting", .{self.timeout_ns / std.time.ns_per_s}) catch "timeout", null);
        const ran = switch (job.outcome) {
            .pending => unreachable,
            .failed => |e| return .fromError(arena, e),
            .done => |d| d,
        };
        return respond(arena, io, &job, ran) catch refuse(arena, .internal_server_error, "internal", "out of memory", null);
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

/// One request's run on the worker: open the model if needed, complete.
const Job = struct {
    item: gpu.Item = .{ .run = run },
    language: *language_mod.Language,
    arena: std.mem.Allocator,
    request: wire.Request,
    collected: wire.Collector,
    outcome: union(enum) { pending, done: language_mod.Language.Ran, failed: ApiError } = .pending,
    /// The model that ran, in `arena`.
    name: []const u8 = "",

    fn run(item: *gpu.Item, io: std.Io) gpu.Item.After {
        const self: *Job = @alignCast(@fieldParentPtr("item", item));
        self.language.ensure(io, self.request.model) catch |err| {
            self.outcome = .{ .failed = openFailure(self.arena, self.request.model orelse self.language.loaded.config.engine.model, err) };
            return .done;
        };
        self.name = self.arena.dupe(u8, self.language.name.?) catch {
            self.outcome = .{ .failed = .init(.internal_server_error, "internal", "out of memory") };
            return .done;
        };
        const ran = self.language.run(io, self.arena, self.request, &self.collected) catch |err| {
            self.outcome = .{ .failed = runFailure(self.arena, self.language, err) };
            return .done;
        };
        self.outcome = if (ran.reply.outcome.stop == .failure)
            .{ .failed = .init(.internal_server_error, "model_failed", "the completion failed on the GPU") }
        else
            .{ .done = ran };
        return .done;
    }
};

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
    const reply = ran.reply;
    const outcome = reply.outcome;
    const finish = wire.finishReason(outcome.stop, job.collected.calls.items.len);
    const call_ids = try arena.alloc([]const u8, job.collected.calls.items.len);
    for (call_ids) |*id| id.* = try wire.newId(arena, io, "call_");
    const name = job.name;
    const usage: wire.Usage = .{
        .prompt = reply.prompt_tokens,
        .cached = reply.reused_tokens,
        .completion = outcome.timing.generated_tokens,
        .reasoning = outcome.reasoning_tokens,
    };
    var out: std.Io.Writer.Allocating = .init(arena);
    try wire.writeResponse(&out.writer, arena, &job.collected, .{
        .id = try wire.newId(arena, io, "chatcmpl-"),
        .created = std.Io.Timestamp.now(io, .real).toSeconds(),
        .model = name,
        .finish = finish,
        .usage = usage,
        .call_ids = call_ids,
    });
    return .{
        .body = out.written(),
        .note = try arena.print("{s} {s}: {d} prompt ({d} reused), {d} generated, {s}", .{ name, @tagName(ran.effort), usage.prompt, usage.cached, usage.completion, @tagName(finish) }),
    };
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
