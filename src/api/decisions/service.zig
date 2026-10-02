//! The decision service: `POST /v1/decisions` (the body `nuclis decide
//! --request` reads, answered with what `nuclis decide --json` writes),
//! `POST /v1/systemone` (Jev's single-state call), and the decision models
//! for `GET /v1/models`. A handler parses and validates on its connection's
//! thread, then waits for one GPU item that opens the model if needed and
//! answers; it never touches a socket or a model directly.
//! docs/reference/api.md § Decisions.
const std = @import("std");
const inference = @import("inference");
const config = @import("../../config.zig");
const wire = @import("../../decision/request.zig");
const response = @import("../../decision/response.zig");
const catalog = @import("../../decision/catalog.zig");
const http = @import("../http.zig");
const errors = @import("../errors.zig");
const gpu = @import("../gpu.zig");
const models = @import("../models.zig");
const router_mod = @import("../router.zig");
const pool_mod = @import("pool.zig");

const ApiError = errors.ApiError;

/// A request waits at most this long for the GPU (queue and run together
/// before it starts; a started pass is never cut).
pub const default_timeout_ns: u64 = 30 * std.time.ns_per_s;

pub const Service = struct {
    executor: *gpu.Executor,
    pool: pool_mod.Pool,
    /// The user root, for model names under `<root>/models`.
    root: ?[]const u8,
    registry: config.Models,
    /// `decide.model`: a request that names no model gets this one.
    default_model: []const u8,
    timeout_ns: u64 = default_timeout_ns,

    pub const Shape = enum { decisions, systemone };

    /// `registry` and `root` are borrowed for the service's lifetime.
    pub fn init(gpa: std.mem.Allocator, executor: *gpu.Executor, backend: inference.decide.Backend, root: ?[]const u8, registry: config.Models, default_model: []const u8) Service {
        return .{ .executor = executor, .pool = .init(gpa, backend), .root = root, .registry = registry, .default_model = default_model };
    }

    /// After the executor's worker stopped.
    pub fn deinit(self: *Service, io: std.Io) void {
        self.pool.deinit(io);
    }

    pub fn register(self: *Service, gpa: std.mem.Allocator, router: *router_mod.Router, listing: *models.Models) !void {
        try router.add(gpa, .POST, router_mod.prefix ++ "/decisions", .{ .context = self, .handle = handleDecisions });
        try router.add(gpa, .POST, router_mod.prefix ++ "/systemone", .{ .context = self, .handle = handleSystemOne });
        try listing.add(gpa, .{ .context = self, .list = list });
    }

    fn handleDecisions(context: *anyopaque, arena: std.mem.Allocator, io: std.Io, request: http.Request) http.Response {
        const self: *Service = @ptrCast(@alignCast(context));
        return self.answer(arena, io, request, .decisions);
    }

    fn handleSystemOne(context: *anyopaque, arena: std.mem.Allocator, io: std.Io, request: http.Request) http.Response {
        const self: *Service = @ptrCast(@alignCast(context));
        return self.answer(arena, io, request, .systemone);
    }

    /// Validates, waits for the GPU, renders. Every failure is a response.
    pub fn answer(self: *Service, arena: std.mem.Allocator, io: std.Io, request: http.Request, shape: Shape) http.Response {
        const deadline = std.Io.Clock.Timestamp.fromNow(io, .{ .raw = .fromNanoseconds(self.timeout_ns), .clock = .awake });
        var diag: config.Diagnostic = .{};
        const root = wire.parseJson(arena, request.body, "the request body", &diag) catch
            return fail(arena, .unprocessable_entity, "invalid_json", diag.message());
        const name = if (root == .object) if (root.object.get("model")) |m| switch (m) {
            .string => |s| self.modelName(s),
            else => return fail(arena, .unprocessable_entity, "invalid_request", "\"model\" must be a string"),
        } else self.default_model else self.default_model;
        if (shape == .systemone and root == .object and root.object.get("states") != null)
            return fail(arena, .unprocessable_entity, "invalid_request", "systemone takes one \"state\"; POST /v1/decisions takes \"states\"");
        const located = catalog.locate(arena, io, self.root, self.registry, name, &diag) catch |err|
            return locateFailure(arena, err, diag.message());
        const decision = wire.fromJson(arena, io, root, "the request body", .refused, &diag) catch |err|
            return requestFailure(arena, err, diag.message());
        wire.checkLimits(decision, &diag) catch |err| return requestFailure(arena, err, diag.message());

        var job: Job = .{
            .pool = &self.pool,
            .arena = arena,
            .directory = located.directory,
            .name = name,
            .request = decision,
        };
        self.executor.submit(io, &job.item) catch |err| return switch (err) {
            error.Busy => fail(arena, http.overloaded, "busy", "the GPU queue is full; retry shortly"),
            error.Stopped => fail(arena, .service_unavailable, "shutting_down", "the server is stopping"),
        };
        self.executor.wait(io, &job.item, deadline) catch
            return fail(arena, http.overloaded, "timeout", std.fmt.allocPrint(arena, "waited {d} s for the GPU without starting", .{self.timeout_ns / std.time.ns_per_s}) catch "timeout");
        const done = switch (job.outcome) {
            .pending => unreachable,
            .failed => |e| return .fromError(arena, e),
            .done => |d| d,
        };
        const body: response.Body = .{
            .request = decision,
            .results = done.results,
            .identity = located.identity,
            .load_ns = done.load_ns,
            .timings = done.timings,
            .explain = explainParam(request),
        };
        var out: std.Io.Writer.Allocating = .init(arena);
        (switch (shape) {
            .decisions => response.write(&out.writer, body),
            .systemone => response.writeSystemOne(&out.writer, body),
        }) catch return fail(arena, .internal_server_error, "internal", "out of memory");
        return .{ .body = out.written() };
    }

    /// The nuclis model a request's `"model"` names. A TypeSafe id
    /// (`jev-latest`, `jev-1.13.0`) is the default decision model unless
    /// the registry names an entry so, which lets a Jev client switch over
    /// by changing only its base URL.
    pub fn modelName(self: *const Service, requested: []const u8) []const u8 {
        if (std.mem.startsWith(u8, requested, "jev-") and self.registry.find(requested) == null) return self.default_model;
        return requested;
    }

    /// Opens `name` now (the `--model` of `nuclis serve`), through the GPU.
    pub fn preload(self: *Service, arena: std.mem.Allocator, io: std.Io, name: []const u8, diag: *config.Diagnostic) !void {
        const located = try catalog.locate(arena, io, self.root, self.registry, name, diag);
        var job: Job = .{ .pool = &self.pool, .arena = arena, .directory = located.directory, .name = name, .request = null };
        try self.executor.submit(io, &job.item);
        job.item.done.waitUncancelable(io);
        if (job.outcome == .failed) {
            diag.set("{s}", .{job.outcome.failed.message});
            return error.ModelOpenFailed;
        }
    }

    fn list(context: *anyopaque, arena: std.mem.Allocator, io: std.Io, out: *std.ArrayList(models.Entry)) anyerror!void {
        const self: *Service = @ptrCast(@alignCast(context));
        for (try catalog.list(arena, io, self.root, self.registry)) |l| {
            var details: std.json.ObjectMap = .empty;
            try details.put(arena, "kind", .{ .string = "decision" });
            try details.put(arena, "present", .{ .bool = l.present });
            try details.put(arena, "loaded", .{ .bool = if (l.directory) |d| self.pool.isOpen(io, d) else false });
            try details.put(arena, "default", .{ .bool = std.mem.eql(u8, l.name, self.default_model) });
            try details.put(arena, "max_len", if (l.budget) |b| .{ .integer = @intCast(b.max_len) } else .null);
            try details.put(arena, "head_max_len", if (l.budget) |b| .{ .integer = @intCast(b.head_max_len) } else .null);
            try details.put(arena, "repo", if (l.repo) |r| .{ .string = r } else .null);
            try details.put(arena, "revision", if (l.revision) |r| .{ .string = r } else .null);
            const owner = if (l.repo) |r| r[0 .. std.mem.indexOfScalar(u8, r, '/') orelse r.len] else "local";
            try out.append(arena, .{ .id = l.name, .owned_by = owner, .details = .{ .object = details } });
        }
    }
};

/// One request's GPU work: open the model if needed, then answer (nothing
/// to answer when preloading). Runs on the worker; `arena` is the request's,
/// untouched by its connection until `wait` returns.
const Job = struct {
    item: gpu.Item = .{ .run = run },
    pool: *pool_mod.Pool,
    arena: std.mem.Allocator,
    directory: []const u8,
    name: []const u8,
    request: ?wire.Request,
    outcome: union(enum) { pending, done: Done, failed: ApiError } = .pending,

    const Done = struct { results: []const inference.decide.StateResult, load_ns: u64, timings: inference.decide.Timings };

    fn run(item: *gpu.Item, io: std.Io) void {
        const job: *Job = @fieldParentPtr("item", item);
        job.outcome = job.execute(io) catch |err| .{ .failed = job.failure(err) };
    }

    fn execute(job: *Job, io: std.Io) !@FieldType(Job, "outcome") {
        const acquired = job.pool.acquire(io, job.directory, job.name) catch |err| return .{ .failed = openFailure(job.arena, job.name, err) };
        const request = job.request orelse return .{ .done = .{ .results = &.{}, .load_ns = acquired.load_ns, .timings = .{} } };
        var timings: inference.decide.Timings = .{};
        const states = try wire.decisionStates(job.arena, request);
        const results = try acquired.decider.decide(job.arena, io, states, request.questions, .{}, &timings);
        return .{ .done = .{ .results = results, .load_ns = acquired.load_ns, .timings = timings } };
    }

    fn failure(job: *Job, err: anyerror) ApiError {
        return switch (err) {
            error.OptionsExceedBudget => .init(.unprocessable_entity, "options_exceed_budget", "a question's options do not fit in the model's budget; shorten them or ask fewer"),
            error.OutOfMemory => .init(.internal_server_error, "internal", "out of memory"),
            error.InvalidUtf8, error.LimitExceeded, error.WorkLimitExceeded => .init(.unprocessable_entity, "invalid_request", std.fmt.allocPrint(job.arena, "a text could not be tokenized ({s})", .{@errorName(err)}) catch "a text could not be tokenized"),
            else => .init(.internal_server_error, "internal", std.fmt.allocPrint(job.arena, "{s}: {s}", .{ job.name, @errorName(err) }) catch @errorName(err)),
        };
    }
};

fn openFailure(arena: std.mem.Allocator, name: []const u8, err: anyerror) ApiError {
    if (err == error.MetalNotEnabled) return .init(.internal_server_error, "model_failed", "this build has no Metal backend; serve with --backend cpu");
    return .init(.internal_server_error, "model_failed", std.fmt.allocPrint(arena, "{s}: not a Laya checkpoint directory nuclis can run ({s})", .{ name, @errorName(err) }) catch "the model failed to open");
}

fn locateFailure(arena: std.mem.Allocator, err: anyerror, message: []const u8) http.Response {
    return switch (err) {
        error.NotADecisionModel => fail(arena, .unprocessable_entity, "not_a_decision_model", message),
        error.ModelFileNotFound => fail(arena, .unprocessable_entity, "model_not_found", message),
        error.InvalidRegistryEntry => fail(arena, .internal_server_error, "invalid_registry_entry", message),
        error.OutOfMemory => fail(arena, .internal_server_error, "internal", "out of memory"),
        else => fail(arena, .internal_server_error, "internal", @errorName(err)),
    };
}

fn requestFailure(arena: std.mem.Allocator, err: anyerror, message: []const u8) http.Response {
    return switch (err) {
        error.FileStateRefused => fail(arena, .unprocessable_entity, "file_state_refused", message),
        error.RequestTooLarge, error.StateTooLarge, error.TooManyOptions => fail(arena, .unprocessable_entity, "request_too_large", message),
        error.OutOfMemory => fail(arena, .internal_server_error, "internal", "out of memory"),
        // InvalidRequest, InvalidQuestion, MissingQuestions: the request's own fault.
        else => fail(arena, .unprocessable_entity, "invalid_request", message),
    };
}

/// `message` may live in a caller's diagnostic buffer: copied into `arena`.
fn fail(arena: std.mem.Allocator, status: std.http.Status, code: []const u8, message: []const u8) http.Response {
    return .fromError(arena, .init(status, code, arena.dupe(u8, message) catch "out of memory"));
}

/// `?explain` or `?explain=1`; `0` and `false` turn it off.
fn explainParam(request: http.Request) bool {
    const value = request.param("explain") orelse return false;
    return !(std.mem.eql(u8, value, "0") or std.mem.eql(u8, value, "false"));
}

// ----- tests: the tiny checkpoint on the CPU -----

const Fixture = struct {
    gpa: std.mem.Allocator,
    tmp: std.testing.TmpDir,
    directory: [:0]u8,
    entries: [1]config.NamedModel,
    executor: gpu.Executor,
    service: Service,

    /// A service over one registry entry, `tiny`, with `max_queued` queue
    /// slots; the worker is the caller's to start.
    fn init(self: *Fixture, max_queued: usize) !void {
        const gpa = std.testing.allocator;
        const io = std.testing.io;
        self.gpa = gpa;
        self.tmp = std.testing.tmpDir(.{});
        errdefer self.tmp.cleanup();
        try @import("../../decision/tiny.zig").write(gpa, io, self.tmp.dir);
        self.directory = try self.tmp.dir.realPathFileAlloc(io, ".", gpa);
        self.entries = .{.{ .name = "tiny", .entry = .{ .kind = .decision, .path = self.directory } }};
        self.executor = .init(max_queued);
        self.service = .init(gpa, &self.executor, .cpu, self.directory, .{ .entries = &self.entries }, "tiny");
    }

    fn deinit(self: *Fixture) void {
        self.service.deinit(std.testing.io);
        self.gpa.free(self.directory);
        self.tmp.cleanup();
    }

    fn post(self: *Fixture, arena: std.mem.Allocator, shape: Service.Shape, query: []const u8, body: []const u8) http.Response {
        return self.service.answer(arena, std.testing.io, .{ .method = .POST, .path = "/v1/decisions", .query = query, .body = body }, shape);
    }
};

const ticket =
    \\{"questions":{"team":{"type":"choice","instructions":"Which team?","criteria":{"billing":"invoices","technical":"bugs"}},
    \\ "urgent":{"type":"noul","instructions":"Urgent?"},"level":{"type":"score","instructions":"How bad?","criteria":["fine","bad","down"]}},
    \\ "states":["Charged twice!",{"customer":"acme","tickets":[1,2]},[{"role":"user","content":"down again"}]]}
;

/// `text` with the timing values replaced, so two runs compare.
fn withoutTimings(arena: std.mem.Allocator, text: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    var rest = text;
    while (rest.len > 0) {
        const at = for ([_][]const u8{ "\"load\": ", "\"tokenize\": ", "\"encode\": " }) |key| {
            if (std.mem.startsWith(u8, rest, key)) break key.len;
        } else 0;
        if (at == 0) {
            try out.append(arena, rest[0]);
            rest = rest[1..];
            continue;
        }
        try out.appendSlice(arena, rest[0..at]);
        try out.append(arena, 'T');
        rest = rest[at..];
        while (rest.len > 0 and (std.ascii.isDigit(rest[0]) or rest[0] == '.')) rest = rest[1..];
    }
    return out.items;
}

test "decisions: the response is what `nuclis decide --json` writes" {
    const io = std.testing.io;
    var f: Fixture = undefined;
    try f.init(4);
    defer f.deinit();
    var worker = try io.concurrent(gpu.Executor.run, .{ &f.executor, io });
    defer {
        f.executor.stop(io);
        worker.await(io);
    }
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    for ([_]bool{ false, true }) |explain| {
        const served = f.post(arena, .decisions, if (explain) "explain=1" else "", ticket);
        try std.testing.expectEqual(std.http.Status.ok, served.status);
        try f.tmp.dir.writeFile(io, .{ .sub_path = "ticket.json", .data = ticket });
        const path = try std.fs.path.join(arena, &.{ f.directory, "ticket.json" });
        const decide = @import("../../decide.zig");
        var diag: config.Diagnostic = .{};
        const args: []const []const u8 = if (explain) &.{ "--request", path, "--json", "--backend", "cpu", "--explain" } else &.{ "--request", path, "--json", "--backend", "cpu" };
        var cli: std.Io.Writer.Allocating = .init(arena);
        try decide.run(std.testing.allocator, io, f.directory, .{ .name = "tiny" }, try decide.parseArgs(arena, args, &diag), &cli.writer, .none, &diag);
        try std.testing.expectEqualStrings(try withoutTimings(arena, cli.written()), try withoutTimings(arena, served.body));
    }

    // TypeSafe's own example request, as a Jev client sends it.
    const jev =
        \\{"state":"Help! My payouts have been failing for 3 days.","model":"jev-latest",
        \\ "questions":{"is_urgent":{"type":"noul","instructions":"Does this convey urgency?","criteria":{"true":"Explicitly time-sensitive","false":"No urgency expressed"}},
        \\ "department":{"type":"choice","instructions":{"question":"Which team?","teams":["billing","technical"]},"criteria":{"billing":"Payments","technical":["Bugs","Outages"],"sales":null}}}}
    ;
    const one = f.post(arena, .systemone, "", jev);
    try std.testing.expectEqual(std.http.Status.ok, one.status);
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, arena, one.body, .{});
    try std.testing.expectEqualStrings("tiny", parsed.object.get("model").?.string);
    const answers = parsed.object.get("answers").?.object;
    const urgent = answers.get("is_urgent").?.object;
    try std.testing.expectEqual(@as(usize, 2), urgent.count());
    try std.testing.expectEqual(@as(f64, 0.5), urgent.get("noul").?.float);
    const department = answers.get("department").?.object;
    for ([_][]const u8{ "type", "choice", "probabilities", "confidence" }, department.keys()) |want, got| try std.testing.expectEqualStrings(want, got);
    try std.testing.expect(parsed.object.get("usage").?.object.get("input_tokens").?.integer > 0);

    var listing: std.ArrayList(models.Entry) = .empty;
    try Service.list(&f.service, arena, io, &listing);
    const tiny = for (listing.items) |e| {
        if (std.mem.eql(u8, e.id, "tiny")) break e;
    } else return error.TestExpectedTiny;
    try std.testing.expect(tiny.details.object.get("present").?.bool and tiny.details.object.get("loaded").?.bool and tiny.details.object.get("default").?.bool);
    try std.testing.expectEqual(@as(i64, 256), tiny.details.object.get("max_len").?.integer);
    try std.testing.expectEqualStrings("local", tiny.owned_by);
}

test "decisions: every refusal is a typed error body" {
    const io = std.testing.io;
    var f: Fixture = undefined;
    try f.init(4);
    defer f.deinit();
    var worker = try io.concurrent(gpu.Executor.run, .{ &f.executor, io });
    defer {
        f.executor.stop(io);
        worker.await(io);
    }
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const q = "\"questions\":{\"a\":{\"type\":\"noul\",\"instructions\":\"x\"}}";
    const cases = [_]struct { shape: Service.Shape = .decisions, body: []const u8, status: std.http.Status, code: []const u8 }{
        .{ .body = "{\"questions\":", .status = .unprocessable_entity, .code = "invalid_json" },
        .{ .body = "{" ++ q ++ ",\"state\":\"s\",\"model\":\"nope\"}", .status = .unprocessable_entity, .code = "model_not_found" },
        .{ .body = "{" ++ q ++ ",\"state\":\"s\",\"model\":7}", .status = .unprocessable_entity, .code = "invalid_request" },
        .{ .body = "{" ++ q ++ ",\"state\":\"s\",\"model\":\"qwen3.8-27b\"}", .status = .unprocessable_entity, .code = "not_a_decision_model" },
        .{ .body = "{" ++ q ++ ",\"state\":{\"file\":\"/etc/passwd\"}}", .status = .unprocessable_entity, .code = "file_state_refused" },
        .{ .body = "{" ++ q ++ "}", .status = .unprocessable_entity, .code = "invalid_request" },
        .{ .body = "{\"questions\":{\"a\":{\"type\":\"maybe\",\"instructions\":\"x\"}},\"state\":\"s\"}", .status = .unprocessable_entity, .code = "invalid_request" },
        .{ .body = "{\"questions\":{},\"state\":\"s\"}", .status = .unprocessable_entity, .code = "invalid_request" },
        .{ .body = "{" ++ q ++ ",\"states\":[" ++ "\"s\"," ** 64 ++ "\"s\"]}", .status = .unprocessable_entity, .code = "request_too_large" },
        .{ .shape = .systemone, .body = "{" ++ q ++ ",\"states\":[\"s\"]}", .status = .unprocessable_entity, .code = "invalid_request" },
    };
    for (cases) |case| {
        errdefer std.debug.print("case: {s}\n", .{case.body[0..@min(case.body.len, 80)]});
        const r = f.post(arena, case.shape, "", case.body);
        try std.testing.expectEqual(case.status, r.status);
        const parsed = try std.json.parseFromSliceLeaky(std.json.Value, arena, r.body, .{});
        try std.testing.expectEqualStrings(case.code, parsed.object.get("error").?.object.get("code").?.string);
    }
}

test "decisions: a full queue is busy; a request that never starts times out" {
    const io = std.testing.io;
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const body = "{\"questions\":{\"a\":{\"type\":\"noul\",\"instructions\":\"x\"}},\"state\":\"s\"}";
    {
        var f: Fixture = undefined;
        try f.init(0);
        defer f.deinit();
        const r = f.post(arena, .decisions, "", body);
        try std.testing.expectEqual(http.overloaded, r.status);
        try std.testing.expect(std.mem.indexOf(u8, r.body, "\"busy\"") != null);
    }
    {
        // No worker runs: the item waits in the queue until its deadline.
        var f: Fixture = undefined;
        try f.init(4);
        defer f.deinit();
        f.service.timeout_ns = 20 * std.time.ns_per_ms;
        const r = f.post(arena, .decisions, "", body);
        try std.testing.expectEqual(http.overloaded, r.status);
        try std.testing.expect(std.mem.indexOf(u8, r.body, "\"timeout\"") != null);
        try std.testing.expectEqual(@as(usize, 0), f.executor.stats(io).queued);
    }
}
