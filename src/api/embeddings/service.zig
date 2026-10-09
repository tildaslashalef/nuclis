//! The embedding service: `POST /v1/embeddings` (OpenAI's embeddings call,
//! with nuclis's `task`, `title`, `truncate` and `image_tokens`, and
//! `input_audio` parts) and the embedding models
//! for `GET /v1/models`. A handler parses and validates on its connection's
//! task, then waits in the batcher, whose GPU item opens the model if needed
//! and embeds every waiting request for it, pass by pass; a handler never
//! touches a model. Models are opened by name only, never by a client's
//! path. docs/guide/api.md § Embeddings.
const std = @import("std");
const inference = @import("inference");
const config = @import("../../config.zig");
const catalog = @import("../../catalog.zig");
const model_files = @import("../../model.zig");
const wire = @import("../../embedding/request.zig");
const response = @import("../../embedding/response.zig");
const names = @import("../../embedding/catalog.zig");
const http = @import("../http.zig");
const errors = @import("../errors.zig");
const gpu = @import("../gpu.zig");
const models = @import("../models.zig");
const router_mod = @import("../router.zig");
const pool_mod = @import("pool.zig");
const batcher_mod = @import("batcher.zig");

const embed = inference.embed;
const ApiError = errors.ApiError;

/// The route's body limit: 2,048 inputs of ordinary text fit.
pub const max_body = 32 << 20;
/// Embedding requests waiting for the GPU at once; one more is `busy`.
pub const max_waiting = 64;

pub const Service = struct {
    gpa: std.mem.Allocator,
    executor: *gpu.Executor,
    pool: pool_mod.Pool,
    batcher: batcher_mod.Batcher,
    /// The user root, for model names under `<root>/models`.
    root: ?[]const u8,
    registry: config.Models,
    /// `embed.model`: a request that names no model gets this one.
    default_model: []const u8,
    timeout_ns: u64,
    /// File digests computed for files without a sidecar, by path; owned.
    digests: std.StringHashMapUnmanaged([]const u8) = .empty,
    digests_mutex: std.Io.Mutex = .init,

    /// `registry` and `root` are borrowed for the service's lifetime. The
    /// batcher points into the service: call `bind` once it sits at its
    /// final address.
    pub fn init(gpa: std.mem.Allocator, executor: *gpu.Executor, backend: embed.Backend, root: ?[]const u8, registry: config.Models, default_model: []const u8, timeout_ns: u64) Service {
        return .{
            .gpa = gpa,
            .executor = executor,
            .pool = .init(gpa, backend),
            .batcher = .init(gpa, executor, undefined, max_waiting),
            .root = root,
            .registry = registry,
            .default_model = default_model,
            .timeout_ns = timeout_ns,
        };
    }

    pub fn bind(self: *Service) void {
        self.batcher.executor = self.executor;
        self.batcher.pool = &self.pool;
    }

    /// After the executor's worker stopped.
    pub fn deinit(self: *Service, io: std.Io) void {
        self.batcher.deinit();
        self.pool.deinit(io);
        var it = self.digests.iterator();
        while (it.next()) |e| {
            self.gpa.free(e.key_ptr.*);
            self.gpa.free(e.value_ptr.*);
        }
        self.digests.deinit(self.gpa);
    }

    pub fn register(self: *Service, gpa: std.mem.Allocator, router: *router_mod.Router, listing: *models.Models) !void {
        try router.addLimited(gpa, .POST, router_mod.prefix ++ "/embeddings", .{ .context = self, .handle = handle }, max_body);
        try listing.add(gpa, .{ .context = self, .list = list });
    }

    fn handle(context: *anyopaque, arena: std.mem.Allocator, io: std.Io, request: http.Request) http.Response {
        const self: *Service = @ptrCast(@alignCast(context));
        return self.answer(arena, io, request.body);
    }

    /// Validates, waits for the GPU, writes OpenAI's body. Every failure is
    /// a response.
    pub fn answer(self: *Service, arena: std.mem.Allocator, io: std.Io, body: []const u8) http.Response {
        const deadline = std.Io.Clock.Timestamp.fromNow(io, .{ .raw = .fromNanoseconds(self.timeout_ns), .clock = .awake });
        const root = std.json.parseFromSliceLeaky(std.json.Value, arena, body, .{}) catch |err| return switch (err) {
            error.OutOfMemory => internal(arena),
            else => refuse(arena, .bad_request, "invalid_json", null, "the request body is not valid JSON ({s})", .{@errorName(err)}),
        };
        var problem: wire.Problem = .{};
        const parsed = wire.fromJson(arena, root, &problem) catch |err| return switch (err) {
            error.Refused => refuse(arena, .bad_request, problem.code, problem.param, "{s}", .{problem.message}),
            error.OutOfMemory => internal(arena),
        };
        const name = parsed.model orelse self.default_model;
        var diag: config.Diagnostic = .{};
        var located = self.locate(arena, io, name, &diag) catch |err| return switch (err) {
            error.NotAnEmbeddingModel => refuse(arena, .bad_request, "not_an_embedding_model", "model", "{s}", .{diag.message()}),
            error.ModelNotFound, error.ModelFileNotFound => refuse(arena, .not_found, "model_not_found", "model", "{s}", .{diag.message()}),
            error.OutOfMemory => internal(arena),
            else => refuse(arena, .internal_server_error, "model_failed", "model", "{s}: {s}", .{ name, if (diag.len > 0) diag.message() else @errorName(err) }),
        };
        if (located.identity.sha256.len == 0) located.identity.sha256 = self.digest(arena, io, located.path) catch |err|
            return refuse(arena, .internal_server_error, "model_failed", "model", "{s}: its digest could not be read ({s})", .{ name, @errorName(err) });

        const request = parsed.request;
        const inputs = arena.alloc(embed.Input, request.inputs.len) catch return internal(arena);
        for (request.inputs, inputs) |input, *out| out.* = wire.render(arena, request, input) catch return internal(arena);
        if ((request.hasImages() or request.hasAudio()) and located.mmproj == null)
            return refuse(arena, .bad_request, "unsupported_feature", "input", "{s}: images and audio need the model's projector, which is not pulled (`nuclis model pull {s} --with mmproj`)", .{ name, name });
        var job: batcher_mod.Job = .{ .arena = arena, .path = located.path, .mmproj = located.mmproj, .name = name, .inputs = inputs, .truncate = request.truncate, .image_tokens = request.image_tokens };
        self.batcher.submit(io, &job) catch |err| return switch (err) {
            error.Busy => refuse(arena, http.overloaded, "busy", null, "overloaded: too many embedding requests wait for the GPU; retry shortly", .{}),
            error.Stopped => refuse(arena, .service_unavailable, "shutting_down", null, "the server is stopping", .{}),
        };
        self.batcher.wait(io, &job, deadline) catch
            return refuse(arena, http.overloaded, "timeout", null, "waited {d} s for the GPU without starting", .{self.timeout_ns / std.time.ns_per_s});
        switch (job.outcome) {
            .pending => unreachable,
            .failed => |e| return .fromError(arena, e),
            .done => {},
        }
        const vectors = arena.alloc(response.Vector, inputs.len) catch return internal(arena);
        for (vectors, job.vectors, job.prepared.?) |*v, values, p| v.* = .{
            .values = if (request.dimensions == embed.dimensions) values else embed.truncate(values, request.dimensions) catch |err|
                return refuse(arena, .internal_server_error, "internal", null, "{s}", .{@errorName(err)}),
            .tokens = p.tokens.len,
            .input_tokens = p.length,
        };
        const out_body: response.Body = .{
            .identity = located.identity,
            .request = request,
            .vectors = vectors,
            .timings = .{ .load_ns = job.timings.load_ns, .tokenize_ns = job.timings.tokenize_ns, .embed_ns = job.timings.embed_ns },
        };
        var out: std.Io.Writer.Allocating = .init(arena);
        response.writeOpenAI(&out.writer, arena, out_body, parsed.encoding) catch return internal(arena);
        return .{ .body = out.written(), .note = note(arena, name, out_body, job.timings) };
    }

    /// An embedding model by registry or catalogue name only: the server
    /// opens no path a client names.
    fn locate(self: *Service, arena: std.mem.Allocator, io: std.Io, name: []const u8, diag: *config.Diagnostic) !names.Located {
        if (self.registry.find(name) == null and catalog.findEmbedding(name) == null) {
            if (catalog.find(name) != null or catalog.findDecision(name) != null) {
                diag.set("{s} is a {s} model; POST /v1/embeddings takes an embedding model ({s})", .{ name, if (catalog.find(name) != null) "language" else "decision", self.default_model });
                return error.NotAnEmbeddingModel;
            }
            diag.set("{s} is not an embedding model of the registry or the catalogue (`GET /v1/models` lists them)", .{name});
            return error.ModelNotFound;
        }
        return names.locate(arena, io, self.root, self.registry, name, diag);
    }

    /// The file's SHA-256, computed once per path for the server's life.
    fn digest(self: *Service, arena: std.mem.Allocator, io: std.Io, path: []const u8) ![]const u8 {
        {
            self.digests_mutex.lockUncancelable(io);
            defer self.digests_mutex.unlock(io);
            if (self.digests.get(path)) |d| return arena.dupe(u8, d);
        }
        const computed = try names.digest(self.gpa, io, path);
        errdefer self.gpa.free(computed);
        const key = try self.gpa.dupe(u8, path);
        errdefer self.gpa.free(key);
        self.digests_mutex.lockUncancelable(io);
        defer self.digests_mutex.unlock(io);
        const slot = try self.digests.getOrPut(self.gpa, key);
        if (slot.found_existing) {
            // Another request computed it meanwhile.
            self.gpa.free(key);
            self.gpa.free(computed);
        } else slot.value_ptr.* = computed;
        return arena.dupe(u8, slot.value_ptr.*);
    }

    fn list(context: *anyopaque, arena: std.mem.Allocator, io: std.Io, out: *std.ArrayList(models.Entry)) anyerror!void {
        const self: *Service = @ptrCast(@alignCast(context));
        var listed: std.ArrayList([]const u8) = .empty;
        for (&catalog.embedding_entries) |*e| if (self.registry.find(e.name) == null) try listed.append(arena, e.name);
        for (self.registry.entries) |named| if (named.entry.kind == .embedding) try listed.append(arena, named.name);
        var tasks: std.json.Array = .init(arena);
        for (std.enums.values(wire.Task)) |t| try tasks.append(.{ .string = @tagName(t) });
        var widths: std.json.Array = .init(arena);
        for (embed.widths) |w| try widths.append(.{ .integer = @intCast(w) });
        for (listed.items) |name| {
            var diag: config.Diagnostic = .{};
            const resolved = names.resolve(arena, self.root, self.registry, name, &diag) catch continue;
            const stat = std.Io.Dir.cwd().statFile(io, resolved.path, .{}) catch null;
            const sidecar = model_files.readSidecar(arena, io, .cwd(), try model_files.sidecarPath(arena, resolved.path)) catch null;
            // The catalogue's facts when it pins the file, by name or digest.
            const pinned = resolved.entry orelse if (sidecar) |s| catalog.embeddingByDigest(s.sha256) else null;
            const registered = self.registry.find(name);
            var details: std.json.ObjectMap = .empty;
            try details.put(arena, "kind", .{ .string = "embedding" });
            try details.put(arena, "name", .{ .string = if (pinned) |p| p.title else name });
            try details.put(arena, "architecture", .{ .string = if (pinned) |p| p.architecture else embed.architecture });
            try details.put(arena, "quantization", if (pinned) |p| .{ .string = p.quantization } else .null);
            try details.put(arena, "size_bytes", .{ .integer = @intCast(if (stat) |s| s.size else if (pinned) |p| p.size else 0) });
            try details.put(arena, "present", .{ .bool = stat != null });
            try details.put(arena, "loaded", .{ .bool = self.pool.isOpen(io, resolved.path) });
            try details.put(arena, "default", .{ .bool = std.mem.eql(u8, name, self.default_model) });
            try details.put(arena, "dimensions", .{ .array = widths });
            // What this server embeds: text, and images and audio once the projector is pulled.
            var modalities: std.json.Array = .init(arena);
            try modalities.append(.{ .string = "text" });
            if (resolved.mmproj) |m| if (std.Io.Dir.cwd().access(io, m, .{})) |_| {
                try modalities.append(.{ .string = "image" });
                try modalities.append(.{ .string = "audio" });
            } else |_| {};
            try details.put(arena, "modalities", .{ .array = modalities });
            try details.put(arena, "max_tokens", .{ .integer = embed.max_tokens });
            try details.put(arena, "tasks", .{ .array = tasks });
            const repo = if (sidecar) |s| s.repo else if (registered) |r| r.repo else if (pinned) |p| p.repo else null;
            const revision = if (sidecar) |s| s.revision else if (registered) |r| r.revision else if (pinned) |p| p.revision else null;
            try details.put(arena, "repo", if (repo) |r| .{ .string = r } else .null);
            try details.put(arena, "revision", if (revision) |r| .{ .string = r } else .null);
            const owner = if (repo) |r| r[0 .. std.mem.indexOfScalar(u8, r, '/') orelse r.len] else "local";
            try out.append(arena, .{ .id = name, .owned_by = owner, .details = .{ .object = details } });
        }
    }
};

fn refuse(arena: std.mem.Allocator, status: std.http.Status, code: []const u8, param: ?[]const u8, comptime format: []const u8, args: anytype) http.Response {
    var err: ApiError = .init(status, code, arena.print(format, args) catch code);
    err.param = param;
    return .fromError(arena, err);
}

fn internal(arena: std.mem.Allocator) http.Response {
    return .fromError(arena, .init(.internal_server_error, "internal", "out of memory"));
}

/// The log's line: the model, the inputs and tokens, the passes it took and
/// shared, and the open it paid for, if any.
fn note(arena: std.mem.Allocator, model: []const u8, body: response.Body, timings: batcher_mod.Job.Timings) []const u8 {
    const n = body.vectors.len;
    const passes = if (timings.passes > 1 or timings.shared > 1) arena.print(" · {d} pass{s}, shared by up to {d}", .{ timings.passes, if (timings.passes == 1) "" else "es", timings.shared }) catch "" else "";
    const opened = if (timings.load_ns > 0) arena.print(" · opened in {d:.0} ms", .{response.ms(timings.load_ns)}) catch "" else "";
    return arena.print("{s} · {d} input{s} · {d} tokens{s}{s}", .{ model, n, if (n == 1) "" else "s", body.tokens(), passes, opened }) catch model;
}

// ----- tests: everything before the GPU, with no model file -----

const Fixture = struct {
    tmp: std.testing.TmpDir,
    root: [:0]u8,
    file: [:0]u8,
    entries: [3]config.NamedModel,
    executor: gpu.Executor,
    service: Service,

    fn init(self: *Fixture, max_queued: usize) !void {
        const gpa = std.testing.allocator;
        const io = std.testing.io;
        self.tmp = std.testing.tmpDir(.{});
        errdefer self.tmp.cleanup();
        try self.tmp.dir.writeFile(io, .{ .sub_path = "e.gguf", .data = "GGUF" });
        self.root = try self.tmp.dir.realPathFileAlloc(io, ".", gpa);
        self.file = try self.tmp.dir.realPathFileAlloc(io, "e.gguf", gpa);
        self.entries = .{
            .{ .name = "e", .entry = .{ .kind = .embedding, .path = self.file } },
            .{ .name = "gone", .entry = .{ .kind = .embedding, .path = "/nowhere/e.gguf" } },
            .{ .name = "chat", .entry = .{ .path = "/nowhere/chat.gguf" } },
        };
        self.executor = .init(max_queued);
        self.service = .init(gpa, &self.executor, .cpu, self.root, .{ .entries = &self.entries }, "e", 20 * std.time.ns_per_ms);
        self.service.bind();
    }

    fn deinit(self: *Fixture) void {
        self.service.deinit(std.testing.io);
        std.testing.allocator.free(self.root);
        std.testing.allocator.free(self.file);
        self.tmp.cleanup();
    }
};

test "embeddings: every refusal before the GPU is a typed error body with its field" {
    var f: Fixture = undefined;
    try f.init(4);
    defer f.deinit();
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const cases = [_]struct { body: []const u8, status: std.http.Status, code: []const u8, param: ?[]const u8 }{
        .{ .body = "{\"input\":", .status = .bad_request, .code = "invalid_json", .param = null },
        .{ .body = "{\"input\":[1,2]}", .status = .bad_request, .code = "unsupported_feature", .param = "input" },
        .{ .body = "{\"input\":\"x\",\"dimensions\":100}", .status = .bad_request, .code = "unsupported_feature", .param = "dimensions" },
        .{ .body = "{\"input\":\"x\",\"model\":\"qwen3.8-27b\"}", .status = .bad_request, .code = "not_an_embedding_model", .param = "model" },
        .{ .body = "{\"input\":\"x\",\"model\":\"laya\"}", .status = .bad_request, .code = "not_an_embedding_model", .param = "model" },
        .{ .body = "{\"input\":\"x\",\"model\":\"chat\"}", .status = .bad_request, .code = "not_an_embedding_model", .param = "model" },
        .{ .body = "{\"input\":\"x\",\"model\":\"nope\"}", .status = .not_found, .code = "model_not_found", .param = "model" },
        .{ .body = "{\"input\":\"x\",\"model\":\"/etc/e.gguf\"}", .status = .not_found, .code = "model_not_found", .param = "model" },
        .{ .body = "{\"input\":\"x\",\"model\":\"gone\"}", .status = .not_found, .code = "model_not_found", .param = "model" },
    };
    for (cases) |case| {
        errdefer std.debug.print("case: {s}\n", .{case.body});
        const r = f.service.answer(arena, std.testing.io, case.body);
        try std.testing.expectEqual(case.status, r.status);
        const parsed = try std.json.parseFromSliceLeaky(std.json.Value, arena, r.body, .{});
        const err = parsed.object.get("error").?.object;
        try std.testing.expectEqualStrings(case.code, err.get("code").?.string);
        if (case.param) |p| try std.testing.expectEqualStrings(p, err.get("param").?.string);
    }
}

test "embeddings: a full queue is busy; a request that never starts times out" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    {
        var f: Fixture = undefined;
        try f.init(0);
        defer f.deinit();
        const r = f.service.answer(arena, std.testing.io, "{\"input\":\"x\"}");
        try std.testing.expectEqual(http.overloaded, r.status);
        try std.testing.expect(std.mem.indexOf(u8, r.body, "\"busy\"") != null);
    }
    {
        // No worker runs: the job waits until its deadline, then leaves.
        var f: Fixture = undefined;
        try f.init(4);
        defer f.deinit();
        const r = f.service.answer(arena, std.testing.io, "{\"input\":\"x\"}");
        try std.testing.expectEqual(http.overloaded, r.status);
        try std.testing.expect(std.mem.indexOf(u8, r.body, "\"timeout\"") != null);
        try std.testing.expectEqual(@as(usize, 0), f.service.batcher.stats(std.testing.io).waiting);
    }
}

test "embeddings: a file that is no model fails the request on the worker, not the server" {
    const io = std.testing.io;
    var f: Fixture = undefined;
    try f.init(4);
    defer f.deinit();
    f.service.timeout_ns = 60 * std.time.ns_per_s;
    var worker = try io.concurrent(gpu.Executor.run, .{ &f.executor, io });
    defer {
        f.executor.stop(io);
        worker.await(io);
    }
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const r = f.service.answer(arena, io, "{\"input\":[\"x\",\"y\"]}");
    try std.testing.expectEqual(std.http.Status.internal_server_error, r.status);
    try std.testing.expect(std.mem.indexOf(u8, r.body, "\"model_failed\"") != null);
    try std.testing.expect(!f.service.pool.isOpen(io, f.file));

    var listing: std.ArrayList(models.Entry) = .empty;
    try Service.list(&f.service, arena, io, &listing);
    const e = for (listing.items) |entry| {
        if (std.mem.eql(u8, entry.id, "e")) break entry;
    } else return error.TestExpectedEntry;
    const d = e.details.object;
    try std.testing.expectEqualStrings("embedding", d.get("kind").?.string);
    try std.testing.expect(d.get("present").?.bool and !d.get("loaded").?.bool and d.get("default").?.bool);
    try std.testing.expectEqual(@as(i64, 8192), d.get("max_tokens").?.integer);
    try std.testing.expectEqual(@as(usize, 4), d.get("dimensions").?.array.items.len);
    try std.testing.expectEqualStrings("local", e.owned_by);
    // The catalogue's entry is listed whether or not it is pulled.
    for (listing.items) |entry| if (std.mem.eql(u8, entry.id, "embeddinggemma-2")) {
        try std.testing.expect(!entry.details.object.get("present").?.bool);
        try std.testing.expectEqualStrings("unsloth", entry.owned_by);
    };
}
