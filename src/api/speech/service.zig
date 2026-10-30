//! The speech service: `POST /v1/audio/speech` (OpenAI's speech call), the
//! saved voices' routes (`voices.zig`), and the speech models for `GET
//! /v1/models`. A Base's saved voice is made ready (encoded on its first use
//! with the model) on the connection's task, before the job is queued, so
//! the GPU worker never waits on the CPU encoders. A handler validates on its
//! connection's task and queues a job; one short GPU item speaks the jobs
//! in arrival order, a chunk of frames per run, so a chunk runs between a
//! chat generation's steps (`gpu.Executor.runShort`) and a voice holds one
//! synthesizer at a time. The response head waits until the job's prompt
//! is prefilled, so a refusal on the worker (an unknown speaker) is still a
//! status; the audio then streams through a `Pipe`. A client that leaves
//! cancels between chunks. docs/guide/api.md § Speech.
const std = @import("std");
const inference = @import("inference");
const config = @import("../../config.zig");
const catalog = @import("../../catalog.zig");
const audio_catalog = @import("../../audio/catalog.zig");
const http = @import("../http.zig");
const request_log = @import("../request_log.zig");
const errors = @import("../errors.zig");
const gpu = @import("../gpu.zig");
const models = @import("../models.zig");
const router_mod = @import("../router.zig");
const slot_mod = @import("../slot.zig");
const Pipe = @import("../pipe.zig").Pipe;
const wire = @import("wire.zig");
const voice_routes = @import("voices.zig");
const voice_store = @import("../../voices.zig");
const paths = @import("../../paths.zig");
const speech = @import("../../audio/speech.zig");

const tts = inference.models.qwen3_tts;
const runtime = inference.models.qwen3_tts_runtime;
const ApiError = errors.ApiError;

const drama = inference.models.drama3_speech;

pub const Voice = speech.Voice;
pub const Slot = slot_mod.Slot(Voice, .speech);

/// Speech requests waiting for the voice at once; one more is `busy`.
pub const max_waiting = 64;
/// Bytes a stream may hold between the worker and the socket: about 5 s
/// of PCM. A chunk waits (the job parks) rather than block the worker.
const pipe_capacity = 256 * 1024;
/// Beside a generation, how far the audio may run ahead of its playback
/// (which starts with its first byte) before speech yields to the generation.
const lead_ms = 1000;
/// How often a waiting request looks at its client; how long an event
/// stream may stay silent before a keepalive comment.
const poll_ms = 1000;
const keepalive_ms = 10_000;

/// Requests queued behind the one speaking, whether one is, and the
/// requests answered since the start.
pub const Stats = struct { waiting: usize, speaking: bool, requests: u64 };

pub const Service = struct {
    gpa: std.mem.Allocator,
    executor: *gpu.Executor,
    slot: Slot,
    root: ?[]const u8,
    registry: config.Models,
    /// `speech.model` and `speech.voice`: what a request that names none gets.
    settings: config.Config.Speech,
    timeout_ns: u64,
    /// Frames in the first chunk and in every later one.
    first_chunk: usize = runtime.default_first_chunk,
    chunk: usize = runtime.default_chunk,

    /// The queue: guards the list, `scheduled`, `parked`, and each job's `state`.
    mutex: std.Io.Mutex = .init,
    head: ?*Job = null,
    tail: ?*Job = null,
    waiting: usize = 0,
    /// The item is queued or running.
    scheduled: bool = false,
    /// The job speaking waits for its client to take what it has.
    parked: bool = false,
    requests: u64 = 0,
    /// Worker-only: the job speaking.
    active: ?*Job = null,
    routes: voice_routes.Routes,
    item: gpu.Item = .{ .run = step, .short = true },

    /// `root` and `registry` are borrowed for the service's lifetime; the
    /// service must not move once a request was queued (the item points
    /// into it).
    pub fn init(gpa: std.mem.Allocator, executor: *gpu.Executor, metal: bool, root: ?[]const u8, registry: config.Models, settings: config.Config.Speech, timeout_ns: u64) Service {
        return .{
            .gpa = gpa,
            .executor = executor,
            .slot = .init(gpa, openVoice, closeVoice, .{ .metal = metal }),
            .root = root,
            .registry = registry,
            .settings = settings,
            .timeout_ns = timeout_ns,
            .routes = .{ .root = root },
        };
    }

    /// After the executor's worker stopped.
    pub fn deinit(self: *Service, io: std.Io) void {
        if (self.active) |job| self.finish(io, job);
        self.slot.deinit(io);
    }

    pub fn register(self: *Service, gpa: std.mem.Allocator, router: *router_mod.Router, listing: *models.Models) !void {
        try router.add(gpa, .POST, router_mod.prefix ++ "/audio/speech", .{ .context = self, .handle = handle });
        try self.routes.register(gpa, router);
        try listing.add(gpa, .{ .context = self, .list = list });
    }

    pub fn stats(self: *Service, io: std.Io) Stats {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        return .{ .waiting = self.waiting, .speaking = self.scheduled or self.parked, .requests = self.requests };
    }

    /// Opens `name` now (a `--model` or `--pin` of `nuclis serve`) through the GPU.
    pub fn preload(self: *Service, arena: std.mem.Allocator, io: std.Io, name: []const u8, diag: *config.Diagnostic) !void {
        const located = try audio_catalog.locate(arena, io, self.root, self.registry, name, .speech, false, diag);
        var opening: Opening = .{ .slot = &self.slot, .directory = located.directory, .name = name, .bytes = located.bytes };
        try self.executor.submit(io, &opening.item);
        opening.item.done.waitUncancelable(io);
        if (opening.failed) |err| {
            diag.set("the speech model {s} did not open: {s}", .{ name, @errorName(err) });
            return err;
        }
    }

    fn handle(context: *anyopaque, arena: std.mem.Allocator, io: std.Io, request: http.Request) http.Response {
        const self: *Service = @ptrCast(@alignCast(context));
        return self.answer(arena, io, request.body, request.peer, request.fields);
    }

    /// Validates, queues, waits until the job's prompt is prefilled, and
    /// answers with the stream or the refusal. Every failure is a response.
    pub fn answer(self: *Service, arena: std.mem.Allocator, io: std.Io, body: []const u8, peer: ?http.Peer, fields: request_log.Fields) http.Response {
        const deadline = std.Io.Clock.Timestamp.fromNow(io, .{ .raw = .fromNanoseconds(self.timeout_ns), .clock = .awake });
        var problem: wire.Problem = .{};
        const parsed = wire.parse(arena, body, &problem) catch |err| return switch (err) {
            error.Refused => refuse(arena, .bad_request, problem.code, problem.param, "{s}", .{problem.message}),
            error.OutOfMemory => internal(arena),
        };
        const open = self.slot.openName(io, arena) catch null;
        const name = parsed.model orelse open orelse self.settings.model;
        var diag: config.Diagnostic = .{};
        const located = audio_catalog.locate(arena, io, self.root, self.registry, name, .speech, false, &diag) catch |err| return switch (err) {
            error.WrongKind => refuse(arena, .bad_request, "not_a_speech_model", "model", "{s} is a {s} model; this route speaks with a speech model ({s})", .{ name, catalog.kindName(catalog.kindOf(self.registry, name)), self.settings.model }),
            error.ModelNotFound, error.ModelFileNotFound => refuse(arena, .not_found, "model_not_found", "model", "{s}", .{diag.message()}),
            error.OutOfMemory => internal(arena),
            else => refuse(arena, .internal_server_error, "model_failed", "model", "{s}: {s}", .{ name, if (diag.len > 0) diag.message() else @errorName(err) }),
        };
        const mode = (audio_catalog.speechMode(arena, io, located.directory) catch return internal(arena)) orelse .speakers;
        fields.text("model", name);
        fields.text("voice_kind", @tagName(mode));
        if (speech.engineOf(located.directory) == .drama3) {
            if (parsed.instructions != null)
                return refuse(arena, .bad_request, "instructions_unsupported", "instructions", "{s} takes its delivery from cues in the input ([whispering], <emphasis>…</emphasis>), not instructions", .{name});
            if (parsed.language) |l| if (!std.mem.eql(u8, l, "auto"))
                return refuse(arena, .bad_request, "invalid_request", "language", "{s} reads the language from the input; it takes only auto", .{name});
        }
        if (parsed.instructions != null) {
            if (mode == .clone) return refuse(arena, .bad_request, "instructions_unsupported", "instructions", "{s} is a Base: it takes no instruction (qwen3-tts-1.7b-customvoice does)", .{name});
            if (located.entry) |entry| if (!entry.instructions)
                return refuse(arena, .bad_request, "instructions_unsupported", "instructions", "{s} performs no instruction; qwen3-tts-1.7b-customvoice does", .{name});
        }
        if (parsed.voice_by_id and mode != .clone)
            return refuse(arena, .bad_request, "voice_cloning_unsupported", "voice", "{s} speaks in {s}; a saved voice needs a Base (qwen3-tts-1.7b-base, qwen3-tts-0.6b-base)", .{ name, if (mode == .design) "a voice its instructions describe" else "its own speakers" });
        // VoiceDesign has no voice to name: OpenAI's clients always send one, so it is ignored.
        const voice_name = if (mode == .design) "" else parsed.voice orelse self.settings.voice;
        if (voice_name.len > 0) fields.text("voice", voice_name);
        var ready: ?voice_store.Ready = null;
        if (mode == .clone) {
            var refusal: ApiError = undefined;
            const started = std.Io.Clock.awake.now(io);
            ready = readyVoice(self.gpa, arena, io, self.root, located.directory, name, voice_name, &refusal) catch return .fromError(arena, refusal);
            fields.ms("voice_ready_ms", @intCast(started.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds()));
        }

        const job = arena.create(Job) catch {
            if (ready) |*r| r.deinit(self.gpa);
            return internal(arena);
        };
        job.* = .{
            .service = self,
            .request = parsed,
            .directory = located.directory,
            .name = name,
            .bytes = located.bytes,
            .voice = voice_name,
            .mode = mode,
            .saved = ready,
            .pipe = Pipe.init(self.gpa, pipe_capacity) catch {
                if (ready) |*r| r.deinit(self.gpa);
                return internal(arena);
            },
            .note_buffer = arena.alloc(u8, 512) catch return internal(arena),
            .message_buffer = arena.alloc(u8, 1024) catch return internal(arena),
            .peer = peer,
            .fields = fields,
        };
        self.submit(io, job) catch |err| {
            job.pipe.deinit(self.gpa);
            job.dropVoice();
            return switch (err) {
                error.Busy => refuse(arena, http.overloaded, "busy", null, "overloaded: {d} speech requests wait already; retry shortly", .{max_waiting}),
                error.Stopped => refuse(arena, .service_unavailable, "shutting_down", null, "the server is stopping", .{}),
            };
        };
        switch (job.waitReady(io, deadline)) {
            .ready => {},
            .timed_out => {
                job.release(io);
                return refuse(arena, http.overloaded, "timeout", null, "overloaded: waited {d} s for the GPU without starting (timeout)", .{self.timeout_ns / std.time.ns_per_s});
            },
            .gone => {
                job.release(io);
                return refuse(arena, .bad_request, "cancelled", null, "the client closed the connection", .{});
            },
        }
        if (job.failure) |failure| {
            job.release(io);
            return .fromError(arena, failure);
        }
        return .{
            .body = "",
            .content_type = wire.contentType(parsed),
            .stream = .{ .context = job, .write = Job.writeStream, .note = Job.streamNote },
        };
    }

    /// Queues `job` behind the others and schedules the item if it is idle.
    fn submit(self: *Service, io: std.Io, job: *Job) error{ Busy, Stopped }!void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (self.waiting >= max_waiting) return error.Busy;
        if (!self.scheduled and !self.parked) {
            try self.executor.submit(io, &self.item);
            self.scheduled = true;
        }
        if (self.tail) |t| t.next = job else self.head = job;
        self.tail = job;
        self.waiting += 1;
        job.state = .queued;
    }

    /// Takes `job` out of the queue if it has not started: true when the
    /// worker will never touch it.
    fn withdraw(self: *Service, io: std.Io, job: *Job) bool {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (job.state != .queued) return false;
        var previous: ?*Job = null;
        var cursor = self.head;
        while (cursor) |c| : ({
            previous = c;
            cursor = c.next;
        }) if (c == job) {
            if (previous) |p| p.next = c.next else self.head = c.next;
            if (self.tail == c) self.tail = previous;
            break;
        };
        self.waiting -= 1;
        job.state = .finished;
        job.done.set(io);
        return true;
    }

    /// The parked job's client took bytes (or left): schedule it again.
    fn unpark(self: *Service, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (!self.parked) return;
        self.executor.submit(io, &self.item) catch return;
        self.parked = false;
        self.scheduled = true;
    }

    /// The item: one step of the job speaking, taking the next when it ends.
    fn step(item: *gpu.Item, io: std.Io) gpu.Item.After {
        const self: *Service = @alignCast(@fieldParentPtr("item", item));
        const job = self.active orelse blk: {
            self.mutex.lockUncancelable(io);
            const next = self.head orelse {
                self.scheduled = false;
                self.mutex.unlock(io);
                return .done;
            };
            self.head = next.next;
            if (self.head == null) self.tail = null;
            self.waiting -= 1;
            next.state = .active;
            self.mutex.unlock(io);
            self.active = next;
            break :blk next;
        };
        if (job.cancel.load(.acquire)) {
            self.finish(io, job);
            return self.afterJob(io);
        }
        if (job.synth == null) {
            // Opening a voice may close the language model a generation is
            // running: it waits for the generation, not between its steps.
            if (item.short and !self.slot.isOpen(io, job.directory)) {
                item.short = false;
                return .again;
            }
            item.short = true;
            self.start(io, job) catch |err| {
                job.failure = startFailure(job, err);
                job.ready.set(io);
                self.finish(io, job);
                return self.afterJob(io);
            };
            job.ready.set(io);
        }
        if (self.executor.betweenSteps()) {
            // A generation waits for this run: a frame at a time, and only
            // while the listener could run out.
            if (job.ahead(io) >= lead_ms * std.time.ns_per_ms) return .again;
            job.synth.?.setChunk(1);
        } else job.synth.?.setChunk(self.chunk);
        // Parked rather than blocked: the worker never waits on a client.
        if (job.pipe.space(io) < wire.maxChunkBytes(@max(self.chunk, self.first_chunk), job.per_frame, job.request.stream)) {
            self.mutex.lockUncancelable(io);
            defer self.mutex.unlock(io);
            self.parked = true;
            self.scheduled = false;
            return .done;
        }
        if (!job.speak(io)) {
            self.finish(io, job);
            return self.afterJob(io);
        }
        return .again;
    }

    /// After a job ended: `.again` while another waits.
    fn afterJob(self: *Service, io: std.Io) gpu.Item.After {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (self.head != null) return .again;
        self.scheduled = false;
        return .done;
    }

    /// Opens the voice if needed, builds the prompt, and prefills it.
    fn start(self: *Service, io: std.Io, job: *Job) !void {
        job.started = std.Io.Clock.awake.now(io);
        const acquired = try self.slot.acquire(io, job.directory, job.name, job.bytes);
        job.load_ns = acquired.load_ns;
        const voice = acquired.model;
        self.slot.hold(io, true);
        job.holding = true;
        job.rate = acquired.model.sampleRate();
        job.per_frame = acquired.model.samplesPerFrame();
        var synth: speech.Synthesizer = switch (voice.model) {
            .qwen3_tts => |v| qwen: {
                // For the refusal's list, should the prompt name an unknown one.
                job.speakers = v.config.speakers;
                job.languages = v.config.languages;
                const language = job.request.language orelse "auto";
                var prompt = switch (job.mode) {
                    .clone => try v.clonePrompt(self.gpa, job.request.input, job.saved.?.reference_text, language, job.saved.?.clone()),
                    .design => try v.prompt(self.gpa, job.request.input, null, language, job.request.instructions),
                    .speakers => try v.prompt(self.gpa, job.request.input, job.voice, language, job.request.instructions),
                };
                defer prompt.deinit(self.gpa);
                job.prompt_rows = prompt.rows.len;
                var s = try v.synthesizer(self.gpa, &prompt, job.request.seed orelse 0);
                s.first_chunk = self.first_chunk;
                break :qwen .{ .qwen3_tts = s };
            },
            .drama3 => |v| d: {
                const saved = &job.saved.?;
                job.reference = .{ .transcript = saved.reference_text.?, .codes = saved.rows(drama.Codes) };
                break :d .{ .drama3 = try drama.Synthesizer.init(self.gpa, v, .{
                    .text = job.request.input,
                    .references = (&job.reference)[0..1],
                    .seed = job.request.seed orelse 0,
                }) };
            },
        };
        errdefer synth.deinit();
        synth.setChunk(self.chunk);
        const frames = @max(self.chunk, self.first_chunk);
        job.samples = try self.gpa.alloc(f32, frames * job.per_frame);
        errdefer self.gpa.free(job.samples);
        // The raw bytes, then (for events) their framing.
        const raw = wire.maxChunkBytes(frames, job.per_frame, .audio);
        const framed = if (job.request.stream == .sse) wire.maxChunkBytes(frames, job.per_frame, .sse) else 0;
        job.bytes_out = try self.gpa.alloc(u8, raw + framed);
        job.raw_len = raw;
        job.synth = synth;
    }

    /// The job is over: its synthesizer and buffers freed, the voice no
    /// longer held, the stream closed, its waiter woken.
    fn finish(self: *Service, io: std.Io, job: *Job) void {
        if (job.synth) |*s| {
            job.frames = s.frames();
            if (s.* == .drama3) job.prompt_rows = s.drama3.timing.prompt_rows;
            s.deinit();
        }
        job.synth = null;
        if (job.samples.len > 0) self.gpa.free(job.samples);
        if (job.bytes_out.len > 0) self.gpa.free(job.bytes_out);
        job.samples = &.{};
        job.bytes_out = &.{};
        if (job.holding) self.slot.hold(io, false);
        job.holding = false;
        job.writeNote();
        job.pipe.close(io);
        self.active = null;
        {
            self.mutex.lockUncancelable(io);
            defer self.mutex.unlock(io);
            job.state = .finished;
            self.requests += 1;
        }
        job.done.set(io);
    }

    fn list(context: *anyopaque, arena: std.mem.Allocator, io: std.Io, out: *std.ArrayList(models.Entry)) anyerror!void {
        const self: *Service = @ptrCast(@alignCast(context));
        try listAudio(arena, io, self.root, self.registry, .speech, self.settings.model, self.settings.voice, &self.slot, out);
    }
};

/// The models of `kind` for `GET /v1/models`: every catalogue entry, pulled
/// or not, then the registry's entries of the kind. Shared with the
/// transcription service.
pub fn listAudio(arena: std.mem.Allocator, io: std.Io, root: ?[]const u8, registry: config.Models, kind: audio_catalog.Kind, default_model: []const u8, default_voice: ?[]const u8, slot: anytype, out: *std.ArrayList(models.Entry)) !void {
    var listed: std.ArrayList([]const u8) = .empty;
    const wanted: catalog.ModelKind = switch (kind) {
        .speech => .speech,
        .transcription => .transcription,
    };
    for (&catalog.audio_entries) |*e| if (e.kind == wanted and registry.find(e.name) == null) try listed.append(arena, e.name);
    for (registry.entries) |named| if (named.entry.kind == wanted) try listed.append(arena, named.name);
    for (listed.items) |name| {
        var diag: config.Diagnostic = .{};
        const resolved = audio_catalog.resolve(arena, root, registry, name, kind, false, &diag) catch continue;
        const located: ?audio_catalog.Located = audio_catalog.locate(arena, io, root, registry, name, kind, false, &diag) catch null;
        const present = located != null;
        const entry = resolved.entry;
        const registered = registry.find(name);
        // A pulled speech model describes itself: its speakers' dialects and
        // its languages, and for one the catalogue does not pin, the rest.
        const described = if (present and kind == .speech) try audio_catalog.describeSpeech(arena, io, resolved.directory) else null;
        var details: std.json.ObjectMap = .empty;
        try details.put(arena, "kind", .{ .string = @tagName(kind) });
        const title = if (entry) |e| e.title else if (registered) |r| if (r.repo) |repo| std.fs.path.basenamePosix(repo) else name else name;
        try details.put(arena, "name", .{ .string = title });
        try details.put(arena, "architecture", if (entry) |e| .{ .string = e.architecture } else if (described != null) .{ .string = "qwen3_tts" } else .null);
        try details.put(arena, "quantization", if (entry) |e| .{ .string = e.quantization } else if (described) |d| .{ .string = d.quantization } else .null);
        try details.put(arena, "size_bytes", .{ .integer = @intCast(if (entry) |e| e.totalSize() else if (located) |l| l.bytes else 0) });
        try details.put(arena, "present", .{ .bool = present });
        try details.put(arena, "source", .{ .string = if (registered != null) "registry" else "catalogue" });
        // What `nuclis model pull <name>` fetches: the weights and, for speech, the codec.
        try details.put(arena, "download_bytes", if (registered == null) if (entry) |e| .{ .integer = @intCast(e.totalSize()) } else .null else .null);
        try details.put(arena, "loaded", .{ .bool = slot.isOpen(io, resolved.directory) });
        try details.put(arena, "default", .{ .bool = std.mem.eql(u8, name, default_model) });
        switch (kind) {
            .speech => {
                const drama3 = if (entry) |e| std.mem.eql(u8, e.architecture, "drama3") else audio_catalog.isFile(resolved.directory);
                const speakers = if (described) |d| d.speakers else if (entry) |e| e.speakers else &.{};
                // The checkpoint's own word once pulled, else the catalogue's; unknown for a registry entry not pulled.
                const mode: ?audio_catalog.Mode = if (described) |d| d.mode else if (entry) |e| e.mode else null;
                // A Base's voices are the saved ones, VoiceDesign's none.
                const voices = if (mode == .clone) try savedVoices(arena, io, root) else try models.strings(arena, speakers);
                try details.put(arena, "voices", .{ .array = voices });
                var objects: std.json.Array = .init(arena);
                for (speakers, 0..) |v, i| {
                    var speaker: std.json.ObjectMap = .empty;
                    try speaker.put(arena, "id", .{ .string = v });
                    const dialect = if (described) |d| d.dialects[i] else null;
                    try speaker.put(arena, "dialect", if (dialect) |d| .{ .string = d } else .null);
                    try objects.append(.{ .object = speaker });
                }
                try details.put(arena, "speakers", .{ .array = objects });
                const languages = if (described) |d| d.languages else &.{};
                try details.put(arena, "languages", .{ .array = try models.strings(arena, languages) });
                const instructions: ?bool = if (entry) |e| e.instructions else if (described) |d| d.instructions else null;
                try details.put(arena, "instructions", if (instructions) |b| .{ .bool = b } else .null);
                try details.put(arena, "cloning", if (mode == .clone) try cloning(arena, drama3) else .null);
                try details.put(arena, "design", if (mode) |m| .{ .bool = m == .design } else .null);
                try details.put(arena, "sample_rate", .{ .integer = if (drama3) drama.sample_rate else wire.sample_rate });
                var formats: std.json.Array = .init(arena);
                for (std.enums.values(wire.Format)) |f| try formats.append(.{ .string = @tagName(f) });
                try details.put(arena, "formats", .{ .array = formats });
                var limits: std.json.ObjectMap = .empty;
                try limits.put(arena, "max_input_bytes", .{ .integer = wire.max_input_bytes });
                try details.put(arena, "limits", .{ .object = limits });
                try details.put(arena, "options", .{ .object = try speechOptions(arena, voices, languages, instructions orelse false, default_voice, mode != .design) });
            },
            .transcription => {
                try details.put(arena, "sample_rate", .{ .integer = 16000 });
                var limits: std.json.ObjectMap = .empty;
                try limits.put(arena, "max_phrases", .{ .integer = inference.audio.boost.max_phrases });
                try limits.put(arena, "max_phrase_bytes", .{ .integer = inference.audio.boost.max_phrase_bytes });
                try details.put(arena, "limits", .{ .object = limits });
            },
        }
        const repo = if (entry) |e| e.repo else if (registered) |r| r.repo else null;
        try details.put(arena, "repo", if (repo) |r| .{ .string = r } else .null);
        try details.put(arena, "revision", if (entry) |e| .{ .string = e.revision } else if (registered) |r| if (r.revision) |v| .{ .string = v } else .null else .null);
        const owner = if (repo) |r| r[0 .. std.mem.indexOfScalar(u8, r, '/') orelse r.len] else "local";
        try out.append(arena, .{ .id = name, .owned_by = owner, .details = .{ .object = details } });
    }
}

/// The saved voices' names, for a Base's `voices`.
fn savedVoices(arena: std.mem.Allocator, io: std.Io, root: ?[]const u8) !std.json.Array {
    var names_list: std.json.Array = .init(arena);
    const home = root orelse return names_list;
    const store: voice_store.Store = .{ .io = io, .path = try paths.voicesPath(arena, home) };
    for (store.list(arena) catch &.{}) |v| try names_list.append(.{ .string = v.name });
    return names_list;
}

/// What a cloning model clones from: a Base, a clip alone or a clip and its
/// transcript; Drama 3 (`transcript_only`), the clip and its transcript.
fn cloning(arena: std.mem.Allocator, transcript_only: bool) !std.json.Value {
    var o: std.json.ObjectMap = .empty;
    try o.put(arena, "modes", .{ .array = try models.strings(arena, if (transcript_only) &.{"audio_and_transcript"} else &.{ "audio", "audio_and_transcript" }) });
    try o.put(arena, "voices_route", .{ .string = router_mod.prefix ++ "/audio/voices" });
    return .{ .object = o };
}

/// The request fields a speech model takes. A request without `voice`
/// gets `speech.voice`, so its default is null when the model has no such
/// speaker or saved voice (the request must name one); VoiceDesign takes
/// no voice (`takes_voice` false).
fn speechOptions(arena: std.mem.Allocator, voices: std.json.Array, languages: []const []const u8, instructions: bool, default_voice: ?[]const u8, takes_voice: bool) !std.json.ObjectMap {
    const Option = models.Option;
    var o: std.json.ObjectMap = .empty;
    var voice_default: std.json.Value = .null;
    if (default_voice) |v| for (voices.items) |item| if (std.mem.eql(u8, item.string, v)) {
        voice_default = .{ .string = v };
    };
    if (takes_voice) try o.put(arena, "voice", try (Option{ .type = .@"enum", .values = voices, .default = voice_default }).value(arena));
    var language_values = try models.strings(arena, &.{"auto"});
    for (languages) |l| try language_values.append(.{ .string = l });
    try o.put(arena, "language", try (Option{ .type = .@"enum", .values = language_values, .default = .{ .string = "auto" } }).value(arena));
    if (instructions) try o.put(arena, "instructions", try (Option{ .type = .string }).value(arena));
    var formats: std.ArrayList([]const u8) = .empty;
    for (std.enums.values(wire.Format)) |f| try formats.append(arena, @tagName(f));
    try o.put(arena, "response_format", try (Option{ .type = .@"enum", .values = try models.strings(arena, formats.items), .default = .{ .string = "wav" } }).value(arena));
    var streams: std.ArrayList([]const u8) = .empty;
    for (std.enums.values(wire.StreamFormat)) |f| try streams.append(arena, @tagName(f));
    try o.put(arena, "stream_format", try (Option{ .type = .@"enum", .values = try models.strings(arena, streams.items), .default = .{ .string = "audio" } }).value(arena));
    try o.put(arena, "seed", try (Option{ .type = .integer, .min = .{ .integer = 0 }, .default = .{ .integer = 0 } }).value(arena));
    return o;
}

/// One request: queued on the connection's task, spoken on the worker. It
/// lives in the connection's arena; the worker allocates only from the
/// service's allocator.
const Job = struct {
    service: *Service,
    next: ?*Job = null,
    request: wire.Request,
    directory: []const u8,
    name: []const u8,
    bytes: u64,
    /// The speaker or saved voice; empty on VoiceDesign.
    voice: []const u8,
    mode: audio_catalog.Mode,
    /// A Base's voice, owned by the service's allocator; the connection
    /// frees it once the worker let go of the job (`release`).
    saved: ?voice_store.Ready = null,
    pipe: Pipe,
    peer: ?http.Peer,
    /// Set by the connection when its client is gone; the worker stops
    /// before the next chunk.
    cancel: std.atomic.Value(bool) = .init(false),
    /// Guarded by the service's mutex.
    state: enum { idle, queued, active, finished } = .idle,
    /// Set once the prompt is prefilled or the job failed before.
    ready: std.Io.Event = .unset,
    /// Set once the worker can no longer touch the job.
    done: std.Io.Event = .unset,
    /// Why the job failed; set before `ready` or `done`.
    failure: ?ApiError = null,
    /// The log's line, in `note_buffer`, written by the worker at the end;
    /// a failure's message lives in `message_buffer`.
    note: []const u8 = "",
    note_buffer: []u8,
    message_buffer: []u8,

    // Worker-only, while it speaks.
    synth: ?speech.Synthesizer = null,
    /// The voice's output rate and frame length, set once it is open.
    rate: u32 = wire.sample_rate,
    per_frame: usize = 1920,
    /// Drama 3's reference, borrowing `saved`.
    reference: inference.models.drama3_text.Reference = undefined,
    samples: []f32 = &.{},
    /// `raw_len` bytes for a chunk's audio, then room to frame it as an event.
    bytes_out: []u8 = &.{},
    raw_len: usize = 0,
    holding: bool = false,
    header_sent: bool = false,
    prompt_rows: usize = 0,
    speakers: []const tts.Speaker = &.{},
    languages: []const tts.Language = &.{},
    started: std.Io.Timestamp = undefined,
    load_ns: u64 = 0,
    first_audio_ns: ?u64 = null,
    first_audio_at: ?std.Io.Timestamp = null,
    audio_samples: usize = 0,
    /// `eos` or `length` once the utterance ended; empty before (or cancelled).
    finish_reason: []const u8 = "",
    /// The frames generated, kept once the synthesizer is freed.
    frames: usize = 0,
    /// The request's log fields, filled once the worker is done.
    fields: request_log.Fields = .{},

    /// One chunk: synthesized, encoded, queued for the client. False once
    /// the utterance ended, the client left, or the device failed.
    fn speak(self: *Job, io: std.Io) bool {
        const synth = &self.synth.?;
        const n = synth.read(self.samples) catch |err| {
            self.failure = apiError(self.message_buffer, .internal_server_error, "model_failed", null, "the synthesis failed: {s}", .{@errorName(err)});
            if (self.request.stream == .sse) self.emitError(io);
            return false;
        };
        if (n == 0) {
            self.finish_reason = @tagName(synth.finish());
            if (self.request.stream == .sse) {
                var out: std.Io.Writer = .fixed(self.bytes_out[self.raw_len..]);
                wire.writeDone(&out, .{ .input_tokens = self.prompt_rows, .output_tokens = synth.frames() }, @tagName(synth.finish())) catch return false;
                self.pipe.write(io, out.buffered()) catch self.cancel.store(true, .release);
            }
            return false;
        }
        if (self.first_audio_at == null) {
            const now = std.Io.Clock.awake.now(io);
            self.first_audio_at = now;
            self.first_audio_ns = @intCast(@max(0, self.started.durationTo(now).toNanoseconds()));
        }
        self.audio_samples += n;
        var raw: std.Io.Writer = .fixed(self.bytes_out[0..self.raw_len]);
        if (self.request.format == .wav and !self.header_sent) raw.writeAll(&wire.streamingWavHeader(self.rate)) catch return false;
        self.header_sent = true;
        const pcm = wire.pcm16(self.samples[0..n], raw.unusedCapacitySlice());
        raw.advance(pcm.len);
        const chunk = switch (self.request.stream) {
            .audio => raw.buffered(),
            .sse => framed: {
                var out: std.Io.Writer = .fixed(self.bytes_out[self.raw_len..]);
                wire.writeDelta(&out, raw.buffered()) catch return false;
                break :framed out.buffered();
            },
        };
        self.pipe.write(io, chunk) catch {
            self.cancel.store(true, .release);
            return false;
        };
        return true;
    }

    /// Nanoseconds the audio made is ahead of its playback, which starts
    /// with the first audio and runs at real time; 0 before it.
    fn ahead(self: *const Job, io: std.Io) i64 {
        const first = self.first_audio_at orelse return 0;
        const made: i64 = @intCast(self.audio_samples * std.time.ns_per_s / self.rate);
        return made - @as(i64, @intCast(first.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds()));
    }

    /// An SSE error line after the stream began.
    fn emitError(self: *Job, io: std.Io) void {
        var line: [512]u8 = undefined;
        var out: std.Io.Writer = .fixed(&line);
        wire.writeStreamError(&out, self.failure.?) catch return;
        self.pipe.write(io, out.buffered()) catch {};
    }

    fn writeNote(self: *Job) void {
        const seconds = @as(f64, @floatFromInt(self.audio_samples)) / @as(f64, @floatFromInt(self.rate));
        self.note = if (self.failure) |f|
            std.fmt.bufPrint(self.note_buffer, "{s}: {s}", .{ f.code, f.message }) catch f.code
        else if (self.cancel.load(.acquire))
            std.fmt.bufPrint(self.note_buffer, "{s} · {s} · cancelled after {d:.2} s of audio", .{ self.name, self.voice, seconds }) catch "cancelled"
        else
            std.fmt.bufPrint(self.note_buffer, "{s} · {s} · {d:.2} s of audio · first audio {d:.0} ms{s}{s}", .{
                self.name,
                if (self.mode == .design) "designed" else self.voice,
                seconds,
                if (self.first_audio_ns) |ns| @as(f64, @floatFromInt(ns)) / std.time.ns_per_ms else 0,
                if (self.load_ns > 0) " · opened the voice" else "",
                if (std.mem.eql(u8, self.finish_reason, "length")) " · stopped at the length budget" else "",
            }) catch self.name;
    }

    const Waited = enum { ready, timed_out, gone };

    /// The connection's side before the head: wait for the prompt's
    /// prefill, looking at the client every second; a job still queued at
    /// `deadline` is withdrawn.
    fn waitReady(self: *Job, io: std.Io, deadline: std.Io.Clock.Timestamp) Waited {
        const service = self.service;
        while (true) {
            self.ready.waitTimeout(io, .{ .deadline = fromNow(io, poll_ms) }) catch {};
            if (self.ready.isSet()) return .ready;
            if (self.peer) |peer| if (peer.gone()) {
                self.cancel.store(true, .release);
                service.unpark(io);
                return .gone;
            };
            if (pastDeadline(io, deadline) and service.withdraw(io, self)) return .timed_out;
        }
    }

    /// The connection's side of the stream: drain the pipe into the body,
    /// wake a parked job once its client took bytes, keep an event stream
    /// alive, and stop the job when the client goes. Returns once the
    /// worker has let go of the job.
    fn writeStream(context: *anyopaque, io: std.Io, body: *http.Body) bool {
        const self: *Job = @ptrCast(@alignCast(context));
        defer self.release(io);
        var buffer: [16 * 1024]u8 = undefined;
        var quiet_since = std.Io.Clock.awake.now(io);
        while (true) {
            if (self.peer) |peer| if (peer.gone()) return self.abandon(io);
            const taken = self.pipe.take(io, &buffer, fromNow(io, poll_ms)) catch return self.abandon(io);
            switch (taken) {
                // A raw stream that failed midway ends without its last
                // chunk, which a client reads as an error.
                .closed => return self.failure == null or self.request.stream == .sse,
                .data => |bytes| {
                    body.send(bytes) catch return self.abandon(io);
                    quiet_since = std.Io.Clock.awake.now(io);
                    self.service.unpark(io);
                },
                .timeout => if (self.request.stream == .sse and quiet_since.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds() >= keepalive_ms) {
                    body.send(": speaking\n\n") catch return self.abandon(io);
                    quiet_since = std.Io.Clock.awake.now(io);
                },
            }
        }
    }

    fn abandon(self: *Job, io: std.Io) bool {
        self.cancel.store(true, .release);
        self.pipe.abandon(io);
        self.service.unpark(io);
        return false;
    }

    fn dropVoice(self: *Job) void {
        if (self.saved) |*r| r.deinit(self.service.gpa);
        self.saved = null;
    }

    /// Waits until the worker can no longer touch the job (or takes it
    /// back unstarted), then frees the pipe and the voice.
    fn release(self: *Job, io: std.Io) void {
        if (!self.service.withdraw(io, self)) {
            while (true) {
                self.done.waitTimeout(io, .{ .deadline = fromNow(io, poll_ms) }) catch {};
                if (self.done.isSet()) break;
                // A parked job finishes only once scheduled again.
                if (self.cancel.load(.acquire)) self.service.unpark(io);
            }
        }
        self.pipe.deinit(self.service.gpa);
        self.dropVoice();
    }

    fn streamNote(context: *anyopaque) []const u8 {
        const self: *Job = @ptrCast(@alignCast(context));
        self.logTo(self.fields);
        return if (self.cancel.load(.acquire) and self.note.len == 0) "cancelled: the client closed the connection" else self.note;
    }

    /// The request log's fields; only once the worker can no longer touch
    /// the job.
    fn logTo(self: *const Job, fields: request_log.Fields) void {
        fields.text("format", @tagName(self.request.format));
        fields.text("stream_format", @tagName(self.request.stream));
        if (self.failure) |f| fields.fail(f);
        fields.int("frames", self.frames);
        fields.float("audio_s", @as(f64, @floatFromInt(self.audio_samples)) / @as(f64, @floatFromInt(self.rate)), 2);
        if (self.first_audio_ns) |ns| fields.ms("first_audio_ms", ns);
        if (self.load_ns > 0) fields.ms("open_ms", self.load_ns);
        const finish = if (self.cancel.load(.acquire)) "cancelled" else self.finish_reason;
        if (finish.len > 0) fields.text("finish_reason", finish);
    }
};

/// `voice_name` ready for the Base in `directory`: read back, or encoded
/// now on this task and saved (`voice_store.Store.ensure`), owned by `gpa`. A
/// refusal is in `refusal`, its message in `arena`.
fn readyVoice(gpa: std.mem.Allocator, arena: std.mem.Allocator, io: std.Io, root: ?[]const u8, directory: []const u8, name: []const u8, voice_name: []const u8, refusal: *ApiError) error{Refused}!voice_store.Ready {
    const fail = struct {
        fn with(out: *ApiError, a: std.mem.Allocator, status: std.http.Status, code: []const u8, param: ?[]const u8, comptime format: []const u8, args: anytype) error{Refused} {
            out.* = .init(status, code, a.print(format, args) catch code);
            out.param = param;
            return error.Refused;
        }
    }.with;
    voice_store.checkName(voice_name) catch |err| return switch (err) {
        error.ReservedVoiceName => fail(refusal, arena, .bad_request, "invalid_request", "voice", "{s} is a speaker of the CustomVoice models; {s} is a Base and speaks in a saved voice (GET /v1/audio/voices)", .{ voice_name, name }),
        else => fail(refusal, arena, .not_found, "voice_not_found", "voice", "no saved voice named {s} (GET /v1/audio/voices)", .{voice_name}),
    };
    const home = root orelse return fail(refusal, arena, .internal_server_error, "no_home", "voice", "the server has no user root (HOME or NUCLIS_HOME) to keep voices in", .{});
    const store: voice_store.Store = .{ .io = io, .path = paths.voicesPath(arena, home) catch return fail(refusal, arena, .internal_server_error, "internal", null, "out of memory", .{}) };
    const keys = voice_store.modelKeys(arena, io, directory) catch |err| return fail(refusal, arena, .internal_server_error, "model_failed", "model", "{s}: its files' digests could not be read ({s})", .{ name, @errorName(err) });
    var encoders = voice_store.FileEncoders.init(gpa, io, directory) catch |err| return fail(refusal, arena, .internal_server_error, "model_failed", "model", "{s}: {s}", .{ name, @errorName(err) });
    defer encoders.deinit();
    return store.ensure(gpa, voice_name, keys, encoders.encoders()) catch |err| switch (err) {
        error.VoiceNotFound => fail(refusal, arena, .not_found, "voice_not_found", "voice", "no saved voice named {s} (GET /v1/audio/voices)", .{voice_name}),
        error.InvalidVoiceRecord => fail(refusal, arena, .internal_server_error, "voice_store_failed", "voice", "{s}: its saved files are damaged (DELETE /v1/audio/voices/{s}, then save it again)", .{ voice_name, voice_name }),
        else => fail(refusal, arena, .internal_server_error, "model_failed", "voice", "{s} could not be made ready: {s}", .{ voice_name, @errorName(err) }),
    };
}

/// A failure before the first chunk, as the client reads it.
fn startFailure(job: *Job, err: anyerror) ApiError {
    const arena_message = job.message_buffer;
    if (speech.inputProblem(err)) |problem| return apiError(arena_message, .bad_request, "invalid_request", "input", "{s} cannot speak \"input\": {s} (docs/guide/api.md § Speech)", .{ job.name, problem });
    return switch (err) {
        error.UnknownSpeaker => apiError(arena_message, .bad_request, "invalid_request", "voice", "{s} is not a speaker of {s}: {f}", .{ job.voice, job.name, names(job.speakers) }),
        error.UnknownLanguage => apiError(arena_message, .bad_request, "invalid_request", "language", "{s} is not a language of {s}: auto, {f}", .{ job.request.language orelse "auto", job.name, names(job.languages) }),
        error.InstructionUnsupported => apiError(arena_message, .bad_request, "instructions_unsupported", "instructions", "{s} performs no instruction; qwen3-tts-1.7b-customvoice does", .{job.name}),
        error.EmptyText => apiError(arena_message, .bad_request, "invalid_request", "input", "\"input\" has no text to speak", .{}),
        error.ModelTooLarge => apiError(arena_message, .bad_request, "model_too_large", "model", "{s} needs {d:.1} GiB, more than the server's budget leaves beside the pinned models (serve.memory_bytes, --memory, --pin)", .{ job.name, @as(f64, @floatFromInt(job.bytes)) / (1 << 30) }),
        error.Pinned => apiError(arena_message, .service_unavailable, "busy", "model", "overloaded: {s} needs memory a running request holds; retry shortly", .{job.name}),
        else => apiError(arena_message, .internal_server_error, "model_failed", "model", "{s} did not speak: {s}", .{ job.name, @errorName(err) }),
    };
}

/// The names of `items` joined by commas, for a message.
fn names(items: anytype) Names(@TypeOf(items)) {
    return .{ .items = items };
}

fn Names(comptime T: type) type {
    return struct {
        items: T,
        pub fn format(self: @This(), w: *std.Io.Writer) std.Io.Writer.Error!void {
            for (self.items, 0..) |item, i| try w.print("{s}{s}", .{ if (i > 0) ", " else "", item.name });
        }
    };
}

/// Opening a model on the worker, outside any request.
const Opening = struct {
    item: gpu.Item = .{ .run = run },
    slot: *Slot,
    directory: []const u8,
    name: []const u8,
    bytes: u64,
    failed: ?anyerror = null,

    fn run(item: *gpu.Item, io: std.Io) gpu.Item.After {
        const self: *Opening = @alignCast(@fieldParentPtr("item", item));
        _ = self.slot.acquire(io, self.directory, self.name, self.bytes) catch |err| {
            self.failed = err;
        };
        return .done;
    }
};

fn openVoice(gpa: std.mem.Allocator, io: std.Io, directory: []const u8, options: Slot.Options) anyerror!*Voice {
    return Voice.open(gpa, io, directory, null, if (options.metal) .metal else .cpu);
}

fn closeVoice(voice: *Voice, io: std.Io) void {
    voice.close(io);
}

fn fromNow(io: std.Io, ms: i64) std.Io.Clock.Timestamp {
    return std.Io.Clock.Timestamp.fromNow(io, .{ .raw = .fromMilliseconds(ms), .clock = .awake });
}

fn pastDeadline(io: std.Io, deadline: std.Io.Clock.Timestamp) bool {
    return std.Io.Clock.Timestamp.now(io, deadline.clock).compare(.gte, deadline);
}

fn apiError(buffer: []u8, status: std.http.Status, code: []const u8, param: ?[]const u8, comptime format: []const u8, args: anytype) ApiError {
    var err: ApiError = .init(status, code, std.fmt.bufPrint(buffer, format, args) catch code);
    err.param = param;
    return err;
}

fn refuse(arena: std.mem.Allocator, status: std.http.Status, code: []const u8, param: ?[]const u8, comptime format: []const u8, args: anytype) http.Response {
    var err: ApiError = .init(status, code, arena.print(format, args) catch code);
    err.param = param;
    return .fromError(arena, err);
}

fn internal(arena: std.mem.Allocator) http.Response {
    return .fromError(arena, .init(.internal_server_error, "internal", "out of memory"));
}

// ----- tests: everything before the GPU, and an open that fails on it -----

const Fixture = struct {
    tmp: std.testing.TmpDir,
    root: [:0]u8,
    executor: gpu.Executor,
    service: Service,

    /// A root whose 0.6B voice has files that are not a checkpoint, and a
    /// 0.6B Base with a config but no weights (what precedes the GPU).
    fn init(self: *Fixture, max_queued: usize) !void {
        const gpa = std.testing.allocator;
        const io = std.testing.io;
        self.tmp = std.testing.tmpDir(.{});
        errdefer self.tmp.cleanup();
        const dir = "models/Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice";
        try self.tmp.dir.createDirPath(io, dir ++ "/speech_tokenizer");
        try self.tmp.dir.writeFile(io, .{ .sub_path = dir ++ "/model.safetensors", .data = "not weights" });
        try self.tmp.dir.writeFile(io, .{ .sub_path = dir ++ "/speech_tokenizer/model.safetensors", .data = "not weights" });
        const base = "models/Qwen/Qwen3-TTS-12Hz-0.6B-Base";
        try self.tmp.dir.createDirPath(io, base ++ "/speech_tokenizer");
        try self.tmp.dir.writeFile(io, .{ .sub_path = base ++ "/model.safetensors", .data = "not weights" });
        try self.tmp.dir.writeFile(io, .{ .sub_path = base ++ "/speech_tokenizer/model.safetensors", .data = "not weights" });
        const config_json = try inference.models.qwen3_tts.testConfigJson(gpa, "0b6", 8, "base");
        defer gpa.free(config_json);
        try self.tmp.dir.writeFile(io, .{ .sub_path = base ++ "/config.json", .data = config_json });
        self.root = try self.tmp.dir.realPathFileAlloc(io, ".", gpa);
        self.executor = .init(max_queued);
        self.service = .init(gpa, &self.executor, false, self.root, .{}, .{ .model = "qwen3-tts-0.6b-customvoice" }, 20 * std.time.ns_per_ms);
    }

    fn deinit(self: *Fixture) void {
        self.service.deinit(std.testing.io);
        std.testing.allocator.free(self.root);
        self.tmp.cleanup();
    }
};

fn expectError(r: http.Response, status: std.http.Status, code: []const u8, param: ?[]const u8) !void {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    errdefer std.debug.print("got {d}: {s}\n", .{ @backingInt(r.status), r.body });
    try std.testing.expect(r.stream == null);
    try std.testing.expectEqual(status, r.status);
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, arena_state.allocator(), r.body, .{});
    const err = parsed.object.get("error").?.object;
    try std.testing.expectEqualStrings(code, err.get("code").?.string);
    if (param) |p| try std.testing.expectEqualStrings(p, err.get("param").?.string);
}

test "speech: every refusal before the GPU is a typed error body with its field" {
    var f: Fixture = undefined;
    try f.init(4);
    defer f.deinit();
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const io = std.testing.io;
    const cases = [_]struct { body: []const u8, status: std.http.Status, code: []const u8, param: ?[]const u8 }{
        .{ .body = "{\"input\":", .status = .bad_request, .code = "invalid_json", .param = null },
        .{ .body = "{\"input\": \"Hi.\", \"response_format\": \"mp3\"}", .status = .bad_request, .code = "unsupported_feature", .param = "response_format" },
        .{ .body = "{\"input\": \"Hi.\", \"speed\": 2}", .status = .bad_request, .code = "unsupported_feature", .param = "speed" },
        .{ .body = "{\"input\": \"Hi.\", \"model\": \"nemotron-asr-0.6b\"}", .status = .bad_request, .code = "not_a_speech_model", .param = "model" },
        .{ .body = "{\"input\": \"Hi.\", \"model\": \"qwen3.8-27b\"}", .status = .bad_request, .code = "not_a_speech_model", .param = "model" },
        .{ .body = "{\"input\": \"Hi.\", \"model\": \"laya\"}", .status = .bad_request, .code = "not_a_speech_model", .param = "model" },
        .{ .body = "{\"input\": \"Hi.\", \"model\": \"tts-1\"}", .status = .not_found, .code = "model_not_found", .param = "model" },
        .{ .body = "{\"input\": \"Hi.\", \"model\": \"/etc/voice\"}", .status = .not_found, .code = "model_not_found", .param = "model" },
        .{ .body = "{\"input\": \"Hi.\", \"model\": \"qwen3-tts-1.7b-customvoice\"}", .status = .not_found, .code = "model_not_found", .param = "model" },
        .{ .body = "{\"input\": \"Hi.\", \"instructions\": \"Whisper.\"}", .status = .bad_request, .code = "instructions_unsupported", .param = "instructions" },
        // A saved voice needs a Base; a Base needs a saved voice, and takes no instruction.
        .{ .body = "{\"input\": \"Hi.\", \"voice\": {\"id\": \"anna\"}}", .status = .bad_request, .code = "voice_cloning_unsupported", .param = "voice" },
        .{ .body = "{\"input\": \"Hi.\", \"model\": \"qwen3-tts-0.6b-base\"}", .status = .bad_request, .code = "invalid_request", .param = "voice" },
        .{ .body = "{\"input\": \"Hi.\", \"model\": \"qwen3-tts-0.6b-base\", \"voice\": \"nobody\"}", .status = .not_found, .code = "voice_not_found", .param = "voice" },
        .{ .body = "{\"input\": \"Hi.\", \"model\": \"qwen3-tts-0.6b-base\", \"voice\": {\"id\": \"Not A Name\"}}", .status = .not_found, .code = "voice_not_found", .param = "voice" },
        .{ .body = "{\"input\": \"Hi.\", \"model\": \"qwen3-tts-0.6b-base\", \"voice\": \"anna\", \"instructions\": \"Whisper.\"}", .status = .bad_request, .code = "instructions_unsupported", .param = "instructions" },
    };
    for (cases) |case| {
        errdefer std.debug.print("case: {s}\n", .{case.body});
        try expectError(f.service.answer(arena, io, case.body, null, .{}), case.status, case.code, case.param);
    }
    try std.testing.expectEqual(Stats{ .waiting = 0, .speaking = false, .requests = 0 }, f.service.stats(io));
}

test "speech: a full queue is busy; a request that never starts times out and leaves the queue" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const io = std.testing.io;
    {
        var f: Fixture = undefined;
        try f.init(0);
        defer f.deinit();
        try expectError(f.service.answer(arena, io, "{\"input\": \"Hi.\"}", null, .{}), http.overloaded, "busy", null);
    }
    {
        // No worker runs: the job waits until its deadline, then leaves.
        var f: Fixture = undefined;
        try f.init(4);
        defer f.deinit();
        try expectError(f.service.answer(arena, io, "{\"input\": \"Hi.\"}", null, .{}), http.overloaded, "timeout", null);
        try std.testing.expectEqual(@as(usize, 0), f.service.stats(io).waiting);
        try std.testing.expect(f.service.head == null);
    }
}

test "speech: a voice that fails to open fails its request before the head, and the next one too" {
    const io = std.testing.io;
    var f: Fixture = undefined;
    try f.init(4);
    defer f.deinit();
    f.service.timeout_ns = 60 * std.time.ns_per_s;
    var budget: @import("../memory.zig").Budget = .init(std.testing.allocator, 1 << 30);
    defer budget.deinit();
    f.service.slot.budget = &budget;
    var worker = try io.concurrent(gpu.Executor.run, .{ &f.executor, io });
    defer {
        f.executor.stop(io);
        worker.await(io);
    }
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    for (0..2) |_| try expectError(f.service.answer(arena, io, "{\"input\": \"Hi.\", \"response_format\": \"pcm\"}", null, .{}), .internal_server_error, "model_failed", "model");
    const stats = f.service.stats(io);
    try std.testing.expectEqual(@as(u64, 2), stats.requests);
    try std.testing.expect(!stats.speaking and stats.waiting == 0);
    try std.testing.expectEqual(@as(u64, 0), budget.resident(io));

    // The listing names the six speech models, pulled or not, the default
    // marked; only CustomVoice has speakers.
    var listing: std.ArrayList(models.Entry) = .empty;
    try Service.list(&f.service, arena, io, &listing);
    var seen: usize = 0;
    for (listing.items) |e| {
        const d = e.details.object;
        try std.testing.expectEqualStrings("speech", d.get("kind").?.string);
        const drama3 = std.mem.eql(u8, e.id, "drama-3");
        try std.testing.expectEqualStrings(if (drama3) "sakasegawa" else "Qwen", e.owned_by);
        try std.testing.expectEqual(@as(i64, if (drama3) 44100 else 24000), d.get("sample_rate").?.integer);
        const small = std.mem.eql(u8, e.id, "qwen3-tts-0.6b-customvoice");
        const speakers = std.mem.endsWith(u8, e.id, "-customvoice");
        // The fixture holds the 0.6B CustomVoice's files and the 0.6B Base's.
        try std.testing.expectEqual(small or std.mem.eql(u8, e.id, "qwen3-tts-0.6b-base"), d.get("present").?.bool);
        try std.testing.expectEqual(small, d.get("default").?.bool);
        try std.testing.expectEqual(std.mem.indexOf(u8, e.id, "1.7b-customvoice") != null or std.mem.endsWith(u8, e.id, "-voicedesign"), d.get("instructions").?.bool);
        try std.testing.expectEqual(@as(usize, if (speakers) 9 else 0), d.get("voices").?.array.items.len);
        // A Base clones (from a clip, or a clip and its words); VoiceDesign designs; both before the pull.
        try std.testing.expectEqual(std.mem.endsWith(u8, e.id, "-base") or drama3, d.get("cloning").? != .null);
        try std.testing.expectEqual(std.mem.endsWith(u8, e.id, "-voicedesign"), d.get("design").?.bool);
        if (std.mem.endsWith(u8, e.id, "-base") or drama3) try std.testing.expectEqual(@as(usize, if (drama3) 1 else 2), d.get("cloning").?.object.get("modes").?.array.items.len);
        seen += 1;
    }
    try std.testing.expectEqual(@as(usize, 6), seen);
}

test "a speech model's voice has no default when speech.voice is not one of its speakers" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const voices = try models.strings(arena, &.{"egyptian_speaker"});
    const fine_tune = try speechOptions(arena, voices, &.{"english"}, true, "ryan", true);
    try std.testing.expect(fine_tune.get("voice").?.object.get("default").? == .null);
    try std.testing.expectEqual(@as(usize, 2), fine_tune.get("language").?.object.get("values").?.array.items.len);
    try std.testing.expect(fine_tune.get("instructions") != null);
    const own = try speechOptions(arena, voices, &.{}, false, "egyptian_speaker", true);
    // VoiceDesign takes no voice.
    const designed = try speechOptions(arena, .init(arena), &.{}, true, "ryan", false);
    try std.testing.expect(designed.get("voice") == null and designed.get("instructions") != null);
    try std.testing.expectEqualStrings("egyptian_speaker", own.get("voice").?.object.get("default").?.string);
    try std.testing.expect(own.get("instructions") == null);
}
