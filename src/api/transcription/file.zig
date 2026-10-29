//! `POST /v1/audio/transcriptions`: OpenAI's transcription call for a whole
//! recording, an hour or more. The handler reads the multipart upload
//! (`multipart.zig`) and opens a piece-by-piece decoder on its connection's
//! task, then queues a job; one short GPU item runs the jobs in arrival
//! order, one 2.56 s piece per run (`audio/transcript.zig`), so a long file
//! interleaves with chat steps and speech. Memory is the upload, the
//! transcriber's caches, and the segments, never the decoded samples. The
//! answer is the whole transcript in the asked format, or with
//! `stream=true` server-sent events as segments become final. A client that
//! leaves cancels between pieces. docs/guide/api.md § Transcription (files).
const std = @import("std");
const inference = @import("inference");
const config = @import("../../config.zig");
const catalog = @import("../../catalog.zig");
const audio_catalog = @import("../../audio/catalog.zig");
const transcript = @import("../../audio/transcript.zig");
const http = @import("../http.zig");
const errors = @import("../errors.zig");
const gpu = @import("../gpu.zig");
const multipart = @import("../multipart.zig");
const router_mod = @import("../router.zig");
const Pipe = @import("../pipe.zig").Pipe;
const service_mod = @import("service.zig");

const audio = inference.audio.audio;
const asr = inference.audio.nemotron_asr;
const lookaheads = inference.audio.nemotron_encoder.lookaheads;
const boost = inference.audio.boost;
const ApiError = errors.ApiError;

pub const path = router_mod.prefix ++ "/audio/transcriptions";
/// Files waiting for the transcriber at once; one more is `busy`. Each
/// holds its upload in memory while it waits.
pub const max_waiting = 16;
/// Bytes an event stream may hold between the worker and the socket; the
/// job parks rather than block the worker.
const pipe_capacity = 64 * 1024;
/// A form field's value, every one but the file and `phrases`.
const max_field_bytes = 1024;
const max_phrases_bytes = 8192;
const phrases_message = std.fmt.comptimePrint("phrases is a JSON array of at most {d} words or names, each at most {d} bytes", .{ boost.max_phrases, boost.max_phrase_bytes });
const poll_ms = 1000;
const keepalive_ms = 10_000;
const progress_ms = 1000;
/// Beside a generation, a file takes at most one part in this many of the
/// GPU's time: it has no deadline, and the generation waits for each piece.
const beside_share = 4;

pub const Format = enum { json, text, verbose_json, srt, vtt };

/// A validated request; its strings are slices of the body.
pub const Request = struct {
    file: []const u8,
    model: ?[]const u8 = null,
    /// A `prompt_dictionary` key; null is `auto`.
    language: ?[]const u8 = null,
    format: Format = .json,
    segments: bool = true,
    words: bool = false,
    stream: bool = false,
    lookahead: usize = 3,
    /// Words to lean toward (`phrases`, a JSON array), in the arena.
    phrases: []const []const u8 = &.{},
};

/// Why a request was refused before it was queued.
pub const Problem = struct {
    code: []const u8 = "invalid_request",
    param: ?[]const u8 = null,
    message: []const u8 = "",
};

/// Reads the form: `file`, `model`, `language`, `response_format`,
/// `timestamp_granularities[]`, `stream`, and nuclis's `lookahead` and
/// `phrases`. Other
/// fields OpenAI defines are ignored, but for those that would change the
/// text (`prompt`, `include[]`), which are refused.
pub fn parse(arena: std.mem.Allocator, body: []const u8, content_type: ?[]const u8, problem: *Problem) error{ Refused, OutOfMemory }!Request {
    const form = multipart.parse(arena, body, content_type orelse "", .{}) catch |err| return refuse(problem, null, switch (err) {
        error.NotMultipart => "the body must be multipart/form-data, with the audio as the part named file",
        error.InvalidBoundary => "the multipart content type names no usable boundary",
        error.TooManyParts => "more than 32 form fields",
        error.HeaderTooLarge => "a form part's headers exceed 1 KiB",
        error.MissingName => "a form part has no content-disposition name",
        error.TooManyFiles => "more than one file part: send one recording per request",
        error.Malformed => "the multipart body is malformed or truncated",
        error.OutOfMemory => return error.OutOfMemory,
    });
    var request: Request = .{ .file = "" };
    const file = form.file() orelse return refuse(problem, "file", "no file part: send the recording as the part named file");
    if (!std.mem.eql(u8, file.name, "file")) return refuse(problem, "file", "the file part must be named file");
    request.file = file.data;
    var granularities = false;
    request.segments = false;
    for (form.parts) |part| {
        if (part.filename != null) continue;
        const name = part.name;
        const value = std.mem.trim(u8, part.data, " \t\r\n");
        const max_bytes: usize = if (std.mem.eql(u8, name, "phrases")) max_phrases_bytes else max_field_bytes;
        if (value.len > max_bytes) return refusef(arena, problem, "invalid_request", name, "{s} is longer than {d} bytes", .{ name, max_bytes });
        if (std.mem.eql(u8, name, "file")) {
            return refuse(problem, "file", "file must be a file part (with a filename), not a field");
        } else if (std.mem.eql(u8, name, "model")) {
            if (value.len > 0) request.model = value;
        } else if (std.mem.eql(u8, name, "language")) {
            if (value.len > 0 and !std.mem.eql(u8, value, "auto")) request.language = value;
        } else if (std.mem.eql(u8, name, "response_format")) {
            request.format = std.meta.stringToEnum(Format, value) orelse
                return refusef(arena, problem, "unsupported_feature", name, "response_format takes json, text, verbose_json, srt, or vtt, not {s}", .{value});
        } else if (std.mem.eql(u8, name, "timestamp_granularities[]") or std.mem.eql(u8, name, "timestamp_granularities")) {
            granularities = true;
            if (std.mem.eql(u8, value, "word")) {
                request.words = true;
            } else if (std.mem.eql(u8, value, "segment")) {
                request.segments = true;
            } else return refusef(arena, problem, "invalid_request", "timestamp_granularities", "timestamp_granularities takes word or segment, not {s}", .{value});
        } else if (std.mem.eql(u8, name, "stream")) {
            request.stream = parseBool(value) orelse return refusef(arena, problem, "invalid_request", name, "stream takes true or false, not {s}", .{value});
        } else if (std.mem.eql(u8, name, "lookahead")) {
            const n = std.fmt.parseInt(usize, value, 10) catch std.math.maxInt(usize);
            if (std.mem.indexOfScalar(usize, &lookaheads, n) == null)
                return refusef(arena, problem, "invalid_request", name, "lookahead takes 0, 3, 6, or 13 (rows of 80 ms), not {s}", .{value});
            request.lookahead = n;
        } else if (std.mem.eql(u8, name, "phrases")) {
            request.phrases = parsePhrases(arena, value) catch |err| switch (err) {
                error.OutOfMemory => return error.OutOfMemory,
                error.Invalid => return refuse(problem, name, phrases_message),
            };
        } else if (std.mem.eql(u8, name, "prompt")) {
            if (value.len > 0) return refusef(arena, problem, "unsupported_feature", name, "the model takes no prompt text; language is its only hint", .{});
        } else if (std.mem.eql(u8, name, "include[]") or std.mem.eql(u8, name, "include")) {
            return refusef(arena, problem, "unsupported_feature", "include", "include ({s}) is not given: the decoder is greedy and keeps no log probabilities", .{value});
        }
    }
    // OpenAI's default: segments in verbose_json unless only words were asked for.
    if (!granularities) request.segments = true;
    if (granularities and request.format != .verbose_json and !request.stream)
        return refuse(problem, "timestamp_granularities", "timestamp_granularities needs response_format verbose_json (or stream)");
    if (request.stream and (request.format == .srt or request.format == .vtt))
        return refusef(arena, problem, "unsupported_feature", "stream", "stream gives events, not {s}: ask for {s} without stream", .{ @tagName(request.format), @tagName(request.format) });
    return request;
}

fn parsePhrases(arena: std.mem.Allocator, value: []const u8) error{ Invalid, OutOfMemory }![]const []const u8 {
    const list = std.json.parseFromSliceLeaky([]const []const u8, arena, value, .{}) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.Invalid,
    };
    if (list.len > boost.max_phrases) return error.Invalid;
    for (list) |p| if (!boost.valid(p)) return error.Invalid;
    return list;
}

fn parseBool(value: []const u8) ?bool {
    if (std.ascii.eqlIgnoreCase(value, "true") or std.mem.eql(u8, value, "1")) return true;
    if (std.ascii.eqlIgnoreCase(value, "false") or std.mem.eql(u8, value, "0")) return false;
    return null;
}

fn refuse(problem: *Problem, param: ?[]const u8, message: []const u8) error{Refused} {
    problem.* = .{ .param = param, .message = message };
    return error.Refused;
}

fn refusef(arena: std.mem.Allocator, problem: *Problem, code: []const u8, param: ?[]const u8, comptime format: []const u8, args: anytype) error{ Refused, OutOfMemory } {
    problem.* = .{ .code = code, .param = param, .message = try arena.print(format, args) };
    return error.Refused;
}

/// The route's queue and its one item; embedded in the transcription
/// service, which must not move once a request was queued.
pub const Files = struct {
    service: *service_mod.Service = undefined,
    /// `serve.transcription_bytes`: the route's body limit.
    max_bytes: usize = transcript.max_file_bytes,
    /// `serve.timeout`: how long a file may wait before it starts.
    timeout_ns: u64 = 300 * std.time.ns_per_s,

    /// Guards the list, `scheduled`, `parked`, `requests`, and each job's `state`.
    mutex: std.Io.Mutex = .init,
    head: ?*Job = null,
    tail: ?*Job = null,
    waiting: usize = 0,
    scheduled: bool = false,
    /// The job running waits for its client to take its events.
    parked: bool = false,
    requests: u64 = 0,
    /// Worker-only: the job running.
    active: ?*Job = null,
    /// Worker-only: when the last piece ended and how long it took.
    last_end: ?std.Io.Timestamp = null,
    last_ns: u64 = 0,
    item: gpu.Item = .{ .run = step, .short = true },

    pub fn register(self: *Files, gpa: std.mem.Allocator, router: *router_mod.Router, service: *service_mod.Service) !void {
        self.service = service;
        try router.addLimited(gpa, .POST, path, .{ .context = self, .handle = handle }, self.max_bytes);
    }

    fn handle(context: *anyopaque, arena: std.mem.Allocator, io: std.Io, request: http.Request) http.Response {
        const self: *Files = @ptrCast(@alignCast(context));
        return self.answer(arena, io, request.body, request.content_type, request.peer);
    }

    /// Validates, queues, and answers: the whole transcript once it is
    /// done, or the event stream once the job started. Every failure is a
    /// response.
    pub fn answer(self: *Files, arena: std.mem.Allocator, io: std.Io, body: []const u8, content_type: ?[]const u8, peer: ?http.Peer) http.Response {
        const service = self.service;
        const gpa = service.slot.gpa;
        const deadline = std.Io.Clock.Timestamp.fromNow(io, .{ .raw = .fromNanoseconds(self.timeout_ns), .clock = .awake });
        var problem: Problem = .{};
        const parsed = parse(arena, body, content_type, &problem) catch |err| return switch (err) {
            error.Refused => refusal(arena, .bad_request, problem.code, problem.param, "{s}", .{problem.message}),
            error.OutOfMemory => internal(arena),
        };
        const open = service.slot.openName(io, arena) catch null;
        const name = parsed.model orelse open orelse service.default_model;
        var diag: config.Diagnostic = .{};
        const located = audio_catalog.locate(arena, io, service.root, service.registry, name, .transcription, false, &diag) catch |err| return switch (err) {
            error.WrongKind => refusal(arena, .bad_request, "not_a_transcription_model", "model", "{s} is a {s} model; this route transcribes with a transcription model ({s})", .{ name, catalog.kindName(catalog.kindOf(service.registry, name)), service.default_model }),
            error.ModelNotFound, error.ModelFileNotFound => refusal(arena, .not_found, "model_not_found", "model", "{s}", .{diag.message()}),
            error.OutOfMemory => internal(arena),
            else => refusal(arena, .internal_server_error, "model_failed", "model", "{s}: {s}", .{ name, if (diag.len > 0) diag.message() else @errorName(err) }),
        };
        const reader = audio.Reader.open(parsed.file) catch |err| return switch (err) {
            error.OutOfMemory => internal(arena),
            else => refusal(arena, .bad_request, "invalid_audio", "file", "the file is not audio nuclis can read ({s}; or a 16 kHz mono WAV)", .{audio.readable}),
        };
        var pipe: ?Pipe = if (parsed.stream) Pipe.init(gpa, pipe_capacity) catch null else null;
        if (parsed.stream and pipe == null) {
            var r = reader;
            r.close();
            return internal(arena);
        }
        const job = arena.create(Job) catch {
            var r = reader;
            r.close();
            if (pipe) |*p| p.deinit(gpa);
            return internal(arena);
        };
        job.* = .{
            .files = self,
            .request = parsed,
            .directory = located.directory,
            .name = name,
            .bytes = located.bytes,
            .reader = reader,
            .segmenter = .init(gpa, parsed.language),
            .pipe = pipe,
            .note_buffer = &.{},
            .message_buffer = &.{},
            .peer = peer,
        };
        job.note_buffer = arena.alloc(u8, 512) catch {
            job.free(gpa);
            return internal(arena);
        };
        job.message_buffer = arena.alloc(u8, 1024) catch {
            job.free(gpa);
            return internal(arena);
        };
        self.submit(io, job) catch |err| {
            job.free(gpa);
            return switch (err) {
                error.Busy => refusal(arena, http.overloaded, "busy", null, "overloaded: {d} transcription requests wait already; retry shortly", .{max_waiting}),
                error.Stopped => refusal(arena, .service_unavailable, "shutting_down", null, "the server is stopping", .{}),
            };
        };
        const event = if (parsed.stream) &job.ready else &job.done;
        switch (job.wait(io, event, deadline)) {
            .ready => {},
            .timed_out => {
                job.release(io);
                return refusal(arena, http.overloaded, "timeout", null, "overloaded: waited {d} s for the GPU without starting (timeout)", .{self.timeout_ns / std.time.ns_per_s});
            },
            .gone => {
                job.release(io);
                return refusal(arena, .bad_request, "cancelled", null, "the client closed the connection", .{});
            },
        }
        if (job.failure) |failure| {
            job.release(io);
            return .fromError(arena, failure);
        }
        if (parsed.stream) return .{
            .body = "",
            .content_type = "text/event-stream",
            .stream = .{ .context = job, .write = Job.writeStream, .note = Job.streamNote },
        };
        defer job.release(io);
        const out = job.writeResponse(arena) catch return internal(arena);
        return .{ .body = out, .content_type = contentType(parsed.format), .note = job.note };
    }

    fn submit(self: *Files, io: std.Io, job: *Job) error{ Busy, Stopped }!void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (self.waiting >= max_waiting) return error.Busy;
        if (!self.scheduled and !self.parked) {
            try self.service.executor.submit(io, &self.item);
            self.scheduled = true;
        }
        if (self.tail) |t| t.next = job else self.head = job;
        self.tail = job;
        self.waiting += 1;
        job.state = .queued;
    }

    /// Takes `job` out of the queue if it has not started: true when the
    /// worker will never touch it.
    fn withdraw(self: *Files, io: std.Io, job: *Job) bool {
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
    fn unpark(self: *Files, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (!self.parked) return;
        self.service.executor.submit(io, &self.item) catch return;
        self.parked = false;
        self.scheduled = true;
    }

    /// The item: one piece of the job running, taking the next when it ends.
    fn step(item: *gpu.Item, io: std.Io) gpu.Item.After {
        const self: *Files = @alignCast(@fieldParentPtr("item", item));
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
        if (job.run == null and !job.over) {
            // Opening the model may close the language model a generation
            // is running: it waits for the generation, not between its steps.
            if (item.short and !self.service.slot.isOpen(io, job.directory)) {
                item.short = false;
                return .again;
            }
            item.short = true;
            self.start(io, job) catch |err| {
                job.failure = job.startFailure(err);
                job.ready.set(io);
                self.finish(io, job);
                return self.afterJob(io);
            };
            job.ready.set(io);
        }
        // Parked rather than blocked: the worker never waits on a client.
        if (!job.flush(io)) {
            self.mutex.lockUncancelable(io);
            defer self.mutex.unlock(io);
            self.parked = true;
            self.scheduled = false;
            return .done;
        }
        if (job.over) {
            self.finish(io, job);
            return self.afterJob(io);
        }
        if (self.service.executor.betweenSteps()) if (self.last_end) |end| {
            const since: u64 = @intCast(@max(0, end.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds()));
            if (since < (beside_share - 1) * self.last_ns) return .again;
        };
        const started = std.Io.Clock.awake.now(io);
        job.advance(io);
        const ended = std.Io.Clock.awake.now(io);
        self.last_ns = nanos(started, ended);
        self.last_end = ended;
        return .again;
    }

    /// After a job ended: `.again` while another waits.
    fn afterJob(self: *Files, io: std.Io) gpu.Item.After {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (self.head != null) return .again;
        self.scheduled = false;
        return .done;
    }

    /// Opens the model if needed, holds it, and sets the job's transcriber.
    fn start(self: *Files, io: std.Io, job: *Job) !void {
        const slot = &self.service.slot;
        const gpa = slot.gpa;
        job.started = std.Io.Clock.awake.now(io);
        job.last_progress = job.started;
        const acquired = try slot.acquire(io, job.directory, job.name, job.bytes);
        job.load_ns = acquired.load_ns;
        const model = acquired.model;
        const prompt = if (job.request.language) |l| model.promptId(l) orelse return error.UnknownLanguage else model.default_prompt;
        slot.hold(io, true);
        job.holding = true;
        errdefer {
            slot.hold(io, false);
            job.holding = false;
        }
        job.transcriber = try asr.Transcriber.init(gpa, model, job.request.lookahead, prompt, if (model.metal != null) .metal else .cpu);
        errdefer {
            job.transcriber.?.deinit();
            job.transcriber = null;
        }
        if (job.request.phrases.len > 0) {
            job.phrases = try model.boost(gpa, job.request.phrases, boost.default_bonus, boost.default_follow);
            job.transcriber.?.setBoost(&job.phrases.?);
        }
        errdefer if (job.phrases) |*p| {
            p.deinit();
            job.phrases = null;
        };
        job.run = try transcript.Run.init(gpa, &job.reader, &job.transcriber.?, &job.segmenter);
    }

    /// The job is over: its transcriber freed, the model no longer held,
    /// the stream closed, its waiter woken.
    fn finish(self: *Files, io: std.Io, job: *Job) void {
        const gpa = self.service.slot.gpa;
        job.writeNote();
        if (job.run) |*r| r.deinit(gpa);
        job.run = null;
        if (job.transcriber) |*t| t.deinit();
        job.transcriber = null;
        if (job.phrases) |*p| p.deinit();
        job.phrases = null;
        if (job.holding) self.service.slot.hold(io, false);
        job.holding = false;
        job.pending.deinit(gpa);
        job.pending = .empty;
        if (job.pipe) |*p| p.close(io);
        self.active = null;
        {
            self.mutex.lockUncancelable(io);
            defer self.mutex.unlock(io);
            job.state = .finished;
            self.requests += 1;
        }
        job.done.set(io);
    }
};

/// One request: queued on the connection's task, transcribed on the
/// worker. It lives in the connection's arena; the worker allocates only
/// from the service's allocator, and the connection reads the segments
/// once the worker let go of the job.
const Job = struct {
    files: *Files,
    next: ?*Job = null,
    request: Request,
    directory: []const u8,
    name: []const u8,
    bytes: u64,
    /// Opened on the connection's task, read by the worker, closed by `free`.
    reader: audio.Reader,
    segmenter: transcript.Segmenter,
    /// With `stream`.
    pipe: ?Pipe,
    peer: ?http.Peer,
    /// Set by the connection when its client is gone; the worker stops
    /// before the next piece.
    cancel: std.atomic.Value(bool) = .init(false),
    /// Guarded by the files' mutex.
    state: enum { idle, queued, active, finished } = .idle,
    /// Set once the job started on the worker, or failed before.
    ready: std.Io.Event = .unset,
    /// Set once the worker can no longer touch the job.
    done: std.Io.Event = .unset,
    failure: ?ApiError = null,
    note: []const u8 = "",
    note_buffer: []u8,
    message_buffer: []u8,

    // Worker-only, while it runs.
    transcriber: ?asr.Transcriber = null,
    /// The phrases the transcriber borrows.
    phrases: ?boost.Boost = null,
    run: ?transcript.Run = null,
    holding: bool = false,
    /// No piece is left: the job ends once `pending` reached the pipe.
    over: bool = false,
    /// Event bytes the pipe has no room for yet.
    pending: std.ArrayList(u8) = .empty,
    /// Segments already sent as events.
    emitted: usize = 0,
    started: std.Io.Timestamp = undefined,
    last_progress: std.Io.Timestamp = undefined,
    load_ns: u64 = 0,
    first_segment_ns: ?u64 = null,
    elapsed_ns: u64 = 0,
    /// The samples transcribed, kept from the run when `finish` frees it.
    fed: u64 = 0,

    /// One piece: decoded, transcribed, segmented, and, for a stream, the
    /// events of the segments it finalized queued.
    fn advance(self: *Job, io: std.Io) void {
        const run = &self.run.?;
        const finalized = run.step() catch |err| {
            self.failure = switch (err) {
                error.UnsupportedAudioFormat => self.apiError(.bad_request, "invalid_audio", "file", "the audio stops decoding after {d:.1} s", .{run.seconds()}),
                else => self.apiError(.internal_server_error, "model_failed", null, "the transcription failed after {d:.1} s of audio: {s}", .{ run.seconds(), @errorName(err) }),
            };
            if (self.pipe != null) self.queueError();
            self.over = true;
            return;
        };
        const now = std.Io.Clock.awake.now(io);
        if (finalized > 0 and self.first_segment_ns == null) self.first_segment_ns = nanos(self.started, now);
        self.over = run.finished;
        if (self.over) self.elapsed_ns = nanos(self.started, now);
        if (self.pipe == null) return;
        self.queueEvents(now) catch {
            self.cancel.store(true, .release);
        };
    }

    fn queueEvents(self: *Job, now: std.Io.Timestamp) !void {
        const gpa = self.files.service.slot.gpa;
        var out: std.Io.Writer.Allocating = .fromArrayList(gpa, &self.pending);
        defer self.pending = out.toArrayList();
        const w = &out.writer;
        const segments = self.segmenter.segments.items;
        for (segments[self.emitted..]) |s| {
            try w.writeAll("data: ");
            var json: std.json.Stringify = .{ .writer = w };
            try json.beginObject();
            try json.objectField("type");
            try json.write("transcript.text.delta");
            try json.objectField("delta");
            try json.beginWriteRaw();
            try w.writeAll(if (s.id > 0) "\" " else "\"");
            try std.json.Stringify.encodeJsonStringChars(s.text, .{}, w);
            try w.writeByte('"');
            json.endWriteRaw();
            try json.endObject();
            try w.writeAll("\n\ndata: ");
            json = .{ .writer = w };
            try json.beginObject();
            try json.objectField("type");
            try json.write("transcript.text.segment");
            try transcript.writeSegmentFields(&json, &self.segmenter, s, self.request.words);
            try json.endObject();
            try w.writeAll("\n\n");
        }
        self.emitted = segments.len;
        const run = &self.run.?;
        if (self.over) {
            try w.writeAll("data: ");
            var json: std.json.Stringify = .{ .writer = w };
            try json.beginObject();
            try json.objectField("type");
            try json.write("transcript.text.done");
            try json.objectField("text");
            try json.beginWriteRaw();
            try w.writeByte('"');
            for (segments, 0..) |s, i| {
                if (i > 0) try w.writeByte(' ');
                try std.json.Stringify.encodeJsonStringChars(s.text, .{}, w);
            }
            try w.writeByte('"');
            json.endWriteRaw();
            try json.objectField("usage");
            try writeUsage(&json, run.seconds());
            try json.endObject();
            try w.writeAll("\n\n");
        } else if (nanos(self.last_progress, now) >= progress_ms * std.time.ns_per_ms) {
            self.last_progress = now;
            try w.print(": progress {d:.2} of {d:.2} s\n\n", .{ run.seconds(), @as(f64, @floatFromInt(self.reader.total)) / audio.sample_rate });
        }
    }

    fn queueError(self: *Job) void {
        const gpa = self.files.service.slot.gpa;
        var out: std.Io.Writer.Allocating = .fromArrayList(gpa, &self.pending);
        defer self.pending = out.toArrayList();
        out.writer.writeAll("data: ") catch return;
        errors.write(&out.writer, self.failure.?) catch return;
        out.writer.writeAll("\n") catch return;
    }

    /// Moves what fits of `pending` into the pipe; true once nothing waits.
    fn flush(self: *Job, io: std.Io) bool {
        const pipe = if (self.pipe) |*p| p else return true;
        if (self.pending.items.len == 0) return true;
        const n = @min(pipe.space(io), self.pending.items.len);
        // Within the free space, a write never waits.
        pipe.write(io, self.pending.items[0..n]) catch {
            self.cancel.store(true, .release);
            self.pending.clearRetainingCapacity();
            return true;
        };
        const rest = self.pending.items.len - n;
        std.mem.copyForwards(u8, self.pending.items[0..rest], self.pending.items[n..]);
        self.pending.shrinkRetainingCapacity(rest);
        return rest == 0;
    }

    /// The whole response once the worker let go: the transcript as asked.
    fn writeResponse(self: *Job, arena: std.mem.Allocator) ![]const u8 {
        var out: std.Io.Writer.Allocating = .init(arena);
        const w = &out.writer;
        const segments = self.segmenter.segments.items;
        const seconds = @as(f64, @floatFromInt(self.audioSamples())) / audio.sample_rate;
        switch (self.request.format) {
            .text => {
                try self.segmenter.writeText(w);
                try w.writeByte('\n');
            },
            .srt => try transcript.writeSrt(w, segments),
            .vtt => try transcript.writeVtt(w, segments),
            .json, .verbose_json => {
                const verbose = self.request.format == .verbose_json;
                var json: std.json.Stringify = .{ .writer = w, .options = .{ .whitespace = .indent_2 } };
                try json.beginObject();
                if (verbose) {
                    try json.objectField("task");
                    try json.write("transcribe");
                    try json.objectField("language");
                    try json.write(self.language());
                    try json.objectField("duration");
                    try json.write(seconds);
                }
                try json.objectField("text");
                var text: std.Io.Writer.Allocating = .init(arena);
                try self.segmenter.writeText(&text.writer);
                try json.write(text.written());
                if (verbose and self.request.segments) {
                    try json.objectField("segments");
                    try json.beginArray();
                    for (segments) |s| {
                        try json.beginObject();
                        try transcript.writeSegmentFields(&json, &self.segmenter, s, false);
                        try json.endObject();
                    }
                    try json.endArray();
                }
                if (verbose and self.request.words) {
                    try json.objectField("words");
                    try transcript.writeWords(&json, self.segmenter.words.items);
                }
                try json.objectField("usage");
                try writeUsage(&json, seconds);
                try json.endObject();
                try w.writeByte('\n');
            },
        }
        return out.written();
    }

    /// The samples transcribed: the run's count, kept past its end.
    fn audioSamples(self: *const Job) u64 {
        return self.fed;
    }

    /// The last tag's locale, or the prompt's.
    fn language(self: *const Job) ?[]const u8 {
        return self.segmenter.language;
    }

    fn startFailure(self: *Job, err: anyerror) ApiError {
        return switch (err) {
            error.UnknownLanguage => self.apiError(.bad_request, "invalid_request", "language", "{s} is not a language of {s}: a locale (en-US, de-DE, fr-FR, es-US, …), a two-letter code (en, de, fr, …), or auto", .{ self.request.language.?, self.name }),
            error.ModelTooLarge => self.apiError(.bad_request, "model_too_large", "model", "{s} needs {d:.1} GiB, more than the server's budget leaves beside the pinned models (serve.memory_bytes, --memory, --pin)", .{ self.name, @as(f64, @floatFromInt(self.bytes)) / (1 << 30) }),
            error.Pinned => self.apiError(http.overloaded, "busy", "model", "overloaded: {s} needs the transcriber another request holds; retry shortly", .{self.name}),
            else => self.apiError(.internal_server_error, "model_failed", "model", "{s} did not transcribe: {s}", .{ self.name, @errorName(err) }),
        };
    }

    fn apiError(self: *Job, status: std.http.Status, code: []const u8, param: ?[]const u8, comptime format: []const u8, args: anytype) ApiError {
        var err: ApiError = .init(status, code, std.fmt.bufPrint(self.message_buffer, format, args) catch code);
        err.param = param;
        return err;
    }

    fn writeNote(self: *Job) void {
        if (self.run) |r| self.fed = r.fed;
        const seconds = @as(f64, @floatFromInt(self.fed)) / audio.sample_rate;
        const ms = @as(f64, @floatFromInt(self.elapsed_ns)) / std.time.ns_per_ms;
        self.note = if (self.failure) |f|
            std.fmt.bufPrint(self.note_buffer, "{s}: {s}", .{ f.code, f.message }) catch f.code
        else if (self.cancel.load(.acquire))
            std.fmt.bufPrint(self.note_buffer, "{s} · cancelled after {d:.1} s of audio", .{ self.name, seconds }) catch "cancelled"
        else
            std.fmt.bufPrint(self.note_buffer, "{s} · {d:.1} s of audio in {d:.2} s ({d:.1}× real time) · {d} segments · first segment {d:.0} ms{s}", .{
                self.name,
                seconds,
                ms / 1000,
                if (ms > 0) seconds * 1000 / ms else 0,
                self.segmenter.segments.items.len,
                if (self.first_segment_ns) |ns| @as(f64, @floatFromInt(ns)) / std.time.ns_per_ms else 0,
                if (self.load_ns > 0) " · opened the model" else "",
            }) catch self.name;
    }

    const Waited = enum { ready, timed_out, gone };

    /// Waits for `event` (the start, or the end), looking at the client
    /// every second; a job still queued at `deadline` is withdrawn.
    fn wait(self: *Job, io: std.Io, event: *std.Io.Event, deadline: std.Io.Clock.Timestamp) Waited {
        const files = self.files;
        while (true) {
            event.waitTimeout(io, .{ .deadline = fromNow(io, poll_ms) }) catch {};
            if (event.isSet()) return .ready;
            if (self.peer) |peer| if (peer.gone()) {
                self.cancel.store(true, .release);
                files.unpark(io);
                return .gone;
            };
            if (pastDeadline(io, deadline) and files.withdraw(io, self)) return .timed_out;
        }
    }

    /// The connection's side of the stream: drain the pipe into the body,
    /// wake a parked job once its client took bytes, keep the stream alive,
    /// and stop the job when the client goes.
    fn writeStream(context: *anyopaque, io: std.Io, body: *http.Body) bool {
        const self: *Job = @ptrCast(@alignCast(context));
        defer self.release(io);
        const pipe = &self.pipe.?;
        var buffer: [16 * 1024]u8 = undefined;
        var quiet_since = std.Io.Clock.awake.now(io);
        while (true) {
            if (self.peer) |peer| if (peer.gone()) return self.abandon(io);
            const taken = pipe.take(io, &buffer, fromNow(io, poll_ms)) catch return self.abandon(io);
            switch (taken) {
                .closed => return true,
                .data => |bytes| {
                    body.send(bytes) catch return self.abandon(io);
                    quiet_since = std.Io.Clock.awake.now(io);
                    self.files.unpark(io);
                },
                .timeout => if (quiet_since.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds() >= keepalive_ms) {
                    body.send(": transcribing\n\n") catch return self.abandon(io);
                    quiet_since = std.Io.Clock.awake.now(io);
                },
            }
        }
    }

    fn abandon(self: *Job, io: std.Io) bool {
        self.cancel.store(true, .release);
        self.pipe.?.abandon(io);
        self.files.unpark(io);
        return false;
    }

    /// Waits until the worker can no longer touch the job (or takes it back
    /// unstarted), then frees what it holds.
    fn release(self: *Job, io: std.Io) void {
        if (!self.files.withdraw(io, self)) {
            while (true) {
                self.done.waitTimeout(io, .{ .deadline = fromNow(io, poll_ms) }) catch {};
                if (self.done.isSet()) break;
                // A parked job finishes only once scheduled again.
                if (self.cancel.load(.acquire)) self.files.unpark(io);
            }
        }
        self.free(self.files.service.slot.gpa);
    }

    fn free(self: *Job, gpa: std.mem.Allocator) void {
        self.reader.close();
        self.segmenter.deinit();
        if (self.pipe) |*p| p.deinit(gpa);
    }

    fn streamNote(context: *anyopaque) []const u8 {
        const self: *Job = @ptrCast(@alignCast(context));
        return if (self.cancel.load(.acquire) and self.note.len == 0) "cancelled: the client closed the connection" else self.note;
    }
};

fn writeUsage(json: *std.json.Stringify, seconds: f64) !void {
    try json.beginObject();
    try json.objectField("type");
    try json.write("duration");
    try json.objectField("seconds");
    try json.write(seconds);
    try json.endObject();
}

pub fn contentType(format: Format) []const u8 {
    return switch (format) {
        .json, .verbose_json => "application/json",
        .text, .srt => "text/plain; charset=utf-8",
        .vtt => "text/vtt; charset=utf-8",
    };
}

fn nanos(from: std.Io.Timestamp, to: std.Io.Timestamp) u64 {
    return @intCast(@max(0, from.durationTo(to).toNanoseconds()));
}

fn fromNow(io: std.Io, ms: i64) std.Io.Clock.Timestamp {
    return std.Io.Clock.Timestamp.fromNow(io, .{ .raw = .fromMilliseconds(ms), .clock = .awake });
}

fn pastDeadline(io: std.Io, deadline: std.Io.Clock.Timestamp) bool {
    return std.Io.Clock.Timestamp.now(io, deadline.clock).compare(.gte, deadline);
}

fn refusal(arena: std.mem.Allocator, status: std.http.Status, code: []const u8, param: ?[]const u8, comptime format: []const u8, args: anytype) http.Response {
    var err: ApiError = .init(status, code, arena.print(format, args) catch code);
    err.param = param;
    return .fromError(arena, err);
}

fn internal(arena: std.mem.Allocator) http.Response {
    return .fromError(arena, .init(.internal_server_error, "internal", "out of memory"));
}

// ----- tests: the form, and everything before the GPU -----

/// A form of `fields` and, when given, a file part named `file`.
fn testForm(arena: std.mem.Allocator, fields: []const [2][]const u8, file_bytes: ?[]const u8) ![]const u8 {
    var out: std.Io.Writer.Allocating = .init(arena);
    for (fields) |f| try out.writer.print("--B\r\nContent-Disposition: form-data; name=\"{s}\"\r\n\r\n{s}\r\n", .{ f[0], f[1] });
    if (file_bytes) |bytes| try out.writer.print("--B\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n{s}\r\n", .{bytes});
    try out.writer.writeAll("--B--\r\n");
    return out.written();
}

const test_type = "multipart/form-data; boundary=B";

/// A 16 kHz mono 16-bit WAV of `n` silent samples.
fn testWav(arena: std.mem.Allocator, n: u32) ![]const u8 {
    var out: std.Io.Writer.Allocating = .init(arena);
    const w = &out.writer;
    try w.writeAll("RIFF");
    try w.writeInt(u32, 36 + 2 * n, .little);
    try w.writeAll("WAVEfmt ");
    try w.writeInt(u32, 16, .little);
    try w.writeInt(u16, 1, .little);
    try w.writeInt(u16, 1, .little);
    try w.writeInt(u32, 16000, .little);
    try w.writeInt(u32, 32000, .little);
    try w.writeInt(u16, 2, .little);
    try w.writeInt(u16, 16, .little);
    try w.writeAll("data");
    try w.writeInt(u32, 2 * n, .little);
    try w.splatByteAll(0, 2 * n);
    return out.written();
}

test "the form: fields, defaults, granularities, and what is refused" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var problem: Problem = .{};
    const plain = try parse(arena, try testForm(arena, &.{.{ "model", "nemotron-asr-0.6b" }}, "RIFF"), test_type, &problem);
    try std.testing.expectEqualStrings("RIFF", plain.file);
    try std.testing.expectEqualStrings("nemotron-asr-0.6b", plain.model.?);
    try std.testing.expect(plain.language == null and plain.format == .json and plain.segments and !plain.words and !plain.stream);
    try std.testing.expectEqual(@as(usize, 3), plain.lookahead);
    const full = try parse(arena, try testForm(arena, &.{
        .{ "language", "de-DE" },
        .{ "response_format", "verbose_json" },
        .{ "timestamp_granularities[]", "word" },
        .{ "stream", "true" },
        .{ "lookahead", "13" },
        .{ "temperature", "0" },
        .{ "prompt", "" },
        .{ "phrases", "[\"Siobhan\", \"Dun Laoghaire\"]" },
    }, "x"), test_type, &problem);
    try std.testing.expectEqual(@as(usize, 2), full.phrases.len);
    try std.testing.expectEqualStrings("Dun Laoghaire", full.phrases[1]);
    try std.testing.expectEqualStrings("de-DE", full.language.?);
    try std.testing.expect(full.format == .verbose_json and full.words and !full.segments and full.stream);
    try std.testing.expectEqual(@as(usize, 13), full.lookahead);
    try std.testing.expect((try parse(arena, try testForm(arena, &.{.{ "language", "auto" }}, "x"), test_type, &problem)).language == null);

    const cases = [_]struct { fields: []const [2][]const u8, file: ?[]const u8 = "x", code: []const u8 = "invalid_request", param: ?[]const u8 }{
        .{ .fields = &.{}, .file = null, .param = "file" },
        .{ .fields = &.{.{ "file", "not a file part" }}, .file = null, .param = "file" },
        .{ .fields = &.{.{ "response_format", "mp3" }}, .code = "unsupported_feature", .param = "response_format" },
        .{ .fields = &.{.{ "timestamp_granularities[]", "token" }}, .param = "timestamp_granularities" },
        .{ .fields = &.{.{ "timestamp_granularities[]", "word" }}, .param = "timestamp_granularities" },
        .{ .fields = &.{.{ "stream", "maybe" }}, .param = "stream" },
        .{ .fields = &.{ .{ "stream", "true" }, .{ "response_format", "srt" } }, .code = "unsupported_feature", .param = "stream" },
        .{ .fields = &.{.{ "lookahead", "4" }}, .param = "lookahead" },
        .{ .fields = &.{.{ "phrases", "Siobhan" }}, .param = "phrases" },
        .{ .fields = &.{.{ "phrases", "[\"\"]" }}, .param = "phrases" },
        .{ .fields = &.{.{ "phrases", "[1]" }}, .param = "phrases" },
        .{ .fields = &.{.{ "prompt", "Names: Goethe." }}, .code = "unsupported_feature", .param = "prompt" },
        .{ .fields = &.{.{ "include[]", "logprobs" }}, .code = "unsupported_feature", .param = "include" },
        .{ .fields = &.{.{ "model", "m" ++ @import("../../text.zig").repeat("mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm", 32) }}, .param = "model" },
    };
    for (cases) |case| {
        errdefer std.debug.print("case: {any}\n", .{case.fields});
        problem = .{};
        try std.testing.expectError(error.Refused, parse(arena, try testForm(arena, case.fields, case.file), test_type, &problem));
        try std.testing.expectEqualStrings(case.code, problem.code);
        try std.testing.expectEqualStrings(case.param.?, problem.param.?);
    }
    try std.testing.expectError(error.Refused, parse(arena, "{\"file\": 1}", "application/json", &problem));
    try std.testing.expectError(error.Refused, parse(arena, "--B\r\n", test_type, &problem));
    try std.testing.expectError(error.Refused, parse(arena, "--B--", null, &problem));
}

const Fixture = struct {
    tmp: std.testing.TmpDir,
    root: [:0]u8,
    executor: gpu.Executor,
    service: service_mod.Service,

    /// A root whose transcription model has weights that are not a checkpoint.
    fn init(self: *Fixture, max_queued: usize) !void {
        const gpa = std.testing.allocator;
        const io = std.testing.io;
        self.tmp = std.testing.tmpDir(.{});
        errdefer self.tmp.cleanup();
        const dir = "models/nvidia/nemotron-3.5-asr-streaming-0.6b";
        try self.tmp.dir.createDirPath(io, dir);
        try self.tmp.dir.writeFile(io, .{ .sub_path = dir ++ "/model.safetensors", .data = "not weights" });
        self.root = try self.tmp.dir.realPathFileAlloc(io, ".", gpa);
        self.executor = .init(max_queued);
        self.service = .init(gpa, &self.executor, false, self.root, .{}, "nemotron-asr-0.6b");
        self.service.files.service = &self.service;
        self.service.files.timeout_ns = 20 * std.time.ns_per_ms;
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

test "files: refusals before the GPU, a full queue, a file that never starts" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const io = std.testing.io;
    const wav = try testWav(arena, 1600);
    {
        var f: Fixture = undefined;
        try f.init(4);
        defer f.deinit();
        const files = &f.service.files;
        try expectError(files.answer(arena, io, try testForm(arena, &.{.{ "model", "qwen3-tts-1.7b-customvoice" }}, wav), test_type, null), .bad_request, "not_a_transcription_model", "model");
        try expectError(files.answer(arena, io, try testForm(arena, &.{.{ "model", "laya" }}, wav), test_type, null), .bad_request, "not_a_transcription_model", "model");
        try expectError(files.answer(arena, io, try testForm(arena, &.{.{ "model", "whisper-1" }}, wav), test_type, null), .not_found, "model_not_found", "model");
        try expectError(files.answer(arena, io, try testForm(arena, &.{}, "not audio at all, not even close"), test_type, null), .bad_request, "invalid_audio", "file");
        try expectError(files.answer(arena, io, try testForm(arena, &.{}, null), test_type, null), .bad_request, "invalid_request", "file");
        // No worker runs: the file waits until its deadline, then leaves.
        try expectError(files.answer(arena, io, try testForm(arena, &.{}, wav), test_type, null), http.overloaded, "timeout", null);
        try std.testing.expect(files.head == null and files.waiting == 0);
    }
    {
        var f: Fixture = undefined;
        try f.init(0);
        defer f.deinit();
        try expectError(f.service.files.answer(arena, io, try testForm(arena, &.{}, wav), test_type, null), http.overloaded, "busy", null);
    }
}

test "files: a model that fails to open fails its request, and the next one too" {
    const io = std.testing.io;
    var f: Fixture = undefined;
    try f.init(4);
    defer f.deinit();
    f.service.files.timeout_ns = 60 * std.time.ns_per_s;
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
    const wav = try testWav(arena, 1600);
    try expectError(f.service.files.answer(arena, io, try testForm(arena, &.{}, wav), test_type, null), .internal_server_error, "model_failed", "model");
    try expectError(f.service.files.answer(arena, io, try testForm(arena, &.{.{ "stream", "true" }}, wav), test_type, null), .internal_server_error, "model_failed", "model");
    try std.testing.expectEqual(@as(u64, 2), f.service.files.requests);
    try std.testing.expect(!f.service.files.scheduled and f.service.files.waiting == 0);
    try std.testing.expectEqual(@as(u64, 0), budget.resident(io));
    try std.testing.expectEqual(@as(u32, 0), f.service.slot.held);
}
