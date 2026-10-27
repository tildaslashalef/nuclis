//! `GET /v1/audio/transcriptions/stream`: live transcription over a
//! WebSocket. The client streams 16 kHz mono s16le PCM in binary messages
//! and sends `finish`, `reset`, and `language` as JSON text; the server
//! answers `ready`, `partial` (when the hypothesis changes), `final` (after
//! `finish`, with word times) and `error`. A connection is one `Session`:
//! a reader task takes frames off the socket, the connection's task writes
//! what is queued for the client, and one short GPU item per session owns
//! its `Transcriber` (created, pushed one encoder chunk per run, and freed
//! on the worker), so chunks run between a chat generation's steps. The
//! model is the service's slot's, shared and held while any session uses
//! it. docs/guide/api.md § Live transcription.
const std = @import("std");
const inference = @import("inference");
const config = @import("../../config.zig");
const catalog = @import("../../catalog.zig");
const audio_catalog = @import("../../audio/catalog.zig");
const http = @import("../http.zig");
const websocket = @import("../websocket.zig");
const errors = @import("../errors.zig");
const gpu = @import("../gpu.zig");
const router_mod = @import("../router.zig");
const Service = @import("service.zig").Service;

const asr = inference.audio.nemotron_asr;
const rnnt = inference.audio.nemotron_rnnt;
const mel = inference.audio.mel;
const lookaheads = inference.audio.nemotron_encoder.lookaheads;
const Code = websocket.Code;

pub const path = router_mod.prefix ++ "/audio/transcriptions/stream";
pub const sample_rate = 16000;
/// Host limits (docs/guide/api.md § Live transcription).
pub const max_message = 1 << 20;
pub const max_sessions = 16;
pub const max_backlog_samples = 10 * sample_rate;
pub const idle_ms = 60_000;
/// Bytes of events waiting for a client that does not read.
const max_outgoing = 1 << 20;
/// How long a client's close waits for the work it asked for before it.
const drain_ms = 5_000;
const tick_ms = 250;
const retry_ms = 20;
/// The longest locale a client may name (`es-419`, `zh-CN`, `auto`).
const max_locale = 32;
/// The longest text message: a command, never a transcript.
const max_command = 4096;

/// What the service keeps for this route.
pub const Live = struct {
    /// Sessions running now, at most `max_sessions`.
    sessions: std.atomic.Value(u32) = .init(0),
};

pub fn register(service: *Service, gpa: std.mem.Allocator, router: *router_mod.Router) !void {
    try router.add(gpa, .GET, path, .{ .context = service, .handle = handle });
}

/// The samples after which chunk `chunk` of an utterance is complete: its
/// last row reads mel frame `8·row` (nemotron-asr.md § Cache-aware streaming).
pub fn chunkEnd(lookahead: usize, chunk: u64) u64 {
    const last_row = (chunk + 1) * (lookahead + 1) - 1;
    return mel.Spec.nemotron.frameEnd(@intCast(8 * last_row));
}

// ----- the handshake -----

pub const Query = struct {
    model: ?[]const u8 = null,
    /// A `prompt_dictionary` key; null is the model's default (`auto`).
    language: ?[]const u8 = null,
    lookahead: usize = 3,
};

/// The query's fields, or the reason it is refused (`param` names the field).
fn parseQuery(request: http.Request, problem: *Problem) error{Refused}!Query {
    var q: Query = .{};
    if (request.param("model")) |m| {
        if (m.len == 0 or m.len > 256 or !plain(m)) return problem.refuse("model", "\"model\" is a registry or catalogue name (letters, digits, . _ -)");
        q.model = m;
    }
    if (request.param("language")) |l| {
        if (l.len == 0 or l.len > max_locale or !plain(l)) return problem.refuse("language", "\"language\" is a locale (en-US, de-DE, fr) or auto");
        if (!std.mem.eql(u8, l, "auto")) q.language = l;
    }
    if (request.param("lookahead")) |r| {
        const n = std.fmt.parseInt(usize, r, 10) catch std.math.maxInt(usize);
        if (std.mem.indexOfScalar(usize, &lookaheads, n) == null) return problem.refuse("lookahead", "\"lookahead\" takes 0, 3, 6, or 13 (rows of 80 ms)");
        q.lookahead = n;
    }
    return q;
}

fn plain(s: []const u8) bool {
    for (s) |c| if (!std.ascii.isAlphanumeric(c) and c != '-' and c != '_' and c != '.') return false;
    return true;
}

const Problem = struct {
    param: ?[]const u8 = null,
    message: []const u8 = "",

    fn refuse(self: *Problem, param: ?[]const u8, message: []const u8) error{Refused} {
        self.param = param;
        self.message = message;
        return error.Refused;
    }
};

fn handle(context: *anyopaque, arena: std.mem.Allocator, io: std.Io, request: http.Request) http.Response {
    const service: *Service = @ptrCast(@alignCast(context));
    const h = request.websocket orelse {
        var r = refuse(arena, .upgrade_required, "upgrade_required", null, "{s} is a WebSocket (RFC 6455): a GET with upgrade: websocket", .{path});
        r.extra_header = .{ .name = "upgrade", .value = "websocket" };
        return r;
    };
    if (websocket.refusal(h, request.method)) |refused| {
        var r: http.Response = .fromError(arena, refused);
        if (refused.status == .upgrade_required) r.extra_header = .{ .name = "sec-websocket-version", .value = "13" };
        return r;
    }
    var problem: Problem = .{};
    const query = parseQuery(request, &problem) catch return refuse(arena, .bad_request, "invalid_request", problem.param, "{s}", .{problem.message});
    const open = service.slot.openName(io, arena) catch null;
    const name = query.model orelse open orelse service.default_model;
    var diag: config.Diagnostic = .{};
    const located = audio_catalog.locate(arena, io, service.root, service.registry, name, .transcription, false, &diag) catch |err| return switch (err) {
        error.WrongKind => refuse(arena, .bad_request, "not_a_transcription_model", "model", "{s} is a {s} model; this route transcribes with a transcription model ({s})", .{ name, catalog.kindName(catalog.kindOf(service.registry, name)), service.default_model }),
        error.ModelNotFound, error.ModelFileNotFound => refuse(arena, .not_found, "model_not_found", "model", "{s}", .{diag.message()}),
        error.OutOfMemory => .fromError(arena, .init(.internal_server_error, "internal", "out of memory")),
        else => refuse(arena, .internal_server_error, "model_failed", "model", "{s}: {s}", .{ name, if (diag.len > 0) diag.message() else @errorName(err) }),
    };
    if (service.live.sessions.load(.monotonic) >= max_sessions)
        return refuse(arena, http.overloaded, "busy", null, "overloaded: {d} live transcriptions run already; retry shortly", .{max_sessions});
    const session = arena.create(Session) catch return .fromError(arena, .init(.internal_server_error, "internal", "out of memory"));
    session.* = .{
        .service = service,
        .gpa = service.slot.gpa,
        .directory = located.directory,
        .name = name,
        .bytes = located.bytes,
        .lookahead = query.lookahead,
        .language = query.language,
        .note_buffer = arena.alloc(u8, 256) catch return .fromError(arena, .init(.internal_server_error, "internal", "out of memory")),
    };
    return .{ .status = .switching_protocols, .body = "", .upgrade = .{ .context = session, .run = Session.run, .note = Session.writtenNote } };
}

fn refuse(arena: std.mem.Allocator, status: std.http.Status, code: []const u8, param: ?[]const u8, comptime format: []const u8, args: anytype) http.Response {
    var err: errors.ApiError = .init(status, code, arena.print(format, args) catch code);
    err.param = param;
    return .fromError(arena, err);
}

// ----- the client's messages -----

pub const Command = struct {
    kind: Kind,
    /// Samples received when it arrived: it applies after them.
    at: u64 = 0,
    locale: [max_locale]u8 = undefined,
    locale_len: usize = 0,

    pub const Kind = enum { finish, reset, language };

    fn localeName(self: *const Command) []const u8 {
        return self.locale[0..self.locale_len];
    }
};

/// A text message as a command, or the error event's message.
fn parseCommand(gpa: std.mem.Allocator, text: []const u8, problem: *Problem) error{ Refused, OutOfMemory }!Command {
    if (text.len > max_command) return problem.refuse(null, "a text message is a command of at most 4 KiB; audio goes in binary messages");
    const parsed = std.json.parseFromSlice(std.json.Value, gpa, text, .{}) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return problem.refuse(null, "a text message is a JSON object with a \"type\""),
    };
    defer parsed.deinit();
    if (parsed.value != .object) return problem.refuse(null, "a text message is a JSON object with a \"type\"");
    const o = parsed.value.object;
    const kind_text = switch (o.get("type") orelse .null) {
        .string => |s| s,
        else => return problem.refuse("type", "\"type\" is finish, reset, or language"),
    };
    const kind = std.meta.stringToEnum(Command.Kind, kind_text) orelse return problem.refuse("type", "\"type\" is finish, reset, or language");
    var command: Command = .{ .kind = kind };
    if (kind == .language) {
        const locale = switch (o.get("language") orelse .null) {
            .string => |s| s,
            else => return problem.refuse("language", "\"language\" is a locale (en-US, de-DE, fr) or auto"),
        };
        if (locale.len == 0 or locale.len > max_locale or !plain(locale)) return problem.refuse("language", "\"language\" is a locale (en-US, de-DE, fr) or auto");
        @memcpy(command.locale[0..locale.len], locale);
        command.locale_len = locale.len;
    }
    return command;
}

/// Appends little-endian 16-bit samples as floats; a byte left over joins
/// the next message's first.
pub fn appendPcm(gpa: std.mem.Allocator, out: *std.ArrayList(f32), odd: *?u8, bytes: []const u8) !void {
    var rest = bytes;
    if (rest.len == 0) return;
    try out.ensureUnusedCapacity(gpa, (rest.len + 1) / 2);
    if (odd.*) |low| {
        out.appendAssumeCapacity(sample(.{ low, rest[0] }));
        odd.* = null;
        rest = rest[1..];
    }
    var i: usize = 0;
    while (i + 2 <= rest.len) : (i += 2) out.appendAssumeCapacity(sample(rest[i..][0..2].*));
    if (i < rest.len) odd.* = rest[i];
}

fn sample(b: [2]u8) f32 {
    // As `audio.decode` reads a 16-bit WAV, so a stream and a file agree.
    return @as(f32, @floatFromInt(std.mem.readInt(i16, &b, .little))) / 32768.0;
}

// ----- the server's events -----

pub const Word = struct { word: []const u8, start: f64, end: f64 };

/// A token's text (spaces decoded, specials left out) and its row.
pub const Piece = struct { text: []const u8, row: u32 };

/// Words from pieces: a piece that starts with a space starts a word; its
/// times are its first row's start and its last row's end.
pub fn groupWords(arena: std.mem.Allocator, pieces: []const Piece) ![]Word {
    var words: std.ArrayList(Word) = .empty;
    var word: std.ArrayList(u8) = .empty;
    var first: u32 = 0;
    var last: u32 = 0;
    for (pieces) |p| {
        if ((p.text.len > 0 and p.text[0] == ' ') or word.items.len == 0) {
            if (word.items.len > 0) try words.append(arena, wordOf(try word.toOwnedSlice(arena), first, last));
            word = .empty;
            first = p.row;
        }
        try word.appendSlice(arena, std.mem.trimStart(u8, p.text, " "));
        last = p.row;
    }
    if (word.items.len > 0) try words.append(arena, wordOf(try word.toOwnedSlice(arena), first, last));
    return words.items;
}

fn wordOf(text: []const u8, first: u32, last: u32) Word {
    return .{ .word = text, .start = @as(f64, @floatFromInt(first)) * rnnt.seconds_per_row, .end = @as(f64, @floatFromInt(last + 1)) * rnnt.seconds_per_row };
}

fn writeSeconds(s: *std.json.Stringify, seconds: f64) !void {
    try s.print("{d:.3}", .{seconds});
}

pub fn writeReady(out: *std.Io.Writer, model: []const u8, lookahead: usize, language: []const u8) !void {
    var s: std.json.Stringify = .{ .writer = out };
    try s.beginObject();
    try s.objectField("type");
    try s.write("ready");
    try s.objectField("model");
    try s.write(model);
    try s.objectField("lookahead");
    try s.write(lookahead);
    try s.objectField("language");
    try s.write(language);
    try s.objectField("sample_rate");
    try s.write(sample_rate);
    try s.endObject();
}

pub fn writePartial(out: *std.Io.Writer, text: []const u8, audio_seconds: f64) !void {
    var s: std.json.Stringify = .{ .writer = out };
    try s.beginObject();
    try s.objectField("type");
    try s.write("partial");
    try s.objectField("text");
    try s.write(text);
    try s.objectField("audio_seconds");
    try writeSeconds(&s, audio_seconds);
    try s.endObject();
}

pub fn writeFinal(out: *std.Io.Writer, text: []const u8, language: ?[]const u8, words: []const Word, audio_seconds: f64) !void {
    var s: std.json.Stringify = .{ .writer = out };
    try s.beginObject();
    try s.objectField("type");
    try s.write("final");
    try s.objectField("text");
    try s.write(text);
    try s.objectField("language");
    try s.write(language);
    try s.objectField("words");
    try s.beginArray();
    for (words) |w| {
        try s.beginObject();
        try s.objectField("word");
        try s.write(w.word);
        try s.objectField("start");
        try writeSeconds(&s, w.start);
        try s.objectField("end");
        try writeSeconds(&s, w.end);
        try s.endObject();
    }
    try s.endArray();
    try s.objectField("audio_seconds");
    try writeSeconds(&s, audio_seconds);
    try s.endObject();
}

pub fn writeError(out: *std.Io.Writer, code: []const u8, message: []const u8) !void {
    var s: std.json.Stringify = .{ .writer = out };
    try s.beginObject();
    try s.objectField("type");
    try s.write("error");
    try s.objectField("code");
    try s.write(code);
    try s.objectField("message");
    try s.write(message);
    try s.endObject();
}

// ----- one connection -----

const End = union(enum) {
    /// The client sent a close frame, with this code or none.
    client: ?u16,
    /// The connection is gone: nothing more can be sent.
    gone,
    /// The server closes with this code and reason.
    server: struct { code: u16, reason: []const u8 },
};

/// One connection. Lives in the connection's arena; everything it
/// allocates is from `gpa`, since the reader task and the worker allocate
/// too.
const Session = struct {
    service: *Service,
    gpa: std.mem.Allocator,
    directory: []const u8,
    name: []const u8,
    bytes: u64,
    lookahead: usize,
    language: ?[]const u8,
    note_buffer: []u8,
    note: []const u8 = "",

    /// Guards everything up to the worker's fields.
    mutex: std.Io.Mutex = .init,
    changed: std.Io.Condition = .init,
    /// The connection's task has something to do; cleared when it looks.
    wake: bool = false,
    /// Samples received and not pushed: `pending[0]` is sample `consumed`.
    pending: std.ArrayList(f32) = .empty,
    odd: ?u8 = null,
    received: u64 = 0,
    consumed: u64 = 0,
    /// The utterance in progress began at this sample; its chunks pushed.
    utterance_start: u64 = 0,
    chunks: u64 = 0,
    commands: std.ArrayList(Command) = .empty,
    /// Server frames for the connection's task to send.
    outgoing: std.ArrayList(u8) = .empty,
    /// The item is queued or running.
    scheduled: bool = false,
    /// A submit found the queue full: the connection's task tries again.
    retry: bool = false,
    /// The session is ending: the worker frees the transcriber.
    closing: bool = false,
    has_transcriber: bool = false,
    /// The model failed: no audio is taken any more.
    broken: bool = false,
    ended: ?End = null,
    reader_done: bool = false,
    last_heard: std.Io.Timestamp = undefined,
    finals: u32 = 0,

    // Worker-only.
    transcriber: ?asr.Transcriber = null,
    scratch: std.ArrayList(f32) = .empty,
    /// The text of the last partial sent.
    shown: std.ArrayList(u8) = .empty,
    item: gpu.Item = .{ .run = step, .short = true },

    /// The connection's task, after the `101`.
    fn run(context: *anyopaque, io: std.Io, in: *std.Io.Reader, out: *std.Io.Writer) void {
        const self: *Session = @ptrCast(@alignCast(context));
        const live = &self.service.live;
        // Counted here, not at the handshake: `run` is skipped when the 101 fails.
        if (live.sessions.fetchAdd(1, .monotonic) >= max_sessions) {
            _ = live.sessions.fetchSub(1, .monotonic);
            self.writeNote(.{ .server = .{ .code = Code.try_again, .reason = "too many live sessions" } });
            sendClose(out, Code.try_again, "too many live sessions");
            return;
        }
        defer _ = live.sessions.fetchSub(1, .monotonic);
        defer self.free();
        self.last_heard = std.Io.Clock.awake.now(io);
        self.mutex.lockUncancelable(io);
        self.kick(io);
        self.mutex.unlock(io);
        var reader = io.concurrent(readLoop, .{ self, io, in }) catch {
            self.release(io);
            self.writeNote(.{ .server = .{ .code = Code.try_again, .reason = "no task for the reader" } });
            sendClose(out, Code.try_again, "the server is overloaded");
            return;
        };
        const end = self.serve(io, out);
        switch (end) {
            .gone => {},
            .client => |code| sendClose(out, code, ""),
            .server => |s| {
                sendClose(out, s.code, s.reason);
                // The client's close answers ours; a second at most.
                const until = fromNow(io, 1000);
                self.mutex.lockUncancelable(io);
                while (!self.reader_done) self.changed.waitTimeout(io, &self.mutex, .{ .deadline = until }) catch break;
                self.mutex.unlock(io);
            },
        }
        _ = reader.cancel(io);
        self.release(io);
        self.writeNote(end);
    }

    /// Sends what the reader and the worker queue until the session ends.
    fn serve(self: *Session, io: std.Io, out: *std.Io.Writer) End {
        var sending: std.ArrayList(u8) = .empty;
        defer sending.deinit(self.gpa);
        var drain_until: ?std.Io.Clock.Timestamp = null;
        while (true) {
            self.mutex.lockUncancelable(io);
            if (self.retry) self.kick(io);
            std.mem.swap(std.ArrayList(u8), &sending, &self.outgoing);
            const ended = self.ended;
            const working = self.scheduled or self.commands.items.len > 0;
            const heard = self.last_heard;
            self.wake = false;
            self.mutex.unlock(io);
            if (sending.items.len > 0) {
                out.writeAll(sending.items) catch return .gone;
                out.flush() catch return .gone;
                sending.clearRetainingCapacity();
            }
            if (ended) |e| switch (e) {
                // What the client asked before its close is answered first.
                .client => {
                    const until = drain_until orelse fromNow(io, drain_ms);
                    drain_until = until;
                    if (!working or past(io, until)) return e;
                },
                else => return e,
            };
            if (heard.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds() >= idle_ms)
                return .{ .server = .{ .code = Code.policy, .reason = "idle: nothing received for 60 s" } };
            self.mutex.lockUncancelable(io);
            defer self.mutex.unlock(io);
            const tick: i64 = if (self.retry) retry_ms else tick_ms;
            while (!self.wake) self.changed.waitTimeout(io, &self.mutex, .{ .duration = .{ .raw = .fromMilliseconds(tick), .clock = .awake } }) catch |err| switch (err) {
                error.Timeout => break,
                error.Canceled => return .{ .server = .{ .code = Code.going_away, .reason = "the server is stopping" } },
            };
        }
    }

    /// The reader task: frames off the socket until a close, an error, or
    /// cancellation.
    fn readLoop(self: *Session, io: std.Io, in: *std.Io.Reader) void {
        var reader: websocket.Reader = .{ .in = in, .max_message = max_message };
        defer reader.deinit(self.gpa);
        defer {
            self.mutex.lockUncancelable(io);
            self.reader_done = true;
            self.signal(io);
            self.mutex.unlock(io);
        }
        while (true) {
            const message = reader.next(self.gpa) catch |err| {
                const code = websocket.closeCode(err) orelse return self.endWith(io, .gone);
                return self.endWith(io, .{ .server = .{ .code = code, .reason = switch (err) {
                    error.TooBig => "a message over 1 MiB",
                    error.InvalidData => "text that is not UTF-8, or a bad close code",
                    error.OutOfMemory => "out of memory",
                    else => "a frame RFC 6455 forbids",
                } } });
            };
            self.mutex.lockUncancelable(io);
            self.last_heard = std.Io.Clock.awake.now(io);
            self.mutex.unlock(io);
            switch (message) {
                .binary => |bytes| if (!self.appendAudio(io, bytes)) return,
                .text => |text| self.command(io, text),
                .ping => |payload| self.queueFrame(io, .pong, payload),
                .pong => {},
                .close => |c| return self.endWith(io, .{ .client = c.code }),
            }
        }
    }

    /// Records why the session ends, unless it already does.
    fn endWith(self: *Session, io: std.Io, e: End) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (self.ended == null) self.ended = e;
        self.signal(io);
    }

    /// Mutex held: wakes the connection's task.
    fn signal(self: *Session, io: std.Io) void {
        self.wake = true;
        self.changed.broadcast(io);
    }

    /// False once the session ends: past the backlog, or out of memory.
    fn appendAudio(self: *Session, io: std.Io, bytes: []const u8) bool {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (self.broken) return true;
        if (self.pending.items.len + (bytes.len + 1) / 2 > max_backlog_samples) {
            self.queueErrorLocked(io, "audio_backlog", "more than 10 s of audio waits for the GPU; the stream is closed");
            if (self.ended == null) self.ended = .{ .server = .{ .code = Code.policy, .reason = "audio backlog over 10 s" } };
            self.signal(io);
            return false;
        }
        const before = self.pending.items.len;
        appendPcm(self.gpa, &self.pending, &self.odd, bytes) catch {
            if (self.ended == null) self.ended = .{ .server = .{ .code = Code.internal, .reason = "out of memory" } };
            self.signal(io);
            return false;
        };
        self.received += self.pending.items.len - before;
        self.kick(io);
        return true;
    }

    fn command(self: *Session, io: std.Io, text: []const u8) void {
        var problem: Problem = .{};
        var c = parseCommand(self.gpa, text, &problem) catch |err| return switch (err) {
            error.Refused => self.queueError(io, "invalid_message", problem.message),
            error.OutOfMemory => self.queueError(io, "internal", "out of memory"),
        };
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        c.at = self.received;
        self.commands.append(self.gpa, c) catch return self.queueErrorLocked(io, "internal", "out of memory");
        self.kick(io);
    }

    /// Mutex held: submits the item when there is work and it is not queued.
    fn kick(self: *Session, io: std.Io) void {
        if (self.scheduled or !self.workReady()) return;
        self.service.executor.submit(io, &self.item) catch |err| switch (err) {
            error.Busy => {
                self.retry = true;
                return;
            },
            error.Stopped => return,
        };
        self.scheduled = true;
        self.retry = false;
    }

    /// Mutex held.
    fn workReady(self: *const Session) bool {
        if (self.closing) return self.has_transcriber;
        if (self.broken) return false;
        if (!self.has_transcriber) return true;
        if (self.commands.items.len > 0) return true;
        return self.nextBoundary() <= self.received;
    }

    fn nextBoundary(self: *const Session) u64 {
        return self.utterance_start + chunkEnd(self.lookahead, self.chunks);
    }

    fn queueFrame(self: *Session, io: std.Io, opcode: websocket.Opcode, payload: []const u8) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.queueFrameLocked(io, opcode, payload);
    }

    /// Mutex held: a frame for the client; a client that reads nothing for
    /// a megabyte of events is closed.
    fn queueFrameLocked(self: *Session, io: std.Io, opcode: websocket.Opcode, payload: []const u8) void {
        defer self.signal(io);
        if (self.outgoing.items.len + payload.len > max_outgoing) {
            if (self.ended == null) self.ended = .{ .server = .{ .code = Code.policy, .reason = "the client reads no events" } };
            return;
        }
        var w: std.Io.Writer.Allocating = .fromArrayList(self.gpa, &self.outgoing);
        defer self.outgoing = w.toArrayList();
        websocket.writeFrame(&w.writer, opcode, payload) catch if (self.ended == null) {
            self.ended = .{ .server = .{ .code = Code.internal, .reason = "out of memory" } };
        };
    }

    fn queueError(self: *Session, io: std.Io, code: []const u8, message: []const u8) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.queueErrorLocked(io, code, message);
    }

    fn queueErrorLocked(self: *Session, io: std.Io, code: []const u8, message: []const u8) void {
        var buffer: [512]u8 = undefined;
        var w: std.Io.Writer = .fixed(&buffer);
        writeError(&w, code, message) catch return;
        self.queueFrameLocked(io, .text, w.buffered());
    }

    /// An event built in `w` (which it frees) for the client.
    fn queueEvent(self: *Session, io: std.Io, w: *std.Io.Writer.Allocating) void {
        defer w.deinit();
        self.queueFrame(io, .text, w.written());
    }

    // ----- the worker's side -----

    const Action = union(enum) { none, chunk: u64, command: Command };

    /// The item: opens the transcriber, then one chunk or one command a run.
    fn step(item: *gpu.Item, io: std.Io) gpu.Item.After {
        const self: *Session = @alignCast(@fieldParentPtr("item", item));
        self.mutex.lockUncancelable(io);
        const closing = self.closing;
        const broken = self.broken;
        self.mutex.unlock(io);
        if (closing) {
            self.teardown(io);
            return .done;
        }
        if (broken) return self.idle(io);
        if (self.transcriber == null) {
            // Opening may close the language model a generation runs on:
            // it waits for the generation, not between its steps.
            if (item.short and !self.service.slot.isOpen(io, self.directory)) {
                item.short = false;
                return .again;
            }
            item.short = true;
            self.open(io) catch |err| self.fail(io, err);
            return self.idle(io);
        }

        var action: Action = .none;
        self.mutex.lockUncancelable(io);
        const head: ?Command = if (self.commands.items.len > 0) self.commands.items[0] else null;
        const limit = if (head) |c| c.at else self.received;
        const boundary = self.nextBoundary();
        if (head != null and head.?.kind == .reset) {
            // Nothing before a reset is worth pushing.
            action = .{ .command = self.commands.orderedRemove(0) };
        } else if (boundary <= limit) {
            action = .{ .chunk = boundary };
        } else if (head != null) {
            action = .{ .command = self.commands.orderedRemove(0) };
        }
        // The samples the action pushes, copied: the reader appends meanwhile.
        const take: u64 = switch (action) {
            .none => 0,
            .chunk => |b| b - self.consumed,
            .command => |c| if (c.kind == .finish) c.at - self.consumed else 0,
        };
        self.scratch.clearRetainingCapacity();
        const copied = if (self.scratch.appendSlice(self.gpa, self.pending.items[0..@intCast(take)])) |_| true else |_| false;
        self.mutex.unlock(io);
        if (!copied) {
            self.fail(io, error.OutOfMemory);
            return self.idle(io);
        }
        switch (action) {
            .none => {},
            .chunk => |b| self.pushChunk(io, b),
            .command => |c| switch (c.kind) {
                .finish => self.finishUtterance(io, c.at),
                .reset => self.resetUtterance(io, c.at),
                .language => self.setLanguage(io, c.localeName()),
            },
        }
        return self.idle(io);
    }

    /// `.again` while there is work, else unscheduled.
    fn idle(self: *Session, io: std.Io) gpu.Item.After {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (self.workReady()) return .again;
        self.scheduled = false;
        return .done;
    }

    fn open(self: *Session, io: std.Io) !void {
        const slot = &self.service.slot;
        // Another transcription model in use cannot be replaced under it.
        if (slot.held > 0 and !slot.isOpen(io, self.directory)) return error.ModelBusy;
        const acquired = try slot.acquire(io, self.directory, self.name, self.bytes);
        const model = acquired.model;
        const prompt = if (self.language) |l| model.promptId(l) orelse return error.UnknownLanguage else model.default_prompt;
        self.transcriber = try asr.Transcriber.init(self.gpa, model, self.lookahead, prompt, if (model.metal != null) .metal else .cpu);
        slot.hold(io, true);
        self.mutex.lockUncancelable(io);
        self.has_transcriber = true;
        self.mutex.unlock(io);
        var w: std.Io.Writer.Allocating = .init(self.gpa);
        writeReady(&w.writer, self.name, self.lookahead, self.language orelse "auto") catch return w.deinit();
        self.queueEvent(io, &w);
    }

    /// A failure the session does not survive: an error event, then close.
    fn fail(self: *Session, io: std.Io, err: anyerror) void {
        var buffer: [256]u8 = undefined;
        const Failure = struct { code: []const u8, close: u16, message: []const u8 };
        const f: Failure = switch (err) {
            error.UnknownLanguage => .{ .code = "invalid_language", .close = Code.policy, .message = std.fmt.bufPrint(&buffer, "{s} is not a locale of {s} (en-US, de-DE, fr, ..., auto)", .{ self.language.?, self.name }) catch "unknown locale" },
            error.ModelTooLarge => .{ .code = "model_too_large", .close = Code.internal, .message = std.fmt.bufPrint(&buffer, "{s} needs more than the server's memory budget leaves beside the pinned models", .{self.name}) catch "model too large" },
            error.Pinned, error.ModelBusy => .{ .code = "busy", .close = Code.try_again, .message = std.fmt.bufPrint(&buffer, "overloaded: the memory or the transcription slot {s} needs is in use; retry shortly", .{self.name}) catch "busy" },
            else => .{ .code = "model_failed", .close = Code.internal, .message = std.fmt.bufPrint(&buffer, "{s} did not open: {s}", .{ self.name, @errorName(err) }) catch "model failed" },
        };
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.broken = true;
        self.queueErrorLocked(io, f.code, f.message);
        if (self.ended == null) self.ended = .{ .server = .{ .code = f.close, .reason = f.code } };
        self.signal(io);
    }

    /// Pushes one chunk (`scratch`), which completes at sample `boundary`.
    fn pushChunk(self: *Session, io: std.Io, boundary: u64) void {
        const t = &self.transcriber.?;
        const emitted = t.push(self.scratch.items) catch |err| return self.pushFailed(io, err);
        self.mutex.lockUncancelable(io);
        self.drop(boundary);
        self.chunks += 1;
        const start = self.utterance_start;
        self.mutex.unlock(io);
        if (emitted == 0) return;
        var text: std.Io.Writer.Allocating = .init(self.gpa);
        defer text.deinit();
        t.writeText(&text.writer) catch return;
        // The model ends a tagged sentence with a space, then the tag.
        const trimmed = std.mem.trim(u8, text.written(), " ");
        if (std.mem.eql(u8, trimmed, self.shown.items)) return;
        self.shown.clearRetainingCapacity();
        self.shown.appendSlice(self.gpa, trimmed) catch return;
        var w: std.Io.Writer.Allocating = .init(self.gpa);
        writePartial(&w.writer, trimmed, @as(f64, @floatFromInt(boundary - start)) / sample_rate) catch return w.deinit();
        self.queueEvent(io, &w);
    }

    /// Mutex held: the samples up to `to` were pushed or dropped.
    fn drop(self: *Session, to: u64) void {
        const n: usize = @intCast(to - self.consumed);
        std.mem.copyForwards(f32, self.pending.items, self.pending.items[n..]);
        self.pending.shrinkRetainingCapacity(self.pending.items.len - n);
        self.consumed = to;
    }

    /// Mutex held: a new utterance begins at `at`.
    fn restart(self: *Session, at: u64) void {
        self.drop(at);
        self.utterance_start = at;
        self.chunks = 0;
    }

    /// A push that failed leaves the stream unusable until reset: the
    /// utterance is dropped and the next audio starts a new one.
    fn pushFailed(self: *Session, io: std.Io, err: anyerror) void {
        var buffer: [128]u8 = undefined;
        self.queueError(io, "model_failed", std.fmt.bufPrint(&buffer, "the transcription failed ({s}); the utterance is dropped", .{@errorName(err)}) catch "the transcription failed");
        self.transcriber.?.reset() catch |e| return self.fail(io, e);
        self.shown.clearRetainingCapacity();
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.restart(self.consumed + self.scratch.items.len);
    }

    /// Ends the utterance at sample `at` (`scratch` holds what is left of
    /// it), sends its final, and starts the next.
    fn finishUtterance(self: *Session, io: std.Io, at: u64) void {
        const t = &self.transcriber.?;
        self.mutex.lockUncancelable(io);
        const start = self.utterance_start;
        self.mutex.unlock(io);
        var arena_state: std.heap.ArenaAllocator = .init(self.gpa);
        defer arena_state.deinit();
        const arena = arena_state.allocator();
        var text: []const u8 = "";
        var language: ?[]const u8 = null;
        var words: []const Word = &.{};
        if (at > start) {
            if (self.scratch.items.len > 0) _ = t.push(self.scratch.items) catch |err| return self.pushFailed(io, err);
            _ = t.finish() catch |err| return self.pushFailed(io, err);
            var written: std.Io.Writer.Allocating = .init(arena);
            t.writeText(&written.writer) catch {};
            text = std.mem.trim(u8, written.written(), " ");
            language = t.language();
            words = wordsOf(arena, t) catch &.{};
        }
        var w: std.Io.Writer.Allocating = .init(self.gpa);
        writeFinal(&w.writer, text, language, words, @as(f64, @floatFromInt(at - start)) / sample_rate) catch {};
        self.queueEvent(io, &w);
        if (at > start) t.reset() catch |err| return self.fail(io, err);
        self.shown.clearRetainingCapacity();
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.restart(at);
        self.finals += 1;
    }

    fn resetUtterance(self: *Session, io: std.Io, at: u64) void {
        self.transcriber.?.reset() catch |err| return self.fail(io, err);
        self.shown.clearRetainingCapacity();
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.restart(at);
    }

    fn setLanguage(self: *Session, io: std.Io, locale: []const u8) void {
        const t = &self.transcriber.?;
        const prompt = if (std.mem.eql(u8, locale, "auto")) t.model.default_prompt else t.model.promptId(locale) orelse {
            var buffer: [128]u8 = undefined;
            return self.queueError(io, "invalid_language", std.fmt.bufPrint(&buffer, "{s} is not a locale of {s}; the language stays", .{ locale, self.name }) catch "unknown locale");
        };
        t.setPrompt(prompt) catch |err| return self.fail(io, err);
    }

    /// Worker (or, once the worker stopped, the connection's task): frees
    /// the transcriber and lets go of the model.
    fn teardown(self: *Session, io: std.Io) void {
        if (self.transcriber) |*t| {
            t.deinit();
            self.transcriber = null;
            self.service.slot.hold(io, false);
        }
        self.scratch.deinit(self.gpa);
        self.shown.deinit(self.gpa);
        self.scratch = .empty;
        self.shown = .empty;
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.has_transcriber = false;
        self.scheduled = false;
    }

    /// The connection's task, last: the worker frees the transcriber, and
    /// the item is no longer the executor's once this returns.
    fn release(self: *Session, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        self.closing = true;
        const wait = self.scheduled or self.has_transcriber;
        while (!self.scheduled and self.has_transcriber) {
            self.service.executor.submit(io, &self.item) catch |err| switch (err) {
                error.Busy => {
                    self.mutex.unlock(io);
                    io.sleep(.fromMilliseconds(retry_ms), .awake) catch {};
                    self.mutex.lockUncancelable(io);
                    continue;
                },
                // The worker is gone: nothing else can touch the transcriber.
                error.Stopped => {
                    self.mutex.unlock(io);
                    self.teardown(io);
                    return;
                },
            };
            self.scheduled = true;
        }
        self.mutex.unlock(io);
        // `done` is set when the item finishes, after its last touch of us.
        if (wait) self.item.done.waitUncancelable(io);
    }

    fn free(self: *Session) void {
        self.pending.deinit(self.gpa);
        self.commands.deinit(self.gpa);
        self.outgoing.deinit(self.gpa);
    }

    fn writeNote(self: *Session, e: End) void {
        const seconds = @as(f64, @floatFromInt(self.received)) / sample_rate;
        var how_buffer: [96]u8 = undefined;
        const how = switch (e) {
            .gone => "the client left",
            .client => |code| std.fmt.bufPrint(&how_buffer, "closed by the client ({?d})", .{code}) catch "closed by the client",
            .server => |s| std.fmt.bufPrint(&how_buffer, "closed {d}: {s}", .{ s.code, s.reason }) catch "closed",
        };
        self.note = std.fmt.bufPrint(self.note_buffer, "{s} · lookahead {d} · {d:.2} s of audio · {d} final{s} · {s}", .{ self.name, self.lookahead, seconds, self.finals, if (self.finals == 1) "" else "s", how }) catch self.name;
    }

    fn writtenNote(context: *anyopaque) []const u8 {
        const self: *Session = @ptrCast(@alignCast(context));
        return self.note;
    }
};

/// The final's words: the transcriber's tokens, specials left out.
fn wordsOf(arena: std.mem.Allocator, t: *const asr.Transcriber) ![]Word {
    const table = &t.model.table;
    var pieces: std.ArrayList(Piece) = .empty;
    for (t.tokens.items) |token| {
        if (table.isSpecial(token.id)) continue;
        var text: std.Io.Writer.Allocating = .init(arena);
        try table.writePiece(table.piece(token.id) orelse continue, &text.writer);
        try pieces.append(arena, .{ .text = text.written(), .row = token.row });
    }
    return groupWords(arena, pieces.items);
}

fn sendClose(out: *std.Io.Writer, code: ?u16, reason: []const u8) void {
    websocket.writeClose(out, code, reason) catch return;
    out.flush() catch {};
}

fn fromNow(io: std.Io, ms: i64) std.Io.Clock.Timestamp {
    return std.Io.Clock.Timestamp.fromNow(io, .{ .raw = .fromMilliseconds(ms), .clock = .awake });
}

fn past(io: std.Io, deadline: std.Io.Clock.Timestamp) bool {
    return std.Io.Clock.Timestamp.now(io, deadline.clock).compare(.gte, deadline);
}

// ----- tests: everything but the model -----

test "chunk boundaries: the processor's sample counts at every lookahead" {
    // nemotron-asr.md § Cache-aware streaming: first chunk, then new audio per chunk.
    const first = [_]u64{ 200, 4040, 7880, 16840 };
    const later = [_]u64{ 1280, 5120, 8960, 17920 };
    for (lookaheads, first, later) |r, f, l| {
        try std.testing.expectEqual(f, chunkEnd(r, 0));
        try std.testing.expectEqual(f + l, chunkEnd(r, 1));
        try std.testing.expectEqual(f + 9 * l, chunkEnd(r, 9));
    }
}

test "the query: lookahead, language, model, each refused by name" {
    var problem: Problem = .{};
    const q = try parseQuery(.{ .method = .GET, .path = path, .query = "model=nemotron-asr-0.6b&language=de-DE&lookahead=13", .body = "" }, &problem);
    try std.testing.expectEqualStrings("nemotron-asr-0.6b", q.model.?);
    try std.testing.expectEqualStrings("de-DE", q.language.?);
    try std.testing.expectEqual(@as(usize, 13), q.lookahead);
    const defaults = try parseQuery(.{ .method = .GET, .path = path, .query = "language=auto", .body = "" }, &problem);
    try std.testing.expect(defaults.language == null and defaults.model == null);
    try std.testing.expectEqual(@as(usize, 3), defaults.lookahead);
    for ([_][2][]const u8{ .{ "lookahead=1", "lookahead" }, .{ "lookahead=x", "lookahead" }, .{ "language=", "language" }, .{ "language=de%2DDE", "language" }, .{ "model=../x", "model" } }) |case| {
        try std.testing.expectError(error.Refused, parseQuery(.{ .method = .GET, .path = path, .query = case[0], .body = "" }, &problem));
        try std.testing.expectEqualStrings(case[1], problem.param.?);
    }
}

test "commands: finish, reset, language; anything else is refused with a reason" {
    const gpa = std.testing.allocator;
    var problem: Problem = .{};
    try std.testing.expectEqual(Command.Kind.finish, (try parseCommand(gpa, "{\"type\":\"finish\"}", &problem)).kind);
    try std.testing.expectEqual(Command.Kind.reset, (try parseCommand(gpa, "{\"type\": \"reset\", \"extra\": 1}", &problem)).kind);
    const language = try parseCommand(gpa, "{\"type\":\"language\",\"language\":\"fr-FR\"}", &problem);
    try std.testing.expectEqualStrings("fr-FR", language.localeName());
    for ([_][]const u8{ "finish", "[]", "{\"type\":\"session.update\"}", "{\"type\":3}", "{\"type\":\"language\"}", "{\"type\":\"language\",\"language\":\"a b\"}", "{}" }) |text| {
        errdefer std.debug.print("text: {s}\n", .{text});
        try std.testing.expectError(error.Refused, parseCommand(gpa, text, &problem));
        try std.testing.expect(problem.message.len > 0);
    }
}

test "PCM: little-endian 16-bit as audio.decode reads it, a sample split across messages" {
    const gpa = std.testing.allocator;
    var out: std.ArrayList(f32) = .empty;
    defer out.deinit(gpa);
    var odd: ?u8 = null;
    try appendPcm(gpa, &out, &odd, "\x00\x40\x00");
    try std.testing.expectEqual(@as(?u8, 0), odd);
    try appendPcm(gpa, &out, &odd, "\x80\xff\x7f");
    try appendPcm(gpa, &out, &odd, "");
    try std.testing.expectEqual(@as(?u8, null), odd);
    try std.testing.expectEqualSlices(f32, &.{ 0.5, -1, 32767.0 / 32768.0 }, out.items);
}

test "words: a leading space starts one, punctuation joins it, times span its rows" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const words = try groupWords(arena_state.allocator(), &.{
        .{ .text = "He", .row = 3 }, .{ .text = "llo", .row = 4 }, .{ .text = ",", .row = 4 },
        .{ .text = " ", .row = 6 },  .{ .text = "wor", .row = 7 }, .{ .text = " ld", .row = 9 },
        .{ .text = ".", .row = 10 },
    });
    try std.testing.expectEqual(@as(usize, 3), words.len);
    try std.testing.expectEqualStrings("Hello,", words[0].word);
    try std.testing.expectApproxEqAbs(@as(f64, 0.24), words[0].start, 1e-9);
    try std.testing.expectApproxEqAbs(@as(f64, 0.40), words[0].end, 1e-9);
    try std.testing.expectEqualStrings("wor", words[1].word);
    // A bare space starts the word; its text, and its time, begin after.
    try std.testing.expectApproxEqAbs(@as(f64, 0.56), words[1].start, 1e-9);
    try std.testing.expectEqualStrings("ld.", words[2].word);
    try std.testing.expectApproxEqAbs(@as(f64, 0.88), words[2].end, 1e-9);
    try std.testing.expectEqual(@as(usize, 0), (try groupWords(arena_state.allocator(), &.{})).len);
}

test "events as the client reads them" {
    var buffer: [512]u8 = undefined;
    var w: std.Io.Writer = .fixed(&buffer);
    try writePartial(&w, "hello \"you\"", 1.28);
    try std.testing.expectEqualStrings("{\"type\":\"partial\",\"text\":\"hello \\\"you\\\"\",\"audio_seconds\":1.280}", w.buffered());
    w.end = 0;
    try writeFinal(&w, "Hi.", "en-US", &.{.{ .word = "Hi.", .start = 0.24, .end = 0.4 }}, 0.75);
    try std.testing.expectEqualStrings("{\"type\":\"final\",\"text\":\"Hi.\",\"language\":\"en-US\",\"words\":[{\"word\":\"Hi.\",\"start\":0.240,\"end\":0.400}],\"audio_seconds\":0.750}", w.buffered());
    w.end = 0;
    try writeFinal(&w, "", null, &.{}, 0);
    try std.testing.expectEqualStrings("{\"type\":\"final\",\"text\":\"\",\"language\":null,\"words\":[],\"audio_seconds\":0.000}", w.buffered());
    w.end = 0;
    try writeError(&w, "audio_backlog", "x");
    try std.testing.expectEqualStrings("{\"type\":\"error\",\"code\":\"audio_backlog\",\"message\":\"x\"}", w.buffered());
    w.end = 0;
    try writeReady(&w, "nemotron-asr-0.6b", 3, "auto");
    try std.testing.expectEqualStrings("{\"type\":\"ready\",\"model\":\"nemotron-asr-0.6b\",\"lookahead\":3,\"language\":\"auto\",\"sample_rate\":16000}", w.buffered());
}

// ----- a session over a socket pair, with a model that fails to open -----

const Fixture = struct {
    tmp: std.testing.TmpDir,
    root: [:0]u8,
    executor: gpu.Executor,
    service: Service,

    /// A root whose transcription model has a file that is not a checkpoint.
    fn init(self: *Fixture) !void {
        const gpa = std.testing.allocator;
        const io = std.testing.io;
        self.tmp = std.testing.tmpDir(.{});
        errdefer self.tmp.cleanup();
        const dir = "models/nvidia/nemotron-3.5-asr-streaming-0.6b";
        try self.tmp.dir.createDirPath(io, dir);
        try self.tmp.dir.writeFile(io, .{ .sub_path = dir ++ "/model.safetensors", .data = "not weights" });
        self.root = try self.tmp.dir.realPathFileAlloc(io, ".", gpa);
        self.executor = .init(4);
        self.service = .init(gpa, &self.executor, false, self.root, .{}, "nemotron-asr-0.6b");
    }

    fn deinit(self: *Fixture) void {
        self.service.deinit(std.testing.io);
        std.testing.allocator.free(self.root);
        self.tmp.cleanup();
    }
};

fn expectRefusal(r: http.Response, status: std.http.Status, code: []const u8) !void {
    errdefer std.debug.print("got {d}: {s}\n", .{ @backingInt(r.status), r.body });
    try std.testing.expect(r.upgrade == null);
    try std.testing.expectEqual(status, r.status);
    try std.testing.expect(std.mem.indexOf(u8, r.body, code) != null);
}

test "live: refusals before the upgrade are HTTP errors" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const io = std.testing.io;
    const ws: websocket.Handshake = .{ .connection_upgrade = true, .key = "dGhlIHNhbXBsZSBub25jZQ==", .version = "13" };
    const plain_get = handle(&f.service, arena, io, .{ .method = .GET, .path = path, .query = "", .body = "" });
    try expectRefusal(plain_get, .upgrade_required, "upgrade_required");
    try std.testing.expectEqualStrings("upgrade", plain_get.extra_header.?.name);
    var old = ws;
    old.version = "8";
    try expectRefusal(handle(&f.service, arena, io, .{ .method = .GET, .path = path, .query = "", .body = "", .websocket = old }), .upgrade_required, "unsupported_version");
    try expectRefusal(handle(&f.service, arena, io, .{ .method = .GET, .path = path, .query = "lookahead=2", .body = "", .websocket = ws }), .bad_request, "invalid_request");
    try expectRefusal(handle(&f.service, arena, io, .{ .method = .GET, .path = path, .query = "model=laya", .body = "", .websocket = ws }), .bad_request, "not_a_transcription_model");
    try expectRefusal(handle(&f.service, arena, io, .{ .method = .GET, .path = path, .query = "model=whisper", .body = "", .websocket = ws }), .not_found, "model_not_found");
    const accepted = handle(&f.service, arena, io, .{ .method = .GET, .path = path, .query = "", .body = "", .websocket = ws });
    try std.testing.expect(accepted.upgrade != null);
}

/// Reads one server frame from `fd`: its opcode and payload in `buffer`.
fn readFrame(fd: std.c.fd_t, buffer: []u8) !struct { opcode: u8, payload: []u8 } {
    var head: [4]u8 = undefined;
    try readExact(fd, head[0..2]);
    var len: usize = head[1] & 0x7f;
    if (len == 126) {
        try readExact(fd, head[2..4]);
        len = std.mem.readInt(u16, head[2..4], .big);
    }
    try readExact(fd, buffer[0..len]);
    return .{ .opcode = head[0] & 0x0f, .payload = buffer[0..len] };
}

fn readExact(fd: std.c.fd_t, out: []u8) !void {
    var at: usize = 0;
    while (at < out.len) {
        const n = std.c.read(fd, out[at..].ptr, out.len - at);
        if (n <= 0) return error.EndOfStream;
        at += @intCast(n);
    }
}

fn sendMasked(fd: std.c.fd_t, opcode: u8, payload: []const u8) void {
    var frame: [256]u8 = undefined;
    frame[0] = 0x80 | opcode;
    frame[1] = 0x80 | @as(u8, @intCast(payload.len));
    const key = [4]u8{ 9, 8, 7, 6 };
    @memcpy(frame[2..6], &key);
    for (payload, 0..) |b, i| frame[6 + i] = b ^ key[i % 4];
    _ = std.c.write(fd, &frame, 6 + payload.len);
}

test "live: a session answers pings and bad messages, reports a model that fails to open, and closes" {
    var f: Fixture = undefined;
    try f.init();
    defer f.deinit();
    const io = std.testing.io;
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const ws: websocket.Handshake = .{ .connection_upgrade = true, .key = "dGhlIHNhbXBsZSBub25jZQ==", .version = "13" };
    const response = handle(&f.service, arena, io, .{ .method = .GET, .path = path, .query = "", .body = "", .websocket = ws });
    const upgrade = response.upgrade.?;

    var fds: [2]std.c.fd_t = undefined;
    try std.testing.expectEqual(@as(c_int, 0), std.c.socketpair(std.c.AF.UNIX, std.c.SOCK.STREAM, 0, &fds));
    defer _ = std.c.close(fds[1]);
    const stream: std.Io.net.Stream = .{ .socket = .{ .handle = fds[0], .address = undefined } };
    var read_buffer: [1024]u8 = undefined;
    var write_buffer: [1024]u8 = undefined;
    var reader = stream.reader(io, &read_buffer);
    var writer = stream.writer(io, &write_buffer);
    const Server = struct {
        fn serve(u: http.Upgrade, server_io: std.Io, in: *std.Io.Reader, out: *std.Io.Writer, fd: std.c.fd_t) void {
            u.run(u.context, server_io, in, out);
            _ = std.c.close(fd);
        }
    };
    // No worker yet: the item that opens the model waits in the queue.
    var session = try io.concurrent(Server.serve, .{ upgrade, io, &reader.interface, &writer.interface, fds[0] });
    var buffer: [1024]u8 = undefined;
    sendMasked(fds[1], 0x9, "hi");
    const pong = try readFrame(fds[1], &buffer);
    try std.testing.expectEqual(@as(u8, 0xa), pong.opcode);
    try std.testing.expectEqualStrings("hi", pong.payload);
    sendMasked(fds[1], 0x1, "{\"type\":\"pause\"}");
    const bad = try readFrame(fds[1], &buffer);
    try std.testing.expect(std.mem.indexOf(u8, bad.payload, "\"code\":\"invalid_message\"") != null);
    sendMasked(fds[1], 0x2, "\x00\x01\x02");

    var worker = try io.concurrent(gpu.Executor.run, .{ &f.executor, io });
    defer {
        f.executor.stop(io);
        worker.await(io);
    }
    const failed = try readFrame(fds[1], &buffer);
    try std.testing.expect(std.mem.indexOf(u8, failed.payload, "\"code\":\"model_failed\"") != null);
    const close = try readFrame(fds[1], &buffer);
    try std.testing.expectEqual(@as(u8, 0x8), close.opcode);
    try std.testing.expectEqual(@as(u16, 1011), std.mem.readInt(u16, close.payload[0..2], .big));
    sendMasked(fds[1], 0x8, "\x03\xf3");
    session.await(io);
    try std.testing.expectEqual(@as(u32, 0), f.service.live.sessions.load(.monotonic));
    try std.testing.expect(std.mem.indexOf(u8, Session.writtenNote(upgrade.context), "closed 1011: model_failed") != null);
}
