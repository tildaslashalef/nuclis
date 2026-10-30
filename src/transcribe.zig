//! `nuclis transcribe`: speech to text with Nemotron 3.5 ASR
//! (`inference.audio.nemotron_asr`). The file is decoded in pieces and fed
//! to one stream as the server's file route feeds it (`audio/transcript.zig`,
//! so both give the same text), or in 80 ms pieces (`--stream`, printing each
//! partial transcript as it changes). `--format` writes the transcript, the
//! JSON report (tokens with their times, the timings), or subtitles from the
//! segments; `--segments` adds the segments to the first two. On Metal by
//! default when built with it, else the CPU reference.
//! docs/models/nemotron-asr.md.
const std = @import("std");
const inference = @import("inference");
const style = @import("tui/style.zig");
const config = @import("config.zig");
const transcript = @import("audio/transcript.zig");

const audio = inference.audio.audio;
const asr = inference.audio.nemotron_asr;
const rnnt = inference.audio.nemotron_rnnt;
const lookaheads = inference.audio.nemotron_encoder.lookaheads;
const boost = inference.audio.boost;

/// One `--stream` piece: 80 ms, one encoder row.
const piece_samples = 1280;

pub const Backend = enum { cpu, metal };
/// Metal where the build has it.
pub const default_backend: Backend = if (inference.metal.enabled) .metal else .cpu;

pub const Format = enum { text, json, srt, vtt };

pub const Options = struct {
    file: ?[]const u8 = null,
    stream: bool = false,
    format: Format = .text,
    /// The segments, with their words in JSON.
    segments: bool = false,
    lookahead: usize = 3,
    /// A `prompt_dictionary` key; null is the model's default (`auto`).
    language: ?[]const u8 = null,
    backend: Backend = default_backend,
    model: ?[]const u8 = null,
    /// `--phrase`, in order: words to lean toward.
    phrase_buf: [boost.max_phrases][]const u8 = undefined,
    phrase_count: usize = 0,

    pub fn phrases(self: *const Options) []const []const u8 {
        return self.phrase_buf[0..self.phrase_count];
    }
};

/// Parses the words after `transcribe`.
pub fn parseArgs(args: []const []const u8, diag: *config.Diagnostic) !Options {
    var o: Options = .{};
    var lookahead_set = false;
    var backend_set = false;
    var format_set = false;
    var i: usize = 0;
    while (i < args.len) : (i += 1) {
        const word = args[i];
        if (!std.mem.startsWith(u8, word, "--")) {
            if (o.file != null) {
                diag.set("transcribe takes one file; {s} is a second", .{word});
                return error.UnknownOption;
            }
            o.file = word;
            continue;
        }
        if (std.mem.eql(u8, word, "--stream") or std.mem.eql(u8, word, "--segments")) {
            const flag = if (std.mem.eql(u8, word, "--stream")) &o.stream else &o.segments;
            if (flag.*) return error.DuplicateOption;
            flag.* = true;
            continue;
        }
        if (std.mem.eql(u8, word, "--json")) {
            if (format_set) return error.DuplicateOption;
            format_set = true;
            o.format = .json;
            continue;
        }
        const takes_value = for ([_][]const u8{ "--lookahead", "--language", "--backend", "--model", "--format", "--phrase" }) |known| {
            if (std.mem.eql(u8, word, known)) break true;
        } else false;
        if (!takes_value) {
            diag.set("{s} is not an option of transcribe (`nuclis transcribe --help`)", .{word});
            return error.UnknownOption;
        }
        i += 1;
        if (i == args.len) {
            diag.set("{s} needs a value", .{word});
            return error.MissingOptionValue;
        }
        const value = args[i];
        if (std.mem.eql(u8, word, "--lookahead")) {
            if (lookahead_set) return error.DuplicateOption;
            lookahead_set = true;
            const n = std.fmt.parseInt(usize, value, 10) catch std.math.maxInt(usize);
            if (std.mem.indexOfScalar(usize, &lookaheads, n) == null) {
                diag.set("--lookahead takes 0, 3, 6, or 13 (rows of 80 ms), not {s}", .{value});
                return error.InvalidOptionValue;
            }
            o.lookahead = n;
        } else if (std.mem.eql(u8, word, "--language")) {
            if (o.language != null) return error.DuplicateOption;
            o.language = value;
        } else if (std.mem.eql(u8, word, "--backend")) {
            if (backend_set) return error.DuplicateOption;
            backend_set = true;
            o.backend = std.meta.stringToEnum(Backend, value) orelse {
                diag.set("--backend takes cpu or metal, not {s}", .{value});
                return error.InvalidOptionValue;
            };
        } else if (std.mem.eql(u8, word, "--model")) {
            if (o.model != null) return error.DuplicateOption;
            o.model = value;
        } else if (std.mem.eql(u8, word, "--phrase")) {
            if (o.phrase_count == boost.max_phrases) {
                diag.set("--phrase is given at most {d} times", .{boost.max_phrases});
                return error.InvalidOptionValue;
            }
            o.phrase_buf[o.phrase_count] = value;
            o.phrase_count += 1;
        } else if (std.mem.eql(u8, word, "--format")) {
            if (format_set) return error.DuplicateOption;
            format_set = true;
            o.format = std.meta.stringToEnum(Format, value) orelse {
                diag.set("--format takes text, json, srt, or vtt, not {s}", .{value});
                return error.InvalidOptionValue;
            };
        }
    }
    if (o.file == null) {
        diag.set("give the audio file to transcribe", .{});
        return error.MissingInput;
    }
    return o;
}

/// What `--format json` writes; the text report shows the same values.
const Report = struct {
    /// The segments' texts joined by spaces (special tokens skipped).
    text: []const u8,
    /// The locale of the language tag the model emitted, if any.
    language: ?[]const u8,
    /// The prompt it ran under (`auto` unless `--language` named one).
    prompt: []const u8,
    lookahead: usize,
    stream: bool,
    backend: []const u8,
    model: []const u8,
    audio_seconds: f64,
    /// The greedy path's log probability, blanks included.
    score: f64,
    tokens: []const TokenReport,
    /// With `--segments`, else null.
    segments: ?Segments = null,
    timings: Timings,
};

/// The segments with their words, as JSON.
const Segments = struct {
    segmenter: *const transcript.Segmenter,

    pub fn jsonStringify(self: Segments, w: *std.json.Stringify) std.Io.Writer.Error!void {
        try w.beginArray();
        for (self.segmenter.segments.items) |s| {
            try w.beginObject();
            try transcript.writeSegmentFields(w, self.segmenter, s, true);
            try w.endObject();
        }
        try w.endArray();
    }
};

const TokenReport = struct {
    id: u32,
    /// The piece with U+2581 as a space; a language tag as written.
    piece: []const u8,
    special: bool,
    start: f64,
    end: f64,
};

const Timings = struct {
    load_ms: f64,
    /// Reading the file and its header: the samples are decoded piece by
    /// piece within `transcribe_ms`.
    decode_audio_ms: f64,
    transcribe_ms: f64,
    /// Transcription time over the audio's duration.
    real_time_factor: f64,
};

/// Transcribes `o.file` with the checkpoint directory `directory`.
pub fn run(gpa: std.mem.Allocator, io: std.Io, directory: []const u8, o: Options, out: *std.Io.Writer, sty: style.Style, diag: *config.Diagnostic) !void {
    if (o.backend == .metal and !inference.metal.enabled) {
        diag.set("this nuclis was built without Metal; run with --backend cpu", .{});
        return error.MetalNotEnabled;
    }
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const path = o.file.?;

    var started = std.Io.Clock.awake.now(io);
    const model = asr.Model.open(gpa, io, directory) catch |err| {
        switch (err) {
            error.FileNotFound => diag.set("{s}: no Nemotron ASR checkpoint there (`nuclis model pull nemotron-asr-0.6b`, or --model <name or directory>)", .{directory}),
            error.NotAnAsrModel => diag.set("{s}: not a Nemotron 3.5 ASR checkpoint (config.json names another architecture)", .{directory}),
            else => diag.set("{s}: not a checkpoint nuclis can transcribe with ({s})", .{ directory, @errorName(err) }),
        }
        return err;
    };
    defer model.close();
    if (o.backend == .metal) model.openMetal() catch |err| {
        diag.set("{s}: the Metal plan failed ({s}); run with --backend cpu", .{ directory, @errorName(err) });
        return err;
    };
    const load_ns = lap(io, &started);

    const prompt_name = o.language orelse "auto";
    var prompts: [asr.max_languages]usize = undefined;
    const languages = model.languages(prompt_name, &prompts) catch |err| {
        switch (err) {
            error.UnknownLanguage => diag.set("--language takes locales the model knows (en-US, de-DE, fr, auto, ...), not {s}", .{prompt_name}),
            error.TooManyLanguages => diag.set("--language takes at most {d} locales", .{asr.max_languages}),
            error.DuplicateLanguage => diag.set("--language names a language twice: {s}", .{prompt_name}),
            error.FallbackNotLast => diag.set("--language takes auto only last, as the fallback: {s}", .{prompt_name}),
        }
        return error.InvalidOptionValue;
    };

    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(transcript.max_file_bytes)) catch |err| {
        switch (err) {
            error.StreamTooLong => diag.set("{s}: larger than {d} MB", .{ path, transcript.max_file_bytes >> 20 }),
            else => diag.set("{s}: {s}", .{ path, @errorName(err) }),
        }
        return err;
    };
    var reader = audio.Reader.open(bytes) catch |err| {
        if (err == error.UnsupportedAudioFormat) diag.set("{s}: not audio nuclis can read ({s}; or a 16 kHz mono WAV)", .{ path, audio.readable });
        return err;
    };
    defer reader.close();
    const decode_ns = lap(io, &started);

    var transcriber = try asr.Transcriber.init(gpa, model, o.lookahead, prompts[0], switch (o.backend) {
        .cpu => .cpu,
        .metal => .metal,
    });
    defer transcriber.deinit();
    if (languages > 1) try transcriber.setLanguages(prompts[0..languages]);
    var phrases: ?inference.audio.boost.Boost = if (o.phrase_count == 0) null else model.boost(gpa, o.phrases(), boost.default_bonus, boost.default_follow) catch |err| switch (err) {
        error.InvalidPhrase => {
            diag.set("--phrase takes words: at most {d} bytes, no control characters", .{boost.max_phrase_bytes});
            return error.InvalidOptionValue;
        },
        else => return err,
    };
    defer if (phrases) |*p| p.deinit();
    if (phrases) |*p| transcriber.setBoost(p);
    var segmenter: transcript.Segmenter = .init(gpa, if (languages == 1 and prompts[0] != model.default_prompt) prompt_name else null);
    defer segmenter.deinit();
    var samples: u64 = 0;
    if (o.stream) {
        var shown: std.Io.Writer.Allocating = .init(arena);
        var piece: [piece_samples]f32 = undefined;
        while (true) {
            const n = reader.read(&piece) catch |err| return midway(diag, path, samples, err);
            if (n == 0) break;
            samples += n;
            const emitted = try transcriber.push(piece[0..n]);
            if (emitted > 0 and o.format == .text) try showPartial(arena, &transcriber, &shown, out, sty);
        }
        _ = try transcriber.finish();
        _ = try segmenter.update(model, transcriber.tokens.items, transcriber.rowsDecoded(), true);
        if (o.format == .text and sty.enabled) try out.writeAll("\r\x1b[2K");
    } else {
        var feeding: transcript.Run = try .init(gpa, &reader, &transcriber, &segmenter);
        defer feeding.deinit(gpa);
        while (!feeding.finished) _ = feeding.step() catch |err| return midway(diag, path, feeding.fed, err);
        samples = feeding.fed;
    }
    const transcribe_ns = lap(io, &started);

    var text: std.Io.Writer.Allocating = .init(arena);
    try segmenter.writeText(&text.writer);
    const seconds = @as(f64, @floatFromInt(samples)) / audio.sample_rate;
    switch (o.format) {
        .srt => return transcript.writeSrt(out, segmenter.segments.items),
        .vtt => return transcript.writeVtt(out, segmenter.segments.items),
        .text, .json => {},
    }
    const tokens = try arena.alloc(TokenReport, transcriber.tokens.items.len);
    for (transcriber.tokens.items, tokens) |t, *r| {
        var piece: std.Io.Writer.Allocating = .init(arena);
        try model.table.writePiece(model.table.piece(t.id) orelse "", &piece.writer);
        const start = @as(f64, @floatFromInt(t.row)) * rnnt.seconds_per_row;
        r.* = .{ .id = t.id, .piece = piece.written(), .special = model.table.isSpecial(t.id), .start = start, .end = start + rnnt.seconds_per_row };
    }
    const report: Report = .{
        .text = text.written(),
        .language = transcriber.language(),
        .prompt = prompt_name,
        .lookahead = o.lookahead,
        .stream = o.stream,
        .backend = @tagName(o.backend),
        .model = directory,
        .audio_seconds = seconds,
        .score = transcriber.score(),
        .tokens = tokens,
        .segments = if (o.segments) .{ .segmenter = &segmenter } else null,
        .timings = .{
            .load_ms = ms(load_ns),
            .decode_audio_ms = ms(decode_ns),
            .transcribe_ms = ms(transcribe_ns),
            .real_time_factor = if (seconds > 0) ms(transcribe_ns) / 1000 / seconds else 0,
        },
    };
    if (o.format == .json) {
        try std.json.Stringify.value(report, .{ .whitespace = .indent_2 }, out);
        return out.writeByte('\n');
    }
    const off = sty.off();
    if (o.segments) {
        for (segmenter.segments.items) |s| {
            try out.print("{s}", .{sty.on(.dim)});
            try transcript.writeTime(out, s.start_ms, '.');
            try out.writeAll(" → ");
            try transcript.writeTime(out, s.end_ms, '.');
            try out.print("{s}{s}{s}  {s}\n", .{ if (s.language != null) "  " else "", s.language orelse "", off, s.text });
        }
    } else try out.print("{s}{s}{s}\n", .{ sty.on(.header), report.text, off });
    try out.print("{s}{s} · prompt {s} · lookahead {d} ({d} ms) · {d:.1} s of audio in {d:.2} s ({d:.2}× real time) · load {d:.1} s{s}\n", .{
        sty.on(.dim),                        if (report.language) |l| l else "no language tag",
        report.prompt,                       report.lookahead,
        (report.lookahead + 1) * 80,         seconds,
        report.timings.transcribe_ms / 1000, report.timings.real_time_factor,
        report.timings.load_ms / 1000,       off,
    });
}

/// The file stopped decoding after `samples`.
fn midway(diag: *config.Diagnostic, path: []const u8, samples: u64, err: anyerror) anyerror {
    if (err == error.UnsupportedAudioFormat) diag.set("{s}: the audio stops decoding after {d:.1} s", .{ path, @as(f64, @floatFromInt(samples)) / audio.sample_rate });
    return err;
}

/// Prints the transcript so far when it changed: in place on a terminal,
/// one line per change otherwise.
fn showPartial(arena: std.mem.Allocator, transcriber: *const asr.Transcriber, shown: *std.Io.Writer.Allocating, out: *std.Io.Writer, sty: style.Style) !void {
    var now: std.Io.Writer.Allocating = .init(arena);
    try transcriber.writeText(&now.writer);
    if (std.mem.eql(u8, now.written(), shown.written())) return;
    shown.clearRetainingCapacity();
    try shown.writer.writeAll(now.written());
    if (sty.enabled) {
        try out.print("\r\x1b[2K{s}{s}{s}", .{ sty.on(.dim), now.written(), sty.off() });
    } else try out.print("{s}\n", .{now.written()});
    try out.flush();
}

fn lap(io: std.Io, started: *std.Io.Timestamp) u64 {
    const now = std.Io.Clock.awake.now(io);
    defer started.* = now;
    return @intCast(started.durationTo(now).toNanoseconds());
}

fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / 1e6;
}

test "transcribe's options: one file, the lookaheads the checkpoint has, cpu or metal, a format" {
    var diag: config.Diagnostic = .{};
    const o = try parseArgs(&.{ "clip.wav", "--stream", "--lookahead", "13", "--language", "de-DE", "--json", "--model", "dir", "--segments" }, &diag);
    try std.testing.expectEqualStrings("clip.wav", o.file.?);
    try std.testing.expect(o.stream and o.segments);
    try std.testing.expectEqual(Format.json, o.format);
    try std.testing.expectEqual(@as(usize, 13), o.lookahead);
    try std.testing.expectEqualStrings("de-DE", o.language.?);
    try std.testing.expectEqualStrings("dir", o.model.?);
    const defaults = try parseArgs(&.{"clip.wav"}, &diag);
    try std.testing.expectEqual(@as(usize, 3), defaults.lookahead);
    try std.testing.expectEqual(default_backend, defaults.backend);
    try std.testing.expectEqual(Format.text, defaults.format);
    try std.testing.expect(!defaults.segments);
    try std.testing.expectEqual(Format.srt, (try parseArgs(&.{ "clip.wav", "--format", "srt" }, &diag)).format);
    try std.testing.expectEqual(Backend.cpu, (try parseArgs(&.{ "clip.wav", "--backend", "cpu" }, &diag)).backend);
    try std.testing.expectError(error.MissingInput, parseArgs(&.{"--stream"}, &diag));
    try std.testing.expectError(error.InvalidOptionValue, parseArgs(&.{ "a.wav", "--lookahead", "1" }, &diag));
    try std.testing.expectError(error.InvalidOptionValue, parseArgs(&.{ "a.wav", "--backend", "gpu" }, &diag));
    try std.testing.expectError(error.InvalidOptionValue, parseArgs(&.{ "a.wav", "--format", "docx" }, &diag));
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "a.wav", "b.wav" }, &diag));
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "a.wav", "--max-tokens", "4" }, &diag));
    try std.testing.expectError(error.DuplicateOption, parseArgs(&.{ "a.wav", "--json", "--json" }, &diag));
    try std.testing.expectError(error.DuplicateOption, parseArgs(&.{ "a.wav", "--json", "--format", "vtt" }, &diag));
    try std.testing.expectError(error.MissingOptionValue, parseArgs(&.{ "a.wav", "--language" }, &diag));
}

test "--backend metal without Metal is refused before anything is opened" {
    if (inference.metal.enabled) return error.SkipZigTest;
    var diag: config.Diagnostic = .{};
    var buffer: [64]u8 = undefined;
    var out: std.Io.Writer = .fixed(&buffer);
    try std.testing.expectError(error.MetalNotEnabled, run(std.testing.allocator, std.testing.io, "nowhere", .{ .file = "a.wav", .backend = .metal }, &out, .none, &diag));
}
