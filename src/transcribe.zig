//! `nuclis transcribe`: speech to text with Nemotron 3.5 ASR
//! (`inference.audio.nemotron_asr`). The file is decoded to 16 kHz samples
//! and fed to one stream, at once or in 80 ms pieces (`--stream`, printing
//! each partial transcript as it changes); the text report is the transcript
//! and the detected language, `--json` adds the tokens with their times and
//! the timings. The CPU only for now. docs/models/nemotron-asr.md.
const std = @import("std");
const inference = @import("inference");
const style = @import("tui/style.zig");
const config = @import("config.zig");

const audio = inference.audio.audio;
const asr = inference.audio.nemotron_asr;
const rnnt = inference.audio.nemotron_rnnt;
const lookaheads = inference.audio.nemotron_encoder.lookaheads;

/// The checkpoint `transcribe` opens when `--model` names none, under the root.
pub const default_model = "models/nvidia/nemotron-3.5-asr-streaming-0.6b";
/// The longest clip read: an hour of samples, about 230 MB of F32.
pub const max_samples: u32 = 3600 * audio.sample_rate;
/// One `--stream` piece: 80 ms, one encoder row.
const piece_samples = 1280;

pub const Backend = enum { cpu, metal };

pub const Options = struct {
    file: ?[]const u8 = null,
    stream: bool = false,
    lookahead: usize = 3,
    /// A `prompt_dictionary` key; null is the model's default (`auto`).
    language: ?[]const u8 = null,
    backend: Backend = .cpu,
    json: bool = false,
    model: ?[]const u8 = null,
};

/// Parses the words after `transcribe`.
pub fn parseArgs(args: []const []const u8, diag: *config.Diagnostic) !Options {
    var o: Options = .{};
    var lookahead_set = false;
    var backend_set = false;
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
        if (std.mem.eql(u8, word, "--stream") or std.mem.eql(u8, word, "--json")) {
            const flag = if (word[2] == 's') &o.stream else &o.json;
            if (flag.*) return error.DuplicateOption;
            flag.* = true;
            continue;
        }
        const takes_value = for ([_][]const u8{ "--lookahead", "--language", "--backend", "--model" }) |known| {
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
        }
    }
    if (o.file == null) {
        diag.set("give the audio file to transcribe", .{});
        return error.MissingInput;
    }
    return o;
}

/// What `--json` writes; the text report shows the same values.
const Report = struct {
    /// Special tokens skipped, spaces at either end trimmed.
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
    tokens: []const TokenReport,
    timings: Timings,
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
    decode_audio_ms: f64,
    transcribe_ms: f64,
    /// Transcription time over the audio's duration.
    real_time_factor: f64,
};

/// Transcribes `o.file` with the checkpoint directory `directory`.
pub fn run(gpa: std.mem.Allocator, io: std.Io, directory: []const u8, o: Options, out: *std.Io.Writer, sty: style.Style, diag: *config.Diagnostic) !void {
    if (o.backend == .metal) {
        diag.set("transcription runs on the CPU only for now; run with --backend cpu", .{});
        return error.BackendNotYetSupported;
    }
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const path = o.file.?;

    var started = std.Io.Clock.awake.now(io);
    const model = asr.Model.open(gpa, io, directory) catch |err| {
        switch (err) {
            error.FileNotFound => diag.set("{s}: no Nemotron ASR checkpoint there (`nuclis model pull nvidia/nemotron-3.5-asr-streaming-0.6b`, or --model <directory>)", .{directory}),
            error.NotAnAsrModel => diag.set("{s}: not a Nemotron 3.5 ASR checkpoint (config.json names another architecture)", .{directory}),
            else => diag.set("{s}: not a checkpoint nuclis can transcribe with ({s})", .{ directory, @errorName(err) }),
        }
        return err;
    };
    defer model.close();
    const load_ns = lap(io, &started);

    const prompt_name = o.language orelse "auto";
    const prompt = if (o.language) |l| model.promptId(l) orelse {
        diag.set("--language takes a locale the model knows (en-US, de-DE, fr, auto, ...), not {s}", .{l});
        return error.InvalidOptionValue;
    } else model.default_prompt;

    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(audio.max_bytes)) catch |err| {
        switch (err) {
            error.StreamTooLong => diag.set("{s}: larger than {d} MB", .{ path, audio.max_bytes >> 20 }),
            else => diag.set("{s}: {s}", .{ path, @errorName(err) }),
        }
        return err;
    };
    var pcm = audio.decode(gpa, bytes, max_samples) catch |err| {
        if (err == error.UnsupportedAudioFormat) diag.set("{s}: not audio nuclis can read ({s}; or a 16 kHz mono WAV)", .{ path, audio.readable });
        return err;
    };
    defer pcm.deinit(gpa);
    if (pcm.total > pcm.samples.len) {
        diag.set("{s}: longer than {d} minutes", .{ path, max_samples / audio.sample_rate / 60 });
        return error.AudioTooLong;
    }
    const decode_ns = lap(io, &started);

    var transcriber = try asr.Transcriber.init(gpa, model, o.lookahead, prompt);
    defer transcriber.deinit();
    if (o.stream) {
        var shown: std.Io.Writer.Allocating = .init(arena);
        var at: usize = 0;
        while (at < pcm.samples.len) : (at += piece_samples) {
            const emitted = try transcriber.push(pcm.samples[at..@min(pcm.samples.len, at + piece_samples)]);
            if (emitted > 0 and !o.json) try showPartial(arena, &transcriber, &shown, out, sty);
        }
        _ = try transcriber.finish();
        if (!o.json and sty.enabled) try out.writeAll("\r\x1b[2K");
    } else {
        _ = try transcriber.push(pcm.samples);
        _ = try transcriber.finish();
    }
    const transcribe_ns = lap(io, &started);

    var text: std.Io.Writer.Allocating = .init(arena);
    try transcriber.writeText(&text.writer);
    const tokens = try arena.alloc(TokenReport, transcriber.tokens.items.len);
    for (transcriber.tokens.items, tokens) |t, *r| {
        var piece: std.Io.Writer.Allocating = .init(arena);
        try model.table.writePiece(model.table.piece(t.id) orelse "", &piece.writer);
        const start = @as(f64, @floatFromInt(t.row)) * rnnt.seconds_per_row;
        r.* = .{ .id = t.id, .piece = piece.written(), .special = model.table.isSpecial(t.id), .start = start, .end = start + rnnt.seconds_per_row };
    }
    const seconds = @as(f64, @floatFromInt(pcm.samples.len)) / audio.sample_rate;
    const report: Report = .{
        // The model ends a tagged sentence with `▁`, then the tag: a space.
        .text = std.mem.trim(u8, text.written(), " "),
        .language = transcriber.language(),
        .prompt = prompt_name,
        .lookahead = o.lookahead,
        .stream = o.stream,
        .backend = @tagName(o.backend),
        .model = directory,
        .audio_seconds = seconds,
        .tokens = tokens,
        .timings = .{
            .load_ms = ms(load_ns),
            .decode_audio_ms = ms(decode_ns),
            .transcribe_ms = ms(transcribe_ns),
            .real_time_factor = if (seconds > 0) ms(transcribe_ns) / 1000 / seconds else 0,
        },
    };
    if (o.json) {
        try std.json.Stringify.value(report, .{ .whitespace = .indent_2 }, out);
        return out.writeByte('\n');
    }
    const off = sty.off();
    try out.print("{s}{s}{s}\n", .{ sty.on(.header), report.text, off });
    try out.print("{s}{s} · prompt {s} · lookahead {d} ({d} ms) · {d:.1} s of audio in {d:.2} s ({d:.2}× real time) · load {d:.1} s{s}\n", .{
        sty.on(.dim),                        if (report.language) |l| l else "no language tag",
        report.prompt,                       report.lookahead,
        (report.lookahead + 1) * 80,         seconds,
        report.timings.transcribe_ms / 1000, report.timings.real_time_factor,
        report.timings.load_ms / 1000,       off,
    });
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

test "transcribe's options: one file, the lookaheads the checkpoint has, cpu or metal" {
    var diag: config.Diagnostic = .{};
    const o = try parseArgs(&.{ "clip.wav", "--stream", "--lookahead", "13", "--language", "de-DE", "--json", "--model", "dir" }, &diag);
    try std.testing.expectEqualStrings("clip.wav", o.file.?);
    try std.testing.expect(o.stream and o.json);
    try std.testing.expectEqual(@as(usize, 13), o.lookahead);
    try std.testing.expectEqualStrings("de-DE", o.language.?);
    try std.testing.expectEqualStrings("dir", o.model.?);
    const defaults = try parseArgs(&.{"clip.wav"}, &diag);
    try std.testing.expectEqual(@as(usize, 3), defaults.lookahead);
    try std.testing.expectEqual(Backend.cpu, defaults.backend);
    try std.testing.expectError(error.MissingInput, parseArgs(&.{"--stream"}, &diag));
    try std.testing.expectError(error.InvalidOptionValue, parseArgs(&.{ "a.wav", "--lookahead", "1" }, &diag));
    try std.testing.expectError(error.InvalidOptionValue, parseArgs(&.{ "a.wav", "--backend", "gpu" }, &diag));
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "a.wav", "b.wav" }, &diag));
    try std.testing.expectError(error.UnknownOption, parseArgs(&.{ "a.wav", "--max-tokens", "4" }, &diag));
    try std.testing.expectError(error.DuplicateOption, parseArgs(&.{ "a.wav", "--json", "--json" }, &diag));
    try std.testing.expectError(error.MissingOptionValue, parseArgs(&.{ "a.wav", "--language" }, &diag));
}

test "--backend metal is refused before anything is opened" {
    var diag: config.Diagnostic = .{};
    var buffer: [64]u8 = undefined;
    var out: std.Io.Writer = .fixed(&buffer);
    try std.testing.expectError(error.BackendNotYetSupported, run(std.testing.allocator, std.testing.io, "nowhere", .{ .file = "a.wav", .backend = .metal }, &out, .none, &diag));
}
