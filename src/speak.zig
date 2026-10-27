//! `nuclis speak`: text into a 24 kHz mono 16-bit WAV through Qwen3-TTS
//! CustomVoice (`inference.models.qwen3_tts_runtime.Voice`): one of the
//! checkpoint's speakers, a language or `auto`, and on the 1.7B an
//! instruction for the delivery. On Metal by default when built with it;
//! `--stream` writes each chunk of frames as it is decoded and reports the
//! time to the first audio. docs/models/qwen3-tts.md.
const std = @import("std");
const inference = @import("inference");
const style = @import("tui/style.zig");
const config = @import("config.zig");

const tts = inference.models.qwen3_tts;
const speech = inference.models.qwen3_tts_runtime;

pub const Backend = speech.Backend;
pub const sample_rate = 24000;
/// The checkpoints `--model` defaults to, under the models directory, the first present.
pub const default_models = [_][]const u8{ "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice" };
/// Longest text accepted, in bytes.
const max_text_bytes = 16 * 1024;

pub const Options = struct {
    words: std.ArrayList([]const u8) = .empty,
    voice: ?[]const u8 = null,
    language: ?[]const u8 = null,
    instruct: ?[]const u8 = null,
    model: ?[]const u8 = null,
    seed: ?u64 = null,
    backend: ?Backend = null,
    output: ?[]const u8 = null,
    /// Write each chunk as it is decoded and report the time to the first audio.
    stream: bool = false,
    /// Frames per chunk, and in the first (1 … `speech.max_chunk`).
    chunk: ?usize = null,
    first_chunk: ?usize = null,
    /// Time every Metal dispatch and print where the device's time went.
    profile: bool = false,
    /// Split a frame's time into the talker and the predictor (on Metal, one
    /// more commit a frame).
    split: bool = false,
    /// On Metal, sample a frame's groups 1–15 on the host, a command buffer each.
    host_sampling: bool = false,
};

/// Parses the words after `speak`; `arena` owns the list. Words that are not
/// options are the text, joined by spaces; after `--`, every word is.
pub fn parseArgs(arena: std.mem.Allocator, args: []const []const u8, diag: *config.Diagnostic) !Options {
    var o: Options = .{};
    var i: usize = 0;
    var literal = false;
    while (i < args.len) : (i += 1) {
        const word = args[i];
        if (literal or !std.mem.startsWith(u8, word, "-") or word.len == 1) {
            try o.words.append(arena, word);
            continue;
        }
        if (std.mem.eql(u8, word, "--")) {
            literal = true;
            continue;
        }
        if (std.mem.eql(u8, word, "--stream")) {
            o.stream = true;
            continue;
        }
        if (std.mem.eql(u8, word, "--profile")) {
            o.profile = true;
            continue;
        }
        if (std.mem.eql(u8, word, "--split")) {
            o.split = true;
            continue;
        }
        if (std.mem.eql(u8, word, "--host-sampling")) {
            o.host_sampling = true;
            continue;
        }
        const known = [_][]const u8{ "--voice", "--language", "--instruct", "--model", "--seed", "--backend", "--output", "-o", "--chunk", "--first-chunk" };
        const takes_value = for (known) |k| {
            if (std.mem.eql(u8, word, k)) break true;
        } else false;
        if (!takes_value) {
            diag.set("{s} is not an option of speak (`nuclis speak --help`)", .{word});
            return error.UnknownOption;
        }
        i += 1;
        if (i == args.len) {
            diag.set("{s} needs a value", .{word});
            return error.MissingOptionValue;
        }
        const value = args[i];
        const slot: *?[]const u8 = if (std.mem.eql(u8, word, "--voice"))
            &o.voice
        else if (std.mem.eql(u8, word, "--language"))
            &o.language
        else if (std.mem.eql(u8, word, "--instruct"))
            &o.instruct
        else if (std.mem.eql(u8, word, "--model"))
            &o.model
        else if (std.mem.eql(u8, word, "--output") or std.mem.eql(u8, word, "-o"))
            &o.output
        else if (std.mem.eql(u8, word, "--chunk") or std.mem.eql(u8, word, "--first-chunk")) {
            const chunk_slot = if (std.mem.eql(u8, word, "--chunk")) &o.chunk else &o.first_chunk;
            if (chunk_slot.* != null) return error.DuplicateOption;
            const n = std.fmt.parseInt(usize, value, 10) catch 0;
            if (n == 0 or n > speech.max_chunk) {
                diag.set("{s} takes a number of frames from 1 to {d}, not {s}", .{ word, speech.max_chunk, value });
                return error.InvalidOptionValue;
            }
            chunk_slot.* = n;
            continue;
        } else if (std.mem.eql(u8, word, "--seed")) {
            if (o.seed != null) return error.DuplicateOption;
            o.seed = std.fmt.parseInt(u64, value, 10) catch {
                diag.set("--seed takes a whole number, not {s}", .{value});
                return error.InvalidOptionValue;
            };
            continue;
        } else {
            if (o.backend != null) return error.DuplicateOption;
            o.backend = std.meta.stringToEnum(Backend, value) orelse {
                diag.set("--backend takes cpu or metal, not {s}", .{value});
                return error.InvalidOptionValue;
            };
            continue;
        };
        if (slot.* != null) return error.DuplicateOption;
        slot.* = value;
    }
    if (o.words.items.len == 0) {
        diag.set("give the text to speak (`nuclis speak \"Hello there.\" -o hello.wav`)", .{});
        return error.MissingInput;
    }
    if (o.output == null) {
        diag.set("name the WAV file to write with -o <file.wav>", .{});
        return error.MissingOutput;
    }
    return o;
}

/// The checkpoint directory: `--model`, else the first of `default_models`
/// present under `<root>/models`.
fn locate(arena: std.mem.Allocator, io: std.Io, root: ?[]const u8, explicit: ?[]const u8, diag: *config.Diagnostic) ![]const u8 {
    if (explicit) |path| return path;
    if (root) |dir| for (default_models) |name| {
        const path = try std.fs.path.join(arena, &.{ dir, "models", name });
        const config_path = try std.fs.path.join(arena, &.{ path, "config.json" });
        if (std.Io.Dir.cwd().access(io, config_path, .{})) |_| return path else |_| {}
    };
    diag.set("no Qwen3-TTS checkpoint under the models directory; pull {s} (`nuclis model pull`) or pass --model <dir>", .{default_models[0]});
    return error.MissingModel;
}

/// A 44-byte RIFF/WAVE header for `samples` 16-bit mono samples at `rate`.
pub fn wavHeader(samples: usize, rate: u32) ![44]u8 {
    const data: u32 = std.math.cast(u32, samples * 2) orelse return error.AudioTooLong;
    if (data > std.math.maxInt(u32) - 36) return error.AudioTooLong;
    var h: [44]u8 = undefined;
    @memcpy(h[0..4], "RIFF");
    std.mem.writeInt(u32, h[4..8], 36 + data, .little);
    @memcpy(h[8..16], "WAVEfmt ");
    std.mem.writeInt(u32, h[16..20], 16, .little);
    std.mem.writeInt(u16, h[20..22], 1, .little); // PCM
    std.mem.writeInt(u16, h[22..24], 1, .little); // mono
    std.mem.writeInt(u32, h[24..28], rate, .little);
    std.mem.writeInt(u32, h[28..32], rate * 2, .little);
    std.mem.writeInt(u16, h[32..34], 2, .little);
    std.mem.writeInt(u16, h[34..36], 16, .little);
    @memcpy(h[36..40], "data");
    std.mem.writeInt(u32, h[40..44], data, .little);
    return h;
}

/// A sample in [−1, 1] as 16-bit PCM, rounded to nearest.
pub fn pcm16(x: f32) i16 {
    const clamped = std.math.clamp(x, -1, 1);
    return @intFromFloat(@round(clamped * 32767));
}

/// `nuclis speak <words…>`: parses `words`, synthesizes, writes the WAV, reports.
pub fn run(alloc: std.mem.Allocator, io: std.Io, root: ?[]const u8, words: []const []const u8, out: *std.Io.Writer, sty: style.Style, diag: *config.Diagnostic) !void {
    var arena_state = std.heap.ArenaAllocator.init(alloc);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const options = try parseArgs(arena, words, diag);
    const backend = options.backend orelse if (inference.metal.enabled) Backend.metal else .cpu;
    if (backend == .metal and !inference.metal.enabled) {
        diag.set("this build has no Metal backend; --backend cpu", .{});
        return error.MetalNotEnabled;
    }
    const text = try std.mem.join(arena, " ", options.words.items);
    if (text.len > max_text_bytes) {
        diag.set("the text is {d} bytes; speak takes at most {d} (a few sentences at a time)", .{ text.len, max_text_bytes });
        return error.TextTooLong;
    }
    const dir = try locate(arena, io, root, options.model, diag);
    const seed = options.seed orelse 0;
    const language = options.language orelse "auto";
    const speaker = options.voice orelse "ryan";

    const load_start = std.Io.Clock.awake.now(io);
    var bind_diag: tts.Diagnostic = .{};
    const voice = speech.Voice.open(alloc, io, dir, &bind_diag, .{ .backend = backend }) catch |err| {
        if (bind_diag.len > 0) diag.set("{s}: {s}", .{ dir, bind_diag.name() }) else diag.set("{s}", .{dir});
        return err;
    };
    defer voice.close(io);
    const load_ms = load_start.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds();
    if (options.profile) {
        const gpu = if (voice.gpu) |*g| g else {
            diag.set("--profile times Metal dispatches; it needs --backend metal", .{});
            return error.InvalidOptionValue;
        };
        var message: [256]u8 = @splat(0);
        gpu.enableProfiling(4096, &message) catch |err| {
            diag.set("{s}", .{std.mem.sliceTo(&message, 0)});
            return err;
        };
    }
    try out.print("{s}{s}{s} ({s}), speaker {s}, language {s}, seed {d}, {s}; loaded in {d:.1} s\n", .{ sty.on(.header), std.fs.path.basename(dir), sty.off(), voice.config.size, speaker, language, seed, @tagName(backend), @as(f64, @floatFromInt(load_ms)) / 1000 });
    try out.flush();

    // The clock of the first audio starts when the text is handed over.
    const start = std.Io.Clock.awake.now(io);
    var prompt = voice.prompt(alloc, text, speaker, language, options.instruct) catch |err| {
        switch (err) {
            error.InstructionUnsupported => diag.set("{s} ({s}) takes no instruction; the 1.7B CustomVoice does", .{ std.fs.path.basename(dir), voice.config.size }),
            error.UnknownSpeaker => diag.set("{s} is not a speaker of this model: {s}", .{ speaker, try names(arena, voice.config.speakers) }),
            error.UnknownLanguage => diag.set("{s} is not a language of this model: auto, {s}", .{ language, try names(arena, voice.config.languages) }),
            else => {},
        }
        return err;
    };
    defer prompt.deinit(alloc);

    const path = options.output.?;
    const file = std.Io.Dir.cwd().createFile(io, path, .{}) catch |err| {
        diag.set("{s}", .{path});
        return err;
    };
    defer file.close(io);
    var file_buffer: [64 * 1024]u8 = undefined;
    var writer = file.writerStreaming(io, &file_buffer);
    // A placeholder header; its sizes are written once the length is known.
    try writer.interface.writeAll(&try wavHeader(0, sample_rate));

    var synth = try voice.synthesizer(alloc, prompt.rows, seed);
    defer synth.deinit();
    synth.chunk = options.chunk orelse speech.default_chunk;
    synth.first_chunk = options.first_chunk orelse speech.default_first_chunk;
    synth.split = options.split;
    synth.host_sampling = options.host_sampling;
    const per = 1920;
    const buffer = try arena.alloc(f32, @max(synth.chunk, synth.first_chunk) * per);
    var first_ms: ?f64 = null;
    var samples: usize = 0;
    var pending: std.ArrayList(i16) = .empty;
    defer pending.deinit(alloc);
    while (true) {
        const n = try synth.read(buffer);
        if (n == 0) break;
        if (first_ms == null) first_ms = @as(f64, @floatFromInt(start.durationTo(std.Io.Clock.awake.now(io)).toMicroseconds())) / 1000;
        samples += n;
        if (options.stream) {
            for (buffer[0..n]) |x| try writer.interface.writeInt(i16, pcm16(x), .little);
            try writer.interface.flush();
        } else {
            try pending.ensureUnusedCapacity(alloc, n);
            for (buffer[0..n]) |x| pending.appendAssumeCapacity(pcm16(x));
        }
    }
    const seconds = @as(f64, @floatFromInt(start.durationTo(std.Io.Clock.awake.now(io)).toMicroseconds())) / 1e6;
    for (pending.items) |x| try writer.interface.writeInt(i16, x, .little);
    try writer.interface.flush();
    try file.writePositionalAll(io, &try wavHeader(samples, sample_rate), 0);

    const frames = synth.frames;
    const audio_seconds = @as(f64, @floatFromInt(samples)) / sample_rate;
    try out.print("wrote {s}: {d} frames, {d:.2} s of audio in {d:.2} s ({d:.0} ms a frame, {d:.2}× real time)\n", .{
        path,
        frames,
        audio_seconds,
        seconds,
        if (frames > 0) seconds * 1000 / @as(f64, @floatFromInt(frames)) else 0,
        if (seconds > 0) audio_seconds / seconds else 0,
    });
    const t = synth.timing;
    const per_frame = struct {
        fn ms(ns: u64, count: usize) f64 {
            return if (count == 0) 0 else @as(f64, @floatFromInt(ns)) / 1e6 / @as(f64, @floatFromInt(count));
        }
    }.ms;
    try out.print("prefill {d:.0} ms; a frame: codes {d:.1} ms, codec {d:.1} ms", .{ @as(f64, @floatFromInt(t.prefill)) / 1e6, per_frame(t.frames, frames), per_frame(t.codec, frames) });
    if (backend == .cpu or synth.split) try out.print(" (talker {d:.1} ms, predictor {d:.1} ms{s})", .{ per_frame(t.talker, frames), per_frame(t.predictor, frames), if (backend == .metal) " on the device" else "" });
    if (first_ms) |ms| try out.print("; first audio {d:.0} ms after the text (chunks of {d} frames, the first {d})", .{ ms, synth.chunk, synth.first_chunk });
    if (voice.plan) |plan| if (plan.disagreements > 0) try out.print("; the host overruled the device's draw {d} times", .{plan.disagreements});
    try out.writeByte('\n');
    if (voice.gpu) |*g| if (g.profile) |*p| try printProfile(arena, out, p, frames);
}

/// The profile's kernels by device time, each with its share and, for a
/// matrix, the weight bytes it read per second.
fn printProfile(arena: std.mem.Allocator, out: *std.Io.Writer, p: *const inference.metal.Profile, frames: usize) !void {
    const Entry = struct { key: inference.metal.Profile.Key, total: inference.metal.Profile.Total };
    var entries: std.ArrayList(Entry) = .empty;
    var sum: f64 = 0;
    var it = p.totals.iterator();
    while (it.next()) |e| {
        try entries.append(arena, .{ .key = e.key_ptr.*, .total = e.value_ptr.* });
        sum += e.value_ptr.seconds;
    }
    std.mem.sort(Entry, entries.items, {}, struct {
        fn more(_: void, a: Entry, b: Entry) bool {
            return a.total.seconds > b.total.seconds;
        }
    }.more);
    try out.print("device time by kernel ({d} command buffers, {d:.1} ms timed of {d:.1} ms busy; per frame over {d} frames):\n", .{ p.command_buffers, sum * 1e3, p.gpu_seconds * 1e3, frames });
    for (entries.items[0..@min(entries.items.len, 24)]) |e| {
        const per = if (frames > 0) e.total.seconds * 1e3 / @as(f64, @floatFromInt(frames)) else 0;
        try out.print("  {s:<26} {d:>5}x{d:<5} {d:>7} dispatches {d:>7.2} ms/frame {d:>5.1} %", .{ @tagName(e.key.kernel), e.key.rows, e.key.columns, e.total.dispatches, per, 100 * e.total.seconds / sum });
        if (e.total.bytes > 0) try out.print(" {d:>6.0} GB/s", .{@as(f64, @floatFromInt(e.total.bytes)) / e.total.seconds / 1e9});
        try out.writeByte('\n');
    }
}

fn names(arena: std.mem.Allocator, items: anytype) ![]const u8 {
    var list: std.ArrayList(u8) = .empty;
    for (items, 0..) |item, i| {
        if (i > 0) try list.appendSlice(arena, ", ");
        try list.appendSlice(arena, item.name);
    }
    return list.items;
}

test "speak's arguments: the text joined, the output required, the chunk bounded" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diag: config.Diagnostic = .{};
    const o = try parseArgs(arena, &.{ "Hello", "there.", "--voice", "vivian", "--instruct", "Whisper.", "-o", "x.wav", "--seed", "4", "--stream", "--chunk", "2" }, &diag);
    try std.testing.expect(o.stream);
    try std.testing.expectEqual(@as(?usize, 2), o.chunk);
    try std.testing.expectError(error.InvalidOptionValue, parseArgs(arena, &.{ "Hi", "-o", "x.wav", "--chunk", "0" }, &diag));
    try std.testing.expectError(error.InvalidOptionValue, parseArgs(arena, &.{ "Hi", "-o", "x.wav", "--chunk", "33" }, &diag));
    try std.testing.expectEqual(@as(usize, 2), o.words.items.len);
    try std.testing.expectEqualStrings("vivian", o.voice.?);
    try std.testing.expectEqualStrings("x.wav", o.output.?);
    try std.testing.expectEqual(@as(?u64, 4), o.seed);
    try std.testing.expectError(error.MissingOutput, parseArgs(arena, &.{"Hello"}, &diag));
    try std.testing.expectError(error.MissingInput, parseArgs(arena, &.{ "-o", "x.wav" }, &diag));
    try std.testing.expectError(error.UnknownOption, parseArgs(arena, &.{ "Hi", "--speed", "2" }, &diag));
    try std.testing.expectError(error.DuplicateOption, parseArgs(arena, &.{ "Hi", "-o", "a.wav", "--output", "b.wav" }, &diag));
    try std.testing.expectError(error.InvalidOptionValue, parseArgs(arena, &.{ "Hi", "--backend", "gpu" }, &diag));
}

test "the WAV header and the 16-bit samples" {
    const h = try wavHeader(3, sample_rate);
    try std.testing.expectEqualStrings("RIFF", h[0..4]);
    try std.testing.expectEqual(@as(u32, 36 + 6), std.mem.readInt(u32, h[4..8], .little));
    try std.testing.expectEqual(@as(u32, 48000), std.mem.readInt(u32, h[28..32], .little));
    try std.testing.expectEqual(@as(u32, 6), std.mem.readInt(u32, h[40..44], .little));
    try std.testing.expectEqual(@as(i16, 32767), pcm16(1.5));
    try std.testing.expectEqual(@as(i16, -32767), pcm16(-1));
    try std.testing.expectEqual(@as(i16, 0), pcm16(0));
}
