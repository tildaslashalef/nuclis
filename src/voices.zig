//! Saved voices for Qwen3-TTS Base: a reference clip, and optionally its
//! transcript, registered once under a name and used by every request that
//! names it. Each lives in `<root>/voices/<name>/`: `reference.wav`, the
//! clip decoded to 24 kHz mono 16-bit, and `voice.json`, its record and what
//! the encoders made of it, keyed by the file that made it (reference codes
//! by the codec file's SHA-256, a speaker embedding by the model file's).
//! A key the model in use lacks is computed from `reference.wav` and written
//! back (`ensure`), so one voice serves either Base. A voice appears whole
//! (a directory renamed into place) and its record is replaced by rename.
//! Bounds are host constants. docs/guide/api.md § Voices.
const std = @import("std");
const inference = @import("inference");
const catalog = @import("catalog.zig");
const model = @import("model.zig");
const speak = @import("speak.zig");

const Allocator = std.mem.Allocator;
const audio = inference.audio.audio;
const speech = inference.models.qwen3_tts_runtime;
pub const Frame = inference.audio.qwen3_codec.Frame;

pub const sample_rate = 24000;
/// A clip's length after decoding: 1 to 30 s.
pub const min_samples = 1 * sample_rate;
pub const max_samples = 30 * sample_rate;
pub const max_transcript_bytes = 2048;
pub const max_voices = 256;
pub const max_name = 64;
const record_file = "voice.json";
const clip_file = "reference.wav";
/// A record holds a few codes and embeddings, each tens of KB as base64.
const max_record_bytes = 1 << 20;
const max_clip_bytes = 44 + 2 * max_samples;

pub const Error = error{
    InvalidVoiceName,
    ReservedVoiceName,
    VoiceExists,
    VoiceNotFound,
    TooManyVoices,
    ClipTooShort,
    ClipTooLong,
    TranscriptTooLong,
    InvalidTranscript,
    /// `voice.json` unreadable, or `reference.wav` not the one it records.
    InvalidVoiceRecord,
};

/// `voice.json`.
const Record = struct {
    schema_version: u32 = 1,
    name: []const u8,
    reference_text: ?[]const u8 = null,
    /// Of `reference.wav`, lowercase hex.
    reference_sha256: []const u8,
    samples: u64,
    /// RFC 3339 UTC.
    created: []const u8,
    /// Codec file SHA-256 → base64 of little-endian u16 `[frames][16]`.
    codes: std.json.ArrayHashMap([]const u8) = .{},
    /// Model file SHA-256 → base64 of little-endian f32 `[width]`.
    embeddings: std.json.ArrayHashMap([]const u8) = .{},
};

/// A voice as listed.
pub const Summary = struct {
    name: []const u8,
    reference_text: ?[]const u8,
    seconds: f64,
    created: []const u8,
};

/// The files a voice's derived data is keyed by: the SHA-256 (lowercase
/// hex) of the model's `model.safetensors` and of its codec's.
pub const Keys = struct { model: []const u8, codec: []const u8 };

/// What makes a voice's data, from 24 kHz mono samples: the speaker
/// embedding (`dim` floats) and the reference codes (owned by `gpa`).
pub const Encoders = struct {
    context: *anyopaque,
    dim: usize,
    embed: *const fn (context: *anyopaque, gpa: Allocator, samples: []const f32, out: []f32) anyerror!void,
    codes: *const fn (context: *anyopaque, gpa: Allocator, samples: []const f32) anyerror![]Frame,
};

/// An open Base's encoders.
pub fn encodersOf(voice: *speech.Voice) Encoders {
    const wrap = struct {
        fn embed(context: *anyopaque, gpa: Allocator, samples: []const f32, out: []f32) anyerror!void {
            const v: *speech.Voice = @ptrCast(@alignCast(context));
            try v.speakerEmbedding(gpa, samples, out);
        }
        fn codes(context: *anyopaque, gpa: Allocator, samples: []const f32) anyerror![]Frame {
            const v: *speech.Voice = @ptrCast(@alignCast(context));
            return v.referenceCodes(gpa, samples);
        }
    };
    return .{ .context = voice, .dim = voice.config.talker.hidden, .embed = wrap.embed, .codes = wrap.codes };
}

/// A voice ready for a prompt, owned (`deinit`).
pub const Ready = struct {
    name: []u8,
    reference_text: ?[]u8,
    embedding: []f32,
    /// The reference codes; empty without a transcript (x-vector only).
    frames: []Frame,

    pub fn clone(self: *const Ready) inference.models.qwen3_tts.Clone {
        return .{ .embedding = self.embedding, .frames = self.frames };
    }

    pub fn deinit(self: *Ready, alloc: Allocator) void {
        alloc.free(self.name);
        if (self.reference_text) |t| alloc.free(t);
        alloc.free(self.embedding);
        alloc.free(self.frames);
        self.* = undefined;
    }
};

/// Whether `name` may name a voice: `[a-z0-9][a-z0-9_-]{0,63}`, and not a
/// catalogue speaker (a request's `voice` would be ambiguous).
pub fn checkName(name: []const u8) Error!void {
    if (name.len == 0 or name.len > max_name) return error.InvalidVoiceName;
    for (name, 0..) |c, i| switch (c) {
        'a'...'z', '0'...'9' => {},
        '_', '-' => if (i == 0) return error.InvalidVoiceName,
        else => return error.InvalidVoiceName,
    };
    for (catalog.audio_entries) |entry| for (entry.speakers) |speaker| {
        if (std.mem.eql(u8, speaker, name)) return error.ReservedVoiceName;
    };
}

/// The voices under one root's `voices/` directory.
pub const Store = struct {
    io: std.Io,
    /// `<root>/voices`, borrowed.
    path: []const u8,
    /// The most voices kept (a host constant; tests lower it).
    limit: usize = max_voices,

    /// Decodes `bytes` (any format `audio.decodeAt` reads) to 24 kHz and
    /// saves it as `name`, with `transcript` (trimmed; empty is none).
    /// Strings in the summary are `alloc`'s.
    pub fn add(self: Store, alloc: Allocator, name: []const u8, bytes: []const u8, transcript: ?[]const u8, now_seconds: i64) !Summary {
        try checkName(name);
        const text = try checkTranscript(transcript);
        var pcm = try audio.decodeAt(alloc, bytes, max_samples + 1, sample_rate);
        defer pcm.deinit(alloc);
        if (pcm.total > max_samples) return error.ClipTooLong;
        if (pcm.samples.len < min_samples) return error.ClipTooShort;

        var root = try std.Io.Dir.cwd().createDirPathOpen(self.io, self.path, .{ .open_options = .{ .iterate = true } });
        defer root.close(self.io);
        if (try exists(self.io, root, name)) return error.VoiceExists;
        if (try countVoices(self.io, root) >= self.limit) return error.TooManyVoices;

        const wav = try alloc.alloc(u8, 44 + 2 * pcm.samples.len);
        defer alloc.free(wav);
        wav[0..44].* = try speak.wavHeader(pcm.samples.len, sample_rate);
        for (pcm.samples, 0..) |x, i| std.mem.writeInt(i16, wav[44 + 2 * i ..][0..2], speak.pcm16(x), .little);
        var stamp: [20]u8 = undefined;
        const record: Record = .{
            .name = name,
            .reference_text = text,
            .reference_sha256 = &sha256Hex(wav),
            .samples = pcm.samples.len,
            .created = model.rfc3339(&stamp, now_seconds),
        };

        // Built beside the voices under a temporary name, then renamed into
        // place without replacing: a voice appears whole or not at all.
        var random: [8]u8 = undefined;
        self.io.random(&random);
        var temp_buffer: [32]u8 = undefined;
        const temp = std.fmt.bufPrint(&temp_buffer, ".add-{x}", .{&random}) catch unreachable;
        try root.createDir(self.io, temp, .default_dir);
        errdefer root.deleteTree(self.io, temp) catch {};
        var dir = try root.openDir(self.io, temp, .{});
        defer dir.close(self.io);
        try dir.writeFile(self.io, .{ .sub_path = clip_file, .data = wav });
        try writeRecord(alloc, self.io, dir, record, false);
        std.Io.Dir.renamePreserve(root, temp, root, name, self.io) catch |err| return switch (err) {
            error.PathAlreadyExists => error.VoiceExists,
            else => err,
        };
        return .{
            .name = try alloc.dupe(u8, name),
            .reference_text = if (text) |t| try alloc.dupe(u8, t) else null,
            .seconds = @as(f64, @floatFromInt(pcm.samples.len)) / sample_rate,
            .created = try alloc.dupe(u8, record.created),
        };
    }

    /// Every voice by name; a directory without a readable record is
    /// skipped. Strings are `alloc`'s (an arena suits).
    pub fn list(self: Store, alloc: Allocator) ![]Summary {
        var root = std.Io.Dir.cwd().openDir(self.io, self.path, .{ .iterate = true }) catch |err| switch (err) {
            error.FileNotFound => return &.{},
            else => return err,
        };
        defer root.close(self.io);
        var out: std.ArrayList(Summary) = .empty;
        var iterator = root.iterate();
        while (try iterator.next(self.io)) |entry| {
            if (entry.kind != .directory or entry.name[0] == '.') continue;
            checkName(entry.name) catch continue;
            var parsed = readRecord(alloc, self.io, root, entry.name) catch continue;
            defer parsed.deinit();
            const r = parsed.value;
            try out.append(alloc, .{
                .name = try alloc.dupe(u8, entry.name),
                .reference_text = if (r.reference_text) |t| try alloc.dupe(u8, t) else null,
                .seconds = @as(f64, @floatFromInt(r.samples)) / sample_rate,
                .created = try alloc.dupe(u8, r.created),
            });
        }
        std.mem.sort(Summary, out.items, {}, struct {
            fn less(_: void, a: Summary, b: Summary) bool {
                return std.mem.lessThan(u8, a.name, b.name);
            }
        }.less);
        return out.toOwnedSlice(alloc);
    }

    /// Deletes `name` and everything derived from it.
    pub fn remove(self: Store, name: []const u8) !void {
        try checkName(name);
        var root = std.Io.Dir.cwd().openDir(self.io, self.path, .{}) catch |err| switch (err) {
            error.FileNotFound => return error.VoiceNotFound,
            else => return err,
        };
        defer root.close(self.io);
        if (!try exists(self.io, root, name)) return error.VoiceNotFound;
        try root.deleteTree(self.io, name);
    }

    /// `name` ready for the model `keys` names: its embedding and (with a
    /// transcript) its reference codes, each computed by `encoders` from
    /// `reference.wav` and saved when the record lacks it.
    pub fn ensure(self: Store, alloc: Allocator, name: []const u8, keys: Keys, encoders: Encoders) !Ready {
        try checkName(name);
        var root = std.Io.Dir.cwd().openDir(self.io, self.path, .{}) catch |err| switch (err) {
            error.FileNotFound => return error.VoiceNotFound,
            else => return err,
        };
        defer root.close(self.io);
        if (!try exists(self.io, root, name)) return error.VoiceNotFound;
        var dir = try root.openDir(self.io, name, .{});
        defer dir.close(self.io);
        var parsed = try readRecord(alloc, self.io, root, name);
        defer parsed.deinit();
        const record = &parsed.value;
        const icl = record.reference_text != null;
        const arena = parsed.arena.allocator();

        const have_embedding = record.embeddings.map.get(keys.model);
        const have_codes = if (icl) record.codes.map.get(keys.codec) else null;
        if (have_embedding == null or (icl and have_codes == null)) {
            const wav = dir.readFileAlloc(self.io, clip_file, alloc, .limited(max_clip_bytes)) catch return error.InvalidVoiceRecord;
            defer alloc.free(wav);
            if (!std.mem.eql(u8, &sha256Hex(wav), record.reference_sha256)) return error.InvalidVoiceRecord;
            var pcm = audio.decodeWavAt(alloc, wav, max_samples, sample_rate) catch return error.InvalidVoiceRecord;
            defer pcm.deinit(alloc);
            if (have_embedding == null) {
                const embedding = try alloc.alloc(f32, encoders.dim);
                defer alloc.free(embedding);
                try encoders.embed(encoders.context, alloc, pcm.samples, embedding);
                try record.embeddings.map.put(arena, try arena.dupe(u8, keys.model), try base64(arena, std.mem.sliceAsBytes(embedding)));
            }
            if (icl and have_codes == null) {
                const frames = try encoders.codes(encoders.context, alloc, pcm.samples);
                defer alloc.free(frames);
                try record.codes.map.put(arena, try arena.dupe(u8, keys.codec), try base64(arena, std.mem.sliceAsBytes(frames)));
            }
            try writeRecord(alloc, self.io, dir, record.*, true);
        }

        const embedding_bytes = try unbase64(alloc, record.embeddings.map.get(keys.model).?);
        defer alloc.free(embedding_bytes);
        if (embedding_bytes.len != encoders.dim * 4) return error.InvalidVoiceRecord;
        const embedding = try alloc.alloc(f32, encoders.dim);
        errdefer alloc.free(embedding);
        for (embedding, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, embedding_bytes[i * 4 ..][0..4], .little));
        var frames: []Frame = &.{};
        errdefer alloc.free(frames);
        if (icl) {
            const code_bytes = try unbase64(alloc, record.codes.map.get(keys.codec).?);
            defer alloc.free(code_bytes);
            const frame_bytes = @sizeOf(Frame);
            if (code_bytes.len == 0 or code_bytes.len % frame_bytes != 0) return error.InvalidVoiceRecord;
            frames = try alloc.alloc(Frame, code_bytes.len / frame_bytes);
            for (frames, 0..) |*frame, t| for (frame, 0..) |*code, g| {
                code.* = std.mem.readInt(u16, code_bytes[(t * 16 + g) * 2 ..][0..2], .little);
                if (code.* >= inference.models.qwen3_tts.codebook_size) return error.InvalidVoiceRecord;
            };
        }
        const owned_name = try alloc.dupe(u8, name);
        errdefer alloc.free(owned_name);
        return .{
            .name = owned_name,
            .reference_text = if (record.reference_text) |t| try alloc.dupe(u8, t) else null,
            .embedding = embedding,
            .frames = frames,
        };
    }
};

/// A transcript as saved: trimmed, null when empty, bounded, UTF-8.
fn checkTranscript(transcript: ?[]const u8) Error!?[]const u8 {
    const text = std.mem.trim(u8, transcript orelse return null, " \t\r\n");
    if (text.len == 0) return null;
    if (text.len > max_transcript_bytes) return error.TranscriptTooLong;
    if (!std.unicode.utf8ValidateSlice(text)) return error.InvalidTranscript;
    return text;
}

fn exists(io: std.Io, root: std.Io.Dir, name: []const u8) !bool {
    var dir = root.openDir(io, name, .{}) catch |err| switch (err) {
        error.FileNotFound => return false,
        else => return err,
    };
    dir.close(io);
    return true;
}

fn countVoices(io: std.Io, root: std.Io.Dir) !usize {
    var count: usize = 0;
    var iterator = root.iterate();
    while (try iterator.next(io)) |entry| {
        if (entry.kind == .directory and entry.name[0] != '.') count += 1;
    }
    return count;
}

fn readRecord(alloc: Allocator, io: std.Io, root: std.Io.Dir, name: []const u8) !std.json.Parsed(Record) {
    var dir = try root.openDir(io, name, .{});
    defer dir.close(io);
    const bytes = dir.readFileAlloc(io, record_file, alloc, .limited(max_record_bytes)) catch return error.InvalidVoiceRecord;
    defer alloc.free(bytes);
    const parsed = std.json.parseFromSlice(Record, alloc, bytes, .{ .ignore_unknown_fields = true, .allocate = .alloc_always }) catch return error.InvalidVoiceRecord;
    if (parsed.value.schema_version != 1 or parsed.value.reference_sha256.len != 64 or !std.mem.eql(u8, parsed.value.name, name)) {
        parsed.deinit();
        return error.InvalidVoiceRecord;
    }
    return parsed;
}

/// Writes `voice.json` into `dir`; with `replace`, through a temporary
/// file renamed over the old record, so a reader sees one or the other.
fn writeRecord(alloc: Allocator, io: std.Io, dir: std.Io.Dir, record: Record, replace: bool) !void {
    var buffer: std.Io.Writer.Allocating = .init(alloc);
    defer buffer.deinit();
    try std.json.Stringify.value(record, .{ .whitespace = .indent_1 }, &buffer.writer);
    try buffer.writer.writeByte('\n');
    if (!replace) return dir.writeFile(io, .{ .sub_path = record_file, .data = buffer.written() });
    var random: [8]u8 = undefined;
    io.random(&random);
    var temp_buffer: [40]u8 = undefined;
    const temp = std.fmt.bufPrint(&temp_buffer, ".voice-{x}.json", .{&random}) catch unreachable;
    try dir.writeFile(io, .{ .sub_path = temp, .data = buffer.written() });
    std.Io.Dir.rename(dir, temp, dir, record_file, io) catch |err| {
        dir.deleteFile(io, temp) catch {};
        return err;
    };
}

fn sha256Hex(bytes: []const u8) [64]u8 {
    var digest: [32]u8 = undefined;
    std.crypto.hash.sha2.Sha256.hash(bytes, &digest, .{});
    return std.fmt.bytesToHex(digest, .lower);
}

fn base64(alloc: Allocator, bytes: []const u8) ![]u8 {
    const encoder = std.base64.standard.Encoder;
    const out = try alloc.alloc(u8, encoder.calcSize(bytes.len));
    _ = encoder.encode(out, bytes);
    return out;
}

fn unbase64(alloc: Allocator, text: []const u8) ![]u8 {
    const decoder = std.base64.standard.Decoder;
    const size = decoder.calcSizeForSlice(text) catch return error.InvalidVoiceRecord;
    const out = try alloc.alloc(u8, size);
    errdefer alloc.free(out);
    decoder.decode(out, text) catch return error.InvalidVoiceRecord;
    return out;
}

/// The keys of the checkpoint in `dir`: each file's SHA-256 from its pull's
/// sidecar, or, without one, from the file itself (seconds for the weights).
pub fn modelKeys(alloc: Allocator, io: std.Io, dir: []const u8) !Keys {
    var opened = try std.Io.Dir.cwd().openDir(io, dir, .{});
    defer opened.close(io);
    var out: [2][]const u8 = undefined;
    for (&out, [_][]const u8{ "model.safetensors", "speech_tokenizer/model.safetensors" }) |*key, file| {
        const sidecar_path = try model.sidecarPath(alloc, file);
        defer alloc.free(sidecar_path);
        if (model.readSidecar(alloc, io, opened, sidecar_path) catch null) |sidecar| {
            key.* = sidecar.sha256;
            continue;
        }
        key.* = try alloc.dupe(u8, &try fileSha256(io, opened, file));
    }
    return .{ .model = out[0], .codec = out[1] };
}

fn fileSha256(io: std.Io, dir: std.Io.Dir, path: []const u8) ![64]u8 {
    const file = try dir.openFile(io, path, .{});
    defer file.close(io);
    var hasher: std.crypto.hash.sha2.Sha256 = .init(.{});
    var buffer: [1 << 16]u8 = undefined;
    var reader = file.reader(io, &.{});
    while (true) {
        const n = try reader.interface.readSliceShort(&buffer);
        if (n == 0) break;
        hasher.update(buffer[0..n]);
    }
    return std.fmt.bytesToHex(hasher.finalResult(), .lower);
}

// ---- tests -----------------------------------------------------------------

test {
    // `encodersOf` has no caller here; this compiles it and the Voice methods it wraps.
    _ = &encodersOf;
}

/// A 24 kHz mono 16-bit WAV of `seconds` of a tone.
fn testClip(alloc: Allocator, seconds: f64) ![]u8 {
    const n: usize = @intFromFloat(seconds * sample_rate);
    const out = try alloc.alloc(u8, 44 + 2 * n);
    out[0..44].* = try speak.wavHeader(n, sample_rate);
    for (0..n) |i| {
        const x: f32 = @floatCast(0.3 * @sin(2 * std.math.pi * 220 * @as(f64, @floatFromInt(i)) / sample_rate));
        std.mem.writeInt(i16, out[44 + 2 * i ..][0..2], speak.pcm16(x), .little);
    }
    return out;
}

/// Encoders that count their calls: the embedding is the clip's mean and
/// length, the codes one frame per 1,920 samples.
const Fake = struct {
    embeds: usize = 0,
    codings: usize = 0,

    fn encoders(self: *Fake) Encoders {
        return .{ .context = self, .dim = 4, .embed = embed, .codes = codes };
    }
    fn embed(context: *anyopaque, _: Allocator, samples: []const f32, out: []f32) anyerror!void {
        const self: *Fake = @ptrCast(@alignCast(context));
        self.embeds += 1;
        var sum: f32 = 0;
        for (samples) |s| sum += s;
        out[0..4].* = .{ sum / @as(f32, @floatFromInt(samples.len)), @floatFromInt(samples.len), 1, -1 };
    }
    fn codes(context: *anyopaque, gpa: Allocator, samples: []const f32) anyerror![]Frame {
        const self: *Fake = @ptrCast(@alignCast(context));
        self.codings += 1;
        const frames = try gpa.alloc(Frame, (samples.len + 1919) / 1920);
        for (frames, 0..) |*f, t| f.* = @splat(@intCast(t));
        return frames;
    }
};

test "a voice is added once, listed, made ready per model, and removed" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(root);
    const path = try std.fs.path.join(gpa, &.{ root, "voices" });
    defer gpa.free(path);
    const store: Store = .{ .io = io, .path = path };
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const clip = try testClip(gpa, 2.5);
    defer gpa.free(clip);
    const added = try store.add(arena, "alice", clip, "  Hello there.\n", 1_800_000_000);
    try std.testing.expectEqualStrings("Hello there.", added.reference_text.?);
    try std.testing.expectEqual(@as(f64, 2.5), added.seconds);
    _ = try store.add(arena, "bob-2", clip, "   ", 1_800_000_000);
    const listed = try store.list(arena);
    try std.testing.expectEqual(@as(usize, 2), listed.len);
    try std.testing.expectEqualStrings("alice", listed[0].name);
    try std.testing.expect(listed[1].reference_text == null);
    try std.testing.expectEqualStrings("2027-01-15T08:00:00Z", listed[0].created);

    // The first use computes both and saves them; the second reads them.
    var fake: Fake = .{};
    const keys: Keys = .{ .model = &@as([64]u8, @splat('m')), .codec = &@as([64]u8, @splat('c')) };
    var ready = try store.ensure(gpa, "alice", keys, fake.encoders());
    defer ready.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 32), ready.frames.len);
    try std.testing.expectEqual(@as(f32, 60000), ready.embedding[1]);
    var again = try store.ensure(gpa, "alice", keys, fake.encoders());
    defer again.deinit(gpa);
    try std.testing.expectEqualSlices(f32, ready.embedding, again.embedding);
    try std.testing.expectEqualSlices(Frame, ready.frames, again.frames);
    try std.testing.expectEqual(@as(usize, 1), fake.embeds);
    try std.testing.expectEqual(@as(usize, 1), fake.codings);
    // Another model with the same codec: a new embedding, the codes kept.
    var other = try store.ensure(gpa, "alice", .{ .model = &@as([64]u8, @splat('n')), .codec = keys.codec }, fake.encoders());
    defer other.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 2), fake.embeds);
    try std.testing.expectEqual(@as(usize, 1), fake.codings);
    // Without a transcript, no codes are made (x-vector only).
    var plain = try store.ensure(gpa, "bob-2", keys, fake.encoders());
    defer plain.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 0), plain.frames.len);
    try std.testing.expectEqual(@as(usize, 1), fake.codings);

    try store.remove("bob-2");
    try std.testing.expectEqual(@as(usize, 1), (try store.list(arena)).len);
    try std.testing.expectError(error.VoiceNotFound, store.remove("bob-2"));
    try std.testing.expectError(error.VoiceNotFound, store.ensure(gpa, "bob-2", keys, fake.encoders()));
}

test "names, clips, transcripts, and the count are bounded; a changed clip is refused" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(root);
    const path = try std.fs.path.join(gpa, &.{ root, "voices" });
    defer gpa.free(path);
    const store: Store = .{ .io = io, .path = path, .limit = 2 };
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const clip = try testClip(gpa, 1.5);
    defer gpa.free(clip);

    for ([_][]const u8{ "", "Alice", "-a", "a b", "a/b", &@as([65]u8, @splat('x')) }) |name| try std.testing.expectError(error.InvalidVoiceName, store.add(arena, name, clip, null, 0));
    try std.testing.expectError(error.ReservedVoiceName, store.add(arena, "ryan", clip, null, 0));
    const short = try testClip(gpa, 0.5);
    defer gpa.free(short);
    try std.testing.expectError(error.ClipTooShort, store.add(arena, "a", short, null, 0));
    const long = try testClip(gpa, 30.5);
    defer gpa.free(long);
    try std.testing.expectError(error.ClipTooLong, store.add(arena, "a", long, null, 0));
    try std.testing.expectError(error.TranscriptTooLong, store.add(arena, "a", clip, &@as([max_transcript_bytes + 1]u8, @splat('x')), 0));
    try std.testing.expectError(error.InvalidTranscript, store.add(arena, "a", clip, "\xff\xfe", 0));
    try std.testing.expectError(error.UnsupportedAudioFormat, store.add(arena, "a", "not audio", null, 0));

    _ = try store.add(arena, "a", clip, "Words.", 0);
    try std.testing.expectError(error.VoiceExists, store.add(arena, "a", clip, null, 0));
    _ = try store.add(arena, "b", clip, null, 0);
    try std.testing.expectError(error.TooManyVoices, store.add(arena, "c", clip, null, 0));

    // A clip that is not the one recorded is not encoded.
    const wav_path = try std.fs.path.join(gpa, &.{ path, "a", clip_file });
    defer gpa.free(wav_path);
    const changed = try testClip(gpa, 1.25);
    defer gpa.free(changed);
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = wav_path, .data = changed });
    var fake: Fake = .{};
    try std.testing.expectError(error.InvalidVoiceRecord, store.ensure(gpa, "a", .{ .model = &@as([64]u8, @splat('m')), .codec = &@as([64]u8, @splat('c')) }, fake.encoders()));
    try std.testing.expectEqual(@as(usize, 0), fake.embeds);
}
