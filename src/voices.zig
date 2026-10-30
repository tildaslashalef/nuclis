//! Saved voices for the cloning speech models (Qwen3-TTS Base, Drama 3): a
//! reference clip, and optionally its transcript, registered once under a
//! name and used by every request that names it. Each lives in
//! `<root>/voices/<name>/`: `reference.wav`, the clip decoded to 24 kHz mono
//! 16-bit, `reference-44k.wav`, the same at 44.1 kHz (Drama 3's codec rate;
//! absent from voices saved before it), and `voice.json`, its record and
//! what the encoders made of it, keyed by the file that made it (reference
//! codes by the codec file's SHA-256, a speaker embedding by the model
//! file's). A key the model in use lacks is computed from the clip and
//! written back (`ensure`), so one voice serves every cloning model. A voice appears whole
//! (a directory renamed into place) and its record is replaced by rename.
//! Bounds are host constants. `run` is `nuclis voice`. docs/guide/api.md
//! § Voices.
const std = @import("std");
const inference = @import("inference");
const catalog = @import("catalog.zig");
const config = @import("config.zig");
const model = @import("model.zig");
const paths = @import("paths.zig");
const speak = @import("speak.zig");
const style = @import("tui/style.zig");

const Allocator = std.mem.Allocator;
const audio = inference.audio.audio;
const speech = inference.models.qwen3_tts_runtime;
const drama3 = inference.models.drama3_speech;
const fish = inference.audio.fish_codec;
pub const Frame = inference.audio.qwen3_codec.Frame;

pub const sample_rate = 24000;
/// The rate of the wide copy, which Drama 3's codec encodes.
pub const wide_rate = 44100;
/// A clip's length after decoding: 1 to 30 s.
pub const min_samples = 1 * sample_rate;
pub const max_samples = 30 * sample_rate;
pub const max_transcript_bytes = 2048;
pub const max_voices = 256;
pub const max_name = 64;
const record_file = "voice.json";
const clip_file = "reference.wav";
const wide_clip_file = "reference-44k.wav";
/// A record holds a few codes and embeddings, each tens of KB as base64.
const max_record_bytes = 1 << 20;
const max_clip_bytes = 44 + 2 * max_samples;
const max_wide_samples = 30 * wide_rate;
const max_wide_clip_bytes = 44 + 2 * max_wide_samples;

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
    /// The model clones only from a clip with its transcript (Drama 3).
    TranscriptRequired,
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
    /// Of `reference-44k.wav`, when the voice has one.
    wide_sha256: ?[]const u8 = null,
    samples: u64,
    /// Unix seconds, as OpenAI's voice object's `created_at`.
    created_at: i64,
    /// Codec file SHA-256 → base64 of little-endian u16 `[frames][codebooks]`.
    codes: std.json.ArrayHashMap([]const u8) = .{},
    /// Model file SHA-256 → base64 of little-endian f32 `[width]`.
    embeddings: std.json.ArrayHashMap([]const u8) = .{},
};

/// A voice as listed.
pub const Summary = struct {
    name: []const u8,
    reference_text: ?[]const u8,
    seconds: f64,
    /// Unix seconds.
    created_at: i64,
};

/// The files a voice's derived data is keyed by: the SHA-256 (lowercase
/// hex) of the model's `model.safetensors` and of its codec's.
pub const Keys = struct { model: []const u8, codec: []const u8 };

/// What makes a voice's data, from mono samples at `rate`: the speaker
/// embedding (`dim` floats; none when 0) and the reference codes, frame by
/// frame, `sizes.len` codebooks of `sizes[k]` codes each (owned by `gpa`).
pub const Encoders = struct {
    context: *anyopaque,
    rate: u32 = sample_rate,
    sizes: []const u16 = &qwen3_sizes,
    dim: usize,
    embed: *const fn (context: *anyopaque, gpa: Allocator, samples: []const f32, out: []f32) anyerror!void,
    codes: *const fn (context: *anyopaque, gpa: Allocator, samples: []const f32) anyerror![]u16,
};

const qwen3_sizes: [16]u16 = @splat(inference.models.qwen3_tts.codebook_size);
const drama3_sizes = blk: {
    var sizes: [fish.codebooks]u16 = @splat(fish.residual_size);
    sizes[0] = fish.semantic_size;
    break :blk sizes;
};

fn noEmbedding(_: *anyopaque, _: Allocator, _: []const f32, _: []f32) anyerror!void {
    unreachable;
}

/// An open Drama 3's encoder, which needs the transcript and makes no
/// embedding.
pub fn drama3Encoders(voice: *drama3.Voice) Encoders {
    const wrap = struct {
        fn codes(context: *anyopaque, gpa: Allocator, samples: []const f32) anyerror![]u16 {
            const v: *drama3.Voice = @ptrCast(@alignCast(context));
            return flat(try v.referenceCodes(gpa, samples));
        }
    };
    return .{ .context = voice, .rate = wide_rate, .sizes = &drama3_sizes, .dim = 0, .embed = noEmbedding, .codes = wrap.codes };
}

/// `frames` as one slice of codes, the same allocation.
fn flat(frames: anytype) []u16 {
    const Row = @typeInfo(@TypeOf(frames)).pointer.child;
    const per = @typeInfo(Row).array.len;
    return @as([*]u16, @ptrCast(frames.ptr))[0 .. frames.len * per];
}

/// An open Base's encoders.
pub fn encodersOf(voice: *speech.Voice) Encoders {
    const wrap = struct {
        fn embed(context: *anyopaque, gpa: Allocator, samples: []const f32, out: []f32) anyerror!void {
            const v: *speech.Voice = @ptrCast(@alignCast(context));
            try v.speakerEmbedding(gpa, samples, out);
        }
        fn codes(context: *anyopaque, gpa: Allocator, samples: []const f32) anyerror![]u16 {
            const v: *speech.Voice = @ptrCast(@alignCast(context));
            return flat(try v.referenceCodes(gpa, samples));
        }
    };
    return .{ .context = voice, .dim = voice.config.talker.hidden, .embed = wrap.embed, .codes = wrap.codes };
}

/// Encoders over a Base checkpoint's directory or a Drama 3 GGUF, mapped on
/// their first call (`speech.Encoding`, `Drama3Encoding`), so a voice
/// already made for the model maps nothing: the server's way, beside the
/// voice its worker holds. Not thread-safe.
pub const FileEncoders = struct {
    gpa: Allocator,
    io: std.Io,
    dir: []const u8,
    dim: usize,
    encoding: ?*speech.Encoding = null,
    drama3: ?*Drama3Encoding = null,

    /// Reads `dir`'s `config.json` only (nothing for a `.gguf`, Drama 3);
    /// `NotABase` when it clones nothing. `dir` is borrowed.
    pub fn init(gpa: Allocator, io: std.Io, dir: []const u8) !FileEncoders {
        if (std.mem.endsWith(u8, dir, ".gguf")) return .{ .gpa = gpa, .io = io, .dir = dir, .dim = 0 };
        var arena: std.heap.ArenaAllocator = .init(gpa);
        defer arena.deinit();
        const a = arena.allocator();
        const bytes = try std.Io.Dir.cwd().readFileAlloc(io, try std.fs.path.join(a, &.{ dir, "config.json" }), a, .limited(1 << 20));
        var parsed = try inference.models.qwen3_tts.parseConfig(gpa, bytes, null);
        defer parsed.deinit();
        if (!parsed.clones()) return error.NotABase;
        return .{ .gpa = gpa, .io = io, .dir = dir, .dim = parsed.talker.hidden };
    }

    pub fn deinit(self: *FileEncoders) void {
        if (self.encoding) |e| e.close();
        if (self.drama3) |e| e.close();
        self.* = undefined;
    }

    pub fn encoders(self: *FileEncoders) Encoders {
        if (std.mem.endsWith(u8, self.dir, ".gguf")) {
            const wrap = struct {
                fn codes(context: *anyopaque, gpa: Allocator, samples: []const f32) anyerror![]u16 {
                    const f: *FileEncoders = @ptrCast(@alignCast(context));
                    if (f.drama3 == null) f.drama3 = try Drama3Encoding.open(f.gpa, f.io, f.dir);
                    return flat(try f.drama3.?.encoder.encode(gpa, samples));
                }
            };
            return .{ .context = self, .rate = wide_rate, .sizes = &drama3_sizes, .dim = 0, .embed = noEmbedding, .codes = wrap.codes };
        }
        const wrap = struct {
            fn opened(context: *anyopaque) !*speech.Encoding {
                const f: *FileEncoders = @ptrCast(@alignCast(context));
                if (f.encoding == null) f.encoding = try speech.Encoding.open(f.gpa, f.io, f.dir);
                return f.encoding.?;
            }
            fn embed(context: *anyopaque, gpa: Allocator, samples: []const f32, out: []f32) anyerror!void {
                try (try opened(context)).speakerEmbedding(gpa, samples, out);
            }
            fn codes(context: *anyopaque, gpa: Allocator, samples: []const f32) anyerror![]u16 {
                return flat(try (try opened(context)).referenceCodes(gpa, samples));
            }
        };
        return .{ .context = self, .dim = self.dim, .embed = wrap.embed, .codes = wrap.codes };
    }
};

/// Drama 3's encoder alone over its GGUF: the mapping and the Fish codec's
/// encoder, without the backbone.
pub const Drama3Encoding = struct {
    gpa: Allocator,
    io: std.Io,
    mapped: inference.weights.Mapped,
    encoder: inference.audio.fish_codec_encoder.Encoder,

    pub fn open(gpa: Allocator, io: std.Io, path: []const u8) !*Drama3Encoding {
        const self = try gpa.create(Drama3Encoding);
        errdefer gpa.destroy(self);
        self.gpa = gpa;
        self.io = io;
        self.mapped = try .open(gpa, io, path);
        errdefer self.mapped.deinit(io);
        self.encoder = try .init(gpa, io, self.mapped.document, self.mapped.view());
        return self;
    }

    pub fn close(self: *Drama3Encoding) void {
        self.encoder.deinit();
        self.mapped.deinit(self.io);
        self.gpa.destroy(self);
    }
};

/// A voice ready for a prompt, owned (`deinit`).
pub const Ready = struct {
    name: []u8,
    reference_text: ?[]u8,
    /// Empty for a model that makes none (Drama 3).
    embedding: []f32,
    /// The reference codes, `codebooks` a frame; empty without a transcript
    /// (x-vector only).
    codes: []u16,
    codebooks: usize,

    pub fn clone(self: *const Ready) inference.models.qwen3_tts.Clone {
        return .{ .embedding = self.embedding, .frames = self.rows(Frame) };
    }

    /// The codes as frames of `Row` (an array of `codebooks` codes).
    pub fn rows(self: *const Ready, comptime Row: type) []const Row {
        std.debug.assert(@typeInfo(Row).array.len == self.codebooks);
        return @as([*]const Row, @ptrCast(self.codes.ptr))[0 .. self.codes.len / self.codebooks];
    }

    pub fn deinit(self: *Ready, alloc: Allocator) void {
        alloc.free(self.name);
        if (self.reference_text) |t| alloc.free(t);
        alloc.free(self.embedding);
        alloc.free(self.codes);
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

        const wav = try wavBytes(alloc, pcm.samples, sample_rate);
        defer alloc.free(wav);
        // The wide copy needs the platform's decoder unless the upload is a
        // 44.1 kHz WAV; without it, Drama 3 encodes the 24 kHz clip.
        var wide: ?audio.Pcm = audio.decodeAt(alloc, bytes, max_wide_samples, wide_rate) catch |err| switch (err) {
            error.OutOfMemory => return err,
            else => null,
        };
        defer if (wide) |*w| w.deinit(alloc);
        const wide_wav: ?[]u8 = if (wide) |w| try wavBytes(alloc, w.samples, wide_rate) else null;
        defer if (wide_wav) |w| alloc.free(w);
        const record: Record = .{
            .name = name,
            .reference_text = text,
            .reference_sha256 = &sha256Hex(wav),
            .wide_sha256 = if (wide_wav) |w| &sha256Hex(w) else null,
            .samples = pcm.samples.len,
            .created_at = now_seconds,
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
        if (wide_wav) |w| try dir.writeFile(self.io, .{ .sub_path = wide_clip_file, .data = w });
        try writeRecord(alloc, self.io, dir, record, false);
        std.Io.Dir.renamePreserve(root, temp, root, name, self.io) catch |err| return switch (err) {
            error.PathAlreadyExists => error.VoiceExists,
            else => err,
        };
        return .{
            .name = try alloc.dupe(u8, name),
            .reference_text = if (text) |t| try alloc.dupe(u8, t) else null,
            .seconds = @as(f64, @floatFromInt(pcm.samples.len)) / sample_rate,
            .created_at = now_seconds,
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
                .created_at = r.created_at,
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

    /// `name` ready for the model `keys` names: its embedding (if the model
    /// makes one) and (with a transcript) its reference codes, each computed
    /// by `encoders` from the clip at their rate and saved when the record
    /// lacks it.
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
        const embeds = encoders.dim > 0;
        if (!embeds and !icl) return error.TranscriptRequired;
        const arena = parsed.arena.allocator();

        const have_embedding = if (embeds) record.embeddings.map.get(keys.model) else null;
        const have_codes = if (icl) record.codes.map.get(keys.codec) else null;
        if ((embeds and have_embedding == null) or (icl and have_codes == null)) {
            var pcm = try readClip(alloc, self.io, dir, record.*, encoders.rate);
            defer pcm.deinit(alloc);
            if (embeds and have_embedding == null) {
                const embedding = try alloc.alloc(f32, encoders.dim);
                defer alloc.free(embedding);
                try encoders.embed(encoders.context, alloc, pcm.samples, embedding);
                try record.embeddings.map.put(arena, try arena.dupe(u8, keys.model), try base64(arena, std.mem.sliceAsBytes(embedding)));
            }
            if (icl and have_codes == null) {
                const codes = try encoders.codes(encoders.context, alloc, pcm.samples);
                defer alloc.free(codes);
                try record.codes.map.put(arena, try arena.dupe(u8, keys.codec), try base64(arena, std.mem.sliceAsBytes(codes)));
            }
            try writeRecord(alloc, self.io, dir, record.*, true);
        }

        const embedding = try alloc.alloc(f32, encoders.dim);
        errdefer alloc.free(embedding);
        if (embeds) {
            const embedding_bytes = try unbase64(alloc, record.embeddings.map.get(keys.model).?);
            defer alloc.free(embedding_bytes);
            if (embedding_bytes.len != encoders.dim * 4) return error.InvalidVoiceRecord;
            for (embedding, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, embedding_bytes[i * 4 ..][0..4], .little));
        }
        const books = encoders.sizes.len;
        var codes: []u16 = &.{};
        errdefer alloc.free(codes);
        if (icl) {
            const code_bytes = try unbase64(alloc, record.codes.map.get(keys.codec).?);
            defer alloc.free(code_bytes);
            if (code_bytes.len == 0 or code_bytes.len % (2 * books) != 0) return error.InvalidVoiceRecord;
            codes = try alloc.alloc(u16, code_bytes.len / 2);
            for (codes, 0..) |*code, i| {
                code.* = std.mem.readInt(u16, code_bytes[i * 2 ..][0..2], .little);
                if (code.* >= encoders.sizes[i % books]) return error.InvalidVoiceRecord;
            }
        }
        const owned_name = try alloc.dupe(u8, name);
        errdefer alloc.free(owned_name);
        return .{
            .name = owned_name,
            .reference_text = if (record.reference_text) |t| try alloc.dupe(u8, t) else null,
            .embedding = embedding,
            .codes = codes,
            .codebooks = books,
        };
    }
};

/// The voice's clip at `rate`: the saved one at its rate, else the 24 kHz
/// clip resampled. Each file must be the one the record names.
fn readClip(alloc: Allocator, io: std.Io, dir: std.Io.Dir, record: Record, rate: u32) !audio.Pcm {
    const wide = rate == wide_rate and record.wide_sha256 != null;
    const file = if (wide) wide_clip_file else clip_file;
    const wav = dir.readFileAlloc(io, file, alloc, .limited(if (wide) max_wide_clip_bytes else max_clip_bytes)) catch return error.InvalidVoiceRecord;
    defer alloc.free(wav);
    if (!std.mem.eql(u8, &sha256Hex(wav), if (wide) record.wide_sha256.? else record.reference_sha256)) return error.InvalidVoiceRecord;
    if (wide or rate == sample_rate) return audio.decodeWavAt(alloc, wav, if (wide) max_wide_samples else max_samples, rate) catch error.InvalidVoiceRecord;
    return audio.decodeAt(alloc, wav, @intCast(@as(u64, max_samples) * rate / sample_rate), rate);
}

/// A mono 16-bit WAV of `samples` at `rate`.
fn wavBytes(alloc: Allocator, samples: []const f32, rate: u32) ![]u8 {
    const wav = try alloc.alloc(u8, 44 + 2 * samples.len);
    wav[0..44].* = try speak.wavHeader(samples.len, rate);
    for (samples, 0..) |x, i| std.mem.writeInt(i16, wav[44 + 2 * i ..][0..2], speak.pcm16(x), .little);
    return wav;
}

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
/// A `.gguf` (Drama 3) holds model and codec both: one key for the two.
pub fn modelKeys(alloc: Allocator, io: std.Io, dir: []const u8) !Keys {
    if (std.mem.endsWith(u8, dir, ".gguf")) {
        var parent = try std.Io.Dir.cwd().openDir(io, std.fs.path.dirname(dir) orelse ".", .{});
        defer parent.close(io);
        const file = std.fs.path.basename(dir);
        const sidecar_path = try model.sidecarPath(alloc, file);
        defer alloc.free(sidecar_path);
        const key = if (model.readSidecar(alloc, io, parent, sidecar_path) catch null) |sidecar| sidecar.sha256 else try alloc.dupe(u8, &try fileSha256(io, parent, file));
        return .{ .model = key, .codec = key };
    }
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

// ---- the command -----------------------------------------------------------

pub const Command = union(enum) {
    add: struct { name: []const u8, audio: []const u8, text: ?[]const u8 },
    list,
    rm: []const u8,
};

pub const Options = struct { command: Command, json: bool = false };

/// Parses the words after `voice`.
pub fn parseArgs(args: []const []const u8, diag: *config.Diagnostic) !Options {
    const usage = "`nuclis voice add <name> <audio> [--text <transcript>]`, `voice list`, or `voice rm <name>`";
    if (args.len == 0) {
        diag.set("give an action: {s}", .{usage});
        return error.MissingAction;
    }
    var json = false;
    var text: ?[]const u8 = null;
    var positionals: [2][]const u8 = undefined;
    var count: usize = 0;
    var i: usize = 1;
    while (i < args.len) : (i += 1) {
        const word = args[i];
        if (std.mem.eql(u8, word, "--json")) {
            json = true;
        } else if (std.mem.eql(u8, word, "--text")) {
            i += 1;
            if (i == args.len) {
                diag.set("--text needs the clip's transcript", .{});
                return error.MissingOptionValue;
            }
            if (text != null) return error.DuplicateOption;
            text = args[i];
        } else if (std.mem.startsWith(u8, word, "-") and word.len > 1) {
            diag.set("{s} is not an option of voice (`nuclis voice --help`)", .{word});
            return error.UnknownOption;
        } else {
            if (count == positionals.len) {
                diag.set("{s}: too many arguments; {s}", .{ word, usage });
                return error.UnexpectedArgument;
            }
            positionals[count] = word;
            count += 1;
        }
    }
    const action = args[0];
    const want: usize = if (std.mem.eql(u8, action, "add")) 2 else if (std.mem.eql(u8, action, "rm")) 1 else if (std.mem.eql(u8, action, "list")) 0 else {
        diag.set("{s} is not an action of voice: {s}", .{ action, usage });
        return error.UnknownAction;
    };
    if (count != want or (text != null and want != 2)) {
        diag.set("{s}", .{usage});
        return error.InvalidArguments;
    }
    const command: Command = if (want == 2) .{ .add = .{ .name = positionals[0], .audio = positionals[1], .text = text } } else if (want == 1) .{ .rm = positionals[0] } else .list;
    return .{ .command = command, .json = json };
}

/// `nuclis voice`: saves a clip under a name, lists the saved voices, or
/// removes one. No model is opened: a voice is encoded on its first use.
pub fn run(alloc: Allocator, io: std.Io, root: ?[]const u8, words: []const []const u8, out: *std.Io.Writer, sty: style.Style, diag: *config.Diagnostic) !void {
    const options = try parseArgs(words, diag);
    const home = root orelse return error.MissingHome;
    var arena_state = std.heap.ArenaAllocator.init(alloc);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const store: Store = .{ .io = io, .path = try paths.voicesPath(arena, home) };
    switch (options.command) {
        .add => |a| {
            const bytes = std.Io.Dir.cwd().readFileAlloc(io, a.audio, arena, .limited(audio.max_bytes)) catch |err| {
                diag.set("{s}: {s}", .{ a.audio, if (err == error.StreamTooLong) "larger than 64 MiB" else @errorName(err) });
                return err;
            };
            const now = std.Io.Timestamp.now(io, .real).toSeconds();
            const added = store.add(arena, a.name, bytes, a.text, now) catch |err| {
                explain(err, a.name, a.audio, diag);
                return err;
            };
            if (options.json) return writeJson(out, added);
            try out.print("saved {s}{s}{s}: {d:.1} s", .{ sty.on(.header), added.name, sty.off(), added.seconds });
            if (added.reference_text) |t| try out.print(", transcript \"{s}\"", .{t}) else try out.writeAll(", no transcript (the speaker embedding alone)");
            try out.print("\nit is encoded on its first use: `nuclis speak --model qwen3-tts-1.7b-base --voice {s} \"…\" -o out.wav`\n", .{added.name});
        },
        .list => {
            const listed = try store.list(arena);
            if (options.json) return writeJson(out, listed);
            if (listed.len == 0) return out.writeAll("no saved voices (`nuclis voice add <name> <audio> --text <transcript>` saves one)\n");
            for (listed) |v| {
                var stamp: [20]u8 = undefined;
                try out.print("{s}{s:<20}{s} {d:>5.1} s  {s}  ", .{ sty.on(.header), v.name, sty.off(), v.seconds, model.rfc3339(&stamp, v.created_at) });
                if (v.reference_text) |t| try out.print("\"{s}\"\n", .{t}) else try out.writeAll("(no transcript)\n");
            }
        },
        .rm => |name| {
            store.remove(name) catch |err| {
                explain(err, name, "", diag);
                return err;
            };
            if (options.json) return out.print("{{\"removed\":\"{s}\"}}\n", .{name});
            try out.print("removed {s}\n", .{name});
        },
    }
}

fn writeJson(out: *std.Io.Writer, value: anytype) !void {
    var w: std.json.Stringify = .{ .writer = out, .options = .{ .whitespace = .indent_2 } };
    try w.write(value);
    try out.writeByte('\n');
}

/// The message for a store error about voice `name` (and clip `path`).
pub fn explain(err: anyerror, name: []const u8, path: []const u8, diag: *config.Diagnostic) void {
    switch (err) {
        error.InvalidVoiceName => diag.set("{s}: a voice's name is 1 to 64 of a-z, 0-9, _ and -, starting with a letter or digit", .{name}),
        error.ReservedVoiceName => diag.set("{s} is a built-in speaker of the CustomVoice models; give the voice another name", .{name}),
        error.VoiceExists => diag.set("{s} is saved already (`nuclis voice rm {s}` removes it)", .{ name, name }),
        error.VoiceNotFound => diag.set("no saved voice named {s} (`nuclis voice list`)", .{name}),
        error.TooManyVoices => diag.set("{d} voices are saved, the most kept; remove one first", .{max_voices}),
        error.ClipTooShort => diag.set("{s}: under 1 s of audio; a voice needs 1 to 30 s", .{path}),
        error.ClipTooLong => diag.set("{s}: over 30 s of audio; a voice needs 1 to 30 s (a few sentences clone best)", .{path}),
        error.TranscriptTooLong => diag.set("the transcript is over {d} bytes", .{max_transcript_bytes}),
        error.InvalidTranscript => diag.set("the transcript is not UTF-8", .{}),
        error.UnsupportedAudioFormat => diag.set("{s}: not audio nuclis reads ({s})", .{ path, audio.readable }),
        error.AudioTooLarge => diag.set("{s}: larger than 64 MiB", .{path}),
        error.InvalidVoiceRecord => diag.set("{s}: its voice.json or reference.wav is damaged (`nuclis voice rm {s}`, then add it again)", .{ name, name }),
        else => {},
    }
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
    _ = &drama3Encoders;
    _ = &FileEncoders.encoders;
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
    fn codes(context: *anyopaque, gpa: Allocator, samples: []const f32) anyerror![]u16 {
        const self: *Fake = @ptrCast(@alignCast(context));
        self.codings += 1;
        const frames = try gpa.alloc(Frame, (samples.len + 1919) / 1920);
        for (frames, 0..) |*f, t| f.* = @splat(@intCast(t));
        return flat(frames);
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
    try std.testing.expectEqual(@as(i64, 1_800_000_000), listed[0].created_at);

    // The first use computes both and saves them; the second reads them.
    var fake: Fake = .{};
    const keys: Keys = .{ .model = &@as([64]u8, @splat('m')), .codec = &@as([64]u8, @splat('c')) };
    var ready = try store.ensure(gpa, "alice", keys, fake.encoders());
    defer ready.deinit(gpa);
    try std.testing.expectEqual(@as(usize, 32), ready.rows(Frame).len);
    try std.testing.expectEqual(@as(f32, 60000), ready.embedding[1]);
    var again = try store.ensure(gpa, "alice", keys, fake.encoders());
    defer again.deinit(gpa);
    try std.testing.expectEqualSlices(f32, ready.embedding, again.embedding);
    try std.testing.expectEqualSlices(u16, ready.codes, again.codes);
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
    try std.testing.expectEqual(@as(usize, 0), plain.codes.len);
    try std.testing.expectEqual(@as(usize, 1), fake.codings);

    try store.remove("bob-2");
    try std.testing.expectEqual(@as(usize, 1), (try store.list(arena)).len);
    try std.testing.expectError(error.VoiceNotFound, store.remove("bob-2"));
    try std.testing.expectError(error.VoiceNotFound, store.ensure(gpa, "bob-2", keys, fake.encoders()));
}

test "a model without an embedding needs the transcript and encodes the 44.1 kHz copy" {
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
    const clip = try testClip(gpa, 2);
    defer gpa.free(clip);
    _ = try store.add(arena, "carol", clip, "Hello.", 0);
    _ = try store.add(arena, "dave", clip, null, 0);

    const Wide = struct {
        rate: usize = 0,
        fn codes(context: *anyopaque, a: Allocator, samples: []const f32) anyerror![]u16 {
            const self: *@This() = @ptrCast(@alignCast(context));
            self.rate = samples.len / 2;
            const out = try a.alloc(u16, 3 * fish.codebooks);
            for (out, 0..) |*c, i| c.* = @intCast(i % fish.codebooks);
            return out;
        }
    };
    var wide: Wide = .{};
    const encoders: Encoders = .{ .context = &wide, .rate = wide_rate, .sizes = &drama3_sizes, .dim = 0, .embed = noEmbedding, .codes = Wide.codes };
    const keys: Keys = .{ .model = &@as([64]u8, @splat('g')), .codec = &@as([64]u8, @splat('g')) };
    try std.testing.expectError(error.TranscriptRequired, store.ensure(gpa, "dave", keys, encoders));
    var ready = try store.ensure(gpa, "carol", keys, encoders);
    defer ready.deinit(gpa);
    try std.testing.expectEqual(@as(usize, wide_rate), wide.rate);
    try std.testing.expectEqual(@as(usize, 0), ready.embedding.len);
    try std.testing.expectEqual(@as(usize, 3), ready.rows([fish.codebooks]u16).len);
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
