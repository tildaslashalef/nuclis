//! Speech and transcription models by name: a registry entry of the kind,
//! an audio catalogue name (`qwen3-tts-1.7b`, `nemotron-asr-0.6b`), or, for
//! the commands only, a checkpoint directory. Shared by `nuclis speak`,
//! `nuclis transcribe`, and `nuclis serve`, which opens no path a client
//! names. docs/models/catalogue.md § The catalogue.
const std = @import("std");
const config = @import("../config.zig");
const catalog = @import("../catalog.zig");
const inference = @import("inference");

pub const Kind = enum {
    speech,
    transcription,

    fn model(self: Kind) catalog.ModelKind {
        return switch (self) {
            .speech => .speech,
            .transcription => .transcription,
        };
    }
};

pub const Located = struct {
    /// The checkpoint directory.
    directory: []const u8,
    /// The catalogue's entry when the name is one.
    entry: ?*const catalog.AudioEntry = null,
    /// What the weights (and a speech model's codec) hold on disk, from the
    /// catalogue when it pins them.
    bytes: u64 = 0,
};

pub const Error = error{ WrongKind, ModelNotFound, ModelFileNotFound, MissingHome, InvalidRegistryEntry, OutOfMemory };

/// The directory `name` names for a model of `kind`, without touching the
/// disk. A model of another kind is `WrongKind`; with `paths` false (the
/// server), a name the registry and the catalogue do not know is
/// `ModelNotFound` instead of a directory.
pub fn resolve(arena: std.mem.Allocator, root: ?[]const u8, registry: config.Models, name: []const u8, kind: Kind, paths: bool, diag: *config.Diagnostic) Error!Located {
    if (registry.find(name)) |entry| {
        const entry_kind = entry.kind orelse .generation;
        if (entry_kind != kind.model()) {
            diag.set("{s} is a registry entry of a {s} model; `{s}` opens a {s} model (an entry with \"kind\": \"{s}\")", .{ name, catalog.kindName(entry_kind), catalog.command(kind.model()), @tagName(kind), @tagName(kind) });
            return error.WrongKind;
        }
        const located = if (entry.path) |p| p else if (entry.repo != null and entry.file != null) try std.fs.path.join(arena, &.{ entry.repo.?, entry.file.? }) else {
            diag.set("models.{s} locates nothing: give it path, or repo and file", .{name});
            return error.InvalidRegistryEntry;
        };
        const full = try models(arena, root, located);
        // A path to the weights names their directory.
        return .{ .directory = if (std.mem.endsWith(u8, full, ".safetensors")) std.fs.path.dirname(full) orelse full else full };
    }
    if (catalog.findAudio(name)) |entry| {
        if (entry.kind != kind.model()) {
            diag.set("{s} is a {s} model of the catalogue; `{s}` opens a {s} model ({s})", .{ name, catalog.kindName(entry.kind), catalog.command(kind.model()), @tagName(kind), defaultName(kind) });
            return error.WrongKind;
        }
        return .{ .directory = try catalog.audioDirectory(arena, try models(arena, root, ""), entry), .entry = entry, .bytes = entry.totalSize() };
    }
    const other = catalog.kindOf(.{}, name);
    if (other != .generation or catalog.find(name) != null) {
        diag.set("{s} is a {s} model of the catalogue; `{s}` opens a {s} model ({s})", .{ name, catalog.kindName(other), catalog.command(kind.model()), @tagName(kind), defaultName(kind) });
        return error.WrongKind;
    }
    if (!paths) {
        diag.set("{s} is not a {s} model of the registry or the catalogue (`GET /v1/models` lists them)", .{ name, @tagName(kind) });
        return error.ModelNotFound;
    }
    return .{ .directory = try models(arena, root, name) };
}

/// `resolve`, then the weights' presence: `ModelFileNotFound` names the
/// pull that fetches them. `bytes` is the files' size on disk.
pub fn locate(arena: std.mem.Allocator, io: std.Io, root: ?[]const u8, registry: config.Models, name: []const u8, kind: Kind, paths: bool, diag: *config.Diagnostic) Error!Located {
    var located = try resolve(arena, root, registry, name, kind, paths, diag);
    var bytes: u64 = 0;
    for (files(kind)) |file| {
        const path = try std.fs.path.join(arena, &.{ located.directory, file });
        const stat = std.Io.Dir.cwd().statFile(io, path, .{}) catch {
            if (located.entry) |e|
                diag.set("{s}: not pulled yet (`nuclis model pull {s}` fetches it, {d} MB)", .{ name, e.name, e.totalSize() / 1_000_000 })
            else
                diag.set("{s}: no {s} (a {s} checkpoint directory, a registry entry of kind {s}, or a catalogue name such as {s})", .{ located.directory, file, @tagName(kind), @tagName(kind), defaultName(kind) });
            return error.ModelFileNotFound;
        };
        bytes += stat.size;
    }
    located.bytes = bytes;
    return located;
}

/// The files a checkpoint of `kind` cannot open without, beside its
/// configuration and tokenizer.
fn files(kind: Kind) []const []const u8 {
    return switch (kind) {
        .speech => &.{ "model.safetensors", "speech_tokenizer/model.safetensors" },
        .transcription => &.{"model.safetensors"},
    };
}

fn defaultName(kind: Kind) []const u8 {
    return switch (kind) {
        .speech => "qwen3-tts-1.7b",
        .transcription => "nemotron-asr-0.6b",
    };
}

fn models(arena: std.mem.Allocator, root: ?[]const u8, path: []const u8) ![]const u8 {
    if (std.fs.path.isAbsolute(path) or std.mem.startsWith(u8, path, "./") or std.mem.startsWith(u8, path, "../")) return arena.dupe(u8, path);
    const dir = root orelse return error.MissingHome;
    return std.fs.path.join(arena, &.{ dir, "models", path });
}

/// What a speech checkpoint on disk says about itself, for a model the
/// catalogue does not describe. Strings live in `arena`.
pub const Described = struct {
    speakers: []const []const u8,
    instructions: bool,
    /// The weights' dtype (`BF16`), or `mixed`.
    quantization: []const u8,
};

/// Reads `directory`'s `config.json` (and `generation_config.json`) with the
/// parse `nuclis speak` runs and the weights' safetensors header. Null when
/// either is missing or not a speech checkpoint nuclis opens.
pub fn describeSpeech(arena: std.mem.Allocator, io: std.Io, directory: []const u8) !?Described {
    const max_config_bytes = 1024 * 1024;
    const cwd = std.Io.Dir.cwd();
    const config_json = cwd.readFileAlloc(io, try std.fs.path.join(arena, &.{ directory, "config.json" }), arena, .limited(max_config_bytes)) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return null,
    };
    const generation_json = cwd.readFileAlloc(io, try std.fs.path.join(arena, &.{ directory, "generation_config.json" }), arena, .limited(max_config_bytes)) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => null,
    };
    var parsed = inference.models.qwen3_tts.parseConfig(arena, config_json, generation_json) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return null,
    };
    defer parsed.deinit();
    const speakers = try arena.alloc([]const u8, parsed.speakers.len);
    for (speakers, parsed.speakers) |*name, s| name.* = try arena.dupe(u8, s.name);
    var weights = inference.safetensors.open(arena, io, try std.fs.path.join(arena, &.{ directory, "model.safetensors" }), .{}) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return null,
    };
    defer weights.deinit();
    var dtype: ?inference.safetensors.Dtype = null;
    for (weights.tensors) |t| {
        if (dtype != null and dtype.? != t.dtype) break;
        dtype = t.dtype;
    } else if (dtype) |d| return .{ .speakers = speakers, .instructions = parsed.takesInstruction(), .quantization = try std.ascii.allocUpperString(arena, @tagName(d)) };
    return .{ .speakers = speakers, .instructions = parsed.takesInstruction(), .quantization = "mixed" };
}

test "names resolve by kind; another kind's name is refused by name; the server takes no path" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diag: config.Diagnostic = .{};
    const voice = try resolve(arena, "/r", .{}, "qwen3-tts-1.7b", .speech, false, &diag);
    try std.testing.expectEqualStrings("/r/models/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", voice.directory);
    try std.testing.expect(voice.entry.?.instructions);
    const ears = try resolve(arena, "/r", .{}, "nemotron-asr-0.6b", .transcription, false, &diag);
    try std.testing.expectEqualStrings("/r/models/nvidia/nemotron-3.5-asr-streaming-0.6b", ears.directory);
    try std.testing.expectError(error.WrongKind, resolve(arena, "/r", .{}, "nemotron-asr-0.6b", .speech, true, &diag));
    try std.testing.expect(std.mem.indexOf(u8, diag.message(), "transcription model") != null);
    try std.testing.expectError(error.WrongKind, resolve(arena, "/r", .{}, "qwen3.8-27b", .speech, true, &diag));
    try std.testing.expectError(error.WrongKind, resolve(arena, "/r", .{}, "laya", .transcription, true, &diag));
    try std.testing.expectError(error.WrongKind, resolve(arena, "/r", .{}, "embeddinggemma-2", .speech, true, &diag));
    try std.testing.expectError(error.ModelNotFound, resolve(arena, "/r", .{}, "/etc/voice", .speech, false, &diag));
    try std.testing.expectEqualStrings("/etc/voice", (try resolve(arena, "/r", .{}, "/etc/voice", .speech, true, &diag)).directory);
    try std.testing.expectEqualStrings("/r/models/Qwen/x", (try resolve(arena, "/r", .{}, "Qwen/x", .speech, true, &diag)).directory);
    const entries = [_]config.NamedModel{
        .{ .name = "mine", .entry = .{ .kind = .speech, .repo = "me/voice", .file = "model.safetensors" } },
        .{ .name = "chat", .entry = .{ .path = "chat.gguf" } },
    };
    const registry: config.Models = .{ .entries = &entries };
    try std.testing.expectEqualStrings("/r/models/me/voice", (try resolve(arena, "/r", registry, "mine", .speech, false, &diag)).directory);
    try std.testing.expectError(error.WrongKind, resolve(arena, "/r", registry, "mine", .transcription, false, &diag));
    try std.testing.expectError(error.WrongKind, resolve(arena, "/r", registry, "chat", .speech, false, &diag));
}

test "a model not pulled names the pull that fetches it" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diag: config.Diagnostic = .{};
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(std.testing.io, ".", arena);
    try std.testing.expectError(error.ModelFileNotFound, locate(arena, std.testing.io, root, .{}, "qwen3-tts-0.6b", .speech, false, &diag));
    try std.testing.expect(std.mem.indexOf(u8, diag.message(), "nuclis model pull qwen3-tts-0.6b") != null);
    try tmp.dir.createDirPath(std.testing.io, "models/nvidia/nemotron-3.5-asr-streaming-0.6b");
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "models/nvidia/nemotron-3.5-asr-streaming-0.6b/model.safetensors", .data = "abcd" });
    try std.testing.expectEqual(@as(u64, 4), (try locate(arena, std.testing.io, root, .{}, "nemotron-asr-0.6b", .transcription, false, &diag)).bytes);
}

test "a pulled speech checkpoint describes its speakers, instructions, and dtype" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const dir = try tmp.dir.realPathFileAlloc(io, ".", arena);
    try std.testing.expectEqual(@as(?Described, null), try describeSpeech(arena, io, dir));
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = try inference.models.qwen3_tts.testConfigJson(arena, "1b7", 16, "custom_voice") });
    const header = "{\"a\":{\"dtype\":\"BF16\",\"shape\":[2],\"data_offsets\":[0,4]},\"b\":{\"dtype\":\"BF16\",\"shape\":[1],\"data_offsets\":[4,6]}}";
    var bytes: [8 + header.len + 6]u8 = @splat(0);
    std.mem.writeInt(u64, bytes[0..8], header.len, .little);
    @memcpy(bytes[8..][0..header.len], header);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = &bytes });
    const d = (try describeSpeech(arena, io, dir)).?;
    try std.testing.expectEqual(@as(usize, 2), d.speakers.len);
    try std.testing.expect(d.instructions);
    try std.testing.expectEqualStrings("BF16", d.quantization);
    try tmp.dir.writeFile(io, .{ .sub_path = "config.json", .data = "{\"model_type\":\"bert\"}" });
    try std.testing.expectEqual(@as(?Described, null), try describeSpeech(arena, io, dir));
}
