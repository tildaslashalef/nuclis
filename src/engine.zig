//! Executable-side view of the engine seam. The types live in the library
//! (`inference.engine`, so every consumer shares one
//! API); this module re-exports them and keeps the two helpers that belong
//! to a command line: prompt sources and millisecond formatting.
const std = @import("std");
const inference = @import("inference");

pub const Backend = inference.engine.Backend;
pub const StopReason = inference.engine.StopReason;
pub const Model = inference.engine.Model;
pub const Engine = inference.engine.Engine;

/// Reads the prompt text from `--prompt` or `--prompt-file`. Inline prompts are
/// limited to 64 KiB, files to 4 MiB; neither is ever truncated. Caller owns
/// the result.
pub fn readPrompt(alloc: std.mem.Allocator, io: std.Io, inline_prompt: ?[]const u8, file_path: ?[]const u8) ![]u8 {
    if (inline_prompt != null and file_path != null) return error.ConflictingPromptSources;
    if (inline_prompt) |text| {
        if (text.len > 64 * 1024) return error.LimitExceeded;
        return alloc.dupe(u8, text);
    }
    const path = file_path orelse return error.MissingPrompt;
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.size == 0) return error.EmptyPrompt;
    if (stat.size > 4 * 1024 * 1024) return error.LimitExceeded;
    const text = try alloc.alloc(u8, @intCast(stat.size));
    errdefer alloc.free(text);
    var buffer: [16 * 1024]u8 = undefined;
    var reader = file.reader(io, &buffer);
    try reader.interface.readSliceAll(text);
    if (!std.unicode.utf8ValidateSlice(text)) return error.InvalidUtf8;
    return text;
}

/// Reads a prompt given as token IDs: a JSON array of integers below
/// `vocabulary_size`, at most 4 MiB, at least one token. This is how a
/// benchmark reproduces a reference run's exact input when that input was a
/// token array rather than text (the reference harness concatenates token
/// sequences, which the tokenizer would not always produce from the joined
/// text). Caller owns the result.
pub fn readPromptTokens(alloc: std.mem.Allocator, io: std.Io, path: []const u8, vocabulary_size: usize) ![]u32 {
    const text = try readPrompt(alloc, io, null, path);
    defer alloc.free(text);
    const parsed = std.json.parseFromSlice([]u32, alloc, text, .{}) catch return error.InvalidPromptTokens;
    defer parsed.deinit();
    if (parsed.value.len == 0) return error.EmptyPrompt;
    for (parsed.value) |id| if (id >= vocabulary_size) return error.InvalidPromptTokens;
    return alloc.dupe(u32, parsed.value);
}

/// The companion draft file's path: `mtp` (a name relative to the model
/// file's directory, or an absolute path) resolved against `model_path`'s
/// directory. Null when the entry names none; caller owns the result.
pub fn draftPath(alloc: std.mem.Allocator, model_path: []const u8, mtp: ?[]const u8) !?[]u8 {
    const file = mtp orelse return null;
    if (std.fs.path.isAbsolute(file)) return try alloc.dupe(u8, file);
    const parent = std.fs.path.dirname(model_path) orelse return error.MissingModelDirectory;
    return try std.fs.path.join(alloc, &.{ parent, file });
}

pub fn milliseconds(duration: std.Io.Duration) f64 {
    return @as(f64, @floatFromInt(duration.nanoseconds)) / std.time.ns_per_ms;
}

test "prompt sources are exclusive and inline prompts are bounded" {
    const alloc = std.testing.allocator;
    try std.testing.expectError(error.MissingPrompt, readPrompt(alloc, std.testing.io, null, null));
    try std.testing.expectError(error.ConflictingPromptSources, readPrompt(alloc, std.testing.io, "a", "b"));
    const text = try readPrompt(alloc, std.testing.io, "hello", null);
    defer alloc.free(text);
    try std.testing.expectEqualStrings("hello", text);
    const big = try alloc.alloc(u8, 64 * 1024 + 1);
    defer alloc.free(big);
    @memset(big, 'x');
    try std.testing.expectError(error.LimitExceeded, readPrompt(alloc, std.testing.io, big, null));
}

test "prompt files are read completely and validated as UTF-8" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "p.txt", .data = "def f():\n    return 1\n" });
    try tmp.dir.writeFile(io, .{ .sub_path = "bad.txt", .data = "\xff\xfe" });
    try tmp.dir.writeFile(io, .{ .sub_path = "empty.txt", .data = "" });
    const good = try tmp.dir.realPathFileAlloc(io, "p.txt", alloc);
    defer alloc.free(good);
    const text = try readPrompt(alloc, io, null, good);
    defer alloc.free(text);
    try std.testing.expectEqualStrings("def f():\n    return 1\n", text);
    const bad = try tmp.dir.realPathFileAlloc(io, "bad.txt", alloc);
    defer alloc.free(bad);
    try std.testing.expectError(error.InvalidUtf8, readPrompt(alloc, io, null, bad));
    const empty = try tmp.dir.realPathFileAlloc(io, "empty.txt", alloc);
    defer alloc.free(empty);
    try std.testing.expectError(error.EmptyPrompt, readPrompt(alloc, io, null, empty));
}

test "token prompts are JSON arrays of in-range IDs" {
    const alloc = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "ok.json", .data = "[248045, 846,\n 198]" });
    try tmp.dir.writeFile(io, .{ .sub_path = "empty.json", .data = "[]" });
    try tmp.dir.writeFile(io, .{ .sub_path = "range.json", .data = "[1, 248320]" });
    try tmp.dir.writeFile(io, .{ .sub_path = "text.json", .data = "[\"a\"]" });
    const ok = try tmp.dir.realPathFileAlloc(io, "ok.json", alloc);
    defer alloc.free(ok);
    const tokens = try readPromptTokens(alloc, io, ok, 248_320);
    defer alloc.free(tokens);
    try std.testing.expectEqualSlices(u32, &.{ 248045, 846, 198 }, tokens);
    inline for (.{ .{ "empty.json", error.EmptyPrompt }, .{ "range.json", error.InvalidPromptTokens }, .{ "text.json", error.InvalidPromptTokens } }) |case| {
        const path = try tmp.dir.realPathFileAlloc(io, case[0], alloc);
        defer alloc.free(path);
        try std.testing.expectError(case[1], readPromptTokens(alloc, io, path, 248_320));
    }
}
