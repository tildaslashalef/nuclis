//! A dropped audio file before anything encodes it: its length, read from the
//! file's header, against the prompt's bound (`inference.engine.max_audio_seconds`,
//! refused at drop, never cut), and the sentences the agent's refusals read.
//! The encode itself (`ahead.zig`, or the turn) checks the decoded length
//! again: a compressed format's header gives the decoder's estimate.
const std = @import("std");
const inference = @import("inference");
const catalog = @import("../catalog.zig");

const audio = inference.audio.audio;

/// The longest clip a prompt takes, in samples at 16 kHz.
pub const max_samples: u64 = inference.engine.max_audio_seconds * audio.sample_rate;

pub const Measure = union(enum) {
    /// Within the bound: its length at 16 kHz.
    fits: u64,
    /// Over the bound: its length at 16 kHz.
    too_long: u64,
    /// No decoder reads it.
    unreadable,
};

/// Measures a clip's encoded bytes by their header, decoding nothing.
pub fn measure(bytes: []const u8) Measure {
    var reader = audio.Reader.open(bytes) catch return .unreadable;
    defer reader.close();
    return if (reader.total > max_samples) .{ .too_long = reader.total } else .{ .fits = reader.total };
}

/// A clip's length from its samples at 16 kHz.
pub fn seconds(samples: u64) f64 {
    return @as(f64, @floatFromInt(samples)) / audio.sample_rate;
}

/// Room for any sentence below with a path's basename.
pub const sentence_bytes = 512;

/// The refusal of a clip over the bound, naming its length and the limit.
pub fn tooLong(buffer: *[sentence_bytes]u8, path: []const u8, samples: u64) []const u8 {
    return std.fmt.bufPrint(buffer, "  — {s} is {d:.1} s: a clip is at most {d} s, and is not cut", .{ std.fs.path.basename(path), seconds(samples), inference.engine.max_audio_seconds }) catch "  — the clip is too long";
}

/// The refusal of a file over the decoder's byte bound, unmeasured.
pub fn tooLarge(buffer: *[sentence_bytes]u8, path: []const u8) []const u8 {
    return std.fmt.bufPrint(buffer, "  — {s} is over {d} MB: a clip is at most {d} s", .{ std.fs.path.basename(path), audio.max_bytes >> 20, inference.engine.max_audio_seconds }) catch "  — the clip is too large";
}

/// The refusal of a file no decoder reads.
pub fn unreadable(buffer: *[sentence_bytes]u8, path: []const u8) []const u8 {
    return std.fmt.bufPrint(buffer, "  — {s} is not audio nuclis can read ({s})", .{ std.fs.path.basename(path), audio.readable }) catch "  — not audio nuclis can read";
}

/// The refusal of a clip by a model that does not hear, naming the
/// catalogue's model that does (or the projector the hearing one lacks).
pub fn deaf(buffer: *[sentence_bytes]u8, model: []const u8) []const u8 {
    const hearing = catalog.hearing() orelse
        return std.fmt.bufPrint(buffer, "  — {s} does not hear audio", .{model}) catch "  — this model does not hear audio";
    if (std.mem.eql(u8, hearing.name, model))
        return std.fmt.bufPrint(buffer, "  — {s} hears audio through its projector: nuclis model pull {s} --with mmproj", .{ model, model }) catch "  — the projector is missing";
    return std.fmt.bufPrint(buffer, "  — {s} does not hear audio; {s} does", .{ model, hearing.name }) catch "  — this model does not hear audio";
}

const testing = std.testing;

/// A 16 kHz mono 16-bit WAV of `samples` silent samples.
fn silentWav(alloc: std.mem.Allocator, samples: u32) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    const data: u32 = samples * 2;
    try out.appendSlice(alloc, "RIFF");
    try out.appendSlice(alloc, &std.mem.toBytes(std.mem.nativeToLittle(u32, 36 + data)));
    try out.appendSlice(alloc, "WAVEfmt ");
    try out.appendSlice(alloc, &std.mem.toBytes(std.mem.nativeToLittle(u32, 16)));
    try out.appendSlice(alloc, &std.mem.toBytes(std.mem.nativeToLittle(u16, 1)));
    try out.appendSlice(alloc, &std.mem.toBytes(std.mem.nativeToLittle(u16, 1)));
    try out.appendSlice(alloc, &std.mem.toBytes(std.mem.nativeToLittle(u32, audio.sample_rate)));
    try out.appendSlice(alloc, &std.mem.toBytes(std.mem.nativeToLittle(u32, audio.sample_rate * 2)));
    try out.appendSlice(alloc, &std.mem.toBytes(std.mem.nativeToLittle(u16, 2)));
    try out.appendSlice(alloc, &std.mem.toBytes(std.mem.nativeToLittle(u16, 16)));
    try out.appendSlice(alloc, "data");
    try out.appendSlice(alloc, &std.mem.toBytes(std.mem.nativeToLittle(u32, data)));
    try out.appendNTimes(alloc, 0, data);
    return out.toOwnedSlice(alloc);
}

test "a clip is measured by its header against the 30 s bound" {
    const alloc = testing.allocator;
    const short = try silentWav(alloc, 12 * audio.sample_rate + 6400);
    defer alloc.free(short);
    try testing.expectEqual(Measure{ .fits = 198_400 }, measure(short));
    const exact = try silentWav(alloc, 30 * audio.sample_rate);
    defer alloc.free(exact);
    try testing.expectEqual(Measure{ .fits = max_samples }, measure(exact));
    const long = try silentWav(alloc, 30 * audio.sample_rate + 1);
    defer alloc.free(long);
    try testing.expectEqual(Measure{ .too_long = max_samples + 1 }, measure(long));
    try testing.expectEqual(Measure.unreadable, measure("not audio at all, only text"));
}

test "the refusals name the clip, the bound, and the model that hears" {
    var buffer: [sentence_bytes]u8 = undefined;
    try testing.expectEqualStrings("  — long.wav is 42.3 s: a clip is at most 30 s, and is not cut", tooLong(&buffer, "/tmp/memos/long.wav", 676_800));
    try testing.expect(std.mem.startsWith(u8, unreadable(&buffer, "~/broken.m4a"), "  — broken.m4a is not audio nuclis can read ("));
    try testing.expectEqualStrings("  — gemma-4-12b-qat does not hear audio; gemma-4-e4b-qat does", deaf(&buffer, "gemma-4-12b-qat"));
    try testing.expectEqualStrings("  — gemma-4-e4b-qat hears audio through its projector: nuclis model pull gemma-4-e4b-qat --with mmproj", deaf(&buffer, "gemma-4-e4b-qat"));
    try testing.expectEqual(@as(f64, 12.4), seconds(198_400));
}
