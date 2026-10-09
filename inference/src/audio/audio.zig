//! Decoded audio for the audio encoders: 16 kHz mono f32 samples in
//! [-1, 1]. Two decoders: a RIFF/WAVE parser for 16-bit PCM at 16 kHz
//! (fixtures, tests, any platform) and, when the macOS bridges are built,
//! AudioToolbox for the platform's formats, which resamples and averages
//! channels. Bounds are host constants, never model-supplied.
const std = @import("std");
const build_options = @import("build_options");

/// The bridge is compiled with the Metal one: both are the macOS bridges.
pub const bridge_enabled = build_options.metal;

pub const sample_rate = 16000;
/// Encoded bytes accepted from a file.
pub const max_bytes: usize = 64 << 20;

pub const Error = error{
    UnsupportedAudioFormat,
    AudioTooLarge,
    OutOfMemory,
};

pub const Pcm = struct {
    /// At most the `max_samples` the decode was given, owned.
    samples: []f32,
    /// The clip's length at 16 kHz before that bound; above `samples.len`
    /// only when it was cut.
    total: u64,

    pub fn deinit(self: *Pcm, alloc: std.mem.Allocator) void {
        alloc.free(self.samples);
        self.* = undefined;
    }
};

/// Decodes `bytes`, keeping at most `max_samples`: a 16 kHz mono 16-bit WAV
/// directly, anything else through the bridge.
pub fn decode(alloc: std.mem.Allocator, bytes: []const u8, max_samples: u32) Error!Pcm {
    if (bytes.len > max_bytes) return error.AudioTooLarge;
    if (decodeWav(alloc, bytes, max_samples)) |pcm| return pcm else |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => {},
    }
    if (!bridge_enabled) return error.UnsupportedAudioFormat;
    var samples: [*]f32 = undefined;
    var count: u32 = 0;
    var total: u64 = 0;
    switch (nu_audio_decode(bytes.ptr, bytes.len, max_samples, &samples, &count, &total)) {
        0 => {},
        1 => return error.UnsupportedAudioFormat,
        else => return error.OutOfMemory,
    }
    defer nu_audio_free(samples);
    return .{ .samples = try alloc.dupe(f32, samples[0..count]), .total = total };
}

extern fn nu_audio_decode([*]const u8, usize, u32, *[*]f32, *u32, *u64) c_int;
extern fn nu_audio_free([*]f32) void;

/// A RIFF/WAVE file of 16-bit PCM, one channel, 16 kHz: each sample over
/// 32768, as every decoder reads it. Other WAVs are `UnsupportedAudioFormat`.
pub fn decodeWav(alloc: std.mem.Allocator, bytes: []const u8, max_samples: u32) Error!Pcm {
    if (bytes.len < 12 or !std.mem.eql(u8, bytes[0..4], "RIFF") or !std.mem.eql(u8, bytes[8..12], "WAVE")) return error.UnsupportedAudioFormat;
    var at: usize = 12;
    var format_ok = false;
    while (at + 8 <= bytes.len) {
        const id = bytes[at..][0..4];
        const size = std.mem.readInt(u32, bytes[at + 4 ..][0..4], .little);
        const body_start = at + 8;
        if (size > bytes.len - body_start) return error.UnsupportedAudioFormat;
        const body = bytes[body_start..][0..size];
        if (std.mem.eql(u8, id, "fmt ")) {
            if (size < 16) return error.UnsupportedAudioFormat;
            const tag = std.mem.readInt(u16, body[0..2], .little);
            const channels = std.mem.readInt(u16, body[2..4], .little);
            const rate = std.mem.readInt(u32, body[4..8], .little);
            const bits = std.mem.readInt(u16, body[14..16], .little);
            format_ok = tag == 1 and channels == 1 and rate == sample_rate and bits == 16;
            if (!format_ok) return error.UnsupportedAudioFormat;
        } else if (std.mem.eql(u8, id, "data")) {
            if (!format_ok) return error.UnsupportedAudioFormat;
            const total = size / 2;
            const kept = @min(total, max_samples);
            const samples = try alloc.alloc(f32, kept);
            for (samples, 0..) |*s, i| s.* = @as(f32, @floatFromInt(std.mem.readInt(i16, body[i * 2 ..][0..2], .little))) / 32768.0;
            return .{ .samples = samples, .total = total };
        }
        // Chunks are padded to an even size.
        at = body_start + size + (size & 1);
    }
    return error.UnsupportedAudioFormat;
}

/// A 16-bit WAV of `samples` with the given header fields, for tests.
fn testWav(alloc: std.mem.Allocator, samples: []const i16, channels: u16, rate: u32) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    const data_bytes: u32 = @intCast(samples.len * 2);
    try out.appendSlice(alloc, "RIFF");
    try out.appendSlice(alloc, &std.mem.toBytes(std.mem.nativeToLittle(u32, 36 + 12 + data_bytes)));
    try out.appendSlice(alloc, "WAVEfmt ");
    for ([_]u32{16}) |v| try out.appendSlice(alloc, &std.mem.toBytes(std.mem.nativeToLittle(u32, v)));
    for ([_]u16{ 1, channels }) |v| try out.appendSlice(alloc, &std.mem.toBytes(std.mem.nativeToLittle(u16, v)));
    for ([_]u32{ rate, rate * 2 * channels }) |v| try out.appendSlice(alloc, &std.mem.toBytes(std.mem.nativeToLittle(u32, v)));
    for ([_]u16{ 2 * channels, 16 }) |v| try out.appendSlice(alloc, &std.mem.toBytes(std.mem.nativeToLittle(u16, v)));
    // An odd-sized chunk before the data, as afconvert writes `FLLR`.
    try out.appendSlice(alloc, "LIST\x03\x00\x00\x00abc\x00");
    try out.appendSlice(alloc, "data");
    try out.appendSlice(alloc, &std.mem.toBytes(std.mem.nativeToLittle(u32, data_bytes)));
    for (samples) |v| try out.appendSlice(alloc, &std.mem.toBytes(std.mem.nativeToLittle(i16, v)));
    return out.toOwnedSlice(alloc);
}

test "a 16-bit mono 16 kHz WAV decodes directly; other WAVs and other bytes are refused" {
    const alloc = std.testing.allocator;
    const wav = try testWav(alloc, &.{ 0, 16384, -32768, 32767 }, 1, sample_rate);
    defer alloc.free(wav);
    var pcm = try decodeWav(alloc, wav, std.math.maxInt(u32));
    defer pcm.deinit(alloc);
    try std.testing.expectEqualSlices(f32, &.{ 0, 0.5, -1, 32767.0 / 32768.0 }, pcm.samples);
    try std.testing.expectEqual(@as(u64, 4), pcm.total);
    var cut = try decodeWav(alloc, wav, 2);
    defer cut.deinit(alloc);
    try std.testing.expectEqual(@as(usize, 2), cut.samples.len);
    try std.testing.expectEqual(@as(u64, 4), cut.total);
    const stereo = try testWav(alloc, &.{ 1, 2 }, 2, sample_rate);
    defer alloc.free(stereo);
    try std.testing.expectError(error.UnsupportedAudioFormat, decodeWav(alloc, stereo, 10));
    const rate = try testWav(alloc, &.{ 1, 2 }, 1, 44100);
    defer alloc.free(rate);
    try std.testing.expectError(error.UnsupportedAudioFormat, decodeWav(alloc, rate, 10));
    try std.testing.expectError(error.UnsupportedAudioFormat, decodeWav(alloc, "RIFF\x00\x00\x00\x00WAVE", 10));
    try std.testing.expectError(error.UnsupportedAudioFormat, decodeWav(alloc, "not audio at all", 10));
}

test "the bridge resamples and mixes what the WAV parser refuses" {
    if (!bridge_enabled) return error.SkipZigTest;
    const alloc = std.testing.allocator;
    // One second of a 44.1 kHz stereo tone: 16,000 mono samples come back.
    var samples: [2 * 44100]i16 = undefined;
    for (0..44100) |i| {
        const v: i16 = @intFromFloat(8000 * @sin(2 * std.math.pi * 440 * @as(f64, @floatFromInt(i)) / 44100));
        samples[2 * i] = v;
        samples[2 * i + 1] = v;
    }
    const wav = try testWav(alloc, &samples, 2, 44100);
    defer alloc.free(wav);
    var pcm = try decode(alloc, wav, std.math.maxInt(u32));
    defer pcm.deinit(alloc);
    try std.testing.expect(pcm.samples.len >= 15990 and pcm.samples.len <= 16010);
    var peak: f32 = 0;
    for (pcm.samples[1000..15000]) |s| peak = @max(peak, @abs(s));
    try std.testing.expectApproxEqAbs(@as(f32, 8000.0 / 32768.0), peak, 0.01);
    try std.testing.expectError(error.UnsupportedAudioFormat, decode(alloc, "not audio at all, not even close", 10));
}
