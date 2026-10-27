//! Decoded audio for the audio encoders: 16 kHz mono f32 samples in
//! [-1, 1]. Two decoders: a RIFF/WAVE parser for 16-bit PCM or 32-bit float
//! at 16 kHz mono (fixtures, tests, any platform) and, when the macOS bridges are built,
//! AudioToolbox for the platform's formats, which resamples and averages
//! channels. `decode` gives a clip whole, within `max_bytes`; a `Reader`
//! gives a long recording in pieces, the bytes bounded by its caller.
//! Bounds are host constants, never model-supplied.
const std = @import("std");
const build_options = @import("build_options");

/// The bridge is compiled with the Metal one: both are the macOS bridges.
pub const bridge_enabled = build_options.metal;
/// The formats `decode` reads, for messages.
pub const readable = if (bridge_enabled) "WAV, AIFF, MP3, M4A, FLAC, CAF" else "none: audio needs the macOS build";

pub const sample_rate = 16000;
/// Encoded bytes `decode` accepts.
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

/// Decodes `bytes`, keeping at most `max_samples`: a 16 kHz mono 16-bit or
/// float WAV directly, anything else through the bridge.
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

/// A RIFF/WAVE file of one channel at 16 kHz, 16-bit PCM (each sample over
/// 32768, as every decoder reads it) or 32-bit IEEE float (as stored, which
/// FLEURS uses), plain or `WAVE_FORMAT_EXTENSIBLE`. Other WAVs are
/// `UnsupportedAudioFormat`.
pub fn decodeWav(alloc: std.mem.Allocator, bytes: []const u8, max_samples: u32) Error!Pcm {
    const wav = try parseWav(bytes);
    const total = wav.count();
    const samples = try alloc.alloc(f32, @min(total, max_samples));
    errdefer alloc.free(samples);
    try wav.convert(0, samples);
    return .{ .samples = samples, .total = total };
}

/// The samples of a WAV `decodeWav` reads, still encoded.
const Wav = struct {
    format: enum { int16, float32 },
    data: []const u8,

    fn width(self: Wav) usize {
        return switch (self.format) {
            .int16 => 2,
            .float32 => 4,
        };
    }

    fn count(self: Wav) usize {
        return self.data.len / self.width();
    }

    /// Samples `first ..` into `out`; a NaN or an infinity is refused.
    fn convert(self: Wav, first: usize, out: []f32) Error!void {
        const bytes = self.data[first * self.width() ..];
        switch (self.format) {
            .int16 => for (out, 0..) |*s, i| {
                s.* = @as(f32, @floatFromInt(std.mem.readInt(i16, bytes[i * 2 ..][0..2], .little))) / 32768.0;
            },
            .float32 => for (out, 0..) |*s, i| {
                s.* = @bitCast(std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little));
                // Untrusted: a NaN or an infinity would reach every feature.
                if (!std.math.isFinite(s.*)) return error.UnsupportedAudioFormat;
            },
        }
    }
};

fn parseWav(bytes: []const u8) Error!Wav {
    if (bytes.len < 12 or !std.mem.eql(u8, bytes[0..4], "RIFF") or !std.mem.eql(u8, bytes[8..12], "WAVE")) return error.UnsupportedAudioFormat;
    var at: usize = 12;
    var format: ?@FieldType(Wav, "format") = null;
    while (at + 8 <= bytes.len) {
        const id = bytes[at..][0..4];
        const size = std.mem.readInt(u32, bytes[at + 4 ..][0..4], .little);
        const body_start = at + 8;
        if (size > bytes.len - body_start) return error.UnsupportedAudioFormat;
        const body = bytes[body_start..][0..size];
        if (std.mem.eql(u8, id, "fmt ")) {
            if (size < 16) return error.UnsupportedAudioFormat;
            var tag = std.mem.readInt(u16, body[0..2], .little);
            const channels = std.mem.readInt(u16, body[2..4], .little);
            const rate = std.mem.readInt(u32, body[4..8], .little);
            const bits = std.mem.readInt(u16, body[14..16], .little);
            // WAVE_FORMAT_EXTENSIBLE: the real tag opens the sub-format GUID.
            if (tag == 0xFFFE) tag = if (size >= 40) std.mem.readInt(u16, body[24..26], .little) else 0;
            if (channels != 1 or rate != sample_rate) return error.UnsupportedAudioFormat;
            format = if (tag == 1 and bits == 16) .int16 else if (tag == 3 and bits == 32) .float32 else return error.UnsupportedAudioFormat;
        } else if (std.mem.eql(u8, id, "data")) {
            return .{ .format = format orelse return error.UnsupportedAudioFormat, .data = body };
        }
        // Chunks are padded to an even size.
        at = body_start + size + (size & 1);
    }
    return error.UnsupportedAudioFormat;
}

/// A decoding in pieces of bounded size, for audio too long to hold
/// decoded: the samples `decode` gives, in the same order, `read` by `read`.
/// `bytes` are borrowed for the reader's life.
pub const Reader = struct {
    source: union(enum) {
        wav: struct { wav: Wav, at: usize = 0 },
        bridge: *anyopaque,
    },
    /// The length at 16 kHz: exact for a WAV read directly, the bridge's
    /// estimate otherwise (the reads may pass it by a frame).
    total: u64,

    pub fn open(bytes: []const u8) Error!Reader {
        if (parseWav(bytes)) |wav| return .{ .source = .{ .wav = .{ .wav = wav } }, .total = wav.count() } else |_| {}
        if (!bridge_enabled) return error.UnsupportedAudioFormat;
        var handle: *anyopaque = undefined;
        var total: u64 = 0;
        return switch (nu_audio_open(bytes.ptr, bytes.len, &handle, &total)) {
            0 => .{ .source = .{ .bridge = handle }, .total = total },
            1 => error.UnsupportedAudioFormat,
            else => error.OutOfMemory,
        };
    }

    pub fn close(self: *Reader) void {
        switch (self.source) {
            .wav => {},
            .bridge => |handle| nu_audio_close(handle),
        }
        self.* = undefined;
    }

    /// Fills `out` from the next samples; fewer only at the end, 0 past it.
    /// `UnsupportedAudioFormat` when the file stops decoding midway.
    pub fn read(self: *Reader, out: []f32) Error!usize {
        switch (self.source) {
            .wav => |*w| {
                const n = @min(out.len, w.wav.count() - w.at);
                try w.wav.convert(w.at, out[0..n]);
                w.at += n;
                return n;
            },
            .bridge => |handle| {
                var n: u32 = 0;
                if (nu_audio_read(handle, out.ptr, @intCast(@min(out.len, std.math.maxInt(u32))), &n) != 0) return error.UnsupportedAudioFormat;
                return n;
            },
        }
    }
};

extern fn nu_audio_open([*]const u8, usize, **anyopaque, *u64) c_int;
extern fn nu_audio_read(*anyopaque, [*]f32, u32, *u32) c_int;
extern fn nu_audio_close(*anyopaque) void;

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

/// A 32-bit float WAV of `samples` as FLEURS writes it (an 18-byte `fmt `
/// and a `fact` chunk), or as `WAVE_FORMAT_EXTENSIBLE` with sub-format `tag`.
fn floatWav(alloc: std.mem.Allocator, samples: []const f32, extensible: ?u16) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    const le = struct {
        fn put(a: std.mem.Allocator, o: *std.ArrayList(u8), comptime T: type, v: T) !void {
            try o.appendSlice(a, &std.mem.toBytes(std.mem.nativeToLittle(T, v)));
        }
    }.put;
    const fmt_size: u32 = if (extensible != null) 40 else 18;
    const data_bytes: u32 = @intCast(samples.len * 4);
    try out.appendSlice(alloc, "RIFF");
    try le(alloc, &out, u32, 4 + 8 + fmt_size + 12 + 8 + data_bytes);
    try out.appendSlice(alloc, "WAVEfmt ");
    try le(alloc, &out, u32, fmt_size);
    try le(alloc, &out, u16, if (extensible != null) 0xFFFE else 3);
    try le(alloc, &out, u16, 1);
    try le(alloc, &out, u32, sample_rate);
    try le(alloc, &out, u32, sample_rate * 4);
    try le(alloc, &out, u16, 4);
    try le(alloc, &out, u16, 32);
    if (extensible) |tag| {
        try le(alloc, &out, u16, 22);
        try le(alloc, &out, u16, 32);
        try le(alloc, &out, u32, 4);
        try le(alloc, &out, u16, tag);
        try out.appendSlice(alloc, "\x00\x00\x00\x00\x10\x00\x80\x00\x00\xAA\x00\x38\x9B\x71");
    } else try le(alloc, &out, u16, 0);
    try out.appendSlice(alloc, "fact");
    try le(alloc, &out, u32, 4);
    try le(alloc, &out, u32, @intCast(samples.len));
    try out.appendSlice(alloc, "data");
    try le(alloc, &out, u32, data_bytes);
    for (samples) |v| try le(alloc, &out, u32, @bitCast(v));
    return out.toOwnedSlice(alloc);
}

test "a 32-bit float WAV, plain or extensible, decodes as stored; NaN and other sub-formats are refused" {
    const alloc = std.testing.allocator;
    const values = [_]f32{ 0, 0.25, -1, 0.123456789, 1 };
    for ([_]?u16{ null, 3 }) |extensible| {
        const wav = try floatWav(alloc, &values, extensible);
        defer alloc.free(wav);
        var pcm = try decodeWav(alloc, wav, std.math.maxInt(u32));
        defer pcm.deinit(alloc);
        try std.testing.expectEqualSlices(f32, &values, pcm.samples);
        try std.testing.expectEqual(@as(u64, 5), pcm.total);
        var cut = try decodeWav(alloc, wav, 3);
        defer cut.deinit(alloc);
        try std.testing.expectEqualSlices(f32, values[0..3], cut.samples);
    }
    // Extensible PCM at 32 bits is integer, not float: refused.
    const pcm32 = try floatWav(alloc, &values, 1);
    defer alloc.free(pcm32);
    try std.testing.expectError(error.UnsupportedAudioFormat, decodeWav(alloc, pcm32, 10));
    const nan = try floatWav(alloc, &.{ 0, std.math.nan(f32) }, null);
    defer alloc.free(nan);
    try std.testing.expectError(error.UnsupportedAudioFormat, decodeWav(alloc, nan, 10));
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

test "a reader gives decode's samples in pieces of any size" {
    const alloc = std.testing.allocator;
    var samples: [1000]i16 = undefined;
    for (&samples, 0..) |*v, i| v.* = @intCast(@as(i32, @intCast(i * 37 % 2000)) - 1000);
    const wav = try testWav(alloc, &samples, 1, sample_rate);
    defer alloc.free(wav);
    var whole = try decode(alloc, wav, std.math.maxInt(u32));
    defer whole.deinit(alloc);
    for ([_]usize{ 1, 7, 333, 4096 }) |piece| {
        var reader = try Reader.open(wav);
        defer reader.close();
        try std.testing.expectEqual(@as(u64, 1000), reader.total);
        var got: std.ArrayList(f32) = .empty;
        defer got.deinit(alloc);
        const buffer = try alloc.alloc(f32, piece);
        defer alloc.free(buffer);
        while (true) {
            const n = try reader.read(buffer);
            if (n == 0) break;
            try got.appendSlice(alloc, buffer[0..n]);
        }
        try std.testing.expectEqualSlices(f32, whole.samples, got.items);
    }
    const nan = try floatWav(alloc, &.{ 0, 1, std.math.nan(f32) }, null);
    defer alloc.free(nan);
    var reader = try Reader.open(nan);
    defer reader.close();
    var two: [2]f32 = undefined;
    try std.testing.expectEqual(@as(usize, 2), try reader.read(&two));
    try std.testing.expectError(error.UnsupportedAudioFormat, reader.read(&two));
    if (!bridge_enabled) try std.testing.expectError(error.UnsupportedAudioFormat, Reader.open("not audio at all, not even close"));
}

test "the bridge's reader gives its whole decode, in pieces" {
    if (!bridge_enabled) return error.SkipZigTest;
    const alloc = std.testing.allocator;
    var samples: [2 * 22050]i16 = undefined;
    for (0..22050) |i| {
        const v: i16 = @intFromFloat(8000 * @sin(2 * std.math.pi * 330 * @as(f64, @floatFromInt(i)) / 22050));
        samples[2 * i] = v;
        samples[2 * i + 1] = -v;
    }
    const wav = try testWav(alloc, &samples, 2, 22050);
    defer alloc.free(wav);
    var whole = try decode(alloc, wav, std.math.maxInt(u32));
    defer whole.deinit(alloc);
    var reader = try Reader.open(wav);
    defer reader.close();
    try std.testing.expect(reader.source == .bridge);
    var got: std.ArrayList(f32) = .empty;
    defer got.deinit(alloc);
    var buffer: [1000]f32 = undefined;
    while (true) {
        const n = try reader.read(&buffer);
        if (n == 0) break;
        try got.appendSlice(alloc, buffer[0..n]);
    }
    try std.testing.expectEqualSlices(f32, whole.samples, got.items);
    try std.testing.expectError(error.UnsupportedAudioFormat, Reader.open("not audio at all, not even close"));
}
