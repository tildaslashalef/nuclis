//! The log-mel front end of Gemma 4's audio encoders, as Google's
//! `Gemma4AudioFeatureExtractor` computes it (docs/engine/audio.md § The
//! front end): 16 kHz samples, 160 zeros in front, 320-sample frames every
//! 160 samples under a periodic Hann window, the magnitude of a 512-point
//! FFT, a 128-filter HTK mel bank over 0–8000 Hz without normalization, and
//! `log(mel + 1e-3)`. Only frames that end inside the clip are kept, as the
//! extractor's mask keeps them. Pure: no I/O.
const std = @import("std");

pub const sample_rate = 16000;
pub const frame = 320;
pub const hop = 160;
pub const fft = 512;
pub const bins = fft / 2 + 1;
pub const filters = 128;
pub const max_frequency: f64 = 8000;
pub const floor: f64 = 1e-3;

/// The frames a clip of `samples` gives: those whose last sample (with
/// the 160 leading zeros) falls inside the clip.
pub fn frameCount(samples: usize) usize {
    if (samples <= hop) return 0;
    return (samples - hop + hop - 1) / hop;
}

/// The window, the filter bank, and the FFT's twiddles, built once.
pub const Bank = struct {
    /// F32, as the extractor stores its window.
    window: [frame]f32,
    /// `[bins][filters]`.
    weights: [bins * filters]f64,
    /// `exp(−2πik/fft)` for k < fft/2.
    twiddles: [fft / 2][2]f64,

    pub fn init(self: *Bank) void {
        // np.hanning(frame + 1)[:frame]: the periodic Hann window.
        for (&self.window, 0..) |*w, n| {
            const x = 2 * std.math.pi * @as(f64, @floatFromInt(n)) / frame;
            w.* = @floatCast(0.5 - 0.5 * @cos(x));
        }
        // HTK mels, filter edges equally spaced on the mel scale, bins
        // equally spaced in hertz; triangles by their two slopes.
        var edges: [filters + 2]f64 = undefined;
        const mel_max = 2595.0 * std.math.log10(1.0 + max_frequency / 700.0);
        for (&edges, 0..) |*e, i| {
            const mel = mel_max * @as(f64, @floatFromInt(i)) / @as(f64, filters + 1);
            e.* = 700.0 * (std.math.pow(f64, 10, mel / 2595.0) - 1.0);
        }
        for (0..bins) |k| {
            const f = @as(f64, @floatFromInt(k)) * (sample_rate / 2) / @as(f64, bins - 1);
            for (0..filters) |m| {
                const down = -(edges[m] - f) / (edges[m + 1] - edges[m]);
                const up = (edges[m + 2] - f) / (edges[m + 2] - edges[m + 1]);
                self.weights[k * filters + m] = @max(0, @min(down, up));
            }
        }
        for (&self.twiddles, 0..) |*t, k| {
            const angle = -2 * std.math.pi * @as(f64, @floatFromInt(k)) / fft;
            t.* = .{ @cos(angle), @sin(angle) };
        }
    }

    /// Writes the log-mel features of `samples` into `out`
    /// (`frameCount(samples.len) × filters`, row-major), cast to F32.
    pub fn features(self: *const Bank, samples: []const f32, out: []f32) !void {
        const count = frameCount(samples.len);
        if (out.len != count * filters) return error.InvalidShape;
        var re: [fft]f64 = undefined;
        var im: [fft]f64 = undefined;
        for (0..count) |i| {
            // Frame i covers clip samples [i·hop − hop, i·hop − hop + frame).
            for (0..fft) |n| {
                re[n] = 0;
                im[n] = 0;
                if (n >= frame) continue;
                const at = @as(i64, @intCast(i * hop + n)) - hop;
                if (at < 0 or at >= samples.len) continue;
                // The product in F32, as the extractor's float32 arrays.
                re[n] = samples[@intCast(at)] * self.window[n];
            }
            self.transform(&re, &im);
            var magnitude: [bins]f64 = undefined;
            for (&magnitude, 0..) |*m, k| m.* = @sqrt(re[k] * re[k] + im[k] * im[k]);
            for (out[i * filters ..][0..filters], 0..) |*o, m| {
                var sum: f64 = 0;
                for (magnitude, 0..) |v, k| sum += v * self.weights[k * filters + m];
                o.* = @floatCast(@log(sum + floor));
            }
        }
    }

    /// In-place iterative radix-2 FFT of `fft` points.
    fn transform(self: *const Bank, re: *[fft]f64, im: *[fft]f64) void {
        comptime std.debug.assert(fft == 1 << 9);
        for (0..fft) |i| {
            const j = @bitReverse(@as(u9, @intCast(i)));
            if (j > i) {
                std.mem.swap(f64, &re[i], &re[j]);
                std.mem.swap(f64, &im[i], &im[j]);
            }
        }
        var size: usize = 2;
        while (size <= fft) : (size *= 2) {
            const half = size / 2;
            const step = fft / size;
            var start: usize = 0;
            while (start < fft) : (start += size) {
                for (0..half) |k| {
                    const t = self.twiddles[k * step];
                    const a = start + k;
                    const b = a + half;
                    const xr = re[b] * t[0] - im[b] * t[1];
                    const xi = re[b] * t[1] + im[b] * t[0];
                    re[b] = re[a] - xr;
                    im[b] = im[a] - xi;
                    re[a] += xr;
                    im[a] += xi;
                }
            }
        }
    }
};

test "frames end inside the clip: 4.8 s gives 479, a hop or less none" {
    try std.testing.expectEqual(@as(usize, 479), frameCount(76800));
    try std.testing.expectEqual(@as(usize, 350), frameCount(56020));
    try std.testing.expectEqual(@as(usize, 0), frameCount(160));
    try std.testing.expectEqual(@as(usize, 1), frameCount(161));
    try std.testing.expectEqual(@as(usize, 1), frameCount(320));
    try std.testing.expectEqual(@as(usize, 2), frameCount(321));
}

test "the FFT matches a direct transform, and a tone peaks in its filter" {
    const bank = try std.testing.allocator.create(Bank);
    defer std.testing.allocator.destroy(bank);
    bank.init();
    var re: [fft]f64 = undefined;
    var im: [fft]f64 = @splat(0);
    for (&re, 0..) |*r, n| r.* = @sin(@as(f64, @floatFromInt(n)) * 0.3) + 0.25 * @cos(@as(f64, @floatFromInt(n * n % 17)));
    const original = re;
    bank.transform(&re, &im);
    for ([_]usize{ 0, 1, 37, 255, 256 }) |k| {
        var dr: f64 = 0;
        var di: f64 = 0;
        for (original, 0..) |x, n| {
            const angle = -2 * std.math.pi * @as(f64, @floatFromInt(k * n % fft)) / fft;
            dr += x * @cos(angle);
            di += x * @sin(angle);
        }
        try std.testing.expectApproxEqAbs(dr, re[k], 1e-9);
        try std.testing.expectApproxEqAbs(di, im[k], 1e-9);
    }
    // The bank's edges: filter 0 starts at 0 Hz, the last ends at 8000 Hz.
    try std.testing.expectEqual(@as(f64, 0), bank.weights[0 * filters + 0]);
    try std.testing.expect(bank.weights[(bins - 1) * filters + filters - 1] < 1e-9);
    // A 1 kHz tone: its strongest filter is the one around 1 kHz.
    var samples: [16000]f32 = undefined;
    for (&samples, 0..) |*s, n| s.* = @floatCast(0.5 * @sin(2 * std.math.pi * 1000 * @as(f64, @floatFromInt(n)) / sample_rate));
    const out = try std.testing.allocator.alloc(f32, frameCount(samples.len) * filters);
    defer std.testing.allocator.free(out);
    try bank.features(&samples, out);
    const row = out[50 * filters ..][0..filters];
    const peak = std.mem.indexOfMax(f32, row);
    const mel_1k = 2595.0 * std.math.log10(1.0 + 1000.0 / 700.0);
    const mel_max = 2595.0 * std.math.log10(1.0 + max_frequency / 700.0);
    const expected: usize = @intFromFloat(@round(mel_1k / mel_max * (filters + 1) - 1));
    try std.testing.expect(@abs(@as(i64, @intCast(peak)) - @as(i64, @intCast(expected))) <= 1);
}
