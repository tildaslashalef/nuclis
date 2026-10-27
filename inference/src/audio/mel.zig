//! Log-mel front ends: 16 kHz samples, a 512-point FFT every 160 samples,
//! 128 mel filters over 0–8000 Hz, and a log. Two extractors share it, and a
//! `Spec` holds what differs (docs/engine/audio.md § The front end):
//! Gemma 4's (`Gemma4AudioFeatureExtractor`: 320-sample periodic Hann frames
//! after 160 leading zeros, magnitude, HTK filters, `log(mel + 1e-3)`) and
//! Nemotron ASR's (`NemotronAsrStreamingFeatureExtractor`: pre-emphasis,
//! centered 400-sample symmetric Hann frames, power, Slaney filters,
//! `log(mel + 2⁻²⁴)`). Sums run in F64 and cast to F32. A frame reads only
//! its own samples, so `frames` also serves a stream. Pure: no I/O.
const std = @import("std");

pub const sample_rate = 16000;
pub const hop = 160;
pub const fft = 512;
pub const bins = fft / 2 + 1;
pub const filters = 128;
pub const max_frequency: f64 = 8000;

pub const Spec = struct {
    /// Samples under the Hann window.
    window: usize,
    /// Periodic (`n / window`) or symmetric (`n / (window − 1)`) Hann.
    periodic: bool,
    /// Where the window sits in the FFT's input; the rest is zero.
    window_at: usize,
    /// Frame k's FFT input starts at clip sample `hop · k − lead`; samples
    /// before the clip are zeros.
    lead: usize,
    count: Count,
    /// `|X|²` rather than `|X|`.
    power: bool,
    scale: Scale,
    /// `log(mel + guard)`.
    guard: f64,
    /// `y[n] = x[n] − c · x[n − 1]` in F32, `y[0] = x[0]`; 0 for none.
    preemphasis: f32,

    pub const Count = enum {
        /// Frames whose last sample, after the lead, lies inside the clip
        /// (Gemma's mask): `⌈(L − hop) / hop⌉`.
        ends_inside,
        /// `⌊L / hop⌋`, the centered STFT's frames less the last (Nemotron's
        /// mask zeroes it).
        centered,
    };

    pub const Scale = enum {
        /// `2595 · log10(1 + f / 700)`, triangles unnormalized.
        htk,
        /// Linear below 1 kHz, logarithmic above (librosa's `htk=False`),
        /// each triangle scaled to unit area, `2 / (f_{m+2} − f_m)`.
        slaney,
    };

    pub const gemma: Spec = .{ .window = 320, .periodic = true, .window_at = 0, .lead = 160, .count = .ends_inside, .power = false, .scale = .htk, .guard = 1e-3, .preemphasis = 0 };
    pub const nemotron: Spec = .{ .window = 400, .periodic = false, .window_at = (fft - 400) / 2, .lead = fft / 2, .count = .centered, .power = true, .scale = .slaney, .guard = 0x1p-24, .preemphasis = 0.97 };

    /// The frames a clip of `samples` gives.
    pub fn frameCount(self: Spec, samples: usize) usize {
        return switch (self.count) {
            .ends_inside => if (samples <= hop) 0 else (samples - hop + hop - 1) / hop,
            .centered => samples / hop,
        };
    }

    /// One past the last clip sample frame k reads under its window: the
    /// samples a stream needs before the frame is final.
    pub fn frameEnd(self: Spec, k: usize) usize {
        return (hop * k + self.window_at + self.window) -| self.lead;
    }

    /// The first clip sample frame k reads under its window.
    pub fn frameStart(self: Spec, k: usize) usize {
        return (hop * k + self.window_at) -| self.lead;
    }
};

/// Samples `[origin, origin + values.len)` of a clip; samples outside are
/// zeros. `previous` is sample `origin − 1`, which pre-emphasis reads.
pub const Span = struct {
    values: []const f32,
    origin: usize = 0,
    previous: f32 = 0,

    fn at(self: Span, index: usize) f32 {
        if (index < self.origin or index - self.origin >= self.values.len) return 0;
        return self.values[index - self.origin];
    }

    /// The (pre-emphasized) sample at clip index `index`.
    fn sample(self: Span, index: usize, c: f32) f32 {
        const x = self.at(index);
        if (c == 0 or index == 0) return x;
        const before = if (index == self.origin) self.previous else self.at(index - 1);
        return x - c * before;
    }
};

/// The window, the filter bank, and the FFT's twiddles, built once.
pub const Bank = struct {
    spec: Spec,
    /// F32, as the extractors store theirs, placed in the FFT's input.
    window: [fft]f32,
    /// `[bins][filters]`.
    weights: [bins * filters]f64,
    /// `exp(−2πik/fft)` for k < fft/2.
    twiddles: [fft / 2][2]f64,

    pub fn init(self: *Bank, spec: Spec) void {
        std.debug.assert(spec.window >= 2 and spec.window_at + spec.window <= fft);
        self.spec = spec;
        @memset(&self.window, 0);
        const period: f64 = @floatFromInt(if (spec.periodic) spec.window else spec.window - 1);
        for (self.window[spec.window_at..][0..spec.window], 0..) |*w, n| {
            const x = 2 * std.math.pi * @as(f64, @floatFromInt(n)) / period;
            w.* = @floatCast(0.5 - 0.5 * @cos(x));
        }
        // Filter edges equally spaced on the mel scale, bins equally spaced
        // in hertz; triangles by their two slopes.
        var edges: [filters + 2]f64 = undefined;
        const mel_max = toMel(spec.scale, max_frequency);
        for (&edges, 0..) |*e, i| e.* = fromMel(spec.scale, mel_max * @as(f64, @floatFromInt(i)) / @as(f64, filters + 1));
        for (0..bins) |k| {
            const f = @as(f64, @floatFromInt(k)) * (sample_rate / 2) / @as(f64, bins - 1);
            for (0..filters) |m| {
                const down = -(edges[m] - f) / (edges[m + 1] - edges[m]);
                const up = (edges[m + 2] - f) / (edges[m + 2] - edges[m + 1]);
                const area: f64 = switch (spec.scale) {
                    .htk => 1,
                    .slaney => 2 / (edges[m + 2] - edges[m]),
                };
                self.weights[k * filters + m] = @max(0, @min(down, up)) * area;
            }
        }
        for (&self.twiddles, 0..) |*t, k| {
            const angle = -2 * std.math.pi * @as(f64, @floatFromInt(k)) / fft;
            t.* = .{ @cos(angle), @sin(angle) };
        }
    }

    /// Writes the log-mel features of `samples` into `out`
    /// (`spec.frameCount(samples.len) × filters`, row-major), cast to F32.
    pub fn features(self: *const Bank, samples: []const f32, out: []f32) !void {
        if (out.len != self.spec.frameCount(samples.len) * filters) return error.InvalidShape;
        self.frames(.{ .values = samples }, 0, out);
    }

    /// Frames `first …` of the clip `span` holds, `out.len / filters` of
    /// them. The caller holds every sample those frames read under their
    /// windows (`Spec.frameStart`/`frameEnd`), and the one before for
    /// pre-emphasis; anything else reads as zero.
    pub fn frames(self: *const Bank, span: Span, first: usize, out: []f32) void {
        const spec = self.spec;
        var re: [fft]f64 = undefined;
        var im: [fft]f64 = undefined;
        for (0..out.len / filters) |i| {
            const k = first + i;
            for (0..fft) |n| {
                re[n] = 0;
                im[n] = 0;
                if (n < spec.window_at or n >= spec.window_at + spec.window) continue;
                const at = @as(i64, @intCast(k * hop + n)) - @as(i64, @intCast(spec.lead));
                if (at < 0) continue;
                // The product in F32, as the extractors' float32 arrays.
                re[n] = span.sample(@intCast(at), spec.preemphasis) * self.window[n];
            }
            self.transform(&re, &im);
            var spectrum: [bins]f64 = undefined;
            for (&spectrum, 0..) |*s, b| {
                const p = re[b] * re[b] + im[b] * im[b];
                s.* = if (spec.power) p else @sqrt(p);
            }
            for (out[i * filters ..][0..filters], 0..) |*o, m| {
                var sum: f64 = 0;
                for (spectrum, 0..) |v, b| sum += v * self.weights[b * filters + m];
                o.* = @floatCast(@log(sum + spec.guard));
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

const slaney_step = 200.0 / 3.0;
const slaney_knee = 1000.0;
const slaney_log_step = @log(6.4) / 27.0;

fn toMel(scale: Spec.Scale, hz: f64) f64 {
    return switch (scale) {
        .htk => 2595.0 * std.math.log10(1.0 + hz / 700.0),
        .slaney => if (hz < slaney_knee) hz / slaney_step else slaney_knee / slaney_step + @log(hz / slaney_knee) / slaney_log_step,
    };
}

fn fromMel(scale: Spec.Scale, mel: f64) f64 {
    return switch (scale) {
        .htk => 700.0 * (std.math.pow(f64, 10, mel / 2595.0) - 1.0),
        .slaney => if (mel < slaney_knee / slaney_step) mel * slaney_step else slaney_knee * @exp(slaney_log_step * (mel - slaney_knee / slaney_step)),
    };
}

test "frames end inside the clip: 4.8 s gives 479, a hop or less none" {
    const gemma = Spec.gemma;
    try std.testing.expectEqual(@as(usize, 479), gemma.frameCount(76800));
    try std.testing.expectEqual(@as(usize, 350), gemma.frameCount(56020));
    try std.testing.expectEqual(@as(usize, 0), gemma.frameCount(160));
    try std.testing.expectEqual(@as(usize, 1), gemma.frameCount(161));
    try std.testing.expectEqual(@as(usize, 1), gemma.frameCount(320));
    try std.testing.expectEqual(@as(usize, 2), gemma.frameCount(321));
}

test "Nemotron's frames: floor(L / 160), each windowed over 160k − 200 … 160k + 199" {
    const nemotron = Spec.nemotron;
    try std.testing.expectEqual(@as(usize, 505), nemotron.frameCount(80900));
    try std.testing.expectEqual(@as(usize, 0), nemotron.frameCount(159));
    try std.testing.expectEqual(@as(usize, 1), nemotron.frameCount(160));
    try std.testing.expectEqual(@as(usize, 0), nemotron.frameStart(1));
    try std.testing.expectEqual(@as(usize, 360), nemotron.frameEnd(1));
    try std.testing.expectEqual(@as(usize, 120), nemotron.frameStart(2));
    // The processor's first chunk at r = 3 is 25 frames over 4040 samples.
    try std.testing.expectEqual(@as(usize, 4040), nemotron.frameEnd(24));
}

test "the FFT matches a direct transform, and a tone peaks in its filter" {
    const bank = try std.testing.allocator.create(Bank);
    defer std.testing.allocator.destroy(bank);
    bank.init(.gemma);
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
    const out = try std.testing.allocator.alloc(f32, Spec.gemma.frameCount(samples.len) * filters);
    defer std.testing.allocator.free(out);
    try bank.features(&samples, out);
    const row = out[50 * filters ..][0..filters];
    const peak = std.mem.indexOfMax(f32, row);
    const mel_1k = 2595.0 * std.math.log10(1.0 + 1000.0 / 700.0);
    const mel_max = 2595.0 * std.math.log10(1.0 + max_frequency / 700.0);
    const expected: usize = @intFromFloat(@round(mel_1k / mel_max * (filters + 1) - 1));
    try std.testing.expect(@abs(@as(i64, @intCast(peak)) - @as(i64, @intCast(expected))) <= 1);
}

test "Nemotron's bank: Slaney edges and areas, and a constant's frame by hand" {
    const bank = try std.testing.allocator.create(Bank);
    defer std.testing.allocator.destroy(bank);
    bank.init(.nemotron);
    // Slaney: 1 kHz is mel 15; 8 kHz is 15 + 27 ln 8 / ln 6.4, so the
    // 130 edges are that over 129 steps. Filter 0 rises from 0 Hz to
    // edge 1 and falls to edge 2, its peak scaled by 2 / edge 2.
    const mel_max = 15.0 + 27.0 * @log(8.0) / @log(6.4);
    const e1 = mel_max / 129.0 * slaney_step;
    const e2 = 2 * mel_max / 129.0 * slaney_step;
    // Edge 1 is 23.4 Hz, so bin 1 (31.25 Hz) lies on filter 0's falling
    // edge, and bin 0 at its foot.
    try std.testing.expectApproxEqRel((e2 - 31.25) / (e2 - e1) * 2 / e2, bank.weights[1 * filters + 0], 1e-12);
    try std.testing.expectEqual(@as(f64, 0), bank.weights[0 * filters + 0]);
    // The window is symmetric, zero at both ends, centered in the FFT.
    try std.testing.expectEqual(@as(f32, 0), bank.window[55]);
    try std.testing.expectEqual(@as(f32, 0), bank.window[56]);
    try std.testing.expectEqual(@as(f32, 0), bank.window[455]);
    try std.testing.expectEqual(bank.window[56 + 100], bank.window[455 - 100]);
    try std.testing.expectApproxEqAbs(@as(f32, 1), bank.window[56 + 200], 1e-4);

    // Silence is the log guard everywhere: log 2⁻²⁴.
    var silence: [800]f32 = @splat(0);
    var quiet: [5 * filters]f32 = undefined;
    try bank.features(&silence, &quiet);
    for (quiet) |v| try std.testing.expectEqual(@as(f32, @floatCast(-24 * @log(2.0))), v);

    // A constant c: pre-emphasis leaves 0.03c after the first sample, so
    // frame 2 (samples 120 … 519, all inside) is 0.03c times the window;
    // its filters, by a direct DFT of that.
    const c: f32 = 0.25;
    var samples: [1000]f32 = @splat(c);
    var out: [6 * filters]f32 = undefined;
    try bank.features(&samples, &out);
    const y: f32 = c - 0.97 * c;
    var direct: [filters]f64 = @splat(0);
    for (0..bins) |b| {
        var dr: f64 = 0;
        var di: f64 = 0;
        for (56..456) |n| {
            const x: f64 = y * bank.window[n];
            const angle = -2 * std.math.pi * @as(f64, @floatFromInt(b * n % fft)) / fft;
            dr += x * @cos(angle);
            di += x * @sin(angle);
        }
        for (&direct, 0..) |*d, m| d.* += (dr * dr + di * di) * bank.weights[b * filters + m];
    }
    for (direct, out[2 * filters ..][0..filters]) |d, got| try std.testing.expectApproxEqAbs(@as(f32, @floatCast(@log(d + 0x1p-24))), got, 1e-5);
    // Frames are their own samples: a span starting mid-clip gives frame 4
    // as the whole clip does, given the sample before it.
    var piece: [filters]f32 = undefined;
    const start = Spec.nemotron.frameStart(4);
    bank.frames(.{ .values = samples[start..Spec.nemotron.frameEnd(4)], .origin = start, .previous = c }, 4, &piece);
    try std.testing.expectEqualSlices(f32, out[4 * filters ..][0..filters], &piece);
}
