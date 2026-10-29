//! Qwen3-TTS Base's speaker encoder on the CPU: 24 kHz mono samples to the
//! log-mel `extract_speaker_embedding` computes (`Front`), then the
//! ECAPA-TDNN (`Encoder`) to one embedding at the talker's width, the
//! prompt's speaker row. It runs once per saved voice, so the F32 CPU path
//! is the only one. Rows are time-major, `[frames][channels]`; every
//! convolution is a matrix product over its gathered taps (`cpu.dense`), so
//! the weights keep PyTorch's `[out][in][kernel]` layout.
//! docs/models/qwen3-tts.md § The speaker encoder.
const std = @import("std");
const cpu = @import("../backends/cpu/root.zig");
const tts = @import("../models/qwen3_tts.zig");
const mel = @import("mel.zig");

const Allocator = std.mem.Allocator;
const dense = cpu.dense;

pub const sample_rate = 24000;
pub const fft = 1024;
pub const hop = 256;
pub const bins = fft / 2 + 1;
pub const filters = 128;
/// Samples reflected before the clip and after it, `(fft − hop) / 2`.
pub const pad = (fft - hop) / 2;
const max_frequency: f64 = sample_rate / 2;
/// The pooling's floor under a variance.
const variance_floor = 1e-12;

pub const Error = Allocator.Error || dense.Error || error{ClipTooShort};

/// The front end's window, filter bank, and twiddles (about 300 KB: keep it
/// on the heap).
pub const Front = struct {
    /// Periodic Hann, F32 as `torch.hann_window`.
    window: [fft]f32,
    /// `[filters][bins]`: librosa's Slaney bank, cast to F32 as it returns it.
    weights: [filters * bins]f32,
    /// `exp(−2πik/fft)` for k < fft/2.
    twiddles: [fft / 2][2]f64,

    pub fn init(self: *Front) void {
        for (&self.window, 0..) |*w, n| {
            const x = 2 * std.math.pi * @as(f64, @floatFromInt(n)) / fft;
            w.* = @floatCast(0.5 - 0.5 * @cos(x));
        }
        // Edges equally spaced on the Slaney mel scale over 0 … 12 kHz,
        // triangles of unit area.
        var edges: [filters + 2]f64 = undefined;
        const top = mel.toMel(.slaney, max_frequency);
        for (&edges, 0..) |*e, i| e.* = mel.fromMel(.slaney, top * @as(f64, @floatFromInt(i)) / @as(f64, filters + 1));
        for (0..filters) |m| for (0..bins) |b| {
            const f = @as(f64, @floatFromInt(b)) * max_frequency / @as(f64, bins - 1);
            const down = (f - edges[m]) / (edges[m + 1] - edges[m]);
            const up = (edges[m + 2] - f) / (edges[m + 2] - edges[m + 1]);
            self.weights[m * bins + b] = @floatCast(@max(0, @min(down, up)) * 2 / (edges[m + 2] - edges[m]));
        };
        for (&self.twiddles, 0..) |*t, k| {
            const angle = -2 * std.math.pi * @as(f64, @floatFromInt(k)) / fft;
            t.* = .{ @cos(angle), @sin(angle) };
        }
    }

    /// The frames a clip of `samples` gives: 0 when it is too short to
    /// reflect (`pad` samples or fewer).
    pub fn frameCount(samples: usize) usize {
        if (samples <= pad) return 0;
        return (samples + 2 * pad - fft) / hop + 1;
    }

    /// Writes the log-mel rows of `samples` into `out`
    /// (`frameCount(samples.len) × filters`).
    pub fn features(self: *const Front, samples: []const f32, out: []f32) Error!void {
        const frames = frameCount(samples.len);
        if (frames == 0) return error.ClipTooShort;
        std.debug.assert(out.len == frames * filters);
        var re: [fft]f64 = undefined;
        var im: [fft]f64 = undefined;
        for (0..frames) |k| {
            for (0..fft) |n| {
                const at = @as(i64, @intCast(k * hop + n)) - pad;
                // The product in F32, as the authors' tensors.
                re[n] = samples[reflect(at, samples.len)] * self.window[n];
                im[n] = 0;
            }
            self.transform(&re, &im);
            var magnitude: [bins]f64 = undefined;
            for (&magnitude, 0..) |*v, b| v.* = @sqrt(re[b] * re[b] + im[b] * im[b] + 1e-9);
            for (out[k * filters ..][0..filters], 0..) |*o, m| {
                var sum: f64 = 0;
                for (magnitude, self.weights[m * bins ..][0..bins]) |v, w| sum += v * w;
                o.* = @floatCast(@log(@max(sum, 1e-5)));
            }
        }
    }

    /// In-place iterative radix-2 FFT of `fft` points.
    fn transform(self: *const Front, re: *[fft]f64, im: *[fft]f64) void {
        comptime std.debug.assert(fft == 1 << 10);
        for (0..fft) |i| {
            const j = @bitReverse(@as(u10, @intCast(i)));
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

/// PyTorch's `reflect` padding of one index into `[0, len)`: the edge
/// sample is not repeated. One reflection only; the callers bound the overhang.
fn reflect(index: i64, len: usize) usize {
    const n: i64 = @intCast(len);
    const i = if (index < 0) -index else if (index >= n) 2 * (n - 1) - index else index;
    std.debug.assert(i >= 0 and i < n);
    return @intCast(i);
}

/// Sees each stage's rows: `mel`, `blocks.0` … `blocks.N`, `mfa`, and
/// `asp` (one row, the pooled mean then standard deviation).
pub const Observer = struct {
    context: *anyopaque,
    record: *const fn (context: *anyopaque, label: []const u8, rows: []const f32) void,

    fn see(self: ?Observer, label: []const u8, rows: []const f32) void {
        if (self) |o| o.record(o.context, label, rows);
    }
};

/// A convolution: `w` is `[out][in][kernel]`, padded "same" by reflection.
const Conv = struct { w: []const f32, b: []const f32, in: usize, out: usize, kernel: usize, dilation: usize };
const Block = struct { tdnn1: Conv, res2net: []const Conv, tdnn2: Conv, se1: Conv, se2: Conv };

/// The speaker encoder's weights decoded to F32, owned by `storage`;
/// read-only once built.
pub const Encoder = struct {
    config: tts.SpeakerEncoder,
    storage: std.heap.ArenaAllocator,
    front: *const Front,
    first: Conv,
    blocks: []const Block,
    mfa: Conv,
    asp_tdnn: Conv,
    asp_conv: Conv,
    fc: Conv,
    /// F32 bytes held.
    bytes: usize,

    /// Decodes `binding` (which borrows its checkpoint only for this call).
    pub fn init(gpa: Allocator, binding: *const tts.SpeakerBinding, config: tts.SpeakerEncoder) !Encoder {
        var storage: std.heap.ArenaAllocator = .init(gpa);
        errdefer storage.deinit();
        var l: Loader = .{ .arena = storage.allocator() };
        const front = try l.arena.create(Front);
        front.init();
        const c = config;
        const blocks = try l.arena.alloc(Block, binding.blocks.len);
        for (blocks, binding.blocks, 1..) |*block, b, i| {
            const res2net = try l.arena.alloc(Conv, b.res2net.len);
            for (res2net, b.res2net) |*r, ref| r.* = try l.conv(ref, c.dilations[i]);
            block.* = .{
                .tdnn1 = try l.conv(b.tdnn1, 1),
                .res2net = res2net,
                .tdnn2 = try l.conv(b.tdnn2, 1),
                .se1 = try l.conv(b.se1, 1),
                .se2 = try l.conv(b.se2, 1),
            };
        }
        const n = c.channels.len;
        return .{
            .config = c,
            .front = front,
            .first = try l.conv(binding.first, c.dilations[0]),
            .blocks = blocks,
            .mfa = try l.conv(binding.mfa, c.dilations[n - 1]),
            .asp_tdnn = try l.conv(binding.asp_tdnn, 1),
            .asp_conv = try l.conv(binding.asp_conv, 1),
            .fc = try l.conv(binding.fc, 1),
            .bytes = l.bytes,
            .storage = storage,
        };
    }

    pub fn deinit(self: *Encoder) void {
        self.storage.deinit();
        self.* = undefined;
    }

    /// The frames a clip of `samples` must give at least: every convolution
    /// reflects at most this many rows at each end.
    fn minFrames(self: *const Encoder) usize {
        var most: usize = 0;
        for (self.config.kernels, self.config.dilations) |k, d| most = @max(most, d * (k - 1) / 2);
        return most + 1;
    }

    /// Writes the embedding of `samples` (24 kHz mono) into `out`
    /// (`config.dim`). Scratch comes from `gpa` and is freed before it returns.
    pub fn embed(self: *const Encoder, gpa: Allocator, io: std.Io, samples: []const f32, out: []f32, observer: ?Observer) Error!void {
        std.debug.assert(out.len == self.config.dim);
        const frames = Front.frameCount(samples.len);
        if (frames < self.minFrames()) return error.ClipTooShort;
        var arena_state: std.heap.ArenaAllocator = .init(gpa);
        defer arena_state.deinit();
        const arena = arena_state.allocator();

        const features = try arena.alloc(f32, frames * filters);
        try self.front.features(samples, features);
        Observer.see(observer, "mel", features);

        var x = try arena.alloc(f32, frames * self.first.out);
        try convolve(arena, io, self.first, features, frames, x, true);
        Observer.see(observer, "blocks.0", x);

        // `mfa` reads every SE-Res2Net block's output, side by side.
        const widths = self.mfa.in;
        const gathered = try arena.alloc(f32, frames * widths);
        var column: usize = 0;
        for (self.blocks, 1..) |*block, i| {
            const y = try arena.alloc(f32, frames * block.tdnn2.out);
            try seRes2Net(arena, io, block, x, frames, y);
            var label: [16]u8 = undefined;
            Observer.see(observer, std.fmt.bufPrint(&label, "blocks.{d}", .{i}) catch unreachable, y);
            for (0..frames) |t| @memcpy(gathered[t * widths + column ..][0..block.tdnn2.out], y[t * block.tdnn2.out ..][0..block.tdnn2.out]);
            column += block.tdnn2.out;
            x = y;
        }
        const aggregated = try arena.alloc(f32, frames * self.mfa.out);
        try convolve(arena, io, self.mfa, gathered, frames, aggregated, true);
        Observer.see(observer, "mfa", aggregated);

        const pooled = try arena.alloc(f32, 2 * self.mfa.out);
        try self.pool(arena, io, aggregated, frames, pooled);
        Observer.see(observer, "asp", pooled);
        try convolve(arena, io, self.fc, pooled, 1, out, false);
    }

    /// Attentive statistics pooling: `[μ_w; σ_w]` of `x` under attention
    /// weights computed from `x` and its global statistics.
    fn pool(self: *const Encoder, arena: Allocator, io: std.Io, x: []const f32, frames: usize, out: []f32) Error!void {
        const c = self.mfa.out;
        const mean = out[0..c];
        const deviation = out[c..][0..c];
        statistics(x, frames, null, mean, deviation);
        const joined = try arena.alloc(f32, frames * 3 * c);
        for (0..frames) |t| {
            const row = joined[t * 3 * c ..][0 .. 3 * c];
            @memcpy(row[0..c], x[t * c ..][0..c]);
            @memcpy(row[c..][0..c], mean);
            @memcpy(row[2 * c ..][0..c], deviation);
        }
        const hidden = try arena.alloc(f32, frames * self.asp_tdnn.out);
        try convolve(arena, io, self.asp_tdnn, joined, frames, hidden, true);
        for (hidden) |*v| v.* = std.math.tanh(v.*);
        const scores = try arena.alloc(f32, frames * c);
        try convolve(arena, io, self.asp_conv, hidden, frames, scores, false);
        // A softmax over time, per channel.
        for (0..c) |j| {
            var most: f32 = -std.math.inf(f32);
            for (0..frames) |t| most = @max(most, scores[t * c + j]);
            var sum: f64 = 0;
            for (0..frames) |t| {
                const e = @exp(scores[t * c + j] - most);
                scores[t * c + j] = e;
                sum += e;
            }
            for (0..frames) |t| scores[t * c + j] = @floatCast(scores[t * c + j] / sum);
        }
        statistics(x, frames, scores, mean, deviation);
    }
};

/// Per channel of `x` (`[frames][mean.len]`): the mean and standard
/// deviation, uniform or under `weights` (each channel's summing to 1).
fn statistics(x: []const f32, frames: usize, weights: ?[]const f32, mean: []f32, deviation: []f32) void {
    const c = mean.len;
    const uniform = 1 / @as(f64, @floatFromInt(frames));
    for (0..c) |j| {
        var m: f64 = 0;
        for (0..frames) |t| m += (if (weights) |w| w[t * c + j] else uniform) * x[t * c + j];
        var v: f64 = 0;
        for (0..frames) |t| {
            const d = x[t * c + j] - m;
            v += (if (weights) |w| w[t * c + j] else uniform) * d * d;
        }
        mean[j] = @floatCast(m);
        deviation[j] = @floatCast(@sqrt(@max(v, variance_floor)));
    }
}

/// One SE-Res2Net block: TDNN, Res2Net, TDNN, squeeze-excitation, plus the
/// block's input. `y` is `[frames][tdnn2.out]`.
fn seRes2Net(arena: Allocator, io: std.Io, block: *const Block, x: []const f32, frames: usize, y: []f32) Error!void {
    const c = block.tdnn1.out;
    const a = try arena.alloc(f32, frames * c);
    try convolve(arena, io, block.tdnn1, x, frames, a, true);
    // Part 0 passes; part 1 through `res2net[0]`; part j ≥ 2 through
    // `res2net[j − 1]` after adding part j − 1's output.
    const scale = block.res2net.len + 1;
    const p = c / scale;
    const split = try arena.alloc(f32, frames * c);
    const part = try arena.alloc(f32, frames * p);
    const previous = try arena.alloc(f32, frames * p);
    for (0..scale) |j| {
        for (0..frames) |t| @memcpy(part[t * p ..][0..p], a[t * c + j * p ..][0..p]);
        if (j >= 1) {
            if (j >= 2) for (part, previous) |*u, v| {
                u.* += v;
            };
            try convolve(arena, io, block.res2net[j - 1], part, frames, previous, true);
        } else @memcpy(previous, part);
        for (0..frames) |t| @memcpy(split[t * c + j * p ..][0..p], previous[t * p ..][0..p]);
    }
    try convolve(arena, io, block.tdnn2, split, frames, y, true);

    // Squeeze-excitation: a gate per channel from the time mean.
    const squeezed = try arena.alloc(f32, c);
    const unused = try arena.alloc(f32, c);
    statistics(y, frames, null, squeezed, unused);
    const excited = try arena.alloc(f32, block.se1.out);
    try convolve(arena, io, block.se1, squeezed, 1, excited, true);
    const gate = try arena.alloc(f32, c);
    try convolve(arena, io, block.se2, excited, 1, gate, false);
    for (gate) |*g| g.* = 1 / (1 + @exp(-g.*));
    for (0..frames) |t| for (y[t * c ..][0..c], gate, x[t * c ..][0..c]) |*v, g, r| {
        v.* = v.* * g + r;
    };
}

/// `y = conv(x) (+ ReLU)` over `frames` rows, "same" length by reflection.
fn convolve(arena: Allocator, io: std.Io, c: Conv, x: []const f32, frames: usize, y: []f32, relu: bool) Error!void {
    std.debug.assert(x.len == frames * c.in and y.len == frames * c.out);
    var taps = x;
    if (c.kernel > 1) {
        // Row t gathers, per input channel, the kernel's taps around t.
        const half = c.dilation * (c.kernel - 1) / 2;
        const width = c.in * c.kernel;
        const gathered = try arena.alloc(f32, frames * width);
        for (0..frames) |t| for (0..c.kernel) |k| {
            const source = reflect(@as(i64, @intCast(t + k * c.dilation)) - @as(i64, @intCast(half)), frames);
            for (0..c.in) |i| gathered[t * width + i * c.kernel + k] = x[source * c.in + i];
        };
        taps = gathered;
    }
    try dense.matmul(io, .{ .rows = frames, .inner = c.in * c.kernel, .outputs = c.out, .x = taps, .w = c.w, .bias = c.b }, y);
    if (relu) for (y) |*v| {
        v.* = @max(v.*, 0);
    };
}

/// Decodes binding refs into the encoder's arena.
const Loader = struct {
    arena: Allocator,
    bytes: usize = 0,

    fn floats(self: *Loader, ref: @import("../formats/safetensors.zig").Ref) ![]f32 {
        const out = try self.arena.alloc(f32, @intCast(ref.tensor.elements));
        try ref.decode(0, out);
        self.bytes += out.len * 4;
        return out;
    }

    fn conv(self: *Loader, c: tts.Conv, dilation: usize) !Conv {
        const shape = c.weight.tensor.shape;
        return .{
            .w = try self.floats(c.weight),
            .b = try self.floats(c.bias),
            .out = @intCast(shape[0]),
            .in = @intCast(shape[1]),
            .kernel = @intCast(shape[2]),
            .dilation = dilation,
        };
    }
};

// ---- tests -----------------------------------------------------------------

test "frames: (L + 768 − 1024) / 256 + 1, none when the clip cannot be reflected" {
    try std.testing.expectEqual(@as(usize, 435), Front.frameCount(111360));
    try std.testing.expectEqual(@as(usize, 1), Front.frameCount(385));
    try std.testing.expectEqual(@as(usize, 0), Front.frameCount(384));
    try std.testing.expectEqual(@as(usize, 2), Front.frameCount(512));
}

test "the front end matches a direct transform over the reflected clip, and a tone peaks in its filter" {
    const gpa = std.testing.allocator;
    const front = try gpa.create(Front);
    defer gpa.destroy(front);
    front.init();
    var samples: [2000]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(3);
    for (&samples, 0..) |*s, n| s.* = 0.3 * @as(f32, @floatCast(@sin(2 * std.math.pi * 1000 * @as(f64, @floatFromInt(n)) / sample_rate))) + 0.01 * (prng.random().float(f32) - 0.5);
    const frames = Front.frameCount(samples.len);
    const out = try gpa.alloc(f32, frames * filters);
    defer gpa.free(out);
    try front.features(&samples, out);

    // Frame 0 by a direct DFT over its reflected samples.
    var magnitude: [bins]f64 = undefined;
    for (&magnitude, 0..) |*v, b| {
        var re: f64 = 0;
        var im: f64 = 0;
        for (0..fft) |n| {
            // Frame 0 starts `pad` samples before the clip: n < pad reads x[pad − n].
            const index = if (n < pad) pad - n else n - pad;
            const x: f64 = samples[index] * front.window[n];
            const angle = -2 * std.math.pi * @as(f64, @floatFromInt(b * n)) / fft;
            re += x * @cos(angle);
            im += x * @sin(angle);
        }
        v.* = @sqrt(re * re + im * im + 1e-9);
    }
    var loudest: usize = 0;
    for (0..filters) |m| {
        var sum: f64 = 0;
        for (magnitude, front.weights[m * bins ..][0..bins]) |v, w| sum += v * w;
        try std.testing.expectApproxEqRel(@log(@max(sum, 1e-5)), @as(f64, out[m]), 1e-5);
        if (out[m] > out[loudest]) loudest = m;
    }
    // 1 kHz lies in the filter whose edges straddle it.
    const top = mel.toMel(.slaney, max_frequency);
    const centre = mel.fromMel(.slaney, top * @as(f64, @floatFromInt(loudest + 1)) / (filters + 1));
    try std.testing.expect(@abs(centre - 1000) < 120);
    try std.testing.expectError(error.ClipTooShort, front.features(samples[0..pad], out[0..0]));
}

test "a convolution reflects at both ends, as padding=\"same\" with reflect does" {
    const gpa = std.testing.allocator;
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    // One input channel, kernel 3, dilation 2: taps at t − 2, t, t + 2.
    const x = [_]f32{ 1, 2, 3, 4, 5, 6 };
    const c: Conv = .{ .w = &.{ 1, 10, 100, 2, 0, 0 }, .b = &.{ 0, -100 }, .in = 1, .out = 2, .kernel = 3, .dilation = 2 };
    var y: [12]f32 = undefined;
    try convolve(arena, std.testing.io, c, &x, 6, &y, true);
    // Rows 0 and 1 reflect t − 2 to x[2] and x[1]; rows 4 and 5 reflect t + 2 to x[4] and x[3].
    const left = [_]f32{ 3, 2, 1, 2, 3, 4 };
    const right = [_]f32{ 3, 4, 5, 6, 5, 4 };
    for (0..6) |t| {
        try std.testing.expectEqual(left[t] + 10 * x[t] + 100 * right[t], y[t * 2]);
        try std.testing.expectEqual(@max(0, 2 * left[t] - 100), y[t * 2 + 1]);
    }
}

test "the encoder runs a scaled-down Base: every stage in order, a finite embedding at the talker's width" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const json = try tts.testConfigJson(gpa, "0b6", 8, "base");
    defer gpa.free(json);
    var config = try tts.parseConfig(gpa, json, null);
    defer config.deinit();
    var file = try tts.TestFile.open(io, &config, .{});
    defer file.close(io);
    var binding = try tts.bind(gpa, &file.checkpoint, &config, null);
    defer binding.deinit(gpa);
    var encoder = try Encoder.init(gpa, &binding.speaker_encoder.?, config.speaker_encoder.?);
    defer encoder.deinit();

    var samples: [6000]f32 = undefined;
    var prng = std.Random.DefaultPrng.init(5);
    for (&samples) |*s| s.* = prng.random().float(f32) - 0.5;
    // Labels are copied: a block's is formatted into a buffer the call reuses.
    const Seen = struct {
        labels: [8][16]u8 = undefined,
        lengths: [8]usize = undefined,
        rows: [8]usize = undefined,
        count: usize = 0,
        fn record(context: *anyopaque, label: []const u8, values: []const f32) void {
            const self: *@This() = @ptrCast(@alignCast(context));
            @memcpy(self.labels[self.count][0..label.len], label);
            self.lengths[self.count] = label.len;
            self.rows[self.count] = values.len;
            self.count += 1;
        }
    };
    var seen: Seen = .{};
    var embedding: [8]f32 = undefined;
    try encoder.embed(gpa, io, &samples, &embedding, .{ .context = &seen, .record = Seen.record });
    for (embedding) |v| try std.testing.expect(std.math.isFinite(v));
    const frames = Front.frameCount(samples.len);
    const want = [_][]const u8{ "mel", "blocks.0", "blocks.1", "blocks.2", "blocks.3", "mfa", "asp" };
    const sizes = [_]usize{ frames * filters, frames * 16, frames * 16, frames * 16, frames * 16, frames * 48, 96 };
    try std.testing.expectEqual(want.len, seen.count);
    for (want, sizes, 0..) |label, size, i| {
        try std.testing.expectEqualStrings(label, seen.labels[i][0..seen.lengths[i]]);
        try std.testing.expectEqual(size, seen.rows[i]);
    }
    // The same clip gives the same embedding.
    var again: [8]f32 = undefined;
    try encoder.embed(gpa, io, &samples, &again, null);
    try std.testing.expectEqualSlices(f32, &embedding, &again);
    try std.testing.expectError(error.ClipTooShort, encoder.embed(gpa, io, samples[0..600], &again, null));
}
