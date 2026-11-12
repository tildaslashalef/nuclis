//! An in-place iterative radix-2 FFT in F64 for the log-mel front ends
//! (mel.zig's 512 points, ecapa.zig's 1024): bit-reversal, then butterflies
//! over a table of twiddles built once. The order of every sum is fixed by
//! `n`, so a front end's features stay bit for bit what they were.
const std = @import("std");

/// The transform of `n` points, `n` a power of two.
pub fn Fft(comptime n: usize) type {
    comptime std.debug.assert(n >= 2 and std.math.isPowerOfTwo(n));
    const Index = @Int(.unsigned, std.math.log2_int(usize, n));
    return struct {
        const Self = @This();

        /// `exp(−2πik/n)` for k < n/2.
        twiddles: [n / 2][2]f64,

        pub fn init(self: *Self) void {
            for (&self.twiddles, 0..) |*t, k| {
                const angle = -2 * std.math.pi * @as(f64, @floatFromInt(k)) / n;
                t.* = .{ @cos(angle), @sin(angle) };
            }
        }

        pub fn transform(self: *const Self, re: *[n]f64, im: *[n]f64) void {
            for (0..n) |i| {
                const j = @bitReverse(@as(Index, @intCast(i)));
                if (j > i) {
                    std.mem.swap(f64, &re[i], &re[j]);
                    std.mem.swap(f64, &im[i], &im[j]);
                }
            }
            var size: usize = 2;
            while (size <= n) : (size *= 2) {
                const half = size / 2;
                const step = n / size;
                var start: usize = 0;
                while (start < n) : (start += size) {
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
}
