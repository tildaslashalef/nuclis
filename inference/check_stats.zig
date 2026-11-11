//! NaN-aware aggregation for the check executables beside it.
//!
//! `@max(worst, d)` returns the operand that is not NaN, and `d > bound` is
//! false for NaN, so a plain running maximum or a plain comparison lets a
//! NaN or infinite output pass a check. Every difference and cosine a check
//! aggregates goes through `Worst` or `Least`, which latch a non-finite
//! observation and fail `within`/`atLeast` for it; a lone comparison against
//! a bound uses `exceeds`/`below`, which are true for NaN. Each counts its
//! observations, so a caller can also fail on comparing nothing.
const std = @import("std");

/// The largest magnitude observed; any NaN or infinity latches `non_finite`.
pub const Worst = struct {
    value: f64 = 0,
    non_finite: bool = false,
    count: usize = 0,

    /// Records |d|.
    pub fn observe(self: *Worst, d: f64) void {
        self.count += 1;
        if (!std.math.isFinite(d)) {
            self.non_finite = true;
            return;
        }
        self.value = @max(self.value, @abs(d));
    }

    /// Folds another aggregate into this one.
    pub fn merge(self: *Worst, other: Worst) void {
        self.count += other.count;
        self.non_finite = self.non_finite or other.non_finite;
        self.value = @max(self.value, other.value);
    }

    /// True only when every observation was finite and none exceeded `bound`.
    pub fn within(self: Worst, bound: f64) bool {
        return !self.non_finite and self.value <= bound;
    }

    /// The value to print: NaN when a non-finite value was observed.
    pub fn shown(self: Worst) f64 {
        return if (self.non_finite) std.math.nan(f64) else self.value;
    }
};

/// The smallest value observed (a cosine, a margin); NaN or infinity latches
/// `non_finite`. With no observation `value` is +inf.
pub const Least = struct {
    value: f64 = std.math.inf(f64),
    non_finite: bool = false,
    count: usize = 0,

    pub fn observe(self: *Least, x: f64) void {
        self.count += 1;
        if (!std.math.isFinite(x)) {
            self.non_finite = true;
            return;
        }
        self.value = @min(self.value, x);
    }

    pub fn merge(self: *Least, other: Least) void {
        self.count += other.count;
        self.non_finite = self.non_finite or other.non_finite;
        self.value = @min(self.value, other.value);
    }

    /// True only when every observation was finite and none fell below `floor`.
    pub fn atLeast(self: Least, floor: f64) bool {
        return !self.non_finite and self.value >= floor;
    }

    /// The value to print: NaN when a non-finite value was observed.
    pub fn shown(self: Least) f64 {
        return if (self.non_finite) std.math.nan(f64) else self.value;
    }
};

/// True when `x` is over `bound` or is NaN.
pub fn exceeds(x: f64, bound: f64) bool {
    return !(x <= bound);
}

/// True when `x` is under `floor` or is NaN.
pub fn below(x: f64, floor: f64) bool {
    return !(x >= floor);
}

const nan = std.math.nan(f64);
const inf = std.math.inf(f64);

test "Worst keeps the largest magnitude and latches NaN and inf" {
    var w: Worst = .{};
    try std.testing.expect(w.within(0));
    w.observe(-3e-6);
    w.observe(1e-6);
    try std.testing.expectEqual(@as(f64, 3e-6), w.value);
    try std.testing.expect(w.within(1e-5));
    try std.testing.expect(!w.within(1e-6));

    var n: Worst = .{};
    n.observe(1e-9);
    n.observe(nan);
    n.observe(1e-9);
    try std.testing.expect(!n.within(1e30));
    try std.testing.expect(std.math.isNan(n.shown()));
    try std.testing.expectEqual(@as(usize, 3), n.count);

    var i: Worst = .{};
    i.observe(-inf);
    try std.testing.expect(!i.within(inf));

    var merged: Worst = .{};
    merged.observe(2);
    merged.merge(n);
    try std.testing.expect(!merged.within(10));
    try std.testing.expectEqual(@as(usize, 4), merged.count);
}

test "Worst observes f32 values" {
    var w: Worst = .{};
    const x: f32 = std.math.nan(f32);
    w.observe(x);
    try std.testing.expect(w.non_finite);
}

test "Least keeps the smallest value and latches NaN and inf" {
    var l: Least = .{};
    try std.testing.expectEqual(@as(usize, 0), l.count);
    l.observe(0.999);
    l.observe(0.98);
    try std.testing.expectEqual(@as(f64, 0.98), l.value);
    try std.testing.expect(l.atLeast(0.95));
    try std.testing.expect(!l.atLeast(0.99));

    var n: Least = .{};
    n.observe(0.999);
    n.observe(nan);
    try std.testing.expect(!n.atLeast(-1));
    try std.testing.expect(std.math.isNan(n.shown()));

    var i: Least = .{};
    i.observe(inf);
    try std.testing.expect(!i.atLeast(0));

    var merged: Least = .{};
    merged.observe(0.5);
    merged.merge(i);
    try std.testing.expect(!merged.atLeast(0));
}

test "exceeds and below are true for NaN" {
    try std.testing.expect(exceeds(nan, 1));
    try std.testing.expect(exceeds(inf, 1));
    try std.testing.expect(!exceeds(1, 1));
    try std.testing.expect(exceeds(1.5, 1));
    try std.testing.expect(below(nan, 0));
    try std.testing.expect(below(-inf, 0));
    try std.testing.expect(!below(0, 0));
}
