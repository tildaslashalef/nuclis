//! Counts what a mixture of experts' routing asks of a batch, from the
//! choices the plan reports through `inference.observer.Routing`
//! (`bench --expert-counts`): how many distinct experts a batch of R rows
//! touches, how a prefill chunk's rows spread over the experts, and how many
//! of a token's experts the next token reuses. Pure aggregation; the
//! reading of the numbers is in docs/models/gemma4.md § What the experts cost.
const std = @import("std");
const inference = @import("inference");

const Routing = inference.observer.Routing;

/// Batch sizes the window and verify tallies cover: one row to a full
/// verify batch of the longest draft.
pub const max_rows = 16;
/// The gathered prefill tiles' row count (`matmulExperts`): a list shorter
/// than this leaves part of its tile idle.
pub const tile_rows = 32;
/// Most experts a configuration may have for these tallies.
pub const max_experts = 256;

/// List-length buckets: 1, 2, 3–4, 5–8, 9–16, 17–31, 32 or more rows.
pub const bucket_bounds = [_]usize{ 1, 2, 4, 8, 16, 31 };

pub const Counts = struct {
    experts: usize = 0,
    k: usize = 0,
    /// Distinct experts over windows of R consecutive prompt rows (index
    /// R - 1), summed over windows and layers, and the windows counted: what
    /// a verify batch of R accepted drafts would touch.
    window_distinct: [max_rows]u64 = @splat(0),
    windows: [max_rows]u64 = @splat(0),
    /// The same over real verify batches, by their row count.
    verify_distinct: [max_rows]u64 = @splat(0),
    verify_batches: [max_rows]u64 = @splat(0),
    /// Prefill chunks: distinct experts and rows summed over chunk-layers,
    /// the non-empty lists, those shorter than a tile, and the lists by
    /// length bucket (`bucket_bounds`, then the rest).
    chunk_layers: u64 = 0,
    chunk_rows: u64 = 0,
    chunk_distinct: u64 = 0,
    lists: u64 = 0,
    short_lists: u64 = 0,
    list_buckets: [bucket_bounds.len + 1]u64 = @splat(0),
    /// Experts a token shares with the token before it, summed, over the
    /// consecutive pairs counted: prompt rows within a chunk, and decode steps.
    prompt_shared: u64 = 0,
    prompt_pairs: u64 = 0,
    decode_shared: u64 = 0,
    decode_pairs: u64 = 0,
    /// Each layer's last decode routing (`k` ids), for the next step's overlap.
    previous: [max_layers][max_k]u32 = undefined,
    has_previous: [max_layers]bool = @splat(false),

    pub const max_layers = 128;
    pub const max_k = 16;

    pub fn add(self: *Counts, r: Routing) error{InvalidShape}!void {
        if (r.k == 0 or r.k > max_k or r.experts == 0 or r.experts > max_experts or r.layer >= max_layers) return error.InvalidShape;
        if (r.indices.len != r.rows * r.k) return error.InvalidShape;
        for (r.indices) |id| if (id >= r.experts) return error.InvalidShape;
        self.experts = r.experts;
        self.k = r.k;
        switch (r.batch) {
            .prefill => {
                self.addWindows(r);
                self.addChunk(r);
                for (1..r.rows) |row| {
                    self.prompt_shared += shared(r.indices[(row - 1) * r.k ..][0..r.k], r.indices[row * r.k ..][0..r.k]);
                    self.prompt_pairs += 1;
                }
            },
            .verify => if (r.rows <= max_rows) {
                self.verify_distinct[r.rows - 1] += distinct(r.indices);
                self.verify_batches[r.rows - 1] += 1;
            },
            .decode => {
                const now = r.indices[0..r.k];
                if (self.has_previous[r.layer]) {
                    self.decode_shared += shared(self.previous[r.layer][0..r.k], now);
                    self.decode_pairs += 1;
                }
                @memcpy(self.previous[r.layer][0..r.k], now);
                self.has_previous[r.layer] = true;
            },
        }
    }

    fn addWindows(self: *Counts, r: Routing) void {
        for (1..@min(r.rows, max_rows) + 1) |w| {
            var start: usize = 0;
            while (start + w <= r.rows) : (start += 1) {
                self.window_distinct[w - 1] += distinct(r.indices[start * r.k ..][0 .. w * r.k]);
                self.windows[w - 1] += 1;
            }
        }
    }

    fn addChunk(self: *Counts, r: Routing) void {
        var per: [max_experts]u32 = @splat(0);
        for (r.indices) |id| per[id] += 1;
        self.chunk_layers += 1;
        self.chunk_rows += r.rows;
        for (per[0..r.experts]) |n| {
            if (n == 0) continue;
            self.chunk_distinct += 1;
            self.lists += 1;
            if (n < tile_rows) self.short_lists += 1;
            self.list_buckets[bucket(n)] += 1;
        }
    }

    /// The tallies as means, for the report.
    pub fn summary(self: *const Counts) Summary {
        var out: Summary = .{ .experts = self.experts, .k = self.k };
        for (0..max_rows) |i| {
            out.window_mean[i] = ratio(self.window_distinct[i], self.windows[i]);
            out.verify_mean[i] = ratio(self.verify_distinct[i], self.verify_batches[i]);
        }
        out.verify_batches = self.verify_batches;
        out.chunk_mean_rows = ratio(self.chunk_rows, self.chunk_layers);
        out.chunk_mean_distinct = ratio(self.chunk_distinct, self.chunk_layers);
        out.mean_list_rows = ratio(self.chunk_rows * self.k, self.lists);
        out.short_list_share = ratio(self.short_lists, self.lists);
        out.list_buckets = self.list_buckets;
        out.prompt_reuse = ratio(self.prompt_shared, self.prompt_pairs);
        out.decode_reuse = ratio(self.decode_shared, self.decode_pairs);
        return out;
    }
};

/// What `bench --expert-counts --json` writes: means over layers and batches.
pub const Summary = struct {
    experts: usize,
    k: usize,
    /// Index R - 1: mean distinct experts over R consecutive prompt rows, and
    /// over real verify batches of R rows (null where none ran).
    window_mean: [max_rows]?f64 = @splat(null),
    verify_mean: [max_rows]?f64 = @splat(null),
    verify_batches: [max_rows]u64 = @splat(0),
    chunk_mean_rows: ?f64 = null,
    chunk_mean_distinct: ?f64 = null,
    /// Rows per non-empty expert list, and the share of lists under a tile.
    mean_list_rows: ?f64 = null,
    short_list_share: ?f64 = null,
    list_buckets: [bucket_bounds.len + 1]u64 = @splat(0),
    /// Mean experts of `k` the next token shares, in a prompt and in decode.
    prompt_reuse: ?f64 = null,
    decode_reuse: ?f64 = null,

    pub fn render(self: Summary, out: *std.Io.Writer) !void {
        try out.print("Experts: {d} of {d} per token\n", .{ self.k, self.experts });
        try out.writeAll("  rows R   distinct over R prompt rows   ratio to k   ratio to kR   verify batches (distinct)\n");
        for (0..max_rows) |i| {
            const w = self.window_mean[i] orelse continue;
            const r: f64 = @floatFromInt(i + 1);
            const k: f64 = @floatFromInt(self.k);
            try out.print("  {d:>6}   {d:>27.1}   {d:>10.2}   {d:>11.2}", .{ i + 1, w, w / k, w / @min(k * r, @as(f64, @floatFromInt(self.experts))) });
            if (self.verify_mean[i]) |v| try out.print("   {d} ({d:.1})\n", .{ self.verify_batches[i], v }) else try out.writeAll("   —\n");
        }
        if (self.chunk_mean_rows) |rows| try out.print("  prefill chunk: {d:.0} rows, {d:.1} distinct experts, {d:.1} rows per list, {d:.0} % of lists under {d} rows\n", .{ rows, self.chunk_mean_distinct.?, self.mean_list_rows.?, self.short_list_share.? * 100, tile_rows });
        try out.writeAll("  list lengths: ");
        const labels = [_][]const u8{ "1", "2", "3-4", "5-8", "9-16", "17-31", "32+" };
        for (labels, self.list_buckets) |label, n| try out.print("{s}: {d}  ", .{ label, n });
        try out.writeAll("\n");
        if (self.prompt_reuse) |p| try out.print("  next token shares {d:.2} of {d} experts in a prompt", .{ p, self.k });
        if (self.decode_reuse) |d| try out.print(", {d:.2} in decode", .{d});
        try out.writeAll("\n");
    }
};

fn ratio(n: u64, d: u64) ?f64 {
    if (d == 0) return null;
    return @as(f64, @floatFromInt(n)) / @as(f64, @floatFromInt(d));
}

fn bucket(n: usize) usize {
    for (bucket_bounds, 0..) |bound, i| if (n <= bound) return i;
    return bucket_bounds.len;
}

/// Distinct ids in `ids`, each below `max_experts`.
fn distinct(ids: []const u32) u64 {
    var seen: std.StaticBitSet(max_experts) = .empty;
    for (ids) |id| seen.set(id);
    return seen.count();
}

/// Ids of `b` that appear in `a`.
fn shared(a: []const u32, b: []const u32) u64 {
    var seen: std.StaticBitSet(max_experts) = .empty;
    for (a) |id| seen.set(id);
    var n: u64 = 0;
    for (b) |id| {
        if (seen.isSet(id)) n += 1;
    }
    return n;
}

test "windows, chunk lists, and reuse count a prompt's routing" {
    var c: Counts = .{};
    // Three rows, k = 2, of 8 experts: {0,1}, {1,2}, {5,6}.
    try c.add(.{ .batch = .prefill, .layer = 0, .rows = 3, .k = 2, .experts = 8, .indices = &.{ 0, 1, 1, 2, 5, 6 } });
    const s = c.summary();
    try std.testing.expectEqual(@as(?f64, 2), s.window_mean[0]);
    // Windows of two: {0,1,2} and {1,2,5,6}.
    try std.testing.expectEqual(@as(?f64, 3.5), s.window_mean[1]);
    try std.testing.expectEqual(@as(?f64, 5), s.window_mean[2]);
    try std.testing.expectEqual(@as(?f64, null), s.window_mean[3]);
    // Five lists (0, 1 twice, 2, 5, 6), all under a tile.
    try std.testing.expectEqual(@as(?f64, 5), s.chunk_mean_distinct);
    try std.testing.expectEqual(@as(?f64, 1), s.short_list_share);
    try std.testing.expectEqual(@as(u64, 4), s.list_buckets[0]);
    try std.testing.expectEqual(@as(u64, 1), s.list_buckets[1]);
    // Pairs share one expert, then none.
    try std.testing.expectEqual(@as(?f64, 0.5), s.prompt_reuse);
}

test "verify batches and decode steps are tallied apart" {
    var c: Counts = .{};
    try c.add(.{ .batch = .verify, .layer = 0, .rows = 2, .k = 2, .experts = 8, .indices = &.{ 3, 4, 4, 7 } });
    try c.add(.{ .batch = .decode, .layer = 1, .rows = 1, .k = 2, .experts = 8, .indices = &.{ 1, 2 } });
    try c.add(.{ .batch = .decode, .layer = 1, .rows = 1, .k = 2, .experts = 8, .indices = &.{ 2, 3 } });
    const s = c.summary();
    try std.testing.expectEqual(@as(?f64, 3), s.verify_mean[1]);
    try std.testing.expectEqual(@as(u64, 1), s.verify_batches[1]);
    try std.testing.expectEqual(@as(?f64, 1), s.decode_reuse);
    try std.testing.expectEqual(@as(?f64, null), s.window_mean[0]);
}

test "an id past the expert count is refused" {
    var c: Counts = .{};
    try std.testing.expectError(error.InvalidShape, c.add(.{ .batch = .decode, .layer = 0, .rows = 1, .k = 1, .experts = 4, .indices = &.{4} }));
    try std.testing.expectError(error.InvalidShape, c.add(.{ .batch = .decode, .layer = 0, .rows = 2, .k = 1, .experts = 4, .indices = &.{1} }));
}
