//! `make test-metal`'s cases for the attention modes: a distance bias table
//! per query row and head, a sink logit per row and head, and queries that
//! arrive scaled (`scale` 1). Every body that takes them (the split pass and
//! its merge for decode and verify, the row-split chunk body, and the
//! materialized chunk body) is held to the CPU F64 reference row by row at
//! each cache precision, windowed at 1,024 and full, with the cache rows
//! after the batch poisoned. A zero table and a sink far below every score
//! reproduce the plain instantiations bit for bit.
const std = @import("std");
const inference = @import("inference");
const check_stats = @import("check_stats.zig");
const Backend = inference.metal.Backend;
const Buffer = inference.metal.Buffer;
const Precision = inference.metal.Precision;

const Body = enum { decode, verify, chunk, materialized };
const Geometry = struct { qh: usize, kvh: usize, hd: usize };
const Modes = struct { bias: bool = true, sink: bool = true, prescaled: bool = false };
/// `sink_scale` is the sinks' spread: a large one makes a sink's share of
/// the sum visible under the half bodies' looser bound.
const Case = struct { g: Geometry, body: Body, precision: Precision, window: usize = 0, total: usize, count: usize = 1, extent: usize = 1024, modes: Modes = .{}, sample: usize = 1, sink_scale: f32 = 1.5 };

const inkling: Geometry = .{ .qh = 24, .kvh = 8, .hd = 128 };
const general: Geometry = .{ .qh = 6, .kvh = 2, .hd = 96 };
const wide: Geometry = .{ .qh = 4, .kvh = 1, .hd = 288 };
const mid: Geometry = .{ .qh = 8, .kvh = 2, .hd = 256 };
const widest: Geometry = .{ .qh = 8, .kvh = 1, .hd = 512 };
const poison_rows = 64;

fn cases() [46]Case {
    var list: [46]Case = undefined;
    var n: usize = 0;
    // Inkling 2's geometry: every body, precision, and window.
    for ([_]Precision{ .f32, .f16, .q8_0 }) |precision| {
        for ([_]usize{ 0, 1024 }) |window| {
            const modes: Modes = .{ .prescaled = window != 0 };
            for ([_]Case{
                .{ .g = inkling, .body = .decode, .precision = precision, .window = window, .total = 1300, .modes = modes },
                .{ .g = inkling, .body = .verify, .precision = precision, .window = window, .total = 1300, .count = 5, .modes = modes },
                .{ .g = inkling, .body = .chunk, .precision = precision, .window = window, .total = 1300, .count = 70, .modes = modes, .sample = 3 },
                .{ .g = inkling, .body = .materialized, .precision = precision, .window = window, .total = 1300, .count = 70, .modes = modes, .sample = 3 },
            }) |case| {
                list[n] = case;
                n += 1;
            }
        }
    }
    // Every other biased split-pass instantiation: the general kernel (and
    // its wide one) at each precision, and the 256- and 512-wide head splits.
    for ([_]Precision{ .f32, .f16, .q8_0 }) |precision| {
        for ([_]Case{
            .{ .g = general, .body = .decode, .precision = precision, .total = 300, .extent = 16 },
            .{ .g = wide, .body = .decode, .precision = precision, .window = 1024, .total = 1100 },
            .{ .g = mid, .body = .decode, .precision = precision, .total = 600, .modes = .{ .prescaled = true } },
            .{ .g = widest, .body = .decode, .precision = precision, .total = 400 },
        }) |case| {
            list[n] = case;
            n += 1;
        }
    }
    // Verify on the general kernel (its own row arithmetic), one mode at a
    // time, a short extent, a sink that weighs in the half bodies, and the
    // first position.
    for ([_]Case{
        .{ .g = general, .body = .verify, .precision = .f32, .window = 1024, .total = 1100, .count = 16 },
        .{ .g = general, .body = .verify, .precision = .f16, .window = 1024, .total = 1100, .count = 16, .modes = .{ .bias = false } },
        .{ .g = general, .body = .chunk, .precision = .f32, .total = 200, .count = 200, .extent = 16, .sample = 7 },
        .{ .g = wide, .body = .chunk, .precision = .f32, .total = 100, .count = 40, .modes = .{ .sink = false } },
        .{ .g = inkling, .body = .chunk, .precision = .f16, .total = 60, .count = 40, .modes = .{ .bias = false }, .sink_scale = 4 },
        .{ .g = inkling, .body = .chunk, .precision = .q8_0, .total = 60, .count = 40, .sink_scale = 4 },
        .{ .g = mid, .body = .materialized, .precision = .f16, .window = 1024, .total = 1200, .count = 100, .sample = 9 },
        .{ .g = general, .body = .materialized, .precision = .f32, .total = 100, .count = 40, .modes = .{ .bias = false }, .sink_scale = 4 },
        .{ .g = general, .body = .materialized, .precision = .f16, .total = 100, .count = 40, .modes = .{ .sink = false } },
        .{ .g = inkling, .body = .decode, .precision = .f32, .total = 1 },
    }) |case| {
        list[n] = case;
        n += 1;
    }
    std.debug.assert(n == list.len);
    return list;
}

/// The cache as one body reads it, and the F32 rows its reference reads.
const Cache = struct {
    keys: Buffer,
    values: Buffer,
    /// What the body is told the cache holds: a Q8_0 cache's chunk bodies
    /// read its F16 view.
    precision: Precision,
    ref_keys: []f32,
    ref_values: []f32,
    owned: [6]?Buffer = @splat(null),

    fn deinit(self: *Cache, alloc: std.mem.Allocator, b: *Backend) void {
        for (self.owned) |o| if (o) |x| b.release(x) catch {};
        alloc.free(self.ref_keys);
        alloc.free(self.ref_values);
    }
};

fn halves(buffer: Buffer, out: []f32) void {
    const h = @as([*]const f16, @ptrCast(@alignCast(buffer.host)))[0..out.len];
    for (h, out) |x, *f| f.* = x;
}

fn makeCache(alloc: std.mem.Allocator, b: *Backend, random: std.Random, case: Case) !Cache {
    const kvw = case.g.kvh * case.g.hd;
    const n = (case.total + poison_rows) * kvw;
    const k32 = try b.create(n * 4);
    const v32 = try b.create(n * 4);
    var c: Cache = .{ .keys = k32, .values = v32, .precision = case.precision, .ref_keys = try alloc.alloc(f32, n), .ref_values = try alloc.alloc(f32, n) };
    c.owned[0] = k32;
    c.owned[1] = v32;
    for (k32.floats()[0 .. case.total * kvw]) |*x| x.* = random.floatNorm(f32) * 0.5;
    for (v32.floats()[0 .. case.total * kvw]) |*x| x.* = random.floatNorm(f32) * 0.5;
    // A future row read by mistake moves the output far past any bound.
    const poison: f32 = if (case.precision == .f32) 1e30 else 6e4;
    @memset(k32.floats()[case.total * kvw ..], poison);
    @memset(v32.floats()[case.total * kvw ..], poison);
    switch (case.precision) {
        .f32 => {
            @memcpy(c.ref_keys, k32.floats()[0..n]);
            @memcpy(c.ref_values, v32.floats()[0..n]);
        },
        .f16 => {
            c.keys = try b.create(n * 2);
            c.values = try b.create(n * 2);
            c.owned[2] = c.keys;
            c.owned[3] = c.values;
            try b.begin();
            try b.packHalf(&.{ .{ .dst = c.keys, .src = k32, .count = n }, .{ .dst = c.values, .src = v32, .count = n } });
            try b.commit();
            halves(c.keys, c.ref_keys);
            halves(c.values, c.ref_values);
        },
        .q8_0 => {
            const bytes = Precision.q8_0.bytes(n);
            c.keys = try b.create(bytes);
            c.values = try b.create(bytes);
            c.owned[2] = c.keys;
            c.owned[3] = c.values;
            try b.begin();
            try b.packQ8(&.{ .{ .dst = c.keys, .src = k32, .count = n }, .{ .dst = c.values, .src = v32, .count = n } });
            try b.commit();
            if (case.body == .decode or case.body == .verify) {
                try inference.quant.row(8, c.keys.host[0..bytes], c.ref_keys);
                try inference.quant.row(8, c.values.host[0..bytes], c.ref_values);
            } else {
                // The plan's route for a chunk: the F16 view.
                const kh = try b.create(n * 2);
                const vh = try b.create(n * 2);
                c.owned[4] = kh;
                c.owned[5] = vh;
                try b.begin();
                try b.unpackQ8(&.{ .{ .dst = kh, .src = c.keys, .count = n }, .{ .dst = vh, .src = c.values, .count = n } });
                try b.commit();
                halves(kh, c.ref_keys);
                halves(vh, c.ref_values);
                c.keys = kh;
                c.values = vh;
                c.precision = .f16;
            }
        },
    }
    return c;
}

/// Runs `case.body` into `out`; `modes` false runs the plain body.
fn dispatch(b: *Backend, case: Case, cache: Cache, queries: Buffer, out: Buffer, scratch: Buffer, partials: Buffer, scale: f32, bias: ?Backend.Bias, sinks: ?Buffer) !void {
    const g = case.g;
    const qw = g.qh * g.hd;
    const kv_bytes = cache.precision.bytes(g.kvh * g.hd);
    const position = case.total - case.count;
    const chunk: Backend.AttentionChunkShape = .{ .query_heads = g.qh, .kv_heads = g.kvh, .key_width = g.hd, .value_width = g.hd, .position = position, .count = case.count, .q_stride = qw, .out_stride = qw, .scale = scale, .precision = cache.precision, .window = case.window, .bias = bias, .sinks = sinks };
    try b.begin();
    switch (case.body) {
        .decode => {
            // The plan slices a window off the cache: row 0 is the earliest key.
            const lo = if (case.window != 0 and case.total > case.window) case.total - case.window else 0;
            const visible = case.total - lo;
            try b.attentionDecode(cache.keys.slice(lo * kv_bytes, visible * kv_bytes), cache.values.slice(lo * kv_bytes, visible * kv_bytes), queries, partials, out, .{ .query_heads = g.qh, .kv_heads = g.kvh, .key_width = g.hd, .value_width = g.hd, .visible = visible, .scale = scale, .precision = cache.precision, .bias = bias, .sinks = sinks });
        },
        .verify => try b.attentionVerify(cache.keys, cache.values, queries, partials, out, chunk),
        .chunk => try b.attentionChunk(cache.keys, cache.values, queries, out, chunk),
        .materialized => try b.attentionChunkMaterialized(cache.keys, cache.values, queries, out, scratch, chunk),
    }
    try b.commit();
}

fn bound(case: Case, cache: Cache, expected: f32) f32 {
    // The row-split body multiplies half queries and probabilities.
    if (case.body == .chunk and cache.precision == .f16) return 2e-3;
    return 2e-5 * @max(1, @abs(expected));
}

fn runCase(alloc: std.mem.Allocator, b: *Backend, random: std.Random, case: Case, worst: *[4]check_stats.Worst) !void {
    const g = case.g;
    const qw = g.qh * g.hd;
    const kvw = g.kvh * g.hd;
    const position = case.total - case.count;
    const rows = Backend.attentionChunkRows(case.count);
    var cache = try makeCache(alloc, b, random, case);
    defer cache.deinit(alloc, b);
    const head_scale = 1.0 / @sqrt(@as(f32, @floatFromInt(g.hd)));
    const scale: f32 = if (case.modes.prescaled) 1 else head_scale;
    // Queries as the body reads them: F32, or half for the row-split body
    // over an F16 cache; `ref_q` is what the reference multiplies.
    const q32 = try b.create(rows * qw * 4);
    defer b.release(q32) catch {};
    for (q32.floats()) |*x| x.* = random.floatNorm(f32) * 0.5 * (if (case.modes.prescaled) head_scale else 1);
    const ref_q = try alloc.alloc(f32, rows * qw);
    defer alloc.free(ref_q);
    var queries = q32;
    var q16: ?Buffer = null;
    defer if (q16) |x| b.release(x) catch {};
    if (case.body == .chunk and cache.precision == .f16) {
        q16 = try b.create(rows * qw * 2);
        try b.begin();
        try b.packHalf(&.{.{ .dst = q16.?, .src = q32, .count = rows * qw }});
        try b.commit();
        halves(q16.?, ref_q);
        queries = q16.?;
    } else @memcpy(ref_q, q32.floats());
    const table = try b.create(@max(1, case.count * g.qh * case.extent) * 4);
    defer b.release(table) catch {};
    for (table.floats()) |*x| x.* = random.floatNorm(f32);
    const sink_buffer = try b.create(case.count * g.qh * 4);
    defer b.release(sink_buffer) catch {};
    for (sink_buffer.floats()) |*x| x.* = random.floatNorm(f32) * case.sink_scale;
    const bias: ?Backend.Bias = if (case.modes.bias) .{ .table = table, .extent = case.extent } else null;
    const sinks: ?Buffer = if (case.modes.sink) sink_buffer else null;

    const out = try b.create(rows * qw * 4);
    defer b.release(out) catch {};
    @memset(out.floats(), std.math.nan(f32));
    const chunk_shape: Backend.AttentionChunkShape = .{ .query_heads = g.qh, .kv_heads = g.kvh, .key_width = g.hd, .value_width = g.hd, .position = position, .count = case.count, .q_stride = qw, .out_stride = qw, .scale = scale, .precision = cache.precision, .window = case.window };
    const strip_floats = Backend.attentionChunkStripFloats(g.qh, @divCeil(rows, 32), Backend.attentionStripKeys(chunk_shape, @divCeil(rows, 32)));
    const scratch = try b.create(if (case.body == .materialized) strip_floats * 4 else 64);
    defer b.release(scratch) catch {};
    const partials = try b.create(Backend.attentionVerifyPartials(@min(case.count, Backend.verify_max_rows), g.qh, g.hd) * 4);
    defer b.release(partials) catch {};
    try dispatch(b, case, cache, queries, out, scratch, partials, scale, bias, sinks);

    const scratch64 = try alloc.alloc(f64, case.total);
    defer alloc.free(scratch64);
    const expected = try alloc.alloc(f32, qw);
    defer alloc.free(expected);
    var r: usize = 0;
    while (r < case.count) : (r += case.sample) {
        const end = position + r + 1;
        const lo = if (case.window != 0 and end > case.window) end - case.window else 0;
        const visible = end - lo;
        try inference.cpu.attention.apply(.{
            .query_heads = g.qh,
            .kv_heads = g.kvh,
            .key_width = g.hd,
            .value_width = g.hd,
            .tokens = visible,
            .visible_tokens = visible,
            .scale = scale,
            .queries = ref_q[r * qw ..][0..qw],
            .keys = cache.ref_keys[lo * kvw .. end * kvw],
            .values = cache.ref_values[lo * kvw .. end * kvw],
            .bias = if (case.modes.bias) .{ .table = table.floats()[r * g.qh * case.extent ..][0 .. g.qh * case.extent], .extent = case.extent } else null,
            .sinks = if (case.modes.sink) sink_buffer.floats()[r * g.qh ..][0..g.qh] else null,
        }, expected, scratch64);
        for (expected, out.floats()[r * qw ..][0..qw]) |e, a| {
            worst[@backingInt(case.body)].observe(@abs(a - e));
            if (!(@abs(a - e) <= bound(case, cache, e))) {
                std.debug.print("attention modes, {s} {s}: got {d} want {d} (geometry {d}/{d}/{d}, total {d}, count {d}, window {d}, extent {d}, bias {}, sink {}, prescaled {}, row {d})\n", .{ @tagName(case.body), @tagName(case.precision), a, e, g.qh, g.kvh, g.hd, case.total, case.count, case.window, case.extent, case.modes.bias, case.modes.sink, case.modes.prescaled, r });
                return error.MetalMismatch;
            }
        }
    }

    // A zero table and a sink below every score leave the plain body's bits.
    if (case.g.hd == inkling.hd and case.precision == .f16 and case.window != 0) {
        @memset(table.floats(), 0);
        @memset(sink_buffer.floats(), -1e30);
        const plain = try b.create(rows * qw * 4);
        defer b.release(plain) catch {};
        @memset(plain.floats(), std.math.nan(f32));
        @memset(out.floats(), std.math.nan(f32));
        try dispatch(b, case, cache, queries, plain, scratch, partials, scale, null, null);
        try dispatch(b, case, cache, queries, out, scratch, partials, scale, .{ .table = table, .extent = case.extent }, sink_buffer);
        const n = (case.count - 1) * qw + qw;
        if (!std.mem.eql(u8, std.mem.sliceAsBytes(plain.floats()[0..n]), std.mem.sliceAsBytes(out.floats()[0..n]))) {
            std.debug.print("attention modes, {s}: a zero table and a -1e30 sink differ from the plain body\n", .{@tagName(case.body)});
            return error.MetalMismatch;
        }
    }
}

/// The refusals: the three-pass and register-reuse bodies take no mode, a
/// mode takes no span, and a table or sink buffer must cover every row.
fn checkRefusals(b: *Backend) !void {
    const any = try b.create(1 << 20);
    defer b.release(any) catch {};
    const short = any.slice(0, 64);
    const table: Backend.Bias = .{ .table = any, .extent = 16 };
    try std.testing.expectError(error.InvalidShape, b.attention(any, any, any, any, any, .{ .query_heads = 2, .kv_heads = 1, .key_width = 64, .value_width = 64, .visible = 8, .scale = 1, .bias = table }));
    try std.testing.expectError(error.InvalidShape, b.attention(any, any, any, any, any, .{ .query_heads = 2, .kv_heads = 1, .key_width = 64, .value_width = 64, .visible = 8, .scale = 1, .sinks = any }));
    const shape: Backend.AttentionChunkShape = .{ .query_heads = 2, .kv_heads = 1, .key_width = 256, .value_width = 256, .position = 0, .count = 8, .q_stride = 512, .out_stride = 512, .scale = 1, .bias = table, .sinks = any };
    try std.testing.expectError(error.InvalidShape, b.attentionChunkReuse(any, any, any, any, shape));
    var spanned = shape;
    spanned.span = .{ .begin = 0, .end = 8 };
    try std.testing.expectError(error.InvalidShape, b.attentionChunk(any, any, any, any, spanned));
    var short_table = shape;
    short_table.bias = .{ .table = short, .extent = 16 };
    try std.testing.expectError(error.InvalidShape, b.attentionChunk(any, any, any, any, short_table));
    var empty_extent = shape;
    empty_extent.bias = .{ .table = any, .extent = 0 };
    try std.testing.expectError(error.InvalidShape, b.attentionChunk(any, any, any, any, empty_extent));
    var short_sinks = shape;
    short_sinks.sinks = any.slice(0, 8);
    try std.testing.expectError(error.InvalidShape, b.attentionChunk(any, any, any, any, short_sinks));
    try std.testing.expectError(error.InvalidShape, b.attentionDecode(any, any, any, any, any, .{ .query_heads = 2, .kv_heads = 1, .key_width = 64, .value_width = 64, .visible = 8, .scale = 1, .sinks = any.slice(0, 4) }));
}

pub fn check(alloc: std.mem.Allocator, b: *Backend) !void {
    var prng = std.Random.DefaultPrng.init(0x51ce);
    const random = prng.random();
    var worst: [4]check_stats.Worst = @splat(.{});
    for (cases()) |case| try runCase(alloc, b, random, case, &worst);
    try checkRefusals(b);
    std.debug.print("Attention modes (distance bias, sinks, scaled queries) vs CPU F64 per row, F32/F16/Q8_0 caches, windows 1,024 and full, future rows poisoned: decode {e:.3}, verify {e:.3}, materialized {e:.3} (bound 2e-5, relative past 1); row-split chunk {e:.3} (2e-3 over half queries); a zero table and a -1e30 sink equal the plain bodies bit for bit\n", .{ worst[0].shown(), worst[1].shown(), worst[3].shown(), worst[2].shown() });
}
