//! `make test-metal`'s cases for Qwen3-TTS's kernels (`kernels/tts.metal`),
//! each against the CPU function it replaces (audio/qwen3_codec.zig) or an
//! F64 reference, and `--tts-bench`, the BF16 matvec's bandwidth against
//! the generic matvec on the talker's and predictor's shapes.
const std = @import("std");
const inference = @import("inference");
const Backend = inference.metal.Backend;
const Buffer = inference.metal.Buffer;
const codec = inference.audio.qwen3_codec;

fn upload(b: *Backend, values: []const f32) !Buffer {
    const buffer = try b.create(values.len * 4);
    @memcpy(buffer.floats()[0..values.len], values);
    return buffer;
}

fn fill(random: std.Random, values: []f32, scale: f32) void {
    for (values) |*v| v.* = (random.float(f32) * 2 - 1) * scale;
}

/// Largest |got − want| over max(|want|, floor).
fn worst(got: []const f32, want: []const f32, floor: f32) f32 {
    var w: f32 = 0;
    for (got, want) |g, x| w = @max(w, @abs(g - x) / @max(@abs(x), floor));
    return w;
}

fn expectWithin(name: []const u8, got: []const f32, want: []const f32, floor: f32, bound: f32) !f32 {
    const w = worst(got, want, floor);
    if (!(w <= bound)) {
        std.debug.print("{s}: worst relative difference {e} over the bound {e}\n", .{ name, w, bound });
        return error.MetalMismatch;
    }
    return w;
}

fn expectEqual(name: []const u8, got: []const f32, want: []const f32) !void {
    if (!std.mem.eql(u8, std.mem.sliceAsBytes(got), std.mem.sliceAsBytes(want))) {
        std.debug.print("{s}: not bit for bit the CPU's\n", .{name});
        return error.MetalMismatch;
    }
}

/// Every TTS kernel against its CPU function.
pub fn check(alloc: std.mem.Allocator, b: *Backend) !void {
    var arena_state: std.heap.ArenaAllocator = .init(alloc);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    var prng = std.Random.DefaultPrng.init(0x7751);
    const random = prng.random();
    var created: std.ArrayList(Buffer) = .empty;
    defer {
        for (created.items) |buffer| b.release(buffer) catch {};
        created.deinit(alloc);
    }
    const Up = struct {
        fn load(bk: *Backend, list: *std.ArrayList(Buffer), gpa: std.mem.Allocator, values: []const f32) !Buffer {
            const buffer = try upload(bk, values);
            try list.append(gpa, buffer);
            return buffer;
        }
        fn zeros(bk: *Backend, list: *std.ArrayList(Buffer), gpa: std.mem.Allocator, count: usize) !Buffer {
            const buffer = try bk.create(count * 4);
            try list.append(gpa, buffer);
            return buffer;
        }
    };

    // The codebook sum: exact, the same additions in the same order.
    const size = 64;
    const dim = 256;
    const frames = 5;
    const books = try a.alloc(f32, codec.groups * size * dim);
    fill(random, books, 3);
    var codes: [frames]codec.Frame = undefined;
    const ids = try a.alloc(f32, frames * codec.groups);
    for (&codes, 0..) |*frame, t| for (frame, 0..) |*code, g| {
        code.* = random.uintLessThan(u16, size);
        ids[t * codec.groups + g] = @bitCast(@as(u32, code.*));
    };
    const first_want = try a.alloc(f32, frames * dim);
    const rest_want = try a.alloc(f32, frames * dim);
    codec.sumCodebooks(books, size, dim, &codes, first_want, rest_want);
    const first = try Up.zeros(b, &created, alloc, frames * dim);
    const rest = try Up.zeros(b, &created, alloc, frames * dim);
    try b.begin();
    try b.ttsCodebookSum(try Up.load(b, &created, alloc, books), try Up.load(b, &created, alloc, ids), first, rest, frames, codec.groups, size, dim);
    try b.commit();
    try expectEqual("tts_codebook_sum first", first.floats()[0 .. frames * dim], first_want);
    try expectEqual("tts_codebook_sum rest", rest.floats()[0 .. frames * dim], rest_want);

    // The gather, causal (dilation 3, padded columns) and transposed: exact.
    const width = 24;
    const rows = 9;
    const joined_values = try a.alloc(f32, (rows + 18) * width);
    fill(random, joined_values, 1);
    const joined = try Up.load(b, &created, alloc, joined_values);
    const Case = struct { taps: usize, base: usize, step: isize, history: usize };
    for ([_]Case{ .{ .taps = 7, .base = 0, .step = 3, .history = 18 }, .{ .taps = 2, .base = 1, .step = -1, .history = 1 } }) |case| {
        const inner = case.taps * width;
        const stride = (inner + 63) / 64 * 64;
        const want = try a.alloc(f32, rows * stride);
        @memset(want, 0);
        const packed_rows = try a.alloc(f32, rows * inner);
        codec.gather(joined_values[0 .. (rows + case.history) * width], width, case.taps, case.base, case.step, 0, rows, packed_rows);
        for (0..rows) |r| @memcpy(want[r * stride ..][0..inner], packed_rows[r * inner ..][0..inner]);
        const out = try Up.zeros(b, &created, alloc, rows * stride);
        @memset(out.floats(), std.math.nan(f32));
        try b.begin();
        try b.ttsGather(joined, out, rows, width, case.taps, stride, case.base, case.step);
        try b.commit();
        try expectEqual("tts_gather", out.floats()[0 .. rows * stride], want);
    }

    // ConvNeXt's depthwise convolution, SnakeBeta, the scaled add.
    const channels = 40;
    const dw_rows = 11;
    const dw_joined = try a.alloc(f32, (dw_rows + 6) * channels);
    fill(random, dw_joined, 2);
    const dw = try a.alloc(f32, channels * 7);
    fill(random, dw, 0.5);
    const dw_b = try a.alloc(f32, channels);
    fill(random, dw_b, 0.5);
    const dw_want = try a.alloc(f32, dw_rows * channels);
    codec.depthwise(dw, dw_b, dw_joined, dw_rows, dw_want);
    const dw_out = try Up.zeros(b, &created, alloc, dw_rows * channels);
    try b.begin();
    try b.ttsDepthwise(try Up.load(b, &created, alloc, dw_joined), try Up.load(b, &created, alloc, dw), try Up.load(b, &created, alloc, dw_b), dw_out, dw_rows, channels, 7);
    try b.commit();
    const dw_worst = try expectWithin("tts_depthwise", dw_out.floats()[0..dw_want.len], dw_want, 1, 1e-6);

    const scale = try a.alloc(f32, channels);
    const inv = try a.alloc(f32, channels);
    for (scale, inv) |*s, *i| {
        s.* = @exp((random.float(f32) * 2 - 1) * 1.5);
        i.* = 1 / (@exp((random.float(f32) * 2 - 1) * 1.5) + 1e-9);
    }
    const snake_want = try a.alloc(f32, dw_want.len);
    codec.snakeBeta(.{ .scale = scale, .inv = inv }, dw_want, snake_want);
    const snake_out = try Up.zeros(b, &created, alloc, dw_want.len);
    try b.begin();
    try b.ttsSnake(dw_out, try Up.load(b, &created, alloc, scale), try Up.load(b, &created, alloc, inv), snake_out, dw_want.len, channels);
    try b.commit();
    // The GPU's input is its own depthwise output; the CPU's sin is not Metal's.
    const snake_cpu = try a.alloc(f32, dw_want.len);
    codec.snakeBeta(.{ .scale = scale, .inv = inv }, dw_out.floats()[0..dw_want.len], snake_cpu);
    const snake_worst = try expectWithin("tts_snake", snake_out.floats()[0..dw_want.len], snake_cpu, 1, 2e-6);

    const add_want = try a.dupe(f32, dw_want);
    codec.addScaled(add_want, scale, snake_cpu);
    const x = try Up.load(b, &created, alloc, dw_want);
    try b.begin();
    try b.ttsScaledAdd(x, try Up.load(b, &created, alloc, scale), try Up.load(b, &created, alloc, snake_cpu), dw_want.len, channels);
    try b.commit();
    const add_worst = try expectWithin("tts_scaled_add", x.floats()[0..dw_want.len], add_want, 1, 2e-7);

    // The window attention against the CPU's F64 scores, with the cache
    // full (past 71) and filling (past 3), the window shorter than the push.
    const heads = 4;
    const stride = heads * 64;
    const window = 72;
    var attention_worst: f32 = 0;
    for ([_][2]usize{ .{ 71, 90 }, .{ 3, 5 }, .{ 0, 1 } }) |shape| {
        const past = shape[0];
        const n = shape[1];
        const q = try a.alloc(f32, n * stride);
        fill(random, q, 1);
        const keys = try a.alloc(f32, (past + n) * stride);
        fill(random, keys, 1);
        const values = try a.alloc(f32, (past + n) * stride);
        fill(random, values, 1);
        const want = try a.alloc(f32, n * stride);
        const scores = try a.alloc(f64, window);
        codec.attention(.{ .heads = heads, .head_dim = 64, .window = window }, q, keys, values, n, past, scores, want);
        const out = try Up.zeros(b, &created, alloc, n * stride);
        try b.begin();
        try b.ttsWindowAttention(try Up.load(b, &created, alloc, q), try Up.load(b, &created, alloc, keys), try Up.load(b, &created, alloc, values), out, n, past, heads, window, 0.125);
        try b.commit();
        attention_worst = @max(attention_worst, try expectWithin("tts_window_attention", out.floats()[0 .. n * stride], want, 1e-2, 2e-5));
    }

    // The output convolution: 7 taps of 96 channels to one, clamped.
    const out_rows = 50;
    const out_width = 96;
    const out_joined = try a.alloc(f32, (out_rows + 6) * out_width);
    fill(random, out_joined, 1);
    const out_w = try a.alloc(f32, 7 * out_width);
    fill(random, out_w, 0.05);
    const out_bias = [_]f32{0.01};
    const conv_want = try a.alloc(f32, out_rows);
    for (conv_want, 0..) |*y, r| {
        var acc: f64 = out_bias[0];
        for (0..7) |j| for (0..out_width) |i| {
            acc += @as(f64, out_w[j * out_width + i]) * out_joined[(r + j) * out_width + i];
        };
        y.* = std.math.clamp(@as(f32, @floatCast(acc)), -1, 1);
    }
    const conv_out = try Up.zeros(b, &created, alloc, out_rows);
    try b.begin();
    try b.ttsConvOut(try Up.load(b, &created, alloc, out_joined), try Up.load(b, &created, alloc, out_w), try Up.load(b, &created, alloc, &out_bias), conv_out, out_rows, out_width, 7);
    try b.commit();
    const conv_worst = try expectWithin("tts_conv_out", conv_out.floats()[0..out_rows], conv_want, 1e-2, 1e-5);

    // The BF16 matvec, each row count, on a 2-byte (not 16-byte) aligned
    // weight range, against the F64 product of the widened weights.
    const m_rows = 37;
    const m_columns = 1032;
    const raw = try a.alloc(u16, m_rows * m_columns + 1);
    for (raw) |*v| {
        const f: f32 = (random.float(f32) * 2 - 1) * 0.1;
        v.* = @truncate(@as(u32, @bitCast(f)) >> 16);
    }
    const input = try a.alloc(f32, m_columns);
    fill(random, input, 1);
    const mv_want = try a.alloc(f32, m_rows);
    for (mv_want, 0..) |*y, r| {
        var acc: f64 = 0;
        for (0..m_columns) |c| acc += @as(f64, @as(f32, @bitCast(@as(u32, raw[1 + r * m_columns + c]) << 16))) * input[c];
        y.* = @floatCast(acc);
    }
    const all = try b.create(raw.len * 2);
    try created.append(alloc, all);
    @memcpy(all.host[0 .. raw.len * 2], std.mem.sliceAsBytes(raw));
    const weights = all.slice(2, m_rows * m_columns * 2);
    const matrix: inference.cpu.Matrix = .{ .encoding = 30, .rows = m_rows, .columns = m_columns, .bytes = weights.host[0 .. m_rows * m_columns * 2] };
    const in = try Up.load(b, &created, alloc, input);
    var matvec_worst: f32 = 0;
    for ([_]usize{ 1, 2, 4 }) |r| {
        const out = try Up.zeros(b, &created, alloc, m_rows);
        try b.begin();
        try b.ttsMatvecBf16(weights, matrix, in, out, r);
        try b.commit();
        matvec_worst = @max(matvec_worst, try expectWithin("tts_matvec_bf16", out.floats()[0..m_rows], mv_want, 1e-1, 2e-5));
    }

    // Three matrices in one dispatch, and gate·up with SiLU: each row summed
    // as the single BF16 matvec sums it, so bit for bit its result (and, for
    // gate·up, `siluMul`'s).
    {
        const widths = [3]usize{ 16, 8, 8 };
        const columns = 512;
        var mats: [3]inference.cpu.Matrix = undefined;
        var bufs: [3]Buffer = undefined;
        var outs: [3]Buffer = undefined;
        var at: usize = 1;
        for (&mats, &bufs, &outs, widths) |*m, *w, *o, height| {
            w.* = try b.create(height * columns * 2);
            try created.append(alloc, w.*);
            @memcpy(w.host[0 .. height * columns * 2], std.mem.sliceAsBytes(raw[at..][0 .. height * columns]));
            at += height * columns;
            m.* = .{ .encoding = 30, .rows = height, .columns = columns, .bytes = w.host[0 .. height * columns * 2] };
            o.* = try Up.zeros(b, &created, alloc, height);
        }
        var singles: [3]Buffer = undefined;
        for (&singles, widths) |*s, height| s.* = try Up.zeros(b, &created, alloc, height);
        const gated = try Up.zeros(b, &created, alloc, 8);
        const unfused = try Up.zeros(b, &created, alloc, 8);
        try b.begin();
        try b.ttsMatvec3Bf16(bufs, mats, in, outs);
        for (bufs, mats, singles) |w, m, s| try b.ttsMatvecBf16(w, m, in, s, 2);
        try b.ttsGateUpBf16(bufs[1], bufs[2], mats[1], in, gated);
        try b.copy(unfused, singles[1], 8);
        try b.siluMul(unfused, singles[2], 8);
        try b.commit();
        for (outs, singles, widths) |o, s, height| try expectEqual("tts_matvec3_bf16", o.floats()[0..height], s.floats()[0..height]);
        try expectEqual("tts_gate_up_bf16", gated.floats()[0..8], unfused.floats()[0..8]);
    }

    // The BF16 tile: the generic F32 tile's result bit for bit, on 32 rows
    // of 1024 columns (a copy: the tile reads whole segments).
    const t_rows = 32;
    const t_columns = 1024;
    const tokens = 21;
    const tile_bytes = try b.create(t_rows * t_columns * 2);
    try created.append(alloc, tile_bytes);
    @memcpy(tile_bytes.host[0 .. t_rows * t_columns * 2], std.mem.sliceAsBytes(raw[1..][0 .. t_rows * t_columns]));
    const t_matrix: inference.cpu.Matrix = .{ .encoding = 30, .rows = t_rows, .columns = t_columns, .bytes = tile_bytes.host[0 .. t_rows * t_columns * 2] };
    const padded = Backend.matmulPadded(tokens);
    const t_input = try a.alloc(f32, padded * t_columns);
    fill(random, t_input, 1);
    const t_in = try Up.load(b, &created, alloc, t_input);
    const generic = try Up.zeros(b, &created, alloc, padded * t_rows);
    const special = try Up.zeros(b, &created, alloc, padded * t_rows);
    try b.begin();
    try b.matmulTile(tile_bytes, t_matrix, t_in, t_columns, generic, t_rows, tokens);
    try b.ttsMatmulBf16(tile_bytes, t_matrix, t_in, t_columns, special, t_rows, tokens);
    try b.commit();
    try expectEqual("tts_matmul_bf16", special.floats()[0 .. tokens * t_rows], generic.floats()[0 .. tokens * t_rows]);

    // A table row by a device code, written then accumulated: exact.
    const codes_buffer = try Up.load(b, &created, alloc, &.{ @bitCast(@as(u32, 3)), @bitCast(@as(u32, 17)) });
    const row = try Up.zeros(b, &created, alloc, t_columns);
    try b.begin();
    try b.ttsEmbedRow(tile_bytes, t_rows, t_columns, codes_buffer, 0, row, false);
    try b.ttsEmbedRow(tile_bytes, t_rows, t_columns, codes_buffer, 1, row, true);
    try b.commit();
    const row_want = try a.alloc(f32, t_columns);
    for (row_want, 0..) |*v, i| {
        const first_value: f32 = @bitCast(@as(u32, raw[1 + 3 * t_columns + i]) << 16);
        v.* = first_value + @as(f32, @bitCast(@as(u32, raw[1 + 17 * t_columns + i]) << 16));
    }
    try expectEqual("tts_embed_row", row.floats()[0..t_columns], row_want);

    // The device sampler against the CPU sampler with the same draw, over
    // top-k (the predictor's 50), a nucleus, and greedy.
    const sampling = inference.sampling;
    const count = 2048;
    const logits = try a.alloc(f32, count);
    const candidates = try a.alloc(sampling.Candidate, count);
    const logits_buffer = try Up.zeros(b, &created, alloc, count);
    const draw_buffer = try Up.zeros(b, &created, alloc, 1);
    const code_buffer = try Up.zeros(b, &created, alloc, 1);
    const Options = struct { o: sampling.Options, trials: usize };
    var trials: usize = 0;
    for ([_]Options{
        .{ .o = .{ .temperature = 0.9, .top_k = 50 }, .trials = 300 },
        .{ .o = .{ .temperature = 0.7, .top_k = 0, .top_p = 0.8 }, .trials = 60 },
        .{ .o = .{ .temperature = 0 }, .trials = 20 },
    }) |case| for (0..case.trials) |trial| {
        fill(random, logits, 6);
        // Ties at the top, which the sort must break by id.
        if (trial % 3 == 0) logits[1500] = logits[std.mem.indexOfMax(f32, logits)];
        var sampler = try sampling.Sampler.init(trial, case.o);
        var copy = sampler.rng;
        const u = copy.random().float(f64);
        const want = try sampler.select(logits, candidates, null);
        @memcpy(logits_buffer.floats()[0..count], logits);
        draw_buffer.floats()[0] = @floatCast(u);
        try b.begin();
        try b.ttsSample(logits_buffer, count, draw_buffer, 0, code_buffer, 0, case.o.top_k, case.o.top_p, case.o.temperature);
        try b.commit();
        const got: u32 = @bitCast(code_buffer.floats()[0]);
        if (got != want) {
            std.debug.print("tts_sample: trial {d} chose {d}, the CPU sampler {d}\n", .{ trial, got, want });
            return error.MetalMismatch;
        }
        trials += 1;
    };
    std.debug.print("Qwen3-TTS kernels vs the CPU: codebook sum, gather, row gather, the fused q/k/v and gate·up matvecs (against the single one), and the BF16 tile (against the generic tile) bit for bit; scaled add {e:.2}, depthwise {e:.2}, SnakeBeta {e:.2}, window attention {e:.2} (F64 scores), output convolution {e:.2}, BF16 matvec {e:.2} (worst relative); the device sampler chose the CPU sampler's id in {d} of {d} draws\n", .{ add_worst, dw_worst, snake_worst, attention_worst, conv_worst, matvec_worst, trials, trials });
}

/// The BF16 matvec against the generic one on the TTS shapes: GB/s of
/// weight bytes over 200 dispatches in one command buffer.
pub fn bench(alloc: std.mem.Allocator) !void {
    var diagnostic: [8192]u8 = @splat(0);
    var b = Backend.init(alloc, &diagnostic) catch |err| {
        std.debug.print("{s}\n", .{std.mem.sliceTo(&diagnostic, 0)});
        return err;
    };
    defer b.deinit();
    const shapes = [_][2]usize{ .{ 1024, 1024 }, .{ 2048, 1024 }, .{ 3072, 1024 }, .{ 1024, 3072 }, .{ 1024, 2048 }, .{ 2048, 2048 }, .{ 6144, 2048 }, .{ 2048, 6144 }, .{ 3072, 2048 } };
    const repeats = 200;
    std.debug.print("{s:>11} {s:>10} {s:>10} {s:>10} {s:>10}  (GB/s of weights)\n", .{ "shape", "generic", "bf16 r1", "bf16 r2", "bf16 r4" });
    for (shapes) |shape| {
        const rows = shape[0];
        const columns = shape[1];
        const bytes = rows * columns * 2;
        // Distinct copies so the 200 reads are not served by the cache.
        const copies = 8;
        const weights = try b.create(bytes * copies);
        defer b.release(weights) catch {};
        for (std.mem.bytesAsSlice(u16, weights.host[0 .. bytes * copies]), 0..) |*v, i| v.* = @truncate(0x3c00 + (i % 251));
        const input = try b.create(columns * 4);
        defer b.release(input) catch {};
        const output = try b.create(rows * 4);
        defer b.release(output) catch {};
        var rates: [4]f64 = undefined;
        for (&rates, 0..) |*rate, variant| {
            try b.begin();
            for (0..repeats) |i| {
                const w = weights.slice((i % copies) * bytes, bytes);
                const m: inference.cpu.Matrix = .{ .encoding = 30, .rows = rows, .columns = columns, .bytes = w.host[0..bytes] };
                switch (variant) {
                    0 => try b.matvec(w, m, input, output),
                    else => try b.ttsMatvecBf16(w, m, input, output, @as(usize, 1) << @intCast(variant - 1)),
                }
            }
            const before = b.gpuSeconds();
            try b.commit();
            rate.* = @as(f64, @floatFromInt(bytes * repeats)) / (b.gpuSeconds() - before) / 1e9;
        }
        std.debug.print("{d:>5}x{d:<5} {d:>10.0} {d:>10.0} {d:>10.0} {d:>10.0}\n", .{ rows, columns, rates[0], rates[1], rates[2], rates[3] });
    }
}
