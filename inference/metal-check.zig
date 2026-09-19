//! Explicit GPU fixture/lifecycle check. No full model needed, never a default test.
//!
//! Every GPU kernel is compared with the CPU reference or with the pinned
//! llama.cpp fixtures the CPU reference itself is tested against. Tolerances are
//! stated per section; they bound F32 GPU arithmetic against F64 CPU references
//! and are not claims of bitwise equality. Dequantization, however, must match
//! the CPU decoders exactly: both index the same bytes with the same equations.
const std = @import("std");
const inference = @import("inference");
const Backend = inference.metal.Backend;
const Buffer = inference.metal.Buffer;

const QuantFixture = struct { rows: []const struct { encoding: u32, bytes: []const u8, values: []const f32 } };
const AttentionCase = struct { query_heads: usize, kv_heads: usize, key_width: usize, value_width: usize, tokens: usize, visible_tokens: usize, scale: f32, queries: []const f32, keys: []const f32, values: []const f32, output: []const f32 };
const AttentionFixture = struct { revision: []const u8, cases: []const AttentionCase };
const DeltaCase = struct { query: []const f32, key: []const f32, value: []const f32, log_decay: f32, beta: f32, scale: f32, state: []const f32, output: []const f32, next_state: []const f32 };
const ConvCase = struct { input: []const f32, weights: []const f32, history: []const f32, kernel: usize, output: []const f32, next_history: []const f32 };
const RecurrentFixture = struct { revision: []const u8, delta: []const DeltaCase, convolution: []const ConvCase };

fn openBackend(alloc: std.mem.Allocator) !Backend {
    var diagnostic: [8192]u8 = @splat(0);
    return Backend.init(alloc, &diagnostic) catch |err| {
        std.debug.print("{s}\n", .{std.mem.sliceTo(&diagnostic, 0)});
        return err;
    };
}

fn expectClose(name: []const u8, got: f32, want: f32, tolerance: f32) !void {
    if (!(@abs(got - want) <= tolerance)) {
        std.debug.print("{s}: got {d} want {d} tolerance {d}\n", .{ name, got, want, tolerance });
        return error.MetalMismatch;
    }
}

fn upload(b: *Backend, values: []const f32) !Buffer {
    const buffer = try b.create(values.len * 4);
    @memcpy(buffer.floats(), values);
    return buffer;
}
fn uploadBytes(b: *Backend, bytes: []const u8) !Buffer {
    const buffer = try b.create(bytes.len);
    @memcpy(buffer.host[0..bytes.len], bytes);
    return buffer;
}
/// One-shot matvec through the recording API; used by the fixture loops.
fn matvecOnce(b: *Backend, matrix: inference.cpu.Matrix, weights: Buffer, input: []const f32, output: []f32) !void {
    const in = try upload(b, input);
    const out = try b.create(output.len * 4);
    try b.begin();
    try b.matvec(weights, matrix, in, out);
    try b.commit();
    @memcpy(output, out.floats()[0..output.len]);
}

/// Tiles fixture blocks of `encoding` into a `rows` x `columns` matrix so the
/// bytes are valid encoded data; caller frees the result.
fn tiledMatrix(alloc: std.mem.Allocator, sample_bytes: []const u8, encoding: u32, rows: usize, columns: usize) ![]u8 {
    const layout = inference.encoding.layout(encoding) orelse return error.UnknownEncoding;
    const block_bytes = layout.bytes_per_block;
    const blocks_per_fixture = sample_bytes.len / block_bytes;
    if (columns % layout.elements_per_block != 0) return error.FixtureNotTileable;
    const repeats = columns / layout.elements_per_block;
    const stride = block_bytes * repeats;
    const region = try alloc.alloc(u8, stride * rows);
    for (0..rows) |r| for (0..repeats) |t| {
        const block = (r * 3 + t) % blocks_per_fixture;
        @memcpy(region[r * stride + t * block_bytes ..][0..block_bytes], sample_bytes[block * block_bytes ..][0..block_bytes]);
    };
    return region;
}

/// Rewrites the F16 block scales of every block in a tiled region to 2^-6 so
/// decoded weights stay O(1..16): the pinned fixtures carry extreme scales
/// (decoded values up to ~1e7) to pin the decoders, which would overflow the
/// half operands of the matmul tiles. The CPU reference reads the same
/// bytes, so the comparison stays exact. Positions are format facts
/// (dequant.metal header): Q4_K/Q5_K d and dmin at 0/2, Q6_K d at 208,
/// Q3_K d at 108, IQ3_S, IQ4_XS, Q4_0, and PQ2_0 d at 0, PTQ1_0 d at 26.
fn tameScales(region: []u8, encoding: u32) void {
    const layout = inference.encoding.layout(encoding) orelse return;
    const positions: []const usize = switch (encoding) {
        12, 13 => &.{ 0, 2 },
        14 => &.{208},
        11 => &.{108},
        143 => &.{26},
        2, 21, 23, 142 => &.{0},
        else => return,
    };
    var offset: usize = 0;
    while (offset + layout.bytes_per_block <= region.len) : (offset += layout.bytes_per_block) {
        for (positions) |at| std.mem.writeInt(u16, region[offset + at ..][0..2], 0x2400, .little);
    }
}

/// `--matvec-bench [ENCODING]`: achieved weight bandwidth of the matvec kernels on
/// model-shaped matrices (GPU busy time per dispatch, one dispatch per command
/// buffer with `repeats` dispatches; best and mean over `rounds`), one encoding
/// when named (a full run heats the GPU progressively). A measurement aid for kernel work, not
/// an acceptance benchmark: it isolates one kernel from the token schedule.
fn matvecBench(alloc: std.mem.Allocator, only: ?[]const u8) !void {
    var backend = try openBackend(alloc);
    defer backend.deinit();
    const b = &backend;
    const Shape = struct { rows: usize, columns: usize, name: []const u8 };
    const shapes = [_]Shape{
        .{ .rows = 69632, .columns = 5120, .name = "69632x5120 (4 ffn_gate)" },
        .{ .rows = 20480, .columns = 17408, .name = "20480x17408 (4 ffn_down)" },
        .{ .rows = 248320, .columns = 5120, .name = "248320x5120 (output)" },
        // Actual down-projection geometry: fewer groups than the tiled cases.
        .{ .rows = 5120, .columns = 17408, .name = "5120x17408 (ffn_down)" },
    };
    const encodings = [_]struct { id: u32, fixture: []const u8, name: []const u8 }{
        .{ .id = 11, .fixture = "k-signed", .name = "Q3_K" },
        .{ .id = 21, .fixture = "iq", .name = "IQ3_S" },
        .{ .id = 12, .fixture = "k-affine", .name = "Q4_K" },
        .{ .id = 13, .fixture = "k-affine", .name = "Q5_K" },
        .{ .id = 14, .fixture = "k-signed", .name = "Q6_K" },
        .{ .id = 23, .fixture = "iq", .name = "IQ4_XS" },
        .{ .id = 2, .fixture = "simple", .name = "Q4_0" },
        .{ .id = 142, .fixture = "ternary", .name = "PQ2_0" },
        .{ .id = 143, .fixture = "ternary", .name = "PTQ1_0" },
    };
    const rounds = 5; // measured command buffers
    // One weight buffer for every case: fresh multi-GB buffers made timings swing
    // with driver residency work rather than kernel speed.
    var max_bytes: usize = 0;
    for (encodings) |enc| for (shapes) |shape| {
        const layout = inference.encoding.layout(enc.id) orelse return error.UnknownEncoding;
        max_bytes = @max(max_bytes, shape.rows * (shape.columns / layout.elements_per_block) * layout.bytes_per_block);
    };
    const weights = try b.create(max_bytes);
    const input = try b.create(17408 * 4);
    for (input.floats(), 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 13)) / 13 - 0.5;
    const output = try b.create(248320 * 4);
    std.debug.print("{s:<8} {s:<26} {s:>9} {s:>8} {s:>9} {s:>9}  rounds (GB/s)\n", .{ "encoding", "shape", "MB", "path", "best", "mean" });
    inline for (.{ "k-affine", "k-signed", "iq", "simple", "ternary" }) |fixture_name| {
        const fixtures = try std.json.parseFromSlice(QuantFixture, alloc, @embedFile("src/quant/fixtures/" ++ fixture_name ++ ".json"), .{ .ignore_unknown_fields = true });
        defer fixtures.deinit();
        for (encodings) |enc| {
            if (!std.mem.eql(u8, enc.fixture, fixture_name)) continue;
            if (only) |name| if (!std.mem.eql(u8, name, enc.name)) continue;
            var sample_bytes: ?[]const u8 = null;
            for (fixtures.value.rows) |sample| if (sample.encoding == enc.id) {
                sample_bytes = sample.bytes;
                break;
            };
            const bytes = sample_bytes orelse return error.FixtureMissing;
            for (shapes) |shape| {
                // Small matrices need longer batches to avoid measuring clock ramp-up.
                const repeats: usize = if (shape.rows < 8192) 64 else 8;
                const region = try tiledMatrix(alloc, bytes, enc.id, shape.rows, shape.columns);
                defer alloc.free(region);
                @memcpy(weights.host[0..region.len], region);
                const matrix: inference.cpu.Matrix = .{ .rows = shape.rows, .columns = shape.columns, .encoding = enc.id, .bytes = region };
                for ([_]bool{ false, true }) |generic| {
                    b.generic_only = generic;
                    // Back-to-back dispatches in one command buffer keep the GPU
                    // clocked as it is inside a token; isolated dispatches measure
                    // low-power ramp-up instead of the kernel.
                    const mb = @as(f64, @floatFromInt(region.len)) / 1e6;
                    var best: f64 = 0;
                    var total: f64 = 0;
                    var samples: [rounds]f64 = undefined;
                    for (0..rounds + 2) |i| {
                        const before = b.gpuSeconds();
                        try b.begin();
                        for (0..repeats) |_| try b.matvec(weights, matrix, input, output);
                        try b.commit();
                        const rate = mb / 1e3 / ((b.gpuSeconds() - before) / @as(f64, @floatFromInt(repeats)));
                        if (i < 2) continue; // warm-up
                        samples[i - 2] = rate;
                        best = @max(best, rate);
                        total += rate;
                    }
                    std.debug.print("{s:<8} {s:<26} {d:>9.1} {s:>8} {d:>9.1} {d:>9.1} ", .{ enc.name, shape.name, mb, if (generic) "generic" else "block", best, total / rounds });
                    for (samples) |r| std.debug.print(" {d:.0}", .{r});
                    std.debug.print("\n", .{});
                }
            }
        }
    }
    b.generic_only = false;
}

/// `--matmul-bench [tokens]`: throughput of the batched prefill matmul on
/// model shapes for a chunk of `tokens` (default 256). One command buffer
/// issues 64 dispatches at t <= 8 and 16 above (the batch keeps the GPU
/// clocked), five measured after two warm-ups; reports GPU ms per dispatch,
/// GFLOP/s of F32 multiply-adds (2 · rows · columns · tokens), and the weight
/// bytes per second the tile streams.
fn matmulBench(alloc: std.mem.Allocator, tokens: usize) !void {
    var backend = try openBackend(alloc);
    defer backend.deinit();
    const b = &backend;
    const Shape = struct { rows: usize, columns: usize, name: []const u8 };
    const shapes = [_]Shape{
        .{ .rows = 17408, .columns = 5120, .name = "17408x5120 (ffn_gate)" },
        .{ .rows = 5120, .columns = 17408, .name = "5120x17408 (ffn_down)" },
    };
    const encodings = [_]struct { id: u32, fixture: []const u8, name: []const u8 }{
        .{ .id = 11, .fixture = "k-signed", .name = "Q3_K" },
        .{ .id = 21, .fixture = "iq", .name = "IQ3_S" },
        .{ .id = 12, .fixture = "k-affine", .name = "Q4_K" },
        .{ .id = 13, .fixture = "k-affine", .name = "Q5_K" },
        .{ .id = 14, .fixture = "k-signed", .name = "Q6_K" },
        .{ .id = 23, .fixture = "iq", .name = "IQ4_XS" },
        .{ .id = 2, .fixture = "simple", .name = "Q4_0" },
        .{ .id = 142, .fixture = "ternary", .name = "PQ2_0" },
        .{ .id = 143, .fixture = "ternary", .name = "PTQ1_0" },
    };
    const rounds = 5;
    // Back-to-back dispatches per command buffer keep the clock where a token
    // keeps it; an isolated short dispatch measures ramp-up, not the kernel
    // (the matvec bench's note). 16 is the floor the slowest shape needs.
    const repeats: usize = if (tokens <= 8) 64 else 16;
    var max_bytes: usize = 0;
    for (encodings) |enc| for (shapes) |shape| {
        const layout = inference.encoding.layout(enc.id) orelse return error.UnknownEncoding;
        max_bytes = @max(max_bytes, shape.rows * (shape.columns / layout.elements_per_block) * layout.bytes_per_block);
    };
    const weights = try b.create(max_bytes);
    const input = try b.create(Backend.matmulPadded(tokens) * 17408 * 4);
    for (input.floats(), 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 13)) / 13 - 0.5;
    const output = try b.create(Backend.matmulPadded(tokens) * 17408 * 4);
    std.debug.print("{s:<8} {s:<24} {s:<11} {s:>8} {s:>9} {s:>10} {s:>8} {s:>8}  rounds (ms), {d} tokens, {d}/round\n", .{ "encoding", "shape", "tile", "MB", "best ms", "GFLOP/s", "tok/s*", "GB/s", tokens, repeats });
    inline for (.{ "k-affine", "k-signed", "iq", "simple", "ternary" }) |fixture_name| {
        const fixtures = try std.json.parseFromSlice(QuantFixture, alloc, @embedFile("src/quant/fixtures/" ++ fixture_name ++ ".json"), .{ .ignore_unknown_fields = true });
        defer fixtures.deinit();
        for (encodings) |enc| {
            if (!std.mem.eql(u8, enc.fixture, fixture_name)) continue;
            var sample_bytes: ?[]const u8 = null;
            for (fixtures.value.rows) |sample| if (sample.encoding == enc.id) {
                sample_bytes = sample.bytes;
                break;
            };
            const bytes = sample_bytes orelse return error.FixtureMissing;
            for (shapes) |shape| for ([_]bool{ true, false }) |generic| {
                const region = try tiledMatrix(alloc, bytes, enc.id, shape.rows, shape.columns);
                defer alloc.free(region);
                @memcpy(weights.host[0..region.len], region);
                const matrix: inference.cpu.Matrix = .{ .rows = shape.rows, .columns = shape.columns, .encoding = enc.id, .bytes = region };
                b.generic_only = generic;
                const kernel: inference.metal.Kernel = if (generic) .matmul else (Backend.specializedMatmul(enc.id, 0, region.len / shape.rows, tokens) orelse .matmul);
                const geometry = Backend.matmulGeometry(kernel);
                // A chunk spanning several token tiles reads the weights once
                // per tile, as `matmul`'s profile attribution counts them.
                const token_tiles = (tokens + geometry.tokens - 1) / geometry.tokens;
                const weight_mb = @as(f64, @floatFromInt(region.len)) / 1e6;
                var best: f64 = std.math.inf(f64);
                var samples: [rounds]f64 = undefined;
                for (0..rounds + 2) |i| {
                    const before = b.gpuSeconds();
                    try b.begin();
                    for (0..repeats) |_| try b.matmul(weights, matrix, input, shape.columns, output, shape.rows, tokens);
                    try b.commit();
                    const ms = (b.gpuSeconds() - before) * 1e3 / @as(f64, @floatFromInt(repeats));
                    if (i < 2) continue; // warm-up
                    samples[i - 2] = ms;
                    best = @min(best, ms);
                }
                const flops = 2.0 * @as(f64, @floatFromInt(shape.rows * shape.columns * tokens));
                const gflops = flops / (best * 1e-3) / 1e9;
                const tile = if (geometry.rows == 16) "16x8" else switch (geometry.tokens) {
                    8 => "8x8",
                    16 => "8x16",
                    32 => "32x32",
                    64 => "64x64",
                    else => "specialized",
                };
                const gbps = weight_mb * @as(f64, @floatFromInt(token_tiles)) / best;
                std.debug.print("{s:<8} {s:<24} {s:<11} {d:>8.1} {d:>9.2} {d:>10.0} {d:>8.1} {d:>8.1} ", .{ enc.name, shape.name, if (generic) "generic" else tile, weight_mb, best, gflops, gflops / 54.0, gbps });
                for (samples) |ms| std.debug.print(" {d:.1}", .{ms});
                std.debug.print("\n", .{});
            };
        }
    }
    b.generic_only = false;
    std.debug.print("* tok/s if the whole 54 GFLOP/token model ran at this rate: a ceiling, not a prediction.\n", .{});
}

/// Mixture-of-experts kernels against `cpu.experts`: the router on rows with
/// ties and lanes past the expert count (indices exact), the gathered matvec
/// on both kernel paths with shared and per-slot inputs and NaN in every
/// unselected expert, the strided GELU pair and the weighted combine, the
/// whole decode chain over three tokens against the F64 reference, the
/// prefill lists and gathered tiles over a skewed chunk on both tile
/// kernels against the reference and the decode path, and the contract
/// rejections.
fn checkExperts(alloc: std.mem.Allocator, b: *Backend) !void {
    var prng = std.Random.DefaultPrng.init(0xe8e8);
    const random = prng.random();
    // Routing: 128 experts, k 8, six strided rows.
    {
        const experts: usize = 128;
        const k: usize = 8;
        const rows: usize = 6;
        const in_stride: usize = 130;
        const logits = try b.create(rows * in_stride * 4);
        for (logits.floats()) |*v| v.* = random.floatNorm(f32);
        const l = logits.floats();
        // Row 1: ties at the top and at the k boundary; row 2: all equal;
        // row 3: a wide spread so most probabilities underflow to zero.
        l[1 * in_stride + 5] = 3;
        l[1 * in_stride + 9] = 3;
        l[1 * in_stride + 70] = 3;
        for (0..experts) |i| if (l[1 * in_stride + i] < 2.5 and l[1 * in_stride + i] > 0.5) {
            l[1 * in_stride + i] = 1.25;
        };
        @memset(l[2 * in_stride ..][0..experts], 0.75);
        for (l[3 * in_stride ..][0..experts]) |*v| v.* *= 40;
        const indices = try b.create(rows * k * 4);
        const weights = try b.create(rows * k * 4);
        try b.begin();
        try b.route(logits, experts, k, rows, in_stride, indices, weights);
        try b.commit();
        var expected_indices: [8]u32 = undefined;
        var expected_weights: [8]f32 = undefined;
        var worst: f64 = 0;
        for (0..rows) |r| {
            try inference.cpu.experts.route(l[r * in_stride ..][0..experts], &expected_indices, &expected_weights);
            const got_indices = @as([*]const u32, @ptrCast(@alignCast(indices.host)))[r * k ..][0..k];
            if (!std.mem.eql(u32, got_indices, &expected_indices)) {
                std.debug.print("route row {d}: got {any} want {any}\n", .{ r, got_indices, expected_indices });
                return error.MetalMismatch;
            }
            for (weights.floats()[r * k ..][0..k], expected_weights) |got, want| {
                worst = @max(worst, @abs(@as(f64, got) - want));
                try expectClose("route weight", got, want, 1e-6);
            }
        }
        // Rows 2: all equal selects 0..7 with weight 1/8.
        try std.testing.expectEqualSlices(u32, &.{ 0, 1, 2, 3, 4, 5, 6, 7 }, @as([*]const u32, @ptrCast(@alignCast(indices.host)))[2 * k ..][0..k]);
        // An expert count that is not a multiple of 32, k 5, one row.
        var small_indices: [5]u32 = undefined;
        var small_weights: [5]f32 = undefined;
        try inference.cpu.experts.route(l[0..37], &small_indices, &small_weights);
        try b.begin();
        try b.route(logits, 37, 5, 1, in_stride, indices, weights);
        try b.commit();
        try std.testing.expectEqualSlices(u32, &small_indices, @as([*]const u32, @ptrCast(@alignCast(indices.host)))[0..5]);
        for (weights.floats()[0..5], small_weights) |got, want| try expectClose("route weight (37 experts)", got, want, 1e-6);
        std.debug.print("Router vs CPU F64 on 128 and 37 experts with ties: indices exact, weights max abs {e:.3} (bound 1e-6)\n", .{worst});
        if (b.route(logits, experts, 129, 1, in_stride, indices, weights)) |_| return error.ExpectedInvalidShape else |err| if (err != error.InvalidShape) return err;
        if (b.route(logits, 257, 8, 1, 257, indices, weights)) |_| return error.ExpectedInvalidShape else |err| if (err != error.InvalidShape) return err;
        if (b.route(logits, 64, 8, rows, 63, indices, weights)) |_| return error.ExpectedInvalidShape else |err| if (err != error.InvalidShape) return err;
        if (b.route(logits, 128, 8, rows, in_stride, indices.slice(2, rows * k * 4 - 2), weights)) |_| return error.ExpectedInvalidShape else |err| if (err != error.InvalidShape) return err;
    }
    // Gathered Q4_0 matvec: six experts of 40 × 1,280 (two full 16-row
    // groups plus a partial one), slots [5, 0, 5, 2], shared and per-slot
    // inputs, both kernel paths, then NaN in every unselected expert.
    const fixtures = try std.json.parseFromSlice(QuantFixture, alloc, @embedFile("src/quant/fixtures/simple.json"), .{ .ignore_unknown_fields = true });
    defer fixtures.deinit();
    var q4_0: ?[]const u8 = null;
    for (fixtures.value.rows) |sample| if (sample.encoding == 2) {
        q4_0 = sample.bytes;
        break;
    };
    const sample_bytes = q4_0 orelse return error.FixtureMissing;
    {
        const experts: usize = 6;
        const rows: usize = 40;
        const columns: usize = 1280;
        const slots = [_]u32{ 5, 0, 5, 2 };
        const region = try tiledMatrix(alloc, sample_bytes, 2, experts * rows, columns);
        defer alloc.free(region);
        const tensor: inference.cpu.ExpertMatrix = .{ .encoding = 2, .experts = experts, .rows = rows, .columns = columns, .bytes = region };
        const weights = try uploadBytes(b, region);
        const indices = try b.create(slots.len * 4);
        @memcpy(@as([*]u32, @ptrCast(@alignCast(indices.host)))[0..slots.len], &slots);
        const input = try b.create(slots.len * columns * 4);
        for (input.floats()) |*v| v.* = random.float(f32) * 2 - 1;
        const output = try b.create(slots.len * rows * 4);
        const expected = try alloc.alloc(f32, rows);
        defer alloc.free(expected);
        const decoded = try alloc.alloc(f32, columns);
        defer alloc.free(decoded);
        var clean: [slots.len * rows]f32 = undefined;
        for ([_]usize{ 0, columns }) |in_stride| for ([_]bool{ false, true }) |generic| {
            b.generic_only = generic;
            for (output.floats()) |*v| v.* = std.math.nan(f32);
            try b.begin();
            try b.matvecExperts(weights, tensor, indices, slots.len, input, in_stride, output, rows);
            try b.commit();
            for (slots, 0..) |e, s| {
                const x = input.floats()[s * in_stride ..][0..columns];
                const matrix = try tensor.expert(e);
                try inference.cpu.matvec(matrix, x, expected, decoded);
                const stride = matrix.bytes.len / rows;
                for (0..rows) |r| {
                    try inference.quant.row(2, matrix.bytes[r * stride ..][0..stride], decoded);
                    var mass: f64 = 0;
                    for (decoded, x) |w, xv| mass += @abs(@as(f64, w) * xv);
                    expectClose(if (generic) "gathered matvec (generic)" else "gathered matvec", output.floats()[s * rows + r], expected[r], @floatCast(mass * 4e-6 + 1e-6)) catch |err| {
                        std.debug.print("  in_stride {d}, slot {d} (expert {d}), row {d}\n", .{ in_stride, s, e, r });
                        return err;
                    };
                }
            }
            if (in_stride == 0 and !generic) @memcpy(&clean, output.floats()[0 .. slots.len * rows]);
        };
        b.generic_only = false;
        // Poison: NaN scales in experts 1, 3, 4; the selected outputs are bit-identical.
        const poisoned = try alloc.dupe(u8, region);
        defer alloc.free(poisoned);
        const per_expert = region.len / experts;
        for ([_]usize{ 1, 3, 4 }) |e| {
            var offset: usize = e * per_expert;
            while (offset < (e + 1) * per_expert) : (offset += 18) std.mem.writeInt(u16, poisoned[offset..][0..2], 0x7e00, .little);
        }
        @memcpy(weights.host[0..poisoned.len], poisoned);
        for (output.floats()) |*v| v.* = std.math.nan(f32);
        try b.begin();
        try b.matvecExperts(weights, tensor, indices, slots.len, input, 0, output, rows);
        try b.commit();
        for (output.floats()[0 .. slots.len * rows], clean) |got, want| if (@as(u32, @bitCast(got)) != @as(u32, @bitCast(want))) {
            std.debug.print("gathered matvec read an unselected expert: {d} vs {d}\n", .{ got, want });
            return error.MetalMismatch;
        };
        @memcpy(weights.host[0..region.len], region);
        // Contract: slot inputs stay float4-aligned, indices word-aligned, one slot at least.
        if (b.matvecExperts(weights, tensor, indices, slots.len, input, columns + 1, output, rows)) |_| return error.ExpectedInvalidShape else |err| if (err != error.InvalidShape) return err;
        if (b.matvecExperts(weights, tensor, indices.slice(2, 8), 2, input, 0, output, rows)) |_| return error.ExpectedInvalidShape else |err| if (err != error.InvalidShape) return err;
        if (b.matvecExperts(weights, tensor, indices, 0, input, 0, output, rows)) |_| return error.ExpectedInvalidShape else |err| if (err != error.InvalidShape) return err;
        var bad = tensor;
        bad.experts = 7;
        if (b.matvecExperts(weights, bad, indices, 1, input, 0, output, rows)) |_| return error.ExpectedInvalidShape else |err| if (err != error.InvalidShape) return err;
        // A dense F32 tensor of 3 experts × 8 rows through the generic branch (rows below one group).
        {
            const f_rows: usize = 8;
            const f_columns: usize = 32;
            const f_region = try alloc.alloc(u8, 3 * f_rows * f_columns * 4);
            defer alloc.free(f_region);
            for (0..3 * f_rows * f_columns) |i| std.mem.writeInt(u32, f_region[i * 4 ..][0..4], @bitCast(random.floatNorm(f32)), .little);
            const f_tensor: inference.cpu.ExpertMatrix = .{ .encoding = 0, .experts = 3, .rows = f_rows, .columns = f_columns, .bytes = f_region };
            const f_weights = try uploadBytes(b, f_region);
            const f_indices = try b.create(8);
            @as([*]u32, @ptrCast(@alignCast(f_indices.host)))[0] = 2;
            @as([*]u32, @ptrCast(@alignCast(f_indices.host)))[1] = 1;
            const f_out = try b.create(2 * f_rows * 4);
            try b.begin();
            try b.matvecExperts(f_weights, f_tensor, f_indices, 2, input, f_columns, f_out, f_rows);
            try b.commit();
            var f_expected: [8]f32 = undefined;
            for ([_]u32{ 2, 1 }, 0..) |e, s| {
                try inference.cpu.matvec(try f_tensor.expert(e), input.floats()[s * f_columns ..][0..f_columns], &f_expected, decoded);
                for (f_out.floats()[s * f_rows ..][0..f_rows], f_expected) |got, want| try expectClose("gathered dense matvec", got, want, 1e-5);
            }
        }
    }
    // The decode chain over three tokens: route → gathered gate-up →
    // gelu·up rows → gathered down → combine (with and without the
    // per-expert down scale) against `cpu.experts.ffn` per token. Six
    // experts of width 256 and the model's ff of 704 (the down projection's
    // 22-block rows on the specialized Q4_0 kernel), k 3.
    {
        const experts: usize = 6;
        const k: usize = 3;
        const tokens: usize = 3;
        const width: usize = 256;
        const ff: usize = 704;
        const gu_region = try tiledMatrix(alloc, sample_bytes, 2, experts * 2 * ff, width);
        defer alloc.free(gu_region);
        const down_region = try tiledMatrix(alloc, sample_bytes, 2, experts * width, ff);
        defer alloc.free(down_region);
        tameScales(gu_region, 2);
        tameScales(down_region, 2);
        const gate_up: inference.cpu.ExpertMatrix = .{ .encoding = 2, .experts = experts, .rows = 2 * ff, .columns = width, .bytes = gu_region };
        const down: inference.cpu.ExpertMatrix = .{ .encoding = 2, .experts = experts, .rows = width, .columns = ff, .bytes = down_region };
        const gu_weights = try uploadBytes(b, gu_region);
        const down_weights = try uploadBytes(b, down_region);
        const scales = try b.create(experts * 4);
        for (scales.floats()) |*v| v.* = random.float(f32) + 0.5;
        const logits = try b.create(tokens * experts * 4);
        for (logits.floats()) |*v| v.* = random.floatNorm(f32);
        const x = try b.create(tokens * width * 4);
        for (x.floats()) |*v| v.* = random.floatNorm(f32) * 0.05;
        const indices = try b.create(tokens * k * 4);
        const weights = try b.create(tokens * k * 4);
        const gu_out = try b.create(tokens * k * 2 * ff * 4);
        const hidden = try b.create(tokens * k * ff * 4);
        const y = try b.create(tokens * k * width * 4);
        const out = try b.create(tokens * width * 4);
        const scratch = try alloc.alloc(f32, (inference.cpu.experts.Ffn{ .gate_up = gate_up, .down = down }).scratchLen());
        defer alloc.free(scratch);
        const acc = try alloc.alloc(f64, width);
        defer alloc.free(acc);
        const expected = try alloc.alloc(f32, width);
        defer alloc.free(expected);
        var worst: f64 = 0;
        for ([_]bool{ true, false }) |scaled| {
            try b.begin();
            try b.route(logits, experts, k, tokens, experts, indices, weights);
            for (0..tokens) |t| {
                try b.matvecExperts(gu_weights, gate_up, indices.slice(t * k * 4, k * 4), k, x.slice(t * width * 4, width * 4), 0, gu_out.slice(t * k * 2 * ff * 4, k * 2 * ff * 4), 2 * ff);
            }
            try b.geluMulRows(gu_out, gu_out.slice(ff * 4, gu_out.len - ff * 4), hidden, ff, tokens * k, 2 * ff, 2 * ff, ff);
            for (0..tokens) |t| {
                try b.matvecExperts(down_weights, down, indices.slice(t * k * 4, k * 4), k, hidden.slice(t * k * ff * 4, k * ff * 4), ff, y.slice(t * k * width * 4, k * width * 4), width);
            }
            try b.combineExperts(y, weights, indices, if (scaled) scales else null, out, .{ .columns = width, .slots = k, .rows = tokens, .experts = experts, .in_stride = width, .out_stride = width });
            try b.commit();
            var cpu_indices: [3]u32 = undefined;
            var cpu_weights: [3]f32 = undefined;
            for (0..tokens) |t| {
                try inference.cpu.experts.route(logits.floats()[t * experts ..][0..experts], &cpu_indices, &cpu_weights);
                try std.testing.expectEqualSlices(u32, &cpu_indices, @as([*]const u32, @ptrCast(@alignCast(indices.host)))[t * k ..][0..k]);
                try inference.cpu.experts.ffn(.{ .gate_up = gate_up, .down = down, .down_scale = if (scaled) scales.floats() else null }, x.floats()[t * width ..][0..width], &cpu_indices, &cpu_weights, expected, scratch, acc);
                var magnitude: f64 = 0;
                for (expected) |v| magnitude = @max(magnitude, @abs(v));
                const tolerance: f64 = 2e-5 * (1 + magnitude);
                for (out.floats()[t * width ..][0..width], expected) |got, want| {
                    worst = @max(worst, @abs(@as(f64, got) - want) / (1 + magnitude));
                    expectClose("expert chain", got, want, @floatCast(tolerance)) catch |err| {
                        std.debug.print("  token {d}, scaled {}\n", .{ t, scaled });
                        return err;
                    };
                }
            }
        }
        std.debug.print("Expert decode chain (route, gathered gate-up, gelu rows, gathered down, combine) vs CPU F64 over 3 tokens: worst |difference| / (1 + max|y|) {e:.3} (bound 2e-5)\n", .{worst});
        // Contract: the pair output must not alias its inputs; combine needs every slot's weight.
        if (b.geluMulRows(gu_out, gu_out.slice(ff * 4, gu_out.len - ff * 4), gu_out, ff, tokens * k, 2 * ff, 2 * ff, ff)) |_| return error.ExpectedInvalidShape else |err| if (err != error.InvalidShape) return err;
        if (b.combineExperts(y, weights.slice(0, 8), indices, null, out, .{ .columns = width, .slots = k, .rows = tokens, .experts = experts, .in_stride = width, .out_stride = width })) |_| return error.ExpectedInvalidShape else |err| if (err != error.InvalidShape) return err;
        if (b.combineExperts(y, weights, indices, scales.slice(0, 8), out, .{ .columns = width, .slots = k, .rows = tokens, .experts = experts, .in_stride = width, .out_stride = width })) |_| return error.ExpectedInvalidShape else |err| if (err != error.InvalidShape) return err;
    }
    // The prefill path: route, expert lists, gathered gate-up tiles, gelu
    // rows, gathered down tiles, combine, over a chunk of 45 tokens (135
    // slot rows: not a multiple of 32), six experts, k 3, expert 0 taking
    // every token (two tiles, one partial) and expert 5 none; both tile
    // kernels against `cpu.experts.ffn` per token and against the decode
    // path row by row; then a chunk of one token; then the contracts.
    {
        const experts: usize = 6;
        const k: usize = 3;
        const width: usize = 256;
        const ff: usize = 704;
        const chunk: usize = 45;
        const n = chunk * k;
        const gu_region = try tiledMatrix(alloc, sample_bytes, 2, experts * 2 * ff, width);
        defer alloc.free(gu_region);
        const down_region = try tiledMatrix(alloc, sample_bytes, 2, experts * width, ff);
        defer alloc.free(down_region);
        tameScales(gu_region, 2);
        tameScales(down_region, 2);
        const gate_up: inference.cpu.ExpertMatrix = .{ .encoding = 2, .experts = experts, .rows = 2 * ff, .columns = width, .bytes = gu_region };
        const down: inference.cpu.ExpertMatrix = .{ .encoding = 2, .experts = experts, .rows = width, .columns = ff, .bytes = down_region };
        const gu_weights = try uploadBytes(b, gu_region);
        const down_weights = try uploadBytes(b, down_region);
        const scales = try b.create(experts * 4);
        for (scales.floats()) |*v| v.* = random.float(f32) + 0.5;
        const logits = try b.create(chunk * experts * 4);
        for (0..chunk) |t| for (0..experts) |e| {
            logits.floats()[t * experts + e] = if (e == 0) 8 else if (e == 5) -50 else random.floatNorm(f32);
        };
        const x = try b.create(chunk * width * 4);
        for (x.floats()) |*v| v.* = random.floatNorm(f32);
        const layout = Backend.expertListsLayout(n, experts);
        const indices = try b.create(n * 4);
        const weights = try b.create(n * 4);
        const lists = try b.create(layout.words * 4);
        const gu_out = try b.create(n * 2 * ff * 4);
        const hidden = try b.create(n * ff * 4);
        const y = try b.create(n * width * 4);
        const out = try b.create(chunk * width * 4);
        const decode_out = try b.create(chunk * width * 4);
        const scratch = try alloc.alloc(f32, (inference.cpu.experts.Ffn{ .gate_up = gate_up, .down = down }).scratchLen());
        defer alloc.free(scratch);
        const acc = try alloc.alloc(f64, width);
        defer alloc.free(acc);
        const expected = try alloc.alloc(f32, chunk * width);
        defer alloc.free(expected);
        // The decode path on the same chunk: the tiles must agree with it row by row.
        try b.begin();
        try b.route(logits, experts, k, chunk, experts, indices, weights);
        for (0..chunk) |t| try b.matvecExperts(gu_weights, gate_up, indices.slice(t * k * 4, k * 4), k, x.slice(t * width * 4, width * 4), 0, gu_out.slice(t * k * 2 * ff * 4, k * 2 * ff * 4), 2 * ff);
        try b.geluMulRows(gu_out, gu_out.slice(ff * 4, gu_out.len - ff * 4), hidden, ff, n, 2 * ff, 2 * ff, ff);
        for (0..chunk) |t| try b.matvecExperts(down_weights, down, indices.slice(t * k * 4, k * 4), k, hidden.slice(t * k * ff * 4, k * ff * 4), ff, y.slice(t * k * width * 4, k * width * 4), width);
        try b.combineExperts(y, weights, indices, scales, decode_out, .{ .columns = width, .slots = k, .rows = chunk, .experts = experts, .in_stride = width, .out_stride = width });
        try b.commit();
        const idx = @as([*]const u32, @ptrCast(@alignCast(indices.host)))[0..n];
        var chunk_max: f64 = 0;
        {
            var cpu_indices: [3]u32 = undefined;
            var cpu_weights: [3]f32 = undefined;
            for (0..chunk) |t| {
                try inference.cpu.experts.route(logits.floats()[t * experts ..][0..experts], &cpu_indices, &cpu_weights);
                try std.testing.expectEqualSlices(u32, &cpu_indices, idx[t * k ..][0..k]);
                try inference.cpu.experts.ffn(.{ .gate_up = gate_up, .down = down, .down_scale = scales.floats() }, x.floats()[t * width ..][0..width], &cpu_indices, &cpu_weights, expected[t * width ..][0..width], scratch, acc);
            }
            for (expected) |v| chunk_max = @max(chunk_max, @abs(v));
        }
        const words = @as([*]u32, @ptrCast(@alignCast(lists.host)))[0..layout.words];
        var worst_cpu: [2]f64 = .{ 0, 0 };
        var worst_decode: [2]f64 = .{ 0, 0 };
        for ([_]bool{ false, true }, 0..) |generic, which| {
            b.generic_only = generic;
            for (out.floats()) |*v| v.* = std.math.nan(f32);
            @memset(words, 0xffffffff);
            try b.begin();
            try b.expertLists(indices, chunk, k, experts, lists);
            try b.matmulExperts(gu_weights, gate_up, lists, chunk, k, x, width, k, gu_out, 2 * ff);
            try b.geluMulRows(gu_out, gu_out.slice(ff * 4, gu_out.len - ff * 4), hidden, ff, n, 2 * ff, 2 * ff, ff);
            try b.matmulExperts(down_weights, down, lists, chunk, k, hidden, ff, 1, y, width);
            try b.combineExperts(y, weights, indices, scales, out, .{ .columns = width, .slots = k, .rows = chunk, .experts = experts, .in_stride = width, .out_stride = width });
            try b.commit();
            b.generic_only = false;
            // The lists: the skew, the offsets, every tile's (expert, first,
            // count), and the row list as a permutation grouped by expert.
            var counts: [experts]usize = @splat(0);
            for (idx) |e| counts[e] += 1;
            if (counts[0] != chunk or counts[5] != 0) return error.SkewNotProduced;
            var offset: usize = 0;
            var tiles: usize = 0;
            for (0..experts) |e| {
                if (words[2 + e] != offset) return error.ListsMismatch;
                var t: usize = 0;
                while (t < counts[e]) : (t += 32) {
                    const entry = words[layout.tiles_at + 3 * tiles ..][0..3];
                    if (entry[0] != e or entry[1] != offset + t or entry[2] != @min(32, counts[e] - t)) return error.ListsMismatch;
                    tiles += 1;
                }
                for (words[layout.rows_at + offset ..][0..counts[e]]) |s| if (s >= n or idx[s] != e) return error.ListsMismatch;
                offset += counts[e];
            }
            if (words[0] != tiles or words[1] != n or words[2 + experts] != n) return error.ListsMismatch;
            const seen = try alloc.alloc(bool, n);
            defer alloc.free(seen);
            @memset(seen, false);
            for (words[layout.rows_at..][0..n]) |s| {
                if (seen[s]) return error.ListsMismatch;
                seen[s] = true;
            }
            // Outputs: the F32 tiles differ from the CPU by rounding order;
            // the half tiles round both operands of both projections.
            const bound: f64 = if (generic) 2e-5 else 2e-3;
            for (out.floats(), expected, decode_out.floats(), 0..) |got, want, dec, i| {
                worst_cpu[which] = @max(worst_cpu[which], @abs(@as(f64, got) - want) / chunk_max);
                worst_decode[which] = @max(worst_decode[which], @abs(@as(f64, got) - dec) / chunk_max);
                expectClose(if (generic) "expert prefill (generic tile)" else "expert prefill (half tile)", got, want, @floatCast(bound * chunk_max)) catch |err| {
                    std.debug.print("  token {d}, column {d}\n", .{ i / width, i % width });
                    return err;
                };
            }
        }
        std.debug.print("Expert prefill chain (lists, gathered gate-up and down tiles) over 45 tokens vs CPU F64: worst |difference| / max|y| F32 tile {e:.3} (bound 2e-5), half tile {e:.3} (bound 2e-3); vs the decode path {e:.3} and {e:.3}\n", .{ worst_cpu[1], worst_cpu[0], worst_decode[1], worst_decode[0] });
        // A chunk of one token: three slot rows, every tile partial; the
        // same token as row 0 of the chunk above.
        {
            for (out.floats()) |*v| v.* = std.math.nan(f32);
            try b.begin();
            try b.route(logits, experts, k, 1, experts, indices, weights);
            try b.expertLists(indices, 1, k, experts, lists);
            try b.matmulExperts(gu_weights, gate_up, lists, 1, k, x, width, k, gu_out, 2 * ff);
            try b.geluMulRows(gu_out, gu_out.slice(ff * 4, gu_out.len - ff * 4), hidden, ff, k, 2 * ff, 2 * ff, ff);
            try b.matmulExperts(down_weights, down, lists, 1, k, hidden, ff, 1, y, width);
            try b.combineExperts(y, weights, indices, scales, out, .{ .columns = width, .slots = k, .rows = 1, .experts = experts, .in_stride = width, .out_stride = width });
            try b.commit();
            if (words[0] != k) return error.ListsMismatch;
            for (out.floats()[0..width], expected[0..width]) |got, want| try expectClose("expert prefill (one token)", got, want, @floatCast(2e-3 * chunk_max));
        }
        // Contracts: rows not a multiple of 8, a short lists buffer, an
        // unaligned input row stride, a zero input group, too many experts.
        var bad = gate_up;
        bad.rows = 2 * ff - 4;
        bad.bytes = gu_region[0 .. gu_region.len / (2 * ff) * (2 * ff - 4)];
        if (b.matmulExperts(gu_weights, bad, lists, chunk, k, x, width, k, gu_out, 2 * ff)) |_| return error.ExpectedInvalidShape else |err| if (err != error.InvalidShape) return err;
        if (b.matmulExperts(gu_weights, gate_up, lists.slice(0, layout.words * 4 - 4), chunk, k, x, width, k, gu_out, 2 * ff)) |_| return error.ExpectedInvalidShape else |err| if (err != error.InvalidShape) return err;
        if (b.matmulExperts(gu_weights, gate_up, lists, chunk, k, x, width + 2, k, gu_out, 2 * ff)) |_| return error.ExpectedInvalidShape else |err| if (err != error.InvalidShape) return err;
        if (b.matmulExperts(gu_weights, gate_up, lists, chunk, k, x, width, 0, gu_out, 2 * ff)) |_| return error.ExpectedInvalidShape else |err| if (err != error.InvalidShape) return err;
        if (b.expertLists(indices, chunk, k, 257, lists)) |_| return error.ExpectedInvalidShape else |err| if (err != error.InvalidShape) return err;
        if (b.expertLists(indices, chunk, k, experts, lists.slice(0, layout.words * 4 - 4))) |_| return error.ExpectedInvalidShape else |err| if (err != error.InvalidShape) return err;
    }
}

/// `--experts-bench [CHUNK]`: achieved bandwidth of the gathered expert
/// kernels on the 26B-A4B shape (128 experts, 8 selected; gate-up 1,408 ×
/// 2,816 and down 2,816 × 704 per expert, Q4_0): GB/s of the **selected**
/// experts' bytes per dispatch, 64 dispatches per command buffer, best and
/// mean of five, beside a dense matvec over the same byte count (the rate a
/// gathered kernel could at most reach); then the prefill tiles over a
/// chunk of CHUNK tokens (256 by default), the bytes counted once per tile.
fn expertsBench(alloc: std.mem.Allocator, chunk: usize) !void {
    var backend = try openBackend(alloc);
    defer backend.deinit();
    const b = &backend;
    const fixtures = try std.json.parseFromSlice(QuantFixture, alloc, @embedFile("src/quant/fixtures/simple.json"), .{ .ignore_unknown_fields = true });
    defer fixtures.deinit();
    var q4_0: ?[]const u8 = null;
    for (fixtures.value.rows) |sample| if (sample.encoding == 2) {
        q4_0 = sample.bytes;
        break;
    };
    const sample_bytes = q4_0 orelse return error.FixtureMissing;
    const experts: usize = 128;
    const k: usize = 8;
    const width: usize = 2816;
    const ff: usize = 704;
    const gu_region = try tiledMatrix(alloc, sample_bytes, 2, experts * 2 * ff, width);
    defer alloc.free(gu_region);
    const down_region = try tiledMatrix(alloc, sample_bytes, 2, experts * width, ff);
    defer alloc.free(down_region);
    const gate_up: inference.cpu.ExpertMatrix = .{ .encoding = 2, .experts = experts, .rows = 2 * ff, .columns = width, .bytes = gu_region };
    const down: inference.cpu.ExpertMatrix = .{ .encoding = 2, .experts = experts, .rows = width, .columns = ff, .bytes = down_region };
    const gu_weights = try uploadBytes(b, gu_region);
    const down_weights = try uploadBytes(b, down_region);
    const indices = try b.create(k * 4);
    @memcpy(@as([*]u32, @ptrCast(@alignCast(indices.host)))[0..k], &[_]u32{ 3, 17, 40, 41, 77, 90, 100, 127 });
    const weights = try b.create(k * 4);
    for (weights.floats()) |*v| v.* = 0.125;
    const x = try b.create(width * 4);
    for (x.floats(), 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 13)) / 13 - 0.5;
    const gu_out = try b.create(k * 2 * ff * 4);
    const hidden = try b.create(k * ff * 4);
    const y = try b.create(k * width * 4);
    const out = try b.create(width * 4);
    const gu_bytes = @as(f64, @floatFromInt(gu_region.len / experts * k));
    const down_bytes = @as(f64, @floatFromInt(down_region.len / experts * k));
    // Dense equivalents: one matrix holding the selected experts' bytes.
    const dense_gu: inference.cpu.Matrix = .{ .encoding = 2, .rows = k * 2 * ff, .columns = width, .bytes = gu_region[0 .. gu_region.len / experts * k] };
    const dense_down: inference.cpu.Matrix = .{ .encoding = 2, .rows = width, .columns = k * ff, .bytes = down_region[0 .. down_region.len / experts * k] };
    const dense_x = try b.create(k * ff * 4);
    for (dense_x.floats(), 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 13)) / 13 - 0.5;
    const dense_out = try b.create(k * 2 * ff * 4);
    // Prefill: a chunk routed by the router over random logits (at 256
    // tokens 2,048 slot rows, every expert touched, 16 rows per expert on
    // average), the lists built once; the bytes are what the tiles read,
    // one expert matrix per tile.
    if (chunk == 0 or chunk > 4096) return error.InvalidChunk;
    const n = chunk * k;
    const layout = Backend.expertListsLayout(n, experts);
    const logits = try b.create(chunk * experts * 4);
    var prng = std.Random.DefaultPrng.init(0x26b);
    for (logits.floats()) |*v| v.* = prng.random().floatNorm(f32);
    const chunk_indices = try b.create(n * 4);
    const chunk_weights = try b.create(n * 4);
    const lists = try b.create(layout.words * 4);
    const chunk_x = try b.create(chunk * width * 4);
    for (chunk_x.floats(), 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 13)) / 13 - 0.5;
    const chunk_gu_out = try b.create(n * 2 * ff * 4);
    const chunk_hidden = try b.create(n * ff * 4);
    const chunk_y = try b.create(n * width * 4);
    const chunk_out = try b.create(chunk * width * 4);
    try b.begin();
    try b.route(logits, experts, k, chunk, experts, chunk_indices, chunk_weights);
    try b.expertLists(chunk_indices, chunk, k, experts, lists);
    try b.commit();
    const tiles = @as([*]const u32, @ptrCast(@alignCast(lists.host)))[0];
    const tile_gu_bytes = @as(f64, @floatFromInt(gu_region.len / experts)) * @as(f64, @floatFromInt(tiles));
    const tile_down_bytes = @as(f64, @floatFromInt(down_region.len / experts)) * @as(f64, @floatFromInt(tiles));
    const Case = struct { name: []const u8, bytes: f64, which: enum { gate_up, down, chain, dense_gate_up, dense_down, prefill_gate_up, prefill_down, prefill_chain }, tokens: usize = 0 };
    const cases = [_]Case{
        .{ .name = "gathered gate-up 8 × (1408×2816)", .bytes = gu_bytes, .which = .gate_up },
        .{ .name = "gathered down 8 × (2816×704)", .bytes = down_bytes, .which = .down },
        .{ .name = "expert chain (gate-up, gelu, down, combine)", .bytes = gu_bytes + down_bytes, .which = .chain },
        .{ .name = "dense 11264×2816 (same bytes as gate-up)", .bytes = gu_bytes, .which = .dense_gate_up },
        .{ .name = "dense 2816×5632 (same bytes as down)", .bytes = down_bytes, .which = .dense_down },
        .{ .name = "prefill gate-up tiles", .bytes = tile_gu_bytes, .which = .prefill_gate_up, .tokens = chunk },
        .{ .name = "prefill down tiles", .bytes = tile_down_bytes, .which = .prefill_down, .tokens = chunk },
        .{ .name = "prefill chain (lists, tiles, gelu, combine)", .bytes = tile_gu_bytes + tile_down_bytes, .which = .prefill_chain, .tokens = chunk },
    };
    const rounds = 5;
    const repeats = 64;
    std.debug.print("prefill chunk of {d} tokens: {d} slot rows in {d} tiles of 32 over {d} experts\n", .{ chunk, n, tiles, experts });
    std.debug.print("{s:<46} {s:>7} {s:>9} {s:>9}  rounds (GB/s of the selected experts' bytes)\n", .{ "case", "MB", "best", "mean" });
    for (cases) |case| {
        var best: f64 = 0;
        var total: f64 = 0;
        var best_seconds: f64 = std.math.inf(f64);
        var samples: [rounds]f64 = undefined;
        for (0..rounds + 2) |i| {
            const before = b.gpuSeconds();
            try b.begin();
            for (0..repeats) |_| switch (case.which) {
                .gate_up => try b.matvecExperts(gu_weights, gate_up, indices, k, x, 0, gu_out, 2 * ff),
                .down => try b.matvecExperts(down_weights, down, indices, k, hidden, ff, y, width),
                .chain => {
                    try b.matvecExperts(gu_weights, gate_up, indices, k, x, 0, gu_out, 2 * ff);
                    try b.geluMulRows(gu_out, gu_out.slice(ff * 4, gu_out.len - ff * 4), hidden, ff, k, 2 * ff, 2 * ff, ff);
                    try b.matvecExperts(down_weights, down, indices, k, hidden, ff, y, width);
                    try b.combineExperts(y, weights, indices, null, out, .{ .columns = width, .slots = k, .experts = experts, .in_stride = width, .out_stride = width });
                },
                .dense_gate_up => try b.matvec(gu_weights, dense_gu, x, dense_out),
                .dense_down => try b.matvec(down_weights, dense_down, dense_x, dense_out),
                .prefill_gate_up => try b.matmulExperts(gu_weights, gate_up, lists, chunk, k, chunk_x, width, k, chunk_gu_out, 2 * ff),
                .prefill_down => try b.matmulExperts(down_weights, down, lists, chunk, k, chunk_hidden, ff, 1, chunk_y, width),
                .prefill_chain => {
                    try b.expertLists(chunk_indices, chunk, k, experts, lists);
                    try b.matmulExperts(gu_weights, gate_up, lists, chunk, k, chunk_x, width, k, chunk_gu_out, 2 * ff);
                    try b.geluMulRows(chunk_gu_out, chunk_gu_out.slice(ff * 4, chunk_gu_out.len - ff * 4), chunk_hidden, ff, n, 2 * ff, 2 * ff, ff);
                    try b.matmulExperts(down_weights, down, lists, chunk, k, chunk_hidden, ff, 1, chunk_y, width);
                    try b.combineExperts(chunk_y, chunk_weights, chunk_indices, null, chunk_out, .{ .columns = width, .slots = k, .rows = chunk, .experts = experts, .in_stride = width, .out_stride = width });
                },
            };
            try b.commit();
            const seconds = (b.gpuSeconds() - before) / @as(f64, @floatFromInt(repeats));
            const rate = case.bytes / 1e9 / seconds;
            if (i < 2) continue; // warm-up
            samples[i - 2] = rate;
            best = @max(best, rate);
            best_seconds = @min(best_seconds, seconds);
            total += rate;
        }
        std.debug.print("{s:<46} {d:>7.1} {d:>9.1} {d:>9.1} ", .{ case.name, case.bytes / 1e6, best, total / rounds });
        for (samples) |r| std.debug.print(" {d:.0}", .{r});
        if (case.tokens != 0) std.debug.print("  ({d:.0} us per chunk; {d:.0} tok/s if 30 such layers were the whole cost)", .{ best_seconds * 1e6, @as(f64, @floatFromInt(case.tokens)) / (best_seconds * 30) });
        std.debug.print("\n", .{});
    }
}

/// Merging must preserve each standalone reduction, including heterogeneous
/// encodings and the generic alignment fallback. Packed outputs also exercise
/// multiple byte offsets into one binding, as used by DeltaNet and KV state.
fn checkSegments(alloc: std.mem.Allocator) !void {
    var backend = try openBackend(alloc);
    defer backend.deinit();
    const b = &backend;
    const columns = 1280;
    const rows = 48;
    const input = try b.create(columns * 4);
    for (input.floats(), 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 23)) / 23 - 0.5;
    const packed_output = try b.create(12 * rows * 4);
    var list: std.ArrayList(Backend.Segment) = .empty;
    defer list.deinit(alloc);
    inline for (.{ "simple", "k-affine", "k-signed", "iq", "ternary" }) |name| {
        const fixtures = try std.json.parseFromSlice(QuantFixture, alloc, @embedFile("src/quant/fixtures/" ++ name ++ ".json"), .{ .ignore_unknown_fields = true });
        defer fixtures.deinit();
        var seen: [256]bool = @splat(false);
        for (fixtures.value.rows) |sample| {
            if (seen[sample.encoding]) continue;
            seen[sample.encoding] = true;
            const region = try tiledMatrix(alloc, sample.bytes, sample.encoding, rows, columns);
            defer alloc.free(region);
            const weights = try uploadBytes(b, region);
            try list.append(alloc, .{ .weights = weights, .matrix = .{ .rows = rows, .columns = columns, .encoding = sample.encoding, .bytes = weights.host[0..region.len] }, .output = packed_output.slice(list.items.len * rows * 4, rows * 4) });
        }
    }
    const expected = try b.create(rows * 4);
    const second = try b.create(rows * 4);
    for ([_]bool{ false, true }) |generic| {
        b.generic_only = generic;
        for (list.items, 0..) |s, i| {
            const next = list.items[(i + 1) % list.items.len];
            const third = list.items[(i + 2) % list.items.len];
            const fourth = list.items[(i + 3) % list.items.len];
            try b.begin();
            try b.matvec(s.weights, s.matrix, input, expected);
            try b.matvecSegments(&.{ s, next, third, fourth }, input, .plain);
            try b.commit();
            for (s.output.floats()[0..rows], expected.floats()[0..rows]) |got, want| if (got != want) return error.MergedMatvecMismatch;
            var up = next;
            up.output = s.output;
            try b.begin();
            try b.matvec(next.weights, next.matrix, input, second);
            try b.siluMul(expected, second, rows);
            try b.matvecSegments(&.{ s, up }, input, .silu_mul_pair);
            try b.commit();
            for (s.output.floats()[0..rows], expected.floats()[0..rows]) |got, want| if (got != want) return error.MergedSiluMismatch;
            try b.begin();
            try b.matvec(s.weights, s.matrix, input, expected);
            try b.matvec(next.weights, next.matrix, input, second);
            try b.geluMul(expected, second, rows);
            try b.matvecSegments(&.{ s, up }, input, .gelu_mul_pair);
            try b.commit();
            for (s.output.floats()[0..rows], expected.floats()[0..rows]) |got, want| if (got != want) return error.MergedGeluMismatch;
        }
    }
    b.generic_only = false;
    // Misaligned specialized weights must use the identical generic arithmetic.
    var s = list.items[0];
    for (list.items) |item| if (item.matrix.encoding == 12) {
        s = item;
        break;
    };
    const unaligned = try b.create(s.matrix.bytes.len + 8);
    @memcpy(unaligned.host[8..][0..s.matrix.bytes.len], s.matrix.bytes);
    s.weights = unaligned.slice(8, s.matrix.bytes.len);
    try b.begin();
    try b.matvec(s.weights, s.matrix, input, expected);
    try b.matvecSegments(&.{s}, input, .plain);
    try b.commit();
    for (s.output.floats()[0..rows], expected.floats()[0..rows]) |got, want| if (got != want) return error.MergedAlignmentMismatch;
    // Rejections happen before recording work, so no begin/commit is required.
    var bad = s;
    bad.matrix.rows = 40;
    try std.testing.expectError(error.InvalidShape, b.matvecSegments(&.{bad}, input, .plain));
    try std.testing.expectError(error.InvalidShape, b.matvecSegments(&.{ s, s }, input, .plain));
    try std.testing.expectError(error.InvalidShape, b.matvecSegments(&.{s}, input, .silu_mul_pair));
    try std.testing.expectError(error.InvalidShape, b.matvecSegments(&.{s}, input, .gelu_mul_pair));
    try std.testing.expectError(error.InvalidShape, b.matvecSegments(&.{}, input, .plain));
    bad = s;
    bad.output = input;
    try std.testing.expectError(error.InvalidShape, b.matvecSegments(&.{bad}, input, .plain));
    bad = s;
    bad.output.len = 4;
    try std.testing.expectError(error.InvalidShape, b.matvecSegments(&.{bad}, input, .plain));
    bad = s;
    bad.matrix.columns += 16;
    try std.testing.expectError(error.InvalidShape, b.matvecSegments(&.{ s, bad }, input, .plain));
    try std.testing.expectError(error.InvalidShape, b.matvecSegments(&.{ s, s, s, s, s }, input, .plain));
    // Four independent weights and outputs exceed the six non-input bindings.
    var too_many = [4]Backend.Segment{ list.items[0], list.items[1], list.items[2], list.items[3] };
    for (&too_many) |*entry| entry.output = try b.create(rows * 4);
    try std.testing.expectError(error.InvalidShape, b.matvecSegments(&too_many, input, .plain));
    // Three independent outputs use all seven data slots (attention layout).
    var three = [3]Backend.Segment{ list.items[0], list.items[1], list.items[2] };
    for (&three) |*entry| entry.output = try b.create(rows * 4);
    var diagnostic: [8192]u8 = @splat(0);
    try b.enableProfiling(16, &diagnostic);
    try b.begin();
    try b.matvecSegments(&three, input, .plain);
    try b.commit();
    const total = b.profile.?.totals.get(.{ .kernel = .matvec_segments, .encoding = null, .rows = 3 * rows, .columns = columns }) orelse return error.ProfileKeyMissing;
    const expected_bytes = three[0].matrix.bytes.len + three[1].matrix.bytes.len + three[2].matrix.bytes.len;
    if (total.dispatches != 1 or total.bytes != expected_bytes) return error.ProfileTotalsMismatch;
    for (three) |entry| {
        try b.begin();
        try b.matvec(entry.weights, entry.matrix, input, expected);
        try b.commit();
        for (entry.output.floats()[0..rows], expected.floats()[0..rows]) |got, want| if (got != want) return error.MergedMatvecMismatch;
    }
}

/// The attention geometry Gemma 4 adds — the sliding window on prefill
/// chunks (`window`), 16 query heads of 512 channels over one KV head (the
/// 12B) or two (the 26B-A4B): the wide decode instantiation and the chunk
/// kernel's value-column splits — against the F64 CPU reference per query
/// row, both cache precisions (F16 over the rounded operands, as in 8d).
/// Scores are unscaled as in the model. For the window, the reference sees
/// only the cache rows `[pos + 1 − window, pos]` of each row.
/// `--hadamard-bench`: GPU time of the transforms one Bonsai token needs
/// at batch 1 — per layer the four rotated activations (5,120, 6,144,
/// 5,120, 17,408) over 64 layers plus the embedding inverse and the
/// output-head input — as 258 dispatches in one command buffer, best and
/// mean of five after two warm-ups; and the time per 1,024-block from a
/// 17,408-wide, 64-row dispatch. What the separate kernel costs a token,
/// against which any fusion is judged.
fn hadamardBench(alloc: std.mem.Allocator) !void {
    var backend = try openBackend(alloc);
    defer backend.deinit();
    const b = &backend;
    const data = try b.create(17408 * 64 * 4);
    for (data.floats(), 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 13)) / 13 - 0.5;
    const signs = try b.create(17408 * 4);
    for (signs.floats(), 0..) |*s, i| s.* = if (i % 3 == 0) -1 else 1;
    const rounds = 5;
    var best: f64 = 1e9;
    var total: f64 = 0;
    for (0..rounds + 2) |i| {
        const before = b.gpuSeconds();
        try b.begin();
        try b.hadamard(data, signs, 5120, 1, 5120, true);
        for (0..64) |_| {
            for ([_]usize{ 5120, 6144, 5120, 17408 }) |width| try b.hadamard(data, signs, width, 1, width, false);
        }
        try b.hadamard(data, signs, 5120, 1, 5120, false);
        try b.commit();
        const ms = (b.gpuSeconds() - before) * 1e3;
        if (i < 2) continue;
        best = @min(best, ms);
        total += ms;
    }
    var block_best: f64 = 1e9;
    for (0..rounds + 2) |i| {
        const before = b.gpuSeconds();
        try b.begin();
        for (0..8) |_| try b.hadamard(data, signs, 17408, 64, 17408, false);
        try b.commit();
        const us = (b.gpuSeconds() - before) * 1e6 / (8 * 64 * 17);
        if (i < 2) continue;
        block_best = @min(block_best, us);
    }
    std.debug.print("hadamard: one token's 258 transforms (64 x {{5120, 6144, 5120, 17408}} + 2 x 5120, 2,197,504 elements) best {d:.3} ms, mean {d:.3} ms over {d} command buffers; {d:.3} us per 1,024-block in a 64-row 17,408-wide dispatch\n", .{ best, total / rounds, rounds, block_best });
}

/// The Hadamard transform against `cpu.hadamard` on the model's three widths
/// over strided rows, forward and inverse, and the round trip; F32
/// butterflies against the F64 reference within rounding.
fn checkHadamard(alloc: std.mem.Allocator) !void {
    var backend = try openBackend(alloc);
    defer backend.deinit();
    const b = &backend;
    var prng = std.Random.DefaultPrng.init(0x4ada);
    const random = prng.random();
    const rows: usize = 3;
    var worst: f64 = 0;
    for ([_]usize{ 5120, 6144, 17408 }) |width| {
        const stride = width + 64;
        const signs = try alloc.alloc(f32, width);
        defer alloc.free(signs);
        for (signs) |*s| s.* = if (random.boolean()) 1 else -1;
        const original = try alloc.alloc(f32, rows * stride);
        defer alloc.free(original);
        for (original) |*x| x.* = random.float(f32) * 4 - 2;
        const expected = try alloc.dupe(f32, original);
        defer alloc.free(expected);
        for (0..rows) |r| try inference.cpu.hadamard.forward(expected[r * stride ..][0..width], signs, Backend.hadamard_block);
        const data = try upload(b, original);
        const sign_buffer = try upload(b, signs);
        try b.begin();
        try b.hadamard(data, sign_buffer, width, rows, stride, false);
        try b.commit();
        for (0..rows) |r| for (0..stride) |c| {
            const got = data.floats()[r * stride + c];
            const want = expected[r * stride + c];
            // Ten F32 butterfly stages over values up to 2 in magnitude, against F64.
            if (c < width) worst = @max(worst, @abs(@as(f64, got) - want));
            try expectClose("hadamard forward", got, want, 3e-5);
        };
        try b.begin();
        try b.hadamard(data, sign_buffer, width, rows, stride, true);
        try b.commit();
        for (data.floats(), original) |got, want| try expectClose("hadamard round trip", got, want, 3e-5);
        // The inverse alone against the CPU (the embedding path).
        @memcpy(data.floats(), original);
        for (0..rows) |r| try inference.cpu.hadamard.inverse(expected[r * stride ..][0..width], signs, Backend.hadamard_block);
        @memcpy(expected[0..], original);
        for (0..rows) |r| try inference.cpu.hadamard.inverse(expected[r * stride ..][0..width], signs, Backend.hadamard_block);
        try b.begin();
        try b.hadamard(data, sign_buffer, width, rows, stride, true);
        try b.commit();
        for (data.floats(), expected) |got, want| try expectClose("hadamard inverse", got, want, 3e-5);
        if (b.hadamard(data, sign_buffer, width - 64, rows, stride, false)) |_| return error.ExpectedInvalidShape else |err| if (err != error.InvalidShape) return err;
        if (b.hadamard(data.slice(4, data.len - 4), sign_buffer, width, rows, stride, false)) |_| return error.ExpectedInvalidShape else |err| if (err != error.InvalidShape) return err;
    }
    std.debug.print("Hadamard transform vs cpu.hadamard on 5,120 / 6,144 / 17,408 over strided rows, forward, inverse, and the round trip: max abs {e:.3} (bound 3e-5)\n", .{worst});
}

/// The row gather on the Bonsai value-head regrouping (48 heads of 128,
/// tiled `rep * 16 + nk` to grouped `nk * 3 + rep`) over strided rows,
/// exact, plus the overlap and map-size refusals.
fn checkGatherRows(alloc: std.mem.Allocator) !void {
    var backend = try openBackend(alloc);
    defer backend.deinit();
    const b = &backend;
    const width: usize = 128;
    const groups: usize = 48;
    const rows: usize = 3;
    const in_stride = groups * width + 64;
    const out_stride = groups * width + 128;
    var map: [48]u32 = undefined;
    for (&map, 0..) |*m, g| m.* = @intCast((g % 3) * 16 + g / 3);
    const src = try alloc.alloc(f32, rows * in_stride);
    defer alloc.free(src);
    for (src, 0..) |*x, i| x.* = @floatFromInt(i);
    const src_buffer = try upload(b, src);
    const dst_buffer = try b.create(rows * out_stride * 4);
    @memset(dst_buffer.floats(), -1);
    const map_buffer = try uploadBytes(b, std.mem.sliceAsBytes(&map));
    try b.begin();
    try b.gatherRows(dst_buffer, src_buffer, map_buffer, width, groups, rows, in_stride, out_stride);
    try b.commit();
    for (0..rows) |r| for (0..out_stride) |c| {
        const got = dst_buffer.floats()[r * out_stride + c];
        const want: f32 = if (c < groups * width) src[r * in_stride + map[c / width] * width + c % width] else -1;
        if (got != want) {
            std.debug.print("gather rows: row {d} column {d} got {d} want {d}\n", .{ r, c, got, want });
            return error.MetalMismatch;
        }
    };
    if (b.gatherRows(src_buffer, src_buffer, map_buffer, width, groups, rows, in_stride, in_stride)) |_| return error.ExpectedInvalidShape else |err| if (err != error.InvalidShape) return err;
    if (b.gatherRows(dst_buffer, src_buffer, map_buffer.slice(0, 47 * 4), width, groups, rows, in_stride, out_stride)) |_| return error.ExpectedInvalidShape else |err| if (err != error.InvalidShape) return err;
    if (b.gatherRows(dst_buffer, src_buffer, map_buffer, width, groups, rows, groups * width - 1, out_stride)) |_| return error.ExpectedInvalidShape else |err| if (err != error.InvalidShape) return err;
    std.debug.print("row gather on the 48-head regrouping over strided rows: exact; overlap, short map, and short stride refused\n", .{});
}

fn checkWindowedAndWideAttention(alloc: std.mem.Allocator, b: *Backend) !void {
    var prng = std.Random.DefaultPrng.init(0x6e44a);
    const random = prng.random();
    const Geometry = struct { qh: usize, kvh: usize, hd: usize };
    const sliding: Geometry = .{ .qh = 16, .kvh = 8, .hd = 256 };
    const global: Geometry = .{ .qh = 16, .kvh = 1, .hd = 512 };
    const global_two: Geometry = .{ .qh = 16, .kvh = 2, .hd = 512 };
    const Case = struct { g: Geometry, position: usize, count: usize, window: usize, half: bool };
    // Chunk cases: a window entirely before the chunk, a window opening
    // inside the chunk, a tiny window with fully hidden key tiles (the −∞
    // guard), and the wide geometries without a window (two value splits).
    const cases = [_]Case{
        .{ .g = sliding, .position = 1023, .count = 256, .window = 1024, .half = false },
        .{ .g = sliding, .position = 900, .count = 300, .window = 1024, .half = false },
        .{ .g = sliding, .position = 0, .count = 40, .window = 8, .half = false },
        .{ .g = sliding, .position = 33, .count = 70, .window = 8, .half = true },
        .{ .g = sliding, .position = 1023, .count = 64, .window = 1024, .half = true },
        .{ .g = global, .position = 5, .count = 37, .window = 0, .half = false },
        .{ .g = global, .position = 300, .count = 40, .window = 0, .half = true },
        .{ .g = global_two, .position = 7, .count = 41, .window = 0, .half = false },
        .{ .g = global_two, .position = 300, .count = 40, .window = 0, .half = true },
    };
    var worst_f32: f32 = 0;
    var worst_f16: f32 = 0;
    for (cases) |case| {
        const g = case.g;
        const qw = g.qh * g.hd;
        const kvw = g.kvh * g.hd;
        const total = case.position + case.count;
        const rows = Backend.attentionChunkRows(case.count);
        const k32 = try b.create(total * kvw * 4);
        const v32 = try b.create(total * kvw * 4);
        const q32 = try b.create(rows * qw * 4);
        for (k32.floats()) |*k| k.* = random.floatNorm(f32) * 0.3;
        for (v32.floats()) |*v| v.* = random.floatNorm(f32) * 0.5;
        for (q32.floats()) |*q| q.* = random.floatNorm(f32) * 0.3;
        const out = try b.create(rows * qw * 4);
        for (out.floats()) |*o| o.* = std.math.nan(f32);
        var keys = k32;
        var values = v32;
        var queries = q32;
        if (case.half) {
            keys = try b.create(total * kvw * 2);
            values = try b.create(total * kvw * 2);
            queries = try b.create(rows * qw * 2);
            try b.begin();
            try b.packHalf(&.{ .{ .dst = keys, .src = k32, .count = total * kvw }, .{ .dst = values, .src = v32, .count = total * kvw } });
            try b.packHalf(&.{.{ .dst = queries, .src = q32, .count = rows * qw }});
            try b.commit();
            for ([_]struct { h: Buffer, f: Buffer }{ .{ .h = keys, .f = k32 }, .{ .h = values, .f = v32 }, .{ .h = queries, .f = q32 } }) |pair| {
                const halves = @as([*]const u16, @ptrCast(@alignCast(pair.h.host)))[0 .. pair.h.len / 2];
                for (halves, pair.f.floats()) |h, *f| f.* = @as(f16, @bitCast(h));
            }
        }
        try b.begin();
        try b.attentionChunk(keys, values, queries, out, .{ .query_heads = g.qh, .kv_heads = g.kvh, .key_width = g.hd, .value_width = g.hd, .position = case.position, .count = case.count, .q_stride = qw, .out_stride = qw, .scale = 1.0, .precision = if (case.half) .f16 else .f32, .window = case.window });
        try b.commit();
        const scratch = try alloc.alloc(f64, total);
        defer alloc.free(scratch);
        const expected = try alloc.alloc(f32, qw);
        defer alloc.free(expected);
        for (0..case.count) |t| {
            const pos = case.position + t;
            const lo = if (case.window != 0 and pos + 1 > case.window) pos + 1 - case.window else 0;
            const visible = pos + 1 - lo;
            try inference.cpu.attention.apply(.{ .query_heads = g.qh, .kv_heads = g.kvh, .key_width = g.hd, .value_width = g.hd, .tokens = visible, .visible_tokens = visible, .scale = 1.0, .queries = q32.floats()[t * qw ..][0..qw], .keys = k32.floats()[lo * kvw ..][0 .. visible * kvw], .values = v32.floats()[lo * kvw ..][0 .. visible * kvw] }, expected, scratch);
            for (expected, out.floats()[t * qw ..][0..qw]) |e, a| {
                const d = @abs(a - e);
                if (case.half) worst_f16 = @max(worst_f16, d) else worst_f32 = @max(worst_f32, d);
                expectClose(if (case.half) "windowed/wide half chunk attention" else "windowed/wide chunk attention", a, e, if (case.half) 2e-3 else 1e-5) catch |err| {
                    std.debug.print("  geometry {d}/{d}/{d}, position {d}, count {d}, window {d}, row {d}\n", .{ g.qh, g.kvh, g.hd, case.position, case.count, case.window, t });
                    return err;
                };
            }
        }
    }
    std.debug.print("Windowed and wide chunk attention vs CPU F64 per row: F32 max abs {e:.3} (bound 1e-5), F16 over the rounded operands {e:.3} (bound 2e-3)\n", .{ worst_f32, worst_f16 });
    // Decode: the wide instantiation (16 heads of 512 over one KV head, four
    // head groups per split; over two, two groups per KV head) and the
    // sliding geometry (groups of 2), F32 and F16 (over the rounded rows),
    // at 257 (two splits) and 1,021 visible.
    const partials = try b.create(Backend.attentionDecodePartials(16, 512) * 4);
    var worst_decode: f32 = 0;
    for ([_]Geometry{ global, global_two, sliding }) |g| for ([_]usize{ 257, 1021 }) |visible| {
        const qw = g.qh * g.hd;
        const kvw = g.kvh * g.hd;
        const keys = try b.create(visible * kvw * 4);
        const values = try b.create(visible * kvw * 4);
        const queries = try b.create(qw * 4);
        for (keys.floats()) |*k| k.* = random.floatNorm(f32) * 0.3;
        for (values.floats()) |*v| v.* = random.floatNorm(f32) * 0.5;
        for (queries.floats()) |*q| q.* = random.floatNorm(f32) * 0.3;
        const out = try b.create(qw * 4);
        const scratch = try alloc.alloc(f64, visible);
        defer alloc.free(scratch);
        const expected = try alloc.alloc(f32, qw);
        defer alloc.free(expected);
        for ([_]bool{ false, true }) |half| {
            var k = keys;
            var v = values;
            if (half) {
                k = try b.create(visible * kvw * 2);
                v = try b.create(visible * kvw * 2);
                try b.begin();
                try b.packHalf(&.{ .{ .dst = k, .src = keys, .count = visible * kvw }, .{ .dst = v, .src = values, .count = visible * kvw } });
                try b.commit();
                for ([_]struct { h: Buffer, f: Buffer }{ .{ .h = k, .f = keys }, .{ .h = v, .f = values } }) |pair| {
                    const halves = @as([*]const u16, @ptrCast(@alignCast(pair.h.host)))[0 .. pair.h.len / 2];
                    for (halves, pair.f.floats()) |hv, *f| f.* = @as(f16, @bitCast(hv));
                }
            }
            try inference.cpu.attention.apply(.{ .query_heads = g.qh, .kv_heads = g.kvh, .key_width = g.hd, .value_width = g.hd, .tokens = visible, .visible_tokens = visible, .scale = 1.0, .queries = queries.floats(), .keys = keys.floats(), .values = values.floats() }, expected, scratch);
            for (out.floats()) |*o| o.* = std.math.nan(f32);
            try b.begin();
            try b.attentionDecode(k, v, queries, partials, out, .{ .query_heads = g.qh, .kv_heads = g.kvh, .key_width = g.hd, .value_width = g.hd, .visible = visible, .scale = 1.0, .precision = if (half) .f16 else .f32 });
            try b.commit();
            for (expected, out.floats()) |e, a| {
                worst_decode = @max(worst_decode, @abs(a - e));
                expectClose("wide decode attention", a, e, 5e-5) catch |err| {
                    std.debug.print("  geometry {d}/{d}/{d}, visible {d}, half {}\n", .{ g.qh, g.kvh, g.hd, visible, half });
                    return err;
                };
            }
        }
    };
    std.debug.print("Wide and grouped decode attention vs CPU F64, 257 and 1,021 visible, both precisions: max abs {e:.3} (bound 5e-5)\n", .{worst_decode});
    // Contract: widths above 512 are refused in both kernels.
    const small = try b.create(64);
    try std.testing.expectError(error.InvalidShape, b.attentionDecode(small, small, small, partials, small, .{ .query_heads = 16, .kv_heads = 1, .key_width = 520, .value_width = 520, .visible = 1, .scale = 1.0 }));
    try std.testing.expectError(error.InvalidShape, b.attentionChunk(small, small, small, small, .{ .query_heads = 16, .kv_heads = 1, .key_width = 512, .value_width = 520, .position = 0, .count = 1, .q_stride = 8192, .out_stride = 8192, .scale = 1.0 }));
}

pub fn main(init: std.process.Init) !void {
    const alloc = init.gpa;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--matvec-bench")) return matvecBench(alloc, if (args.len > 2) args[2] else null);
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--matmul-bench")) return matmulBench(alloc, if (args.len > 2) try std.fmt.parseInt(usize, args[2], 10) else 256);
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--hadamard-bench")) return hadamardBench(alloc);
    if (args.len >= 2 and std.mem.eql(u8, args[1], "--experts-bench")) return expertsBench(alloc, if (args.len > 2) try std.fmt.parseInt(usize, args[2], 10) else 256);
    if (args.len != 1) return error.UnknownOption;
    try checkSegments(alloc);
    try checkHadamard(alloc);
    try checkGatherRows(alloc);

    // 1. Every quantized fixture column through the GPU matvec, exact. The
    // backend is recreated per fixture file to exercise object cleanup.
    inline for (.{ "simple", "k-affine", "k-signed", "iq", "ternary" }) |name| {
        var backend = try openBackend(alloc);
        defer backend.deinit();
        const fixtures = try std.json.parseFromSlice(QuantFixture, alloc, @embedFile("src/quant/fixtures/" ++ name ++ ".json"), .{ .ignore_unknown_fields = true });
        defer fixtures.deinit();
        for (fixtures.value.rows) |sample| {
            const region = try uploadBytes(&backend, sample.bytes);
            const input = try alloc.alloc(f32, sample.values.len);
            defer alloc.free(input);
            @memset(input, 0);
            var output: [1]f32 = undefined;
            const matrix: inference.cpu.Matrix = .{ .columns = sample.values.len, .rows = 1, .encoding = sample.encoding, .bytes = sample.bytes };
            const in = try upload(&backend, input);
            const out = try backend.create(4);
            for (0..sample.values.len) |column| {
                in.floats()[column] = 1;
                try backend.begin();
                try backend.matvec(region, matrix, in, out);
                try backend.commit();
                output[0] = out.floats()[0];
                if ((sample.encoding == 11 or sample.encoding == 12 or sample.encoding == 21) and output[0] != sample.values[column]) return error.MetalDecodeMismatch;
                try expectClose("fixture column", output[0], sample.values[column], 0.00001 * @max(1, @abs(sample.values[column])));
                in.floats()[column] = 0;
            }
            // The embedding kernel decodes the whole row: must be bit-identical to the CPU decoder.
            const decoded = try backend.create(sample.values.len * 4);
            try backend.begin();
            try backend.embed(region, matrix, 0, decoded);
            try backend.commit();
            for (decoded.floats()[0..sample.values.len], sample.values) |got, want| if (got != want) {
                std.debug.print("embed decode mismatch for encoding {d}: {d} vs {d}\n", .{ sample.encoding, got, want });
                return error.MetalDecodeMismatch;
            };
        }
    }

    var backend = try openBackend(alloc);
    defer backend.deinit();
    const b = &backend;

    // 2. Randomized rows: tile fixture blocks into matrices of 1,280, 5,120,
    // and 17,408 columns (5, 20, and 68 blocks: a partial and two whole final
    // iterations of the specialized kernels), plus 704 columns for Q4_0 alone
    // (22 of its 32-value blocks, the expert down projection's row: not a
    // whole 256-value stride), by 35 rows (two full threadgroups
    // plus a SIMD group with only three valid rows), and compare F32 GPU
    // accumulation against the F64 CPU reference through both the specialized
    // and the generic kernel. The tolerance is relative to the L1 mass of the
    // products: sum|w*x| * 4e-6 plus a floor.
    {
        var prng = std.Random.DefaultPrng.init(0x5eed);
        const random = prng.random();
        const rows: usize = 35;
        const expected = try alloc.alloc(f32, rows);
        defer alloc.free(expected);
        const actual = try alloc.alloc(f32, rows);
        defer alloc.free(actual);
        for ([_]usize{ 704, 1280, 5120, 17408 }) |columns| {
            const input = try alloc.alloc(f32, columns);
            defer alloc.free(input);
            for (input) |*x| x.* = random.float(f32) * 2 - 1;
            const decoded = try alloc.alloc(f32, columns);
            defer alloc.free(decoded);
            inline for (.{ "simple", "k-affine", "k-signed", "iq", "ternary" }) |name| {
                const fixtures = try std.json.parseFromSlice(QuantFixture, alloc, @embedFile("src/quant/fixtures/" ++ name ++ ".json"), .{ .ignore_unknown_fields = true });
                defer fixtures.deinit();
                var seen: [256]bool = @splat(false);
                for (fixtures.value.rows) |sample| {
                    if (sample.encoding >= seen.len or seen[sample.encoding]) continue;
                    if (columns % 256 != 0 and sample.encoding != 2) continue; // 704: Q4_0 alone
                    seen[sample.encoding] = true;
                    if (columns % 256 != 0 and Backend.specializedMatvec(sample.encoding, 0, columns / 32 * 18, 0) == null) return error.SpecializedPathNotSelected;
                    const region = try tiledMatrix(alloc, sample.bytes, sample.encoding, rows, columns);
                    defer alloc.free(region);
                    const stride = region.len / rows;
                    const matrix: inference.cpu.Matrix = .{ .rows = rows, .columns = columns, .encoding = sample.encoding, .bytes = region };
                    try inference.cpu.matvec(matrix, input, expected, decoded);
                    const weights = try uploadBytes(b, region);
                    for ([_]bool{ false, true }) |generic| {
                        b.generic_only = generic;
                        try matvecOnce(b, matrix, weights, input, actual);
                        for (0..rows) |r| {
                            try inference.quant.row(sample.encoding, region[r * stride ..][0..stride], decoded);
                            var mass: f64 = 0;
                            for (decoded, input) |w, x| mass += @abs(@as(f64, w) * x);
                            try expectClose(if (generic) "wide row (generic)" else "wide row", actual[r], expected[r], @floatCast(mass * 4e-6 + 1e-6));
                        }
                    }
                    b.generic_only = false;
                }
            }
        }
        // 2b. Batched matmul: 40 rows (a full 32-row tile plus a partial
        // one) by the same columns, 37 tokens (two token tiles, the second
        // partial), every fixture encoding. Each token's output row must match
        // the CPU matvec of that token's input within the same tolerance; the
        // padding rows past 37 are never read.
        {
            const mm_rows: usize = 40;
            // 37 tokens select the 64×64 tiles (one full and one partial token
            // tile of 32 under the old geometry; a partial one now), 20 the
            // 32×32 tiles, and 8/5/1 the 8×8 split-K tiles (8 on a full tile,
            // 5 and 1 on a partial one); the buffers hold the larger padding.
            const token_counts = [_]usize{ 37, 20, 16, 9, 8, 5, 1 };
            const padded = Backend.matmulPadded(37);
            const mm_expected = try alloc.alloc(f32, mm_rows);
            defer alloc.free(mm_expected);
            var mm_worst: f64 = 0;
            for ([_]usize{ 1280, 5120 }) |columns| {
                const activations = try b.create(padded * columns * 4);
                for (activations.floats()) |*x| x.* = random.float(f32) * 2 - 1;
                const out = try b.create(padded * mm_rows * 4);
                const generic_out = try b.create(padded * mm_rows * 4);
                const decoded = try alloc.alloc(f32, columns);
                defer alloc.free(decoded);
                inline for (.{ "simple", "k-affine", "k-signed", "iq", "ternary" }) |name| {
                    const fixtures = try std.json.parseFromSlice(QuantFixture, alloc, @embedFile("src/quant/fixtures/" ++ name ++ ".json"), .{ .ignore_unknown_fields = true });
                    defer fixtures.deinit();
                    var seen: [256]bool = @splat(false);
                    for (fixtures.value.rows) |sample| {
                        if (sample.encoding >= seen.len or seen[sample.encoding]) continue;
                        seen[sample.encoding] = true;
                        for (token_counts) |mm_tokens| {
                            const region = try tiledMatrix(alloc, sample.bytes, sample.encoding, mm_rows, columns);
                            defer alloc.free(region);
                            tameScales(region, sample.encoding);
                            const stride = region.len / mm_rows;
                            const matrix: inference.cpu.Matrix = .{ .rows = mm_rows, .columns = columns, .encoding = sample.encoding, .bytes = region };
                            const weights = try uploadBytes(b, region);
                            for (out.floats()) |*v| v.* = std.math.nan(f32);
                            try b.begin();
                            try b.matmul(weights, matrix, activations, columns, out, mm_rows, mm_tokens);
                            try b.commit();
                            // The specialized tile (when the encoding has one) is
                            // compared with the generic F32 tile on the same inputs: its
                            // error against that tile is the half rounding of both
                            // operands alone, bounded relative to Σ|w·x| below.
                            const specialized_kernel = Backend.specializedMatmul(sample.encoding, weights.offset, stride, mm_tokens);
                            if (specialized_kernel != null) {
                                for (generic_out.floats()) |*v| v.* = std.math.nan(f32);
                                b.generic_only = true;
                                try b.begin();
                                try b.matmul(weights, matrix, activations, columns, generic_out, mm_rows, mm_tokens);
                                try b.commit();
                                b.generic_only = false;
                            }
                            const half = if (specialized_kernel) |k| Backend.matmulGeometry(k).half else false;
                            for (0..mm_tokens) |t| {
                                const x = activations.floats()[t * columns ..][0..columns];
                                try inference.cpu.matvec(matrix, x, mm_expected, decoded);
                                for (0..mm_rows) |r| {
                                    try inference.quant.row(sample.encoding, region[r * stride ..][0..stride], decoded);
                                    var mass: f64 = 0;
                                    for (decoded, x) |w, xv| mass += @abs(@as(f64, w) * xv);
                                    const got = out.floats()[t * mm_rows + r];
                                    // F32 tiles differ from the CPU by rounding order only; half
                                    // tiles round both operands to 11 bits (relative 2^-11 each).
                                    expectClose("matmul token row", got, mm_expected[r], @floatCast(mass * @as(f64, if (half) 2e-4 else 4e-6) + 1e-6)) catch |err| {
                                        std.debug.print("  encoding {d}, token {d}, row {d}\n", .{ sample.encoding, t, r });
                                        return err;
                                    };
                                    if (specialized_kernel != null) {
                                        const generic = generic_out.floats()[t * mm_rows + r];
                                        if (half) {
                                            if (mass > 0) mm_worst = @max(mm_worst, @abs(@as(f64, got) - generic) / mass);
                                            try expectClose("matmul half tile vs generic F32 tile", got, generic, @floatCast(mass * 2e-4 + 1e-6));
                                        } else if (@as(u32, @bitCast(got)) != @as(u32, @bitCast(generic))) {
                                            std.debug.print("matmul specialized tile differs from generic (encoding {d}, token {d}, row {d}): {d} vs {d}\n", .{ sample.encoding, t, r, got, generic });
                                            return error.MetalMismatch;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            std.debug.print("Matmul half tiles vs the generic F32 tile: worst |difference| / Σ|w·x| {e:.2} (bound 2e-4)\n", .{mm_worst});
            // Shape rules: rows must be a multiple of 8, columns of 64, and the
            // buffers must hold the padded token rows.
            const small = try b.create(32 * 64 * 4);
            const region = try alloc.alloc(u8, 40 * 64 * 4);
            defer alloc.free(region);
            @memset(region, 0);
            const w = try uploadBytes(b, region);
            const bad_rows: inference.cpu.Matrix = .{ .rows = 36, .columns = 64, .encoding = 0, .bytes = region[0 .. 36 * 64 * 4] };
            if (b.matmul(w, bad_rows, small, 64, small, 64, 1)) |_| return error.ExpectedInvalidShape else |err| if (err != error.InvalidShape) return err;
            const ok: inference.cpu.Matrix = .{ .rows = 40, .columns = 64, .encoding = 0, .bytes = region };
            if (b.matmul(w, ok, small, 64, small, 64, 33)) |_| return error.ExpectedInvalidShape else |err| if (err != error.InvalidShape) return err;
        }
        // The specialized encodings must actually take the specialized path
        // when aligned, and fall back when the weight range or input is not.
        for ([_]u32{ 2, 11, 12, 13, 14, 21, 23 }) |encoding| {
            if (Backend.specializedMatvec(encoding, 0, 2880, 0) == null) return error.SpecializedPathNotSelected;
            if (Backend.specializedMatvec(encoding, 1, 2880, 0) != null) return error.MisalignedWeightsAccepted;
            if (Backend.specializedMatmul(encoding, 0, 2880, 64) == null) return error.SpecializedPathNotSelected;
            if (Backend.specializedMatmul(encoding, 1, 2880, 64) != null) return error.MisalignedWeightsAccepted;
            if (Backend.matmulGeometry(Backend.specializedMatmul(encoding, 0, 2880, 8).?).tokens != 8) return error.SmallTileNotSelected;
            if (Backend.matmulGeometry(Backend.specializedMatmul(encoding, 0, 2880, 16).?).tokens != 8) return error.SmallTileNotSelected;
            if (Backend.matmulGeometry(Backend.specializedMatmul(encoding, 0, 2880, 17).?).tokens != 32) return error.SmallTileNotSelected;
            if (Backend.matmulGeometry(Backend.specializedMatmul(encoding, 0, 2880, 32).?).tokens != 32) return error.SmallTileNotSelected;
            if (Backend.matmulGeometry(Backend.specializedMatmul(encoding, 0, 2880, 33).?).tokens != 64) return error.LargeTileNotSelected;
            if (Backend.specializedMatvec(encoding, 0, 2880, 4) != null) return error.MisalignedInputAccepted;
        }
        for ([_]u32{ 11, 21 }) |encoding| {
            if (Backend.specializedMatvec(encoding, 2, 110, 0) == null) return error.AlignmentRuleMismatch;
            if (Backend.specializedMatvec(encoding, 0, 111, 0) != null) return error.MisalignedWeightsAccepted;
        }
        if (Backend.specializedMatvec(12, 8, 2880, 0) != null or Backend.specializedMatvec(23, 8, 2880, 0) == null or Backend.specializedMatvec(14, 2, 2880, 0) == null) return error.AlignmentRuleMismatch;
        // Ternary: any whole-block row at two-byte (PQ2_0) or four-byte (PTQ1_0) alignment.
        if (Backend.specializedMatvec(142, 0, 42 * 34, 0) == null or Backend.specializedMatvec(142, 2, 42 * 34, 0) == null or Backend.specializedMatvec(142, 1, 42 * 34, 0) != null) return error.AlignmentRuleMismatch;
        if (Backend.specializedMatvec(143, 0, 42 * 28, 0) == null or Backend.specializedMatvec(143, 4, 42 * 28, 0) == null or Backend.specializedMatvec(143, 2, 42 * 28, 0) != null) return error.AlignmentRuleMismatch;
        if (Backend.specializedMatmul(142, 0, 42 * 34, 64) == null or Backend.specializedMatmul(143, 0, 42 * 28, 20) == null or Backend.matmulGeometry(Backend.specializedMatmul(143, 0, 42 * 28, 20).?).tokens != 32) return error.AlignmentRuleMismatch;
        if (Backend.matmulGeometry(Backend.specializedMatmul(142, 0, 42 * 34, 8).?).tokens != 8 or Backend.matmulGeometry(Backend.specializedMatmul(143, 0, 42 * 28, 8).?).tokens != 8) return error.AlignmentRuleMismatch;
        // Q4_0: both kernels take any whole-block row (a 42-block row here), at 2-byte alignment.
        if (Backend.specializedMatvec(2, 0, 756, 0) == null or Backend.specializedMatmul(2, 0, 756, 64) == null or Backend.specializedMatvec(2, 2, 2880, 0) == null) return error.AlignmentRuleMismatch;
        for ([_]u32{ 0, 1, 8, 20, 30 }) |encoding| if (Backend.specializedMatvec(encoding, 0, 4096, 0) != null or Backend.specializedMatmul(encoding, 0, 4096, 64) != null) return error.GenericEncodingSpecialized;
        // A misaligned weight slice still computes correctly through the fallback:
        // the same Q4_K matrix copied 8 bytes into a larger buffer.
        {
            const fixtures = try std.json.parseFromSlice(QuantFixture, alloc, @embedFile("src/quant/fixtures/k-affine.json"), .{ .ignore_unknown_fields = true });
            defer fixtures.deinit();
            var sample_bytes: ?[]const u8 = null;
            for (fixtures.value.rows) |sample| if (sample.encoding == 12) {
                sample_bytes = sample.bytes;
                break;
            };
            const region = try tiledMatrix(alloc, sample_bytes orelse return error.FixtureMissing, 12, rows, 1280);
            defer alloc.free(region);
            const shifted = try b.create(region.len + 16);
            @memcpy(shifted.host[8..][0..region.len], region);
            const matrix: inference.cpu.Matrix = .{ .rows = rows, .columns = 1280, .encoding = 12, .bytes = region };
            const input = try alloc.alloc(f32, 1280);
            defer alloc.free(input);
            for (input) |*x| x.* = random.float(f32) * 2 - 1;
            const decoded = try alloc.alloc(f32, 1280);
            defer alloc.free(decoded);
            try inference.cpu.matvec(matrix, input, expected, decoded);
            try matvecOnce(b, matrix, shifted.slice(8, region.len), input, actual);
            const stride = region.len / rows;
            for (0..rows) |r| {
                try inference.quant.row(12, region[r * stride ..][0..stride], decoded);
                var mass: f64 = 0;
                for (decoded, input) |w, x| mass += @abs(@as(f64, w) * x);
                try expectClose("misaligned fallback", actual[r], expected[r], @floatCast(mass * 4e-6 + 1e-6));
            }
        }
    }

    // 3. Regression: Q4_K half subnormal scales must decode like the CPU.
    {
        var block: [144]u8 = @splat(0);
        block[0] = 1; // smallest positive half
        @memset(block[4..16], 1); // scale fields
        @memset(block[16..144], 0xff);
        var decoded: [256]f32 = undefined;
        try inference.quant.row(12, &block, &decoded);
        var input: [256]f32 = @splat(1);
        var actual: [1]f32 = undefined;
        try matvecOnce(b, .{ .columns = 256, .rows = 1, .encoding = 12, .bytes = &block }, try uploadBytes(b, &block), &input, &actual);
        var expected: f32 = 0;
        for (decoded) |value| expected += value;
        try expectClose("subnormal Q4_K", actual[0], expected, 1e-9);
    }

    // 4. Dense F32/F16 use the same dispatch geometry, including multiple rows.
    for ([_]u32{ 0, 1 }) |encoding| {
        const width: usize = if (encoding == 0) 4 else 2;
        const region = try alloc.alloc(u8, 3 * 32 * width);
        defer alloc.free(region);
        for (0..96) |i| {
            const v: f32 = @as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) / 8;
            if (encoding == 0) std.mem.writeInt(u32, region[i * 4 ..][0..4], @bitCast(v), .little) else std.mem.writeInt(u16, region[i * 2 ..][0..2], @bitCast(@as(f16, @floatCast(v))), .little);
        }
        var x: [32]f32 = undefined;
        for (&x, 0..) |*value, i| value.* = @as(f32, @floatFromInt(i)) / 32;
        var expected_dense: [3]f32 = undefined;
        var actual_dense: [3]f32 = undefined;
        var row: [32]f32 = undefined;
        const matrix: inference.cpu.Matrix = .{ .rows = 3, .columns = 32, .encoding = encoding, .bytes = region };
        try inference.cpu.matvec(matrix, &x, &expected_dense, &row);
        try matvecOnce(b, matrix, try uploadBytes(b, region), &x, &actual_dense);
        for (expected_dense, actual_dense) |e, a| try expectClose("dense", a, e, 1e-6);
    }

    // 5. Pinned recurrent fixtures: carry GPU state through all DeltaNet steps
    // (one head, explicit scale) and all convolution steps (history in place).
    {
        const fixture = try std.json.parseFromSlice(RecurrentFixture, alloc, @embedFile("src/backends/cpu/fixtures/recurrent.json"), .{});
        defer fixture.deinit();
        const first = fixture.value.delta[0];
        const keys = first.key.len;
        const values = first.value.len;
        const state = try upload(b, first.state);
        const qkv = try b.create((2 * keys + values) * 4);
        const decay = try b.create(4);
        const beta = try b.create(4);
        const out = try b.create(values * 4);
        for (fixture.value.delta) |case| {
            @memcpy(qkv.floats()[0..keys], case.query);
            @memcpy(qkv.floats()[keys..][0..keys], case.key);
            @memcpy(qkv.floats()[2 * keys ..][0..values], case.value);
            decay.floats()[0] = case.log_decay;
            beta.floats()[0] = case.beta;
            try b.begin();
            try b.delta(state, qkv, decay, beta, out, .{ .qheads = 1, .vheads = 1, .keys = keys, .values = values, .scale = case.scale });
            try b.commit();
            for (out.floats()[0..values], case.output) |got, want| try expectClose("delta output", got, want, 1e-5);
            for (state.floats(), case.next_state) |got, want| try expectClose("delta state", got, want, 1e-5);
        }
        const conv_first = fixture.value.convolution[0];
        const history = try upload(b, conv_first.history);
        const weights = try b.create(conv_first.weights.len * 4);
        const input = try b.create(conv_first.input.len * 4);
        const conv_out = try b.create(conv_first.input.len * 4);
        for (fixture.value.convolution) |case| {
            @memcpy(weights.floats(), case.weights);
            @memcpy(input.floats(), case.input);
            try b.begin();
            try b.convolution(history, input, weights, conv_out, case.input.len, case.kernel);
            try b.commit();
            for (conv_out.floats(), case.output) |got, want| try expectClose("convolution output", got, want, 1e-6);
            for (history.floats(), case.next_history) |got, want| try expectClose("convolution history", got, want, 0);
        }
    }

    // 6. Two heads sharing Q/K, rectangular state, three recurrent updates
    // against the CPU reference with our own state carried on both sides.
    {
        var expected_state: [12]f32 = undefined;
        for (&expected_state, 0..) |*cpu, i| cpu.* = @as(f32, @floatFromInt(i)) / 20;
        const state = try upload(b, &expected_state);
        const qkv = try upload(b, &.{ 0.2, -0.3, 0.1, 0.4, 0.2, -0.1, 0.5, -0.2, 0.1, 0.8 });
        const decay = try upload(b, &.{ -0.2, -0.5 });
        const beta = try upload(b, &.{ 0.3, 0.7 });
        const gpu_out = try b.create(4 * 4);
        var cpu_out: [4]f32 = undefined;
        var scratch: [8]f64 = undefined;
        for (0..3) |_| {
            try b.begin();
            try b.delta(state, qkv, decay, beta, gpu_out, .{ .qheads = 1, .vheads = 2, .keys = 3, .values = 2, .scale = 1.0 / @sqrt(@as(f32, 3)) });
            try b.commit();
            const q = qkv.floats();
            for (0..2) |head| {
                const slice = expected_state[head * 6 ..][0..6];
                try inference.cpu.recurrent.delta(.{ .query = q[0..3], .key = q[3..6], .value = q[6 + head * 2 ..][0..2], .log_decay = decay.floats()[head], .beta = beta.floats()[head], .scale = 1.0 / @sqrt(@as(f32, 3)) }, slice, slice, cpu_out[head * 2 ..][0..2], &scratch);
            }
            for (state.floats(), expected_state) |g, c| try expectClose("multihead state", g, c, 1e-6);
            for (gpu_out.floats(), cpu_out) |g, c| try expectClose("multihead output", g, c, 1e-6);
        }
    }

    // 7. Pinned attention fixtures through the three-pass GPU attention.
    {
        const fixture = try std.json.parseFromSlice(AttentionFixture, alloc, @embedFile("src/backends/cpu/fixtures/attention.json"), .{});
        defer fixture.deinit();
        for (fixture.value.cases) |case| {
            const keys = try upload(b, case.keys);
            const values = try upload(b, case.values);
            const queries = try upload(b, case.queries);
            const scores = try b.create(case.query_heads * case.visible_tokens * 4);
            const out = try b.create(case.output.len * 4);
            try b.begin();
            try b.attention(keys, values, queries, scores, out, .{ .query_heads = case.query_heads, .kv_heads = case.kv_heads, .key_width = case.key_width, .value_width = case.value_width, .visible = case.visible_tokens, .scale = case.scale });
            try b.commit();
            for (out.floats(), case.output) |got, want| try expectClose("attention", got, want, 1e-5);
        }
    }

    // 8. Model-shaped attention at a long visible prefix against the CPU
    // reference: 24 query heads, 4 KV heads, width 256, 1,021 visible tokens.
    {
        var prng = std.Random.DefaultPrng.init(0xa77e);
        const random = prng.random();
        const tokens: usize = 1024;
        const keys = try b.create(tokens * 4 * 256 * 4);
        const values = try b.create(tokens * 4 * 256 * 4);
        for (keys.floats()) |*k| k.* = random.floatNorm(f32) * 0.5;
        for (values.floats()) |*v| v.* = random.floatNorm(f32) * 0.5;
        const queries = try b.create(24 * 256 * 4);
        for (queries.floats()) |*q| q.* = random.floatNorm(f32) * 0.5;
        const input: inference.cpu.attention.Input = .{ .query_heads = 24, .kv_heads = 4, .key_width = 256, .value_width = 256, .tokens = tokens, .visible_tokens = tokens - 3, .scale = 1.0 / 16.0, .queries = queries.floats(), .keys = keys.floats(), .values = values.floats() };
        var expected: [24 * 256]f32 = undefined;
        const scratch = try alloc.alloc(f64, tokens);
        defer alloc.free(scratch);
        try inference.cpu.attention.apply(input, &expected, scratch);
        const scores = try b.create(24 * tokens * 4);
        const out = try b.create(24 * 256 * 4);
        try b.begin();
        try b.attention(keys, values, queries, scores, out, .{ .query_heads = 24, .kv_heads = 4, .key_width = 256, .value_width = 256, .visible = tokens - 3, .scale = 1.0 / 16.0 });
        try b.commit();
        for (expected, out.floats()) |e, a| try expectClose("long attention", a, e, 2e-5);
    }

    // 8b. Causal tiled attention for a prefill chunk against the CPU
    // reference per query row: 256 queries at positions 1,792..2,047 over a
    // 2,048-row cache (every row compared, F64 reference, its own visible
    // prefix), 8 queries over a 16,384-row cache, then tails and poison: a chunk whose count and total are not
    // multiples of 8 or 32, with 1e30 in every key/value row after query row
    // 20's horizon — rows 0..20 must match the clean reference and the later
    // rows (which legitimately see the poison) must stay finite — plus the
    // single-token and three-token prompts from an empty cache.
    {
        var prng = std.Random.DefaultPrng.init(0xe21);
        const random = prng.random();
        const Case = struct { position: usize, count: usize, poison_after: ?usize };
        const cases = [_]Case{ .{ .position = 1792, .count = 256, .poison_after = null }, .{ .position = 5, .count = 37, .poison_after = 20 }, .{ .position = 0, .count = 1, .poison_after = null }, .{ .position = 0, .count = 3, .poison_after = null }, .{ .position = 16376, .count = 8, .poison_after = null } };
        var worst: f32 = 0;
        for (cases) |case| {
            const total = case.position + case.count;
            const rows = Backend.attentionChunkRows(case.count);
            const keys = try b.create(total * 4 * 256 * 4);
            const values = try b.create(total * 4 * 256 * 4);
            const queries = try b.create(rows * 24 * 256 * 4);
            const out = try b.create(rows * 6144 * 4);
            for (keys.floats()) |*k| k.* = random.floatNorm(f32) * 0.5;
            for (values.floats()) |*v| v.* = random.floatNorm(f32) * 0.5;
            for (queries.floats()) |*q| q.* = random.floatNorm(f32) * 0.5;
            const checked_rows = if (case.poison_after) |t| t + 1 else case.count;
            if (case.poison_after) |t| {
                @memset(keys.floats()[(case.position + t + 1) * 1024 ..], 1e30);
                @memset(values.floats()[(case.position + t + 1) * 1024 ..], 1e30);
            }
            try b.begin();
            try b.attentionChunk(keys, values, queries, out, .{ .query_heads = 24, .kv_heads = 4, .key_width = 256, .value_width = 256, .position = case.position, .count = case.count, .q_stride = 24 * 256, .out_stride = 6144, .scale = 1.0 / 16.0 });
            try b.commit();
            const scratch = try alloc.alloc(f64, total);
            defer alloc.free(scratch);
            var expected: [24 * 256]f32 = undefined;
            for (0..checked_rows) |t| {
                const input: inference.cpu.attention.Input = .{ .query_heads = 24, .kv_heads = 4, .key_width = 256, .value_width = 256, .tokens = total, .visible_tokens = case.position + t + 1, .scale = 1.0 / 16.0, .queries = queries.floats()[t * 24 * 256 ..][0 .. 24 * 256], .keys = keys.floats(), .values = values.floats() };
                try inference.cpu.attention.apply(input, &expected, scratch);
                for (expected, out.floats()[t * 6144 ..][0 .. 24 * 256]) |e, a| {
                    worst = @max(worst, @abs(a - e));
                    try expectClose("chunk attention", a, e, 1e-5);
                }
            }
            for (out.floats()[checked_rows * 6144 .. case.count * 6144]) |a| if (!std.math.isFinite(a)) return error.NonFiniteOutput;
        }
        std.debug.print("Chunk attention vs CPU F64 per row, up to 16,384 visible: max abs {e:.3} (bound 1e-5)\n", .{worst});
        // Contract: padded rows, widths, and cache coverage are validated before dispatch.
        const small = try b.create(8 * 4);
        try b.begin();
        if (b.attentionChunk(small, small, small, small, .{ .query_heads = 24, .kv_heads = 4, .key_width = 256, .value_width = 256, .position = 0, .count = 1, .q_stride = 24 * 256, .out_stride = 6144, .scale = 1.0 / 16.0 })) |_| return error.ExpectedInvalidShape else |err| if (err != error.InvalidShape) return err;
        if (b.attentionChunk(small, small, small, small, .{ .query_heads = 24, .kv_heads = 4, .key_width = 256, .value_width = 264, .position = 0, .count = 1, .q_stride = 24 * 256, .out_stride = 6144, .scale = 1.0 / 16.0 })) |_| return error.ExpectedInvalidShape else |err| if (err != error.InvalidShape) return err;
        try b.commit();
    }

    // 8d. F16 KV cache. `nu_pack_half` must round exactly as the
    // CPU's `@floatCast` (nearest even), so a host conversion of the same
    // floats is bit-identical. The half decode attention (1,021 visible,
    // F32 queries) is checked two ways: against the CPU reference over the
    // *rounded* values (the kernel's own error, same bound as the F32
    // kernel) and against the reference over the original floats (what the
    // rounding costs, printed for the record). The half chunk attention
    // reads half keys, values, and queries and rounds its probability tile
    // to half (matrix operands share one type), so its bound against the
    // rounded reference allows a 2^-11 relative perturbation of every
    // softmax weight over values of magnitude up to about 2: 1e-3
    // absolute, reached only when few keys are visible (the errors average
    // out over long prefixes; the measured worst is printed). Its poison
    // value is 6e4 (finite in half).
    {
        var prng = std.Random.DefaultPrng.init(0xc13);
        const random = prng.random();
        const tokens: usize = 1024;
        const keys = try b.create(tokens * 1024 * 4);
        const values = try b.create(tokens * 1024 * 4);
        const queries = try b.create(24 * 256 * 4);
        for (keys.floats()) |*k| k.* = random.floatNorm(f32) * 0.5;
        for (values.floats()) |*v| v.* = random.floatNorm(f32) * 0.5;
        for (queries.floats()) |*q| q.* = random.floatNorm(f32) * 0.5;
        const keys_h = try b.create(tokens * 1024 * 2);
        const values_h = try b.create(tokens * 1024 * 2);
        try b.begin();
        try b.packHalf(&.{ .{ .dst = keys_h, .src = keys, .count = tokens * 1024 }, .{ .dst = values_h, .src = values, .count = tokens * 1024 } });
        try b.commit();
        const rounded_keys = try alloc.alloc(f32, tokens * 1024);
        defer alloc.free(rounded_keys);
        const rounded_values = try alloc.alloc(f32, tokens * 1024);
        defer alloc.free(rounded_values);
        for ([_]struct { h: Buffer, f: Buffer, r: []f32 }{ .{ .h = keys_h, .f = keys, .r = rounded_keys }, .{ .h = values_h, .f = values, .r = rounded_values } }) |pair| {
            const halves = @as([*]const u16, @ptrCast(@alignCast(pair.h.host)))[0 .. tokens * 1024];
            for (halves, pair.f.floats(), pair.r) |got, original, *r| {
                const want: f16 = @floatCast(original);
                if (got != @as(u16, @bitCast(want))) return error.MetalPackMismatch;
                r.* = want;
            }
        }
        const visible = tokens - 3;
        var expected_rounded: [24 * 256]f32 = undefined;
        var expected_exact: [24 * 256]f32 = undefined;
        const scratch = try alloc.alloc(f64, tokens);
        defer alloc.free(scratch);
        try inference.cpu.attention.apply(.{ .query_heads = 24, .kv_heads = 4, .key_width = 256, .value_width = 256, .tokens = tokens, .visible_tokens = visible, .scale = 1.0 / 16.0, .queries = queries.floats(), .keys = rounded_keys, .values = rounded_values }, &expected_rounded, scratch);
        try inference.cpu.attention.apply(.{ .query_heads = 24, .kv_heads = 4, .key_width = 256, .value_width = 256, .tokens = tokens, .visible_tokens = visible, .scale = 1.0 / 16.0, .queries = queries.floats(), .keys = keys.floats(), .values = values.floats() }, &expected_exact, scratch);
        const scores = try b.create(24 * tokens * 4);
        const out = try b.create(24 * 256 * 4);
        try b.begin();
        try b.attention(keys_h, values_h, queries, scores, out, .{ .query_heads = 24, .kv_heads = 4, .key_width = 256, .value_width = 256, .visible = visible, .scale = 1.0 / 16.0, .precision = .f16 });
        try b.commit();
        var worst_exact: f32 = 0;
        var worst_rounded: f32 = 0;
        var sum_sq: f64 = 0;
        var ref_sq: f64 = 0;
        for (expected_rounded, expected_exact, out.floats()) |r, e, a| {
            try expectClose("half attention", a, r, 2e-5);
            worst_rounded = @max(worst_rounded, @abs(a - r));
            worst_exact = @max(worst_exact, @abs(a - e));
            sum_sq += @as(f64, a - e) * (a - e);
            ref_sq += @as(f64, e) * e;
        }
        std.debug.print("F16 KV decode attention, 1,021 visible: vs CPU over the rounded cache max abs {e:.3} (bound 2e-5); vs CPU over the F32 cache max abs {e:.3}, relative RMS {e:.3}\n", .{ worst_rounded, worst_exact, @sqrt(sum_sq / ref_sq) });

        // Half chunk attention on the cases, against the CPU reference over
        // the rounded keys, values, and queries.
        const Case = struct { position: usize, count: usize, poison_after: ?usize };
        const cases = [_]Case{ .{ .position = 1792, .count = 256, .poison_after = null }, .{ .position = 5, .count = 37, .poison_after = 20 }, .{ .position = 0, .count = 1, .poison_after = null }, .{ .position = 0, .count = 3, .poison_after = null }, .{ .position = 16376, .count = 8, .poison_after = null } };
        var worst_chunk: f32 = 0;
        for (cases) |case| {
            const total = case.position + case.count;
            const rows = Backend.attentionChunkRows(case.count);
            const k32 = try b.create(total * 1024 * 4);
            const v32 = try b.create(total * 1024 * 4);
            const q32 = try b.create(rows * 24 * 256 * 4);
            for (k32.floats()) |*k| k.* = random.floatNorm(f32) * 0.5;
            for (v32.floats()) |*v| v.* = random.floatNorm(f32) * 0.5;
            for (q32.floats()) |*q| q.* = random.floatNorm(f32) * 0.5;
            const checked_rows = if (case.poison_after) |t| t + 1 else case.count;
            if (case.poison_after) |t| {
                @memset(k32.floats()[(case.position + t + 1) * 1024 ..], 6e4);
                @memset(v32.floats()[(case.position + t + 1) * 1024 ..], 6e4);
            }
            const k16 = try b.create(total * 1024 * 2);
            const v16 = try b.create(total * 1024 * 2);
            const q16 = try b.create(rows * 24 * 256 * 2);
            const out_c = try b.create(rows * 6144 * 4);
            try b.begin();
            try b.packHalf(&.{ .{ .dst = k16, .src = k32, .count = total * 1024 }, .{ .dst = v16, .src = v32, .count = total * 1024 } });
            try b.packHalf(&.{.{ .dst = q16, .src = q32, .count = rows * 24 * 256 }});
            try b.attentionChunk(k16, v16, q16, out_c, .{ .query_heads = 24, .kv_heads = 4, .key_width = 256, .value_width = 256, .position = case.position, .count = case.count, .q_stride = 24 * 256, .out_stride = 6144, .scale = 1.0 / 16.0, .precision = .f16 });
            try b.commit();
            // The rounded operands, read back from the packed buffers, are the reference's inputs.
            for ([_]struct { h: Buffer, f: Buffer }{ .{ .h = k16, .f = k32 }, .{ .h = v16, .f = v32 }, .{ .h = q16, .f = q32 } }) |pair| {
                const halves = @as([*]const u16, @ptrCast(@alignCast(pair.h.host)))[0 .. pair.h.len / 2];
                for (halves, pair.f.floats()) |h, *f| f.* = @as(f16, @bitCast(h));
            }
            const chunk_scratch = try alloc.alloc(f64, total);
            defer alloc.free(chunk_scratch);
            var expected: [24 * 256]f32 = undefined;
            for (0..checked_rows) |t| {
                try inference.cpu.attention.apply(.{ .query_heads = 24, .kv_heads = 4, .key_width = 256, .value_width = 256, .tokens = total, .visible_tokens = case.position + t + 1, .scale = 1.0 / 16.0, .queries = q32.floats()[t * 24 * 256 ..][0 .. 24 * 256], .keys = k32.floats(), .values = v32.floats() }, &expected, chunk_scratch);
                for (expected, out_c.floats()[t * 6144 ..][0 .. 24 * 256]) |e, a| {
                    worst_chunk = @max(worst_chunk, @abs(a - e));
                    try expectClose("half chunk attention", a, e, 1e-3);
                }
            }
            for (out_c.floats()[checked_rows * 6144 .. case.count * 6144]) |a| if (!std.math.isFinite(a)) return error.NonFiniteOutput;
        }
        std.debug.print("F16 KV chunk attention vs CPU over the rounded operands per row: max abs {e:.3} (bound 1e-3)\n", .{worst_chunk});
        // Contract: pair count, zero count, overlap, and half alignment are refused before dispatch.
        const small = try b.create(64);
        try b.begin();
        try std.testing.expectError(error.InvalidShape, b.packHalf(&.{}));
        try std.testing.expectError(error.InvalidShape, b.packHalf(&.{.{ .dst = small, .src = small, .count = 0 }}));
        try std.testing.expectError(error.InvalidShape, b.packHalf(&.{.{ .dst = small, .src = small, .count = 8 }}));
        try std.testing.expectError(error.InvalidShape, b.packHalf(&.{ .{ .dst = small, .src = keys, .count = 8 }, .{ .dst = small, .src = keys, .count = 8 }, .{ .dst = small, .src = keys, .count = 8 } }));
        try std.testing.expectError(error.InvalidShape, b.attention(keys_h.slice(1, keys_h.len - 1), values_h, queries, scores, out, .{ .query_heads = 24, .kv_heads = 4, .key_width = 256, .value_width = 256, .visible = 1, .scale = 1.0 / 16.0, .precision = .f16 }));
        try std.testing.expectError(error.InvalidShape, b.attention(keys_h, values_h, queries, scores, out, .{ .query_heads = 24, .kv_heads = 4, .key_width = 256, .value_width = 256, .visible = tokens + 1, .scale = 1.0 / 16.0, .precision = .f16 }));
        try b.commit();
    }

    // 8e. Flash-decoding attention: the pinned fixtures (tiny widths,
    // one split, mostly empty SIMD groups) at the three-pass tolerance, then
    // the model shape at 257 visible (two splits, the second short), 1,021,
    // 16,385 (64 uneven splits), and 32,000 rows against the F64 CPU
    // reference, F32 cache and F16 cache (over the rounded rows: this
    // kernel keeps queries and probabilities F32, so the F16 bound is the
    // F32 one). The bound widens with the row count: F32 online sums over
    // 32,000 terms.
    {
        const fixture = try std.json.parseFromSlice(AttentionFixture, alloc, @embedFile("src/backends/cpu/fixtures/attention.json"), .{});
        defer fixture.deinit();
        for (fixture.value.cases) |case| {
            const keys = try upload(b, case.keys);
            const values = try upload(b, case.values);
            const queries = try upload(b, case.queries);
            const partials = try b.create(Backend.attentionDecodePartials(case.query_heads, case.value_width) * 4);
            const out = try b.create(case.output.len * 4);
            try b.begin();
            try b.attentionDecode(keys, values, queries, partials, out, .{ .query_heads = case.query_heads, .kv_heads = case.kv_heads, .key_width = case.key_width, .value_width = case.value_width, .visible = case.visible_tokens, .scale = case.scale });
            try b.commit();
            for (out.floats(), case.output) |got, want| try expectClose("decode attention fixture", got, want, 1e-5);
        }
        var prng = std.Random.DefaultPrng.init(0xc14);
        const random = prng.random();
        const partials = try b.create(Backend.attentionDecodePartials(24, 256) * 4);
        const out = try b.create(24 * 256 * 4);
        const queries = try b.create(24 * 256 * 4);
        for (queries.floats()) |*q| q.* = random.floatNorm(f32) * 0.5;
        const Case = struct { visible: usize, bound: f32 };
        for ([_]Case{ .{ .visible = 257, .bound = 2e-5 }, .{ .visible = 1021, .bound = 2e-5 }, .{ .visible = 16385, .bound = 1e-4 }, .{ .visible = 32000, .bound = 1e-4 } }) |case| {
            const tokens = case.visible;
            const keys = try b.create(tokens * 1024 * 4);
            const values = try b.create(tokens * 1024 * 4);
            for (keys.floats()) |*k| k.* = random.floatNorm(f32) * 0.5;
            for (values.floats()) |*v| v.* = random.floatNorm(f32) * 0.5;
            const scratch = try alloc.alloc(f64, tokens);
            defer alloc.free(scratch);
            var expected: [24 * 256]f32 = undefined;
            try inference.cpu.attention.apply(.{ .query_heads = 24, .kv_heads = 4, .key_width = 256, .value_width = 256, .tokens = tokens, .visible_tokens = tokens, .scale = 1.0 / 16.0, .queries = queries.floats(), .keys = keys.floats(), .values = values.floats() }, &expected, scratch);
            try b.begin();
            try b.attentionDecode(keys, values, queries, partials, out, .{ .query_heads = 24, .kv_heads = 4, .key_width = 256, .value_width = 256, .visible = tokens, .scale = 1.0 / 16.0 });
            try b.commit();
            var worst: f32 = 0;
            for (expected, out.floats()) |e, a| {
                worst = @max(worst, @abs(a - e));
                try expectClose("decode attention f32", a, e, case.bound);
            }
            // The same rows rounded to half, the reference over the rounded values.
            const keys_h = try b.create(tokens * 1024 * 2);
            const values_h = try b.create(tokens * 1024 * 2);
            try b.begin();
            try b.packHalf(&.{ .{ .dst = keys_h, .src = keys, .count = tokens * 1024 }, .{ .dst = values_h, .src = values, .count = tokens * 1024 } });
            try b.commit();
            for ([_]struct { h: Buffer, f: Buffer }{ .{ .h = keys_h, .f = keys }, .{ .h = values_h, .f = values } }) |pair| {
                const halves = @as([*]const u16, @ptrCast(@alignCast(pair.h.host)))[0 .. tokens * 1024];
                for (halves, pair.f.floats()) |h, *f| f.* = @as(f16, @bitCast(h));
            }
            try inference.cpu.attention.apply(.{ .query_heads = 24, .kv_heads = 4, .key_width = 256, .value_width = 256, .tokens = tokens, .visible_tokens = tokens, .scale = 1.0 / 16.0, .queries = queries.floats(), .keys = keys.floats(), .values = values.floats() }, &expected, scratch);
            try b.begin();
            try b.attentionDecode(keys_h, values_h, queries, partials, out, .{ .query_heads = 24, .kv_heads = 4, .key_width = 256, .value_width = 256, .visible = tokens, .scale = 1.0 / 16.0, .precision = .f16 });
            try b.commit();
            var worst_h: f32 = 0;
            for (expected, out.floats()) |e, a| {
                worst_h = @max(worst_h, @abs(a - e));
                try expectClose("decode attention f16", a, e, case.bound);
            }
            std.debug.print("Flash-decoding attention, {d} visible ({d} splits) vs CPU F64: F32 cache max abs {e:.3}, F16 cache over the rounded rows {e:.3} (bound {e:.0})\n", .{ tokens, Backend.attentionDecodeSplits(tokens), worst, worst_h, case.bound });
        }
        // Contract: group size, widths, partial and cache coverage are refused before dispatch.
        const small = try b.create(64);
        try b.begin();
        try std.testing.expectError(error.InvalidShape, b.attentionDecode(small, small, small, partials, out, .{ .query_heads = 36, .kv_heads = 5, .key_width = 256, .value_width = 256, .visible = 1, .scale = 1.0 / 16.0 }));
        try std.testing.expectError(error.InvalidShape, b.attentionDecode(small, small, small, partials, out, .{ .query_heads = 24, .kv_heads = 4, .key_width = 256, .value_width = 264, .visible = 1, .scale = 1.0 / 16.0 }));
        try std.testing.expectError(error.InvalidShape, b.attentionDecode(small, small, queries, small, out, .{ .query_heads = 24, .kv_heads = 4, .key_width = 256, .value_width = 256, .visible = 1, .scale = 1.0 / 16.0 }));
        try std.testing.expectError(error.InvalidShape, b.attentionDecode(small, small, queries, partials, out, .{ .query_heads = 24, .kv_heads = 4, .key_width = 256, .value_width = 256, .visible = 1, .scale = 1.0 / 16.0 }));
        try b.commit();
    }

    // 8f. The Gemma 4 attention geometry.
    try checkWindowedAndWideAttention(alloc, b);

    // 8c. Chunkwise DeltaNet: a 70-token layer chunk (sub-chunks of
    // 32, 32, and 6) on the model shape (16 Q/K heads broadcast to 48 value
    // heads by h % 16, 128×128 state per head, L2-normalized q/k) against the
    // CPU chunkwise reference per head (F64) and against 70 sequential CPU
    // steps per head (F32 state between steps). The qkv rows past `count`
    // are NaN: the kernel must never read them.
    {
        var prng = std.Random.DefaultPrng.init(0xde17a);
        const random = prng.random();
        const count: usize = 70;
        const rows = Backend.deltaChunkRows(count);
        const in_stride: usize = 10240;
        const qkv = try b.create(rows * in_stride * 4);
        const decays = try b.create(count * 48 * 4);
        const betas = try b.create(count * 48 * 4);
        const out = try b.create(count * 6144 * 4);
        const state = try b.create(48 * 128 * 128 * 4);
        for (qkv.floats()[0 .. count * in_stride]) |*x| x.* = random.floatNorm(f32);
        @memset(qkv.floats()[count * in_stride ..], std.math.nan(f32));
        for (0..count) |t| {
            const row = qkv.floats()[t * in_stride ..][0..in_stride];
            for (0..32) |h| { // q heads 0..16, k heads 16..32: L2-normalize each 128-wide head
                const v = row[h * 128 ..][0..128];
                var norm: f64 = 0;
                for (v) |x| norm += @as(f64, x) * x;
                const inv: f32 = @floatCast(1.0 / @sqrt(norm + 1e-6));
                for (v) |*x| x.* *= inv;
            }
        }
        for (decays.floats()) |*d| d.* = -random.float(f32) * 0.5;
        for (betas.floats()) |*x| x.* = random.float(f32);
        for (state.floats()) |*x| x.* = random.floatNorm(f32) * 0.1;
        const start = try alloc.dupe(f32, state.floats());
        defer alloc.free(start);
        const scale: f32 = 1.0 / @sqrt(@as(f32, 128));
        try b.begin();
        try b.deltaChunk(state, qkv, decays, betas, out, .{ .qheads = 16, .vheads = 48, .keys = 128, .values = 128, .count = count, .in_stride = in_stride, .gate_stride = 48, .out_stride = 6144, .scale = scale });
        try b.commit();
        // Per head: gather contiguous rows for the CPU references.
        const q = try alloc.alloc(f32, count * 128);
        defer alloc.free(q);
        const k = try alloc.alloc(f32, count * 128);
        defer alloc.free(k);
        const v = try alloc.alloc(f32, count * 128);
        defer alloc.free(v);
        const ld = try alloc.alloc(f32, count);
        defer alloc.free(ld);
        const bt = try alloc.alloc(f32, count);
        defer alloc.free(bt);
        const chunk_state = try alloc.alloc(f32, 128 * 128);
        defer alloc.free(chunk_state);
        const chunk_out = try alloc.alloc(f32, count * 128);
        defer alloc.free(chunk_out);
        const seq_state = try alloc.alloc(f32, 128 * 128);
        defer alloc.free(seq_state);
        const seq_out = try alloc.alloc(f32, 128);
        defer alloc.free(seq_out);
        const step_scratch = try alloc.alloc(f64, 128 * 128 + 128);
        defer alloc.free(step_scratch);
        var worst_out: f32 = 0;
        var worst_state: f32 = 0;
        var worst_seq: f32 = 0;
        for (0..48) |h| {
            const kh = h % 16;
            for (0..count) |t| {
                const row = qkv.floats()[t * in_stride ..][0..in_stride];
                @memcpy(q[t * 128 ..][0..128], row[kh * 128 ..][0..128]);
                @memcpy(k[t * 128 ..][0..128], row[2048 + kh * 128 ..][0..128]);
                @memcpy(v[t * 128 ..][0..128], row[4096 + h * 128 ..][0..128]);
                ld[t] = decays.floats()[t * 48 + h];
                bt[t] = betas.floats()[t * 48 + h];
            }
            const x: inference.cpu.recurrent.DeltaChunk = .{ .tokens = count, .key_width = 128, .value_width = 128, .queries = q, .keys = k, .values = v, .log_decays = ld, .betas = bt, .scale = scale };
            const scratch = try alloc.alloc(f64, try inference.cpu.recurrent.deltaChunkScratch(x));
            defer alloc.free(scratch);
            @memcpy(chunk_state, start[h * 128 * 128 ..][0 .. 128 * 128]);
            try inference.cpu.recurrent.deltaChunk(x, chunk_state, chunk_state, chunk_out, scratch);
            @memcpy(seq_state, start[h * 128 * 128 ..][0 .. 128 * 128]);
            for (0..count) |t| {
                try inference.cpu.recurrent.delta(.{ .query = q[t * 128 ..][0..128], .key = k[t * 128 ..][0..128], .value = v[t * 128 ..][0..128], .log_decay = ld[t], .beta = bt[t], .scale = scale }, seq_state, seq_state, seq_out, step_scratch);
                for (seq_out, out.floats()[t * 6144 + h * 128 ..][0..128]) |want, got| worst_seq = @max(worst_seq, @abs(got - want));
            }
            for (0..count) |t| for (chunk_out[t * 128 ..][0..128], out.floats()[t * 6144 + h * 128 ..][0..128]) |want, got| {
                worst_out = @max(worst_out, @abs(got - want));
                try expectClose("delta chunk output", got, want, 1e-5);
            };
            for (chunk_state, state.floats()[h * 128 * 128 ..][0 .. 128 * 128]) |want, got| {
                worst_state = @max(worst_state, @abs(got - want));
                try expectClose("delta chunk state", got, want, 1e-4);
            }
            for (seq_state, state.floats()[h * 128 * 128 ..][0 .. 128 * 128]) |want, got| worst_seq = @max(worst_seq, @abs(got - want));
        }
        if (worst_seq > 1e-4) return error.DeltaChunkSequentialMismatch;
        std.debug.print("Chunk DeltaNet vs CPU chunkwise F64: output max abs {e:.3} (bound 1e-5), state {e:.3} (bound 1e-4); vs 70 sequential CPU steps: {e:.3} (bound 1e-4)\n", .{ worst_out, worst_state, worst_seq });
        const small = try b.create(8 * 4);
        try b.begin();
        if (b.deltaChunk(small, small, small, small, small, .{ .qheads = 16, .vheads = 48, .keys = 128, .values = 100, .count = 1, .in_stride = in_stride, .gate_stride = 48, .out_stride = 6144, .scale = scale })) |_| return error.ExpectedInvalidShape else |err| if (err != error.InvalidShape) return err;
        if (b.deltaChunk(state, qkv, decays, betas, out, .{ .qheads = 16, .vheads = 48, .keys = 128, .values = 128, .count = 0, .in_stride = in_stride, .gate_stride = 48, .out_stride = 6144, .scale = scale })) |_| return error.ExpectedInvalidShape else |err| if (err != error.InvalidShape) return err;
        try b.commit();
    }

    // 9. Elementwise, normalization, and rotary kernels against cpu.* references
    // on model-shaped vectors. Tolerances: 1e-5 absolute for norms of unit-scale
    // data, 1e-6 for activations, 2e-6 for rotation with F64-tabulated angles.
    {
        var prng = std.Random.DefaultPrng.init(0x0e1e);
        const random = prng.random();
        const width: usize = 5120;
        const x = try b.create(width * 4);
        const w = try b.create(width * 4);
        const y = try b.create(width * 4);
        const z = try b.create(width * 4);
        for (x.floats()) |*v| v.* = random.floatNorm(f32) * 3;
        for (w.floats()) |*v| v.* = random.float(f32) + 0.5;
        for (z.floats()) |*v| v.* = random.floatNorm(f32);
        const expected = try alloc.alloc(f32, width);
        defer alloc.free(expected);
        // Weighted RMSNorm, out of place.
        try inference.cpu.rmsNorm(x.floats(), expected, 1e-6);
        for (expected, w.floats()) |*e, ww| e.* *= ww;
        try b.begin();
        try b.rmsNorm(x, w, y, .{ .rows = 1, .width = width, .in_stride = width, .out_stride = width });
        try b.commit();
        for (y.floats(), expected) |got, want| try expectClose("rmsnorm", got, want, 1e-5);
        // Strided multi-row RMSNorm with silu multiplier, in place: 48 rows of 128.
        const rows_in = try b.create(48 * 128 * 4);
        for (rows_in.floats()) |*v| v.* = random.floatNorm(f32);
        const rows_w = try b.create(128 * 4);
        for (rows_w.floats()) |*v| v.* = random.float(f32) + 0.5;
        const mult = try b.create(48 * 128 * 4);
        for (mult.floats()) |*v| v.* = random.floatNorm(f32);
        const rows_expected = try alloc.alloc(f32, 48 * 128);
        defer alloc.free(rows_expected);
        for (0..48) |r| {
            try inference.cpu.rmsNorm(rows_in.floats()[r * 128 ..][0..128], rows_expected[r * 128 ..][0..128], 1e-6);
            for (rows_expected[r * 128 ..][0..128], rows_w.floats(), mult.floats()[r * 128 ..][0..128]) |*e, ww, m| e.* *= ww * inference.cpu.silu(m);
        }
        try b.begin();
        try b.rmsNorm(rows_in, rows_w, rows_in, .{ .rows = 48, .width = 128, .in_stride = 128, .out_stride = 128, .silu_multiplier = .{ .buffer = mult, .stride = 128 } });
        try b.commit();
        for (rows_in.floats(), rows_expected) |got, want| try expectClose("rmsnorm rows", got, want, 1e-5);
        // Strided input rows (query heads packed with gates): 24 rows, stride 512 -> 256.
        const qg = try b.create(24 * 512 * 4);
        for (qg.floats()) |*v| v.* = random.floatNorm(f32);
        const qnorm = try b.create(256 * 4);
        for (qnorm.floats()) |*v| v.* = random.float(f32) + 0.5;
        const q = try b.create(24 * 256 * 4);
        const q_expected = try alloc.alloc(f32, 24 * 256);
        defer alloc.free(q_expected);
        for (0..24) |h| {
            try inference.cpu.rmsNorm(qg.floats()[h * 512 ..][0..256], q_expected[h * 256 ..][0..256], 1e-6);
            for (q_expected[h * 256 ..][0..256], qnorm.floats()) |*e, ww| e.* *= ww;
        }
        try b.begin();
        try b.rmsNorm(qg, qnorm, q, .{ .rows = 24, .width = 256, .in_stride = 512, .out_stride = 256 });
        try b.commit();
        for (q.floats(), q_expected) |got, want| try expectClose("rmsnorm strided", got, want, 1e-5);
        // RoPE on those heads at a late position, against cpu.rope.apply.
        const positions: usize = 32768;
        const table = try b.create(positions * 32 * 8);
        try Backend.ropeTable(table, positions, 64, 1e7, null);
        for (0..24) |h| try inference.cpu.rope.apply(q_expected[h * 256 ..][0..256], q_expected[h * 256 ..][0..256], .{ .dimensions = 64, .base = 1e7, .position = 32767 });
        try b.begin();
        try b.rope(q, table, 24, 256, 64, 32767, .split_half);
        try b.commit();
        for (q.floats(), q_expected) |got, want| try expectClose("rope", got, want, 2e-6 * @max(1, @abs(want)));
        // RoPE over a whole 512-wide head with per-pair factors (the
        // Gemma global layers' 64 unit factors then 192 of 1e30), 16 heads
        // at position 32767, against cpu.rope.apply with the same factors.
        {
            var factors: [256]f32 = @splat(1e30);
            @memset(factors[0..64], 1);
            const wide_table = try b.create(positions * 256 * 8);
            try Backend.ropeTable(wide_table, positions, 512, 1e6, &factors);
            const wide = try b.create(16 * 512 * 4);
            for (wide.floats()) |*v| v.* = random.floatNorm(f32);
            const wide_expected = try alloc.alloc(f32, 16 * 512);
            defer alloc.free(wide_expected);
            for (0..16) |h| try inference.cpu.rope.apply(wide.floats()[h * 512 ..][0..512], wide_expected[h * 512 ..][0..512], .{ .dimensions = 512, .base = 1e6, .position = 32767, .factors = &factors });
            try b.begin();
            try b.rope(wide, wide_table, 16, 512, 512, 32767, .split_half);
            try b.commit();
            for (wide.floats(), wide_expected, 0..) |got, want, i| {
                try expectClose("rope with factors", got, want, 2e-6 * @max(1, @abs(want)));
                // Pairs 64..255 are unrotated to F32 precision.
                const pair = i % 512 % 256;
                if (pair >= 64 and got != want) return error.RopeFactorsMismatch;
            }
            try std.testing.expectError(error.InvalidShape, Backend.ropeTable(wide_table, positions, 512, 1e6, factors[0..100]));
            factors[3] = 0;
            try std.testing.expectError(error.InvalidShape, Backend.ropeTable(wide_table, positions, 512, 1e6, &factors));
        }
        // Adjacent pairing (Muse Glimmer: the whole 128-wide head at base
        // 5e5), 32 heads at position 32767, against cpu.rope.apply in that
        // mode; the same table serves both pairings.
        {
            const adjacent_table = try b.create(positions * 64 * 8);
            try Backend.ropeTable(adjacent_table, positions, 128, 5e5, null);
            const adjacent = try b.create(32 * 128 * 4);
            for (adjacent.floats()) |*v| v.* = random.floatNorm(f32);
            const adjacent_expected = try alloc.alloc(f32, 32 * 128);
            defer alloc.free(adjacent_expected);
            for (0..32) |h| try inference.cpu.rope.apply(adjacent.floats()[h * 128 ..][0..128], adjacent_expected[h * 128 ..][0..128], .{ .dimensions = 128, .base = 5e5, .position = 32767, .pairing = .adjacent });
            try b.begin();
            try b.rope(adjacent, adjacent_table, 32, 128, 128, 32767, .adjacent);
            try b.commit();
            for (adjacent.floats(), adjacent_expected) |got, want| try expectClose("rope adjacent", got, want, 2e-6 * @max(1, @abs(want)));
        }
        // L2 norm over 32 rows of 128.
        const l2 = try b.create(32 * 128 * 4);
        for (l2.floats()) |*v| v.* = random.floatNorm(f32);
        const l2_expected = try alloc.alloc(f32, 32 * 128);
        defer alloc.free(l2_expected);
        for (0..32) |r| try inference.cpu.l2Norm(l2.floats()[r * 128 ..][0..128], l2_expected[r * 128 ..][0..128], 1e-6);
        try b.begin();
        try b.l2Norm(l2, 32, 128, 128, 1e-6);
        try b.commit();
        for (l2.floats(), l2_expected) |got, want| try expectClose("l2norm", got, want, 1e-6);
        // silu*mul, add, silu in place, delta gates, sigmoid gate.
        const g = try b.create(width * 4);
        const u = try b.create(width * 4);
        for (g.floats()) |*v| v.* = random.floatNorm(f32) * 4;
        for (u.floats()) |*v| v.* = random.floatNorm(f32);
        for (expected, g.floats(), u.floats()) |*e, gg, uu| e.* = inference.cpu.silu(gg) * uu;
        try b.begin();
        try b.siluMul(g, u, width);
        try b.commit();
        for (g.floats(), expected) |got, want| try expectClose("silu_mul", got, want, 1e-6 * @max(1, @abs(want)));
        for (expected, x.floats(), z.floats()) |*e, xx, zz| e.* = xx + zz;
        try b.begin();
        try b.add(x, z, width);
        try b.commit();
        for (x.floats(), expected) |got, want| try expectClose("add", got, want, 0);
        for (expected, z.floats()) |*e, zz| e.* = inference.cpu.silu(zz);
        try b.begin();
        try b.silu(z, width);
        try b.commit();
        for (z.floats(), expected) |got, want| try expectClose("silu", got, want, 1e-6 * @max(1, @abs(want)));
        // Epilogues: gelu·mul with values past tanh's F32 overflow (the
        // CPU saturates; the kernel must too), scale, add·scale, softcap.
        for (g.floats()) |*v| v.* = random.floatNorm(f32) * 4;
        g.floats()[0] = 60;
        g.floats()[1] = -60;
        g.floats()[2] = 200;
        g.floats()[3] = -3e3;
        for (expected, g.floats(), u.floats()) |*e, gg, uu| e.* = inference.cpu.gelu(gg) * uu;
        try b.begin();
        try b.geluMul(g, u, width);
        try b.commit();
        for (g.floats(), expected) |got, want| try expectClose("gelu_mul", got, want, 2e-6 * @max(1, @abs(want)));
        for (expected, x.floats()) |*e, xx| e.* = xx * 61.967735;
        try b.begin();
        try b.scale(x, width, 61.967735);
        try b.commit();
        for (x.floats(), expected) |got, want| try expectClose("scale", got, want, 0);
        for (expected, x.floats(), z.floats()) |*e, xx, zz| e.* = (xx + zz) * 0.053;
        try b.begin();
        try b.addScale(x, z, width, 0.053);
        try b.commit();
        for (x.floats(), expected) |got, want| try expectClose("add_scale", got, want, 0);
        for (z.floats()) |*v| v.* = random.floatNorm(f32) * 40;
        z.floats()[0] = 3e3;
        z.floats()[1] = -3e3;
        for (expected, z.floats()) |*e, zz| e.* = 30.0 * std.math.tanh(zz / 30.0);
        try b.begin();
        try b.softcap(z, width, 30.0);
        try b.commit();
        for (z.floats(), expected) |got, want| try expectClose("softcap", got, want, 2e-6 * @max(1, @abs(want)));
        if (b.softcap(z, width, 0)) |_| return error.ExpectedInvalidShape else |err| if (err != error.InvalidShape) return err;
        const alpha = try b.create(48 * 4);
        const beta = try b.create(48 * 4);
        const a = try b.create(48 * 4);
        const bias = try b.create(48 * 4);
        var alpha_expected: [48]f32 = undefined;
        var beta_expected: [48]f32 = undefined;
        for (0..48) |h| {
            alpha.floats()[h] = random.floatNorm(f32) * 3;
            beta.floats()[h] = random.floatNorm(f32) * 3;
            a.floats()[h] = -random.float(f32);
            bias.floats()[h] = random.floatNorm(f32);
            alpha_expected[h] = a.floats()[h] * inference.cpu.softplus(alpha.floats()[h] + bias.floats()[h]);
            beta_expected[h] = inference.cpu.sigmoid(beta.floats()[h]);
        }
        try b.begin();
        try b.deltaGates(alpha, beta, a, bias, 48);
        try b.commit();
        for (alpha.floats(), alpha_expected) |got, want| try expectClose("delta gate alpha", got, want, 1e-6 * @max(1, @abs(want)));
        for (beta.floats(), beta_expected) |got, want| try expectClose("delta gate beta", got, want, 1e-6);
        const gated = try b.create(24 * 256 * 4);
        for (gated.floats()) |*v| v.* = random.floatNorm(f32);
        const gated_expected = try alloc.alloc(f32, 24 * 256);
        defer alloc.free(gated_expected);
        for (0..24) |h| for (0..256) |i| {
            gated_expected[h * 256 + i] = gated.floats()[h * 256 + i] * inference.cpu.sigmoid(qg.floats()[h * 512 + 256 + i]);
        };
        try b.begin();
        try b.sigmoidGate(gated, qg, 24, 256, 512, 256);
        try b.commit();
        for (gated.floats(), gated_expected) |got, want| try expectClose("sigmoid gate", got, want, 1e-6 * @max(1, @abs(want)));
        // Greedy argmax over a vocabulary-sized vector, with a tie at two indices.
        const logits = try b.create(248320 * 4);
        for (logits.floats()) |*v| v.* = random.floatNorm(f32);
        logits.floats()[123456] = 50;
        logits.floats()[200000] = 50;
        const values = try b.create(Backend.argmax_partials * 4);
        const indices = try b.create(Backend.argmax_partials * 4);
        const result = try b.create(4);
        try b.begin();
        try b.argmax(logits, 248320, values, indices, result);
        try b.commit();
        const chosen = @as(*const u32, @ptrCast(@alignCast(result.host))).*;
        if (chosen != 123456) {
            std.debug.print("argmax chose {d}\n", .{chosen});
            return error.MetalArgmaxMismatch;
        }
        // Partial top-k over the same vocabulary-sized vector, with the
        // two-way tie at the top and a run of equal values further down: the
        // 256 (value, index) pairs must equal the CPU sort's first 256 by
        // (value desc, index asc), and the exp-sum must match F64 within the
        // bound `sampling.total_band` relies on.
        for (0..64) |i| logits.floats()[1000 + 7 * i] = 12.5;
        const k = inference.sampling.TopK.capacity;
        const temperature: f32 = 0.7;
        const scratch = try b.topkBuffers(k);
        try b.begin();
        try b.topk(logits, 248320, k, temperature, scratch);
        try b.commit();
        const sorted = try alloc.alloc(inference.sampling.Candidate, 248320);
        defer alloc.free(sorted);
        for (sorted, logits.floats(), 0..) |*c, v, i| c.* = .{ .id = @intCast(i), .weight = v };
        std.mem.sort(inference.sampling.Candidate, sorted, {}, struct {
            fn less(_: void, lhs: inference.sampling.Candidate, rhs: inference.sampling.Candidate) bool {
                return lhs.weight > rhs.weight or (lhs.weight == rhs.weight and lhs.id < rhs.id);
            }
        }.less);
        const ids = @as([*]const u32, @ptrCast(@alignCast(scratch.indices.host)))[0..k];
        for (ids, scratch.values.floats()[0..k], sorted[0..k], 0..) |id, value, want, i| {
            if (id != want.id or value != @as(f32, @floatCast(want.weight))) {
                std.debug.print("top-k entry {d}: got ({d}, {d}) want ({d}, {d})\n", .{ i, id, value, want.id, want.weight });
                return error.MetalTopKMismatch;
            }
        }
        const flags = @as([*]const u32, @ptrCast(@alignCast(scratch.flags.host)))[0..Backend.topk_partials];
        for (flags) |flag| if (flag != 0) return error.MetalTopKFlagMismatch;
        // The exp-sum bound is measured on a flat vector (no dominating
        // logits, temperature 1.5) where ~10^5 terms contribute; the peaked
        // vector above sums to ~2 and would hide the F32 reduction error.
        for (logits.floats()) |*v| v.* = random.floatNorm(f32) * 2;
        const flat_temperature: f32 = 1.5;
        try b.begin();
        try b.topk(logits, 248320, k, flat_temperature, scratch);
        try b.commit();
        var maximum: f32 = -std.math.inf(f32);
        for (logits.floats()) |v| maximum = @max(maximum, v);
        if (scratch.values.floats()[0] != maximum) return error.MetalTopKMismatch;
        var exact: f64 = 0;
        for (logits.floats()) |v| exact += @exp((@as(f64, v) - maximum) / flat_temperature);
        var total: f64 = 0;
        for (scratch.sums.floats()[0..Backend.topk_partials]) |partial| total += partial;
        const relative = @abs(total - exact) / exact;
        std.debug.print("top-k exp-sum over 248,320 flat logits: gpu {d:.6} exact {d:.6} relative error {e:.2} (bound 2e-6)\n", .{ total, exact, relative });
        if (!(relative <= 2e-6)) return error.MetalTopKSumTolerance;
        // A single NaN raises its partition's flag and never enters the list.
        logits.floats()[77777] = std.math.nan(f32);
        try b.begin();
        try b.topk(logits, 248320, k, temperature, scratch);
        try b.commit();
        var raised = false;
        for (flags) |flag| raised = raised or flag != 0;
        if (!raised) return error.MetalTopKFlagMismatch;
        for (ids) |id| if (id == 77777) return error.MetalTopKMismatch;
        // Shape rejection: too many logits for the register-resident partial pass, k out of range, bad temperature.
        if (b.topk(logits, Backend.topk_max_count + 1, k, temperature, scratch)) |_| return error.ExpectedInvalidShape else |err| if (err != error.InvalidShape) return err;
        if (b.topk(logits, 248320, 0, temperature, scratch)) |_| return error.ExpectedInvalidShape else |err| if (err != error.InvalidShape) return err;
        if (b.topk(logits, 248320, k, 0, scratch)) |_| return error.ExpectedInvalidShape else |err| if (err != error.InvalidShape) return err;
    }

    // 9b. Chunk kernels equal their sequential single-token forms bit
    // for bit: rope over rows, convolution over rows plus history update,
    // grouped L2 norm, repeated delta gates, and copy.
    {
        var prng = std.Random.DefaultPrng.init(0xc4);
        const random = prng.random();
        const rows: usize = 5;
        // Rope: 3 heads of 8 dims per row at row stride 32, positions 7..11.
        const table = try b.create(16 * 4 * 8);
        try Backend.ropeTable(table, 16, 8, 1e4, null);
        const seq = try b.create(rows * 32 * 4);
        const chunk = try b.create(rows * 32 * 4);
        for (seq.floats(), chunk.floats()) |*x, *y| {
            x.* = random.floatNorm(f32);
            y.* = x.*;
        }
        try b.begin();
        for (0..rows) |t| try b.rope(seq.slice(t * 32 * 4, 32 * 4), table, 3, 8, 8, 7 + t, .split_half);
        try b.ropeRows(chunk, table, 3, 8, 8, 7, rows, 32, .split_half);
        // The adjacent pairing on top: the composition is still bit-identical.
        for (0..rows) |t| try b.rope(seq.slice(t * 32 * 4, 32 * 4), table, 3, 8, 8, 7 + t, .adjacent);
        try b.ropeRows(chunk, table, 3, 8, 8, 7, rows, 32, .adjacent);
        try b.commit();
        if (!std.mem.eql(f32, seq.floats(), chunk.floats())) return error.RopeRowsMismatch;
        // Convolution: 6 channels, 4 taps, row stride 9 (channels 6..8 untouched).
        const channels: usize = 6;
        const history_seq = try b.create(channels * 3 * 4);
        const history_chunk = try b.create(channels * 3 * 4);
        const conv_w = try b.create(channels * 4 * 4);
        const conv_in = try b.create(rows * 9 * 4);
        const out_seq = try b.create(rows * 9 * 4);
        const out_chunk = try b.create(rows * 9 * 4);
        for (history_seq.floats(), history_chunk.floats()) |*x, *y| {
            x.* = random.floatNorm(f32);
            y.* = x.*;
        }
        for (conv_w.floats()) |*w| w.* = random.floatNorm(f32);
        for (conv_in.floats()) |*x| x.* = random.floatNorm(f32);
        for (out_seq.floats(), out_chunk.floats()) |*x, *y| {
            x.* = 7;
            y.* = 7;
        }
        try b.begin();
        for (0..rows) |t| try b.convolution(history_seq, conv_in.slice(t * 9 * 4, 9 * 4), conv_w, out_seq.slice(t * 9 * 4, 9 * 4), channels, 4);
        try b.convolutionRows(history_chunk, conv_in, conv_w, out_chunk, channels, 4, rows, 9);
        try b.convolutionHistory(history_chunk, conv_in, channels, 4, rows, 9);
        try b.commit();
        if (!std.mem.eql(f32, out_seq.floats(), out_chunk.floats())) return error.ConvolutionRowsMismatch;
        if (!std.mem.eql(f32, history_seq.floats(), history_chunk.floats())) return error.ConvolutionHistoryMismatch;
        // L2 norm: 2 heads of width 4 at stride 8 within rows of stride 20.
        const l2_seq = try b.create(rows * 20 * 4);
        const l2_chunk = try b.create(rows * 20 * 4);
        for (l2_seq.floats(), l2_chunk.floats()) |*x, *y| {
            x.* = random.floatNorm(f32);
            y.* = x.*;
        }
        try b.begin();
        for (0..rows) |t| try b.l2Norm(l2_seq.slice(t * 20 * 4, 20 * 4), 2, 4, 8, 1e-6);
        try b.l2NormRows(l2_chunk, rows, 2, 4, 8, 20, 1e-6);
        try b.commit();
        if (!std.mem.eql(f32, l2_seq.floats(), l2_chunk.floats())) return error.L2NormRowsMismatch;
        // Delta gates: 3 heads per row.
        const ga = try b.create(3 * 4);
        const gb = try b.create(3 * 4);
        const alpha_seq = try b.create(rows * 3 * 4);
        const alpha_chunk = try b.create(rows * 3 * 4);
        const beta_seq = try b.create(rows * 3 * 4);
        const beta_chunk = try b.create(rows * 3 * 4);
        for (ga.floats(), gb.floats()) |*x, *y| {
            x.* = random.floatNorm(f32);
            y.* = random.floatNorm(f32);
        }
        for (alpha_seq.floats(), alpha_chunk.floats(), beta_seq.floats(), beta_chunk.floats()) |*a1, *a2, *b1, *b2| {
            a1.* = random.floatNorm(f32);
            a2.* = a1.*;
            b1.* = random.floatNorm(f32);
            b2.* = b1.*;
        }
        try b.begin();
        for (0..rows) |t| try b.deltaGates(alpha_seq.slice(t * 12, 12), beta_seq.slice(t * 12, 12), ga, gb, 3);
        try b.deltaGatesRows(alpha_chunk, beta_chunk, ga, gb, 3, rows);
        try b.commit();
        if (!std.mem.eql(f32, alpha_seq.floats(), alpha_chunk.floats()) or !std.mem.eql(f32, beta_seq.floats(), beta_chunk.floats())) return error.DeltaGatesRowsMismatch;
        // Copy: exact, and overlapping ranges are rejected.
        const dst = try b.create(64 * 4);
        try b.begin();
        try b.copy(dst, l2_chunk, 64);
        try b.commit();
        if (!std.mem.eql(f32, dst.floats(), l2_chunk.floats()[0..64])) return error.CopyMismatch;
        if (b.copy(dst, dst.slice(16, 128), 32)) |_| return error.ExpectedInvalidShape else |err| if (err != error.InvalidShape) return err;
    }

    // 9b. Mixture-of-experts kernels against the CPU references.
    try checkExperts(alloc, b);

    // 10. Recording contract: dispatch outside begin/commit is rejected, and a
    // committed empty pass is fine.
    {
        const x = try b.create(64);
        if (b.add(x, x, 16)) |_| return error.ExpectedNotRecording else |err| if (err != error.MetalNotRecording) return err;
        try b.begin();
        try b.commit();
    }

    // 11. Profiling: with a capacity of three timed dispatches per command
    // buffer, five recorded dispatches (two matvecs on a 35x1280 Q4_K matrix,
    // three adds) must yield three timed and two unsampled, keyed by kernel
    // and shape, with weight bytes attributed to the matvecs and results
    // unchanged by the per-dispatch encoders. `clear` empties the totals.
    {
        var diagnostic: [512]u8 = @splat(0);
        b.enableProfiling(3, &diagnostic) catch |err| {
            std.debug.print("profiling: {s}\n", .{std.mem.sliceTo(&diagnostic, 0)});
            return err;
        };
        if (b.enableProfiling(3, &diagnostic)) |_| return error.ProfilingEnabledTwice else |err| if (err != error.MetalAlreadyRecording) return err;
        const fixtures = try std.json.parseFromSlice(QuantFixture, alloc, @embedFile("src/quant/fixtures/k-affine.json"), .{ .ignore_unknown_fields = true });
        defer fixtures.deinit();
        var sample_bytes: ?[]const u8 = null;
        for (fixtures.value.rows) |sample| if (sample.encoding == 12) {
            sample_bytes = sample.bytes;
            break;
        };
        const region = try tiledMatrix(alloc, sample_bytes orelse return error.FixtureMissing, 12, 35, 1280);
        defer alloc.free(region);
        const matrix: inference.cpu.Matrix = .{ .rows = 35, .columns = 1280, .encoding = 12, .bytes = region };
        const weights = try uploadBytes(b, region);
        const input = try b.create(1280 * 4);
        for (input.floats(), 0..) |*x, i| x.* = @as(f32, @floatFromInt(i % 7)) / 7 - 0.5;
        const out = try b.create(35 * 4);
        const sum = try b.create(35 * 4);
        var expected: [35]f32 = undefined;
        var decoded: [1280]f32 = undefined;
        try inference.cpu.matvec(matrix, input.floats(), &expected, &decoded);
        try b.begin();
        try b.matvec(weights, matrix, input, out);
        try b.matvec(weights, matrix, input, sum);
        try b.add(sum, out, 35);
        try b.add(sum, out, 35);
        try b.add(sum, out, 35);
        try b.commit();
        for (sum.floats(), expected) |got, want| try expectClose("profiled matvec + adds", got, 4 * want, 1e-4 * @max(1, @abs(want)));
        const p = &b.profile.?;
        if (p.command_buffers != 1 or p.unsampled != 2 or p.pending.items.len != 0 or p.totals.count() != 2) {
            std.debug.print("profile: buffers {d} unsampled {d} pending {d} keys {d}\n", .{ p.command_buffers, p.unsampled, p.pending.items.len, p.totals.count() });
            return error.ProfileAccountingMismatch;
        }
        const mv = p.totals.get(.{ .kernel = .matvec_q4_k, .encoding = 12, .rows = 35, .columns = 1280 }) orelse return error.ProfileKeyMissing;
        const add = p.totals.get(.{ .kernel = .add, .encoding = null, .rows = 0, .columns = 0 }) orelse return error.ProfileKeyMissing;
        if (mv.dispatches != 2 or mv.bytes != 2 * region.len or !(mv.seconds > 0) or add.dispatches != 1 or add.bytes != 0 or !(add.seconds > 0)) {
            std.debug.print("profile: matvec {d} dispatches {d} bytes {d} s; add {d} dispatches {d} s\n", .{ mv.dispatches, mv.bytes, mv.seconds, add.dispatches, add.seconds });
            return error.ProfileTotalsMismatch;
        }
        // Encoder stamps lie inside the command buffer's GPU span, so their sum
        // cannot exceed it; the difference is the encoder-boundary cost.
        // Encoder stamps and the command buffer's GPUStart/EndTime come from
        // the same timeline but are sampled independently; on this ~100 µs
        // buffer the stamps overran the span by 9 µs in one of four runs
        // (2026-09-08: 52.2 µs span, 61.4 µs attributed). Allow that jitter.
        const attributed = mv.seconds + add.seconds;
        std.debug.print("profile check: command buffer {d:.1} us, attributed {d:.1} us\n", .{ p.gpu_seconds * 1e6, attributed * 1e6 });
        if (!(p.gpu_seconds + 20e-6 >= attributed)) return error.ProfileExceedsCommandBuffer;
        p.clear();
        if (p.totals.count() != 0 or p.command_buffers != 0 or p.unsampled != 0) return error.ProfileClearFailed;
        // An empty pass commits nothing and must not count as a command buffer.
        try b.begin();
        try b.commit();
        if (p.command_buffers != 0) return error.ProfileCountedEmptyPass;
    }

    std.debug.print("Metal fixtures passed: exact quantized decode for all encodings, randomized 1,280/5,120/17,408-column rows through specialized and generic matvec kernels with alignment fallback, subnormal Q4_K, dense F32/F16, pinned DeltaNet/convolution steps, multihead state, pinned and long attention, causal chunk attention against F64 per row with poisoned future rows, the F16 KV cache (exact half packing, half decode and chunk attention against the CPU over the rounded operands), flash-decoding attention on the pinned fixtures and up to 32,000 visible rows in both precisions, chunkwise DeltaNet against the F64 chunk reference and sequential steps with NaN past the chunk, windowed and wide chunk attention with the wide decode instantiation, norms, RoPE at 32767 with and without factors and in both pairings, activations and the epilogues, gates, argmax, partial top-k with exp-sum, batched matmul tiles for every encoding (half tiles against the generic F32 tile), chunk kernels equal to their sequential forms, the expert router with ties, the gathered expert matvec on both paths with NaN in the unselected experts and the decode chain against the F64 reference, the prefill lists and gathered tiles over a skewed chunk against the F64 reference and the decode path, repeated bridge lifetimes, the signed Hadamard transform and its inverse against the CPU on the rotated widths, and per-dispatch profiling with capacity overflow.\n", .{});
}
