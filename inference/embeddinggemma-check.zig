//! Explicit full-model check of EmbeddingGemma 2's forward against the
//! recorded oracles (tests/fixtures/provenance.md, `embeddinggemma-*` rows).
//!
//!   embeddinggemma-check MODEL.gguf traces|vectors|google|images|bench [--backend cpu|metal] [--mmproj MMPROJ.gguf] [--only CASE] [FIXTURES_DIR]
//!
//! - `traces` (the Q8_0 file): the traced text cases stage by stage against
//!   llama.cpp's CPU pass over the same weights in F32, within bounds set by
//!   that reference's own approximations (its f16 GELU table, F32 angles).
//! - `vectors` (the Q8_0 file): every text-only case against llama.cpp's
//!   Q8_0 vectors.
//! - `google` (the BF16 file): every text-only case against Google's float32
//!   pass on the same weights, the accuracy claim.
//! - `images` (either file, with `--mmproj`): the vision encoder from
//!   llama.cpp's pixels (its resize, F16 patches and gelu_quick) against
//!   its projector rows, then every image case through Google's pipeline
//!   against Google's float32 vectors (at the `google` floor on the BF16
//!   file). The rows are checked once per distinct image; `--only` keeps
//!   one case (the CPU reference takes about 3 minutes an image).
//! - `bench`: wall-clock rates, no oracle. 64 inputs of 256 tokens embedded
//!   as one batch, then one input of 512 and one of 8192 tokens; each input
//!   is BOS, " the" repeated, EOS (the cost does not depend on the text).
//!   One untimed warm-up of each shape, then the median of the timed runs.
//!
//! Token ids must equal the oracle's, except in the `literal_special` cases,
//! where they must differ and, tokenized the oracle's way, must match. On
//! Metal, `vectors` and `google` also embed every text case in packed
//! batches, which must give the one-at-a-time vectors bit for bit.
//! FIXTURES_DIR defaults to `tests/fixtures`, read at run time.
const std = @import("std");
const inference = @import("inference");
const embed = inference.embed;

/// Per stage: the largest difference over the largest reference value, and
/// the difference's RMS over the reference's.
const max_relative = 2e-3;
const max_relative_rms = 5e-4;
/// Cosine floors: against the reference build on the same Q8_0 file, and
/// against Google's float32 on the BF16 file.
const min_cosine_llama = 0.99999;
const min_cosine_google = 0.9999999;
const layers = 24;
const width = 512;
/// Cases whose text contains a control token's spelling: the oracles parsed
/// it into the token, nuclis keeps it literal (the input contract).
const literal_special = [_][]const u8{"long-8k.doc"};

const Vectors = struct {
    ids: std.StringHashMap([]const u32),
    values: std.StringHashMap([]const f32),

    fn load(arena: std.mem.Allocator, io: std.Io, prefix: []const u8) !Vectors {
        const meta_text = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(arena, "{s}.json", .{prefix}), arena, .limited(64 << 20));
        const raw = try std.Io.Dir.cwd().readFileAlloc(io, try std.fmt.allocPrint(arena, "{s}.f32", .{prefix}), arena, .limited(64 << 20));
        const meta = try std.json.parseFromSliceLeaky(std.json.Value, arena, meta_text, .{});
        const dims: usize = @intCast(meta.object.get("dimensions").?.integer);
        const cases = meta.object.get("cases").?.array.items;
        if (raw.len != cases.len * dims * 4) return error.InvalidFixture;
        var out: Vectors = .{ .ids = .init(arena), .values = .init(arena) };
        for (cases, 0..) |c, i| {
            const id = c.object.get("id").?.string;
            var ids: std.ArrayList(u32) = .empty;
            for (c.object.get("ids").?.array.items) |item| switch (item) {
                .integer => |n| try ids.append(arena, @intCast(n)),
                .array => |run| for (0..@intCast(run.items[1].integer)) |_| try ids.append(arena, @intCast(run.items[0].integer)),
                else => return error.InvalidFixture,
            };
            const values = try arena.alloc(f32, dims);
            for (values, 0..) |*v, k| v.* = @bitCast(std.mem.readInt(u32, raw[(i * dims + k) * 4 ..][0..4], .little));
            try out.ids.put(id, ids.items);
            try out.values.put(id, values);
        }
        return out;
    }
};

const Case = struct { id: []const u8, text: []const u8 };

/// The text-only cases of inputs.json, their parts joined.
fn textCases(arena: std.mem.Allocator, io: std.Io, fixtures: []const u8) ![]Case {
    const path = try std.fmt.allocPrint(arena, "{s}/embeddinggemma-inputs/inputs.json", .{fixtures});
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, arena, try std.Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(16 << 20)), .{});
    var out: std.ArrayList(Case) = .empty;
    outer: for (doc.object.get("cases").?.array.items) |c| {
        var text: std.ArrayList(u8) = .empty;
        for (c.object.get("parts").?.array.items) |part| {
            const t = part.object.get("text") orelse continue :outer;
            try text.appendSlice(arena, t.string);
        }
        try out.append(arena, .{ .id = c.object.get("id").?.string, .text = text.items });
    }
    return out.items;
}

fn cosine(a: []const f32, b: []const f32) f64 {
    var dot: f64 = 0;
    var na: f64 = 0;
    var nb: f64 = 0;
    for (a, b) |x, y| {
        dot += @as(f64, x) * y;
        na += @as(f64, x) * x;
        nb += @as(f64, y) * y;
    }
    return dot / @sqrt(na * nb);
}

/// Compares each observed stage with the reference trace directory.
const Trace = struct {
    arena: std.mem.Allocator,
    io: std.Io,
    directory: []const u8,
    rows: usize,
    worst_max: f64 = 0,
    worst_rel: f64 = 0,
    worst: []const u8 = "",
    failed: bool = false,
    missing: bool = false,
    /// Stages the directory does not hold are skipped, not missing.
    partial: bool = false,

    fn file(self: *Trace, name: []const u8) ?[]const u8 {
        const path = std.fmt.allocPrint(self.arena, "{s}/{s}.f32", .{ self.directory, name }) catch return null;
        return std.Io.Dir.cwd().readFileAlloc(self.io, path, self.arena, .limited(256 << 20)) catch null;
    }

    fn record(context: *anyopaque, stage: embed.Stage, rows: []const f32) void {
        const self: *Trace = @ptrCast(@alignCast(context));
        var buffer: [32]u8 = undefined;
        const name, const raw, const offset, const stride = switch (stage) {
            .input => .{ "inp_scaled", self.file("inp_scaled"), 0, width },
            .layer => |l| blk: {
                const n = std.fmt.bufPrint(&buffer, "l_out-{d}", .{l}) catch unreachable;
                break :blk .{ n, self.file(n), 0, width };
            },
            // [512, 24, rows] in ggml order: row r of layer l at (r·24 + l)·512.
            .per_layer => |l| .{ "inp_per_layer", self.file("inp_per_layer"), l * width, layers * width },
            .final_norm => .{ "result_norm", self.file("result_norm"), 0, width },
            .projected => .{ "result_embd", self.file("result_embd"), 0, embed.dimensions },
        };
        const bytes = raw orelse {
            if (!self.partial) self.missing = true;
            return;
        };
        const columns = rows.len / self.rows;
        var diff2: f64 = 0;
        var ref2: f64 = 0;
        var worst: f64 = 0;
        var largest: f64 = 0;
        for (0..self.rows) |r| for (0..columns) |c| {
            const at = r * stride + offset + c;
            if ((at + 1) * 4 > bytes.len) {
                self.missing = true;
                return;
            }
            const want: f64 = @as(f32, @bitCast(std.mem.readInt(u32, bytes[at * 4 ..][0..4], .little)));
            const delta = rows[r * columns + c] - want;
            worst = @max(worst, @abs(delta));
            largest = @max(largest, @abs(want));
            diff2 += delta * delta;
            ref2 += want * want;
        };
        const rel = @sqrt(diff2 / @max(ref2, 1e-30));
        const max_rel = worst / @max(largest, 1e-30);
        const label = switch (stage) {
            .per_layer => |l| std.fmt.allocPrint(self.arena, "inp_per_layer[{d}]", .{l}) catch "inp_per_layer",
            else => self.arena.dupe(u8, name) catch name,
        };
        if (max_rel > max_relative or rel > max_relative_rms) {
            self.failed = true;
            std.debug.print("  {s}: relative max {e:.3}, relative RMS {e:.3} (bounds {e}, {e})\n", .{ label, max_rel, rel, max_relative, max_relative_rms });
        }
        if (rel > self.worst_rel) {
            self.worst_rel = rel;
            self.worst = label;
        }
        self.worst_max = @max(self.worst_max, max_rel);
    }
};

/// An input of `tokens` rows: BOS, `filler` repeated, EOS.
fn benchInput(arena: std.mem.Allocator, embedder: *const embed.Embedder, filler: u32, tokens: usize) !embed.Prepared {
    const ids = try arena.alloc(u32, tokens);
    @memset(ids, filler);
    ids[0] = embedder.bos.?;
    ids[tokens - 1] = embedder.eos.?;
    return .{ .tokens = ids, .length = tokens };
}

const Timing = struct { wall_ms: f64, gpu_ms: f64 };

/// Median wall and GPU milliseconds of `runs` timed batches, after one
/// untimed; GPU time is the Metal command buffers' (0 on the CPU).
fn timeBatch(io: std.Io, embedder: *embed.Embedder, inputs: []const embed.Prepared, outs: []const []f32, runs: usize) !Timing {
    var walls: [16]f64 = undefined;
    var gpus: [16]f64 = undefined;
    try embedder.embedBatch(inputs, outs);
    for (walls[0..runs], gpus[0..runs]) |*wall, *gpu| {
        const gpu_start = gpuSeconds(embedder);
        const start = std.Io.Clock.awake.now(io);
        try embedder.embedBatch(inputs, outs);
        wall.* = @as(f64, @floatFromInt(start.durationTo(std.Io.Clock.awake.now(io)).toNanoseconds())) / 1e6;
        gpu.* = (gpuSeconds(embedder) - gpu_start) * 1000;
    }
    std.mem.sort(f64, walls[0..runs], {}, std.sort.asc(f64));
    std.mem.sort(f64, gpus[0..runs], {}, std.sort.asc(f64));
    return .{ .wall_ms = walls[runs / 2], .gpu_ms = gpus[runs / 2] };
}

fn gpuSeconds(embedder: *const embed.Embedder) f64 {
    return switch (embedder.engine) {
        .metal => |plan| plan.backend.gpuSeconds(),
        .cpu => 0,
    };
}

fn bench(arena: std.mem.Allocator, io: std.Io, embedder: *embed.Embedder) !void {
    const filler = try embedder.encoder.encode(arena, " the", false, .{});
    if (filler.len != 1) return error.InvalidFixture;
    const batch = 64;
    const short = 256;
    const inputs = try arena.alloc(embed.Prepared, batch);
    const outs = try arena.alloc([]f32, batch);
    for (inputs, outs) |*input, *out| {
        input.* = try benchInput(arena, embedder, filler[0], short);
        out.* = try arena.alloc(f32, embed.dimensions);
    }
    const t = try timeBatch(io, embedder, inputs, outs, 7);
    std.debug.print("{d} inputs of {d} tokens: {d:.1} ms ({d:.1} ms GPU), {d:.1} inputs/s, {d:.0} tokens/s\n", .{ batch, short, t.wall_ms, t.gpu_ms, batch * 1000 / t.wall_ms, batch * short * 1000 / t.wall_ms });
    const long = [_]embed.Prepared{try benchInput(arena, embedder, filler[0], embed.max_tokens)};
    for ([_]usize{ 512, embed.max_tokens }) |tokens| {
        const one = [_]embed.Prepared{try benchInput(arena, embedder, filler[0], tokens)};
        const one_t = try timeBatch(io, embedder, &one, outs[0..1], 5);
        std.debug.print("one input of {d} tokens: {d:.1} ms ({d:.1} ms GPU), {d:.0} tokens/s\n", .{ tokens, one_t.wall_ms, one_t.gpu_ms, @as(f64, @floatFromInt(tokens)) * 1000 / one_t.wall_ms });
    }
    // Per kernel and shape, from the GPU's timestamps: the batch, then the long input.
    const plan = switch (embedder.engine) {
        .metal => |plan| plan,
        .cpu => return,
    };
    var diagnostic: [512]u8 = @splat(0);
    try plan.backend.enableProfiling(4096, &diagnostic);
    for ([_][]const embed.Prepared{ inputs, &long }) |set| {
        plan.backend.profile.?.clear();
        try embedder.embedBatch(set, outs[0..set.len]);
        const profile = &plan.backend.profile.?;
        const Row = struct { key: inference.metal.Profile.Key, total: inference.metal.Profile.Total };
        var rows: std.ArrayList(Row) = .empty;
        var it = profile.totals.iterator();
        while (it.next()) |e| try rows.append(arena, .{ .key = e.key_ptr.*, .total = e.value_ptr.* });
        std.mem.sort(Row, rows.items, {}, struct {
            fn more(_: void, a: Row, b: Row) bool {
                return a.total.seconds > b.total.seconds;
            }
        }.more);
        var sum: f64 = 0;
        for (rows.items) |r| sum += r.total.seconds;
        std.debug.print("profile, {d} inputs: {d:.1} ms of dispatches, {d:.1} ms GPU\n", .{ set.len, sum * 1000, profile.gpu_seconds * 1000 });
        for (rows.items[0..@min(rows.items.len, 14)]) |r| std.debug.print("  {s:<28} {d:>5}x{d:<5} enc {?d:<3} {d:>4} calls {d:>8.1} ms\n", .{ @tagName(r.key.kernel), r.key.rows, r.key.columns, r.key.encoding, r.total.dispatches, r.total.seconds * 1000 });
    }
}

/// Bounds for the vision encoder's rows against llama.cpp's, recorded on
/// Metal, whose batched matmuls stage activations as half or bf16 through
/// 16 blocks: measured 2.7e-2 and 8.3e-3. A layout or weight defect is
/// of order one.
const max_relative_media = 5e-2;
const max_relative_rms_media = 1.5e-2;
/// Google's pipeline on the Q8_0 file against Google's float32 vectors.
const min_cosine_images_q8 = 0.9999;

const MediaCase = struct { id: []const u8, parts: []const embed.Part };

/// The cases of inputs.json that hold an image, their image parts read
/// from the paths the file gives (relative to the working directory).
fn imageCases(arena: std.mem.Allocator, io: std.Io, fixtures: []const u8) ![]MediaCase {
    const path = try std.fmt.allocPrint(arena, "{s}/embeddinggemma-inputs/inputs.json", .{fixtures});
    const doc = try std.json.parseFromSliceLeaky(std.json.Value, arena, try std.Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(16 << 20)), .{});
    var out: std.ArrayList(MediaCase) = .empty;
    for (doc.object.get("cases").?.array.items) |c| {
        const items = c.object.get("parts").?.array.items;
        var parts: std.ArrayList(embed.Part) = .empty;
        var has_image = false;
        for (items) |part| {
            if (part.object.get("text")) |t| {
                try parts.append(arena, .{ .text = t.string });
            } else if (part.object.get("image")) |file| {
                has_image = true;
                try parts.append(arena, .{ .image = try std.Io.Dir.cwd().readFileAlloc(io, file.string, arena, .limited(64 << 20)) });
            } else break;
        } else if (has_image) try out.append(arena, .{ .id = c.object.get("id").?.string, .parts = parts.items });
    }
    return out.items;
}

fn readF32(arena: std.mem.Allocator, io: std.Io, path: []const u8) ![]f32 {
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(256 << 20));
    const out = try arena.alloc(f32, bytes.len / 4);
    for (out, 0..) |*v, i| v.* = @bitCast(std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little));
    return out;
}

/// Relative max and relative RMS of `got` against `want`.
fn relative(got: []const f32, want: []const f32) struct { max: f64, rms: f64 } {
    var diff2: f64 = 0;
    var ref2: f64 = 0;
    var worst: f64 = 0;
    var largest: f64 = 0;
    for (got, want) |g, w| {
        const delta = @as(f64, g) - w;
        worst = @max(worst, @abs(delta));
        largest = @max(largest, @abs(@as(f64, w)));
        diff2 += delta * delta;
        ref2 += @as(f64, w) * w;
    }
    return .{ .max = worst / @max(largest, 1e-30), .rms = @sqrt(diff2 / @max(ref2, 1e-30)) };
}

fn images(gpa: std.mem.Allocator, arena: std.mem.Allocator, io: std.Io, embedder: *embed.Embedder, fixtures: []const u8, only: ?[]const u8) !void {
    const vision = inference.vision;
    const gemma4v = vision.gemma4;
    const google = try Vectors.load(arena, io, try std.fmt.allocPrint(arena, "{s}/embeddinggemma-vectors/st-f32", .{fixtures}));
    var cases = try imageCases(arena, io, fixtures);
    if (only) |id| cases = for (cases, 0..) |c, i| {
        if (std.mem.eql(u8, c.id, id)) break cases[i..][0..1];
    } else return error.UnknownCase;
    const projector = &embedder.projector.?;
    var checked: std.ArrayList([]const u8) = .empty;
    var failures: usize = 0;
    var vector: [embed.dimensions]f32 = undefined;

    for (cases) |case| {
        const directory = try std.fmt.allocPrint(arena, "{s}/embeddinggemma-{s}", .{ fixtures, case.id });
        const want_rows = readF32(arena, io, try std.fmt.allocPrint(arena, "{s}/media-0.f32", .{directory})) catch continue;
        // The reference's pipeline: its smart size, letterboxed Pillow bicubic, F16 patches, gelu_quick.
        const bytes = for (case.parts) |p| switch (p) {
            .image => |b| break b,
            else => {},
        } else unreachable;
        const seen = for (checked.items) |b| {
            if (std.mem.eql(u8, b, bytes)) break true;
        } else false;
        if (seen) continue;
        try checked.append(arena, bytes);
        var decoded = try vision.image.decode(gpa, bytes);
        defer decoded.deinit(gpa);
        const grid = gemma4v.gridFor(.{ .width = decoded.width, .height = decoded.height }, gemma4v.min_tokens, embed.default_image_tokens);
        const target = grid.pixels();
        const resized = try vision.preprocess.resizeLetterbox(gpa, decoded, target);
        defer gpa.free(resized);
        var patches = try vision.preprocess.patches(gpa, resized, target, gemma4v.patchOptions(.siglip, projector.binding.mean, projector.binding.std));
        defer patches.deinit(gpa);
        const rows = try arena.alloc(f32, grid.tokens() * width);
        projector.binding.activation = .gelu_quick;
        try embedder.encodePatches(patches, grid, rows);
        projector.binding.activation = .gelu_tanh;
        if (rows.len != want_rows.len) return error.InvalidFixture;
        const r = relative(rows, want_rows);
        const rows_ok = r.max <= max_relative_media and r.rms <= max_relative_rms_media;
        if (!rows_ok) failures += 1;
        std.debug.print("{s}: projector rows from the reference's pixels, {d} rows: relative max {e:.3}, relative RMS {e:.3} (bounds {e}, {e})\n", .{ case.id, grid.tokens(), r.max, r.rms, max_relative_media, max_relative_rms_media });
    }

    // Google's pipeline end to end.
    // Google's float32 on the BF16 file (encoding 30), as the text gate; the Q8_0 file's
    // own cost (about 5e-5 of cosine on text) on the other.
    const floor: f64 = if (embedder.binding.token_embedding.encoding_id == 30) min_cosine_google else min_cosine_images_q8;
    var low: f64 = 1;
    const prepared_all = try arena.alloc(embed.Prepared, cases.len);
    const singles = try arena.alloc([embed.dimensions]f32, cases.len);
    for (cases, prepared_all, singles) |case, *prepared, *single| {
        const started = std.Io.Clock.awake.now(io);
        prepared.* = try embedder.prepare(arena, .{ .parts = case.parts }, .{});
        const prepare_ms = started.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds();
        if (!std.mem.eql(u32, prepared.tokens, google.ids.get(case.id).?)) {
            failures += 1;
            std.debug.print("{s}: token ids differ from Google's\n", .{case.id});
            continue;
        }
        try embedder.embed(prepared.*, &vector, null);
        single.* = vector;
        const c = cosine(&vector, google.values.get(case.id).?);
        low = @min(low, c);
        if (c < floor) failures += 1;
        std.debug.print("{s}: {d} tokens, prepared in {d} ms, cosine against Google's f32 {d:.7} (1 - c = {e:.2})\n", .{ case.id, prepared.tokens.len, prepare_ms, c, 1 - c });
    }
    std.debug.print("{d} image cases: min cosine against Google's f32 {d:.7} (floor {d}); the vision encoder holds {d} MB\n", .{ cases.len, low, floor, embedder.mediaBytes() / 1_000_000 });
    if (embedder.engine == .metal and failures == 0) {
        // Batching changes timing, never answers, with media rows as with text.
        const outs = try arena.alloc([]f32, cases.len);
        const batched = try arena.alloc([embed.dimensions]f32, cases.len);
        for (outs, batched) |*o, *b| o.* = b;
        try embedder.embedBatch(prepared_all, outs);
        var differing: usize = 0;
        for (singles, batched) |a, b| {
            if (!std.mem.eql(u8, std.mem.asBytes(&a), std.mem.asBytes(&b))) differing += 1;
        }
        std.debug.print("{d} image cases in one packed batch: {d} vectors differ from one at a time\n", .{ cases.len, differing });
        if (differing != 0) failures += 1;
    }
    if (failures != 0) {
        std.debug.print("{d} failures\n", .{failures});
        std.process.exit(1);
    }
}

fn isLiteral(id: []const u8) bool {
    for (literal_special) |l| if (std.mem.eql(u8, l, id)) return true;
    return false;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    const usage = "usage: embeddinggemma-check MODEL.gguf traces|vectors|google|images|bench [--backend cpu|metal] [--mmproj MMPROJ.gguf] [--only CASE] [FIXTURES_DIR]\n";
    if (args.len < 3) {
        std.debug.print(usage, .{});
        return error.InvalidArguments;
    }
    var backend: embed.Backend = .cpu;
    var fixtures: []const u8 = "tests/fixtures";
    var rest = args[3..];
    if (rest.len >= 2 and std.mem.eql(u8, rest[0], "--backend")) {
        backend = std.meta.stringToEnum(embed.Backend, rest[1]) orelse {
            std.debug.print(usage, .{});
            return error.InvalidArguments;
        };
        rest = rest[2..];
    }
    var mmproj: ?[]const u8 = null;
    if (rest.len >= 2 and std.mem.eql(u8, rest[0], "--mmproj")) {
        mmproj = rest[1];
        rest = rest[2..];
    }
    var only: ?[]const u8 = null;
    if (rest.len >= 2 and std.mem.eql(u8, rest[0], "--only")) {
        only = rest[1];
        rest = rest[2..];
    }
    if (rest.len > 1) {
        std.debug.print(usage, .{});
        return error.InvalidArguments;
    }
    if (rest.len == 1) fixtures = rest[0];
    const mode = args[2];
    const google_mode = std.mem.eql(u8, mode, "google");
    const bench_mode = std.mem.eql(u8, mode, "bench");
    const images_mode = std.mem.eql(u8, mode, "images");
    if (images_mode and mmproj == null) {
        std.debug.print("images needs --mmproj\n", .{});
        return error.InvalidArguments;
    }
    if (!google_mode and !bench_mode and !images_mode and !std.mem.eql(u8, mode, "vectors") and !std.mem.eql(u8, mode, "traces")) return error.InvalidArguments;
    const load_start = std.Io.Clock.awake.now(io);
    var embedder = try embed.Embedder.open(gpa, io, args[1], backend, .{ .projector = mmproj });
    defer embedder.deinit();
    if (images_mode) return images(gpa, arena, io, embedder, fixtures, only);
    if (bench_mode) {
        std.debug.print("open ({s}): {d} ms\n", .{ @tagName(backend), load_start.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds() });
        return bench(arena, io, embedder);
    }
    const reference = try Vectors.load(arena, io, try std.fmt.allocPrint(arena, "{s}/embeddinggemma-vectors/{s}", .{ fixtures, if (google_mode) "st-f32" else "llama-q8_0" }));
    const floor: f64 = if (google_mode) min_cosine_google else min_cosine_llama;
    const cases = try textCases(arena, io, fixtures);
    var vector: [embed.dimensions]f32 = undefined;
    var failures: usize = 0;

    if (std.mem.eql(u8, mode, "traces")) {
        for ([_][]const u8{ "north.raw", "north.query" }) |id| {
            const case = for (cases) |c| {
                if (std.mem.eql(u8, c.id, id)) break c;
            } else return error.InvalidFixture;
            var prepared = try embedder.prepare(gpa, .{ .parts = &.{.{ .text = case.text }} }, .{});
            defer prepared.deinit(gpa);
            if (!std.mem.eql(u32, prepared.tokens, reference.ids.get(id).?)) return error.TokenMismatch;
            var trace: Trace = .{ .arena = arena, .io = io, .directory = try std.fmt.allocPrint(arena, "{s}/embeddinggemma-{s}", .{ fixtures, id }), .rows = prepared.tokens.len };
            try embedder.embed(prepared, &vector, .{ .context = &trace, .record = Trace.record });
            const c = cosine(&vector, reference.values.get(id).?);
            if (trace.missing) return error.MissingTrace;
            if (trace.failed or c < floor) failures += 1;
            std.debug.print("{s}: {d} rows, worst stage {s} (relative RMS {e:.3}), worst relative max {e:.3}, cosine {d:.7}\n", .{ id, trace.rows, trace.worst, trace.worst_rel, trace.worst_max, c });
        }
    } else {
        var low: f64 = 1;
        var sum: f64 = 0;
        var counted: usize = 0;
        // Each case's prepared input and its one-at-a-time vector, for the batch check.
        const prepared_all = try arena.alloc(embed.Prepared, cases.len);
        const singles = try arena.alloc([embed.dimensions]f32, cases.len);
        @memset(singles, @splat(0));
        for (cases, prepared_all, singles) |case, *kept, *single| {
            const prepared = try embedder.prepare(arena, .{ .parts = &.{.{ .text = case.text }} }, .{});
            kept.* = prepared;
            const want_ids = reference.ids.get(case.id).?;
            const want = reference.values.get(case.id).?;
            if (isLiteral(case.id)) {
                // The literal tokens must differ; the oracle's way must match it exactly.
                const parsed = try embedder.encoder.encode(gpa, case.text, true, .{});
                defer gpa.free(parsed);
                const oracle_way = try std.mem.concat(gpa, u32, &.{ &.{embedder.bos.?}, parsed, &.{embedder.eos.?} });
                defer gpa.free(oracle_way);
                if (std.mem.eql(u32, prepared.tokens, want_ids) or !std.mem.eql(u32, oracle_way, want_ids)) {
                    failures += 1;
                    std.debug.print("{s}: literal ids should differ from the oracle's and parsed ones equal them\n", .{case.id});
                    continue;
                }
                try embedder.embed(.{ .tokens = oracle_way, .length = oracle_way.len }, &vector, null);
                const parsed_cosine = cosine(&vector, want);
                try embedder.embed(prepared, &vector, null);
                single.* = vector;
                std.debug.print("{s}: {d} literal tokens, cosine {d:.7}; specials parsed as the oracle does, {d} tokens, cosine {d:.7}\n", .{ case.id, prepared.tokens.len, cosine(&vector, want), oracle_way.len, parsed_cosine });
                if (parsed_cosine < floor) failures += 1;
                continue;
            }
            if (!std.mem.eql(u32, prepared.tokens, want_ids)) {
                failures += 1;
                std.debug.print("{s}: token ids differ from the oracle's\n", .{case.id});
                continue;
            }
            try embedder.embed(prepared, &vector, null);
            single.* = vector;
            const c = cosine(&vector, want);
            low = @min(low, c);
            sum += c;
            counted += 1;
            if (c < floor) {
                failures += 1;
                std.debug.print("{s}: cosine {d:.7} ({d} tokens)\n", .{ case.id, c, prepared.tokens.len });
            }
        }
        std.debug.print("{d} text cases with the oracle's ids ({s}): cosine against {s} min {d:.7} (1 - min = {e:.2}), mean {d:.7} (floor {d})\n", .{ counted, @tagName(backend), if (google_mode) "Google f32" else "llama.cpp Q8_0", low, 1 - low, sum / @as(f64, @floatFromInt(counted)), floor });
        if (backend == .metal) {
            // Batching changes timing, never answers.
            const batched = try arena.alloc([embed.dimensions]f32, cases.len);
            const outs = try arena.alloc([]f32, cases.len);
            for (outs, batched) |*o, *b| o.* = b;
            try embedder.embedBatch(prepared_all, outs);
            var differing: usize = 0;
            var worst: f32 = 0;
            for (singles, batched) |a, b| {
                if (!std.mem.eql(u8, std.mem.asBytes(&a), std.mem.asBytes(&b))) differing += 1;
                for (a, b) |x, y| worst = @max(worst, @abs(x - y));
            }
            std.debug.print("{d} cases in packed batches: {d} vectors differ from one at a time, worst |difference| {e:.3}\n", .{ cases.len, differing, worst });
            if (differing != 0) failures += 1;
        }
    }
    if (failures != 0) {
        std.debug.print("{d} failures\n", .{failures});
        std.process.exit(1);
    }
}
