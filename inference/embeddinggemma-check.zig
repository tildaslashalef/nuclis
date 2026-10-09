//! Explicit full-model check of EmbeddingGemma 2's CPU forward against the
//! recorded oracles (tests/fixtures/provenance.md, `embeddinggemma-*` rows).
//!
//!   embeddinggemma-check MODEL.gguf traces|vectors|google [FIXTURES_DIR]
//!
//! - `traces` (the Q8_0 file): the traced text cases stage by stage against
//!   llama.cpp's CPU pass over the same weights in F32, within bounds set by
//!   that reference's own approximations (its f16 GELU table, F32 angles).
//! - `vectors` (the Q8_0 file): every text-only case against llama.cpp's
//!   Q8_0 vectors.
//! - `google` (the BF16 file): every text-only case against Google's float32
//!   pass on the same weights, the accuracy claim.
//!
//! Token ids must equal the oracle's, except in the `literal_special` cases,
//! where they must differ and, tokenized the oracle's way, must match.
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
            self.missing = true;
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

fn isLiteral(id: []const u8) bool {
    for (literal_special) |l| if (std.mem.eql(u8, l, id)) return true;
    return false;
}

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len < 3 or args.len > 4) {
        std.debug.print("usage: embeddinggemma-check MODEL.gguf traces|vectors|google [FIXTURES_DIR]\n", .{});
        return error.InvalidArguments;
    }
    const fixtures = if (args.len == 4) args[3] else "tests/fixtures";
    const mode = args[2];
    const google_mode = std.mem.eql(u8, mode, "google");
    if (!google_mode and !std.mem.eql(u8, mode, "vectors") and !std.mem.eql(u8, mode, "traces")) return error.InvalidArguments;
    var embedder = try embed.Embedder.open(gpa, io, args[1]);
    defer embedder.deinit();
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
        for (cases) |case| {
            var prepared = try embedder.prepare(gpa, .{ .parts = &.{.{ .text = case.text }} }, .{});
            defer prepared.deinit(gpa);
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
            const c = cosine(&vector, want);
            low = @min(low, c);
            sum += c;
            counted += 1;
            if (c < floor) {
                failures += 1;
                std.debug.print("{s}: cosine {d:.7} ({d} tokens)\n", .{ case.id, c, prepared.tokens.len });
            }
        }
        std.debug.print("{d} text cases with the oracle's ids: cosine against {s} min {d:.7} (1 - min = {e:.2}), mean {d:.7} (floor {d})\n", .{ counted, if (google_mode) "Google f32" else "llama.cpp Q8_0", low, 1 - low, sum / @as(f64, @floatFromInt(counted)), floor });
    }
    if (failures != 0) {
        std.debug.print("{d} failures\n", .{failures});
        std.process.exit(1);
    }
}
