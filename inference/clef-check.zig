//! Explicit check of clef-flash against the oracle's fixtures
//! (scripts/clef-reference.py at tag v0.11.0); CLEF is the clef GGUF.
//! `sequences CLEF` rebuilds every request's model input and compares the
//! ids and the head's spans with `sequences.json`, tokenizing with the
//! file's vocabulary (only its directory is read, not the weights).
//! `head CLEF DUMPS HEAD_JSON` runs nuclis's head on the dumped inputs
//! (`DUMPS/<name>.hidden.f32`, `.lexical.f32`) and compares its logits with
//! the reference head's on the same inputs. `run CLEF` is the whole
//! model on a backend: every text request through the backbone and the
//! head, its logits against `head.json` (the reference head fed the Metal
//! backbone's dumps) and its answers against the sanity set's; `--dump`
//! writes the head's inputs for the script, `--max-tokens` skips longer
//! requests (the CPU backbone runs about a token a second), `--only` runs
//! one; the image requests run when `--mmproj` names the projector.
const std = @import("std");
const inference = @import("inference");
const check_stats = @import("check_stats.zig");
const clef = inference.profiles.clef;
const Encoder = inference.tokenizer.Encoder;

const sequences_json = @embedFile("src/models/fixtures/clef/sequences.json");

pub const Field = struct { id: []const u8, type: u32, span: [2]u32, option_spans: []const [2]u32, option_ids: []const []const u8 };
pub const Media = struct { token_offset: usize, image_grid_thw: []const [3]u32, image_tokens: []const u32 };
pub const Entry = struct {
    name: []const u8,
    request: std.json.Value,
    expect: std.json.Value,
    rendered_state: []const u8,
    input_ids: []const u32,
    questions: []const Field,
    media: ?Media = null,
};
pub const File = struct { requests: []const Entry };

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    if (args.len == 5 and std.mem.eql(u8, args[1], "head")) return checkHead(gpa, io, arena, args[2], args[3], args[4]);
    if (args.len >= 3 and std.mem.eql(u8, args[1], "run")) return checkRun(gpa, io, arena, args[2], args[3..]);
    if (args.len != 3 or !std.mem.eql(u8, args[1], "sequences")) {
        std.debug.print("usage: clef-check sequences CLEF | head CLEF DUMPS HEAD_JSON | run CLEF [--backend cpu|metal] [--mmproj GGUF] [--dump DIR] [--max-tokens N] [--only NAME]\n", .{});
        return error.InvalidArguments;
    }
    var mapped = try inference.weights.Mapped.open(gpa, io, args[2]);
    defer mapped.deinit(io);
    var vocab = try inference.vocabulary.load(gpa, mapped.document, mapped.mapping.memory[0..@intCast(mapped.document.directory_bytes)], .{});
    defer vocab.deinit();
    var tokenizer = try Encoder.init(gpa, &vocab);
    defer tokenizer.deinit();
    const file = try std.json.parseFromSliceLeaky(File, arena, sequences_json, .{ .ignore_unknown_fields = true });
    const frame = try clef.Frame.init(arena, &tokenizer);
    if (file.requests.len == 0) return error.NoRequests;
    var failed = false;
    for (file.requests) |entry| {
        const built = try sequenceFor(arena, &tokenizer, frame, entry);
        const same_ids = std.mem.eql(u32, built.ids, entry.input_ids);
        var same_spans = built.fields.len == entry.questions.len;
        if (same_spans) for (built.fields, entry.questions) |f, q| {
            same_spans = same_spans and f.span.start == q.span[0] and f.span.end == q.span[1] and f.options.len == q.option_spans.len and clef.typeId(f.kind) == q.type;
            if (same_spans) for (f.options, q.option_spans) |o, s| {
                same_spans = same_spans and o.start == s[0] and o.end == s[1];
            };
        };
        const ok = same_ids and same_spans;
        failed = failed or !ok;
        std.debug.print("{s}: {d} tokens, ids {s}, spans {s}\n", .{ entry.name, built.ids.len, if (same_ids) "equal" else "DIFFER", if (same_spans) "equal" else "DIFFER" });
        if (!same_ids) for (built.ids, 0..) |id, i| {
            if (i >= entry.input_ids.len or id != entry.input_ids[i]) {
                std.debug.print("  first difference at {d}: {d} against {d} (lengths {d}, {d})\n", .{ i, id, if (i < entry.input_ids.len) entry.input_ids[i] else 0, built.ids.len, entry.input_ids.len });
                break;
            }
        };
    }
    if (failed) return error.SequenceMismatch;
    std.debug.print("every sequence equals the reference's ({d} requests)\n", .{file.requests.len});
}

/// A fixture request through the profile, images as the reference's token
/// counts (the projector's own count is the vision check's business).
pub fn sequenceFor(arena: std.mem.Allocator, tokenizer: *const Encoder, frame: clef.Frame, entry: Entry) !clef.Sequence {
    const object = entry.request.object;
    const map = object.get("questions").?.object;
    const ids = try arena.alloc([]const u8, map.count());
    const questions = try arena.alloc(clef.Question, map.count());
    for (map.keys(), map.values(), ids, questions) |key, value, *id, *q| {
        var diag: clef.Diagnostic = .{};
        id.* = key;
        q.* = clef.parseQuestion(arena, key, value, &diag) catch |err| {
            std.debug.print("{s}: {s}\n", .{ entry.name, diag.message() });
            return err;
        };
    }
    const schema = try clef.schema(arena, tokenizer, ids, questions);
    var counts: std.ArrayList(usize) = .empty;
    if (entry.media) |m| for (m.image_tokens) |n| try counts.append(arena, n);
    const media = try clef.mediaIds(arena, tokenizer, counts.items);
    const state = try clef.tokens(arena, tokenizer, try clef.render(arena, object.get("state").?));
    return clef.assemble(arena, frame, schema, media, state, null);
}

/// Per logit, the difference over max(1, |logit|) from the oracle's head
/// fed the same rows: the head's Q8_0 matrices moved them by at most
/// 2.62e-3 (multi_question, 2026-11-20); summation order adds about 1e-5.
const logit_bound = 5e-3;

const HeadResult = struct { name: []const u8, logits: []const []const f32 };
const HeadFile = struct { results: []const HeadResult };

fn checkHead(gpa: std.mem.Allocator, io: std.Io, arena: std.mem.Allocator, clef_path: []const u8, dumps: []const u8, head_json: []const u8) !void {
    var mapped = try inference.weights.Mapped.open(gpa, io, clef_path);
    defer mapped.deinit(io);
    var head = try inference.models.clef.Head.open(gpa, &mapped.document, mapped.view());
    defer head.deinit();
    const file = try std.json.parseFromSliceLeaky(File, arena, sequences_json, .{ .ignore_unknown_fields = true });
    const expected_bytes = try std.Io.Dir.cwd().readFileAlloc(io, head_json, arena, .limited(16 << 20));
    const expected = try std.json.parseFromSliceLeaky(HeadFile, arena, expected_bytes, .{ .ignore_unknown_fields = true });
    if (expected.results.len == 0) return error.NoRequests;
    var worst: check_stats.Worst = .{};
    for (expected.results) |result| {
        const entry = for (file.requests) |e| {
            if (std.mem.eql(u8, e.name, result.name)) break e;
        } else return error.ArtifactMismatch;
        const sequence = try fixtureSequence(arena, entry);
        const hidden = try readF32(arena, io, dumps, entry.name, "hidden");
        const lexical = try readF32(arena, io, dumps, entry.name, "lexical");
        const logits = try arena.alloc([]f32, sequence.fields.len);
        for (logits, sequence.fields) |*l, f| l.* = try arena.alloc(f32, f.options.len);
        const start = std.Io.Clock.awake.now(io);
        try head.forward(gpa, io, hidden, sequence, lexical, logits);
        const ms = start.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds();
        var request_worst: check_stats.Worst = .{};
        for (logits, result.logits) |got, want| for (got, want) |g, x| {
            request_worst.observe(@abs(g - x) / @max(1.0, @abs(x)));
        };
        worst.merge(request_worst);
        std.debug.print("{s}: {d} tokens, head {d} ms, worst logit difference {e:.2}\n", .{ entry.name, sequence.ids.len, ms, request_worst.shown() });
    }
    if (worst.count == 0 or !worst.within(logit_bound)) {
        std.debug.print("{d} logits compared, worst difference {e:.2}, bound {e}\n", .{ worst.count, worst.shown(), logit_bound });
        return error.HeadMismatch;
    }
    std.debug.print("the head's logits equal the reference's within {e} ({d} requests, {d} logits)\n", .{ logit_bound, expected.results.len, worst.count });
}

/// A fixture request's sequence from its recorded ids and spans.
pub fn fixtureSequence(arena: std.mem.Allocator, entry: Entry) !clef.Sequence {
    const fields = try arena.alloc(clef.Field, entry.questions.len);
    for (fields, entry.questions) |*f, q| {
        const options = try arena.alloc(clef.Span, q.option_spans.len);
        for (options, q.option_spans) |*o, s| o.* = .{ .start = s[0], .end = s[1] };
        const kinds = [_]clef.Kind{ .noul, .choice, .score };
        f.* = .{ .kind = kinds[q.type], .span = .{ .start = q.span[0], .end = q.span[1] }, .options = options };
    }
    return .{ .ids = entry.input_ids, .fields = fields, .state_offset = 0, .state_tokens = 0, .truncated = false };
}

fn readF32(arena: std.mem.Allocator, io: std.Io, dir: []const u8, name: []const u8, kind: []const u8) ![]f32 {
    const path = try arena.print("{s}/{s}.{s}.f32", .{ dir, name, kind });
    const bytes = try std.Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(1 << 30));
    const out = try arena.alloc(f32, bytes.len / 4);
    for (out, 0..) |*x, i| x.* = @bitCast(std.mem.readInt(u32, bytes[i * 4 ..][0..4], .little));
    return out;
}

/// Cloudflare's head on the rows of the Q6_K backbone nuclis first ran
/// (`scripts/clef-reference.py`): the oracle `run` holds probabilities to.
const head_fixture = @embedFile("src/models/fixtures/clef/head.json");
/// nuclis's own Metal logits for the catalogue's file (`run --pin`): what
/// `run` holds the logits to.
const pinned_fixture = @embedFile("src/models/fixtures/clef/logits.json");
/// Per question, the softmax against the oracle's: the Q4_K_M backbone and
/// the Q8_0 head moved them by at most 1.55e-2 (card_systemone, 2026-11-20).
const oracle_probability_bound = 2e-2;
fn checkRun(gpa: std.mem.Allocator, io: std.Io, arena: std.mem.Allocator, clef_path: []const u8, rest: []const [:0]const u8) !void {
    var backend: inference.decide.Backend = .cpu;
    var dump: ?[]const u8 = null;
    var pin: ?[]const u8 = null;
    var mmproj: ?[]const u8 = null;
    var only: ?[]const u8 = null;
    var max_tokens: usize = std.math.maxInt(usize);
    var i: usize = 0;
    while (i < rest.len) : (i += 1) {
        const flag = rest[i];
        if (i + 1 >= rest.len) return error.InvalidArguments;
        i += 1;
        if (std.mem.eql(u8, flag, "--max-tokens")) {
            max_tokens = try std.fmt.parseInt(usize, rest[i], 10);
        } else if (std.mem.eql(u8, flag, "--backend")) {
            backend = std.meta.stringToEnum(inference.decide.Backend, rest[i]) orelse return error.UnknownBackend;
        } else if (std.mem.eql(u8, flag, "--only")) {
            only = rest[i];
        } else if (std.mem.eql(u8, flag, "--pin")) pin = rest[i] else if (std.mem.eql(u8, flag, "--dump")) dump = rest[i] else if (std.mem.eql(u8, flag, "--mmproj")) {
            mmproj = rest[i];
        } else return error.InvalidArguments;
    }
    const load_start = std.Io.Clock.awake.now(io);
    var model = try inference.models.clef.Model.open(gpa, io, clef_path, mmproj, backend);
    defer model.deinit(io);
    std.debug.print("clef-flash loaded on the {s} in {d} ms\n", .{ @tagName(backend), load_start.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds() });
    const file = try std.json.parseFromSliceLeaky(File, arena, sequences_json, .{ .ignore_unknown_fields = true });
    const oracle = try std.json.parseFromSliceLeaky(HeadFile, arena, head_fixture, .{ .ignore_unknown_fields = true });
    const expected = try std.json.parseFromSliceLeaky(HeadFile, arena, pinned_fixture, .{ .ignore_unknown_fields = true });
    var pinned: std.ArrayList(HeadResult) = .empty;
    if (dump) |d| try std.Io.Dir.cwd().createDirPath(io, d);
    // The fixture is the Metal run's; the CPU reference's F32 cache and
    // kernels move the logits by about 1e-4.
    const bound: f64 = if (backend == .metal) 1e-3 else 2e-3;
    var worst: check_stats.Worst = .{};
    var worst_gap: check_stats.Worst = .{};
    var wrong: usize = 0;
    var ran: usize = 0;
    var unpinned: usize = 0;
    const h = model.hiddenSize();
    for (file.requests) |entry| {
        if (entry.media != null and mmproj == null) continue;
        if (entry.input_ids.len > max_tokens) continue;
        if (only) |name| if (!std.mem.eql(u8, name, entry.name)) continue;
        ran += 1;
        const sequence = try fixtureSequence(arena, entry);
        const hidden = try gpa.alloc(f32, sequence.ids.len * h);
        defer gpa.free(hidden);
        const start = std.Io.Clock.awake.now(io);
        const images = try fixtureImages(arena, &model, entry);
        var features: []f32 = &.{};
        var spans: []const inference.vision.Span = &.{};
        if (entry.media) |m| {
            var tokens: usize = 0;
            const starts = try arena.alloc(usize, images.len);
            var at = m.token_offset;
            for (images, starts, m.image_tokens) |img, *s, want| {
                if (img.grid.tokens() != want) {
                    std.debug.print("{s}: an image is {d} tokens, the reference's {d}\n", .{ entry.name, img.grid.tokens(), want });
                    return error.ImageTokenMismatch;
                }
                s.* = at + 1;
                at += want + 2;
                tokens += want;
            }
            features = try arena.alloc(f32, tokens * h);
            spans = try model.projectImages(arena, images, starts, features);
        }
        try model.hiddenRows(sequence.ids, spans, features, hidden);
        const backbone_ms = start.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds();
        const lexical = try gpa.alloc(f32, inference.models.clef.optionCount(sequence) * h);
        defer gpa.free(lexical);
        const scratch = try gpa.alloc(f32, h);
        defer gpa.free(scratch);
        try model.lexical(sequence, scratch, lexical);
        const logits = try arena.alloc([]f32, sequence.fields.len);
        for (logits, sequence.fields) |*l, f| l.* = try arena.alloc(f32, f.options.len);
        const head_start = std.Io.Clock.awake.now(io);
        try model.head.forward(gpa, io, hidden, sequence, lexical, logits);
        const head_ms = head_start.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds();
        if (dump) |d| {
            try writeF32(io, arena, d, entry.name, "hidden", hidden);
            try writeF32(io, arena, d, entry.name, "lexical", lexical);
        }
        var difference: ?f64 = null;
        var gap: f64 = 0;
        for (expected.results) |r| if (std.mem.eql(u8, r.name, entry.name)) {
            var d: check_stats.Worst = .{};
            for (logits, r.logits) |got, want| for (got, want) |g, x| {
                d.observe(@abs(g - x) / @max(1.0, @abs(x)));
            };
            difference = d.shown();
            worst.merge(d);
        };
        for (oracle.results) |r| if (std.mem.eql(u8, r.name, entry.name)) {
            var pg: check_stats.Worst = .{};
            for (logits, r.logits) |got, want| pg.observe(probabilityGap(got, want));
            gap = pg.shown();
            worst_gap.merge(pg);
        };
        if (pin != null) try pinned.append(arena, .{ .name = entry.name, .logits = try copyLogits(arena, logits) });
        if (difference == null) unpinned += 1;
        var picks: std.Io.Writer.Allocating = .init(arena);
        for (entry.questions, logits) |q, l| {
            const best = std.mem.indexOfMax(f32, l);
            const want = if (entry.expect.object.get(q.id)) |v| v.string else null;
            const ok = if (want) |w| std.mem.eql(u8, w, q.option_ids[best]) else true;
            if (!ok) wrong += 1;
            try picks.writer.print(" {s}={s}", .{ q.id, q.option_ids[best] });
            if (!ok) try picks.writer.print(" (expected {s})", .{want.?});
        }
        const tokens_per_s = @as(f64, @floatFromInt(sequence.ids.len)) * 1000 / @as(f64, @floatFromInt(@max(1, backbone_ms)));
        if (difference) |d|
            std.debug.print("{s}: {d} tokens, backbone {d} ms ({d:.0} tok/s), head {d} ms, logits {e:.2} from the pinned run, probabilities {e:.2} from the oracle;{s}\n", .{ entry.name, sequence.ids.len, backbone_ms, tokens_per_s, head_ms, d, gap, picks.written() })
        else
            std.debug.print("{s}: {d} tokens, backbone {d} ms ({d:.0} tok/s), head {d} ms, probabilities {e:.2} from the oracle;{s}\n", .{ entry.name, sequence.ids.len, backbone_ms, tokens_per_s, head_ms, gap, picks.written() });
    }
    std.debug.print("{d} requests run, {d} logits compared with the pinned run, {d} requests it does not pin; probabilities within {e:.2} of the oracle\n", .{ ran, worst.count, unpinned, worst_gap.shown() });
    if (ran == 0) {
        std.debug.print("no request selected (--only, --max-tokens, --mmproj)\n", .{});
        return error.NothingCompared;
    }
    if (wrong > 0) {
        std.debug.print("{d} sanity answers differ from the expected options\n", .{wrong});
        return error.SanityMismatch;
    }
    if (worst_gap.count == 0 or !worst_gap.within(oracle_probability_bound)) {
        std.debug.print("probabilities differ from the oracle's by {e:.2}, over {e}\n", .{ worst_gap.shown(), oracle_probability_bound });
        return error.ProbabilityMismatch;
    }
    if (pin) |path| {
        var out: std.Io.Writer.Allocating = .init(arena);
        try std.json.Stringify.value(.{ .repo = "ggml-org/Clef-Flash-GGUF", .revision = "c3d925454595d792d5333da8b35f3c09911f7568", .file = std.fs.path.basename(clef_path), .backend = @tagName(backend), .results = pinned.items }, .{ .whitespace = .indent_1 }, &out.writer);
        try out.writer.writeByte('\n');
        try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = out.written() });
        std.debug.print("pinned {d} requests' logits to {s}\n", .{ pinned.items.len, path });
        return;
    }
    if (worst.count == 0 or !worst.within(bound)) {
        std.debug.print("logits differ from the pinned run by {e:.2}, over {e}\n", .{ worst.shown(), bound });
        return error.LogitMismatch;
    }
    std.debug.print("every sanity answer is the expected option; logits within {e} of the pinned run, probabilities within {e} of the oracle\n", .{ bound, oracle_probability_bound });
}

fn copyLogits(arena: std.mem.Allocator, logits: []const []f32) ![]const []const f32 {
    const out = try arena.alloc([]const f32, logits.len);
    for (logits, out) |l, *o| o.* = try arena.dupe(f32, l);
    return out;
}

/// The largest difference between one question's softmax over `got` and
/// over `want`.
fn probabilityGap(got: []const f32, want: []const f32) f64 {
    var gm: f64 = -std.math.inf(f64);
    var wm: f64 = -std.math.inf(f64);
    for (got, want) |g, w| {
        gm = @max(gm, g);
        wm = @max(wm, w);
    }
    var gs: f64 = 0;
    var ws: f64 = 0;
    for (got, want) |g, w| {
        gs += @exp(@as(f64, g) - gm);
        ws += @exp(@as(f64, w) - wm);
    }
    var out: f64 = 0;
    for (got, want) |g, w| out = @max(out, @abs(@exp(@as(f64, g) - gm) / gs - @exp(@as(f64, w) - wm) / ws));
    return out;
}

/// A fixture request's images: solid colours of the recorded sizes.
fn fixtureImages(arena: std.mem.Allocator, model: *const inference.models.clef.Model, entry: Entry) ![]inference.models.clef.Image {
    const specs = if (entry.request.object.get("images")) |v| v.array.items else return &.{};
    const out = try arena.alloc(inference.models.clef.Image, specs.len);
    for (specs, out) |spec, *img| {
        const color = spec.object.get("color").?.array.items;
        const size = spec.object.get("size").?.array.items;
        const width: u32 = @intCast(size[0].integer);
        const height: u32 = @intCast(size[1].integer);
        const pixels = try arena.alloc(u8, @as(usize, width) * height * 3);
        for (0..@as(usize, width) * height) |p| for (0..3) |c| {
            pixels[p * 3 + c] = @intCast(color[c].integer);
        };
        img.* = .{ .pixels = .{ .width = width, .height = height, .pixels = pixels }, .grid = try model.imageGrid(.{ .width = width, .height = height }) };
    }
    return out;
}

fn writeF32(io: std.Io, arena: std.mem.Allocator, dir: []const u8, name: []const u8, kind: []const u8, values: []const f32) !void {
    const path = try arena.print("{s}/{s}.{s}.f32", .{ dir, name, kind });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = std.mem.sliceAsBytes(values) });
}
