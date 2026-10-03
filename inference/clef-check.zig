//! Explicit check of clef-flash against the oracle's fixtures
//! (scripts/clef-reference.py). `sequences CLEF_DIR` rebuilds every
//! request's model input and compares the ids and the head's spans with
//! `sequences.json`, tokenizing with the backbone GGUF's vocabulary (only
//! its directory is read, not the weights).
//! `head CLEF_DIR DUMPS HEAD_JSON` runs nuclis's head on the dumped inputs
//! (`DUMPS/<name>.hidden.f32`, `.lexical.f32`) and compares its logits with
//! the reference head's on the same inputs. `run CLEF_DIR` is the whole
//! model on a backend: every text request through the backbone and the
//! head, its logits against `head.json` (the reference head fed the Metal
//! backbone's dumps) and its answers against the sanity set's; `--dump`
//! writes the head's inputs for the script, `--max-tokens` skips longer
//! requests (the CPU backbone runs about a token a second), `--only` runs
//! one; the image requests run when `--mmproj` names the projector. The backbone defaults to the catalogue's GGUF beside
//! CLEF_DIR's repository (`<models>/bartowski/...`).
const std = @import("std");
const inference = @import("inference");
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
        std.debug.print("usage: clef-check sequences CLEF_DIR | head CLEF_DIR DUMPS HEAD_JSON | run CLEF_DIR [--backbone GGUF] [--backend cpu|metal] [--mmproj GGUF] [--dump DIR] [--max-tokens N] [--only NAME]\n", .{});
        return error.InvalidArguments;
    }
    var mapped = try inference.weights.Mapped.open(gpa, io, try backboneBeside(arena, args[2]));
    defer mapped.deinit(io);
    var vocab = try inference.vocabulary.load(gpa, mapped.document, mapped.mapping.memory[0..@intCast(mapped.document.directory_bytes)], .{});
    defer vocab.deinit();
    var tokenizer = try Encoder.init(gpa, &vocab);
    defer tokenizer.deinit();
    const file = try std.json.parseFromSliceLeaky(File, arena, sequences_json, .{ .ignore_unknown_fields = true });
    const frame = try clef.Frame.init(arena, &tokenizer);
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
    std.debug.print("every sequence equals the reference's\n", .{});
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

/// Per logit, the difference over max(1, |logit|): F32 summation order
/// differs between `cpu.dense` and PyTorch, nothing else.
const logit_bound = 1e-4;

const HeadResult = struct { name: []const u8, logits: []const []const f32 };
const HeadFile = struct { results: []const HeadResult };

fn checkHead(gpa: std.mem.Allocator, io: std.Io, arena: std.mem.Allocator, clef_dir: []const u8, dumps: []const u8, head_json: []const u8) !void {
    var head = try inference.models.clef.Head.open(gpa, io, clef_dir);
    defer head.deinit();
    const file = try std.json.parseFromSliceLeaky(File, arena, sequences_json, .{ .ignore_unknown_fields = true });
    const expected_bytes = try std.Io.Dir.cwd().readFileAlloc(io, head_json, arena, .limited(16 << 20));
    const expected = try std.json.parseFromSliceLeaky(HeadFile, arena, expected_bytes, .{ .ignore_unknown_fields = true });
    var worst: f64 = 0;
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
        var request_worst: f64 = 0;
        for (logits, result.logits) |got, want| for (got, want) |g, x| {
            request_worst = @max(request_worst, @abs(g - x) / @max(1.0, @abs(x)));
        };
        worst = @max(worst, request_worst);
        std.debug.print("{s}: {d} tokens, head {d} ms, worst logit difference {e:.2}\n", .{ entry.name, sequence.ids.len, ms, request_worst });
    }
    if (worst > logit_bound) return error.HeadMismatch;
    std.debug.print("the head's logits equal the reference's within {e}\n", .{logit_bound});
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

const head_fixture = @embedFile("src/models/fixtures/clef/head.json");
const default_backbone = "bartowski/Cloudflare_clef-flash-GGUF/Cloudflare_clef-flash-Q6_K.gguf";

fn checkRun(gpa: std.mem.Allocator, io: std.Io, arena: std.mem.Allocator, clef_dir: []const u8, rest: []const [:0]const u8) !void {
    var backbone: ?[]const u8 = null;
    var backend: inference.decide.Backend = .cpu;
    var dump: ?[]const u8 = null;
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
        } else if (std.mem.eql(u8, flag, "--backbone")) backbone = rest[i] else if (std.mem.eql(u8, flag, "--backend")) {
            backend = std.meta.stringToEnum(inference.decide.Backend, rest[i]) orelse return error.UnknownBackend;
        } else if (std.mem.eql(u8, flag, "--only")) {
            only = rest[i];
        } else if (std.mem.eql(u8, flag, "--dump")) dump = rest[i] else if (std.mem.eql(u8, flag, "--mmproj")) {
            mmproj = rest[i];
        } else return error.InvalidArguments;
    }
    const backbone_path = backbone orelse try backboneBeside(arena, clef_dir);
    const load_start = std.Io.Clock.awake.now(io);
    var model = try inference.models.clef.Model.open(gpa, io, clef_dir, backbone_path, mmproj, backend);
    defer model.deinit(io);
    std.debug.print("clef-flash loaded on the {s} in {d} ms\n", .{ @tagName(backend), load_start.durationTo(std.Io.Clock.awake.now(io)).toMilliseconds() });
    const file = try std.json.parseFromSliceLeaky(File, arena, sequences_json, .{ .ignore_unknown_fields = true });
    const expected = std.json.parseFromSliceLeaky(HeadFile, arena, head_fixture, .{ .ignore_unknown_fields = true }) catch HeadFile{ .results = &.{} };
    if (dump) |d| try std.Io.Dir.cwd().createDirPath(io, d);
    // The fixture is the Metal run's; the CPU reference's F32 cache and
    // kernels move the logits by about 1e-4 (8.5e-5 on card_invoice).
    const bound: f64 = if (backend == .metal) 1e-3 else 2e-3;
    var worst: f64 = 0;
    var wrong: usize = 0;
    const h = model.hiddenSize();
    for (file.requests) |entry| {
        if (entry.media != null and mmproj == null) continue;
        if (entry.input_ids.len > max_tokens) continue;
        if (only) |name| if (!std.mem.eql(u8, name, entry.name)) continue;
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
        for (expected.results) |r| if (std.mem.eql(u8, r.name, entry.name)) {
            var d: f64 = 0;
            for (logits, r.logits) |got, want| for (got, want) |g, x| {
                d = @max(d, @abs(g - x) / @max(1.0, @abs(x)));
            };
            difference = d;
            worst = @max(worst, d);
        };
        var picks: std.Io.Writer.Allocating = .init(arena);
        for (entry.questions, logits) |q, l| {
            const best = std.mem.indexOfMax(f32, l);
            const want = if (entry.expect.object.get(q.id)) |v| v.string else null;
            const ok = if (want) |w| std.mem.eql(u8, w, q.option_ids[best]) else true;
            if (!ok) wrong += 1;
            try picks.writer.print(" {s}={s}{s}", .{ q.id, q.option_ids[best], if (ok) "" else " (expected " ++ "other)" });
        }
        const tokens_per_s = @as(f64, @floatFromInt(sequence.ids.len)) * 1000 / @as(f64, @floatFromInt(@max(1, backbone_ms)));
        if (difference) |d|
            std.debug.print("{s}: {d} tokens, backbone {d} ms ({d:.0} tok/s), head {d} ms, logits {e:.2} from the fixture;{s}\n", .{ entry.name, sequence.ids.len, backbone_ms, tokens_per_s, head_ms, d, picks.written() })
        else
            std.debug.print("{s}: {d} tokens, backbone {d} ms ({d:.0} tok/s), head {d} ms;{s}\n", .{ entry.name, sequence.ids.len, backbone_ms, tokens_per_s, head_ms, picks.written() });
    }
    if (wrong > 0) {
        std.debug.print("{d} sanity answers differ from the expected options\n", .{wrong});
        return error.SanityMismatch;
    }
    if (worst > bound) {
        std.debug.print("logits differ from the fixture by {e:.2}, over {e}\n", .{ worst, bound });
        return error.LogitMismatch;
    }
    std.debug.print("every sanity answer is the expected option; logits within {e} of the fixture\n", .{bound});
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

/// The catalogue's backbone for a head directory `<models>/Cloudflare/clef-flash`.
fn backboneBeside(arena: std.mem.Allocator, clef_dir: []const u8) ![]const u8 {
    const models_dir = std.fs.path.dirname(std.fs.path.dirname(clef_dir) orelse ".") orelse ".";
    return std.fs.path.join(arena, &.{ models_dir, default_backbone });
}

fn writeF32(io: std.Io, arena: std.mem.Allocator, dir: []const u8, name: []const u8, kind: []const u8, values: []const f32) !void {
    const path = try arena.print("{s}/{s}.{s}.f32", .{ dir, name, kind });
    try std.Io.Dir.cwd().writeFile(io, .{ .sub_path = path, .data = std.mem.sliceAsBytes(values) });
}
