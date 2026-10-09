//! `nuclis embed`: inputs into unit vectors through an embedding model
//! (`inference.embed`). Inputs come from the words on the command line, from
//! image files, and from inputs files (one JSON input per line), in the
//! order given; each gives one vector. The text report shows each vector's head and, for
//! several inputs, their cosines; `--json` writes the shared response
//! (`embedding/response.zig`). docs/models/embeddinggemma.md.
const std = @import("std");
const inference = @import("inference");
const style = @import("tui/style.zig");
const config = @import("config.zig");
const wire = @import("embedding/request.zig");
const response = @import("embedding/response.zig");
const files = @import("decision/request.zig");

const embed = inference.embed;

/// An inputs file's bytes, at most.
const max_file_bytes = 64 * 1024 * 1024;
/// The most inputs the text report crosses in a cosine matrix.
const max_matrix = 12;

const Source = union(enum) { text: []const u8, file: []const u8, image: []const u8 };

pub const Options = struct {
    sources: std.ArrayList(Source) = .empty,
    task: ?wire.Task = null,
    title: ?[]const u8 = null,
    dimensions: ?usize = null,
    truncate: bool = false,
    json: bool = false,
    model: ?[]const u8 = null,
    backend: ?embed.Backend = null,
    image_tokens: ?u32 = null,
};

/// Parses the words after `embed`; `arena` owns the lists. A word that is
/// not an option is an input's text; after `--`, every word is.
pub fn parseArgs(arena: std.mem.Allocator, args: []const []const u8, diag: *config.Diagnostic) !Options {
    var o: Options = .{};
    var i: usize = 0;
    var literal = false;
    while (i < args.len) : (i += 1) {
        const word = args[i];
        if (literal or !std.mem.startsWith(u8, word, "-") or word.len == 1) {
            try o.sources.append(arena, .{ .text = word });
            continue;
        }
        if (std.mem.eql(u8, word, "--")) {
            literal = true;
            continue;
        }
        if (std.mem.eql(u8, word, "--json") or std.mem.eql(u8, word, "--truncate")) {
            const flag = if (word[2] == 'j') &o.json else &o.truncate;
            if (flag.*) return error.DuplicateOption;
            flag.* = true;
            continue;
        }
        const takes_value = for ([_][]const u8{ "--task", "--title", "--dimensions", "--input-file", "--model", "--backend", "--image", "--image-tokens", "--audio" }) |known| {
            if (std.mem.eql(u8, word, known)) break true;
        } else false;
        if (!takes_value) {
            diag.set("{s} is not an option of embed (`nuclis embed --help`)", .{word});
            return error.UnknownOption;
        }
        i += 1;
        if (i == args.len) {
            diag.set("{s} needs a value", .{word});
            return error.MissingOptionValue;
        }
        const value = args[i];
        if (std.mem.eql(u8, word, "--input-file")) {
            try o.sources.append(arena, .{ .file = value });
        } else if (std.mem.eql(u8, word, "--image")) {
            try o.sources.append(arena, .{ .image = value });
        } else if (std.mem.eql(u8, word, "--image-tokens")) {
            if (o.image_tokens != null) return error.DuplicateOption;
            const n = std.fmt.parseInt(u32, value, 10) catch 0;
            if (std.mem.indexOfScalar(u32, &embed.image_budgets, n) == null) {
                diag.set("--image-tokens takes {s}, not {s}", .{ wire.image_budgets_text, value });
                return error.InvalidOptionValue;
            }
            o.image_tokens = n;
        } else if (std.mem.eql(u8, word, "--audio")) {
            diag.set("{s}: audio input is not supported yet; nuclis embeds text and images", .{value});
            return error.UnsupportedModality;
        } else if (std.mem.eql(u8, word, "--task")) {
            if (o.task != null) return error.DuplicateOption;
            o.task = std.meta.stringToEnum(wire.Task, value) orelse {
                diag.set("--task takes search_query, document, question_answering, fact_checking, code_retrieval, classification, clustering, or similarity, not {s}", .{value});
                return error.InvalidOptionValue;
            };
        } else if (std.mem.eql(u8, word, "--title")) {
            if (o.title != null) return error.DuplicateOption;
            o.title = value;
        } else if (std.mem.eql(u8, word, "--dimensions")) {
            if (o.dimensions != null) return error.DuplicateOption;
            o.dimensions = std.fmt.parseInt(usize, value, 10) catch {
                diag.set("--dimensions takes 768, 512, 256, or 128, not {s}", .{value});
                return error.InvalidOptionValue;
            };
        } else if (std.mem.eql(u8, word, "--model")) {
            if (o.model != null) return error.DuplicateOption;
            o.model = value;
        } else if (std.mem.eql(u8, word, "--backend")) {
            if (o.backend != null) return error.DuplicateOption;
            o.backend = std.meta.stringToEnum(embed.Backend, value) orelse {
                diag.set("--backend takes cpu or metal, not {s}", .{value});
                return error.InvalidOptionValue;
            };
        }
    }
    if (o.sources.items.len == 0) {
        diag.set("give the text to embed (each argument is one input: quote a sentence), --image <file>, or --input-file <file>", .{});
        return error.MissingInput;
    }
    return o;
}

fn buildRequest(arena: std.mem.Allocator, io: std.Io, o: Options, diag: *config.Diagnostic) !wire.Request {
    var inputs: std.ArrayList(wire.Input) = .empty;
    for (o.sources.items) |source| switch (source) {
        .text => |t| {
            const parts = try arena.alloc(wire.Part, 1);
            parts[0] = .{ .text = t };
            try inputs.append(arena, .{ .parts = parts, .label = try arena.print("input[{d}]", .{inputs.items.len}) });
        },
        .file => |path| {
            const bytes = try files.readBounded(arena, io, path, max_file_bytes, diag);
            try wire.inputsFromJsonLines(arena, bytes, if (std.mem.eql(u8, path, "-")) "stdin" else path, &inputs, diag);
        },
        .image => |path| {
            const parts = try arena.alloc(wire.Part, 1);
            parts[0] = .{ .image = try files.readBounded(arena, io, path, wire.max_image_bytes, diag) };
            try inputs.append(arena, .{ .parts = parts, .label = path });
        },
    };
    const request: wire.Request = .{
        .inputs = inputs.items,
        .task = o.task,
        .title = o.title,
        .dimensions = o.dimensions orelse embed.dimensions,
        .truncate = o.truncate,
        .image_tokens = o.image_tokens orelse embed.default_image_tokens,
    };
    try wire.check(request, diag);
    return request;
}

/// Embeds every input of `o` with the model at `path` (and its projector
/// `mmproj` for images, null when not pulled), and reports.
pub fn run(gpa: std.mem.Allocator, io: std.Io, path: []const u8, mmproj: ?[]const u8, identity_in: response.Identity, o: Options, out: *std.Io.Writer, sty: style.Style, diag: *config.Diagnostic) !void {
    var arena_state: std.heap.ArenaAllocator = .init(gpa);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const request = try buildRequest(arena, io, o, diag);
    var identity = identity_in;
    // A file nuclis did not pull has no recorded digest; its space needs one.
    if (identity.sha256.len == 0) identity.sha256 = try @import("embedding/catalog.zig").digest(arena, io, path);

    var timings: response.Timings = .{};
    var started = std.Io.Clock.awake.now(io);
    const backend = o.backend orelse if (inference.metal.enabled) embed.Backend.metal else .cpu;
    const images = request.hasImages();
    if (images and mmproj == null) {
        diag.set("{s}: images need the model's projector, which is not pulled (`nuclis model pull {s} --with mmproj`)", .{ identity.name, identity.name });
        return error.NoProjector;
    }
    const embedder = embed.Embedder.open(gpa, io, path, backend, .{ .projector = if (images) mmproj else null }) catch |err| {
        if (err == error.MetalNotEnabled)
            diag.set("this build has no Metal backend; run with --backend cpu", .{})
        else if (err == error.NotAnEmbeddingModel)
            diag.set("{s}: not an embedding model (its general.architecture is not {s})", .{ path, embed.architecture })
        else
            diag.set("{s}: not an embedding model nuclis can run ({s})", .{ path, @errorName(err) });
        return err;
    };
    defer embedder.deinit();
    timings.load_ns = lap(io, &started);

    const prepared = try arena.alloc(embed.Prepared, request.inputs.len);
    for (request.inputs, prepared) |input, *p| {
        const rendered = try wire.render(arena, request, input, diag);
        // Prepared with truncation always, so a refusal can give the count.
        p.* = embedder.prepare(arena, rendered, .{ .truncate = true, .image_tokens = request.image_tokens }) catch |err| {
            switch (err) {
                error.UnsupportedImageFormat, error.MalformedImage => diag.set("{s}: not an image nuclis can read (PNG, JPEG, HEIC, WebP, TIFF, GIF, BMP)", .{input.label}),
                error.ImageTooLarge => diag.set("{s}: the image is too large to decode", .{input.label}),
                else => {},
            }
            return err;
        };
        if (p.truncated() and !request.truncate) {
            diag.set("{s} is {d} tokens; the model reads at most {d} (--truncate cuts the end)", .{ input.label, p.length, embed.max_tokens });
            return error.InputTooLong;
        }
    }
    timings.tokenize_ns = lap(io, &started);

    const full = try arena.alloc([]f32, request.inputs.len);
    for (full) |*v| v.* = try arena.alloc(f32, embed.dimensions);
    try embedder.embedBatch(prepared, full);
    const vectors = try arena.alloc(response.Vector, request.inputs.len);
    for (vectors, full, prepared) |*v, values, p| v.* = .{
        .values = if (request.dimensions == embed.dimensions) values else try embed.truncate(values, request.dimensions),
        .tokens = p.tokens.len,
        .input_tokens = p.length,
    };
    timings.embed_ns = lap(io, &started);

    const body: response.Body = .{ .identity = identity, .request = request, .vectors = vectors, .timings = timings };
    if (o.json) return response.write(arena, out, body);
    try writeText(arena, out, sty, body);
}

/// Nanoseconds since `started`, which moves to now.
fn lap(io: std.Io, started: *std.Io.Timestamp) u64 {
    const now = std.Io.Clock.awake.now(io);
    defer started.* = now;
    return @intCast(started.durationTo(now).toNanoseconds());
}

/// Values shown per vector in the text report.
const head_values = 6;

fn writeText(arena: std.mem.Allocator, out: *std.Io.Writer, sty: style.Style, body: response.Body) !void {
    const off = sty.off();
    const n = body.vectors.len;
    try out.print("{s}{s}{s} {s}· {d} input{s} · {d} tokens · load {d:.1} s · tokenize {d:.0} ms · embed {d:.0} ms{s}\n", .{
        sty.on(.header),                    body.identity.name,                       off,
        sty.on(.dim),                       n,                                        if (n == 1) "" else "s",
        body.tokens(),                      response.ms(body.timings.load_ns) / 1000, response.ms(body.timings.tokenize_ns),
        response.ms(body.timings.embed_ns), off,
    });
    try out.print("{s}space{s} {s}{s}{s}", .{ sty.on(.label), off, sty.on(.code), try response.space(arena, body.identity, body.request.dimensions), off });
    if (body.request.task) |t| try out.print("  {s}task{s} {s}", .{ sty.on(.label), off, @tagName(t) });
    if (body.request.hasImages()) try out.print("  {s}image tokens{s} {d}", .{ sty.on(.label), off, body.request.image_tokens });
    try out.writeAll("\n\n");
    for (body.vectors, body.request.inputs, 0..) |v, input, i| {
        try out.print("{s}{d:>3}{s}  {s}{s}{s}\n", .{ sty.on(.number), i, off, sty.on(.code), try snippet(arena, input), off });
        try out.print("     {d} tokens · {d} values · norm {d:.4} · ", .{ v.tokens, v.values.len, @sqrt(response.cosine(v.values, v.values)) });
        for (v.values[0..@min(head_values, v.values.len)]) |x| try out.print("{s}{d: >7.4}{s} ", .{ sty.on(.number), x, off });
        try out.writeAll("…\n");
        if (v.truncated()) try out.print("     {s}cut from {d} tokens to the model's {d}{s}\n", .{ sty.on(.warning), v.input_tokens, v.tokens, off });
    }
    if (n == 2) {
        try out.print("\n{s}cosine{s} {s}{d:.4}{s}\n", .{ sty.on(.label), off, sty.on(.number), response.cosine(body.vectors[0].values, body.vectors[1].values), off });
    } else if (n > 2 and n <= max_matrix) {
        try out.print("\n{s}cosines{s}\n     ", .{ sty.on(.label), off });
        for (0..n) |j| try out.print("{s}{d:>7}{s}", .{ sty.on(.dim), j, off });
        try out.writeByte('\n');
        for (body.vectors, 0..) |a, i| {
            try out.print("{s}{d:>3}{s}  ", .{ sty.on(.number), i, off });
            for (body.vectors, 0..) |b, j| {
                if (i == j) try out.print("{s}{s:>7}{s}", .{ sty.on(.dim), "·", off }) else try out.print("{d:>7.4}", .{response.cosine(a.values, b.values)});
            }
            try out.writeByte('\n');
        }
    } else if (n > max_matrix) {
        try out.print("\n{s}cosines are crossed for up to {d} inputs; --json gives every vector{s}\n", .{ sty.on(.dim), max_matrix, off });
    }
}

/// An input as one short line: its first text, else its label.
fn snippet(arena: std.mem.Allocator, input: wire.Input) ![]const u8 {
    const text = for (input.parts) |part| switch (part) {
        .text => |t| break t,
        else => {},
    } else return input.label;
    const limit = 64;
    var line: std.ArrayList(u8) = .empty;
    var it = std.unicode.Utf8View.initUnchecked(text).iterator();
    var count: usize = 0;
    while (it.nextCodepointSlice()) |cp| : (count += 1) {
        if (count == limit) {
            try line.appendSlice(arena, "…");
            break;
        }
        try line.appendSlice(arena, if (cp.len == 1 and cp[0] < 0x20) " " else cp);
    }
    const shown = try std.fmt.allocPrint(arena, "\"{s}\"", .{line.items});
    return if (std.mem.startsWith(u8, input.label, "input[")) shown else arena.print("{s} {s}", .{ input.label, shown });
}

test "arguments: words are inputs, options take values, -- ends options" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diag: config.Diagnostic = .{};
    const o = try parseArgs(arena, &.{ "first", "--task", "document", "--title", "T", "--dimensions", "256", "second", "--input-file", "in.jsonl", "--json", "--", "--literal" }, &diag);
    try std.testing.expectEqual(@as(usize, 4), o.sources.items.len);
    try std.testing.expectEqualStrings("second", o.sources.items[1].text);
    try std.testing.expectEqualStrings("in.jsonl", o.sources.items[2].file);
    try std.testing.expectEqualStrings("--literal", o.sources.items[3].text);
    try std.testing.expectEqual(wire.Task.document, o.task.?);
    try std.testing.expectEqual(@as(usize, 256), o.dimensions.?);
    try std.testing.expect(o.json and !o.truncate);
    const pictures = try parseArgs(arena, &.{ "--image", "a.png", "caption", "--image-tokens", "560" }, &diag);
    try std.testing.expectEqualStrings("a.png", pictures.sources.items[0].image);
    try std.testing.expectEqualStrings("caption", pictures.sources.items[1].text);
    try std.testing.expectEqual(@as(u32, 560), pictures.image_tokens.?);
    const cases = .{
        .{ &[_][]const u8{"--json"}, error.MissingInput },
        .{ &[_][]const u8{ "x", "--task", "search" }, error.InvalidOptionValue },
        .{ &[_][]const u8{ "x", "--dimensions", "wide" }, error.InvalidOptionValue },
        .{ &[_][]const u8{ "x", "--backend", "gpu" }, error.InvalidOptionValue },
        .{ &[_][]const u8{ "x", "--wat" }, error.UnknownOption },
        .{ &[_][]const u8{ "x", "--truncate", "--truncate" }, error.DuplicateOption },
        .{ &[_][]const u8{ "x", "--model" }, error.MissingOptionValue },
        .{ &[_][]const u8{ "--audio", "a.wav" }, error.UnsupportedModality },
        .{ &[_][]const u8{ "--image", "a.png", "--image-tokens", "300" }, error.InvalidOptionValue },
        .{ &[_][]const u8{ "--image", "a.png", "--image-tokens", "70", "--image-tokens", "70" }, error.DuplicateOption },
    };
    inline for (cases) |case| try std.testing.expectError(case[1], parseArgs(arena, case[0], &diag));
}

test "the request: inputs in order, the width checked, a title only for documents" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var diag: config.Diagnostic = .{};
    const request = try buildRequest(arena, std.testing.io, try parseArgs(arena, &.{ "a", "b", "--dimensions", "128" }, &diag), &diag);
    try std.testing.expectEqual(@as(usize, 2), request.inputs.len);
    try std.testing.expectEqualStrings("input[1]", request.inputs[1].label);
    try std.testing.expectEqual(@as(usize, 128), request.dimensions);
    try std.testing.expectError(error.UnsupportedDimensions, buildRequest(arena, std.testing.io, try parseArgs(arena, &.{ "a", "--dimensions", "100" }, &diag), &diag));
    try std.testing.expectError(error.InvalidRequest, buildRequest(arena, std.testing.io, try parseArgs(arena, &.{ "a", "--title", "t" }, &diag), &diag));
}

test "the text report: a snippet per input, one cosine for a pair" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const a = [_]f32{ 1, 0 };
    const b = [_]f32{ 0.6, 0.8 };
    const long: [70]u8 = @splat('x');
    const inputs = [_]wire.Input{
        .{ .parts = &.{.{ .text = "first\nline" }}, .label = "input[0]" },
        .{ .parts = &.{.{ .text = &long }}, .label = "docs.jsonl:4" },
    };
    var out: std.Io.Writer.Allocating = .init(arena);
    try writeText(arena, &out.writer, .none, .{
        .identity = .{ .name = "e", .sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef" },
        .request = .{ .inputs = &inputs, .dimensions = 768 },
        .vectors = &.{ .{ .values = &a, .tokens = 3, .input_tokens = 3 }, .{ .values = &b, .tokens = 4, .input_tokens = 9 } },
        .timings = .{},
    });
    const text = out.written();
    try std.testing.expect(std.mem.indexOf(u8, text, "\"first line\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "docs.jsonl:4 \"" ++ long[0..64] ++ "…\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "cut from 9 tokens") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "cosine 0.6000") != null);
    try std.testing.expect(std.mem.indexOf(u8, text, "space e@0123456789ab/768") != null);
}
