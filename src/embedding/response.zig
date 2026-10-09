//! The embedding response: one unit vector per input, its token counts, and
//! the space the vectors live in, written as the JSON `nuclis embed --json`
//! prints. Knows no HTTP.
const std = @import("std");
const catalog = @import("../catalog.zig");
const request_mod = @import("request.zig");

pub const schema_version = 1;

/// The model as the caller named it, where its file came from, and the
/// file's SHA-256 (64 hex), which names the vectors' space.
pub const Identity = struct {
    name: []const u8,
    repo: ?[]const u8 = null,
    revision: ?[]const u8 = null,
    sha256: []const u8,
};

/// `<checkpoint>@<sha256[0:12]>/<width>`: vectors are comparable only
/// within one space (one file, one width). The checkpoint is the
/// catalogue's name for the file when it pins it, else the caller's.
pub fn space(arena: std.mem.Allocator, identity: Identity, dimensions: usize) ![]const u8 {
    const checkpoint = if (catalog.embeddingByDigest(identity.sha256)) |e| e.name else identity.name;
    return arena.print("{s}@{s}/{d}", .{ checkpoint, identity.sha256[0..12], dimensions });
}

pub const Vector = struct {
    /// Unit length, `dimensions` values.
    values: []const f32,
    /// What the model read, framing included.
    tokens: usize,
    /// The input's whole length; above `tokens` only when it was cut.
    input_tokens: usize,

    pub fn truncated(self: Vector) bool {
        return self.input_tokens != self.tokens;
    }
};

pub const Timings = struct { load_ns: u64 = 0, tokenize_ns: u64 = 0, embed_ns: u64 = 0 };

pub const Body = struct {
    identity: Identity,
    request: request_mod.Request,
    vectors: []const Vector,
    timings: Timings,

    pub fn tokens(self: Body) usize {
        var total: usize = 0;
        for (self.vectors) |v| total += v.tokens;
        return total;
    }
};

/// The `--json` body, then a newline. Each vector's values sit on one line,
/// as the shortest decimal that reads back to the same f32.
pub fn write(arena: std.mem.Allocator, out: *std.Io.Writer, body: Body) !void {
    var s: std.json.Stringify = .{ .writer = out, .options = .{ .whitespace = .indent_2 } };
    try s.beginObject();
    try s.objectField("schema_version");
    try s.write(schema_version);
    try s.objectField("model");
    try s.write(body.identity.name);
    try s.objectField("repo");
    try s.write(body.identity.repo);
    try s.objectField("revision");
    try s.write(body.identity.revision);
    try s.objectField("space");
    try s.write(try space(arena, body.identity, body.request.dimensions));
    try s.objectField("dimensions");
    try s.write(body.request.dimensions);
    try s.objectField("task");
    try s.write(if (body.request.task) |t| @tagName(t) else null);
    try s.objectField("tokens");
    try s.write(body.tokens());
    try s.objectField("timings_ms");
    try s.beginObject();
    inline for (.{ .{ "load", body.timings.load_ns }, .{ "tokenize", body.timings.tokenize_ns }, .{ "embed", body.timings.embed_ns } }) |field| {
        try s.objectField(field[0]);
        try s.write(@round(ms(field[1]) * 10) / 10);
    }
    try s.endObject();
    try s.objectField("vectors");
    try s.beginArray();
    for (body.vectors, body.request.inputs, 0..) |v, input, i| {
        try s.beginObject();
        try s.objectField("index");
        try s.write(i);
        try s.objectField("input");
        try s.write(input.label);
        try s.objectField("tokens");
        try s.write(v.tokens);
        try s.objectField("input_tokens");
        try s.write(v.input_tokens);
        try s.objectField("truncated");
        try s.write(v.truncated());
        try s.objectField("values");
        try s.beginWriteRaw();
        try out.writeByte('[');
        for (v.values, 0..) |x, j| try out.print("{s}{}", .{ if (j > 0) "," else "", x });
        try out.writeByte(']');
        s.endWriteRaw();
        try s.endObject();
    }
    try s.endArray();
    try s.endObject();
    try out.writeByte('\n');
}

/// OpenAI's embeddings response on one line: `data` in input order,
/// `model`, `usage`, then the `nuclis` object (the space, the width, the
/// task, each input's tokens, the indexes of the inputs that were cut, and
/// the timings). `base64` writes each vector's little-endian f32 bytes.
pub fn writeOpenAI(out: *std.Io.Writer, arena: std.mem.Allocator, body: Body, encoding: request_mod.Encoding) !void {
    var s: std.json.Stringify = .{ .writer = out };
    try s.beginObject();
    try s.objectField("object");
    try s.write("list");
    try s.objectField("data");
    try s.beginArray();
    for (body.vectors, 0..) |v, i| {
        try s.beginObject();
        try s.objectField("object");
        try s.write("embedding");
        try s.objectField("index");
        try s.write(i);
        try s.objectField("embedding");
        try s.beginWriteRaw();
        switch (encoding) {
            .float => {
                try out.writeByte('[');
                for (v.values, 0..) |x, j| try out.print("{s}{}", .{ if (j > 0) "," else "", x });
                try out.writeByte(']');
            },
            .base64 => {
                var bytes = try arena.alloc(u8, v.values.len * 4);
                for (v.values, 0..) |x, j| std.mem.writeInt(u32, bytes[j * 4 ..][0..4], @bitCast(x), .little);
                try out.writeByte('"');
                try std.base64.standard.Encoder.encodeWriter(out, bytes);
                try out.writeByte('"');
            },
        }
        s.endWriteRaw();
        try s.endObject();
    }
    try s.endArray();
    try s.objectField("model");
    try s.write(body.identity.name);
    try s.objectField("usage");
    try s.beginObject();
    try s.objectField("prompt_tokens");
    try s.write(body.tokens());
    try s.objectField("total_tokens");
    try s.write(body.tokens());
    try s.endObject();
    try s.objectField("nuclis");
    try s.beginObject();
    try s.objectField("space");
    try s.write(try space(arena, body.identity, body.request.dimensions));
    try s.objectField("dimensions");
    try s.write(body.request.dimensions);
    try s.objectField("task");
    try s.write(if (body.request.task) |t| @tagName(t) else null);
    try s.objectField("tokens");
    try s.beginArray();
    for (body.vectors) |v| try s.write(v.tokens);
    try s.endArray();
    try s.objectField("truncated");
    try s.beginArray();
    for (body.vectors, 0..) |v, i| if (v.truncated()) try s.write(i);
    try s.endArray();
    try s.objectField("timings_ms");
    try s.beginObject();
    inline for (.{ .{ "load", body.timings.load_ns }, .{ "tokenize", body.timings.tokenize_ns }, .{ "embed", body.timings.embed_ns } }) |field| {
        try s.objectField(field[0]);
        try s.write(@round(ms(field[1]) * 10) / 10);
    }
    try s.endObject();
    try s.endObject();
    try s.endObject();
    try out.writeByte('\n');
}

pub fn ms(ns: u64) f64 {
    return @as(f64, @floatFromInt(ns)) / std.time.ns_per_ms;
}

/// The cosine of two unit vectors of one space: their dot product.
pub fn cosine(a: []const f32, b: []const f32) f64 {
    var dot: f64 = 0;
    for (a, b) |x, y| dot += @as(f64, x) * y;
    return dot;
}

test "OpenAI's body: float and base64 carry the same f32 bits" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const values = [_]f32{ 0.6, -0.8 };
    const inputs = [_]request_mod.Input{ .{ .parts = &.{.{ .text = "x" }}, .label = "input[0]" }, .{ .parts = &.{.{ .text = "y" }}, .label = "input[1]" } };
    const body: Body = .{
        .identity = .{ .name = "e", .sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef" },
        .request = .{ .inputs = &inputs, .dimensions = 128 },
        .vectors = &.{ .{ .values = &values, .tokens = 3, .input_tokens = 3 }, .{ .values = &values, .tokens = 8192, .input_tokens = 9000 } },
        .timings = .{},
    };
    var float: std.Io.Writer.Allocating = .init(arena);
    try writeOpenAI(&float.writer, arena, body, .float);
    var b64: std.Io.Writer.Allocating = .init(arena);
    try writeOpenAI(&b64.writer, arena, body, .base64);
    const f = try std.json.parseFromSliceLeaky(std.json.Value, arena, float.written(), .{});
    const b = try std.json.parseFromSliceLeaky(std.json.Value, arena, b64.written(), .{});
    try std.testing.expectEqualStrings("list", f.object.get("object").?.string);
    const first = f.object.get("data").?.array.items[1].object;
    try std.testing.expectEqualStrings("embedding", first.get("object").?.string);
    try std.testing.expectEqual(@as(i64, 1), first.get("index").?.integer);
    try std.testing.expectEqual(@as(i64, 8195), f.object.get("usage").?.object.get("total_tokens").?.integer);
    const nuclis = f.object.get("nuclis").?.object;
    try std.testing.expectEqual(@as(i64, 1), nuclis.get("truncated").?.array.items[0].integer);
    try std.testing.expectEqualStrings("e@0123456789ab/128", nuclis.get("space").?.string);
    const encoded = b.object.get("data").?.array.items[0].object.get("embedding").?.string;
    var decoded: [8]u8 = undefined;
    try std.base64.standard.Decoder.decode(&decoded, encoded);
    for (values, 0..) |x, j| try std.testing.expectEqual(@as(u32, @bitCast(x)), std.mem.readInt(u32, decoded[j * 4 ..][0..4], .little));
    const numbers = first.get("embedding").?.array.items;
    try std.testing.expectEqual(values[1], @as(f32, @floatCast(numbers[1].float)));
}

test "the space names the catalogue checkpoint by digest, else the caller's name" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const pinned = catalog.findEmbedding("embeddinggemma-2").?;
    try std.testing.expectEqualStrings("embeddinggemma-2@6f1bd4ac6c5d/256", try space(arena, .{ .name = "my-entry", .sha256 = pinned.sha256 }, 256));
    try std.testing.expectEqualStrings("bf16@f315cbbb30dd/768", try space(arena, .{ .name = "bf16", .sha256 = "f315cbbb30dd487e44d501c8902abe88808755e43753a96beed1964f0a48aa4f" }, 768));
}

test "the JSON body: one line of values per vector, which read back exactly" {
    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const values = [_]f32{ 0.6, -0.8 };
    const inputs = [_]request_mod.Input{.{ .parts = &.{.{ .text = "x" }}, .label = "input[0]" }};
    var out: std.Io.Writer.Allocating = .init(arena);
    try write(arena, &out.writer, .{
        .identity = .{ .name = "e", .sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef" },
        .request = .{ .inputs = &inputs, .task = .search_query, .dimensions = 128 },
        .vectors = &.{.{ .values = &values, .tokens = 4, .input_tokens = 9 }},
        .timings = .{ .embed_ns = 1_250_000 },
    });
    try std.testing.expect(std.mem.indexOf(u8, out.written(), "\"values\": [0.6,-0.8]") != null);
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, arena, out.written(), .{});
    try std.testing.expectEqualStrings("e@0123456789ab/128", parsed.object.get("space").?.string);
    try std.testing.expectEqualStrings("search_query", parsed.object.get("task").?.string);
    const vector = parsed.object.get("vectors").?.array.items[0].object;
    try std.testing.expect(vector.get("truncated").?.bool);
    try std.testing.expectEqual(@as(f32, -0.8), @as(f32, @floatCast(vector.get("values").?.array.items[1].float)));
    try std.testing.expectEqual(@as(f64, 1.3), parsed.object.get("timings_ms").?.object.get("embed").?.float);
}
