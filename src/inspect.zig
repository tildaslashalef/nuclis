//! One inspection snapshot feeds both output formats. It borrows the Document:
//! render it before freeing the parsed directory.
const std = @import("std");
const inference = @import("inference");
const style = @import("tui/style.zig");

pub const EncodingCount = struct {
    id: u32,
    name: []const u8,
    tensors: u64 = 0,
    elements: u64 = 0,
    bytes: u64 = 0,
};

pub const Snapshot = struct {
    schema_version: u32 = 2,
    model_path: []const u8,
    gguf_version: u32,
    architecture: ?[]const u8,
    name: ?[]const u8,
    declared_block_count: ?u64 = null,
    declared_context_length: ?u64 = null,
    embedding_length: ?u64 = null,
    file_bytes: u64,
    directory_bytes: u64,
    data_offset: u64,
    metadata_count: usize,
    tensor_count: usize,
    stored_elements: u64,
    tensor_bytes: u64,
    encodings: []const EncodingCount,

    pub fn render(self: Snapshot, out: *std.Io.Writer, json: bool, sty: style.Style) !void {
        if (json) {
            try std.json.Stringify.value(self, .{ .whitespace = .indent_2 }, out);
            return out.writeByte('\n');
        }
        try self.renderText(out, sty);
        try out.print("\n{s}Directory and tensor ranges validated. Run `validate` to check architecture support.{s}\n", .{ sty.on(.dim), sty.off() });
    }

    /// The human-readable body without the closing advice, so a caller
    /// with its own verdict (`model inspect`) can append it.
    pub fn renderText(self: Snapshot, out: *std.Io.Writer, sty: style.Style) !void {
        const label = sty.on(.label);
        const number = sty.on(.number);
        const off = sty.off();
        try out.print("{s}Model:{s} {s}", .{ label, off, sty.on(.code) });
        // JSON string escaping also keeps file-controlled names from emitting
        // terminal control sequences in the human-readable output.
        try std.json.Stringify.value(self.name orelse "(unnamed)", .{}, out);
        try out.print("{s}\n{s}Architecture:{s} {s}", .{ off, label, off, sty.on(.code) });
        try std.json.Stringify.value(self.architecture orelse "(unspecified)", .{}, out);
        try out.writeAll(off);
        if (self.declared_block_count) |count| try out.print("\n{s}Declared blocks:{s} {s}{d}{s}", .{ label, off, number, count, off });
        if (self.declared_context_length) |count| try out.print("\n{s}Declared context:{s} {s}{d}{s} tokens", .{ label, off, number, count, off });
        if (self.embedding_length) |count| try out.print("\n{s}Embedding length:{s} {s}{d}{s}", .{ label, off, number, count, off });
        try out.print("\n{s}GGUF:{s} v{d}, {s}{d}{s} metadata keys, {s}{d}{s} tensors\n", .{ label, off, self.gguf_version, number, self.metadata_count, off, number, self.tensor_count, off });
        try out.print("{s}File:{s} {s}{d}{s} bytes; directory: {s}{d}{s} bytes; data offset: {s}{d}{s}\n", .{ label, off, number, self.file_bytes, off, number, self.directory_bytes, off, number, self.data_offset, off });
        try out.print("{s}Stored tensor elements:{s} {s}{d}{s}; tensor bytes: {s}{d}{s}\n\n", .{ label, off, number, self.stored_elements, off, number, self.tensor_bytes, off });
        try out.print("{s}Encoding      Tensors       Bytes{s}\n", .{ sty.on(.header), off });
        for (self.encodings) |entry| {
            try out.print("{s}{s: <12}{s}  {s}{d: >7}{s}  {s}{d: >12}{s}\n", .{ sty.on(.code), entry.name, off, number, entry.tensors, off, number, entry.bytes, off });
        }
    }
};

/// histogram is caller-provided scratch storage. Every parsed encoding must fit;
/// Document rejects unknown layouts, so its IDs are bounded by our layout table.
pub fn snapshot(doc: inference.gguf.Document, path: []const u8, histogram: []EncodingCount) !Snapshot {
    var count: usize = 0;
    var elements: u64 = 0;
    var bytes: u64 = 0;
    for (doc.tensors) |tensor| {
        elements = try std.math.add(u64, elements, tensor.elements);
        bytes = try std.math.add(u64, bytes, tensor.bytes);
        var index: usize = 0;
        while (index < count and histogram[index].id != tensor.encoding_id) : (index += 1) {}
        if (index == count) {
            if (count == histogram.len) return error.InsufficientHistogramStorage;
            histogram[count] = .{
                .id = tensor.encoding_id,
                .name = inference.encoding.layout(tensor.encoding_id).?.name,
            };
            count += 1;
        }
        histogram[index].tensors += 1;
        histogram[index].elements = try std.math.add(u64, histogram[index].elements, tensor.elements);
        histogram[index].bytes = try std.math.add(u64, histogram[index].bytes, tensor.bytes);
    }
    std.mem.sort(EncodingCount, histogram[0..count], {}, struct {
        fn less(_: void, a: EncodingCount, b: EncodingCount) bool {
            return a.id < b.id;
        }
    }.less);
    return .{
        .model_path = path,
        .gguf_version = doc.version,
        .architecture = doc.string("general.architecture"),
        .name = doc.string("general.name"),
        .declared_block_count = architectureInteger(doc, "block_count"),
        .declared_context_length = architectureInteger(doc, "context_length"),
        .embedding_length = architectureInteger(doc, "embedding_length"),
        .file_bytes = doc.file_bytes,
        .directory_bytes = doc.directory_bytes,
        .data_offset = doc.data_offset,
        .metadata_count = doc.metadata.len,
        .tensor_count = doc.tensors.len,
        .stored_elements = elements,
        .tensor_bytes = bytes,
        .encodings = histogram[0..count],
    };
}

// These are common GGUF key suffixes, not Qwen equations. "Declared" values
// intentionally make no claim about runtime support or auxiliary layer semantics.
fn architectureInteger(doc: inference.gguf.Document, suffix: []const u8) ?u64 {
    const architecture = doc.string("general.architecture") orelse return null;
    var buffer: [256]u8 = undefined;
    const key = std.fmt.bufPrint(&buffer, "{s}.{s}", .{ architecture, suffix }) catch return null;
    return switch (doc.get(key) orelse return null) {
        .unsigned => |value| value,
        else => null,
    };
}

test "JSON output preserves paths and clearly reports inspection-only capability" {
    const value: Snapshot = .{
        .model_path = "a\\b\"c.gguf",
        .gguf_version = 3,
        .architecture = "test",
        .name = "model\x1b[31m",
        .file_bytes = 64,
        .directory_bytes = 32,
        .data_offset = 32,
        .metadata_count = 0,
        .tensor_count = 0,
        .stored_elements = 0,
        .tensor_bytes = 0,
        .encodings = &.{},
    };
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    try value.render(&out.writer, true, .none);
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, out.written(), .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings(value.model_path, parsed.value.object.get("model_path").?.string);
    try std.testing.expectEqual(@as(i64, 2), parsed.value.object.get("schema_version").?.integer);
    try std.testing.expect(parsed.value.object.get("inference_available") == null);
    var text_out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer text_out.deinit();
    try value.render(&text_out.writer, false, .none);
    try std.testing.expect(std.mem.findScalar(u8, text_out.written(), 0x1b) == null);
    try std.testing.expect(std.mem.startsWith(u8, text_out.written(), "Model: \"model\\u001b[31m\"\nArchitecture: \"test\"\nGGUF: v3, 0 metadata keys, 0 tensors\n"));
}
