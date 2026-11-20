//! One inspection snapshot feeds both output formats. It borrows the Document:
//! render it before freeing the parsed directory.
const std = @import("std");
const inference = @import("inference");
const style = @import("tui/style.zig");
const kit = @import("ui.zig");

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

    /// `inspect --model`: the header, the directory's facts, the encodings,
    /// and the next step.
    pub fn render(self: Snapshot, arena: std.mem.Allocator, out: *std.Io.Writer, json: bool, ui: kit.Ui) !void {
        if (json) {
            try std.json.Stringify.value(self, .{ .whitespace = .indent_2 }, out);
            return out.writeByte('\n');
        }
        const path = try kit.text.clean(arena, self.model_path);
        try kit.blocks.header(out, ui, std.fs.path.basename(path), std.fs.path.dirname(path) orelse "");
        try self.renderText(arena, out, ui, &.{});
        try out.writeByte('\n');
        try kit.blocks.result(out, ui, .ok, "directory and tensor ranges valid", "");
        try kit.blocks.hint(out, ui, "", try std.fmt.allocPrint(arena, "nuclis validate --model {s}", .{path}), " checks architecture support");
    }

    /// The directory's facts as one detail section after `lead` (a
    /// caller's own fields: `model inspect` names the file and digest),
    /// then the encodings table. Names from the file are cleaned.
    pub fn renderText(self: Snapshot, arena: std.mem.Allocator, out: *std.Io.Writer, ui: kit.Ui, lead: []const kit.blocks.Field) !void {
        const v = kit.values;
        var fields: std.ArrayList(kit.blocks.Field) = .empty;
        try fields.appendSlice(arena, lead);
        try fields.append(arena, .{ .key = "Model", .value = if (self.name) |n| try kit.text.clean(arena, n) else v.missing });
        try fields.append(arena, .{ .key = "Architecture", .value = if (self.architecture) |a| try kit.text.clean(arena, a) else v.missing, .role = .id });
        if (self.declared_block_count) |n| try fields.append(arena, .{ .key = "Blocks", .value = try std.fmt.allocPrint(arena, "{f}", .{v.grouped(n)}) });
        if (self.declared_context_length) |n| try fields.append(arena, .{ .key = "Context", .value = try std.fmt.allocPrint(arena, "{f} tokens", .{v.grouped(n)}) });
        if (self.embedding_length) |n| try fields.append(arena, .{ .key = "Embedding", .value = try std.fmt.allocPrint(arena, "{f}", .{v.grouped(n)}) });
        try fields.append(arena, .{ .key = "GGUF", .value = try std.fmt.allocPrint(arena, "v{d} · {f} · {f}", .{ self.gguf_version, v.count(self.metadata_count, "metadata key"), v.count(self.tensor_count, "tensor") }) });
        try fields.append(arena, .{ .key = "Size", .value = try std.fmt.allocPrint(arena, "{f} · directory {f} · data at byte {f}", .{ v.size(self.file_bytes), v.size(self.directory_bytes), v.grouped(self.data_offset) }) });
        try fields.append(arena, .{ .key = "Tensors", .value = try std.fmt.allocPrint(arena, "{f} elements · {f}", .{ v.grouped(self.stored_elements), v.size(self.tensor_bytes) }) });
        try kit.blocks.detail(out, ui, &.{.{ .fields = fields.items }});
        if (self.encodings.len == 0) return;
        try out.writeByte('\n');
        const columns = [_]kit.table.Column{
            .{ .title = "ENCODING", .truncate = false },
            .{ .title = "TENSORS", .alignment = .right },
            .{ .title = "SIZE", .alignment = .right },
        };
        const rows = try arena.alloc([]const kit.table.Cell, self.encodings.len);
        for (self.encodings, rows) |e, *row| row.* = try arena.dupe(kit.table.Cell, &.{
            .{ .text = e.name, .role = .id },
            .{ .text = try std.fmt.allocPrint(arena, "{d}", .{e.tensors}) },
            .{ .text = try std.fmt.allocPrint(arena, "{f}", .{v.size(e.bytes)}) },
        });
        try kit.table.render(out, ui, arena, 0, &columns, rows);
    }
};

/// A safetensors checkpoint (one file or an indexed shard set): the header
/// facts per shard and the dtype histogram over every tensor. Borrows the
/// checkpoint: render before `deinit`.
pub const SafetensorsSnapshot = struct {
    schema_version: u32 = 1,
    format: []const u8 = "safetensors",
    model_path: []const u8,
    /// The file or index the path resolved to.
    resolved: []const u8,
    tensor_count: usize,
    stored_elements: u64,
    tensor_bytes: u64,
    shards: []const Shard,
    dtypes: []const DtypeCount,
    /// The first shard's `__metadata__` (writers repeat it per shard).
    metadata: []const inference.safetensors.Metadata,

    pub const Shard = struct { file: []const u8, file_bytes: u64, header_bytes: u64, tensors: usize };
    pub const DtypeCount = struct { dtype: []const u8, tensors: u64 = 0, elements: u64 = 0, bytes: u64 = 0 };

    pub fn render(self: SafetensorsSnapshot, out: *std.Io.Writer, json: bool, sty: style.Style) !void {
        if (json) {
            try std.json.Stringify.value(self, .{ .whitespace = .indent_2 }, out);
            return out.writeByte('\n');
        }
        const label = sty.on(.label);
        const number = sty.on(.number);
        const off = sty.off();
        try out.print("{s}Checkpoint:{s} {s}", .{ label, off, sty.on(.code) });
        try std.json.Stringify.value(self.resolved, .{}, out);
        try out.print("{s}\n{s}Safetensors:{s} {s}{d}{s} shard(s), {s}{d}{s} tensors, {s}{d}{s} metadata keys\n", .{ off, label, off, number, self.shards.len, off, number, self.tensor_count, off, number, self.metadata.len, off });
        try out.print("{s}Stored tensor elements:{s} {s}{d}{s}; tensor bytes: {s}{d}{s}\n", .{ label, off, number, self.stored_elements, off, number, self.tensor_bytes, off });
        for (self.metadata) |m| {
            try out.print("{s}Metadata{s} ", .{ label, off });
            // Escaped: keys and values come from the file.
            try std.json.Stringify.value(m.key, .{}, out);
            try out.writeAll(" = ");
            try std.json.Stringify.value(m.value, .{}, out);
            try out.writeByte('\n');
        }
        try out.print("\n{s}Shard                                        Header bytes  Tensors          Bytes{s}\n", .{ sty.on(.header), off });
        for (self.shards) |shard| {
            try out.print("{s}{s: <44}{s} {s}{d: >12}{s}  {s}{d: >7}{s}  {s}{d: >13}{s}\n", .{ sty.on(.code), shard.file, off, number, shard.header_bytes, off, number, shard.tensors, off, number, shard.file_bytes, off });
        }
        try out.print("\n{s}Dtype      Tensors        Elements          Bytes{s}\n", .{ sty.on(.header), off });
        for (self.dtypes) |d| {
            try out.print("{s}{s: <8}{s}  {s}{d: >7}{s}  {s}{d: >14}{s}  {s}{d: >13}{s}\n", .{ sty.on(.code), d.dtype, off, number, d.tensors, off, number, d.elements, off, number, d.bytes, off });
        }
        try out.print("\n{s}Header and byte buffer validated. No text-generation family runs safetensors; `decide`, `speak`, and `transcribe` do.{s}\n", .{ sty.on(.dim), off });
    }
};

/// One tensor: its descriptor and its first values decoded to f32 (null for
/// a dtype the loader does not decode).
pub const TensorPeek = struct {
    name: []const u8,
    dtype: []const u8,
    shape: []const u64,
    elements: u64,
    bytes: u64,
    first: ?[]const f32,

    pub const shown = 8;

    pub fn render(self: TensorPeek, out: *std.Io.Writer, json: bool, sty: style.Style) !void {
        if (json) {
            try std.json.Stringify.value(self, .{ .whitespace = .indent_2 }, out);
            return out.writeByte('\n');
        }
        const label = sty.on(.label);
        const off = sty.off();
        try out.print("{s}Tensor:{s} {s}", .{ label, off, sty.on(.code) });
        try std.json.Stringify.value(self.name, .{}, out);
        try out.print("{s}\n{s}Dtype:{s} {s}; {s}shape:{s} {any}; {d} elements, {d} bytes\n", .{ off, label, off, self.dtype, label, off, self.shape, self.elements, self.bytes });
        if (self.first) |values| {
            try out.print("{s}First {d}:{s}", .{ label, values.len, off });
            for (values) |v| try out.print(" {d}", .{v});
            try out.writeByte('\n');
        } else try out.print("{s}(values not decoded: the loader decodes F32, F16, and BF16){s}\n", .{ sty.on(.dim), off });
    }
};

pub fn tensorPeek(arena: std.mem.Allocator, ref: inference.safetensors.Ref) !TensorPeek {
    const t = ref.tensor;
    const count: usize = @intCast(@min(t.elements, TensorPeek.shown));
    const values = try arena.alloc(f32, count);
    const first: ?[]const f32 = if (ref.decode(0, values)) values else |err| switch (err) {
        error.UnsupportedDtype => null,
        else => return err,
    };
    return .{ .name = t.name, .dtype = t.dtype.name(), .shape = t.shape, .elements = t.elements, .bytes = t.bytes, .first = first };
}

/// Allocates the shard and dtype tables in `arena`.
pub fn safetensorsSnapshot(arena: std.mem.Allocator, checkpoint: *const inference.safetensors.Checkpoint, path: []const u8) !SafetensorsSnapshot {
    const Dtype = inference.safetensors.Dtype;
    var counts: [std.enums.values(Dtype).len]SafetensorsSnapshot.DtypeCount = @splat(.{ .dtype = "" });
    const shards = try arena.alloc(SafetensorsSnapshot.Shard, checkpoint.shards.len);
    var elements: u64 = 0;
    var bytes: u64 = 0;
    for (checkpoint.shards, shards) |*shard, *summary| {
        const doc = &shard.document;
        summary.* = .{ .file = std.fs.path.basename(shard.path), .file_bytes = doc.file_bytes, .header_bytes = doc.header_bytes, .tensors = doc.tensors.len };
        for (doc.tensors) |t| {
            elements = try std.math.add(u64, elements, t.elements);
            bytes = try std.math.add(u64, bytes, t.bytes);
            const c = &counts[@backingInt(t.dtype)];
            c.dtype = t.dtype.name();
            c.tensors += 1;
            c.elements = try std.math.add(u64, c.elements, t.elements);
            c.bytes = try std.math.add(u64, c.bytes, t.bytes);
        }
    }
    var dtypes: std.ArrayList(SafetensorsSnapshot.DtypeCount) = .empty;
    for (counts) |c| if (c.tensors > 0) try dtypes.append(arena, c);
    return .{
        .model_path = path,
        .resolved = checkpoint.path,
        .tensor_count = checkpoint.tensorCount(),
        .stored_elements = elements,
        .tensor_bytes = bytes,
        .shards = shards,
        .dtypes = dtypes.items,
        .metadata = if (checkpoint.shards.len > 0) checkpoint.shards[0].document.metadata else &.{},
    };
}

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
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer out.deinit();
    try value.render(arena, &out.writer, true, .plain);
    const parsed = try std.json.parseFromSlice(std.json.Value, std.testing.allocator, out.written(), .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings(value.model_path, parsed.value.object.get("model_path").?.string);
    try std.testing.expectEqual(@as(i64, 2), parsed.value.object.get("schema_version").?.integer);
    try std.testing.expect(parsed.value.object.get("inference_available") == null);
    var text_out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer text_out.deinit();
    try value.render(arena, &text_out.writer, false, .plain);
    try std.testing.expect(std.mem.findScalar(u8, text_out.written(), 0x1b) == null);
    try std.testing.expectEqualStrings(
        \\◆ a\b"c.gguf
        \\
        \\Model         model?[31m
        \\Architecture  test
        \\GGUF          v3 · 0 metadata keys · 0 tensors
        \\Size          64 B · directory 32 B · data at byte 32
        \\Tensors       0 elements · 0 B
        \\
        \\✓ directory and tensor ranges valid
        \\→ nuclis validate --model a\b"c.gguf checks architecture support
        \\
    , text_out.written());
}
