//! One inspection snapshot feeds both output formats. It borrows the Document:
//! render it before freeing the parsed directory.
const std = @import("std");
const inference = @import("inference");
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
    /// Every metadata key, with `--metadata`.
    metadata: ?[]const MetadataRow = null,
    /// Every tensor's descriptor, in file order, with `--tensors`.
    tensors: ?[]const TensorRow = null,

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
        if (self.metadata) |m| try renderMetadata(arena, out, ui, m);
        if (self.tensors) |t| try renderTensors(arena, out, ui, t);
    }
};

/// What `--metadata`, `--tensors`, and `--tensor <name>` add to a GGUF
/// inspection.
pub const Detail = struct {
    metadata: bool = false,
    tensors: bool = false,
    tensor: ?[]const u8 = null,
};

pub const Scalar = union(enum) {
    unsigned: u64,
    signed: i64,
    /// A `float32` value at its own precision, so it prints as stored.
    float32: f32,
    float: f64,
    boolean: bool,
    string: []const u8,
};

/// A metadata value: a scalar, or an array's element type and count with
/// its values when the parser retained them (small arrays; vocabulary
/// arrays stay lazy and show their count only, `token_type` its counts by type).
pub const MetadataValue = union(enum) {
    scalar: Scalar,
    array: struct { element_type: []const u8, count: u64, values: ?[]const Scalar, token_types: ?[]const TypeCount = null },
};

/// How many tokens of `tokenizer.ggml.token_type` carry one type.
pub const TypeCount = struct { type: []const u8, count: u64 };

pub const MetadataRow = struct {
    key: []const u8,
    type: []const u8,
    value: MetadataValue,
};

/// A tensor's descriptor; `offset` is from the start of the file.
pub const TensorRow = struct {
    name: []const u8,
    encoding: []const u8,
    shape: []const u64,
    elements: u64,
    bytes: u64,
    offset: u64,
};

/// Copies every metadata key into `arena`, so the rows outlive `doc`.
/// `directory`, the file's bytes up to its tensor data, lets the lazy
/// `tokenizer.ggml.token_type` array be counted by type.
pub fn metadataRows(arena: std.mem.Allocator, doc: inference.gguf.Document, directory: ?[]const u8) ![]const MetadataRow {
    const rows = try arena.alloc(MetadataRow, doc.metadata.len);
    for (doc.metadata, rows) |m, *row| row.* = .{
        .key = try arena.dupe(u8, m.key),
        .type = @tagName(m.kind),
        .value = switch (m.value) {
            .array => |a| .{ .array = .{
                .element_type = @tagName(a.element_type),
                .count = a.count,
                .values = if (a.values) |values| blk: {
                    const out = try arena.alloc(Scalar, values.len);
                    for (values, out) |v, *o| o.* = try scalar(arena, v);
                    break :blk out;
                } else null,
                .token_types = if (a.values == null and std.mem.eql(u8, m.key, "tokenizer.ggml.token_type")) if (directory) |bytes| try tokenTypes(arena, a, bytes) else null else null,
            } },
            .float => |x| .{ .scalar = if (m.kind == .float32) .{ .float32 = @floatCast(x) } else .{ .float = x } },
            else => .{ .scalar = try scalar(arena, m.value) },
        },
    };
    return rows;
}

/// Counts by type, in type order; a value outside the GGUF token types is
/// counted as `other`. Null when the array cannot be read from `directory`.
fn tokenTypes(arena: std.mem.Allocator, array: inference.gguf.Array, directory: []const u8) !?[]const TypeCount {
    const names = [_][]const u8{ "other", "normal", "unknown", "control", "user_defined", "unused", "byte" };
    var counts: [names.len]u64 = @splat(0);
    var reader = inference.gguf.ArrayReader.init(array, directory, .{}) catch return null;
    while (reader.nextInt32() catch return null) |value| counts[if (value >= 1 and value < names.len) @intCast(value) else 0] += 1;
    var out: std.ArrayList(TypeCount) = .empty;
    for (names[1..], counts[1..]) |name, count| if (count > 0) try out.append(arena, .{ .type = name, .count = count });
    if (counts[0] > 0) try out.append(arena, .{ .type = names[0], .count = counts[0] });
    return out.items;
}

fn scalar(arena: std.mem.Allocator, v: inference.gguf.Value) !Scalar {
    return switch (v) {
        .unsigned => |x| .{ .unsigned = x },
        .signed => |x| .{ .signed = x },
        .float => |x| .{ .float = x },
        .boolean => |x| .{ .boolean = x },
        .string => |x| .{ .string = try arena.dupe(u8, x) },
        .array => error.InvalidMetadataType,
    };
}

/// Copies every tensor's descriptor into `arena`, in file order.
pub fn tensorRows(arena: std.mem.Allocator, doc: inference.gguf.Document) ![]const TensorRow {
    const rows = try arena.alloc(TensorRow, doc.tensors.len);
    for (doc.tensors, rows) |t, *row| row.* = .{
        .name = try arena.dupe(u8, t.name),
        .encoding = inference.encoding.layout(t.encoding_id).?.name,
        .shape = try arena.dupe(u64, t.dimensions),
        .elements = t.elements,
        .bytes = t.bytes,
        .offset = try std.math.add(u64, doc.data_offset, t.offset),
    };
    return rows;
}

pub fn findTensor(doc: inference.gguf.Document, name: []const u8) ?*const inference.gguf.Tensor {
    for (doc.tensors) |*t| if (std.mem.eql(u8, t.name, name)) return t;
    return null;
}

/// The bytes of a GGUF tensor's first row (its first dimension's span),
/// what `ggufPeek` decodes; at most `peek_row_bytes`.
pub fn firstRowBytes(tensor: *const inference.gguf.Tensor) u64 {
    const columns = if (tensor.dimensions.len > 0) tensor.dimensions[0] else 1;
    const rows = if (columns == 0) 1 else @max(1, tensor.elements / columns);
    return @min(tensor.bytes / rows, peek_row_bytes);
}
pub const peek_row_bytes = 1 << 20;

/// A GGUF tensor's descriptor and the first values of `row` (its first
/// row's bytes, `firstRowBytes` long) decoded to f32; null when nuclis has
/// no decoder for the encoding or the row is cut by `peek_row_bytes`.
pub fn ggufPeek(arena: std.mem.Allocator, doc: inference.gguf.Document, tensor: *const inference.gguf.Tensor, row: []const u8) !TensorPeek {
    const columns: usize = @intCast(if (tensor.dimensions.len > 0) tensor.dimensions[0] else 1);
    const decoded = try arena.alloc(f32, columns);
    const first: ?[]const f32 = if (inference.quant.row(tensor.encoding_id, row, decoded)) decoded[0..@min(columns, TensorPeek.shown)] else |_| null;
    return .{
        .name = try arena.dupe(u8, tensor.name),
        .dtype = inference.encoding.layout(tensor.encoding_id).?.name,
        .shape = try arena.dupe(u64, tensor.dimensions),
        .elements = tensor.elements,
        .bytes = tensor.bytes,
        .offset = try std.math.add(u64, doc.data_offset, tensor.offset),
        .first = first,
    };
}

/// A string value on one line: cleaned, cut at `shown_string` bytes with
/// the full length marked.
const shown_string = 96;

fn scalarText(arena: std.mem.Allocator, s: Scalar) ![]const u8 {
    return switch (s) {
        .unsigned => |x| std.fmt.allocPrint(arena, "{d}", .{x}),
        .signed => |x| std.fmt.allocPrint(arena, "{d}", .{x}),
        .float32 => |x| std.fmt.allocPrint(arena, "{d}", .{x}),
        .float => |x| std.fmt.allocPrint(arena, "{d}", .{x}),
        .boolean => |x| if (x) "true" else "false",
        .string => |x| if (x.len <= shown_string)
            kit.text.clean(arena, x)
        else
            std.fmt.allocPrint(arena, "{s}… ({d} bytes)", .{ try kit.text.clean(arena, x[0..shown_string]), x.len }),
    };
}

fn renderMetadata(arena: std.mem.Allocator, out: *std.Io.Writer, ui: kit.Ui, rows: []const MetadataRow) !void {
    try out.writeByte('\n');
    const columns = [_]kit.table.Column{
        .{ .title = "KEY", .truncate = false },
        .{ .title = "TYPE", .truncate = false },
        .{ .title = "VALUE", .truncate = false },
    };
    const cells = try arena.alloc([]const kit.table.Cell, rows.len);
    for (rows, cells) |r, *row| {
        var value: std.ArrayList(u8) = .empty;
        var kind: []const u8 = r.type;
        switch (r.value) {
            .scalar => |s| try value.appendSlice(arena, try scalarText(arena, s)),
            .array => |a| {
                kind = try std.fmt.allocPrint(arena, "array<{s}>", .{a.element_type});
                if (a.values) |values| {
                    try value.append(arena, '[');
                    for (values, 0..) |v, i| try value.print(arena, "{s}{s}", .{ if (i > 0) ", " else "", try scalarText(arena, v) });
                    try value.append(arena, ']');
                } else {
                    try value.print(arena, "{d} items", .{a.count});
                    if (a.token_types) |types| for (types, 0..) |t, i| try value.print(arena, "{s}{d} {s}", .{ if (i == 0) ": " else ", ", t.count, t.type });
                }
            },
        }
        row.* = try arena.dupe(kit.table.Cell, &.{
            .{ .text = try kit.text.clean(arena, r.key), .role = .id },
            .{ .text = kind, .role = .muted },
            .{ .text = value.items },
        });
    }
    try kit.table.render(out, ui, arena, 0, &columns, cells);
}

fn renderTensors(arena: std.mem.Allocator, out: *std.Io.Writer, ui: kit.Ui, rows: []const TensorRow) !void {
    const v = kit.values;
    try out.writeByte('\n');
    const columns = [_]kit.table.Column{
        .{ .title = "TENSOR", .truncate = false },
        .{ .title = "ENCODING", .truncate = false },
        .{ .title = "SHAPE", .truncate = false },
        .{ .title = "SIZE", .alignment = .right },
        .{ .title = "OFFSET", .alignment = .right },
    };
    const cells = try arena.alloc([]const kit.table.Cell, rows.len);
    for (rows, cells) |r, *row| row.* = try arena.dupe(kit.table.Cell, &.{
        .{ .text = try kit.text.clean(arena, r.name), .role = .id },
        .{ .text = r.encoding },
        .{ .text = try shapeText(arena, r.shape) },
        .{ .text = try std.fmt.allocPrint(arena, "{f}", .{v.size(r.bytes)}) },
        .{ .text = try std.fmt.allocPrint(arena, "{d}", .{r.offset}), .role = .muted },
    });
    try kit.table.render(out, ui, arena, 0, &columns, cells);
}

fn shapeText(arena: std.mem.Allocator, shape: []const u64) ![]const u8 {
    var text: std.ArrayList(u8) = .empty;
    for (shape, 0..) |d, i| try text.print(arena, "{s}{d}", .{ if (i > 0) " × " else "", d });
    return text.items;
}

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

    /// The checkpoint's facts and metadata as one detail section, then the
    /// shard and dtype tables. Names and metadata come from the file and
    /// are cleaned.
    pub fn render(self: SafetensorsSnapshot, arena: std.mem.Allocator, out: *std.Io.Writer, json: bool, ui: kit.Ui) !void {
        if (json) {
            try std.json.Stringify.value(self, .{ .whitespace = .indent_2 }, out);
            return out.writeByte('\n');
        }
        const resolved = try kit.text.clean(arena, self.resolved);
        try kit.blocks.header(out, ui, std.fs.path.basename(resolved), std.fs.path.dirname(resolved) orelse "");
        try self.renderText(arena, out, ui, &.{});
        try out.writeByte('\n');
        try kit.blocks.result(out, ui, .ok, "header and byte buffer valid", "");
        try kit.blocks.hint(out, ui, "no text-generation family runs safetensors; ", "decide, speak, transcribe", " do");
    }

    /// The facts after `lead`, then the shard and dtype tables.
    pub fn renderText(self: SafetensorsSnapshot, arena: std.mem.Allocator, out: *std.Io.Writer, ui: kit.Ui, lead: []const kit.blocks.Field) !void {
        const v = kit.values;
        var fields: std.ArrayList(kit.blocks.Field) = .empty;
        try fields.appendSlice(arena, lead);
        try fields.append(arena, .{ .key = "Safetensors", .value = try std.fmt.allocPrint(arena, "{f} · {f} · {f}", .{ v.count(self.shards.len, "shard"), v.count(self.tensor_count, "tensor"), v.count(self.metadata.len, "metadata key") }) });
        try fields.append(arena, .{ .key = "Tensors", .value = try std.fmt.allocPrint(arena, "{f} elements · {f}", .{ v.grouped(self.stored_elements), v.size(self.tensor_bytes) }) });
        for (self.metadata) |m| try fields.append(arena, .{ .key = try kit.text.clean(arena, m.key), .value = try kit.text.clean(arena, m.value), .role = .muted });
        try kit.blocks.detail(out, ui, &.{.{ .fields = fields.items }});
        try out.writeByte('\n');
        const shard_rows = try arena.alloc([]const kit.table.Cell, self.shards.len);
        for (self.shards, shard_rows) |shard, *row| row.* = try arena.dupe(kit.table.Cell, &.{
            .{ .text = try kit.text.clean(arena, shard.file), .role = .id },
            .{ .text = try std.fmt.allocPrint(arena, "{f}", .{v.size(shard.header_bytes)}) },
            .{ .text = try std.fmt.allocPrint(arena, "{d}", .{shard.tensors}) },
            .{ .text = try std.fmt.allocPrint(arena, "{f}", .{v.size(shard.file_bytes)}) },
        });
        try kit.table.render(out, ui, arena, 0, &.{ .{ .title = "SHARD" }, .{ .title = "HEADER", .alignment = .right, .drop = 1 }, .{ .title = "TENSORS", .alignment = .right }, .{ .title = "SIZE", .alignment = .right } }, shard_rows);
        try out.writeByte('\n');
        const dtype_rows = try arena.alloc([]const kit.table.Cell, self.dtypes.len);
        for (self.dtypes, dtype_rows) |d, *row| row.* = try arena.dupe(kit.table.Cell, &.{
            .{ .text = d.dtype, .role = .id },
            .{ .text = try std.fmt.allocPrint(arena, "{d}", .{d.tensors}) },
            .{ .text = try std.fmt.allocPrint(arena, "{f}", .{v.grouped(d.elements)}) },
            .{ .text = try std.fmt.allocPrint(arena, "{f}", .{v.size(d.bytes)}) },
        });
        try kit.table.render(out, ui, arena, 0, &.{ .{ .title = "DTYPE", .truncate = false }, .{ .title = "TENSORS", .alignment = .right }, .{ .title = "ELEMENTS", .alignment = .right, .drop = 1 }, .{ .title = "SIZE", .alignment = .right } }, dtype_rows);
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
    /// A GGUF tensor's first byte in the file; null for safetensors.
    offset: ?u64 = null,
    first: ?[]const f32,

    pub const shown = 8;

    pub fn render(self: TensorPeek, arena: std.mem.Allocator, out: *std.Io.Writer, json: bool, ui: kit.Ui) !void {
        if (json) {
            try std.json.Stringify.value(self, .{ .whitespace = .indent_2 }, out);
            return out.writeByte('\n');
        }
        const v = kit.values;
        const shape = try shapeText(arena, self.shape);
        var first: std.ArrayList(u8) = .empty;
        if (self.first) |values| {
            for (values, 0..) |x, i| try first.print(arena, "{s}{d}", .{ if (i > 0) "  " else "", x });
        }
        try kit.blocks.header(out, ui, try kit.text.clean(arena, self.name), self.dtype);
        var fields: std.ArrayList(kit.blocks.Field) = .empty;
        try fields.append(arena, .{ .key = "Shape", .value = shape });
        try fields.append(arena, .{ .key = "Elements", .value = try std.fmt.allocPrint(arena, "{f} · {f}", .{ v.grouped(self.elements), v.size(self.bytes) }) });
        if (self.offset) |o| try fields.append(arena, .{ .key = "Offset", .value = try std.fmt.allocPrint(arena, "byte {f} of the file", .{v.grouped(o)}), .role = .muted });
        try fields.append(arena, if (self.first) |values|
            .{ .key = try std.fmt.allocPrint(arena, "First {d}", .{values.len}), .value = first.items }
        else
            .{ .key = "Values", .value = "not decoded: nuclis has no decoder for this encoding", .role = .muted });
        try kit.blocks.detail(out, ui, &.{.{ .fields = fields.items }});
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
    const shards = try arena.alloc(ShardHeader, checkpoint.shards.len);
    for (checkpoint.shards, shards) |*shard, *s| s.* = .{ .file = std.fs.path.basename(shard.path), .document = &shard.document };
    return safetensorsHeaders(arena, shards, path, checkpoint.path);
}

/// One shard's parsed header and the name it is shown by.
pub const ShardHeader = struct { file: []const u8, document: *const inference.safetensors.Document };

/// The snapshot of a set of parsed headers, local or read remotely; the
/// snapshot borrows the documents' strings.
pub fn safetensorsHeaders(arena: std.mem.Allocator, headers: []const ShardHeader, path: []const u8, resolved: []const u8) !SafetensorsSnapshot {
    const Dtype = inference.safetensors.Dtype;
    var counts: [std.enums.values(Dtype).len]SafetensorsSnapshot.DtypeCount = @splat(.{ .dtype = "" });
    const shards = try arena.alloc(SafetensorsSnapshot.Shard, headers.len);
    var elements: u64 = 0;
    var bytes: u64 = 0;
    var tensor_count: usize = 0;
    for (headers, shards) |header, *summary| {
        const doc = header.document;
        tensor_count += doc.tensors.len;
        summary.* = .{ .file = header.file, .file_bytes = doc.file_bytes, .header_bytes = doc.header_bytes, .tensors = doc.tensors.len };
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
        .resolved = resolved,
        .tensor_count = tensor_count,
        .stored_elements = elements,
        .tensor_bytes = bytes,
        .shards = shards,
        .dtypes = dtypes.items,
        .metadata = if (headers.len > 0) headers[0].document.metadata else &.{},
    };
}

/// One `config.json` value shown by `inspect`: a scalar as written, an
/// array of scalars inline when short, anything else by its size.
pub const ConfigRow = struct { key: []const u8, value: []const u8 };

/// The most rows `configRows` returns; the rest are counted, not shown.
pub const max_config_rows = 64;

/// A model configuration's scalars, in file order, with one level of
/// nesting flattened (`text_config.hidden_size`). Fails on text that is not
/// a JSON object. Every string lives in `arena`.
pub fn configRows(arena: std.mem.Allocator, text: []const u8) !struct { rows: []const ConfigRow, omitted: usize } {
    const root = std.json.parseFromSliceLeaky(std.json.Value, arena, text, .{ .max_value_len = text.len }) catch return error.InvalidConfig;
    if (root != .object) return error.InvalidConfig;
    var rows: std.ArrayList(ConfigRow) = .empty;
    var omitted: usize = 0;
    var it = root.object.iterator();
    while (it.next()) |entry| {
        if (entry.value_ptr.* == .object) {
            var inner = entry.value_ptr.object.iterator();
            while (inner.next()) |sub| {
                // Two levels down is a sub-model's own detail; counted only.
                if (sub.value_ptr.* == .object) {
                    omitted += 1;
                    continue;
                }
                if (rows.items.len == max_config_rows) {
                    omitted += 1;
                    continue;
                }
                try rows.append(arena, .{ .key = try std.fmt.allocPrint(arena, "{s}.{s}", .{ entry.key_ptr.*, sub.key_ptr.* }), .value = try configValue(arena, sub.value_ptr.*) });
            }
            continue;
        }
        if (rows.items.len == max_config_rows) {
            omitted += 1;
            continue;
        }
        try rows.append(arena, .{ .key = entry.key_ptr.*, .value = try configValue(arena, entry.value_ptr.*) });
    }
    return .{ .rows = rows.items, .omitted = omitted };
}

fn configValue(arena: std.mem.Allocator, value: std.json.Value) ![]const u8 {
    return switch (value) {
        .null => "null",
        .bool => |b| if (b) "true" else "false",
        .integer => |n| std.fmt.allocPrint(arena, "{d}", .{n}),
        .float => |f| std.fmt.allocPrint(arena, "{d}", .{f}),
        .number_string, .string => |t| t,
        .array => |list| blk: {
            const scalars = for (list.items) |item| {
                if (item == .array or item == .object) break false;
            } else true;
            if (!scalars or list.items.len > 8) break :blk std.fmt.allocPrint(arena, "[{d} items]", .{list.items.len});
            var text: std.ArrayList(u8) = .empty;
            try text.append(arena, '[');
            for (list.items, 0..) |item, i| {
                if (i > 0) try text.appendSlice(arena, ", ");
                try text.appendSlice(arena, try configValue(arena, item));
            }
            try text.append(arena, ']');
            break :blk text.items;
        },
        .object => |o| std.fmt.allocPrint(arena, "{{{d} keys}}", .{o.count()}),
    };
}

test "config rows: scalars in order, one level flattened, the rest counted" {
    var arena = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const r = try configRows(a,
        \\{"architectures":["Qwen3ForCausalLM"],"hidden_size":4096,"rope_theta":1000000.0,
        \\"tie_word_embeddings":false,"layer_types":[1,2,3,4,5,6,7,8,9],
        \\"text_config":{"num_hidden_layers":36,"rope_scaling":{"type":"yarn"}},"torch_dtype":"bfloat16"}
    );
    try std.testing.expectEqual(@as(usize, 1), r.omitted);
    const want = [_][2][]const u8{
        .{ "architectures", "[Qwen3ForCausalLM]" }, .{ "hidden_size", "4096" },      .{ "rope_theta", "1000000" },
        .{ "tie_word_embeddings", "false" },        .{ "layer_types", "[9 items]" }, .{ "text_config.num_hidden_layers", "36" },
        .{ "torch_dtype", "bfloat16" },
    };
    try std.testing.expectEqual(want.len, r.rows.len);
    for (want, r.rows) |w, row| {
        try std.testing.expectEqualStrings(w[0], row.key);
        try std.testing.expectEqualStrings(w[1], row.value);
    }
    try std.testing.expectError(error.InvalidConfig, configRows(a, "[1, 2]"));
    try std.testing.expectError(error.InvalidConfig, configRows(a, "{"));
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

test "a lazy token_type array is counted by type, an unknown value as other" {
    var arena_state = std.heap.ArenaAllocator.init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    var directory: [4 + 6 * 4]u8 = undefined;
    for ([_]i32{ 1, 3, 1, 9, 1, 6 }, 0..) |value, i| std.mem.writeInt(i32, directory[4 + 4 * i ..][0..4], value, .little);
    const array: inference.gguf.Array = .{ .element_type = .int32, .count = 6, .file_offset = 4 };
    const counts = (try tokenTypes(arena, array, &directory)).?;
    try std.testing.expectEqual(@as(usize, 4), counts.len);
    for (counts, [_][]const u8{ "normal", "control", "byte", "other" }, [_]u64{ 3, 1, 1, 1 }) |c, name, n| {
        try std.testing.expectEqualStrings(name, c.type);
        try std.testing.expectEqual(n, c.count);
    }
    try std.testing.expect(try tokenTypes(arena, array, directory[0..8]) == null);
}
