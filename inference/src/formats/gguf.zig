//! Bounded GGUF v3 (little-endian) directory reader.
//!
//! parse() consumes only the header, metadata, and tensor descriptors. The caller
//! supplies a Reader and file size, so tests use bytes and production uses a file.
//! The returned Document owns an arena: names and scalar metadata survive reader
//! reuse. Large metadata arrays retain descriptors, not millions of allocations.
//! No tensor weights are read. Their complete byte ranges are validated instead.
const std = @import("std");
const encoding = @import("../tensor/encoding.zig");
const Allocator = std.mem.Allocator;

pub const Limits = struct {
    directory_bytes: u64 = 64 * 1024 * 1024,
    metadata_count: u64 = 16_384,
    tensor_count: u64 = 100_000,
    string_bytes: u64 = 1024 * 1024,
    array_items: u64 = 4_000_000,
};

pub const Type = enum(u32) {
    uint8 = 0,
    int8 = 1,
    uint16 = 2,
    int16 = 3,
    uint32 = 4,
    int32 = 5,
    float32 = 6,
    boolean = 7,
    string = 8,
    array = 9,
    uint64 = 10,
    int64 = 11,
    float64 = 12,
};

/// Numeric and boolean arrays of at most this many elements keep their
/// values in the Document; larger ones keep a descriptor. Sized for
/// per-layer arrays (Gemma 4's 48-layer patterns) with room for the
/// families in the roadmap; vocabulary arrays are millions of elements.
pub const retained_array_items = 64;

pub const Array = struct {
    element_type: Type,
    count: u64,
    file_offset: u64,
    /// Small numeric arrays (up to `retained_array_items`) are retained for
    /// architecture settings such as RoPE sections and per-layer patterns.
    /// Vocabulary-sized arrays remain lazy.
    values: ?[]const Value = null,
};

/// Bounded, allocation-free access to lazy arrays in a borrowed directory image.
/// `directory` starts at file offset zero and excludes tensor data. Returned
/// strings borrow it, not the Document. Only the two tokenizer array types are
/// supported here; extending access does not change inspection's lazy storage.
pub const ArrayReader = struct {
    bytes: []const u8,
    remaining: u64,
    kind: Type,
    string_bytes: u64,

    pub fn init(array: Array, directory: []const u8, limits: Limits) Error!ArrayReader {
        if (directory.len > limits.directory_bytes or array.count > limits.array_items)
            return error.LimitExceeded;
        if (array.element_type != .string and array.element_type != .int32)
            return error.InvalidMetadataType;
        if (array.file_offset > directory.len) return error.EndOfStream;
        const bytes = directory[@intCast(array.file_offset)..];
        const minimum: u64 = if (array.element_type == .string) 8 else 4;
        if (array.count > bytes.len / minimum) return error.EndOfStream;
        return .{ .bytes = bytes, .remaining = array.count, .kind = array.element_type, .string_bytes = limits.string_bytes };
    }

    pub fn nextString(self: *ArrayReader) Error!?[]const u8 {
        if (self.kind != .string) return error.InvalidMetadataType;
        if (self.remaining == 0) return null;
        if (self.bytes.len < 8) return error.EndOfStream;
        const len = std.mem.readInt(u64, self.bytes[0..8], .little);
        if (len > self.string_bytes) return error.LimitExceeded;
        if (len > self.bytes.len - 8) return error.EndOfStream;
        const end = 8 + @as(usize, @intCast(len));
        const value = self.bytes[8..end];
        self.bytes = self.bytes[end..];
        self.remaining -= 1;
        return value;
    }

    pub fn nextInt32(self: *ArrayReader) Error!?i32 {
        if (self.kind != .int32) return error.InvalidMetadataType;
        if (self.remaining == 0) return null;
        if (self.bytes.len < 4) return error.EndOfStream;
        const value = std.mem.readInt(i32, self.bytes[0..4], .little);
        self.bytes = self.bytes[4..];
        self.remaining -= 1;
        return value;
    }
};
pub const Value = union(enum) {
    unsigned: u64,
    signed: i64,
    float: f64,
    boolean: bool,
    string: []const u8,
    array: Array,
};
pub const Metadata = struct { key: []const u8, kind: Type, value: Value };
pub const Tensor = struct {
    name: []const u8,
    dimensions: []const u64,
    encoding_id: u32,
    /// Relative to Document.data_offset, not to the file's beginning.
    offset: u64,
    bytes: u64,
    elements: u64,
};

pub const Document = struct {
    storage: std.heap.ArenaAllocator,
    version: u32,
    file_bytes: u64,
    directory_bytes: u64,
    data_offset: u64,
    alignment: u32,
    metadata: []const Metadata,
    tensors: []const Tensor,

    /// Releases all directory storage. Do not copy an owning Document and then
    /// deinit both copies; returned metadata/tensor slices borrow from this owner.
    pub fn deinit(self: *Document) void {
        self.storage.deinit();
        self.* = undefined;
    }

    pub fn get(self: Document, key: []const u8) ?Value {
        for (self.metadata) |entry| {
            if (std.mem.eql(u8, entry.key, key)) return entry.value;
        }
        return null;
    }

    pub fn string(self: Document, key: []const u8) ?[]const u8 {
        return switch (self.get(key) orelse return null) {
            .string => |s| s,
            else => null,
        };
    }
};

pub const Error = Allocator.Error || std.Io.Reader.Error || error{
    InvalidMagic,
    UnsupportedVersion,
    LimitExceeded,
    InvalidMetadataType,
    UnsupportedNestedArray,
    InvalidBoolean,
    InvalidName,
    DuplicateMetadata,
    DuplicateTensor,
    InvalidAlignment,
    InvalidShape,
    Overflow,
    UnsupportedTensorType,
    TensorOutOfBounds,
    OverlappingTensors,
};

const Cursor = struct {
    reader: *std.Io.Reader,
    position: u64 = 0,
    file_bytes: u64,
    limits: Limits,

    fn advance(self: *Cursor, n: u64) Error!void {
        // Subtraction after the position check avoids attacker-controlled addition
        // overflowing before we have a chance to reject it.
        if (self.position > self.file_bytes or n > self.file_bytes - self.position)
            return error.EndOfStream;
        if (self.position > self.limits.directory_bytes or n > self.limits.directory_bytes - self.position)
            return error.LimitExceeded;
        self.position += n;
    }

    fn int(self: *Cursor, comptime T: type) Error!T {
        try self.advance(@sizeOf(T));
        return self.reader.takeInt(T, .little);
    }

    fn skip(self: *Cursor, n: u64) Error!void {
        try self.advance(n);
        try self.reader.discardAll64(n);
    }

    fn string(self: *Cursor, alloc: Allocator) Error![]const u8 {
        const len = try self.int(u64);
        if (len > self.limits.string_bytes) return error.LimitExceeded;
        try self.advance(len);
        const bytes = try alloc.alloc(u8, std.math.cast(usize, len) orelse return error.LimitExceeded);
        try self.reader.readSliceAll(bytes);
        return bytes;
    }

    fn kind(self: *Cursor) Error!Type {
        return std.enums.fromInt(Type, try self.int(u32)) orelse error.InvalidMetadataType;
    }

    fn value(self: *Cursor, alloc: Allocator, tag: Type) Error!Value {
        return switch (tag) {
            .uint8 => .{ .unsigned = try self.int(u8) },
            .uint16 => .{ .unsigned = try self.int(u16) },
            .uint32 => .{ .unsigned = try self.int(u32) },
            .uint64 => .{ .unsigned = try self.int(u64) },
            .int8 => .{ .signed = try self.int(i8) },
            .int16 => .{ .signed = try self.int(i16) },
            .int32 => .{ .signed = try self.int(i32) },
            .int64 => .{ .signed = try self.int(i64) },
            .float32 => .{ .float = @as(f32, @bitCast(try self.int(u32))) },
            .float64 => .{ .float = @bitCast(try self.int(u64)) },
            .boolean => .{ .boolean = switch (try self.int(u8)) {
                0 => false,
                1 => true,
                else => return error.InvalidBoolean,
            } },
            .string => .{ .string = try self.string(alloc) },
            .array => blk: {
                const element_type = try self.kind();
                if (element_type == .array) return error.UnsupportedNestedArray;
                const count = try self.int(u64);
                if (count > self.limits.array_items) return error.LimitExceeded;
                var array: Array = .{ .element_type = element_type, .count = count, .file_offset = self.position };
                if (element_type != .string and count <= retained_array_items) {
                    const values = try alloc.alloc(Value, @intCast(count));
                    for (values) |*item| item.* = try self.value(alloc, element_type);
                    array.values = values;
                } else if (element_type == .string) {
                    for (0..@intCast(count)) |_| {
                        const len = try self.int(u64);
                        if (len > self.limits.string_bytes) return error.LimitExceeded;
                        try self.skip(len);
                    }
                } else if (element_type == .boolean) {
                    for (0..@intCast(count)) |_| {
                        if (try self.int(u8) > 1) return error.InvalidBoolean;
                    }
                } else {
                    const size: u64 = switch (element_type) {
                        .uint8, .int8 => 1,
                        .uint16, .int16 => 2,
                        .uint32, .int32, .float32 => 4,
                        .uint64, .int64, .float64 => 8,
                        else => unreachable,
                    };
                    try self.skip(try std.math.mul(u64, count, size));
                }
                break :blk .{ .array = array };
            },
        };
    }
};

fn validateName(name: []const u8) Error!void {
    if (name.len == 0 or name.len > 65_535 or !std.unicode.utf8ValidateSlice(name))
        return error.InvalidName;
    for (name) |byte| if (byte < 0x20 or byte == 0x7f) return error.InvalidName;
}

/// Names the tensor a rejection concerned, since the errors carry no
/// payload: filled for `UnsupportedTensorType`, `InvalidShape`, and
/// `Overflow` raised while sizing a tensor, so an inspection can report
/// the first offending tensor and encoding instead of a bare error name.
/// The name is copied (the directory storage is freed on error) and
/// truncated to the buffer.
pub const Rejection = struct {
    buffer: [128]u8 = undefined,
    len: usize = 0,
    encoding_id: ?u32 = null,

    pub fn tensor(self: *const Rejection) ?[]const u8 {
        return if (self.len == 0) null else self.buffer[0..self.len];
    }

    fn note(self: *Rejection, name: []const u8, encoding_id: u32) void {
        self.len = @min(name.len, self.buffer.len);
        @memcpy(self.buffer[0..self.len], name[0..self.len]);
        self.encoding_id = encoding_id;
    }
};

/// Reader must start at file offset zero and have at least 8 bytes of buffer
/// capacity. file_bytes is the length of the complete file, including weights.
pub fn parse(gpa: Allocator, reader: *std.Io.Reader, file_bytes: u64, limits: Limits) Error!Document {
    return parseDiagnosed(gpa, reader, file_bytes, limits, null);
}

/// `parse` with an optional `Rejection` that receives the tensor behind a
/// per-tensor rejection (see `Rejection`).
pub fn parseDiagnosed(gpa: Allocator, reader: *std.Io.Reader, file_bytes: u64, limits: Limits, rejection: ?*Rejection) Error!Document {
    var storage = std.heap.ArenaAllocator.init(gpa);
    errdefer storage.deinit();
    const alloc = storage.allocator();
    var cursor: Cursor = .{ .reader = reader, .file_bytes = file_bytes, .limits = limits };
    if (try cursor.int(u32) != 0x46554747) return error.InvalidMagic;
    const version = try cursor.int(u32);
    if (version != 3) return error.UnsupportedVersion;
    const tensor_count = try cursor.int(u64);
    const metadata_count = try cursor.int(u64);
    if (tensor_count > limits.tensor_count or metadata_count > limits.metadata_count)
        return error.LimitExceeded;
    const metadata = try alloc.alloc(Metadata, std.math.cast(usize, metadata_count) orelse return error.LimitExceeded);
    var keys = std.StringHashMap(void).init(alloc);
    var alignment: u32 = 32;
    for (metadata) |*entry| {
        const key = try cursor.string(alloc);
        try validateName(key);
        const inserted = try keys.getOrPut(key);
        if (inserted.found_existing) return error.DuplicateMetadata;
        const tag = try cursor.kind();
        const value = try cursor.value(alloc, tag);
        entry.* = .{ .key = key, .kind = tag, .value = value };
        if (std.mem.eql(u8, key, "general.alignment")) {
            if (tag != .uint32) return error.InvalidAlignment;
            alignment = @intCast(value.unsigned);
            if (alignment == 0 or !std.math.isPowerOfTwo(alignment)) return error.InvalidAlignment;
        }
    }
    const tensors = try alloc.alloc(Tensor, std.math.cast(usize, tensor_count) orelse return error.LimitExceeded);
    var names = std.StringHashMap(void).init(alloc);
    for (tensors) |*tensor| {
        const name = try cursor.string(alloc);
        try validateName(name);
        if ((try names.getOrPut(name)).found_existing) return error.DuplicateTensor;
        const rank = try cursor.int(u32);
        if (rank == 0 or rank > 4) return error.InvalidShape;
        const dimensions = try alloc.alloc(u64, rank);
        var elements: u64 = 1;
        for (dimensions) |*dim| {
            dim.* = try cursor.int(u64);
            if (dim.* == 0) return error.InvalidShape;
            elements = try std.math.mul(u64, elements, dim.*);
        }
        const encoding_id = try cursor.int(u32);
        const layout = encoding.layout(encoding_id) orelse {
            if (rejection) |r| r.note(name, encoding_id);
            return error.UnsupportedTensorType;
        };
        const bytes = layout.byteCount(dimensions) catch |err| {
            if (rejection) |r| r.note(name, encoding_id);
            return err;
        };
        const offset = try cursor.int(u64);
        if (offset % alignment != 0) return error.InvalidAlignment;
        tensor.* = .{
            .name = name,
            .dimensions = dimensions,
            .encoding_id = encoding_id,
            .offset = offset,
            .bytes = bytes,
            .elements = elements,
        };
    }
    const padding = (alignment - cursor.position % alignment) % alignment;
    const data_offset = try std.math.add(u64, cursor.position, padding);
    if (data_offset > file_bytes) return error.EndOfStream;
    // Tensor descriptors need not be stored in offset order. Sort a copy so
    // callers still see the original directory, while overlap validation is O(n log n).
    const ordered = try alloc.dupe(Tensor, tensors);
    std.mem.sort(Tensor, ordered, {}, struct {
        fn less(_: void, a: Tensor, b: Tensor) bool {
            return a.offset < b.offset;
        }
    }.less);
    var previous_end: u64 = 0;
    for (ordered) |tensor| {
        if (tensor.offset > file_bytes - data_offset or tensor.bytes > file_bytes - data_offset - tensor.offset)
            return error.TensorOutOfBounds;
        if (tensor.offset < previous_end) return error.OverlappingTensors;
        previous_end = tensor.offset + tensor.bytes;
    }
    return .{
        .storage = storage,
        .version = version,
        .file_bytes = file_bytes,
        .directory_bytes = cursor.position,
        .data_offset = data_offset,
        .alignment = alignment,
        .metadata = metadata,
        .tensors = tensors,
    };
}

/// Thin I/O shell: closes the file before returning because the Document owns
/// all of its directory data and contains no borrowed file-reader buffers.
pub fn open(gpa: Allocator, io: std.Io, path: []const u8, limits: Limits) !Document {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.kind != .file) return error.NotRegularFile;
    var buffer: [64 * 1024]u8 = undefined;
    var reader = file.reader(io, &buffer);
    return parse(gpa, &reader.interface, stat.size, limits);
}

// Fixtures are generated from the wire format, not from host struct layouts.
// This intentionally exercises little-endian reads and unaligned string lengths.
const FixtureOptions = struct {
    version: u32 = 3,
    alignment: u32 = 32,
    duplicate_metadata: bool = false,
    duplicate_tensor: bool = false,
    second_offset: u64 = 32,
    encoding_id: u32 = 0,
    dimension: u64 = 4,
};

fn fixtureString(writer: *std.Io.Writer, text: []const u8) !void {
    try writer.writeInt(u64, text.len, .little);
    try writer.writeAll(text);
}

fn fixture(options: FixtureOptions) !std.Io.Writer.Allocating {
    var out: std.Io.Writer.Allocating = .init(std.testing.allocator);
    errdefer out.deinit();
    const w = &out.writer;
    try w.writeAll("GGUF");
    try w.writeInt(u32, options.version, .little);
    try w.writeInt(u64, 2, .little);
    try w.writeInt(u64, 3, .little);
    try fixtureString(w, "general.architecture");
    try w.writeInt(u32, 8, .little);
    try fixtureString(w, "fixture");
    try fixtureString(w, if (options.duplicate_metadata) "general.architecture" else "general.alignment");
    try w.writeInt(u32, 4, .little);
    try w.writeInt(u32, options.alignment, .little);
    try fixtureString(w, "tokenizer.ggml.tokens");
    try w.writeInt(u32, 9, .little);
    try w.writeInt(u32, 8, .little);
    try w.writeInt(u64, 2, .little);
    try fixtureString(w, "a");
    try fixtureString(w, "unicode: λ");
    for (0..2) |i| {
        try fixtureString(w, if (i == 0 or options.duplicate_tensor) "first" else "second");
        try w.writeInt(u32, 1, .little);
        try w.writeInt(u64, options.dimension, .little);
        try w.writeInt(u32, options.encoding_id, .little);
        try w.writeInt(u64, if (i == 0) 0 else options.second_offset, .little);
    }
    try w.splatByteAll(0, (32 - out.written().len % 32) % 32);
    try w.splatByteAll(0, 64);
    return out;
}

test "a rejected tensor is named for the caller that asks" {
    var bytes = try fixture(.{ .encoding_id = 999 });
    defer bytes.deinit();
    var reader: std.Io.Reader = .fixed(bytes.written());
    var rejection: Rejection = .{};
    try std.testing.expectError(error.UnsupportedTensorType, parseDiagnosed(std.testing.allocator, &reader, bytes.written().len, .{}, &rejection));
    try std.testing.expectEqualStrings("first", rejection.tensor().?);
    try std.testing.expectEqual(@as(u32, 999), rejection.encoding_id.?);
    // A row that does not hold whole blocks is the same kind of rejection.
    var odd = try fixture(.{ .encoding_id = 12, .dimension = 100 });
    defer odd.deinit();
    reader = .fixed(odd.written());
    rejection = .{};
    try std.testing.expectError(error.InvalidShape, parseDiagnosed(std.testing.allocator, &reader, odd.written().len, .{}, &rejection));
    try std.testing.expectEqualStrings("first", rejection.tensor().?);
    try std.testing.expectEqual(@as(u32, 12), rejection.encoding_id.?);
    // Without a rejection the behavior is `parse`'s.
    reader = .fixed(bytes.written());
    try std.testing.expectError(error.UnsupportedTensorType, parse(std.testing.allocator, &reader, bytes.written().len, .{}));
    var untouched: Rejection = .{};
    try std.testing.expect(untouched.tensor() == null);
}

fn parseFixture(alloc: Allocator, bytes: []const u8) !void {
    var reader: std.Io.Reader = .fixed(bytes);
    var document = try parse(alloc, &reader, bytes.len, .{});
    defer document.deinit();
    try std.testing.expectEqualStrings("fixture", document.string("general.architecture").?);
    try std.testing.expectEqual(@as(usize, 2), document.tensors.len);
}

test "read metadata and tensor ranges without consuming weights" {
    var bytes = try fixture(.{});
    defer bytes.deinit();
    var reader: std.Io.Reader = .fixed(bytes.written());
    var doc = try parse(std.testing.allocator, &reader, bytes.written().len, .{});
    defer doc.deinit();
    try std.testing.expectEqualStrings("fixture", doc.string("general.architecture").?);
    try std.testing.expectEqual(@as(u64, 16), doc.tensors[0].bytes);
    try std.testing.expectEqual(@as(u64, 2), doc.get("tokenizer.ggml.tokens").?.array.count);
    try std.testing.expectEqual(doc.directory_bytes, reader.seek);
    try std.testing.expect(doc.directory_bytes < doc.data_offset);
}

test "lazy array access bounds offsets, counts, strings, and element types" {
    const bytes = "\x01\x00\x00\x00\x00\x00\x00\x00a";
    const descriptor: Array = .{ .element_type = .string, .count = 1, .file_offset = 0 };
    var reader = try ArrayReader.init(descriptor, bytes, .{});
    try std.testing.expectEqualStrings("a", (try reader.nextString()).?);
    try std.testing.expectEqual(@as(?[]const u8, null), try reader.nextString());
    try std.testing.expectError(error.InvalidMetadataType, reader.nextInt32());
    for (0..bytes.len) |len| {
        if (ArrayReader.init(descriptor, bytes[0..len], .{})) |value| {
            var truncated = value;
            try std.testing.expectError(error.EndOfStream, truncated.nextString());
        } else |err| try std.testing.expectEqual(error.EndOfStream, err);
    }
    var limited = try ArrayReader.init(descriptor, bytes, .{ .string_bytes = 0 });
    try std.testing.expectError(error.LimitExceeded, limited.nextString());
    try std.testing.expectError(error.LimitExceeded, ArrayReader.init(descriptor, bytes, .{ .array_items = 0 }));
    try std.testing.expectError(error.LimitExceeded, ArrayReader.init(descriptor, bytes, .{ .directory_bytes = 8 }));
    try std.testing.expectError(error.EndOfStream, ArrayReader.init(.{ .element_type = .string, .count = 0, .file_offset = std.math.maxInt(u64) }, bytes, .{}));
    var signed = try ArrayReader.init(.{ .element_type = .int32, .count = 1, .file_offset = 0 }, "\xff\xff\xff\xff", .{});
    try std.testing.expectEqual(@as(?i32, -1), try signed.nextInt32());
    try std.testing.expectEqual(@as(?i32, null), try signed.nextInt32());
    try std.testing.expectError(error.InvalidMetadataType, signed.nextString());
}

test "truncated directory and tensor payloads fail cleanly" {
    var bytes = try fixture(.{});
    defer bytes.deinit();
    var full_reader: std.Io.Reader = .fixed(bytes.written());
    var doc = try parse(std.testing.allocator, &full_reader, bytes.written().len, .{});
    defer doc.deinit();
    const required_size: usize = @intCast(doc.data_offset + 48);
    for (0..required_size) |len| {
        var reader: std.Io.Reader = .fixed(bytes.written()[0..len]);
        if (parse(std.testing.allocator, &reader, len, .{})) |parsed| {
            var unexpected = parsed;
            unexpected.deinit();
            return error.UnexpectedParseSuccess;
        } else |err| {
            try std.testing.expect(err == error.EndOfStream or err == error.TensorOutOfBounds);
        }
    }
}

test "reject unsupported containers, duplicate names, invalid layouts and ranges" {
    const Case = struct { options: FixtureOptions, expected: anyerror };
    const cases = [_]Case{
        .{ .options = .{ .version = 2 }, .expected = error.UnsupportedVersion },
        .{ .options = .{ .alignment = 0 }, .expected = error.InvalidAlignment },
        .{ .options = .{ .alignment = 3 }, .expected = error.InvalidAlignment },
        .{ .options = .{ .duplicate_metadata = true }, .expected = error.DuplicateMetadata },
        .{ .options = .{ .duplicate_tensor = true }, .expected = error.DuplicateTensor },
        .{ .options = .{ .dimension = 0 }, .expected = error.InvalidShape },
        .{ .options = .{ .dimension = std.math.maxInt(u64) }, .expected = error.Overflow },
        .{ .options = .{ .encoding_id = 999 }, .expected = error.UnsupportedTensorType },
        .{ .options = .{ .encoding_id = 12 }, .expected = error.InvalidShape },
        .{ .options = .{ .second_offset = 1 }, .expected = error.InvalidAlignment },
        .{ .options = .{ .second_offset = 0 }, .expected = error.OverlappingTensors },
        .{ .options = .{ .second_offset = 1024 }, .expected = error.TensorOutOfBounds },
    };
    for (cases) |case| {
        var bytes = try fixture(case.options);
        defer bytes.deinit();
        var reader: std.Io.Reader = .fixed(bytes.written());
        try std.testing.expectError(case.expected, parse(std.testing.allocator, &reader, bytes.written().len, .{}));
    }
}

test "hostile lengths and counts are rejected before allocating" {
    var bytes = try fixture(.{});
    defer bytes.deinit();
    const cases = [_]Limits{
        .{ .tensor_count = 1 }, .{ .metadata_count = 1 }, .{ .directory_bytes = 24 },
        .{ .string_bytes = 4 }, .{ .array_items = 1 },
    };
    for (cases) |limits| {
        var reader: std.Io.Reader = .fixed(bytes.written());
        try std.testing.expectError(error.LimitExceeded, parse(std.testing.allocator, &reader, bytes.written().len, limits));
    }
    // First metadata key length follows the 24-byte fixed header.
    std.mem.writeInt(u64, bytes.written()[24..32], std.math.maxInt(u64), .little);
    var reader: std.Io.Reader = .fixed(bytes.written());
    try std.testing.expectError(error.LimitExceeded, parse(std.testing.allocator, &reader, bytes.written().len, .{}));
}

test "arena cleanup survives every allocation failure" {
    var bytes = try fixture(.{});
    defer bytes.deinit();
    try std.testing.checkAllAllocationFailures(std.testing.allocator, parseFixture, .{bytes.written()});
}

test "reject bad magic and unknown metadata tags before interpreting values" {
    var bytes = try fixture(.{});
    defer bytes.deinit();
    bytes.written()[0] = 'X';
    var reader: std.Io.Reader = .fixed(bytes.written());
    try std.testing.expectError(error.InvalidMagic, parse(std.testing.allocator, &reader, bytes.written().len, .{}));
    bytes.written()[0] = 'G';
    // Fixed header, key length, then the first key's bytes precede its type tag.
    const tag_offset = 24 + 8 + "general.architecture".len;
    std.mem.writeInt(u32, bytes.written()[tag_offset..][0..4], 99, .little);
    reader = .fixed(bytes.written());
    try std.testing.expectError(error.InvalidMetadataType, parse(std.testing.allocator, &reader, bytes.written().len, .{}));
}

test "invalid boolean values and nested arrays are explicit errors" {
    var bytes = try fixture(.{});
    defer bytes.deinit();
    const tag_offset = 24 + 8 + "general.architecture".len;
    std.mem.writeInt(u32, bytes.written()[tag_offset..][0..4], 7, .little);
    bytes.written()[tag_offset + 4] = 2;
    var reader: std.Io.Reader = .fixed(bytes.written());
    try std.testing.expectError(error.InvalidBoolean, parse(std.testing.allocator, &reader, bytes.written().len, .{}));
    std.mem.writeInt(u32, bytes.written()[tag_offset..][0..4], 9, .little);
    std.mem.writeInt(u32, bytes.written()[tag_offset + 4 ..][0..4], 9, .little);
    reader = .fixed(bytes.written());
    try std.testing.expectError(error.UnsupportedNestedArray, parse(std.testing.allocator, &reader, bytes.written().len, .{}));
}

test "retain small numeric arrays while leaving large arrays as descriptors" {
    var bytes: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer bytes.deinit();
    const w = &bytes.writer;
    try w.writeAll("GGUF");
    try w.writeInt(u32, 3, .little);
    try w.writeInt(u64, 0, .little);
    try w.writeInt(u64, 2, .little);
    for ([_]u64{ 4, retained_array_items + 1 }) |count| {
        try fixtureString(w, if (count == 4) "sections" else "large");
        try w.writeInt(u32, 9, .little);
        try w.writeInt(u32, 5, .little);
        try w.writeInt(u64, count, .little);
        for (0..@intCast(count)) |i| try w.writeInt(i32, @as(i32, @intCast(i)) - 1, .little);
    }
    try w.splatByteAll(0, (32 - bytes.written().len % 32) % 32);
    var reader: std.Io.Reader = .fixed(bytes.written());
    var doc = try parse(std.testing.allocator, &reader, bytes.written().len, .{});
    defer doc.deinit();
    const sections = doc.get("sections").?.array;
    try std.testing.expectEqual(@as(i64, -1), sections.values.?[0].signed);
    try std.testing.expectEqual(@as(i64, 2), sections.values.?[3].signed);
    try std.testing.expect(doc.get("large").?.array.values == null);
}
