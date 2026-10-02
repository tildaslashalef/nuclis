//! Bounded safetensors reader: the header directory, and a mapped checkpoint
//! of one file or an indexed shard set.
//!
//! parse() reads the 8-byte length and the JSON header only, never weights,
//! and checks the byte buffer the way the format defines it: every tensor's
//! size is its shape times its dtype, and sorted by offset the tensors tile
//! the buffer exactly, without holes or overlap. The Document owns an arena.
//! Offsets need not be aligned, so values are read as unaligned little
//! endian. Format summary: docs/reference/safetensors.md.
const std = @import("std");
const Allocator = std.mem.Allocator;

pub const Limits = struct {
    /// The format's own bound on the JSON header.
    header_bytes: u64 = 100_000_000,
    tensor_count: u64 = 100_000,
    metadata_count: u64 = 16_384,
    rank: u32 = 8,
    /// An index (`*.safetensors.index.json`) and its shard count.
    index_bytes: u64 = 64 * 1024 * 1024,
    shard_count: u32 = 10_000,
};

/// The element types of the format. Sub-byte and complex types are not
/// recognized and are rejected by name.
pub const Dtype = enum {
    bool,
    u8,
    i8,
    f8_e5m2,
    f8_e4m3,
    f8_e8m0,
    i16,
    u16,
    f16,
    bf16,
    i32,
    u32,
    f32,
    i64,
    u64,
    f64,

    const wire = [_]struct { []const u8, Dtype }{
        .{ "BOOL", .bool },       .{ "U8", .u8 },           .{ "I8", .i8 },
        .{ "F8_E5M2", .f8_e5m2 }, .{ "F8_E4M3", .f8_e4m3 }, .{ "F8_E8M0", .f8_e8m0 },
        .{ "I16", .i16 },         .{ "U16", .u16 },         .{ "F16", .f16 },
        .{ "BF16", .bf16 },       .{ "I32", .i32 },         .{ "U32", .u32 },
        .{ "F32", .f32 },         .{ "I64", .i64 },         .{ "U64", .u64 },
        .{ "F64", .f64 },
    };

    pub fn fromName(text: []const u8) ?Dtype {
        for (wire) |entry| if (std.mem.eql(u8, entry[0], text)) return entry[1];
        return null;
    }

    /// The spelling the format uses (`BF16`).
    pub fn name(self: Dtype) []const u8 {
        for (wire) |entry| if (entry[1] == self) return entry[0];
        unreachable;
    }

    pub fn size(self: Dtype) u8 {
        return switch (self) {
            .bool, .u8, .i8, .f8_e5m2, .f8_e4m3, .f8_e8m0 => 1,
            .i16, .u16, .f16, .bf16 => 2,
            .i32, .u32, .f32 => 4,
            .i64, .u64, .f64 => 8,
        };
    }
};

pub const Tensor = struct {
    name: []const u8,
    dtype: Dtype,
    /// Row-major, outermost first (the format's order, unlike GGUF's).
    /// Empty for a scalar.
    shape: []const u64,
    /// Relative to Document.data_offset.
    offset: u64,
    bytes: u64,
    elements: u64,
};

pub const Metadata = struct { key: []const u8, value: []const u8 };

pub const Document = struct {
    storage: std.heap.ArenaAllocator,
    file_bytes: u64,
    header_bytes: u64,
    /// `8 + header_bytes`: where the byte buffer begins.
    data_offset: u64,
    metadata: []const Metadata,
    /// In header order.
    tensors: []const Tensor,
    by_name: std.StringHashMapUnmanaged(u32),

    /// Releases all header storage; tensors and metadata borrow from it.
    pub fn deinit(self: *Document) void {
        self.storage.deinit();
        self.* = undefined;
    }

    pub fn find(self: *const Document, name: []const u8) ?*const Tensor {
        const index = self.by_name.get(name) orelse return null;
        return &self.tensors[index];
    }

    pub fn metadataValue(self: *const Document, key: []const u8) ?[]const u8 {
        for (self.metadata) |entry| if (std.mem.eql(u8, entry.key, key)) return entry.value;
        return null;
    }
};

pub const Error = Allocator.Error || std.Io.Reader.Error || error{
    LimitExceeded,
    InvalidHeader,
    DuplicateTensor,
    UnsupportedDtype,
    InvalidShape,
    Overflow,
    TensorOutOfBounds,
    OverlappingTensors,
    UnindexedBytes,
};

/// Names the tensor behind a per-tensor rejection, as the GGUF reader's does.
pub const Rejection = struct {
    buffer: [128]u8 = undefined,
    len: usize = 0,

    pub fn tensor(self: *const Rejection) ?[]const u8 {
        return if (self.len == 0) null else self.buffer[0..self.len];
    }

    fn note(self: ?*Rejection, name: []const u8) void {
        const r = self orelse return;
        r.len = @min(name.len, r.buffer.len);
        @memcpy(r.buffer[0..r.len], name[0..r.len]);
    }
};

/// Reader must start at file offset zero; file_bytes is the whole file.
pub fn parse(gpa: Allocator, reader: *std.Io.Reader, file_bytes: u64, limits: Limits) Error!Document {
    return parseDiagnosed(gpa, reader, file_bytes, limits, null);
}

pub fn parseDiagnosed(gpa: Allocator, reader: *std.Io.Reader, file_bytes: u64, limits: Limits, rejection: ?*Rejection) Error!Document {
    if (file_bytes < 8) return error.EndOfStream;
    var length: [8]u8 = undefined;
    try reader.readSliceAll(&length);
    const header_bytes = std.mem.readInt(u64, &length, .little);
    // Checked before allocating: a hostile length never reaches the allocator.
    if (header_bytes > limits.header_bytes) return error.LimitExceeded;
    if (header_bytes < 2 or header_bytes > file_bytes - 8) return error.InvalidHeader;
    const header = try gpa.alloc(u8, @intCast(header_bytes));
    defer gpa.free(header);
    try reader.readSliceAll(header);
    return parseHeader(gpa, header, file_bytes, limits, rejection);
}

fn parseHeader(gpa: Allocator, header: []const u8, file_bytes: u64, limits: Limits, rejection: ?*Rejection) Error!Document {
    if (header[0] != '{') return error.InvalidHeader;
    var storage = std.heap.ArenaAllocator.init(gpa);
    errdefer storage.deinit();
    const arena = storage.allocator();
    var scanner = std.json.Scanner.initCompleteInput(gpa, header);
    defer scanner.deinit();
    const data_offset = 8 + @as(u64, header.len);
    const buffer_bytes = file_bytes - data_offset;

    var tensors: std.ArrayList(Tensor) = .empty;
    var metadata: std.ArrayList(Metadata) = .empty;
    var by_name: std.StringHashMapUnmanaged(u32) = .empty;
    var seen_metadata = false;
    try expect(&scanner, .object_begin);
    while (true) {
        const key = switch (try token(&scanner, arena)) {
            .object_end => break,
            .allocated_string => |s| s,
            else => return error.InvalidHeader,
        };
        if (std.mem.eql(u8, key, "__metadata__")) {
            if (seen_metadata) return error.InvalidHeader;
            seen_metadata = true;
            try expect(&scanner, .object_begin);
            while (true) {
                const k = switch (try token(&scanner, arena)) {
                    .object_end => break,
                    .allocated_string => |s| s,
                    else => return error.InvalidHeader,
                };
                const v = switch (try token(&scanner, arena)) {
                    .allocated_string => |s| s,
                    else => return error.InvalidHeader,
                };
                for (metadata.items) |m| if (std.mem.eql(u8, m.key, k)) return error.InvalidHeader;
                if (metadata.items.len == limits.metadata_count) return error.LimitExceeded;
                try metadata.append(arena, .{ .key = k, .value = v });
            }
            continue;
        }
        if (tensors.items.len == limits.tensor_count) return error.LimitExceeded;
        const inserted = try by_name.getOrPut(arena, key);
        if (inserted.found_existing) {
            Rejection.note(rejection, key);
            return error.DuplicateTensor;
        }
        inserted.value_ptr.* = @intCast(tensors.items.len);
        const tensor = parseTensor(&scanner, arena, key, limits) catch |err| {
            Rejection.note(rejection, key);
            return err;
        };
        try tensors.append(arena, tensor);
    }
    const last = scanner.next() catch return error.InvalidHeader;
    if (last != .end_of_document) return error.InvalidHeader;

    // The buffer is tiled exactly: sorted by offset, each tensor starts
    // where the previous ended, and the last ends at the file's end.
    const ordered = try gpa.dupe(Tensor, tensors.items);
    defer gpa.free(ordered);
    std.mem.sort(Tensor, ordered, {}, struct {
        fn less(_: void, a: Tensor, b: Tensor) bool {
            return a.offset < b.offset or (a.offset == b.offset and a.bytes < b.bytes);
        }
    }.less);
    var end: u64 = 0;
    for (ordered) |t| {
        if (t.offset > buffer_bytes or t.bytes > buffer_bytes - t.offset) {
            Rejection.note(rejection, t.name);
            return error.TensorOutOfBounds;
        }
        if (t.offset < end) {
            Rejection.note(rejection, t.name);
            return error.OverlappingTensors;
        }
        if (t.offset > end) {
            Rejection.note(rejection, t.name);
            return error.UnindexedBytes;
        }
        end = t.offset + t.bytes;
    }
    if (end != buffer_bytes) return error.UnindexedBytes;
    return .{
        .storage = storage,
        .file_bytes = file_bytes,
        .header_bytes = header.len,
        .data_offset = data_offset,
        .metadata = metadata.items,
        .tensors = tensors.items,
        .by_name = by_name,
    };
}

/// One tensor entry: exactly `dtype`, `shape`, and `data_offsets`, once each.
fn parseTensor(scanner: *std.json.Scanner, arena: Allocator, name: []const u8, limits: Limits) Error!Tensor {
    try expect(scanner, .object_begin);
    var dtype: ?Dtype = null;
    var shape: ?[]const u64 = null;
    var offsets: ?[2]u64 = null;
    while (true) {
        const field = switch (try token(scanner, arena)) {
            .object_end => break,
            .allocated_string => |s| s,
            else => return error.InvalidHeader,
        };
        if (std.mem.eql(u8, field, "dtype")) {
            if (dtype != null) return error.InvalidHeader;
            const value = switch (try token(scanner, arena)) {
                .allocated_string => |s| s,
                else => return error.InvalidHeader,
            };
            dtype = Dtype.fromName(value) orelse return error.UnsupportedDtype;
        } else if (std.mem.eql(u8, field, "shape")) {
            if (shape != null) return error.InvalidHeader;
            var dims: std.ArrayList(u64) = .empty;
            try expect(scanner, .array_begin);
            while ((scanner.peekNextTokenType() catch return error.InvalidHeader) != .array_end) {
                if (dims.items.len == limits.rank) return error.InvalidShape;
                try dims.append(arena, try unsigned(scanner));
            }
            try expect(scanner, .array_end);
            shape = dims.items;
        } else if (std.mem.eql(u8, field, "data_offsets")) {
            if (offsets != null) return error.InvalidHeader;
            try expect(scanner, .array_begin);
            offsets = .{ try unsigned(scanner), try unsigned(scanner) };
            try expect(scanner, .array_end);
        } else return error.InvalidHeader;
    }
    const d = dtype orelse return error.InvalidHeader;
    const dims = shape orelse return error.InvalidHeader;
    const range = offsets orelse return error.InvalidHeader;
    if (range[1] < range[0]) return error.InvalidShape;
    var elements: u64 = 1;
    for (dims) |dim| elements = try std.math.mul(u64, elements, dim);
    const bytes = try std.math.mul(u64, elements, d.size());
    if (bytes != range[1] - range[0]) return error.InvalidShape;
    return .{ .name = name, .dtype = d, .shape = dims, .offset = range[0], .bytes = bytes, .elements = elements };
}

/// Strings are copied into the arena (they outlive the header buffer);
/// numbers borrow the header, which a complete input always allows.
fn token(scanner: *std.json.Scanner, arena: Allocator) Error!std.json.Token {
    const next = scanner.peekNextTokenType() catch return error.InvalidHeader;
    return (if (next == .string)
        scanner.nextAlloc(arena, .alloc_always)
    else
        scanner.next()) catch |err| switch (err) {
        error.OutOfMemory => error.OutOfMemory,
        else => error.InvalidHeader,
    };
}

fn expect(scanner: *std.json.Scanner, comptime kind: std.meta.Tag(std.json.Token)) Error!void {
    const t = scanner.next() catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidHeader,
    };
    if (t != kind) return error.InvalidHeader;
}

fn unsigned(scanner: *std.json.Scanner) Error!u64 {
    const t = scanner.next() catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidHeader,
    };
    const digits = switch (t) {
        .number => |n| n,
        else => return error.InvalidHeader,
    };
    return std.fmt.parseInt(u64, digits, 10) catch error.InvalidHeader;
}

/// Thin I/O shell over `parse`; closes the file before returning.
pub fn open(gpa: Allocator, io: std.Io, path: []const u8, limits: Limits) !Document {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.kind != .file) return error.NotRegularFile;
    var buffer: [64 * 1024]u8 = undefined;
    var reader = file.reader(io, &buffer);
    return parse(gpa, &reader.interface, stat.size, limits);
}

/// A tensor of a checkpoint: its descriptor and its bytes in the mapping.
pub const Ref = struct {
    tensor: *const Tensor,
    bytes: []const u8,

    /// Decodes `out.len` elements from element `first` to f32. F32, F16,
    /// and BF16 only; unaligned little-endian reads.
    pub fn decode(self: Ref, first: u64, out: []f32) error{ UnsupportedDtype, TensorOutOfBounds }!void {
        const t = self.tensor;
        if (first > t.elements or out.len > t.elements - first) return error.TensorOutOfBounds;
        const width = t.dtype.size();
        const start: usize = @intCast(first * width);
        switch (t.dtype) {
            .f32 => for (out, 0..) |*x, i| {
                x.* = @bitCast(std.mem.readInt(u32, self.bytes[start + i * 4 ..][0..4], .little));
            },
            .f16 => for (out, 0..) |*x, i| {
                const h: f16 = @bitCast(std.mem.readInt(u16, self.bytes[start + i * 2 ..][0..2], .little));
                x.* = h;
            },
            // BF16 is the high half of an f32.
            .bf16 => for (out, 0..) |*x, i| {
                x.* = @bitCast(@as(u32, std.mem.readInt(u16, self.bytes[start + i * 2 ..][0..2], .little)) << 16);
            },
            else => return error.UnsupportedDtype,
        }
    }
};

/// A checkpoint mapped read-only: one `.safetensors` file, or the shards an
/// index names. Every `Ref` borrows a mapping and dies with `deinit`; the
/// files must not change while open.
pub const Checkpoint = struct {
    storage: std.heap.ArenaAllocator,
    /// The file opened, or the index; owned.
    path: []const u8,
    shards: []Shard,
    locations: std.StringHashMapUnmanaged(Location),

    pub const Shard = struct {
        /// The shard's path; owned by the checkpoint.
        path: []const u8,
        file: std.Io.File,
        mapping: std.Io.File.MemoryMap,
        document: Document,
    };
    const Location = struct { shard: u32, tensor: u32 };

    /// `path` is a `.safetensors` file, an `*.safetensors.index.json`, or a
    /// directory holding exactly one of `model.safetensors.index.json`,
    /// `model.safetensors`, a sole index, or a sole `.safetensors` file.
    pub fn open(gpa: Allocator, io: std.Io, path: []const u8, limits: Limits) !Checkpoint {
        var self: Checkpoint = .{ .storage = .init(gpa), .path = undefined, .shards = &.{}, .locations = .empty };
        errdefer self.deinit(io);
        const arena = self.storage.allocator();
        self.path = try resolve(arena, io, path);
        var names: []const []const u8 = undefined;
        var weight_map: ?std.json.ArrayHashMap([]const u8) = null;
        if (std.mem.endsWith(u8, self.path, ".json")) {
            const map = try readIndex(arena, io, self.path, limits);
            weight_map = map;
            names = try shardNames(arena, map, limits);
        } else names = &.{std.fs.path.basename(self.path)};
        const directory = std.fs.path.dirname(self.path) orelse ".";
        self.shards = try arena.alloc(Shard, names.len);
        var opened: usize = 0;
        // On failure the errdefer's `deinit` closes what `shards` holds: this
        // (declared later, so run first) trims it to the shards opened.
        defer self.shards = self.shards[0..opened];
        for (names, 0..) |name, i| {
            const shard_path = if (weight_map == null) self.path else try std.fs.path.join(arena, &.{ directory, name });
            self.shards[i] = try openShard(gpa, io, shard_path, limits);
            opened += 1;
            for (self.shards[i].document.tensors, 0..) |*t, j| {
                if (weight_map) |map| {
                    const named = map.map.get(t.name) orelse return error.IndexMismatch;
                    if (!std.mem.eql(u8, named, name)) return error.IndexMismatch;
                }
                const inserted = try self.locations.getOrPut(arena, t.name);
                if (inserted.found_existing) return error.DuplicateTensor;
                inserted.value_ptr.* = .{ .shard = @intCast(i), .tensor = @intCast(j) };
            }
        }
        // Every tensor the index names was found in its shard.
        if (weight_map) |map| if (map.map.count() != self.locations.count()) return error.IndexMismatch;
        return self;
    }

    pub fn deinit(self: *Checkpoint, io: std.Io) void {
        for (self.shards) |*s| closeShard(s, io);
        self.storage.deinit();
        self.* = undefined;
    }

    pub fn tensorCount(self: *const Checkpoint) usize {
        return self.locations.count();
    }

    pub fn get(self: *const Checkpoint, name: []const u8) ?Ref {
        const at = self.locations.get(name) orelse return null;
        const shard = &self.shards[at.shard];
        const t = &shard.document.tensors[at.tensor];
        const start: usize = @intCast(shard.document.data_offset + t.offset);
        return .{ .tensor = t, .bytes = shard.mapping.memory[start..][0..@intCast(t.bytes)] };
    }
};

fn openShard(gpa: Allocator, io: std.Io, path: []const u8, limits: Limits) !Checkpoint.Shard {
    const file = try std.Io.Dir.cwd().openFile(io, path, .{});
    errdefer file.close(io);
    const stat = try file.stat(io);
    if (stat.kind != .file) return error.NotRegularFile;
    var buffer: [64 * 1024]u8 = undefined;
    var reader = file.reader(io, &buffer);
    var document = try parse(gpa, &reader.interface, stat.size, limits);
    errdefer document.deinit();
    const len = std.math.cast(usize, stat.size) orelse return error.LimitExceeded;
    const mapping = try file.createMemoryMap(io, .{ .len = len, .protection = .{ .read = true, .write = false }, .populate = false });
    return .{ .path = path, .file = file, .mapping = mapping, .document = document };
}

fn closeShard(shard: *Checkpoint.Shard, io: std.Io) void {
    shard.mapping.destroy(io);
    shard.document.deinit();
    shard.file.close(io);
}

/// The file a checkpoint path names (see `Checkpoint.open`), owned by `arena`.
fn resolve(arena: Allocator, io: std.Io, path: []const u8) ![]const u8 {
    const stat = try std.Io.Dir.cwd().statFile(io, path, .{});
    if (stat.kind != .directory) {
        if (!std.mem.endsWith(u8, path, ".safetensors") and !std.mem.endsWith(u8, path, ".safetensors.index.json")) return error.NotSafetensors;
        return arena.dupe(u8, path);
    }
    for ([_][]const u8{ "model.safetensors.index.json", "model.safetensors" }) |name| {
        const candidate = try std.fs.path.join(arena, &.{ path, name });
        if (std.Io.Dir.cwd().access(io, candidate, .{})) |_| return candidate else |_| {}
    }
    var dir = try std.Io.Dir.cwd().openDir(io, path, .{ .iterate = true });
    defer dir.close(io);
    var index: ?[]const u8 = null;
    var single: ?[]const u8 = null;
    var singles: usize = 0;
    var indexes: usize = 0;
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file) continue;
        if (std.mem.endsWith(u8, entry.name, ".safetensors.index.json")) {
            indexes += 1;
            index = try arena.dupe(u8, entry.name);
        } else if (std.mem.endsWith(u8, entry.name, ".safetensors")) {
            singles += 1;
            single = try arena.dupe(u8, entry.name);
        }
    }
    const name = if (indexes == 1) index.? else if (indexes == 0 and singles == 1) single.? else if (indexes + singles == 0) return error.NoCheckpoint else return error.AmbiguousCheckpoint;
    return std.fs.path.join(arena, &.{ path, name });
}

/// The index's `weight_map` (tensor name → shard file name).
fn readIndex(arena: Allocator, io: std.Io, path: []const u8, limits: Limits) !std.json.ArrayHashMap([]const u8) {
    const bytes = std.Io.Dir.cwd().readFileAlloc(io, path, arena, .limited(@intCast(limits.index_bytes))) catch |err| switch (err) {
        error.StreamTooLong => return error.LimitExceeded,
        else => return err,
    };
    const Index = struct { weight_map: std.json.ArrayHashMap([]const u8) };
    const parsed = std.json.parseFromSliceLeaky(Index, arena, bytes, .{ .ignore_unknown_fields = true, .allocate = .alloc_always }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return error.InvalidIndex,
    };
    if (parsed.weight_map.map.count() > limits.tensor_count) return error.LimitExceeded;
    return parsed.weight_map;
}

/// The distinct shard files an index names, sorted; each a plain file name
/// beside the index (no directories, so an index cannot reach elsewhere).
fn shardNames(arena: Allocator, map: std.json.ArrayHashMap([]const u8), limits: Limits) ![]const []const u8 {
    var names: std.ArrayList([]const u8) = .empty;
    for (map.map.values()) |name| {
        if (name.len == 0 or std.mem.indexOfAny(u8, name, "/\\\x00") != null or std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..") or !std.mem.endsWith(u8, name, ".safetensors")) return error.InvalidIndex;
        for (names.items) |n| {
            if (std.mem.eql(u8, n, name)) break;
        } else {
            if (names.items.len == limits.shard_count) return error.LimitExceeded;
            try names.append(arena, name);
        }
    }
    if (names.items.len == 0) return error.InvalidIndex;
    std.mem.sort([]const u8, names.items, {}, struct {
        fn less(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.less);
    return names.items;
}

// Fixtures are written from the wire format: an 8-byte little-endian length,
// the JSON header, then the byte buffer.
fn fixture(gpa: Allocator, header: []const u8, buffer: []const u8) ![]u8 {
    const out = try gpa.alloc(u8, 8 + header.len + buffer.len);
    std.mem.writeInt(u64, out[0..8], header.len, .little);
    @memcpy(out[8..][0..header.len], header);
    @memcpy(out[8 + header.len ..], buffer);
    return out;
}

fn parseBytes(gpa: Allocator, bytes: []const u8) Error!Document {
    var reader: std.Io.Reader = .fixed(bytes);
    return parse(gpa, &reader, bytes.len, .{});
}

fn expectRejected(expected: anyerror, header: []const u8, buffer_len: usize) !void {
    const gpa = std.testing.allocator;
    const buffer = try gpa.alloc(u8, buffer_len);
    defer gpa.free(buffer);
    @memset(buffer, 0);
    const bytes = try fixture(gpa, header, buffer);
    defer gpa.free(bytes);
    try std.testing.expectError(expected, parseBytes(gpa, bytes));
}

// An unaligned F16 after a 3-element F32, a scalar, and an empty tensor;
// padded with spaces as writers may do.
const valid_header =
    \\{"b":{"dtype":"F16","shape":[2,2],"data_offsets":[12,20]},"__metadata__":{"format":"pt"},
    \\"a":{"dtype":"F32","shape":[3],"data_offsets":[0,12]},"s":{"dtype":"BF16","shape":[],"data_offsets":[20,22]},
    \\"e":{"dtype":"I64","shape":[4,0],"data_offsets":[22,22]}}
++ "   ";

fn validBuffer() [22]u8 {
    var b: [22]u8 = undefined;
    for ([_]f32{ 1, -2, 0.5 }, 0..) |v, i| std.mem.writeInt(u32, b[i * 4 ..][0..4], @bitCast(v), .little);
    for ([_]f16{ 1, 2, 3, -4 }, 0..) |v, i| std.mem.writeInt(u16, b[12 + i * 2 ..][0..2], @bitCast(v), .little);
    std.mem.writeInt(u16, b[20..22], 0x3fc0, .little); // BF16 1.5
    return b;
}

test "read the header, metadata, and tensor ranges without the weights" {
    const gpa = std.testing.allocator;
    const buffer = validBuffer();
    const bytes = try fixture(gpa, valid_header, &buffer);
    defer gpa.free(bytes);
    var doc = try parseBytes(gpa, bytes);
    defer doc.deinit();
    try std.testing.expectEqual(@as(usize, 4), doc.tensors.len);
    try std.testing.expectEqualStrings("b", doc.tensors[0].name);
    try std.testing.expectEqual(@as(u64, 8 + valid_header.len), doc.data_offset);
    const b = doc.find("b").?;
    try std.testing.expectEqual(Dtype.f16, b.dtype);
    try std.testing.expectEqualSlices(u64, &.{ 2, 2 }, b.shape);
    try std.testing.expectEqual(@as(u64, 12), b.offset);
    try std.testing.expectEqual(@as(u64, 1), doc.find("s").?.elements);
    try std.testing.expectEqual(@as(u64, 0), doc.find("e").?.bytes);
    try std.testing.expectEqualStrings("pt", doc.metadataValue("format").?);
    try std.testing.expect(doc.find("missing") == null);
    try std.testing.expectEqualStrings("BF16", Dtype.bf16.name());
}

test "every rejection is typed, and hostile lengths fail before allocating" {
    const t =
        \\"t":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}
    ;
    try expectRejected(error.InvalidHeader, "[]", 0);
    try expectRejected(error.InvalidHeader, "{" ++ t, 8);
    try expectRejected(error.InvalidHeader, "{" ++ t ++ "}{}", 8);
    try expectRejected(error.DuplicateTensor, "{" ++ t ++ "," ++ t ++ "}", 8);
    try expectRejected(error.UnsupportedDtype, "{\"t\":{\"dtype\":\"F4\",\"shape\":[2],\"data_offsets\":[0,1]}}", 1);
    try expectRejected(error.InvalidShape, "{\"t\":{\"dtype\":\"F32\",\"shape\":[3],\"data_offsets\":[0,8]}}", 8);
    try expectRejected(error.InvalidShape, "{\"t\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[8,0]}}", 8);
    try expectRejected(error.InvalidHeader, "{\"t\":{\"dtype\":\"F32\",\"shape\":[-2],\"data_offsets\":[0,8]}}", 8);
    try expectRejected(error.InvalidHeader, "{\"t\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[0,8],\"extra\":1}}", 8);
    try expectRejected(error.InvalidHeader, "{\"t\":{\"dtype\":\"F32\",\"shape\":[2]}}", 8);
    try expectRejected(error.InvalidHeader, "{\"__metadata__\":{\"k\":1}}", 0);
    try expectRejected(error.Overflow, "{\"t\":{\"dtype\":\"F32\",\"shape\":[4294967296,4294967296],\"data_offsets\":[0,8]}}", 8);
    try expectRejected(error.InvalidShape, "{\"t\":{\"dtype\":\"U8\",\"shape\":[1,1,1,1,1,1,1,1,1],\"data_offsets\":[0,1]}}", 1);
    // Tiling: a hole, an overlap, out of bounds, bytes nothing indexes.
    try expectRejected(error.UnindexedBytes, "{\"t\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[4,12]}}", 12);
    try expectRejected(error.OverlappingTensors, "{" ++ t ++ ",\"u\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[4,8]}}", 8);
    try expectRejected(error.TensorOutOfBounds, "{" ++ t ++ "}", 4);
    try expectRejected(error.UnindexedBytes, "{" ++ t ++ "}", 9);

    const gpa = std.testing.allocator;
    var huge: [16]u8 = undefined;
    std.mem.writeInt(u64, huge[0..8], std.math.maxInt(u64), .little);
    try std.testing.expectError(error.LimitExceeded, parseBytes(gpa, &huge));
    std.mem.writeInt(u64, huge[0..8], 100, .little);
    try std.testing.expectError(error.InvalidHeader, parseBytes(gpa, &huge));
    try std.testing.expectError(error.EndOfStream, parseBytes(gpa, huge[0..4]));

    // The rejected tensor is named for a caller that asks.
    const bytes = try fixture(gpa, "{\"weird\":{\"dtype\":\"C64\",\"shape\":[1],\"data_offsets\":[0,8]}}", &[_]u8{0} ** 8);
    defer gpa.free(bytes);
    var rejection: Rejection = .{};
    var reader: std.Io.Reader = .fixed(bytes);
    try std.testing.expectError(error.UnsupportedDtype, parseDiagnosed(gpa, &reader, bytes.len, .{}, &rejection));
    try std.testing.expectEqualStrings("weird", rejection.tensor().?);
}

/// Parses `bytes` as a file of `file_bytes`: a typed error or a document,
/// which is released. A panic, an overflow, or a leak fails the caller.
fn parseAny(bytes: []const u8, file_bytes: u64) void {
    var reader: std.Io.Reader = .fixed(bytes);
    var doc = parse(std.testing.allocator, &reader, file_bytes, .{}) catch return;
    doc.deinit();
}

test "corrupted files return an error or a document, never a panic or a leak" {
    const gpa = std.testing.allocator;
    const buffer = validBuffer();
    const good = try fixture(gpa, valid_header, &buffer);
    defer gpa.free(good);
    const bytes = try gpa.dupe(u8, good);
    defer gpa.free(bytes);
    for (0..good.len) |n| for ([_]u64{ n, std.math.maxInt(u64) }) |claimed| parseAny(good[0..n], claimed);
    // JSON structure and a byte no JSON allows, at every offset; the random
    // pass below covers digits and names.
    for (0..good.len) |at| for ("{}[]\":\xff") |v| {
        @memcpy(bytes, good);
        bytes[at] = v;
        parseAny(bytes, bytes.len);
    };
    for ([_]u64{ 0, 1, 7, valid_header.len - 1, valid_header.len + 1, good.len, std.math.maxInt(u64), 1 << 63 }) |v| {
        @memcpy(bytes, good);
        std.mem.writeInt(u64, bytes[0..8], v, .little);
        parseAny(bytes, bytes.len);
    }
    var prng: std.Random.DefaultPrng = .init(0x7361_6665);
    const random = prng.random();
    const alphabet = "{}[]\":,-.0123456789eE \"abcdtypeshapeF32U8";
    for (0..500) |_| {
        @memcpy(bytes, good);
        for (0..random.intRangeAtMost(usize, 1, 4)) |_| bytes[random.uintLessThan(usize, bytes.len)] = alphabet[random.uintLessThan(usize, alphabet.len)];
        parseAny(bytes, bytes.len);
    }
}

fn parseValid(gpa: Allocator) !void {
    const buffer = validBuffer();
    const bytes = try fixture(gpa, valid_header, &buffer);
    defer gpa.free(bytes);
    var doc = try parseBytes(gpa, bytes);
    doc.deinit();
}

test "cleanup survives every allocation failure" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, parseValid, .{});
}

test "a checkpoint maps a file and decodes unaligned F32, F16, and BF16" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const buffer = validBuffer();
    const bytes = try fixture(gpa, valid_header, &buffer);
    defer gpa.free(bytes);
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors", .data = bytes });
    const dir = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(dir);
    // The directory resolves to its `model.safetensors`.
    var checkpoint = try Checkpoint.open(gpa, io, dir, .{});
    defer checkpoint.deinit(io);
    try std.testing.expectEqual(@as(usize, 4), checkpoint.tensorCount());
    var out: [4]f32 = undefined;
    try checkpoint.get("a").?.decode(0, out[0..3]);
    try std.testing.expectEqualSlices(f32, &.{ 1, -2, 0.5 }, out[0..3]);
    try checkpoint.get("b").?.decode(1, out[0..3]);
    try std.testing.expectEqualSlices(f32, &.{ 2, 3, -4 }, out[0..3]);
    try checkpoint.get("s").?.decode(0, out[0..1]);
    try std.testing.expectEqual(@as(f32, 1.5), out[0]);
    try std.testing.expectError(error.TensorOutOfBounds, checkpoint.get("b").?.decode(2, out[0..3]));
    try std.testing.expectError(error.UnsupportedDtype, checkpoint.get("e").?.decode(0, out[0..0]));
    try std.testing.expect(checkpoint.get("missing") == null);
}

test "an index maps its shards and must agree with them exactly" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var one: [4]u8 = undefined;
    std.mem.writeInt(u32, &one, @bitCast(@as(f32, 7)), .little);
    for ([_][]const u8{ "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors" }, [_][]const u8{ "x", "y" }) |file, name| {
        const header = try std.fmt.allocPrint(gpa, "{{\"{s}\":{{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}}}", .{name});
        defer gpa.free(header);
        const bytes = try fixture(gpa, header, &one);
        defer gpa.free(bytes);
        try tmp.dir.writeFile(io, .{ .sub_path = file, .data = bytes });
    }
    const index =
        \\{"metadata":{"total_size":8},"weight_map":{"x":"model-00001-of-00002.safetensors","y":"model-00002-of-00002.safetensors"}}
    ;
    try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = index });
    const dir = try tmp.dir.realPathFileAlloc(io, ".", gpa);
    defer gpa.free(dir);
    {
        var checkpoint = try Checkpoint.open(gpa, io, dir, .{});
        defer checkpoint.deinit(io);
        try std.testing.expectEqual(@as(usize, 2), checkpoint.shards.len);
        var out: [1]f32 = undefined;
        try checkpoint.get("y").?.decode(0, &out);
        try std.testing.expectEqual(@as(f32, 7), out[0]);
    }
    const path = try std.fs.path.join(gpa, &.{ dir, "model.safetensors.index.json" });
    defer gpa.free(path);
    // A tensor the index places in another shard, one it does not name, one
    // it names that no shard holds, and a shard path leaving the directory.
    for ([_][]const u8{
        \\{"weight_map":{"x":"model-00002-of-00002.safetensors","y":"model-00002-of-00002.safetensors"}}
        ,
        \\{"weight_map":{"w":"model-00001-of-00002.safetensors","y":"model-00002-of-00002.safetensors"}}
        ,
        \\{"weight_map":{"x":"model-00001-of-00002.safetensors","y":"model-00002-of-00002.safetensors","z":"model-00002-of-00002.safetensors"}}
        ,
        \\{"weight_map":{"x":"../model-00001-of-00002.safetensors"}}
        ,
    }, [_]anyerror{ error.IndexMismatch, error.IndexMismatch, error.IndexMismatch, error.InvalidIndex }) |text, expected| {
        try tmp.dir.writeFile(io, .{ .sub_path = "model.safetensors.index.json", .data = text });
        try std.testing.expectError(expected, Checkpoint.open(gpa, io, path, .{}));
    }
    try tmp.dir.writeFile(io, .{ .sub_path = "notes.txt", .data = "x" });
    const notes = try std.fs.path.join(gpa, &.{ dir, "notes.txt" });
    defer gpa.free(notes);
    try std.testing.expectError(error.NotSafetensors, Checkpoint.open(gpa, io, notes, .{}));
    // Two sets in one directory without an index: the caller must name one.
    try tmp.dir.deleteFile(io, "model.safetensors.index.json");
    try std.testing.expectError(error.AmbiguousCheckpoint, Checkpoint.open(gpa, io, dir, .{}));
}
