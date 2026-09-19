//! Architecture-neutral, owned mutable state. Layouts come from model adapters.
//! A failed token poisons the session until reset: partial recurrence must never
//! be mistaken for committed state or rewound by simply changing a KV length.
//!
//! All layer state lives in one page-aligned byte block so a GPU backend can
//! expose it once as a shared buffer; layers borrow typed views of it. The
//! block is bytes rather than floats because an attention layer may store
//! its rows as F16 (`Layout.attention.precision`) while recurrent
//! state stays F32. GPU invariant: `reset`, `fail`, and any CPU read of
//! layer slices are only valid while no submitted GPU work can still write
//! this memory. The synchronous backend guarantees that by waiting on every
//! command buffer; an asynchronous backend must fence before these.
const std = @import("std");

/// Storage precision of an attention layer's cache rows. The CPU reference
/// runtime always uses `f32`; `f16` is a GPU option that
/// halves cache traffic and memory at long context.
pub const Precision = enum {
    f32,
    f16,
    pub fn size(self: Precision) usize {
        return switch (self) {
            .f32 => 4,
            .f16 => 2,
        };
    }
};

pub const Layout = union(enum) {
    attention: struct { key_row: usize, value_row: usize, precision: Precision = .f32 },
    recurrent: struct { history: usize, matrix: usize },
};

/// A typed view of one attention region: `capacity` rows of `row` elements
/// stored at `precision`. `bytes` borrows from `Session.memory`; the view is
/// what a backend binds by byte range and what the CPU reference reads as
/// floats.
pub const Rows = struct {
    bytes: []u8,
    row: usize,
    precision: Precision,

    pub fn rowBytes(self: Rows) usize {
        return self.row * self.precision.size();
    }
    /// Byte range of `count` rows starting at row `first`.
    pub fn range(self: Rows, first: usize, count: usize) []u8 {
        return self.bytes[first * self.rowBytes() ..][0 .. count * self.rowBytes()];
    }
    /// F32 view of `count` rows from `first`. Only an `f32` region has one;
    /// the CPU reference never carves an `f16` layout, so this is an
    /// assertion, not an error.
    pub fn floats(self: Rows, first: usize, count: usize) []f32 {
        std.debug.assert(self.precision == .f32);
        return asFloats(self.range(first, count));
    }
};

pub const Layer = union(enum) {
    attention: struct { keys: Rows, values: Rows },
    recurrent: struct { history: []f32, matrix: []f32 },
};

/// A copy of a session's used state, owned by the caller: each
/// attention layer's rows `[0, position)` and each recurrent layer's whole
/// history and matrix, packed in layer order. `Session.restore` accepts it
/// only into a session of the same capacity and layout (`layout_digest`),
/// which is what makes it a checkpoint rather than a rewind: the recurrent
/// state cannot be recovered by truncating a KV length, so it is copied.
/// Plain heap bytes: a snapshot is never bound to the GPU, so it needs no
/// page alignment.
pub const Snapshot = struct {
    gpa: std.mem.Allocator,
    memory: []u8,
    position: usize,
    capacity: usize,
    layout_digest: u64,

    pub fn deinit(self: *Snapshot) void {
        self.gpa.free(self.memory);
        self.* = undefined;
    }
    /// Bytes the snapshot holds (its used extent, no padding).
    pub fn bytes(self: *const Snapshot) usize {
        return self.memory.len;
    }
};

pub const page = 16384;
/// Every region starts on a 16-byte boundary so its F32 view is aligned and
/// a GPU can bind it for vector loads.
const region_alignment = 16;
/// Upper bound on one session's block.
const max_bytes = 16 * @as(usize, 1024 * 1024 * 1024);

fn asFloats(bytes: []u8) []f32 {
    return @as([*]f32, @ptrCast(@alignCast(bytes.ptr)))[0 .. bytes.len / @sizeOf(f32)];
}

pub const Session = struct {
    gpa: std.mem.Allocator,
    /// One contiguous block; every layer view borrows from it.
    memory: []align(page) u8,
    layers: []Layer,
    capacity: usize,
    /// Hash of the layouts and the capacity; two sessions with equal digests
    /// have byte-compatible state, which `restore` requires.
    layout_digest: u64,
    position: usize = 0,
    status: enum { ready, updating, failed } = .ready,
    /// Whether the caller asked for a checkpoint region. An attention-only
    /// layout yields an empty region but still a valid checkpoint position.
    checkpoint_enabled: bool = false,
    /// One copy of every recurrent layer's history and matrix, packed in
    /// layer order. A page-aligned slice of `memory`, so a GPU sees it in the
    /// same shared buffer; never part of the layout digest or a snapshot.
    checkpoint_region: []u8 = &.{},
    /// The position `checkpoint` recorded, or null when none is live. Cleared
    /// by `reset` and `restore`, whose rewrites make the region stale.
    checkpoint_position: ?usize = null,

    pub fn init(gpa: std.mem.Allocator, layouts: []const Layout, capacity: usize, want_checkpoint: bool) !Session {
        if (capacity == 0 or capacity > 32768 or layouts.len == 0 or layouts.len > 1024) return error.InvalidShape;
        var total: usize = 0;
        for (layouts) |layout| {
            const n = try sizes(layout, capacity);
            total = try std.math.add(usize, total, try std.math.add(usize, aligned(n[0]), aligned(n[1])));
            if (total > max_bytes) return error.LimitExceeded;
        }
        const checkpoint_bytes = if (want_checkpoint) try checkpointBytes(layouts) else 0;
        if (want_checkpoint) {
            // The region starts on its own page so it is GPU-bindable alone.
            total = try std.math.add(usize, std.mem.alignForward(usize, total, page), checkpoint_bytes);
            if (total > max_bytes) return error.LimitExceeded;
        }
        const layers = try gpa.alloc(Layer, layouts.len);
        errdefer gpa.free(layers);
        // Round up to whole pages so a no-copy GPU buffer covers exactly this block.
        const padded = std.mem.alignForward(usize, total, page);
        const memory = try gpa.alignedAlloc(u8, .fromByteUnits(page), padded);
        errdefer gpa.free(memory);
        @memset(memory, 0);
        var cursor: usize = 0;
        for (layouts, layers) |layout, *layer| {
            // Views are exact; the padding up to the next boundary belongs to nobody.
            const n = try sizes(layout, capacity);
            const first = memory[cursor..][0..n[0]];
            cursor += aligned(n[0]);
            const second = memory[cursor..][0..n[1]];
            cursor += aligned(n[1]);
            layer.* = switch (layout) {
                .attention => |x| .{ .attention = .{
                    .keys = .{ .bytes = first, .row = x.key_row, .precision = x.precision },
                    .values = .{ .bytes = second, .row = x.value_row, .precision = x.precision },
                } },
                .recurrent => .{ .recurrent = .{ .history = asFloats(first), .matrix = asFloats(second) } },
            };
        }
        const region: []u8 = if (want_checkpoint) memory[std.mem.alignForward(usize, cursor, page)..][0..checkpoint_bytes] else &.{};
        return .{ .gpa = gpa, .memory = memory, .layers = layers, .capacity = capacity, .layout_digest = digest(layouts, capacity), .checkpoint_enabled = want_checkpoint, .checkpoint_region = region };
    }
    /// Order-sensitive hash of every layout field and the capacity.
    fn digest(layouts: []const Layout, capacity: usize) u64 {
        var hasher = std.hash.Wyhash.init(0xd16);
        hasher.update(std.mem.asBytes(&capacity));
        for (layouts) |layout| {
            hasher.update(std.mem.asBytes(&@intFromEnum(layout)));
            switch (layout) {
                .attention => |x| {
                    hasher.update(std.mem.asBytes(&x.key_row));
                    hasher.update(std.mem.asBytes(&x.value_row));
                    hasher.update(std.mem.asBytes(&@intFromEnum(x.precision)));
                },
                .recurrent => |x| {
                    hasher.update(std.mem.asBytes(&x.history));
                    hasher.update(std.mem.asBytes(&x.matrix));
                },
            }
        }
        return hasher.final();
    }
    fn aligned(n: usize) usize {
        return std.mem.alignForward(usize, n, region_alignment);
    }
    /// Bytes one checkpoint copy of every recurrent layer needs, each region
    /// on the same 16-byte boundary the layer regions use, so `checkpoint`
    /// and `rewind` walk the region identically.
    fn checkpointBytes(layouts: []const Layout) !usize {
        var total: usize = 0;
        for (layouts) |layout| switch (layout) {
            .attention => {},
            .recurrent => |x| for ([_]usize{ x.history, x.matrix }) |count| {
                total = try std.math.add(usize, total, aligned(try std.math.mul(usize, count, @sizeOf(f32))));
            },
        };
        return total;
    }
    /// Exact byte sizes of a layout's two regions, overflow-checked so a
    /// hostile capacity cannot wrap.
    fn sizes(layout: Layout, capacity: usize) ![2]usize {
        const n: [2]usize = switch (layout) {
            .attention => |x| .{
                try std.math.mul(usize, try std.math.mul(usize, x.key_row, capacity), x.precision.size()),
                try std.math.mul(usize, try std.math.mul(usize, x.value_row, capacity), x.precision.size()),
            },
            .recurrent => |x| .{ try std.math.mul(usize, x.history, @sizeOf(f32)), try std.math.mul(usize, x.matrix, @sizeOf(f32)) },
        };
        if (n[0] == 0 or n[1] == 0) return error.InvalidShape;
        return n;
    }
    pub fn deinit(self: *Session) void {
        self.gpa.free(self.layers);
        self.gpa.free(self.memory);
        self.* = undefined;
    }
    /// Bytes the session holds for its layers, page padding included: what a
    /// GPU wraps and what a memory record reports.
    pub fn bytes(self: *const Session) usize {
        return self.memory.len;
    }
    /// Byte offset of a layer region within `memory`, for GPU buffer binding.
    pub fn offsetOf(self: *const Session, region: []const u8) usize {
        const base = @intFromPtr(self.memory.ptr);
        const address = @intFromPtr(region.ptr);
        std.debug.assert(address >= base and address + region.len <= base + self.memory.len);
        return address - base;
    }
    pub fn begin(self: *Session) !void {
        try self.beginChunk(1);
    }
    pub fn commit(self: *Session) !void {
        try self.commitChunk(1);
    }
    /// Admission for `count` positions at once (chunked prefill): the whole
    /// chunk must fit, so a chunk never half-fills the context.
    pub fn beginChunk(self: *Session, count: usize) !void {
        if (self.status != .ready) return error.SessionNotReady;
        if (count == 0) return error.InvalidShape;
        if (count > self.capacity - self.position) return error.ContextFull;
        self.status = .updating;
    }
    pub fn commitChunk(self: *Session, count: usize) !void {
        if (self.status != .updating) return error.SessionNotUpdating;
        self.position += count;
        self.status = .ready;
    }
    pub fn fail(self: *Session) void {
        self.status = .failed;
    }
    /// Whether any layer carries recurrent state, which cannot be rewound by
    /// changing a position alone.
    pub fn hasRecurrent(self: *const Session) bool {
        for (self.layers) |layer| switch (layer) {
            .recurrent => return true,
            .attention => {},
        };
        return false;
    }
    /// Records the current recurrent state and position. A ready session only
    /// (`SessionNotReady` otherwise) that was built with a region
    /// (`NoCheckpointRegion`). The copy is valid on both backends because the
    /// block is quiescent: the synchronous GPU backend has waited.
    pub fn checkpoint(self: *Session) !void {
        if (self.status != .ready) return error.SessionNotReady;
        if (!self.checkpoint_enabled) return error.NoCheckpointRegion;
        var cursor: usize = 0;
        for (self.layers) |layer| switch (layer) {
            .attention => {},
            .recurrent => |r| for ([_][]f32{ r.history, r.matrix }) |values| {
                const slice = std.mem.sliceAsBytes(values);
                @memcpy(self.checkpoint_region[cursor..][0..slice.len], slice);
                cursor += aligned(slice.len);
            },
        };
        std.debug.assert(cursor == self.checkpoint_region.len);
        self.checkpoint_position = self.position;
    }
    /// Returns to the last `checkpoint`: the recurrent copy is restored and
    /// the position set. `NoCheckpoint` without one; `SessionNotReady` on an
    /// updating or failed session.
    pub fn rewind(self: *Session) !void {
        if (self.status != .ready) return error.SessionNotReady;
        const at = self.checkpoint_position orelse return error.NoCheckpoint;
        if (!self.checkpoint_enabled) return error.NoCheckpointRegion;
        var cursor: usize = 0;
        for (self.layers) |layer| switch (layer) {
            .attention => {},
            .recurrent => |r| for ([_][]f32{ r.history, r.matrix }) |values| {
                const slice = std.mem.sliceAsBytes(values);
                @memcpy(slice, self.checkpoint_region[cursor..][0..slice.len]);
                cursor += aligned(slice.len);
            },
        };
        std.debug.assert(cursor == self.checkpoint_region.len);
        self.position = at;
    }
    /// Moves the position back without touching layer memory. Only an
    /// attention-only layout may truncate: a recurrent state is a function of
    /// every token fed, so moving its position alone would lie about it.
    /// `position` must lie between the checkpoint and the current position;
    /// rows past it are left as they are and nothing reads them.
    pub fn truncate(self: *Session, position: usize) !void {
        if (self.status != .ready) return error.SessionNotReady;
        const at = self.checkpoint_position orelse return error.NoCheckpoint;
        if (position < self.position and self.hasRecurrent()) return error.RecurrentStateNotRewindable;
        if (position < at or position > self.position) return error.RewindOutOfRange;
        self.position = position;
    }
    pub fn reset(self: *Session) void {
        @memset(self.memory, 0);
        self.position = 0;
        self.status = .ready;
        self.checkpoint_position = null;
    }

    /// The used extent of each layer, in layer order: attention rows
    /// `[0, position)` of keys then values, recurrent history then matrix.
    fn usedBytes(self: *const Session) usize {
        var total: usize = 0;
        for (self.layers) |layer| total += switch (layer) {
            .attention => |a| a.keys.range(0, self.position).len + a.values.range(0, self.position).len,
            .recurrent => |r| std.mem.sliceAsBytes(r.history).len + std.mem.sliceAsBytes(r.matrix).len,
        };
        return total;
    }
    /// Copies the used state into a caller-owned `Snapshot`. Only a ready
    /// session has committed state worth keeping; an updating or failed one
    /// is refused. GPU invariant as for `reset`: no submitted work may still
    /// write this memory (true after `commit()` on the synchronous backend).
    pub fn snapshot(self: *const Session, gpa: std.mem.Allocator) !Snapshot {
        if (self.status != .ready) return error.SessionNotReady;
        const memory = try gpa.alloc(u8, self.usedBytes());
        errdefer gpa.free(memory);
        var cursor: usize = 0;
        for (self.layers) |layer| switch (layer) {
            .attention => |a| {
                for ([_][]const u8{ a.keys.range(0, self.position), a.values.range(0, self.position) }) |region| {
                    @memcpy(memory[cursor..][0..region.len], region);
                    cursor += region.len;
                }
            },
            .recurrent => |r| {
                for ([_][]const f32{ r.history, r.matrix }) |values| {
                    const region = std.mem.sliceAsBytes(values);
                    @memcpy(memory[cursor..][0..region.len], region);
                    cursor += region.len;
                }
            },
        };
        std.debug.assert(cursor == memory.len);
        return .{ .gpa = gpa, .memory = memory, .position = self.position, .capacity = self.capacity, .layout_digest = self.layout_digest };
    }
    /// Copies a snapshot back and sets the position. The session must be
    /// ready (reset a failed one first) and byte-compatible: same capacity
    /// and layout digest, else `SnapshotMismatch` with the session untouched.
    /// Rows past the restored position are not cleared; nothing reads them.
    /// Same GPU invariant as `snapshot`.
    pub fn restore(self: *Session, snap: *const Snapshot) !void {
        if (self.status != .ready) return error.SessionNotReady;
        if (snap.capacity != self.capacity or snap.layout_digest != self.layout_digest or snap.position > self.capacity) return error.SnapshotMismatch;
        var cursor: usize = 0;
        for (self.layers) |layer| switch (layer) {
            .attention => |a| {
                for ([_][]u8{ a.keys.range(0, snap.position), a.values.range(0, snap.position) }) |region| {
                    if (cursor + region.len > snap.memory.len) return error.SnapshotMismatch;
                    @memcpy(region, snap.memory[cursor..][0..region.len]);
                    cursor += region.len;
                }
            },
            .recurrent => |r| {
                for ([_][]f32{ r.history, r.matrix }) |values| {
                    const region = std.mem.sliceAsBytes(values);
                    if (cursor + region.len > snap.memory.len) return error.SnapshotMismatch;
                    @memcpy(region, snap.memory[cursor..][0..region.len]);
                    cursor += region.len;
                }
            },
        };
        if (cursor != snap.memory.len) return error.SnapshotMismatch;
        self.position = snap.position;
        // The rewrite leaves the region stale: it is only read after a fresh
        // `checkpoint`, and no recorded position may point at it.
        self.checkpoint_position = null;
    }
};

fn snapshotRoundTrip(a: std.mem.Allocator) !void {
    const layouts = [_]Layout{ .{ .attention = .{ .key_row = 2, .value_row = 2, .precision = .f16 } }, .{ .recurrent = .{ .history = 2, .matrix = 2 } } };
    var s = try Session.init(a, &layouts, 3, false);
    defer s.deinit();
    // Two committed positions with distinct bytes in every region.
    try s.beginChunk(2);
    try s.commitChunk(2);
    const attn = s.layers[0].attention;
    @memset(attn.keys.range(0, 3), 0x11);
    @memset(attn.values.range(0, 3), 0x22);
    s.layers[1].recurrent.history[1] = 3;
    s.layers[1].recurrent.matrix[0] = 4;
    var snap = try s.snapshot(a);
    defer snap.deinit();
    // Rows 0..2 of keys and values (4 bytes each), then 8 + 8 bytes of recurrent state.
    try std.testing.expectEqual(@as(usize, 8 + 8 + 16), snap.bytes());
    try std.testing.expectEqual(@as(usize, 2), snap.position);
    // Advance and scribble, then restore: state and position come back, the
    // unused third row is left as it was.
    try s.begin();
    try s.commit();
    @memset(attn.keys.range(0, 3), 0x33);
    s.layers[1].recurrent.matrix[0] = 5;
    try s.restore(&snap);
    try std.testing.expectEqual(@as(usize, 2), s.position);
    try std.testing.expectEqual(@as(u8, 0x11), attn.keys.range(1, 1)[0]);
    try std.testing.expectEqual(@as(u8, 0x33), attn.keys.range(2, 1)[0]);
    try std.testing.expectEqual(@as(f32, 4), s.layers[1].recurrent.matrix[0]);
    try std.testing.expectEqual(@as(f32, 3), s.layers[1].recurrent.history[1]);
    try s.begin();
    try s.commit();
    try std.testing.expectEqual(@as(usize, 3), s.position);
}
test "snapshot copies the used extent and restore brings it back" {
    try snapshotRoundTrip(std.testing.allocator);
    try std.testing.checkAllAllocationFailures(std.testing.allocator, snapshotRoundTrip, .{});
}

test "restore refuses other capacities, other layouts, and unready sessions" {
    const a = std.testing.allocator;
    const layouts = [_]Layout{ .{ .attention = .{ .key_row = 2, .value_row = 2 } }, .{ .recurrent = .{ .history = 2, .matrix = 2 } } };
    var s = try Session.init(a, &layouts, 3, false);
    defer s.deinit();
    try s.begin();
    try s.commit();
    var snap = try s.snapshot(a);
    defer snap.deinit();
    var other_capacity = try Session.init(a, &layouts, 4, false);
    defer other_capacity.deinit();
    try std.testing.expectError(error.SnapshotMismatch, other_capacity.restore(&snap));
    try std.testing.expectEqual(@as(usize, 0), other_capacity.position);
    var other_precision = try Session.init(a, &.{ .{ .attention = .{ .key_row = 2, .value_row = 2, .precision = .f16 } }, .{ .recurrent = .{ .history = 2, .matrix = 2 } } }, 3, false);
    defer other_precision.deinit();
    try std.testing.expectError(error.SnapshotMismatch, other_precision.restore(&snap));
    try std.testing.expect(other_precision.layout_digest != s.layout_digest);
    // A poisoned session neither snapshots nor restores until reset.
    try s.begin();
    s.fail();
    try std.testing.expectError(error.SessionNotReady, s.snapshot(a));
    try std.testing.expectError(error.SessionNotReady, s.restore(&snap));
    try std.testing.expectEqual(@as(usize, 1), s.position);
    s.reset();
    try s.restore(&snap);
    try std.testing.expectEqual(@as(usize, 1), s.position);
    // A snapshot whose byte count does not match the claimed position is refused whole.
    var truncated: Snapshot = .{ .gpa = a, .memory = snap.memory[0 .. snap.memory.len - 1], .position = snap.position, .capacity = snap.capacity, .layout_digest = snap.layout_digest };
    try std.testing.expectError(error.SnapshotMismatch, s.restore(&truncated));
    truncated.position = 3;
    try std.testing.expectError(error.SnapshotMismatch, s.restore(&truncated));
}

fn exercise(a: std.mem.Allocator) !void {
    var s = try Session.init(a, &.{ .{ .attention = .{ .key_row = 2, .value_row = 3 } }, .{ .recurrent = .{ .history = 2, .matrix = 4 } } }, 1, false);
    defer s.deinit();
    try s.begin();
    s.layers[1].recurrent.matrix[0] = 1;
    s.fail();
    try std.testing.expectError(error.SessionNotReady, s.begin());
    s.reset();
    try std.testing.expectEqual(@as(f32, 0), s.layers[1].recurrent.matrix[0]);
    try s.begin();
    try s.commit();
    try std.testing.expectError(error.ContextFull, s.begin());
    // Regions of 8, 12, 8, and 16 bytes each start on a 16-byte boundary.
    try std.testing.expectEqual(@as(usize, 16), s.offsetOf(s.layers[0].attention.values.bytes));
    try std.testing.expectEqual(@as(usize, 48), s.offsetOf(std.mem.sliceAsBytes(s.layers[1].recurrent.matrix)));
    try std.testing.expectEqual(@as(usize, 0), s.memory.len % page);
    try std.testing.expectEqual(@as(usize, page), s.bytes());
}
test "session failures require reset and allocation failures release all layers" {
    try exercise(std.testing.allocator);
    try std.testing.checkAllAllocationFailures(std.testing.allocator, exercise, .{});
}

test "attention views carve rows at the layout's precision" {
    var s = try Session.init(std.testing.allocator, &.{ .{ .attention = .{ .key_row = 4, .value_row = 8, .precision = .f16 } }, .{ .attention = .{ .key_row = 4, .value_row = 8 } } }, 3, false);
    defer s.deinit();
    const half = s.layers[0].attention;
    try std.testing.expectEqual(@as(usize, 8), half.keys.rowBytes());
    try std.testing.expectEqual(@as(usize, 3 * 8), half.keys.bytes.len);
    try std.testing.expectEqual(@as(usize, 3 * 16), half.values.bytes.len);
    try std.testing.expectEqual(@as(usize, 16), half.values.range(1, 1).len);
    try std.testing.expectEqual(@as(usize, 16), s.offsetOf(half.values.range(1, 1)) - s.offsetOf(half.values.bytes));
    const full = s.layers[1].attention;
    try std.testing.expectEqual(@as(usize, 16), full.keys.rowBytes());
    const row = full.keys.floats(2, 1);
    try std.testing.expectEqual(@as(usize, 4), row.len);
    row[3] = 1.5;
    try std.testing.expectEqual(@as(f32, 1.5), full.keys.floats(0, 3)[11]);
    // Byte-count invariant: two 16-byte-aligned regions of each layer.
    try std.testing.expectEqual(@as(usize, 32 + 48), s.offsetOf(full.keys.bytes));
}

test "chunk admission requires the whole chunk to fit" {
    var s = try Session.init(std.testing.allocator, &.{.{ .recurrent = .{ .history = 1, .matrix = 1 } }}, 4, false);
    defer s.deinit();
    try std.testing.expectError(error.InvalidShape, s.beginChunk(0));
    try std.testing.expectError(error.ContextFull, s.beginChunk(5));
    try s.beginChunk(3);
    try std.testing.expectError(error.SessionNotReady, s.beginChunk(1));
    try s.commitChunk(3);
    try std.testing.expectEqual(@as(usize, 3), s.position);
    try std.testing.expectError(error.ContextFull, s.beginChunk(2));
    try s.begin();
    try s.commit();
    try std.testing.expectError(error.ContextFull, s.begin());
}

fn checkpointRoundTrip(a: std.mem.Allocator) !void {
    const layouts = [_]Layout{ .{ .attention = .{ .key_row = 2, .value_row = 2 } }, .{ .recurrent = .{ .history = 2, .matrix = 2 } } };
    var s = try Session.init(a, &layouts, 4, true);
    defer s.deinit();
    // The region sits on its own page, after the 96 bytes of layer regions.
    try std.testing.expectEqual(@as(usize, page), s.offsetOf(s.checkpoint_region));
    try std.testing.expectEqual(@as(usize, 2 * page), s.bytes());
    try s.beginChunk(2);
    try s.commitChunk(2);
    s.layers[1].recurrent.history[0] = 1;
    s.layers[1].recurrent.history[1] = 2;
    s.layers[1].recurrent.matrix[0] = 3;
    s.layers[1].recurrent.matrix[1] = 4;
    try s.checkpoint();
    try std.testing.expectEqual(@as(?usize, 2), s.checkpoint_position);
    // Advance two positions and scribble over both state kinds.
    try s.beginChunk(2);
    try s.commitChunk(2);
    s.layers[1].recurrent.history[0] = 9;
    s.layers[1].recurrent.matrix[1] = 8;
    @memset(s.layers[0].attention.keys.range(0, 4), 0x77);
    try s.rewind();
    try std.testing.expectEqual(@as(usize, 2), s.position);
    try std.testing.expectEqual(@as(f32, 1), s.layers[1].recurrent.history[0]);
    try std.testing.expectEqual(@as(f32, 2), s.layers[1].recurrent.history[1]);
    try std.testing.expectEqual(@as(f32, 3), s.layers[1].recurrent.matrix[0]);
    try std.testing.expectEqual(@as(f32, 4), s.layers[1].recurrent.matrix[1]);
    // Rewind restores recurrent state only; the cache is left for the caller
    // to rewind by position (attention rows past it are ignored).
    try std.testing.expectEqual(@as(u8, 0x77), s.layers[0].attention.keys.range(0, 1)[0]);
    // A later checkpoint re-records the region.
    try s.beginChunk(2);
    try s.commitChunk(2);
    s.layers[1].recurrent.matrix[0] = 5;
    try s.checkpoint();
    s.layers[1].recurrent.matrix[0] = 6;
    try s.rewind();
    try std.testing.expectEqual(@as(f32, 5), s.layers[1].recurrent.matrix[0]);
    try std.testing.expectEqual(@as(usize, 4), s.position);
}
test "checkpoint copies recurrent state and rewind brings it back" {
    try checkpointRoundTrip(std.testing.allocator);
    try std.testing.checkAllAllocationFailures(std.testing.allocator, checkpointRoundTrip, .{});
}

test "checkpoint and rewind obey the session state machine" {
    const a = std.testing.allocator;
    var s = try Session.init(a, &.{.{ .recurrent = .{ .history = 1, .matrix = 1 } }}, 4, false);
    defer s.deinit();
    // No region: checkpoint is refused, and there is nothing to rewind.
    try std.testing.expectError(error.NoCheckpointRegion, s.checkpoint());
    try std.testing.expectError(error.NoCheckpoint, s.rewind());
    // A poisoned session neither checkpoints nor rewinds until reset.
    var with_region = try Session.init(a, &.{.{ .recurrent = .{ .history = 1, .matrix = 1 } }}, 4, true);
    defer with_region.deinit();
    try with_region.begin();
    with_region.fail();
    try std.testing.expectError(error.SessionNotReady, with_region.checkpoint());
    try std.testing.expectError(error.SessionNotReady, with_region.rewind());
    try std.testing.expectEqual(@as(?usize, null), with_region.checkpoint_position);
    with_region.reset();
    // Reset clears a live checkpoint: rewinding now has none to use.
    try with_region.begin();
    try with_region.commit();
    try with_region.checkpoint();
    try std.testing.expect(with_region.checkpoint_position != null);
    with_region.reset();
    try std.testing.expectEqual(@as(?usize, null), with_region.checkpoint_position);
    try std.testing.expectError(error.NoCheckpoint, with_region.rewind());
    // Restore clears it too, even though it rewrites the recurrent state.
    try with_region.begin();
    try with_region.commit();
    try with_region.checkpoint();
    var snap = try with_region.snapshot(a);
    defer snap.deinit();
    try with_region.restore(&snap);
    try std.testing.expectEqual(@as(?usize, null), with_region.checkpoint_position);
    try std.testing.expectError(error.NoCheckpoint, with_region.rewind());
}

test "truncate rewinds attention and refuses recurrent state" {
    const a = std.testing.allocator;
    var attn = try Session.init(a, &.{.{ .attention = .{ .key_row = 2, .value_row = 2 } }}, 8, true);
    defer attn.deinit();
    // The region is empty on an attention-only layout but still records a position.
    try std.testing.expectEqual(@as(usize, 0), attn.checkpoint_region.len);
    try attn.beginChunk(4);
    try attn.commitChunk(4);
    try attn.checkpoint();
    try attn.beginChunk(2);
    try attn.commitChunk(2);
    try attn.truncate(4);
    try std.testing.expectEqual(@as(usize, 4), attn.position);
    // Same position is a no-op; outside [checkpoint, position] is refused.
    try attn.truncate(4);
    try std.testing.expectError(error.RewindOutOfRange, attn.truncate(3));
    try std.testing.expectError(error.RewindOutOfRange, attn.truncate(7));
    // Rewind on attention-only restores the checkpoint position alone.
    try attn.rewind();
    try std.testing.expectEqual(@as(usize, 4), attn.position);

    var rec = try Session.init(a, &.{.{ .recurrent = .{ .history = 1, .matrix = 1 } }}, 8, true);
    defer rec.deinit();
    try rec.beginChunk(4);
    try rec.commitChunk(4);
    try rec.checkpoint();
    try rec.beginChunk(2);
    try rec.commitChunk(2);
    try std.testing.expectError(error.RecurrentStateNotRewindable, rec.truncate(5));
    try std.testing.expectEqual(@as(usize, 6), rec.position);
    // Truncating to the current position is a no-op, not a rewind of state.
    try rec.truncate(6);
    try std.testing.expectError(error.RewindOutOfRange, rec.truncate(7));
}

test "rewind restores the recorded bytes exactly and leaves a peer untouched" {
    const a = std.testing.allocator;
    const layouts = [_]Layout{.{ .recurrent = .{ .history = 3, .matrix = 2 } }};
    var first = try Session.init(a, &layouts, 4, true);
    defer first.deinit();
    var peer = try Session.init(a, &layouts, 4, true);
    defer peer.deinit();
    try first.beginChunk(2);
    try first.commitChunk(2);
    first.layers[0].recurrent.history[0] = 1;
    first.layers[0].recurrent.history[1] = 2;
    first.layers[0].recurrent.history[2] = 3;
    first.layers[0].recurrent.matrix[0] = 4;
    first.layers[0].recurrent.matrix[1] = 5;
    try first.checkpoint();
    const region = try a.dupe(u8, first.checkpoint_region);
    defer a.free(region);
    // The peer runs the same tokens; both stay independent, and only the
    // first rewinds.
    try peer.beginChunk(2);
    try peer.commitChunk(2);
    peer.layers[0].recurrent.history[0] = 7;
    first.layers[0].recurrent.history[0] = 9;
    try first.rewind();
    try std.testing.expectEqualSlices(f32, &.{ 1, 2, 3 }, first.layers[0].recurrent.history[0..3]);
    try std.testing.expectEqualSlices(f32, &.{ 4, 5 }, first.layers[0].recurrent.matrix[0..2]);
    try std.testing.expectEqualSlices(u8, region, first.checkpoint_region);
    try std.testing.expectEqualSlices(f32, &.{ 7, 0, 0 }, peer.layers[0].recurrent.history[0..3]);
}
