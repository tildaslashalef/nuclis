//! Architecture-neutral, owned mutable state. Layouts come from model adapters.
//! A failed token poisons the session until reset: partial recurrence must never
//! be mistaken for committed state or rewound by simply changing a KV length.
//!
//! All layer state lives in one page-aligned byte block so a GPU backend can
//! expose it once as a shared buffer; layers borrow typed views of it. The
//! block is bytes rather than floats because an attention layer may store
//! its rows as F16 or Q8_0 (`Layout.attention.precision`) while recurrent
//! state stays F32. GPU invariant: `reset`, `fail`, and any CPU read of
//! layer slices are only valid while no submitted GPU work can still write
//! this memory. The synchronous backend guarantees that by waiting on every
//! command buffer; an asynchronous backend must fence before these.
const std = @import("std");
const alloc_check = @import("../alloc_check.zig");

/// The most token positions a session holds: the context ceiling every
/// executor, the engine, and the configuration check against. 131,072
/// unless the build sets another (`-Dmax-context`).
pub const max_capacity: usize = @import("build_options").max_context;
/// The longest context the gates and the acceptance workload cover; a
/// longer one runs, unmeasured.
pub const supported_context: usize = 65536;

/// Storage precision of an attention layer's cache rows. The CPU reference
/// runtime always uses `f32`; `f16` and `q8_0` are GPU options that cut
/// cache traffic and memory at long context. `q8_0` is GGML's block of 32
/// values (an F16 scale, then 32 `i8` codes; docs/engine/quantization.md),
/// so its rows are whole blocks.
pub const Precision = enum {
    f32,
    f16,
    q8_0,
    pub const q8_block = 32;
    pub const q8_block_bytes = 34;
    /// Bytes of `elements` values; for `q8_0` a multiple of the block.
    pub fn bytes(self: Precision, elements: usize) usize {
        return switch (self) {
            .f32 => elements * 4,
            .f16 => elements * 2,
            .q8_0 => elements / q8_block * q8_block_bytes,
        };
    }
    /// Whether a row of `elements` values is storable at this precision.
    pub fn fits(self: Precision, elements: usize) bool {
        return self != .q8_0 or elements % q8_block == 0;
    }
};

pub const Layout = union(enum) {
    /// `window`: the layer attends to at most this many rows back, so its
    /// region holds `windowRows` rows rather than the capacity and its base
    /// moves forward (`Rows.base`). Null keeps every row. `room`: the
    /// largest chunk the plan admits, when larger than `room_rows`.
    attention: struct { key_row: usize, value_row: usize, precision: Precision = .f32, window: ?usize = null, room: usize = 0 },
    recurrent: struct { history: usize, matrix: usize },
};

/// Rows a windowed region keeps behind its window for rewinds: the default
/// output budget, so a step's own output stays in reach.
pub const rewind_rows = 8192;
/// Rows a windowed region keeps ahead for the chunk being admitted.
pub const room_rows = 1024;
/// Rows below the window a compaction and a snapshot keep, so a restored
/// state can still step one or two positions back.
pub const window_margin = 16;
/// Rows a windowed region holds at `capacity`.
pub fn windowRows(window: usize, room: usize, capacity: usize) usize {
    return @min(capacity, window + window_margin + rewind_rows + @max(room_rows, room));
}

/// A typed view of one attention region: `slots` rows of `row` elements
/// stored at `precision`, slot 0 holding logical row `base` (0 unless the
/// layer is windowed). `bytes` borrows from `Session.memory`; the view is
/// what a backend binds by byte range and what the CPU reference reads as
/// floats. Callers name logical rows.
pub const Rows = struct {
    bytes: []u8,
    row: usize,
    precision: Precision,
    base: usize = 0,
    window: ?usize = null,

    pub fn rowBytes(self: Rows) usize {
        return self.precision.bytes(self.row);
    }
    /// Rows the region holds.
    pub fn slots(self: Rows) usize {
        return self.bytes.len / self.rowBytes();
    }
    /// Byte range of `count` logical rows starting at row `first`, which
    /// must lie at or past `base` with the range inside the region.
    pub fn range(self: Rows, first: usize, count: usize) []u8 {
        std.debug.assert(first >= self.base and first - self.base + count <= self.slots());
        return self.bytes[(first - self.base) * self.rowBytes() ..][0 .. count * self.rowBytes()];
    }
    /// F32 view of `count` rows from `first`. Only an `f32` region has one;
    /// the CPU reference never carves another layout, so this is an
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
    /// Rows below each windowed layer's window the snapshot holds
    /// (at most `window_margin`): a windowed layer packs rows
    /// `[position -| (window + margin), position)`.
    margin: usize = 0,
    spans: [max_spans]PositionSpan = undefined,
    span_count: usize = 0,
    /// State kept outside the session that belongs to the same point (a
    /// drafter's pending hidden row), owned like `memory`; the session
    /// neither writes nor reads it.
    carried: []f32 = &.{},

    pub fn deinit(self: *Snapshot) void {
        self.gpa.free(self.memory);
        self.gpa.free(self.carried);
        self.* = undefined;
    }
    /// Bytes the snapshot holds (its used extent, no padding).
    pub fn bytes(self: *const Snapshot) usize {
        return self.memory.len;
    }
};

/// A run of rows whose rotary positions do not advance one per row: an
/// image span under multi-axis RoPE occupies `count` rows (a grid of
/// `columns` per line) but moves the text position by `advance` (the larger
/// grid side). Rows after it carry the rotary position
/// `row - Σ (count - advance)` of the spans before them; row `i` inside
/// carries the triple `(t, t + i / columns, t + i % columns)` with `t` the
/// span's own position (`ropeTriple`).
pub const PositionSpan = struct { row: usize, count: usize, advance: usize, columns: usize };
/// Spans a session can hold at once (images across a conversation).
pub const max_spans = 64;

pub const page = 16384;
/// Every region starts on a 16-byte boundary so its F32 view is aligned and
/// a GPU can bind it for vector loads.
const region_alignment = 16;
/// Upper bound on one session's block; past it `init` is `SessionTooLarge`.
pub const max_bytes = 16 * @as(usize, 1024 * 1024 * 1024);

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
    /// Rows the last forward admitted whose recurrent update waits for the
    /// plan's replay (a verify batch that kept a tape): the recurrent layers
    /// still hold the state at `position - pending_rows`. Forwards,
    /// checkpoints, snapshots, and truncation refuse (`ReplayPending`) until
    /// `settleRows`; `rewind`, `reset`, and `restore` drop the rows.
    pending_rows: usize = 0,
    /// Position spans in row order (`PositionSpan`); `span_count` are live.
    spans: [max_spans]PositionSpan = undefined,
    span_count: usize = 0,
    /// `span_count` when `checkpoint` ran, restored by `rewind`.
    checkpoint_span_count: usize = 0,
    /// The lowest position a later `rewindTo` may name (`hold`); a windowed
    /// compaction keeps its window while it fits, and drops the hold when
    /// it does not.
    hold_position: ?usize = null,

    /// Records a span whose rows `[row, row + count)` are about to be
    /// committed; spans are added in row order and never overlap.
    pub fn addSpan(self: *Session, span: PositionSpan) !void {
        if (span.count == 0 or span.advance > span.count or span.row + span.count > self.capacity or span.columns == 0) return error.InvalidShape;
        if (self.span_count == max_spans) return error.TooManySpans;
        if (self.span_count > 0) {
            const last = self.spans[self.span_count - 1];
            if (span.row < last.row + last.count) return error.InvalidShape;
        }
        if (span.row < self.position) return error.InvalidShape;
        self.spans[self.span_count] = span;
        self.span_count += 1;
    }
    /// The live spans, in row order.
    pub fn positionSpans(self: *const Session) []const PositionSpan {
        return self.spans[0..self.span_count];
    }
    /// The rotary position of a row outside every span: the row minus the
    /// rows the spans before it did not advance. Inside a span the caller
    /// positions the row from the span's grid; this returns the span's start.
    pub fn ropePosition(self: *const Session, row: usize) usize {
        var delta: usize = 0;
        for (self.positionSpans()) |span| {
            if (row < span.row) break;
            if (row < span.row + span.count) return span.row - delta;
            delta += span.count - span.advance;
        }
        return row - delta;
    }
    /// The rotary triple `(t, h, w)` of a row: equal components outside a
    /// span (the text position), the span's grid inside.
    pub fn ropeTriple(self: *const Session, row: usize) [3]usize {
        var delta: usize = 0;
        for (self.positionSpans()) |span| {
            if (row < span.row) break;
            if (row < span.row + span.count) {
                const t = span.row - delta;
                const i = row - span.row;
                return .{ t, t + i / span.columns, t + i % span.columns };
            }
            delta += span.count - span.advance;
        }
        return .{ row - delta, row - delta, row - delta };
    }
    /// Drops the spans that start at or after `position`; a span straddling it
    /// is refused (`SpanStraddlesPosition`), since no row of it can stay alone.
    fn truncateSpans(self: *Session, position: usize) !void {
        var keep: usize = 0;
        while (keep < self.span_count) : (keep += 1) {
            const span = self.spans[keep];
            if (span.row >= position) break;
            if (span.row + span.count > position) return error.SpanStraddlesPosition;
        }
        self.span_count = keep;
    }

    pub fn init(gpa: std.mem.Allocator, layouts: []const Layout, capacity: usize, want_checkpoint: bool) !Session {
        if (capacity == 0 or capacity > max_capacity or layouts.len == 0 or layouts.len > 1024) return error.InvalidShape;
        var total: usize = 0;
        for (layouts) |layout| {
            const n = try sizes(layout, capacity);
            total = try std.math.add(usize, total, try std.math.add(usize, aligned(n[0]), aligned(n[1])));
            if (total > max_bytes) return error.SessionTooLarge;
        }
        const checkpoint_bytes = if (want_checkpoint) try checkpointBytes(layouts) else 0;
        if (want_checkpoint) {
            // The region starts on its own page so it is GPU-bindable alone.
            total = try std.math.add(usize, std.mem.alignForward(usize, total, page), checkpoint_bytes);
            if (total > max_bytes) return error.SessionTooLarge;
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
                    .keys = .{ .bytes = first, .row = x.key_row, .precision = x.precision, .window = x.window },
                    .values = .{ .bytes = second, .row = x.value_row, .precision = x.precision, .window = x.window },
                } },
                .recurrent => .{ .recurrent = .{ .history = asFloats(first), .matrix = asFloats(second) } },
            };
        }
        const region_start = std.mem.alignForward(usize, cursor, page);
        const region: []u8 = if (want_checkpoint) memory[region_start..][0..checkpoint_bytes] else &.{};
        return .{
            .gpa = gpa,
            .memory = memory,
            .layers = layers,
            .capacity = capacity,
            .layout_digest = digest(layouts, capacity),
            .checkpoint_enabled = want_checkpoint,
            .checkpoint_region = region,
        };
    }
    /// Order-sensitive hash of every layout field and the capacity.
    fn digest(layouts: []const Layout, capacity: usize) u64 {
        var hasher = std.hash.Wyhash.init(0xd16);
        hasher.update(std.mem.asBytes(&capacity));
        for (layouts) |layout| {
            hasher.update(std.mem.asBytes(&@backingInt(layout)));
            switch (layout) {
                .attention => |x| {
                    hasher.update(std.mem.asBytes(&x.key_row));
                    hasher.update(std.mem.asBytes(&x.value_row));
                    hasher.update(std.mem.asBytes(&@backingInt(x.precision)));
                    if (x.window) |w| {
                        const rows = windowRows(w, x.room, capacity);
                        hasher.update(std.mem.asBytes(&w));
                        hasher.update(std.mem.asBytes(&rows));
                    }
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
            .attention => |x| blk: {
                if (!x.precision.fits(x.key_row) or !x.precision.fits(x.value_row)) return error.InvalidShape;
                if (x.window) |w| if (w == 0) return error.InvalidShape;
                const rows = if (x.window) |w| windowRows(w, x.room, capacity) else capacity;
                break :blk .{
                    try std.math.mul(usize, x.precision.bytes(x.key_row), rows),
                    try std.math.mul(usize, x.precision.bytes(x.value_row), rows),
                };
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
        if (self.pending_rows > 0) return error.ReplayPending;
        if (count == 0) return error.InvalidShape;
        if (count > self.capacity - self.position) return error.ContextFull;
        try self.compact(count);
        self.status = .updating;
    }
    /// Makes room for `count` more rows in every windowed region: when the
    /// region would overflow, rows from the lowest of the position, the
    /// hold, and the checkpoint, less the window and `window_margin`, move
    /// to slot 0 and `base` follows. The hold is dropped when keeping it
    /// leaves no room; a chunk that still does not fit (an image span
    /// larger than the region's slack) is `ContextFull`. A CPU memmove of
    /// the shared block, valid under the GPU invariant above.
    fn compact(self: *Session, count: usize) !void {
        for (self.layers) |*layer| switch (layer.*) {
            .recurrent => {},
            .attention => |*a| {
                const w = a.keys.window orelse continue;
                const slots = a.keys.slots();
                if (self.position + count - a.keys.base <= slots) continue;
                var keep = self.keepFrom(w, true, a.keys.base);
                if (self.position + count - keep > slots) {
                    self.hold_position = null;
                    keep = self.keepFrom(w, false, a.keys.base);
                }
                if (self.position + count - keep > slots) return error.ContextFull;
                for ([_]*Rows{ &a.keys, &a.values }) |rows| {
                    const n = (self.position - keep) * rows.rowBytes();
                    const from = (keep - rows.base) * rows.rowBytes();
                    std.mem.copyForwards(u8, rows.bytes[0..n], rows.bytes[from..][0..n]);
                    rows.base = keep;
                }
            },
        };
    }
    /// The first row a compaction keeps for a window of `w`, at `base` or later.
    fn keepFrom(self: *const Session, w: usize, with_hold: bool, base: usize) usize {
        var low = self.position;
        if (with_hold) if (self.hold_position) |h| {
            low = @min(low, h);
        };
        if (self.checkpoint_position) |c| low = @min(low, c);
        return @max(base, low -| (w + window_margin));
    }
    /// Whether every windowed region still holds the rows a session moved
    /// back to `position` attends to.
    fn reaches(self: *const Session, position: usize) bool {
        for (self.layers) |layer| switch (layer) {
            .recurrent => {},
            .attention => |a| if (a.keys.window) |w| {
                if (position -| w < a.keys.base) return false;
            },
        };
        return true;
    }
    /// Asks windowed compactions to keep `position` in reach of `rewindTo`
    /// while they can; the latest hold replaces an earlier one.
    pub fn hold(self: *Session, position: usize) void {
        self.hold_position = position;
    }
    pub fn release(self: *Session) void {
        self.hold_position = null;
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
        if (self.pending_rows > 0) return error.ReplayPending;
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
        self.checkpoint_span_count = self.span_count;
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
        self.span_count = self.checkpoint_span_count;
        self.pending_rows = 0;
    }
    /// Marks the last forward's final `count` rows as pending: the plan kept
    /// their recurrent update on a tape instead of applying it.
    pub fn deferRows(self: *Session, count: usize) !void {
        if (self.status != .ready) return error.SessionNotReady;
        if (count == 0 or count > self.position) return error.InvalidShape;
        self.pending_rows = count;
    }
    /// Settles the pending rows after the plan replayed the first `kept` of
    /// them into the recurrent layers: the position drops the rest, as a
    /// rejected suffix of a verify batch. `kept` may be zero (nothing
    /// replayed, the state is the batch's start).
    pub fn settleRows(self: *Session, kept: usize) !void {
        if (self.status != .ready) return error.SessionNotReady;
        if (self.pending_rows == 0 or kept > self.pending_rows) return error.InvalidShape;
        const position = self.position - self.pending_rows + kept;
        try self.truncateSpans(position);
        self.position = position;
        self.pending_rows = 0;
    }
    /// Moves the position back without touching layer memory. Only an
    /// attention-only layout may truncate: a recurrent state is a function of
    /// every token fed, so moving its position alone would lie about it.
    /// `position` must lie between the checkpoint and the current position;
    /// rows past it are left as they are and nothing reads them.
    pub fn truncate(self: *Session, position: usize) !void {
        if (self.status != .ready) return error.SessionNotReady;
        if (self.pending_rows > 0) return error.ReplayPending;
        const at = self.checkpoint_position orelse return error.NoCheckpoint;
        if (position < self.position and self.hasRecurrent()) return error.RecurrentStateNotRewindable;
        if (position < at or position > self.position or !self.reaches(position)) return error.RewindOutOfRange;
        try self.truncateSpans(position);
        self.position = position;
    }
    /// Moves an attention-only session back to any earlier position: rows
    /// `[0, position)` do not depend on later ones, so the state is exactly
    /// the one after `position` tokens. Clears the checkpoint, which may lie
    /// past it. `RecurrentStateNotRewindable` with a recurrent layer.
    pub fn rewindTo(self: *Session, position: usize) !void {
        if (self.status != .ready) return error.SessionNotReady;
        if (self.pending_rows > 0) return error.ReplayPending;
        if (self.hasRecurrent()) return error.RecurrentStateNotRewindable;
        if (position > self.position or !self.reaches(position)) return error.RewindOutOfRange;
        try self.truncateSpans(position);
        self.position = position;
        self.checkpoint_position = null;
    }
    pub fn reset(self: *Session) void {
        @memset(self.memory, 0);
        self.position = 0;
        self.status = .ready;
        self.checkpoint_position = null;
        self.pending_rows = 0;
        self.span_count = 0;
        self.checkpoint_span_count = 0;
        self.hold_position = null;
        self.setBases(0, 0);
    }
    /// Sets every windowed region's base to `position -| (window + margin)`.
    fn setBases(self: *Session, position: usize, margin: usize) void {
        for (self.layers) |*layer| switch (layer.*) {
            .recurrent => {},
            .attention => |*a| if (a.keys.window) |w| {
                a.keys.base = position -| (w + margin);
                a.values.base = a.keys.base;
            },
        };
    }
    /// The logical rows a snapshot at `position` with `margin` packs for a layer.
    fn packedRows(a: anytype, position: usize, margin: usize) [2]usize {
        const first = if (a.keys.window) |w| position -| (w + margin) else 0;
        return .{ first, position - first };
    }
    /// The margin a snapshot can keep: `window_margin`, or less where a
    /// region's base already sits closer to its window.
    fn snapshotMargin(self: *const Session) usize {
        var margin: usize = window_margin;
        for (self.layers) |layer| switch (layer) {
            .recurrent => {},
            .attention => |a| if (a.keys.window) |w| {
                margin = @min(margin, (self.position -| w) -| a.keys.base);
            },
        };
        return margin;
    }

    /// The used extent of each layer, in layer order: attention rows
    /// `[0, position)` of keys then values, recurrent history then matrix.
    fn usedBytes(self: *const Session, margin: usize) usize {
        var total: usize = 0;
        for (self.layers) |layer| total += switch (layer) {
            .attention => |a| blk: {
                const r = packedRows(a, self.position, margin);
                break :blk a.keys.range(r[0], r[1]).len + a.values.range(r[0], r[1]).len;
            },
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
        if (self.pending_rows > 0) return error.ReplayPending;
        const margin = self.snapshotMargin();
        const memory = try gpa.alloc(u8, self.usedBytes(margin));
        errdefer gpa.free(memory);
        var cursor: usize = 0;
        for (self.layers) |layer| switch (layer) {
            .attention => |a| {
                const r = packedRows(a, self.position, margin);
                for ([_][]const u8{ a.keys.range(r[0], r[1]), a.values.range(r[0], r[1]) }) |region| {
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
        return .{ .gpa = gpa, .memory = memory, .position = self.position, .capacity = self.capacity, .layout_digest = self.layout_digest, .margin = margin, .spans = self.spans, .span_count = self.span_count };
    }
    /// Copies a snapshot back and sets the position. The session must be
    /// ready (reset a failed one first) and byte-compatible: same capacity
    /// and layout digest, else `SnapshotMismatch` with the session untouched.
    /// Rows past the restored position are not cleared; nothing reads them.
    /// Same GPU invariant as `snapshot`.
    pub fn restore(self: *Session, snap: *const Snapshot) !void {
        if (self.status != .ready) return error.SessionNotReady;
        if (snap.capacity != self.capacity or snap.layout_digest != self.layout_digest or snap.position > self.capacity or snap.margin > window_margin) return error.SnapshotMismatch;
        // Checked whole before a byte moves, so a mismatch leaves the session untouched.
        var expected: usize = 0;
        for (self.layers) |layer| expected += switch (layer) {
            .attention => |a| packedRows(a, snap.position, snap.margin)[1] * (a.keys.rowBytes() + a.values.rowBytes()),
            .recurrent => |r| std.mem.sliceAsBytes(r.history).len + std.mem.sliceAsBytes(r.matrix).len,
        };
        if (expected != snap.memory.len) return error.SnapshotMismatch;
        self.setBases(snap.position, snap.margin);
        var cursor: usize = 0;
        for (self.layers) |layer| switch (layer) {
            .attention => |a| {
                const r = packedRows(a, snap.position, snap.margin);
                for ([_][]u8{ a.keys.range(r[0], r[1]), a.values.range(r[0], r[1]) }) |region| {
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
        self.spans = snap.spans;
        self.span_count = snap.span_count;
        // The rewrite leaves the region stale: it is only read after a fresh
        // `checkpoint`, and no recorded position may point at it.
        self.checkpoint_position = null;
        self.pending_rows = 0;
        self.hold_position = null;
    }
};

test "position spans: rope positions, truncation, rewind, and snapshots" {
    const a = std.testing.allocator;
    const layouts = [_]Layout{.{ .attention = .{ .key_row = 2, .value_row = 2, .precision = .f32 } }};
    var s = try Session.init(a, &layouts, 64, true);
    defer s.deinit();
    // Four text rows, a checkpoint, a 12-row image advancing by 4, then text.
    try s.beginChunk(4);
    try s.commitChunk(4);
    try s.checkpoint();
    try s.addSpan(.{ .row = 4, .count = 12, .advance = 4, .columns = 4 });
    try std.testing.expectError(error.InvalidShape, s.addSpan(.{ .row = 10, .count = 2, .advance = 1, .columns = 1 }));
    try s.beginChunk(14);
    try s.commitChunk(14);
    try std.testing.expectEqual(@as(usize, 3), s.ropePosition(3));
    try std.testing.expectEqual(@as(usize, 4), s.ropePosition(4));
    try std.testing.expectEqual(@as(usize, 4), s.ropePosition(15));
    try std.testing.expectEqual(@as(usize, 8), s.ropePosition(16));
    try std.testing.expectEqual(@as(usize, 9), s.ropePosition(17));
    try std.testing.expectEqual([3]usize{ 4, 5, 6 }, s.ropeTriple(10));
    try std.testing.expectEqual([3]usize{ 4, 4, 4 }, s.ropeTriple(4));
    try std.testing.expectEqual([3]usize{ 8, 8, 8 }, s.ropeTriple(16));
    try std.testing.expectError(error.SpanStraddlesPosition, s.truncate(10));
    var snap = try s.snapshot(a);
    defer snap.deinit();
    try s.addSpan(.{ .row = 18, .count = 6, .advance = 2, .columns = 3 });
    try s.beginChunk(6);
    try s.commitChunk(6);
    try std.testing.expectEqual(@as(usize, 12), s.ropePosition(24));
    try s.rewind();
    try std.testing.expectEqual(@as(usize, 4), s.position);
    try std.testing.expectEqual(@as(usize, 0), s.positionSpans().len);
    try s.restore(&snap);
    try std.testing.expectEqual(@as(usize, 1), s.positionSpans().len);
    try std.testing.expectEqual(@as(usize, 8), s.ropePosition(16));
    s.reset();
    try std.testing.expectEqual(@as(usize, 0), s.positionSpans().len);
}

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
    try alloc_check.checkAll(std.testing.allocator, snapshotRoundTrip, .{});
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
    try alloc_check.checkAll(std.testing.allocator, exercise, .{});
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

test "q8_0 rows are whole 34-byte blocks" {
    var s = try Session.init(std.testing.allocator, &.{.{ .attention = .{ .key_row = 64, .value_row = 32, .precision = .q8_0 } }}, 3, false);
    defer s.deinit();
    const q = s.layers[0].attention;
    try std.testing.expectEqual(@as(usize, 68), q.keys.rowBytes());
    try std.testing.expectEqual(@as(usize, 3 * 68), q.keys.bytes.len);
    try std.testing.expectEqual(@as(usize, 34), q.values.range(2, 1).len);
    try std.testing.expectEqual(@as(usize, 208), s.offsetOf(q.values.bytes)); // 3 × 68 bytes, 16-aligned
    try std.testing.expectError(error.InvalidShape, Session.init(std.testing.allocator, &.{.{ .attention = .{ .key_row = 48, .value_row = 32, .precision = .q8_0 } }}, 3, false));
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
    try alloc_check.checkAll(std.testing.allocator, checkpointRoundTrip, .{});
}

test "rewindTo moves an attention-only session back anywhere, never a recurrent one" {
    const a = std.testing.allocator;
    const attention = [_]Layout{.{ .attention = .{ .key_row = 2, .value_row = 2 } }};
    var s = try Session.init(a, &attention, 8, true);
    defer s.deinit();
    try s.beginChunk(5);
    try s.commitChunk(5);
    try s.checkpoint();
    try std.testing.expectError(error.RewindOutOfRange, s.rewindTo(6));
    // Before the checkpoint, where `truncate` refuses.
    try std.testing.expectError(error.RewindOutOfRange, s.truncate(2));
    try s.rewindTo(2);
    try std.testing.expectEqual(@as(usize, 2), s.position);
    try std.testing.expectEqual(@as(?usize, null), s.checkpoint_position);
    try s.rewindTo(0);
    try std.testing.expectEqual(@as(usize, 0), s.position);

    const mixed = [_]Layout{ .{ .attention = .{ .key_row = 2, .value_row = 2 } }, .{ .recurrent = .{ .history = 2, .matrix = 2 } } };
    var r = try Session.init(a, &mixed, 8, false);
    defer r.deinit();
    try r.beginChunk(3);
    try r.commitChunk(3);
    try std.testing.expectError(error.RecurrentStateNotRewindable, r.rewindTo(1));
    try std.testing.expectEqual(@as(usize, 3), r.position);
}

test "pending rows refuse until settled or rewound" {
    const a = std.testing.allocator;
    const layouts = [_]Layout{
        .{ .attention = .{ .key_row = 2, .value_row = 2 } },
        .{ .recurrent = .{ .history = 2, .matrix = 2 } },
    };
    var s = try Session.init(a, &layouts, 8, true);
    defer s.deinit();
    try s.beginChunk(1);
    try s.commitChunk(1);
    try s.checkpoint();
    try s.beginChunk(4);
    try s.commitChunk(4);
    try std.testing.expectError(error.InvalidShape, s.deferRows(6));
    try s.deferRows(4);
    try std.testing.expectError(error.ReplayPending, s.begin());
    try std.testing.expectError(error.ReplayPending, s.checkpoint());
    try std.testing.expectError(error.ReplayPending, s.snapshot(a));
    try std.testing.expectError(error.ReplayPending, s.truncate(2));
    try std.testing.expectError(error.InvalidShape, s.settleRows(5));
    // Two of the four rows kept: the position drops the other two.
    try s.settleRows(2);
    try std.testing.expectEqual(@as(usize, 3), s.position);
    try std.testing.expectError(error.InvalidShape, s.settleRows(0));
    try s.begin();
    try s.commit();
    // A rewind drops pending rows with the rest of the batch.
    try s.beginChunk(2);
    try s.commitChunk(2);
    try s.deferRows(2);
    try s.rewind();
    try std.testing.expectEqual(@as(usize, 1), s.position);
    try std.testing.expectEqual(@as(usize, 0), s.pending_rows);
    // Zero kept rows return to the batch's start.
    try s.beginChunk(3);
    try s.commitChunk(3);
    try s.deferRows(3);
    try s.settleRows(0);
    try std.testing.expectEqual(@as(usize, 1), s.position);
    s.reset();
    try std.testing.expectEqual(@as(usize, 0), s.pending_rows);
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

fn windowedSession(a: std.mem.Allocator, capacity: usize) !Session {
    // A windowed layer of window 4 beside a full one, one f32 per row.
    const layouts = [_]Layout{ .{ .attention = .{ .key_row = 1, .value_row = 1, .window = 4, .room = 2 } }, .{ .attention = .{ .key_row = 1, .value_row = 1 } } };
    return Session.init(a, &layouts, capacity, true);
}
/// Feeds `count` rows one chunk at a time, row `p` of every region holding `p`.
fn feedRows(s: *Session, count: usize, chunk: usize) !void {
    var left = count;
    while (left > 0) {
        const n = @min(chunk, left);
        try s.beginChunk(n);
        for (s.layers) |layer| for ([_]Rows{ layer.attention.keys, layer.attention.values }) |rows| {
            for (rows.floats(s.position, n), s.position..) |*v, p| v.* = @floatFromInt(p);
        };
        try s.commitChunk(n);
        left -= n;
    }
}
fn expectRowsHold(s: *const Session, first: usize, end: usize) !void {
    for (s.layers) |layer| for ([_]Rows{ layer.attention.keys, layer.attention.values }) |rows| {
        for (rows.floats(first, end - first), first..) |v, p| try std.testing.expectEqual(@as(f32, @floatFromInt(p)), v);
    };
}

test "a windowed region compacts, keeps its hold while it fits, and refuses rewinds past its reach" {
    const a = std.testing.allocator;
    const slots = windowRows(4, 2, max_capacity);
    try std.testing.expectEqual(@as(usize, 4 + window_margin + rewind_rows + room_rows), slots);
    var s = try windowedSession(a, max_capacity);
    defer s.deinit();
    try std.testing.expectEqual(slots, s.layers[0].attention.keys.slots());
    try std.testing.expectEqual(@as(usize, max_capacity), s.layers[1].attention.keys.slots());
    // Without a hold: the base moves to the window and margin behind the position.
    try feedRows(&s, slots + 10, 7);
    const base = s.layers[0].attention.keys.base;
    try std.testing.expect(base > 0 and base <= s.position - 4 - window_margin);
    try std.testing.expectEqual(base, s.layers[0].attention.values.base);
    try std.testing.expectEqual(@as(usize, 0), s.layers[1].attention.keys.base);
    try expectRowsHold(&s, base, s.position);
    // A rewind inside the reach works; one row past it is refused.
    try s.rewindTo(base + 4);
    try std.testing.expectError(error.RewindOutOfRange, s.rewindTo(base + 3));
    try expectRowsHold(&s, base, s.position);
    // A hold keeps its rows through the next compaction while they fit.
    s.reset();
    try std.testing.expectEqual(@as(usize, 0), s.layers[0].attention.keys.base);
    try feedRows(&s, 100, 7);
    s.hold(100);
    try feedRows(&s, slots - 50, 7);
    try std.testing.expect(s.hold_position != null);
    try std.testing.expect(s.layers[0].attention.keys.base <= 100 - 4 - window_margin);
    try s.rewindTo(100);
    try expectRowsHold(&s, 96, 100);
    // A hold that no longer fits is dropped, not the chunk.
    s.hold(100);
    try feedRows(&s, slots, 7);
    try std.testing.expect(s.hold_position == null);
    try std.testing.expectError(error.RewindOutOfRange, s.rewindTo(100));
    s.release();
}

test "a windowed region keeps its checkpoint, truncates across a compaction, and snapshots its window" {
    const a = std.testing.allocator;
    const slots = windowRows(4, 2, max_capacity);
    var s = try windowedSession(a, max_capacity);
    defer s.deinit();
    try feedRows(&s, slots - 3, 7);
    // A verify-shaped batch crosses a compaction; truncate back to its start.
    try s.checkpoint();
    const start = s.position;
    try feedRows(&s, 6, 6);
    try std.testing.expect(s.layers[0].attention.keys.base > 0);
    try s.truncate(start + 2);
    try std.testing.expectEqual(start + 2, s.position);
    try expectRowsHold(&s, start - 4, start + 2);
    // A snapshot holds the window and the margin; restore puts the base back.
    var snap = try s.snapshot(a);
    defer snap.deinit();
    try std.testing.expectEqual(@as(usize, window_margin), snap.margin);
    const window_bytes = (4 + window_margin) * 2 * 4;
    try std.testing.expectEqual(window_bytes + s.position * 2 * 4, snap.bytes());
    s.reset();
    try s.restore(&snap);
    try std.testing.expectEqual(snap.position, s.position);
    try std.testing.expectEqual(s.position - 4 - window_margin, s.layers[0].attention.keys.base);
    try expectRowsHold(&s, s.position - 4 - window_margin, s.position);
    // Restored, the session can step two rows back and go on.
    try s.rewindTo(s.position - 2);
    try feedRows(&s, 3, 3);
    try expectRowsHold(&s, s.layers[0].attention.keys.base, s.position);
    // The window and its row count are part of the layout digest.
    var plain = try Session.init(a, &.{ .{ .attention = .{ .key_row = 1, .value_row = 1 } }, .{ .attention = .{ .key_row = 1, .value_row = 1 } } }, max_capacity, true);
    defer plain.deinit();
    try std.testing.expect(plain.layout_digest != s.layout_digest);
    var wider = try Session.init(a, &.{ .{ .attention = .{ .key_row = 1, .value_row = 1, .window = 5, .room = 2 } }, .{ .attention = .{ .key_row = 1, .value_row = 1 } } }, max_capacity, true);
    defer wider.deinit();
    try std.testing.expectError(error.SnapshotMismatch, wider.restore(&snap));
    // A chunk larger than the region's slack is refused, not admitted half.
    s.reset();
    try feedRows(&s, 100, 7);
    try std.testing.expectError(error.ContextFull, s.beginChunk(slots - 4));
    try std.testing.expectEqual(@as(usize, 100), s.position);
    try s.beginChunk(1);
    s.fail();
    try std.testing.expectError(error.InvalidShape, Session.init(a, &.{.{ .attention = .{ .key_row = 1, .value_row = 1, .window = 0 } }}, 64, false));
}

test "a session past the byte limit is refused by name" {
    // Qwen3.8's 16 attention layers (1,024-wide keys and values) at the
    // release ceiling with an F32 cache are 16 GiB of rows; its recurrent
    // state takes the block past the limit.
    var layouts: [17]Layout = @splat(.{ .attention = .{ .key_row = 1024, .value_row = 1024, .precision = .f32 } });
    layouts[16] = .{ .recurrent = .{ .history = 1 << 20, .matrix = 1 << 20 } };
    if (max_capacity < 131072) return error.SkipZigTest;
    if (Session.init(std.testing.allocator, &layouts, 131072, false)) |ok| {
        var unexpected = ok;
        unexpected.deinit();
        return error.TestExpectedError;
    } else |err| try std.testing.expectEqual(error.SessionTooLarge, err);
}
