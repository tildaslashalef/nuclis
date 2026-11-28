//! `inspect --context`: the memory a file needs at a context length, from
//! its tensors alone. Weights are the tensors' bytes; the cache is, per
//! token, the output widths of each block's own K and V projections,
//! stored at the engine's cache precisions (`inference.session.Precision`).
//! A block without its own K/V (shared KV) adds nothing; a recurrent block's
//! state is constant per sequence and is not counted. A declared sliding
//! window is not applied (which blocks slide is the adapter's knowledge),
//! so the cache is an upper bound whenever one is declared. Pure, except
//! `machineMemory`.
const std = @import("std");
const builtin = @import("builtin");
const inference = @import("inference");
const compare = @import("inventory.zig");
const kit = @import("ui.zig");

const Precision = inference.session.Precision;

pub const Row = struct {
    context: u64,
    cache_f16: u64,
    cache_q8_0: u64,
    total_f16: u64,
    total_q8_0: u64,
};

pub const Memory = struct {
    weights: u64,
    /// Blocks with their own K and V projections, and the values one token
    /// adds to all of them together.
    cache_blocks: u32,
    values_per_token: u64,
    /// Blocks holding recurrent state (`ssm_*` tensors), not counted.
    recurrent_blocks: u32,
    /// The declared window, when the file names one: the rows are then an
    /// upper bound.
    sliding_window: ?u64,
    declared_context: ?u64,
    rows: []const Row,
    /// This machine's memory (macOS `hw.memsize`); null where unknown.
    machine: ?u64,
};

/// The most context lengths one `--context` lists.
pub const max_contexts = 8;

/// `spec` is a comma-separated list of token counts and `max` (the declared
/// context length). Fails with `InvalidContext` on anything else, a zero,
/// `max` without a declared length, or more than `max_contexts` items.
pub fn estimate(arena: std.mem.Allocator, inv: compare.Inventory, weights: u64, spec: []const u8, machine: ?u64) !Memory {
    const declared = declaredContext(inv);
    var contexts: std.ArrayList(u64) = .empty;
    var it = std.mem.splitScalar(u8, spec, ',');
    while (it.next()) |raw| {
        const item = std.mem.trim(u8, raw, " ");
        if (contexts.items.len == max_contexts) return error.InvalidContext;
        const n = if (std.mem.eql(u8, item, "max")) declared orelse return error.InvalidContext else std.fmt.parseInt(u64, item, 10) catch return error.InvalidContext;
        if (n == 0) return error.InvalidContext;
        try contexts.append(arena, n);
    }
    if (contexts.items.len == 0) return error.InvalidContext;

    // Per block: the K and V output widths, and whether it is recurrent.
    const Block = struct { k: u64 = 0, v: u64 = 0, recurrent: bool = false };
    var blocks: std.AutoArrayHashMapUnmanaged(u32, Block) = .empty;
    for (inv.tensors) |t| {
        const f = try compare.fold(arena, t.name, null);
        const block = f.block orelse continue;
        const b = try blocks.getOrPutValue(arena, block, .{});
        if (isProjection(f.name, .k)) b.value_ptr.k += outputWidth(t.shape, inv.outermost_first);
        if (isProjection(f.name, .v)) b.value_ptr.v += outputWidth(t.shape, inv.outermost_first);
        if (std.mem.indexOf(u8, f.name, ".ssm_") != null or std.mem.indexOf(u8, f.name, ".linear_attn.") != null) b.value_ptr.recurrent = true;
    }
    var cache_blocks: u32 = 0;
    var recurrent: u32 = 0;
    var per_token_f16: u64 = 0;
    var per_token_q8: u64 = 0;
    var values: u64 = 0;
    for (blocks.values()) |b| {
        if (b.recurrent) recurrent += 1;
        if (b.k == 0) continue;
        cache_blocks += 1;
        values += b.k + b.v;
        per_token_f16 += Precision.f16.bytes(@intCast(b.k)) + Precision.f16.bytes(@intCast(b.v));
        per_token_q8 += Precision.q8_0.bytes(@intCast(b.k)) + Precision.q8_0.bytes(@intCast(b.v));
    }
    const rows = try arena.alloc(Row, contexts.items.len);
    for (contexts.items, rows) |n, *r| {
        const half = try std.math.mul(u64, n, per_token_f16);
        const q8 = try std.math.mul(u64, n, per_token_q8);
        r.* = .{ .context = n, .cache_f16 = half, .cache_q8_0 = q8, .total_f16 = try std.math.add(u64, weights, half), .total_q8_0 = try std.math.add(u64, weights, q8) };
    }
    return .{
        .weights = weights,
        .cache_blocks = cache_blocks,
        .values_per_token = values,
        .recurrent_blocks = recurrent,
        .sliding_window = slidingWindow(inv),
        .declared_context = declared,
        .rows = rows,
        .machine = machine,
    };
}

const Projection = enum { k, v };

/// `attn_k.weight` / `self_attn.k_proj.weight` and their V twins; a fused
/// QKV is not a cache (Qwen 3.5's is the recurrent block's).
fn isProjection(folded: []const u8, which: Projection) bool {
    const names: []const []const u8 = switch (which) {
        .k => &.{ ".attn_k.weight", ".k_proj.weight" },
        .v => &.{ ".attn_v.weight", ".v_proj.weight" },
    };
    for (names) |n| if (std.mem.endsWith(u8, folded, n)) return true;
    return false;
}

/// A projection's output width: GGUF stores the innermost (input) dimension
/// first, safetensors the outermost (output) first.
fn outputWidth(shape: []const u64, outermost_first: bool) u64 {
    if (shape.len < 2) return 0;
    return if (outermost_first) shape[0] else shape[shape.len - 1];
}

fn declaredContext(inv: compare.Inventory) ?u64 {
    for (inv.keys) |k| {
        const base = if (std.mem.lastIndexOfScalar(u8, k.name, '.')) |i| k.name[i + 1 ..] else k.name;
        if (std.mem.eql(u8, base, "context_length") or std.mem.eql(u8, k.name, "max_position_embeddings") or std.mem.eql(u8, k.name, "text_config.max_position_embeddings")) {
            return std.fmt.parseInt(u64, k.value, 10) catch continue;
        }
    }
    return null;
}

fn slidingWindow(inv: compare.Inventory) ?u64 {
    for (inv.keys) |k| {
        const base = if (std.mem.lastIndexOfScalar(u8, k.name, '.')) |i| k.name[i + 1 ..] else k.name;
        if (std.mem.eql(u8, base, "sliding_window")) {
            const n = std.fmt.parseInt(u64, k.value, 10) catch continue;
            if (n > 0) return n;
        }
    }
    return null;
}

/// This machine's memory: macOS's `hw.memsize`; null elsewhere or when the
/// call fails.
pub fn machineMemory() ?u64 {
    if (builtin.os.tag != .macos) return null;
    var value: u64 = 0;
    var len: usize = @sizeOf(u64);
    if (std.c.sysctlbyname("hw.memsize", &value, &len, null, 0) != 0) return null;
    return value;
}

pub fn render(m: Memory, arena: std.mem.Allocator, out: *std.Io.Writer, ui: kit.Ui) !void {
    const v = kit.values;
    var fields: std.ArrayList(kit.blocks.Field) = .empty;
    try fields.append(arena, .{ .key = "Weights", .value = try std.fmt.allocPrint(arena, "{f}", .{v.size(m.weights)}) });
    try fields.append(arena, .{ .key = "Cache", .value = if (m.cache_blocks == 0)
        "no block has its own K and V projections"
    else
        try std.fmt.allocPrint(arena, "{d} blocks with their own K and V · {f} values a token", .{ m.cache_blocks, v.grouped(m.values_per_token) }) });
    if (m.recurrent_blocks > 0) try fields.append(arena, .{ .key = "Recurrent", .value = try std.fmt.allocPrint(arena, "{d} blocks: constant state per sequence, not counted", .{m.recurrent_blocks}), .role = .muted });
    if (m.sliding_window) |w| try fields.append(arena, .{ .key = "Window", .value = try std.fmt.allocPrint(arena, "{f} tokens declared, not applied: the cache is an upper bound", .{v.grouped(w)}), .role = .muted });
    if (m.machine) |bytes| try fields.append(arena, .{ .key = "Machine", .value = try std.fmt.allocPrint(arena, "{f} of memory ({d} GiB)", .{ v.size(bytes), bytes >> 30 }) });
    try kit.blocks.detail(out, ui, &.{.{ .fields = fields.items }});
    try out.writeByte('\n');
    const rows = try arena.alloc([]const kit.table.Cell, m.rows.len);
    for (m.rows, rows) |r, *row| row.* = try arena.dupe(kit.table.Cell, &.{
        .{ .text = try std.fmt.allocPrint(arena, "{f}", .{v.grouped(r.context)}), .role = .id },
        .{ .text = try std.fmt.allocPrint(arena, "{f}", .{v.size(r.cache_f16)}) },
        .{ .text = try total(arena, r.total_f16, m.machine) },
        .{ .text = try std.fmt.allocPrint(arena, "{f}", .{v.size(r.cache_q8_0)}) },
        .{ .text = try total(arena, r.total_q8_0, m.machine) },
    });
    try kit.table.render(out, ui, arena, 0, &.{
        .{ .title = "CONTEXT", .alignment = .right },
        .{ .title = "CACHE F16", .alignment = .right },
        .{ .title = "TOTAL F16", .alignment = .right },
        .{ .title = "CACHE Q8_0", .alignment = .right, .drop = 1 },
        .{ .title = "TOTAL Q8_0", .alignment = .right, .drop = 1 },
    }, rows);
    try out.print("{s}estimate from the tensors: weights plus K/V; activations and scratch buffers not included{s}\n", .{ ui.on(.muted), ui.off(.muted) });
}

fn total(arena: std.mem.Allocator, bytes: u64, machine: ?u64) ![]const u8 {
    const m = machine orelse return std.fmt.allocPrint(arena, "{f}", .{kit.values.size(bytes)});
    return std.fmt.allocPrint(arena, "{f} ({d}%)", .{ kit.values.size(bytes), bytes * 100 / @max(m, 1) });
}

const testing = std.testing;

fn fixture(arena: std.mem.Allocator, document: anytype) !compare.Inventory {
    var doc = try document(testing.allocator);
    defer doc.deinit();
    return compare.fromGguf(arena, doc);
}

test "the cache counts blocks with their own K and V, at the engine's precisions" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const models = inference.models;
    // Qwen 3.5: 16 attention blocks and the MTP block hold K/V; the 48
    // delta-net blocks hold constant state.
    const qwen = try estimate(a, try fixture(a, models.qwen35.inventoryDocument), 1000, "4096,max", null);
    try testing.expectEqual(@as(u32, 17), qwen.cache_blocks);
    try testing.expectEqual(@as(u32, 48), qwen.recurrent_blocks);
    try testing.expectEqual(@as(u64, 17 * 2 * 1024), qwen.values_per_token);
    try testing.expectEqual(@as(u64, 4096 * 17 * 2 * 1024 * 2), qwen.rows[0].cache_f16);
    try testing.expectEqual(qwen.rows[0].cache_f16 + 1000, qwen.rows[0].total_f16);
    try testing.expectEqual(@as(u64, 262144), qwen.rows[1].context);
    try testing.expect(qwen.rows[0].cache_q8_0 < qwen.rows[0].cache_f16);
    // Gemma 4 E4B: the shared-KV blocks add nothing; a window is declared.
    const e4b = try estimate(a, try fixture(a, models.gemma4.inventoryDocumentE4b), 0, "1024", null);
    try testing.expect(e4b.cache_blocks > 0 and e4b.cache_blocks < 42);
    try testing.expect(e4b.sliding_window != null);
    try testing.expectError(error.InvalidContext, estimate(a, try fixture(a, models.qwen35.inventoryDocument), 0, "0", null));
    try testing.expectError(error.InvalidContext, estimate(a, try fixture(a, models.qwen35.inventoryDocument), 0, "4k", null));
    try testing.expectError(error.InvalidContext, estimate(a, try fixture(a, models.qwen35.inventoryDocument), 0, "1,2,3,4,5,6,7,8,9", null));
}
