//! `inspect --layout`: an artifact's block structure from its tensor names
//! and shapes alone, adapter-neutral. A block is the first numbered path
//! segment (`blk.N.`, `model.layers.N.`); its template is the set of
//! (tensor suffix, shape) it holds, so blocks that differ only in encoding
//! share one. Templates are lettered in first-seen order and the schedule
//! printed as repeats (`(A A A B) × 16 · C`). Each dimension is named by
//! the integer metadata or config values it equals, and by products of a
//! head count with a head width, so a shape reads as the model's facts.
//! Pure: no I/O, every string in the caller's arena.
const std = @import("std");
const compare = @import("inventory.zig");
const kit = @import("ui.zig");

pub const Entry = struct {
    /// The suffix after the block number (`attn_q.weight`), or the whole
    /// name outside blocks; further numbers folded to `N`.
    name: []const u8,
    /// How many tensors fold to this name in one block (experts), else 1.
    count: u32 = 1,
    shape: []const u64,
    /// The shape with each dimension's names: `4096 (embedding_length)`.
    dims: []const u8,
    /// Every encoding the entry has across its blocks: `Q4_K/Q6_K`.
    encodings: []const u8,
};

pub const Template = struct {
    letter: []const u8,
    /// The blocks that have it, as `compare.ranges` writes them.
    blocks: []const u8,
    entries: []const Entry,
};

pub const Layout = struct {
    /// Distinct block numbers.
    block_count: u32,
    /// `(A A A B) × 16 · C`; "" without blocks.
    schedule: []const u8,
    templates: []const Template,
    /// Tensors outside any block (embeddings, output, final norm).
    outside: []const Entry,
};

/// The most templates lettered; past it the rest share `?` (a file with
/// that many is a list, not a pattern).
const max_templates = 26;

pub fn layout(arena: std.mem.Allocator, inv: compare.Inventory) !Layout {
    const names = try dimensionNames(arena, inv);
    // Per block: its entries keyed by suffix (structural: suffix and shape).
    const Block = struct { entries: std.StringArrayHashMapUnmanaged(Building) = .empty };
    var blocks: std.AutoArrayHashMapUnmanaged(u32, Block) = .empty;
    var outside: std.StringArrayHashMapUnmanaged(Building) = .empty;
    for (inv.tensors) |t| {
        const f = try compare.fold(arena, t.name, null);
        const block = f.block orelse {
            try addTo(arena, &outside, f.name, t);
            continue;
        };
        const suffix = afterFirstN(f.name);
        const b = try blocks.getOrPutValue(arena, block, .{});
        try addTo(arena, &b.value_ptr.entries, suffix, t);
    }
    // Templates by signature, in first-seen block order.
    const order = try arena.dupe(u32, blocks.keys());
    std.mem.sort(u32, order, {}, std.sort.asc(u32));
    var signatures: std.StringArrayHashMapUnmanaged(struct { index: usize, blocks: std.ArrayList(u32) }) = .empty;
    var schedule_letters: std.ArrayList(usize) = .empty;
    for (order) |b| {
        const sig = try signature(arena, &blocks.getPtr(b).?.entries);
        const got = try signatures.getOrPut(arena, sig);
        if (!got.found_existing) got.value_ptr.* = .{ .index = signatures.count() - 1, .blocks = .empty };
        try got.value_ptr.blocks.append(arena, b);
        try schedule_letters.append(arena, got.value_ptr.index);
    }
    const templates = try arena.alloc(Template, signatures.count());
    for (signatures.values(), templates) |v, *t| {
        const merged = try mergeEncodings(arena, &blocks, v.blocks.items);
        t.* = .{ .letter = letter(v.index), .blocks = try compare.ranges(arena, v.blocks.items), .entries = try finish(arena, merged, names) };
    }
    return .{
        .block_count = @intCast(blocks.count()),
        .schedule = try scheduleText(arena, schedule_letters.items),
        .templates = templates,
        .outside = try finish(arena, &outside, names),
    };
}

const Building = struct {
    count: u32 = 0,
    shape: []const u64,
    encodings: std.ArrayList([]const u8) = .empty,
};

fn addTo(arena: std.mem.Allocator, map: *std.StringArrayHashMapUnmanaged(Building), name: []const u8, t: compare.Tensor) !void {
    // One folded name with several shapes (rare) keeps them apart.
    const key = if (map.get(name)) |old| (if (std.mem.eql(u64, old.shape, t.shape)) name else try std.fmt.allocPrint(arena, "{s} [{s}]", .{ name, try compare.shapeText(arena, t.shape) })) else name;
    const got = try map.getOrPut(arena, key);
    if (!got.found_existing) got.value_ptr.* = .{ .shape = t.shape };
    got.value_ptr.count += 1;
    try addEncoding(arena, &got.value_ptr.encodings, t.encoding);
}

fn addEncoding(arena: std.mem.Allocator, list: *std.ArrayList([]const u8), encoding: []const u8) !void {
    for (list.items) |e| if (std.mem.eql(u8, e, encoding)) return;
    try list.append(arena, encoding);
}

fn afterFirstN(name: []const u8) []const u8 {
    if (std.mem.startsWith(u8, name, "N.")) return name[2..];
    if (std.mem.indexOf(u8, name, ".N.")) |i| return name[i + 3 ..];
    return name;
}

/// The block's structure as one string: sorted "suffix count shape" lines.
fn signature(arena: std.mem.Allocator, entries: *const std.StringArrayHashMapUnmanaged(Building)) ![]const u8 {
    const lines = try arena.alloc([]const u8, entries.count());
    for (entries.keys(), entries.values(), lines) |k, v, *line| line.* = try std.fmt.allocPrint(arena, "{s} {d} {s}", .{ k, v.count, try compare.shapeText(arena, v.shape) });
    std.mem.sort([]const u8, lines, {}, struct {
        fn less(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.less);
    return std.mem.join(arena, "\n", lines);
}

/// The first block's entries with every block's encodings merged in.
fn mergeEncodings(arena: std.mem.Allocator, blocks: anytype, members: []const u32) !*std.StringArrayHashMapUnmanaged(Building) {
    const out = try arena.create(std.StringArrayHashMapUnmanaged(Building));
    out.* = .empty;
    for (members) |b| {
        const entries = &blocks.getPtr(b).?.entries;
        for (entries.keys(), entries.values()) |k, v| {
            const got = try out.getOrPut(arena, k);
            if (!got.found_existing) got.value_ptr.* = .{ .count = v.count, .shape = v.shape };
            for (v.encodings.items) |e| try addEncoding(arena, &got.value_ptr.encodings, e);
        }
    }
    return out;
}

fn finish(arena: std.mem.Allocator, map: *const std.StringArrayHashMapUnmanaged(Building), names: Names) ![]const Entry {
    const out = try arena.alloc(Entry, map.count());
    for (map.keys(), map.values(), out) |k, v, *e| e.* = .{
        .name = k,
        .count = v.count,
        .shape = v.shape,
        .dims = try dimsText(arena, v.shape, names),
        .encodings = try std.mem.join(arena, "/", v.encodings.items),
    };
    return out;
}

fn letter(index: usize) []const u8 {
    const letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ";
    return if (index < max_templates) letters[index .. index + 1] else "?";
}

/// The schedule as repeats: the period (1–16) that covers the longest
/// prefix at least twice, then the rest the same way, joined by ` · `.
pub fn scheduleText(arena: std.mem.Allocator, sequence: []const usize) ![]const u8 {
    var text: std.ArrayList(u8) = .empty;
    var at: usize = 0;
    while (at < sequence.len) {
        if (text.items.len > 0) try text.appendSlice(arena, " · ");
        var best_p: usize = 1;
        var best_k: usize = 1;
        var p: usize = 1;
        while (p <= 16 and at + p <= sequence.len) : (p += 1) {
            var k: usize = 1;
            while (at + (k + 1) * p <= sequence.len and std.mem.eql(usize, sequence[at .. at + p], sequence[at + k * p .. at + (k + 1) * p])) k += 1;
            if (k >= 2 and k * p > best_k * best_p) {
                best_p = p;
                best_k = k;
            }
        }
        const unit = sequence[at .. at + best_p];
        if (best_p > 1) try text.append(arena, '(');
        for (unit, 0..) |l, i| {
            if (i > 0) try text.append(arena, ' ');
            try text.appendSlice(arena, letter(l));
        }
        if (best_p > 1) try text.append(arena, ')');
        if (best_k > 1) try text.print(arena, " × {d}", .{best_k});
        at += best_p * best_k;
    }
    return text.items;
}

/// Integer values the file declares, by value, with the keys that hold
/// them (architecture prefix dropped), and products of a head count with
/// a head width.
const Names = std.AutoHashMapUnmanaged(u64, std.ArrayList([]const u8));

fn dimensionNames(arena: std.mem.Allocator, inv: compare.Inventory) !Names {
    var names: Names = .empty;
    var counts: std.ArrayList(compare.Key) = .empty;
    var widths: std.ArrayList(compare.Key) = .empty;
    for (inv.keys) |k| {
        if (std.mem.startsWith(u8, k.name, "general.") or std.mem.startsWith(u8, k.name, "tokenizer.") or std.mem.startsWith(u8, k.name, "quantize.")) continue;
        const value = std.fmt.parseInt(u64, k.value, 10) catch continue;
        if (value < 2) continue;
        const short = shortKey(k.name, inv.architecture);
        try addName(arena, &names, value, short);
        if (isHeadCount(short)) try counts.append(arena, .{ .name = short, .value = k.value });
        if (isHeadWidth(short)) try widths.append(arena, .{ .name = short, .value = k.value });
    }
    for (counts.items) |c| for (widths.items) |w| {
        const product = std.math.mul(u64, try std.fmt.parseInt(u64, c.value, 10), try std.fmt.parseInt(u64, w.value, 10)) catch continue;
        const width = if (std.mem.lastIndexOfScalar(u8, w.name, '.')) |i| w.name[i + 1 ..] else w.name;
        try addName(arena, &names, product, try std.fmt.allocPrint(arena, "{s}·{s}", .{ c.name, width }));
    };
    return names;
}

fn addName(arena: std.mem.Allocator, names: *Names, value: u64, name: []const u8) !void {
    const got = try names.getOrPut(arena, value);
    if (!got.found_existing) got.value_ptr.* = .empty;
    for (got.value_ptr.items) |n| if (std.mem.eql(u8, n, name)) return;
    try got.value_ptr.append(arena, name);
}

fn shortKey(name: []const u8, architecture: ?[]const u8) []const u8 {
    if (architecture) |a| if (name.len > a.len + 1 and std.mem.startsWith(u8, name, a) and name[a.len] == '.') return name[a.len + 1 ..];
    return name;
}

fn isHeadCount(key: []const u8) bool {
    const base = if (std.mem.lastIndexOfScalar(u8, key, '.')) |i| key[i + 1 ..] else key;
    return std.mem.startsWith(u8, base, "head_count") or std.mem.eql(u8, base, "num_attention_heads") or std.mem.eql(u8, base, "num_key_value_heads");
}

fn isHeadWidth(key: []const u8) bool {
    const base = if (std.mem.lastIndexOfScalar(u8, key, '.')) |i| key[i + 1 ..] else key;
    return std.mem.eql(u8, base, "key_length") or std.mem.eql(u8, base, "value_length") or std.mem.eql(u8, base, "head_dim");
}

/// The names a dimension shows (a declared key before a product); the
/// rest are counted.
const shown_names = 1;

fn dimsText(arena: std.mem.Allocator, shape: []const u64, names: Names) ![]const u8 {
    var text: std.ArrayList(u8) = .empty;
    for (shape, 0..) |d, i| {
        if (i > 0) try text.appendSlice(arena, " × ");
        try text.print(arena, "{d}", .{d});
        const list = names.get(d) orelse continue;
        try text.appendSlice(arena, " (");
        for (list.items[0..@min(list.items.len, shown_names)], 0..) |n, j| {
            if (j > 0) try text.appendSlice(arena, " | ");
            try text.appendSlice(arena, n);
        }
        if (list.items.len > shown_names) try text.print(arena, " +{d}", .{list.items.len - shown_names});
        try text.append(arena, ')');
    }
    if (shape.len == 0) try text.appendSlice(arena, "scalar");
    return text.items;
}

pub fn render(l: Layout, arena: std.mem.Allocator, out: *std.Io.Writer, ui: kit.Ui) !void {
    try kit.blocks.detail(out, ui, &.{.{ .fields = &.{
        .{ .key = "Blocks", .value = try std.fmt.allocPrint(arena, "{d} · {d} template{s}", .{ l.block_count, l.templates.len, if (l.templates.len == 1) "" else "s" }) },
        .{ .key = "Schedule", .value = if (l.schedule.len > 0) l.schedule else kit.values.missing, .role = .id },
    } }});
    for (l.templates) |t| {
        try out.print("\n{s}{s}{s}  {s}blocks {s}{s}\n", .{ ui.on(.id), t.letter, ui.off(.id), ui.on(.muted), t.blocks, ui.off(.muted) });
        try renderEntries(arena, out, ui, t.entries);
    }
    if (l.outside.len > 0) {
        try out.print("\n{s}outside blocks{s}\n", .{ ui.on(.muted), ui.off(.muted) });
        try renderEntries(arena, out, ui, l.outside);
    }
}

fn renderEntries(arena: std.mem.Allocator, out: *std.Io.Writer, ui: kit.Ui, list: []const Entry) !void {
    const rows = try arena.alloc([]const kit.table.Cell, list.len);
    for (list, rows) |e, *row| row.* = try arena.dupe(kit.table.Cell, &.{
        .{ .text = if (e.count > 1) try std.fmt.allocPrint(arena, "{s} × {d}", .{ try kit.text.clean(arena, e.name), e.count }) else try kit.text.clean(arena, e.name), .role = .id },
        .{ .text = try kit.text.clean(arena, e.dims) },
        .{ .text = e.encodings, .role = .muted },
    });
    try kit.table.render(out, ui, arena, 0, &.{ .{ .title = "TENSOR" }, .{ .title = "SHAPE" }, .{ .title = "ENCODING", .drop = 1 } }, rows);
}

const testing = std.testing;

test "the schedule is written as its repeats" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    try testing.expectEqualStrings("A × 28", try scheduleText(a, &@as([28]usize, @splat(0))));
    var qwen: [65]usize = undefined;
    for (&qwen, 0..) |*x, i| x.* = if (i == 64) 2 else if (i % 4 == 3) 1 else 0;
    try testing.expectEqualStrings("(A A A B) × 16 · C", try scheduleText(a, &qwen));
    try testing.expectEqualStrings("A · B", try scheduleText(a, &.{ 0, 1 }));
    try testing.expectEqualStrings("", try scheduleText(a, &.{}));
}

test "layouts of files nuclis runs, from their fixtures" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const inference = @import("inference");
    const Fixture = struct {
        fn load(arena: std.mem.Allocator, document: anytype) !compare.Inventory {
            var doc = try document(testing.allocator);
            defer doc.deinit();
            return compare.fromGguf(arena, doc);
        }
    };
    const qwen = try layout(a, try Fixture.load(a, inference.models.qwen35.inventoryDocument));
    // Three delta-net blocks to one attention block, then the MTP block.
    try testing.expectEqualStrings("(A A A B) × 16 · C", qwen.schedule);
    try testing.expectEqual(@as(u32, 65), qwen.block_count);
    var named = false;
    for (qwen.outside) |e| if (std.mem.eql(u8, e.name, "token_embd.weight")) {
        try testing.expect(std.mem.indexOf(u8, e.dims, "5120 (embedding_length") != null);
        named = true;
    };
    try testing.expect(named);

    const gemma = try layout(a, try Fixture.load(a, inference.models.gemma4.inventoryDocument));
    try testing.expect(gemma.templates.len >= 2);
    try testing.expect(std.mem.indexOf(u8, gemma.schedule, "×") != null);

    const moe = try layout(a, try Fixture.load(a, inference.models.gemma4.inventoryDocument26bA4b));
    var expert = false;
    for (moe.templates[0].entries) |e| if (std.mem.indexOf(u8, e.name, "exps") != null) {
        expert = true;
    };
    try testing.expect(expert);

    var buffer: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buffer.deinit();
    try render(qwen, a, &buffer.writer, .plain);
    try testing.expect(std.mem.indexOf(u8, buffer.written(), "outside blocks") != null);
}
