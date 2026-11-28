//! An artifact's names, adapter-neutral: metadata keys with their values as
//! text, and tensors with their shapes and encodings, built from a GGUF
//! directory or safetensors headers (with `config.json`'s scalars as keys).
//! `diff` compares two of them the way docs/models/README.md § Which kind
//! of new reads a new file: names folded (block numbers become `N`, a
//! GGUF key's architecture prefix `<arch>`), then what only one side has,
//! the values and shapes that differ under a shared name, and which of the
//! three kinds of new the difference is. Pure: no I/O, every string in the
//! caller's arena.
const std = @import("std");
const inference = @import("inference");
const kit = @import("ui.zig");

pub const Key = struct { name: []const u8, value: []const u8 };
pub const Tensor = struct { name: []const u8, shape: []const u64, encoding: []const u8 };

pub const Inventory = struct {
    /// `general.architecture`, or a safetensors config's `model_type`.
    architecture: ?[]const u8,
    keys: []const Key,
    tensors: []const Tensor,
};

/// The most elements of a retained array a value's text spells out.
const shown_elements = 64;

/// Borrows nothing from `doc`: every string is copied into `arena`.
pub fn fromGguf(arena: std.mem.Allocator, doc: inference.gguf.Document) !Inventory {
    const keys = try arena.alloc(Key, doc.metadata.len);
    for (doc.metadata, keys) |m, *k| k.* = .{ .name = try arena.dupe(u8, m.key), .value = try valueText(arena, m.kind, m.value) };
    const tensors = try arena.alloc(Tensor, doc.tensors.len);
    for (doc.tensors, tensors) |t, *o| o.* = .{
        .name = try arena.dupe(u8, t.name),
        .shape = try arena.dupe(u64, t.dimensions),
        .encoding = if (inference.encoding.layout(t.encoding_id)) |l| l.name else try std.fmt.allocPrint(arena, "id {d}", .{t.encoding_id}),
    };
    const architecture = if (doc.string("general.architecture")) |a| try arena.dupe(u8, a) else null;
    return .{ .architecture = architecture, .keys = keys, .tensors = tensors };
}

/// The tensors of every shard, and `config` (`inspect`'s config rows) as
/// the keys; the architecture is the config's `model_type`. `config` is a
/// slice of anything with `key` and `value` strings.
pub fn fromSafetensors(arena: std.mem.Allocator, shards: []const *const inference.safetensors.Document, config: anytype) !Inventory {
    var tensors: std.ArrayList(Tensor) = .empty;
    for (shards) |doc| for (doc.tensors) |t| try tensors.append(arena, .{ .name = try arena.dupe(u8, t.name), .shape = try arena.dupe(u64, t.shape), .encoding = t.dtype.name() });
    const keys = try arena.alloc(Key, config.len);
    var architecture: ?[]const u8 = null;
    for (config, keys) |c, *k| {
        k.* = .{ .name = try arena.dupe(u8, c.key), .value = try arena.dupe(u8, c.value) };
        if (std.mem.eql(u8, c.key, "model_type")) architecture = k.value;
    }
    return .{ .architecture = architecture, .keys = keys, .tensors = tensors.items };
}

fn valueText(arena: std.mem.Allocator, kind: inference.gguf.Type, value: inference.gguf.Value) ![]const u8 {
    return switch (value) {
        .unsigned => |x| std.fmt.allocPrint(arena, "{d}", .{x}),
        .signed => |x| std.fmt.allocPrint(arena, "{d}", .{x}),
        .float => |x| if (kind == .float32) std.fmt.allocPrint(arena, "{d}", .{@as(f32, @floatCast(x))}) else std.fmt.allocPrint(arena, "{d}", .{x}),
        .boolean => |x| if (x) "true" else "false",
        .string => |x| arena.dupe(u8, x),
        .array => |a| blk: {
            const values = a.values orelse break :blk std.fmt.allocPrint(arena, "{s}[{d}]", .{ @tagName(a.element_type), a.count });
            var text: std.ArrayList(u8) = .empty;
            try text.append(arena, '[');
            for (values[0..@min(values.len, shown_elements)], 0..) |v, i| {
                if (i > 0) try text.appendSlice(arena, ", ");
                try text.appendSlice(arena, try valueText(arena, a.element_type, v));
            }
            if (values.len > shown_elements) try text.print(arena, ", … {d} in all", .{values.len});
            try text.append(arena, ']');
            break :blk text.items;
        },
    };
}

/// `name` with every all-digit segment as `N`, and the first such number:
/// the block a tensor or key belongs to. A key's `<architecture>.` prefix
/// becomes `<arch>.`, so two architectures' keys meet.
pub fn fold(arena: std.mem.Allocator, name: []const u8, architecture: ?[]const u8) !struct { name: []const u8, block: ?u32 } {
    var rest = name;
    var out: std.ArrayList(u8) = .empty;
    if (architecture) |a| if (a.len > 0 and name.len > a.len + 1 and std.mem.startsWith(u8, name, a) and name[a.len] == '.') {
        try out.appendSlice(arena, "<arch>");
        rest = name[a.len + 1 ..];
    };
    var block: ?u32 = null;
    var it = std.mem.splitScalar(u8, rest, '.');
    var first = true;
    while (it.next()) |part| {
        if (!first or out.items.len > 0) try out.append(arena, '.');
        first = false;
        const digits = part.len > 0 and for (part) |c| {
            if (!std.ascii.isDigit(c)) break false;
        } else true;
        if (digits) {
            if (block == null) block = std.fmt.parseInt(u32, part, 10) catch null;
            try out.append(arena, 'N');
        } else try out.appendSlice(arena, part);
    }
    return .{ .name = out.items, .block = block };
}

/// A comparison as `inspect` reports it: the reference's name and the diff.
pub const Against = struct { label: []const u8, diff: Diff };

pub const Kind = enum {
    /// Only encodings (or `general.*` facts) differ: a new file of a model.
    same_model,
    /// One architecture; names, values, or shapes differ.
    configuration,
    /// Another `general.architecture` (or `model_type`).
    new_family,

    pub fn sentence(k: Kind) []const u8 {
        return switch (k) {
            .same_model => "a new file of the same model: only encodings differ",
            .configuration => "a configuration of the same architecture: names, values, or shapes differ",
            .new_family => "a new family: another architecture",
        };
    }
};

/// A folded name and the blocks it occurs in on each side ("" when none).
pub const Named = struct { name: []const u8, blocks: []const u8 = "" };
pub const ValueChange = struct { key: []const u8, here: []const u8, there: []const u8 };
/// A tensor whose shape, encoding, or block set differs under one folded name.
pub const TensorChange = struct { name: []const u8, blocks: []const u8, here: []const u8, there: []const u8 };

pub const Diff = struct {
    kind: Kind,
    here: ?[]const u8,
    there: ?[]const u8,
    keys_here: []const Named,
    keys_there: []const Named,
    tensors_here: []const Named,
    tensors_there: []const Named,
    values: []const ValueChange,
    /// Shapes, or the blocks a shared name occurs in.
    shapes: []const TensorChange,
    encodings: []const TensorChange,
    /// `general.*` and `quantize.*` keys that differ or only one side has
    /// (names, sources, licences, the importance matrix): counted, not
    /// listed, and never a reason for the kind.
    general_differences: usize,
};

const Side = struct { blocks: std.ArrayList(u32) = .empty, present: bool = false };
const Pair = struct { here: Side = .{}, there: Side = .{} };

/// Compares `here` (the inspected file) with `there` (the reference).
pub fn diff(arena: std.mem.Allocator, here: Inventory, there: Inventory) !Diff {
    // Keys, folded; a numbered key (`general.base_model.0.name`) compares
    // its first value only.
    var key_values: std.StringArrayHashMapUnmanaged([2]?[]const u8) = .empty;
    for ([_]Inventory{ here, there }, 0..) |inv, side| for (inv.keys) |k| {
        const f = try fold(arena, k.name, inv.architecture);
        const entry = try key_values.getOrPutValue(arena, f.name, .{ null, null });
        if (entry.value_ptr[side] == null) entry.value_ptr[side] = k.value;
    };
    var keys_here: std.ArrayList(Named) = .empty;
    var keys_there: std.ArrayList(Named) = .empty;
    var values: std.ArrayList(ValueChange) = .empty;
    var general: usize = 0;
    var it = key_values.iterator();
    while (it.next()) |e| {
        const v = e.value_ptr.*;
        if (std.mem.eql(u8, e.key_ptr.*, "general.architecture")) continue;
        const differs = v[0] == null or v[1] == null or !std.mem.eql(u8, v[0].?, v[1].?);
        if (!differs) continue;
        if (provenance(e.key_ptr.*)) {
            general += 1;
        } else if (v[1] == null) {
            try keys_here.append(arena, .{ .name = e.key_ptr.* });
        } else if (v[0] == null) {
            try keys_there.append(arena, .{ .name = e.key_ptr.* });
        } else try values.append(arena, .{ .key = e.key_ptr.*, .here = v[0].?, .there = v[1].? });
    }

    // Tensors: by folded name for presence and blocks, by exact name for
    // shape and encoding, grouped back by folded name and change.
    var tensor_blocks: std.StringArrayHashMapUnmanaged(Pair) = .empty;
    var exact: std.StringHashMapUnmanaged(*const Tensor) = .empty;
    for ([_]Inventory{ here, there }, 0..) |inv, side| for (inv.tensors) |*t| {
        const f = try fold(arena, t.name, null);
        const pair = try tensor_blocks.getOrPutValue(arena, f.name, .{});
        const s = if (side == 0) &pair.value_ptr.here else &pair.value_ptr.there;
        s.present = true;
        if (f.block) |b| try s.blocks.append(arena, b);
        if (side == 1) try exact.put(arena, t.name, t);
    };
    var tensors_here: std.ArrayList(Named) = .empty;
    var tensors_there: std.ArrayList(Named) = .empty;
    var shapes: Grouped = .{};
    var encodings: Grouped = .{};
    var tb = tensor_blocks.iterator();
    while (tb.next()) |e| {
        const p = e.value_ptr.*;
        if (p.here.present and !p.there.present) {
            try tensors_here.append(arena, .{ .name = e.key_ptr.*, .blocks = try ranges(arena, p.here.blocks.items) });
        } else if (!p.here.present and p.there.present) {
            try tensors_there.append(arena, .{ .name = e.key_ptr.*, .blocks = try ranges(arena, p.there.blocks.items) });
        } else {
            const a = try ranges(arena, p.here.blocks.items);
            const b = try ranges(arena, p.there.blocks.items);
            if (!std.mem.eql(u8, a, b)) try shapes.add(arena, e.key_ptr.*, try std.fmt.allocPrint(arena, "blocks {s}", .{a}), try std.fmt.allocPrint(arena, "blocks {s}", .{b}), null);
        }
    }
    for (here.tensors) |t| {
        const other = exact.get(t.name) orelse continue;
        const f = try fold(arena, t.name, null);
        if (!std.mem.eql(u64, t.shape, other.shape)) {
            try shapes.add(arena, f.name, try shapeText(arena, t.shape), try shapeText(arena, other.shape), f.block);
        } else if (!std.mem.eql(u8, t.encoding, other.encoding)) {
            try encodings.add(arena, f.name, t.encoding, other.encoding, f.block);
        }
    }

    const same_architecture = if (here.architecture != null and there.architecture != null) std.mem.eql(u8, here.architecture.?, there.architecture.?) else here.architecture == null and there.architecture == null;
    const structural = keys_here.items.len + keys_there.items.len + tensors_here.items.len + tensors_there.items.len + values.items.len + shapes.list.items.len;
    return .{
        .kind = if (!same_architecture) .new_family else if (structural > 0) .configuration else .same_model,
        .here = here.architecture,
        .there = there.architecture,
        .keys_here = keys_here.items,
        .keys_there = keys_there.items,
        .tensors_here = tensors_here.items,
        .tensors_there = tensors_there.items,
        .values = values.items,
        .shapes = try shapes.finish(arena),
        .encodings = try encodings.finish(arena),
        .general_differences = general,
    };
}

/// Facts about where a file came from rather than what the model is: its
/// names and sources (`general.*`) and how it was quantized (`quantize.*`).
fn provenance(key: []const u8) bool {
    return std.mem.startsWith(u8, key, "general.") or std.mem.startsWith(u8, key, "quantize.");
}

/// Changes grouped by (folded name, here, there), each with its blocks.
const Grouped = struct {
    list: std.ArrayList(struct { name: []const u8, here: []const u8, there: []const u8, blocks: std.ArrayList(u32) }) = .empty,

    fn add(g: *Grouped, arena: std.mem.Allocator, name: []const u8, here: []const u8, there: []const u8, block: ?u32) !void {
        for (g.list.items) |*item| if (std.mem.eql(u8, item.name, name) and std.mem.eql(u8, item.here, here) and std.mem.eql(u8, item.there, there)) {
            if (block) |b| try item.blocks.append(arena, b);
            return;
        };
        try g.list.append(arena, .{ .name = name, .here = here, .there = there, .blocks = .empty });
        if (block) |b| try g.list.items[g.list.items.len - 1].blocks.append(arena, b);
    }

    fn finish(g: *Grouped, arena: std.mem.Allocator) ![]const TensorChange {
        const out = try arena.alloc(TensorChange, g.list.items.len);
        for (g.list.items, out) |item, *o| o.* = .{ .name = item.name, .blocks = try ranges(arena, item.blocks.items), .here = item.here, .there = item.there };
        return out;
    }
};

/// "0–3, 8, 12–47" from block numbers in any order (duplicates once); a
/// set of more than four runs with a period of at most 16 as
/// "48 of 0–63: i mod 4 ∈ {0, 1, 2}", any other as its first runs and its
/// count. "" for none.
pub fn ranges(arena: std.mem.Allocator, blocks: []const u32) ![]const u8 {
    if (blocks.len == 0) return "";
    const all = try arena.dupe(u32, blocks);
    std.mem.sort(u32, all, {}, std.sort.asc(u32));
    var unique: std.ArrayList(u32) = .empty;
    for (all) |b| if (unique.items.len == 0 or unique.items[unique.items.len - 1] != b) try unique.append(arena, b);
    const sorted = unique.items;
    var runs: std.ArrayList([2]u32) = .empty;
    var i: usize = 0;
    while (i < sorted.len) {
        var j = i;
        while (j + 1 < sorted.len and sorted[j + 1] == sorted[j] + 1) j += 1;
        try runs.append(arena, .{ sorted[i], sorted[j] });
        i = j + 1;
    }
    var text: std.ArrayList(u8) = .empty;
    const lo = sorted[0];
    const hi = sorted[sorted.len - 1];
    if (runs.items.len > 4) {
        if (period(sorted, lo, hi)) |p| {
            try text.print(arena, "{d} of {d}–{d}: i mod {d} ∈ {{", .{ sorted.len, lo, hi, p });
            var first = true;
            var r: u32 = 0;
            while (r < p) : (r += 1) if (member(sorted, lo + r)) {
                try text.print(arena, "{s}{d}", .{ if (first) "" else ", ", (lo + r) % p });
                first = false;
            };
            try text.append(arena, '}');
            return text.items;
        }
    }
    for (runs.items[0..@min(runs.items.len, 4)], 0..) |run, k| {
        if (k > 0) try text.appendSlice(arena, ", ");
        if (run[0] == run[1]) try text.print(arena, "{d}", .{run[0]}) else try text.print(arena, "{d}–{d}", .{ run[0], run[1] });
    }
    if (runs.items.len > 4) try text.print(arena, ", … ({d} of {d}–{d})", .{ sorted.len, lo, hi });
    return text.items;
}

/// The smallest period (2–16, at most half the span) whose first window
/// repeats over all of `lo`–`hi`; null when none does.
fn period(sorted: []const u32, lo: u32, hi: u32) ?u32 {
    var p: u32 = 2;
    while (p <= 16 and 2 * p <= hi - lo + 1) : (p += 1) {
        var x = lo;
        const repeats = while (x <= hi) : (x += 1) {
            if (member(sorted, x) != member(sorted, lo + (x - lo) % p)) break false;
        } else true;
        if (repeats) return p;
    }
    return null;
}

fn member(sorted: []const u32, x: u32) bool {
    return std.sort.binarySearch(u32, sorted, x, struct {
        fn order(target: u32, item: u32) std.math.Order {
            return std.math.order(target, item);
        }
    }.order) != null;
}

/// A value as a table shows it: at most 48 characters, then its length.
fn short(arena: std.mem.Allocator, text: []const u8) ![]const u8 {
    const limit = 48;
    if (text.len <= limit) return kit.text.clean(arena, text);
    var end: usize = limit;
    while (end > 0 and (text[end] & 0xC0) == 0x80) end -= 1;
    return std.fmt.allocPrint(arena, "{s}… ({d} bytes)", .{ try kit.text.clean(arena, text[0..end]), text.len });
}

pub fn shapeText(arena: std.mem.Allocator, shape: []const u64) ![]const u8 {
    var text: std.ArrayList(u8) = .empty;
    for (shape, 0..) |d, i| try text.print(arena, "{s}{d}", .{ if (i > 0) " × " else "", d });
    if (shape.len == 0) try text.appendSlice(arena, "scalar");
    return text.items;
}

/// The most rows each section of the rendered diff lists.
pub const shown_rows = 32;

/// The kind, then each non-empty section as a table, `label` naming the
/// reference. Text from files is cleaned.
pub fn render(d: Diff, arena: std.mem.Allocator, out: *std.Io.Writer, ui: kit.Ui, label: []const u8) !void {
    const arch = struct {
        fn text(a: ?[]const u8) []const u8 {
            return a orelse "none";
        }
    }.text;
    try kit.blocks.detail(out, ui, &.{.{ .fields = &.{
        .{ .key = "Against", .value = try kit.text.clean(arena, label), .role = .id },
        .{ .key = "Architecture", .value = try kit.text.clean(arena, try std.fmt.allocPrint(arena, "{s} here · {s} there", .{ arch(d.here), arch(d.there) })) },
        .{ .key = "Kind", .value = d.kind.sentence() },
    } }});
    try names(arena, out, ui, "KEY ONLY HERE", d.keys_here);
    try names(arena, out, ui, "KEY ONLY THERE", d.keys_there);
    try names(arena, out, ui, "TENSOR ONLY HERE", d.tensors_here);
    try names(arena, out, ui, "TENSOR ONLY THERE", d.tensors_there);
    if (d.values.len > 0) {
        const rows = try arena.alloc([]const kit.table.Cell, @min(d.values.len, shown_rows));
        for (d.values[0..rows.len], rows) |v, *row| row.* = try arena.dupe(kit.table.Cell, &.{
            .{ .text = try kit.text.clean(arena, v.key), .role = .id },
            .{ .text = try short(arena, v.here) },
            .{ .text = try short(arena, v.there) },
        });
        try out.writeByte('\n');
        try kit.table.render(out, ui, arena, 0, &.{ .{ .title = "VALUE" }, .{ .title = "HERE" }, .{ .title = "THERE" } }, rows);
        try more(out, ui, d.values.len);
    }
    try changes(arena, out, ui, "SHAPE OR BLOCKS", d.shapes);
    try changes(arena, out, ui, "ENCODING", d.encodings);
    if (d.general_differences > 0) try out.print("\n{s}{d} general.* and quantize.* keys differ (names, sources, quantization); not listed{s}\n", .{ ui.on(.muted), d.general_differences, ui.off(.muted) });
}

fn names(arena: std.mem.Allocator, out: *std.Io.Writer, ui: kit.Ui, title: []const u8, list: []const Named) !void {
    if (list.len == 0) return;
    const rows = try arena.alloc([]const kit.table.Cell, @min(list.len, shown_rows));
    for (list[0..rows.len], rows) |n, *row| row.* = try arena.dupe(kit.table.Cell, &.{
        .{ .text = try kit.text.clean(arena, n.name), .role = .id },
        .{ .text = if (n.blocks.len > 0) try std.fmt.allocPrint(arena, "blocks {s}", .{n.blocks}) else "" },
    });
    try out.writeByte('\n');
    try kit.table.render(out, ui, arena, 0, &.{ .{ .title = title }, .{ .title = "WHERE" } }, rows);
    try more(out, ui, list.len);
}

fn changes(arena: std.mem.Allocator, out: *std.Io.Writer, ui: kit.Ui, title: []const u8, list: []const TensorChange) !void {
    if (list.len == 0) return;
    const rows = try arena.alloc([]const kit.table.Cell, @min(list.len, shown_rows));
    for (list[0..rows.len], rows) |c, *row| row.* = try arena.dupe(kit.table.Cell, &.{
        .{ .text = try kit.text.clean(arena, c.name), .role = .id },
        .{ .text = if (c.blocks.len > 0) try std.fmt.allocPrint(arena, "blocks {s}", .{c.blocks}) else "" },
        .{ .text = c.here },
        .{ .text = c.there },
    });
    try out.writeByte('\n');
    try kit.table.render(out, ui, arena, 0, &.{ .{ .title = title }, .{ .title = "WHERE" }, .{ .title = "HERE" }, .{ .title = "THERE" } }, rows);
    try more(out, ui, list.len);
}

fn more(out: *std.Io.Writer, ui: kit.Ui, total: usize) !void {
    if (total > shown_rows) try out.print("{s}… {d} more (--json lists every one){s}\n", .{ ui.on(.muted), total - shown_rows, ui.off(.muted) });
}

const testing = std.testing;

fn fixture(arena: std.mem.Allocator, document: anytype) !Inventory {
    var doc = try document(testing.allocator);
    defer doc.deinit();
    return fromGguf(arena, doc);
}

test "names fold their block numbers and their architecture prefix" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const t = try fold(a, "blk.12.ffn_down_exps.weight", null);
    try testing.expectEqualStrings("blk.N.ffn_down_exps.weight", t.name);
    try testing.expectEqual(@as(?u32, 12), t.block);
    const st = try fold(a, "model.layers.3.mlp.experts.17.down_proj.weight", null);
    try testing.expectEqualStrings("model.layers.N.mlp.experts.N.down_proj.weight", st.name);
    try testing.expectEqual(@as(?u32, 3), st.block);
    const k = try fold(a, "qwen35.attention.head_count", "qwen35");
    try testing.expectEqualStrings("<arch>.attention.head_count", k.name);
    try testing.expectEqualStrings("qwen35x.block_count", (try fold(a, "qwen35x.block_count", "qwen35")).name);
    try testing.expectEqualStrings("0–3, 8, 12–13", try ranges(a, &.{ 13, 0, 1, 2, 3, 8, 12, 3 }));
    // Three of every four blocks: Qwen 3.5's delta-net layers.
    var pattern: [48]u32 = undefined;
    var n: usize = 0;
    for (0..64) |b| if (b % 4 != 3) {
        pattern[n] = @intCast(b);
        n += 1;
    };
    try testing.expectEqualStrings("48 of 0–62: i mod 4 ∈ {0, 1, 2}", try ranges(a, &pattern));
    try testing.expectEqualStrings("6 of 3–23: i mod 4 ∈ {3}", try ranges(a, &.{ 3, 7, 11, 15, 19, 23 }));
    try testing.expectEqualStrings("0, 2, 5, 9, … (6 of 0–20)", try ranges(a, &.{ 0, 2, 5, 9, 14, 20 }));
    try testing.expectEqualStrings("abc", try short(a, "abc"));
    try testing.expect(std.mem.endsWith(u8, try short(a, &@as([100]u8, @splat('x'))), "(100 bytes)"));
}

test "the inspected file against a model nuclis runs: configuration, family, same model" {
    var arena_state = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena_state.deinit();
    const a = arena_state.allocator();
    const models = inference.models;
    const qwen = try fixture(a, models.qwen35.inventoryDocument);
    const bonsai = try fixture(a, models.qwen35.bonsaiInventoryDocument);
    const e4b = try fixture(a, models.gemma4.inventoryDocumentE4b);
    const moe = try fixture(a, models.gemma4.inventoryDocument26bA4b);

    const same = try diff(a, qwen, qwen);
    try testing.expectEqual(Kind.same_model, same.kind);
    try testing.expectEqual(@as(usize, 0), same.values.len + same.shapes.len + same.encodings.len);

    // Bonsai is a Qwen 3.5 checkpoint in a rotated basis: one architecture.
    const rotated = try diff(a, bonsai, qwen);
    try testing.expect(rotated.kind != .new_family);

    // The E4B against the 26B-A4B: per-layer embeddings on one side, the
    // expert tensors on the other.
    const gemma = try diff(a, e4b, moe);
    try testing.expectEqual(Kind.configuration, gemma.kind);
    try testing.expect(contains(gemma.tensors_here, "per_layer_token_embd.weight"));
    try testing.expect(contains(gemma.tensors_there, "blk.N.ffn_down_exps.weight"));
    try testing.expect(gemma.values.len > 0);

    const family = try diff(a, qwen, e4b);
    try testing.expectEqual(Kind.new_family, family.kind);
    // Shared `<arch>.` keys meet across the two architectures.
    try testing.expect(!contains(family.keys_here, "<arch>.block_count"));

    var buffer: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buffer.deinit();
    try render(gemma, a, &buffer.writer, .plain, "gemma-4-26b-a4b");
    try testing.expect(std.mem.indexOf(u8, buffer.written(), "TENSOR ONLY THERE") != null);
}

fn contains(list: []const Named, name: []const u8) bool {
    for (list) |n| if (std.mem.eql(u8, n.name, name)) return true;
    return false;
}
