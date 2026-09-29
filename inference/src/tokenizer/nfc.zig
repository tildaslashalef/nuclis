//! Unicode Normalization Form C (UAX #15) of validated UTF-8: full canonical
//! decomposition, canonical ordering, canonical composition. Data is the
//! generated nfc_table.zig (scripts/tokenizer-nfc.py); Hangul is algorithmic.
//! Text with no scalar at or above U+0300 is already NFC and is not copied.
const std = @import("std");
const table = @import("nfc_table.zig");

pub const Error = std.mem.Allocator.Error || error{InvalidUtf8};

const Scalar = struct { cp: u21, class: u8 };

// Hangul syllable arithmetic (Unicode §3.12).
const s_base = 0xAC00;
const l_base = 0x1100;
const v_base = 0x1161;
const t_base = 0x11A7;
const l_count = 19;
const v_count = 21;
const t_count = 28;
const n_count = v_count * t_count;
const s_count = l_count * n_count;

/// Returns null when `text` is already NFC by the quick check (keep
/// using `text`), else the normalized copy, owned by the caller.
pub fn normalize(alloc: std.mem.Allocator, text: []const u8) Error!?[]u8 {
    // Every scalar at or above U+0300 has a lead byte of at least 0xCC.
    if (for (text) |byte| {
        if (byte >= 0xCC) break false;
    } else true) return null;
    if (!std.unicode.utf8ValidateSlice(text)) return error.InvalidUtf8;
    var scalars: std.ArrayList(Scalar) = .empty;
    defer scalars.deinit(alloc);
    try scalars.ensureTotalCapacity(alloc, text.len);
    var it: std.unicode.Utf8Iterator = .{ .bytes = text, .i = 0 };
    while (it.nextCodepoint()) |cp| try decompose(alloc, &scalars, cp);
    reorder(scalars.items);
    const composed = compose(scalars.items);
    var size: usize = 0;
    for (composed) |s| size += std.unicode.utf8CodepointSequenceLength(s.cp) catch unreachable;
    const out = try alloc.alloc(u8, size);
    var n: usize = 0;
    for (composed) |s| n += std.unicode.utf8Encode(s.cp, out[n..]) catch unreachable;
    return out;
}

fn decompose(alloc: std.mem.Allocator, out: *std.ArrayList(Scalar), cp: u21) Error!void {
    if (cp >= s_base and cp < s_base + s_count) {
        const index = cp - s_base;
        try out.append(alloc, .{ .cp = l_base + index / n_count, .class = 0 });
        try out.append(alloc, .{ .cp = v_base + (index % n_count) / t_count, .class = 0 });
        if (index % t_count != 0) try out.append(alloc, .{ .cp = t_base + index % t_count, .class = 0 });
        return;
    }
    if (find(&table.decompositions, cp)) |entry| {
        for (table.pool[entry[1]..][0..entry[2]]) |part| try out.append(alloc, .{ .cp = part, .class = class(part) });
        return;
    }
    try out.append(alloc, .{ .cp = cp, .class = class(cp) });
}

fn find(entries: []const [3]u21, cp: u21) ?[3]u21 {
    const i = std.sort.binarySearch([3]u21, entries, cp, struct {
        fn order(key: u21, entry: [3]u21) std.math.Order {
            return std.math.order(key, entry[0]);
        }
    }.order) orelse return null;
    return entries[i];
}

fn class(cp: u21) u8 {
    if (cp < 0x300) return 0;
    const i = std.sort.binarySearch([3]u21, &table.classes, cp, struct {
        fn order(key: u21, range: [3]u21) std.math.Order {
            return if (key < range[0]) .lt else if (key > range[1]) .gt else .eq;
        }
    }.order) orelse return 0;
    return @intCast(table.classes[i][2]);
}

/// Canonical ordering: a stable sort by class of every run of nonzero classes.
fn reorder(scalars: []Scalar) void {
    var i: usize = 0;
    while (i < scalars.len) {
        if (scalars[i].class == 0) {
            i += 1;
            continue;
        }
        const start = i;
        while (i < scalars.len and scalars[i].class != 0) i += 1;
        std.sort.block(Scalar, scalars[start..i], {}, struct {
            fn less(_: void, a: Scalar, b: Scalar) bool {
                return a.class < b.class;
            }
        }.less);
    }
}

/// Canonical composition in place; returns the composed prefix. A mark
/// composes with the last starter unless a scalar between them has class 0
/// or a class at least its own (it is blocked).
fn compose(scalars: []Scalar) []Scalar {
    var n: usize = 0;
    var starter: ?usize = null;
    // Class of the last scalar kept after the starter; null while adjacent.
    var last: ?u8 = null;
    for (scalars) |s| {
        if (starter) |at| {
            const blocked = if (last) |c| c == 0 or c >= s.class else false;
            if (!blocked) if (pair(scalars[at].cp, s.cp)) |composite| {
                scalars[at].cp = composite;
                continue;
            };
        }
        if (s.class == 0) {
            starter = n;
            last = null;
        } else last = s.class;
        scalars[n] = s;
        n += 1;
    }
    return scalars[0..n];
}

fn pair(first: u21, second: u21) ?u21 {
    if (first >= l_base and first < l_base + l_count and second >= v_base and second < v_base + v_count)
        return s_base + ((first - l_base) * v_count + (second - v_base)) * t_count;
    if (first >= s_base and first < s_base + s_count and (first - s_base) % t_count == 0 and
        second > t_base and second < t_base + t_count)
        return first + (second - t_base);
    const Key = struct { u21, u21 };
    const i = std.sort.binarySearch([3]u21, &table.compositions, Key{ first, second }, struct {
        fn order(key: Key, entry: [3]u21) std.math.Order {
            const a = std.math.order(key[0], entry[0]);
            return if (a != .eq) a else std.math.order(key[1], entry[1]);
        }
    }.order) orelse return null;
    return table.compositions[i][2];
}

fn expectNfc(input: []const u8, expected: []const u8) !void {
    const gpa = std.testing.allocator;
    const out = try normalize(gpa, input);
    defer if (out) |o| gpa.free(o);
    try std.testing.expectEqualStrings(expected, out orelse input);
}

test "composition, reordering, blocking, exclusions, and Hangul" {
    try expectNfc("", "");
    try expectNfc("plain ASCII, Latin-1 caf\u{e9}", "plain ASCII, Latin-1 caf\u{e9}");
    try expectNfc("cafe\u{301}", "caf\u{e9}");
    // Reordered below-then-above, then composed with the below mark first.
    try expectNfc("d\u{307}\u{323}", "\u{1e0d}\u{307}");
    try expectNfc("a\u{301}\u{327}", "\u{e1}\u{327}");
    // Singletons decompose and never recompose.
    try expectNfc("\u{212a} \u{2126} \u{212b}", "K \u{3a9} \u{c5}");
    // Composition exclusion (Devanagari QA) and a non-starter decomposition.
    try expectNfc("\u{958}", "\u{915}\u{93c}");
    try expectNfc("\u{1d160}", "\u{1d158}\u{1d165}\u{1d16e}");
    // Conjoining jamo compose; an LV syllable takes a trailing jamo.
    try expectNfc("\u{1100}\u{1161}\u{11a8} \u{ac00}\u{11a8}", "\u{ac01} \u{ac01}");
    // A class-0 scalar between them blocks the second mark.
    try expectNfc("e\u{301}\u{5d0}\u{301}", "\u{e9}\u{5d0}\u{301}");
    // Same class blocks: the second acute stays.
    try expectNfc("e\u{301}\u{301}", "\u{e9}\u{301}");
    try expectNfc(">\u{338}", "\u{226f}");
    try std.testing.expectError(error.InvalidUtf8, normalize(std.testing.allocator, "\xcc"));
}

test "allocation failures leave nothing behind" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, struct {
        fn run(gpa: std.mem.Allocator) !void {
            const out = (try normalize(gpa, "d\u{307}\u{323} \u{ac00}\u{11a8}")).?;
            gpa.free(out);
        }
    }.run, .{});
}

// UAX #15 conformance: every NFC column of the pinned version's
// NormalizationTest.txt, which scripts/tokenizer-nfc.py caches.
test "NormalizationTest.txt conformance when cached" {
    const gpa = std.testing.allocator;
    const io = std.testing.io;
    const path = ".reference/ucd/" ++ table.version ++ "/NormalizationTest.txt";
    const data = std.Io.Dir.cwd().readFileAlloc(io, path, gpa, .limited(8 * 1024 * 1024)) catch |err| switch (err) {
        error.FileNotFound => return error.SkipZigTest,
        else => return err,
    };
    defer gpa.free(data);
    var lines = std.mem.splitScalar(u8, data, '\n');
    var cases: usize = 0;
    while (lines.next()) |line| {
        if (line.len == 0 or line[0] == '#' or line[0] == '@') continue;
        var columns: [5][]u8 = undefined;
        var fields = std.mem.splitScalar(u8, line, ';');
        for (&columns) |*column| column.* = try hexColumn(gpa, fields.next().?);
        defer for (columns) |column| gpa.free(column);
        // c2 == toNFC(c1) == toNFC(c2) == toNFC(c3); c4 == toNFC(c4) == toNFC(c5).
        for ([_]usize{ 0, 1, 2, 3, 4 }, [_]usize{ 1, 1, 1, 3, 3 }) |source, expected| {
            const out = try normalize(gpa, columns[source]);
            defer if (out) |o| gpa.free(o);
            std.testing.expectEqualStrings(columns[expected], out orelse columns[source]) catch |err| {
                std.debug.print("NormalizationTest: {s}\n", .{line});
                return err;
            };
        }
        cases += 1;
    }
    try std.testing.expect(cases > 10_000);
}

/// UTF-8 of one space-separated hex column.
fn hexColumn(gpa: std.mem.Allocator, column: []const u8) ![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(gpa);
    var it = std.mem.tokenizeScalar(u8, column, ' ');
    while (it.next()) |hex| {
        var buf: [4]u8 = undefined;
        const n = try std.unicode.utf8Encode(try std.fmt.parseInt(u21, hex, 16), &buf);
        try out.appendSlice(gpa, buf[0..n]);
    }
    return out.toOwnedSlice(gpa);
}
