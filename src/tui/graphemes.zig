//! Grapheme clusters and terminal width.
//!
//! The terminal is addressed in cells, but text is made of *grapheme clusters*:
//! a base plus its combining marks, an emoji ZWJ sequence, a flag, or a Hangul
//! syllable. Measuring or breaking by code point splits those apart — a family
//! emoji becomes four cells wide, Backspace deletes half a cluster, a ZWJ
//! sequence wraps mid-cluster. This module is the one place that knows the
//! Unicode rules (UAX #29 boundaries, UAX #11 width) so `view.zig` and
//! `editor.zig` can stay about layout.
//!
//! The property data is [`graphemes_table.zig`](graphemes_table.zig), generated
//! from a pinned UCD by `scripts/grapheme-table.py`. Nothing outside this file
//! reads the table, so replacing the data layer is a local change.
//!
//! Width policy, pinned by the tests: a cluster that carries VS15 is one cell;
//! an emoji-presentation cluster, a skin-tone modifier, or a two-regional-
//! indicator pair is two; otherwise the cluster sums the widths of its code
//! points, where combining marks, joiners, variation selectors, Hangul V/T
//! jamo, prepend characters, and default-ignorables are zero. East Asian
//! *Ambiguous* is treated as one cell. This is a terminal approximation, not a
//! claim about every terminal.
//!
//! Cost. Boundary state is recomputed per cluster, and `clusterPrev` scans
//! from the start of the text, so editing is O(n) in the buffer. Buffers here
//! are bounded by the editor's 128 KiB input limit; a wrap pass scans once.
const std = @import("std");
const table = @import("graphemes_table.zig");

pub const Gcb = table.Gcb;
pub const Incb = table.Incb;
pub const Props = table.Props;
pub const unicode_version = table.unicode_version;

/// The properties of one code point, by binary search over the interval table.
pub fn props(cp: u21) Props {
    const list = &table.intervals;
    var lo: usize = 0;
    var hi: usize = list.len;
    while (lo + 1 < hi) {
        const mid = lo + (hi - lo) / 2;
        if (list[mid].start <= cp) lo = mid else hi = mid;
    }
    return list[lo].props;
}

/// One decoded code point and the number of bytes it took.
pub const Decoded = struct { cp: u21, len: usize };

/// Decodes the code point at `offset`. A malformed or truncated sequence is
/// U+FFFD in one byte, so callers always make progress.
pub fn decode(text: []const u8, offset: usize) Decoded {
    const first = text[offset];
    const n = std.unicode.utf8ByteSequenceLength(first) catch return .{ .cp = 0xfffd, .len = 1 };
    if (offset + n > text.len) return .{ .cp = 0xfffd, .len = 1 };
    const cp = std.unicode.utf8Decode(text[offset..][0..n]) catch return .{ .cp = 0xfffd, .len = 1 };
    return .{ .cp = cp, .len = n };
}

/// UAX #29 boundary state carried across a cluster. It is reset at every
/// cluster start: the rules that use it (RI parity, GB11) only look within a
/// cluster, so a break makes the state irrelevant.
const State = struct {
    /// Whether the current run of regional indicators has odd length.
    ri_odd: bool = false,
    /// `Extended_Pictographic Extend* ZWJ` tracking for GB11.
    ep: enum { none, ep, ep_extend, zwj } = .none,
    /// `Consonant (Extend|Linker)* Linker (Extend|Linker)*` for GB9c.
    incb: enum { none, consonant, chain } = .none,
};

/// Folds one code point into the boundary state. `joined` is whether it formed
/// a cluster with the preceding code point (false at a cluster start).
fn advance(state: *State, p: Props, joined: bool) void {
    state.ri_odd = if (p.gcb == .ri) (if (joined) !state.ri_odd else true) else false;

    if (p.extended_pictographic) {
        state.ep = .ep;
    } else switch (p.gcb) {
        .extend => state.ep = if (state.ep == .ep or state.ep == .ep_extend) .ep_extend else .none,
        .zwj => state.ep = if (state.ep == .ep or state.ep == .ep_extend) .zwj else .none,
        else => state.ep = .none,
    }

    switch (p.incb) {
        .consonant => state.incb = .consonant,
        .linker => state.incb = if (state.incb == .none) .none else .chain,
        .extend => {},
        .none => state.incb = .none,
    }
}

/// UAX #29 GB3–GB13, in rule order: does a cluster boundary fall between the
/// two code points? `state` describes the run ending at `prev_cp`.
fn breakBefore(prev_cp: u21, cur_cp: u21, state: State) bool {
    const p = props(prev_cp);
    const c = props(cur_cp);
    if (p.gcb == .cr and c.gcb == .lf) return false; // GB3
    if (p.gcb == .control or p.gcb == .cr or p.gcb == .lf) return true; // GB4
    if (c.gcb == .control or c.gcb == .cr or c.gcb == .lf) return true; // GB5
    if (p.gcb == .l and (c.gcb == .l or c.gcb == .v or c.gcb == .lv or c.gcb == .lvt)) return false; // GB6
    if ((p.gcb == .lv or p.gcb == .v) and (c.gcb == .v or c.gcb == .t)) return false; // GB7
    if ((p.gcb == .lvt or p.gcb == .t) and c.gcb == .t) return false; // GB8
    if (c.gcb == .extend or c.gcb == .zwj) return false; // GB9
    if (c.gcb == .spacing_mark) return false; // GB9a
    if (p.gcb == .prepend) return false; // GB9b
    if (state.incb == .chain and c.incb == .consonant) return false; // GB9c
    if (p.gcb == .zwj and c.extended_pictographic and state.ep == .zwj) return false; // GB11
    if (p.gcb == .ri and c.gcb == .ri and state.ri_odd) return false; // GB12/13
    return true;
}

/// The byte offset just past the cluster starting at `from`. `from` must be
/// a code-point boundary below `text.len`.
fn clusterEnd(text: []const u8, from: usize) usize {
    const first = decode(text, from);
    var state: State = .{};
    advance(&state, props(first.cp), false);
    var prev = first.cp;
    var i = from + first.len;
    while (i < text.len) {
        const cur = decode(text, i);
        if (breakBefore(prev, cur.cp, state)) break;
        advance(&state, props(cur.cp), true);
        prev = cur.cp;
        i += cur.len;
    }
    return i;
}

/// Byte offset of the next cluster boundary at or after `from`.
pub fn clusterNext(text: []const u8, from: usize) usize {
    return if (from >= text.len) text.len else clusterEnd(text, from);
}

/// Byte offset of the cluster boundary at or before `from`. Scans from the
/// start, so it is O(n); see the module comment.
pub fn clusterPrev(text: []const u8, from: usize) usize {
    if (from == 0) return 0;
    var i: usize = 0;
    while (i < from) {
        const end = clusterEnd(text, i);
        if (end >= from) break;
        i = end;
    }
    return i;
}

/// Terminal cells one cluster occupies (see the module comment for the policy).
pub fn clusterWidth(cluster: []const u8) usize {
    var emitted: usize = 0;
    var emoji = false;
    var vs15 = false;
    var ris: usize = 0;
    var after_zwj = false;
    var i: usize = 0;
    while (i < cluster.len) {
        const d = decode(cluster, i);
        const p = props(d.cp);
        i += d.len;
        if (d.cp == 0xfe0e) {
            vs15 = true;
            continue;
        }
        if (d.cp == 0xfe0f) {
            emoji = true;
            continue;
        }
        if (p.emoji_modifier) emoji = true;
        if (p.gcb == .ri) ris += 1;
        if (p.gcb == .zwj) {
            after_zwj = true;
            continue;
        }
        // Regional indicators are Emoji_Presentation too; leave them to the
        // pair rule so a lone one stays one cell.
        if (p.emoji_presentation and p.gcb != .ri) emoji = true;
        if (after_zwj or zeroWidth(p)) continue;
        emitted += if (p.wide) 2 else 1;
    }
    if (vs15) return 1;
    if (ris == 2) return 2;
    if (emoji) return @max(emitted, 2);
    return emitted;
}

/// Whether a code point contributes no cells inside a cluster.
fn zeroWidth(p: Props) bool {
    if (p.ignorable) return true;
    return switch (p.gcb) {
        .control, .cr, .lf, .extend, .zwj, .prepend, .v, .t => true,
        else => false,
    };
}

/// Terminal cells a whole string occupies.
pub fn width(text: []const u8) usize {
    var total: usize = 0;
    var i: usize = 0;
    while (i < text.len) {
        const end = clusterEnd(text, i);
        total += clusterWidth(text[i..end]);
        i = end;
    }
    return total;
}

/// Forward iterator over clusters. Borrows `text`.
pub const Iterator = struct {
    text: []const u8,
    i: usize = 0,

    pub fn next(self: *Iterator) ?[]const u8 {
        if (self.i >= self.text.len) return null;
        const end = clusterNext(self.text, self.i);
        const cluster = self.text[self.i..end];
        self.i = end;
        return cluster;
    }
};

test "width policy: emoji, flags, combining, Hangul, CJK" {
    try std.testing.expectEqual(@as(usize, 1), width("a"));
    try std.testing.expectEqual(@as(usize, 4), width("ab中"));
    try std.testing.expectEqual(@as(usize, 1), width("e\u{301}"));
    try std.testing.expectEqual(@as(usize, 2), width("👋🏿")); // wave + skin tone
    try std.testing.expectEqual(@as(usize, 2), width("👩‍🚀")); // ZWJ family
    try std.testing.expectEqual(@as(usize, 2), width("👨🏻‍❤️‍👨🏿"));
    try std.testing.expectEqual(@as(usize, 2), width("🏳️‍🌈"));
    try std.testing.expectEqual(@as(usize, 2), width("🇨🇭")); // flag pair
    try std.testing.expectEqual(@as(usize, 1), width("🇨")); // lone regional indicator
    try std.testing.expectEqual(@as(usize, 1), width("☺︎")); // VS15 keeps text presentation
    try std.testing.expectEqual(@as(usize, 2), width("❤️")); // VS16 promotes to emoji
    try std.testing.expectEqual(@as(usize, 1), width("❤"));
    try std.testing.expectEqual(@as(usize, 2), width("\u{1100}\u{1161}\u{11a8}")); // Hangul L V T
    try std.testing.expectEqual(@as(usize, 2), width("가"));
    try std.testing.expectEqual(@as(usize, 0), width("\u{200b}")); // zero-width space
    try std.testing.expectEqual(@as(usize, 4), width("a👩‍🚀b"));
}

test "clusters are split at UAX #29 boundaries" {
    // A ZWJ sequence is one cluster; the next character starts a new one.
    try std.testing.expectEqual(@as(usize, 11), clusterNext("👩‍🚀x", 0));
    // Base + combining mark.
    try std.testing.expectEqual(@as(usize, 3), clusterNext("e\u{301}b", 0));
    // CRLF is a single cluster.
    try std.testing.expectEqual(@as(usize, 2), clusterNext("\r\nx", 0));
    // A flag pair is one cluster; a third indicator starts another.
    const ri = "🇨🇭🇩"; // CH + D
    try std.testing.expectEqual(@as(usize, 8), clusterNext(ri, 0));
    try std.testing.expectEqual(@as(usize, 0), clusterPrev(ri, 8)); // before the pair
    try std.testing.expectEqual(@as(usize, 8), clusterPrev(ri, 12)); // before the third
    try std.testing.expectEqual(@as(usize, 1), clusterPrev("abc", 2));
    try std.testing.expectEqual(@as(usize, 2), clusterPrev("abc", 3));
}

test "the full Unicode grapheme break test passes" {
    const alloc = std.testing.allocator;
    const content = @embedFile("fixtures/GraphemeBreakTest.txt");
    var lines = std.mem.splitScalar(u8, content, '\n');
    var checked: usize = 0;
    while (lines.next()) |line| {
        const comment = std.mem.indexOfScalar(u8, line, '#') orelse line.len;
        const body = std.mem.trim(u8, line[0..comment], " \t\r");
        if (body.len == 0 or body[0] == '@') continue;

        // The marker before a code point says whether a boundary falls there.
        // Group the code points between the ÷ markers into expected clusters.
        var expected: std.ArrayList(std.ArrayList(u21)) = .empty;
        defer {
            for (expected.items) |*cluster| cluster.deinit(alloc);
            expected.deinit(alloc);
        }
        var current: std.ArrayList(u21) = .empty;
        defer current.deinit(alloc);
        var tokens = std.mem.tokenizeAny(u8, body, " \t");
        while (tokens.next()) |token| {
            if (std.mem.eql(u8, token, "\u{00f7}")) { // ÷
                if (current.items.len != 0) try expected.append(alloc, current);
                current = .empty;
            } else if (std.mem.eql(u8, token, "\u{00d7}")) { // ×
                // no boundary here
            } else {
                try current.append(alloc, try std.fmt.parseInt(u21, token, 16));
            }
        }
        if (current.items.len != 0) try expected.append(alloc, current);

        var bytes: std.ArrayList(u8) = .empty;
        defer bytes.deinit(alloc);
        for (expected.items) |cluster| for (cluster.items) |cp| {
            var buffer: [4]u8 = undefined;
            const n: usize = try std.unicode.utf8Encode(cp, &buffer);
            try bytes.appendSlice(alloc, buffer[0..n]);
        };

        var index: usize = 0;
        var it: Iterator = .{ .text = bytes.items };
        while (it.next()) |cluster| {
            const want = expected.items[index];
            var got: std.ArrayList(u21) = .empty;
            defer got.deinit(alloc);
            var j: usize = 0;
            while (j < cluster.len) {
                const d = decode(cluster, j);
                try got.append(alloc, d.cp);
                j += d.len;
            }
            if (!std.mem.eql(u21, want.items, got.items)) {
                std.debug.print("GraphemeBreakTest mismatch: {s}\n", .{body});
                return error.TestUnexpectedResult;
            }
            index += 1;
        }
        if (index != expected.items.len) {
            std.debug.print("GraphemeBreakTest cluster count: {s}\n", .{body});
            return error.TestUnexpectedResult;
        }
        checked += 1;
    }
    // UCD 17.0.0 ships 766 cases; every one must pass.
    try std.testing.expect(checked > 700);
}
