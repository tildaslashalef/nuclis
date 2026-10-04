//! The one fuzzy matcher of the surface: `@path` completion and every
//! picker's search box rank with it, so a query finds the same things in
//! both. Pure; no allocation.
const std = @import("std");

/// How well `query` matches `path` as an in-order, case-insensitive
/// subsequence, or null when it does not. Matched inside the last component
/// when it can be, else across the path; consecutive characters, a
/// character starting a word, and a basename that starts with the query
/// score higher, and a longer path slightly lower.
pub fn score(query: []const u8, path: []const u8) ?i32 {
    if (query.len == 0) return 0;
    const trimmed = std.mem.trimEnd(u8, path, "/");
    const base = if (std.mem.lastIndexOfScalar(u8, trimmed, '/')) |i| i + 1 else 0;
    var total: i32 = undefined;
    if (subsequence(query, trimmed, base)) |s| {
        total = s + 10;
        if (std.ascii.startsWithIgnoreCase(trimmed[base..], query)) total += 20;
    } else total = subsequence(query, trimmed, 0) orelse return null;
    return total - @as(i32, @intCast(@min(trimmed.len, 400) / 8));
}

fn subsequence(query: []const u8, text: []const u8, from: usize) ?i32 {
    var total: i32 = 0;
    var at = from;
    var previous: ?usize = null;
    for (query) |c| {
        const lower = std.ascii.toLower(c);
        while (at < text.len and std.ascii.toLower(text[at]) != lower) at += 1;
        if (at == text.len) return null;
        total += 1;
        if (previous != null and previous.? + 1 == at) total += 5;
        if (at == 0 or std.mem.indexOfScalar(u8, "/_-. ", text[at - 1]) != null) total += 8;
        previous = at;
        at += 1;
    }
    return total;
}

const testing = std.testing;

test "fuzzy scores prefer word starts and consecutive characters" {
    try testing.expect(score("faq", "docs/faq.md").? > score("faq", "docs/fix_a_queue.md").?);
    try testing.expect(score("rd", "README.md").? > score("rd", "src/bird.py").?);
    try testing.expect(score("xyz", "docs/faq.md") == null);
    try testing.expectEqual(@as(?i32, 0), score("", "anything"));
}
