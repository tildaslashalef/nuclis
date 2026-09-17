//! The welcome the agent writes above the first prompt: the wordmark when
//! the terminal is wide enough for it, then what this session runs with.
//! It is transcript, not a title bar: it scrolls away with the rest, and the
//! status bar carries the model's name from then on. Rows are plain ASCII
//! (no glyph-set fallback needed); the caller inserts them above the live
//! region and owns the arena they are built in.
const std = @import("std");
const screen = @import("screen.zig");

pub const Row = screen.Row;

/// Six rows, 57 columns: the mark and the width it needs, plus a margin.
pub const wordmark = [_][]const u8{
    ".__   __.  __    __    ______  __       __       _______.",
    "|  \\ |  | |  |  |  |  /      ||  |     |  |     /       |",
    "|   \\|  | |  |  |  | |  ,----'|  |     |  |    |   (----`",
    "|  . `  | |  |  |  | |  |     |  |     |  |     \\   \\",
    "|  |\\   | |  `--'  | |  `----.|  `----.|  | .----)   |",
    "|__| \\__|  \\______/   \\______||_______||__| |_______/",
};
pub const wordmark_width: usize = 57;
const margin: usize = 3;

/// What the welcome says about the session. Strings are borrowed for the
/// call.
pub const Info = struct {
    version: []const u8,
    /// The registry or catalogue name the model was reached through, when
    /// it was one (a path has none).
    name: ?[]const u8,
    /// The artifact's own `general.name`.
    model: []const u8,
    backend: []const u8,
    profile: []const u8,
    /// The profile was forced (`--prompt-profile`, the entry's `profile`),
    /// not selected by the file's template.
    forced: bool,
    ctx_size: usize,
    effort: []const u8,
    /// The workspace, already shortened for display (`~/Code/nuclis`).
    workspace: []const u8,
};

/// The welcome's rows for a terminal `width` cells wide: the wordmark, a
/// blank row, and the three description rows; below the mark's width, the
/// one-line header the surface always had.
pub fn rows(a: std.mem.Allocator, info: Info, width: usize) ![]const Row {
    var out: std.ArrayList(Row) = .empty;
    const wide = width >= wordmark_width + margin;
    if (wide) {
        for (wordmark) |line| try out.append(a, .{ .text = try std.fmt.allocPrint(a, "  {s}", .{line}), .style = .header });
        try out.append(a, .{ .text = "" });
        try out.append(a, .{ .text = try std.fmt.allocPrint(a, "  nuclis agent {s}", .{info.version}), .style = .header });
    } else {
        try out.append(a, .{ .text = try std.fmt.allocPrint(a, " nuclis agent {s}", .{info.version}), .style = .header });
    }
    const indent: []const u8 = if (wide) "  " else " ";
    var model: std.Io.Writer.Allocating = .init(a);
    if (info.name) |name| try model.writer.print("{s} · ", .{name});
    try model.writer.print("{s} · {s} · {s} profile", .{ info.model, info.backend, info.profile });
    if (info.forced) try model.writer.writeAll(" (forced)");
    try out.append(a, .{ .text = try std.fmt.allocPrint(a, "{s}{s}", .{ indent, model.written() }), .style = .dim });
    try out.append(a, .{ .text = try std.fmt.allocPrint(a, "{s}ctx {d} · think {s} · {s}", .{ indent, info.ctx_size, info.effort, info.workspace }), .style = .dim });
    try out.append(a, .{ .text = "" });
    return out.items;
}

/// `path` with the home directory replaced by `~`, when it starts with one.
pub fn shortened(a: std.mem.Allocator, path: []const u8, home: ?[]const u8) ![]const u8 {
    const h = home orelse return path;
    if (h.len == 0 or !std.mem.startsWith(u8, path, h)) return path;
    if (path.len > h.len and path[h.len] != '/') return path;
    return std.mem.concat(a, u8, &.{ "~", path[h.len..] });
}

const testing = std.testing;

test "the wordmark is six rows of the stated width and pure ASCII" {
    try testing.expectEqual(@as(usize, 6), wordmark.len);
    for (wordmark) |line| {
        try testing.expect(line.len <= wordmark_width);
        for (line) |byte| try testing.expect(byte >= 0x20 and byte < 0x7f);
    }
}

test "a wide terminal gets the mark and the description; a narrow one the single line" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const info: Info = .{ .version = "0.2.0-dev", .name = "hauhau", .model = "Gemma4-12B", .backend = "metal", .profile = "gemma4", .forced = true, .ctx_size = 8192, .effort = "low", .workspace = "~/Code/nuclis" };
    const wide = try rows(a, info, 120);
    try testing.expectEqual(@as(usize, 6 + 1 + 1 + 2 + 1), wide.len);
    try testing.expectEqualStrings("  " ++ wordmark[0], wide[0].text);
    try testing.expectEqualStrings("  nuclis agent 0.2.0-dev", wide[7].text);
    try testing.expectEqualStrings("  hauhau · Gemma4-12B · metal · gemma4 profile (forced)", wide[8].text);
    try testing.expectEqualStrings("  ctx 8192 · think low · ~/Code/nuclis", wide[9].text);
    try testing.expectEqualStrings("", wide[10].text);
    const narrow = try rows(a, info, 50);
    try testing.expectEqual(@as(usize, 4), narrow.len);
    try testing.expectEqualStrings(" nuclis agent 0.2.0-dev", narrow[0].text);
    // Without a registry name and without a forced profile the line is shorter.
    var plain = info;
    plain.name = null;
    plain.forced = false;
    const bare = try rows(a, plain, 120);
    try testing.expectEqualStrings("  Gemma4-12B · metal · gemma4 profile", bare[8].text);
}

test "the home directory shortens to a tilde only at a path boundary" {
    const a = testing.allocator;
    const short = try shortened(a, "/Users/x/Code", "/Users/x");
    defer a.free(short);
    try testing.expectEqualStrings("~/Code", short);
    try testing.expectEqualStrings("/Users/xy/Code", try shortened(a, "/Users/xy/Code", "/Users/x"));
    try testing.expectEqualStrings("/tmp", try shortened(a, "/tmp", null));
    const exact = try shortened(a, "/Users/x", "/Users/x");
    defer a.free(exact);
    try testing.expectEqualStrings("~", exact);
}
