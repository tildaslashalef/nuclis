//! The welcome the agent writes above the first prompt: the wordmark and
//! what this session runs with, framed in a box when the terminal is wide
//! enough, one plain line when it is not. It is transcript, not a title bar:
//! it scrolls away with the rest, and the status bar carries the model's name
//! from then on. The caller inserts the rows above the live region and owns
//! the arena they are built in.
const std = @import("std");
const screen = @import("screen.zig");
const theme = @import("theme.zig");
const view = @import("view.zig");

pub const Row = screen.Row;

/// Six rows, 57 columns: the mark and the width it needs.
pub const wordmark = [_][]const u8{
    ".__   __.  __    __    ______  __       __       _______.",
    "|  \\ |  | |  |  |  |  /      ||  |     |  |     /       |",
    "|   \\|  | |  |  |  | |  ,----'|  |     |  |    |   (----`",
    "|  . `  | |  |  |  | |  |     |  |     |  |     \\   \\",
    "|  |\\   | |  `--'  | |  `----.|  `----.|  | .----)   |",
    "|__| \\__|  \\______/   \\______||_______||__| |_______/",
};
pub const wordmark_width: usize = 57;
/// Cells the frame adds to its content: the two edges and a space inside each.
const frame_cells: usize = 4;
/// Columns kept free to the right of the box.
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

/// The welcome's rows for a terminal `width` cells wide: the box around the
/// wordmark, a blank row, and the three description rows, then a blank row
/// under the box; below `wordmark_width + frame_cells + margin` columns, the
/// one-line header the surface always had. Box rows are raw (they carry
/// their own styles); every text in them is sanitized here.
pub fn rows(a: std.mem.Allocator, info: Info, width: usize, th: theme.Theme) ![]const Row {
    var out: std.ArrayList(Row) = .empty;
    const model_line = try modelLine(a, info);
    const settings_line = try std.fmt.allocPrint(a, "ctx {d} · think {s} · {s}", .{ info.ctx_size, info.effort, info.workspace });
    const version_line = try std.fmt.allocPrint(a, "nuclis agent {s}", .{info.version});
    if (width < wordmark_width + frame_cells + margin) {
        try out.append(a, .{ .text = try std.fmt.allocPrint(a, " nuclis agent {s}", .{info.version}), .style = .header });
        try out.append(a, .{ .text = try std.fmt.allocPrint(a, " {s}", .{model_line}), .style = .dim });
        try out.append(a, .{ .text = try std.fmt.allocPrint(a, " {s}", .{settings_line}), .style = .dim });
        try out.append(a, .{ .text = "" });
        return out.items;
    }
    // The content is as wide as its widest row, never wider than the space
    // the terminal leaves for it; a longer row is cut, not wrapped.
    const inner = @min(@max(wordmark_width, @max(view.width(model_line), view.width(settings_line))), width - frame_cells - margin);
    const gl = th.glyphs();
    try out.append(a, try edge(a, gl.box_tl, gl.box_h, gl.box_tr, inner, th));
    for (wordmark) |line| try out.append(a, try framed(a, line, .header, inner, th));
    try out.append(a, try framed(a, "", null, inner, th));
    try out.append(a, try framed(a, version_line, .header, inner, th));
    try out.append(a, try framed(a, model_line, .dim, inner, th));
    try out.append(a, try framed(a, settings_line, .dim, inner, th));
    try out.append(a, try edge(a, gl.box_bl, gl.box_h, gl.box_br, inner, th));
    try out.append(a, .{ .text = "" });
    return out.items;
}

fn modelLine(a: std.mem.Allocator, info: Info) ![]const u8 {
    var w: std.Io.Writer.Allocating = .init(a);
    if (info.name) |name| try w.writer.print("{s} · ", .{name});
    try w.writer.print("{s} · {s} · {s} profile", .{ info.model, info.backend, info.profile });
    if (info.forced) try w.writer.writeAll(" (forced)");
    return w.written();
}

/// The top or bottom edge: a corner, `inner + 2` horizontal cells, a corner.
fn edge(a: std.mem.Allocator, left: []const u8, h: []const u8, right: []const u8, inner: usize, th: theme.Theme) !Row {
    var w: std.Io.Writer.Allocating = .init(a);
    try w.writer.print("  {s}{s}", .{ th.paint(.header), left });
    for (0..inner + 2) |_| try w.writer.writeAll(h);
    try w.writer.print("{s}{s}", .{ right, theme.reset });
    return .{ .text = w.written(), .raw = true };
}

/// One content row: the edge, a space, the text in its style cut and padded
/// to `inner` cells, a space, the edge.
fn framed(a: std.mem.Allocator, text: []const u8, style: ?theme.Style, inner: usize, th: theme.Theme) !Row {
    var w: std.Io.Writer.Allocating = .init(a);
    const gl = th.glyphs();
    const cut = try view.fit(a, text, inner);
    try w.writer.print("  {s}{s}{s} ", .{ th.paint(.header), gl.box_v, theme.reset });
    if (style) |s| try w.writer.writeAll(th.paint(s));
    try view.safe(&w.writer, cut);
    if (style != null) try w.writer.writeAll(theme.reset);
    for (0..inner - view.width(cut)) |_| try w.writer.writeByte(' ');
    try w.writer.print(" {s}{s}{s}", .{ th.paint(.header), gl.box_v, theme.reset });
    return .{ .text = w.written(), .raw = true };
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

/// A row's text without its escapes: the goldens pin the cells, the styles
/// are pinned in theme.zig.
fn stripped(a: std.mem.Allocator, text: []const u8) ![]const u8 {
    var out: std.ArrayList(u8) = .empty;
    var i: usize = 0;
    while (i < text.len) : (i += 1) {
        if (text[i] == 0x1b) {
            i += 1;
            while (i < text.len and text[i] != 'm') i += 1;
            continue;
        }
        try out.append(a, text[i]);
    }
    return out.items;
}

const sample: Info = .{ .version = "0.2.0-dev", .name = "hauhau", .model = "Gemma4-12B", .backend = "metal", .profile = "gemma4", .forced = true, .ctx_size = 8192, .effort = "low", .workspace = "~/Code/nuclis" };

test "a wide terminal gets the boxed mark and the description; a narrow one the plain lines" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const plain: theme.Theme = .{ .kind = .plain };
    const wide = try rows(a, sample, 120, plain);
    // Edge, six mark rows, blank, three description rows, edge, blank.
    try testing.expectEqual(@as(usize, 1 + 6 + 1 + 3 + 1 + 1), wide.len);
    const inner = wordmark_width;
    for (wide[0 .. wide.len - 1]) |row| {
        try testing.expect(row.raw);
        try testing.expectEqual(inner + frame_cells + 2, view.styledWidth(row.text));
    }
    try testing.expectEqualStrings("  ╭" ++ ("─" ** 59) ++ "╮", try stripped(a, wide[0].text));
    try testing.expectEqualStrings("  │ " ++ wordmark[0] ++ " │", try stripped(a, wide[1].text));
    try testing.expectEqualStrings("  │ " ++ (" " ** 57) ++ " │", try stripped(a, wide[7].text));
    try testing.expectEqualStrings("  │ nuclis agent 0.2.0-dev" ++ (" " ** 35) ++ " │", try stripped(a, wide[8].text));
    try testing.expectEqualStrings("  │ hauhau · Gemma4-12B · metal · gemma4 profile (forced)" ++ (" " ** 4) ++ " │", try stripped(a, wide[9].text));
    try testing.expectEqualStrings("  │ ctx 8192 · think low · ~/Code/nuclis" ++ (" " ** 21) ++ " │", try stripped(a, wide[10].text));
    try testing.expectEqualStrings("  ╰" ++ ("─" ** 59) ++ "╯", try stripped(a, wide[11].text));
    try testing.expectEqualStrings("", wide[12].text);

    // 80 columns still fits the 61-cell box; 60 does not.
    try testing.expectEqual(@as(usize, 13), (try rows(a, sample, 80, plain)).len);
    const narrow = try rows(a, sample, 60, plain);
    try testing.expectEqual(@as(usize, 4), narrow.len);
    try testing.expectEqualStrings(" nuclis agent 0.2.0-dev", narrow[0].text);
    try testing.expectEqualStrings(" hauhau · Gemma4-12B · metal · gemma4 profile (forced)", narrow[1].text);
    try testing.expect(!narrow[0].raw);
}

test "a description wider than the mark widens the box, up to the terminal" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    var long = sample;
    const path = "~/Code/a/very/long/path/that/goes/on/and/on/for/a/while/more";
    long.workspace = path;
    const settings = "ctx 8192 · think low · " ++ path;
    const wide = try rows(a, long, 160, .{ .kind = .plain });
    try testing.expectEqual(settings.len - 2 + frame_cells + 2, view.styledWidth(wide[0].text)); // two `·`, 2 bytes and 1 cell each
    try testing.expectEqualStrings("  │ " ++ settings ++ " │", try stripped(a, wide[10].text));
    // Cut at the terminal's width minus the margin: every row still ends
    // with the edge and control bytes never reach the row.
    long.workspace = "~/\x1b[2J" ++ ("x" ** 200);
    const cut = try rows(a, long, 100, .{ .kind = .plain });
    try testing.expectEqual(100 - margin + 2, view.styledWidth(cut[10].text));
    try testing.expect(std.mem.indexOf(u8, cut[10].text, "\x1b[2J") == null);
    try testing.expect(std.mem.endsWith(u8, try stripped(a, cut[10].text), " │"));
}

test "the ascii set frames with plus and pipe, and a styled theme colours the frame" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const ascii = try rows(a, sample, 120, .{ .kind = .plain, .glyph_set = .ascii });
    try testing.expectEqualStrings("  +" ++ ("-" ** 59) ++ "+", try stripped(a, ascii[0].text));
    try testing.expect(std.mem.startsWith(u8, try stripped(a, ascii[1].text), "  | "));
    const styled = try rows(a, sample, 120, .{ .kind = .truecolor });
    try testing.expect(std.mem.indexOf(u8, styled[1].text, "\x1b[") != null);
    try testing.expectEqual(wordmark_width + frame_cells + 2, view.styledWidth(styled[1].text));
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
