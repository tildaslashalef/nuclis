//! The kit's blocks: the header line, aligned detail sections, the plan
//! shown before work that downloads or writes, status lines, and the
//! summary of a run. All strings must already be clean (`text.clean`).
const std = @import("std");
const text = @import("text.zig");
const style = @import("style.zig");
const Ui = @import("Ui.zig");

/// `◆ <subject> · <context>`: nuclis's mark, then the subject of a view
/// about one thing (a model, a file) in the title colour, then the context
/// muted, and a blank line. The command is not echoed.
pub fn header(w: *std.Io.Writer, ui: Ui, subject: ?[]const u8, context: []const u8) std.Io.Writer.Error!void {
    try w.print("{s}{s}{s}", .{ ui.on(.title), ui.glyphs.mark, ui.off(.title) });
    if (subject) |s| {
        try w.print(" {s}{s}{s}", .{ ui.on(.title), s, ui.off(.title) });
        if (context.len > 0) try w.print(" {s}", .{ui.glyphs.dot});
    }
    if (context.len > 0) try w.print(" {s}{s}{s}", .{ ui.on(.muted), context, ui.off(.muted) });
    try w.writeAll("\n\n");
}

pub const Field = struct {
    key: []const u8,
    value: []const u8,
    role: style.Role = .text,
};

pub const Section = struct {
    title: ?[]const u8 = null,
    fields: []const Field,
};

/// Aligned key/value sections, keys muted in one column across all of
/// them; a value longer than the terminal wraps at spaces under its column,
/// and an embedded newline continues there too.
pub fn detail(w: *std.Io.Writer, ui: Ui, sections: []const Section) std.Io.Writer.Error!void {
    var key_w: usize = 0;
    for (sections) |s| {
        for (s.fields) |f| key_w = @max(key_w, text.width(f.key));
    }
    for (sections, 0..) |s, si| {
        if (si > 0) try w.writeByte('\n');
        const indent: usize = if (s.title != null) 2 else 0;
        if (s.title) |t| try w.print("{s}{s}{s}\n", .{ ui.on(.title), t, ui.off(.title) });
        for (s.fields) |f| {
            try w.splatByteAll(' ', indent);
            try w.writeAll(ui.on(.muted));
            try text.writeFit(w, f.key, key_w, .left, "", true);
            try w.writeAll(ui.off(.muted));
            try w.writeAll("  ");
            const role: style.Role = if (std.mem.eql(u8, f.value, "--")) .muted else f.role;
            const col = indent + key_w + 2;
            const room: ?usize = if (ui.width) |cols| (if (cols > col + 20) cols - col else null) else null;
            var lines = std.mem.splitScalar(u8, f.value, '\n');
            var first = true;
            while (lines.next()) |whole| {
                var rest = whole;
                while (true) {
                    if (!first) {
                        try w.writeByte('\n');
                        try w.splatByteAll(' ', col);
                    }
                    first = false;
                    const cut = if (room) |r| wrapAt(rest, r) else rest.len;
                    try w.print("{s}{s}{s}", .{ ui.on(role), rest[0..cut], ui.off(role) });
                    if (cut >= rest.len) break;
                    rest = std.mem.trimStart(u8, rest[cut..], " ");
                }
            }
            try w.writeByte('\n');
        }
    }
}

/// Where to break `line` to fit `room` cells: the last space that fits,
/// else the whole line (a long word overflows rather than being cut).
fn wrapAt(row: []const u8, room: usize) usize {
    if (text.width(row) <= room) return row.len;
    var best: ?usize = null;
    var i: usize = 0;
    while (std.mem.indexOfScalarPos(u8, row, i, ' ')) |sp| : (i = sp + 1) {
        if (text.width(row[0..sp]) > room) break;
        best = sp;
    }
    return best orelse row.len;
}

pub const Change = struct {
    field: []const u8,
    /// Null: absent until now.
    from: ?[]const u8,
    /// Null: removed.
    to: ?[]const u8,
};

pub const Plan = struct {
    /// What will happen: "Pull the catalogue".
    action: []const u8,
    target: ?[]const u8 = null,
    changes: []const Change = &.{},
    impact: []const []const u8 = &.{},
    notes: []const []const u8 = &.{},
};

/// The preview shown before work that downloads or writes.
pub fn plan(w: *std.Io.Writer, ui: Ui, pl: Plan) std.Io.Writer.Error!void {
    try w.print("{s}Plan {s} {s}{s}\n", .{ ui.on(.title), ui.glyphs.dot, pl.action, ui.off(.title) });
    if (pl.target) |t| {
        try label(w, ui, "Target");
        try w.print("{s}\n", .{t});
    }
    var field_w: usize = 0;
    for (pl.changes) |c| field_w = @max(field_w, text.width(c.field));
    for (pl.changes, 0..) |c, i| {
        try label(w, ui, if (i == 0) "Change" else "");
        try text.writeFit(w, c.field, field_w, .left, "", true);
        try w.writeAll("  ");
        try w.print("{s}{s}{s} {s} ", .{ ui.on(.muted), c.from orelse "(none)", ui.off(.muted), ui.glyphs.arrow });
        const to_role: style.Role = if (c.to == null) .fail else .changed;
        try w.print("{s}{s}{s}\n", .{ ui.on(to_role), c.to orelse "(removed)", ui.off(to_role) });
    }
    for (pl.impact, 0..) |item, i| {
        try label(w, ui, if (i == 0) "Impact" else "");
        try w.print("{s}\n", .{item});
    }
    for (pl.notes, 0..) |item, i| {
        try label(w, ui, if (i == 0) "Note" else "");
        try w.print("{s}{s}{s}\n", .{ ui.on(.muted), item, ui.off(.muted) });
    }
}

fn label(w: *std.Io.Writer, ui: Ui, name: []const u8) std.Io.Writer.Error!void {
    try w.print("  {s}", .{ui.on(.muted)});
    try text.writeFit(w, name, "Target".len, .left, "", true);
    try w.print("{s}  ", .{ui.off(.muted)});
}

/// `✓ word`: a status glyph and its word in the status colour, unpadded.
pub fn status(w: *std.Io.Writer, ui: Ui, s: style.Status, word: ?[]const u8) std.Io.Writer.Error!void {
    try w.print("{s}{s} {s}{s}", .{ ui.on(s.role()), s.glyph(ui.glyphs), word orelse s.word(), ui.off(s.role()) });
}

/// A status as table text (`✓ installed`), allocated in `arena`.
pub fn statusText(arena: std.mem.Allocator, ui: Ui, s: style.Status, word: ?[]const u8) ![]const u8 {
    return std.fmt.allocPrint(arena, "{s} {s}", .{ s.glyph(ui.glyphs), word orelse s.word() });
}

/// One result line: `✓ <subject>  <detail>`, the detail muted.
pub fn result(w: *std.Io.Writer, ui: Ui, s: style.Status, subject: []const u8, detail_text: []const u8) std.Io.Writer.Error!void {
    try w.print("{s}{s}{s} {s}", .{ ui.on(s.role()), s.glyph(ui.glyphs), ui.off(s.role()), subject });
    if (detail_text.len > 0) try w.print("  {s}{s}{s}", .{ ui.on(.muted), detail_text, ui.off(.muted) });
    try w.writeByte('\n');
}

/// A hint after a result: `→ <text>`, muted, the command in the id colour.
pub fn hint(w: *std.Io.Writer, ui: Ui, before: []const u8, command: []const u8, after: []const u8) std.Io.Writer.Error!void {
    try w.print("{s}{s} {s}{s}{s}{s}{s}{s}{s}{s}\n", .{ ui.on(.muted), ui.glyphs.arrow, before, ui.off(.muted), ui.on(.id), command, ui.off(.id), ui.on(.muted), after, ui.off(.muted) });
}

pub const Tally = struct { ok: usize = 0, warnings: usize = 0, failed: usize = 0 };

/// The run's summary: `12 ok, 1 warning, 0 failed`, each count in its
/// colour when it is not zero.
pub fn summary(w: *std.Io.Writer, ui: Ui, t: Tally) std.Io.Writer.Error!void {
    const ok_role: style.Role = if (t.ok > 0) .ok else .muted;
    const warn_role: style.Role = if (t.warnings > 0) .warn else .muted;
    const fail_role: style.Role = if (t.failed > 0) .fail else .muted;
    try w.print("{s}{d} ok{s}, {s}{d} warning{s}{s}, {s}{d} failed{s}\n", .{
        ui.on(ok_role),    t.ok,             ui.off(ok_role),
        ui.on(warn_role),  t.warnings,       if (t.warnings == 1) "" else "s",
        ui.off(warn_role), ui.on(fail_role), t.failed,
        ui.off(fail_role),
    });
}

const testing = std.testing;

test "header: the mark, the subject, and the context" {
    var buf: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buf.deinit();
    try header(&buf.writer, .plain, null, "~/.nuclis/models · 16.46 GB");
    try testing.expectEqualStrings("◆ ~/.nuclis/models · 16.46 GB\n\n", buf.written());
    buf.clearRetainingCapacity();
    try header(&buf.writer, .plain, "qwen3.8-27b", "unsloth/Qwen3.8-27B-GGUF @ ce331efc");
    try testing.expectEqualStrings("◆ qwen3.8-27b · unsloth/Qwen3.8-27B-GGUF @ ce331efc\n\n", buf.written());
    buf.clearRetainingCapacity();
    try header(&buf.writer, .{ .glyphs = &style.ascii }, "x", "");
    try testing.expectEqualStrings("nuclis x\n\n", buf.written());
}

test "detail: keys aligned across sections, long values wrap under their column" {
    var buf: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buf.deinit();
    try detail(&buf.writer, .plain, &.{
        .{ .title = "Model", .fields = &.{
            .{ .key = "Name", .value = "qwen3.8-27b" },
            .{ .key = "Status", .value = "✓ installed", .role = .ok },
        } },
        .{ .title = "Files", .fields = &.{
            .{ .key = "Companions", .value = "mmproj\ndflash2" },
            .{ .key = "Registry", .value = "--" },
        } },
    });
    try testing.expectEqualStrings(
        \\Model
        \\  Name        qwen3.8-27b
        \\  Status      ✓ installed
        \\
        \\Files
        \\  Companions  mmproj
        \\              dflash2
        \\  Registry    --
        \\
    , buf.written());
    buf.clearRetainingCapacity();
    try detail(&buf.writer, .{ .width = 40 }, &.{.{ .fields = &.{
        .{ .key = "Note", .value = "one two three four five six seven eight nine ten" },
    } }});
    try testing.expectEqualStrings(
        \\Note  one two three four five six seven
        \\      eight nine ten
        \\
    , buf.written());
}

test "plan, lines, hints, and the summary" {
    var buf: std.Io.Writer.Allocating = .init(testing.allocator);
    defer buf.deinit();
    const w = &buf.writer;
    try plan(w, .plain, .{
        .action = "Write the configuration",
        .target = "~/.nuclis/nuclis.json",
        .changes = &.{
            .{ .field = "engine.model", .from = "gemma-4-e4b-qat", .to = "qwen3.8-27b" },
            .{ .field = "models.old", .from = "acme/old.gguf", .to = null },
        },
        .notes = &.{"Nothing is downloaded."},
    });
    try result(w, .plain, .ok, "qwen3.8-27b", "19.55 GB");
    try result(w, .plain, .failed, "laya", "the Hub did not answer");
    try hint(w, .plain, "next: ", "nuclis config init", "");
    try summary(w, .plain, .{ .ok = 6, .failed = 1 });
    try testing.expectEqualStrings(
        \\Plan · Write the configuration
        \\  Target  ~/.nuclis/nuclis.json
        \\  Change  engine.model  gemma-4-e4b-qat → qwen3.8-27b
        \\          models.old    acme/old.gguf → (removed)
        \\  Note    Nothing is downloaded.
        \\✓ qwen3.8-27b  19.55 GB
        \\✗ laya  the Hub did not answer
        \\→ next: nuclis config init
        \\6 ok, 0 warnings, 1 failed
        \\
    , buf.written());
}
