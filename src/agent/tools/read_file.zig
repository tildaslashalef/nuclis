//! `read_file` — a bounded, line-addressed region of a UTF-8 text file, or
//! its outline (markdown headings, code definitions) with line numbers.
//!
//! Host constants, never model-supplied: at most `max_lines` lines and
//! `max_bytes` bytes per call. Exceeding either is a result with `truncated`
//! set (and the text stops at the bound), not an abort. A partial read ends
//! with a bracketed note the model reads (the range, the total, where to
//! continue); a whole-file read carries none, since every token is prefill.
//! A file that is not valid UTF-8 is a typed error result.
const std = @import("std");
const root = @import("root.zig");

pub const tool: root.Tool = .{
    .name = "read_file",
    .description = "Read a UTF-8 text file in the workspace a page at a time: `offset` is a 1-based line, `count` the number of lines (default 120, at most 2000); a partial page ends with its range and where to continue. `outline: true` instead lists a markdown file's headings or a source file's definitions with their line numbers.",
    .parameters = "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"offset\":{\"type\":\"integer\"},\"count\":{\"type\":\"integer\"},\"outline\":{\"type\":\"boolean\"}},\"required\":[\"path\"]}",
    .label = "Reading",
    .display = "Read",
    .subject = "path",
    .run = run,
};

const max_lines: usize = 2000;
/// A page the model asks past, rather than a whole file it did not need.
const default_lines: usize = 120;
const max_bytes: usize = 1024 * 1024;
/// Outline entries per call, and the bytes kept of each entry's line.
const max_outline: usize = 400;
const max_outline_line: usize = 160;

const Args = struct {
    path: []const u8,
    offset: ?usize = null,
    count: ?usize = null,
    outline: bool = false,
};

fn run(workspace: root.Workspace, alloc: std.mem.Allocator, arguments: []const u8) std.mem.Allocator.Error!root.Result {
    const parsed = std.json.parseFromSlice(Args, alloc, arguments, .{ .ignore_unknown_fields = true }) catch {
        return root.fail(alloc, "read_file: arguments must be a JSON object with a \"path\" string", .{});
    };
    defer parsed.deinit();
    const path = parsed.value.path;
    const offset = parsed.value.offset orelse 1;
    const count = @min(parsed.value.count orelse default_lines, max_lines);
    if (offset == 0) return root.fail(alloc, "read_file: offset is 1-based", .{});
    const syntax: ?Syntax = if (parsed.value.outline) syntaxOf(path) orelse
        return root.fail(alloc, "read_file: no outline for {s} (markdown and Python, Zig, JS/TS, Go, Rust, C-family, shell sources have one); use grep to find its definitions", .{path}) else null;

    const abs = workspace.resolve(alloc, path) catch |err| switch (err) {
        error.OutsideWorkspace => return root.fail(alloc, "read_file: {s} is outside the workspace", .{path}),
        else => return root.fail(alloc, "read_file: {s}: {s}", .{ path, @errorName(err) }),
    };
    defer alloc.free(abs);

    // The first `max_bytes` of a larger file are served, never a refusal:
    // the size comes from `stat`, so the read itself stays within the bound.
    const file = workspace.dir.openFile(workspace.io, abs, .{}) catch |err| {
        return root.fail(alloc, "read_file: {s}: {s}", .{ path, @errorName(err) });
    };
    defer file.close(workspace.io);
    const stat = file.stat(workspace.io) catch |err| {
        return root.fail(alloc, "read_file: {s}: {s}", .{ path, @errorName(err) });
    };
    if (stat.kind == .directory) return root.fail(alloc, "read_file: {s} is a directory", .{path});
    const size: usize = @intCast(stat.size);
    const byte_truncated = size > max_bytes;
    const bytes = try alloc.alloc(u8, @min(size, max_bytes));
    defer alloc.free(bytes);
    var buffer: [16 * 1024]u8 = undefined;
    var reader = file.reader(workspace.io, &buffer);
    const read_len = reader.interface.readSliceShort(bytes) catch |err| {
        return root.fail(alloc, "read_file: {s}: {s}", .{ path, @errorName(err) });
    };
    var content = bytes[0..read_len];
    // A cut inside a scalar is the bound's doing, not the file's.
    if (byte_truncated) content = content[0 .. std.mem.lastIndexOfScalar(u8, content, '\n') orelse content.len];
    if (!std.unicode.utf8ValidateSlice(content)) {
        return root.fail(alloc, "read_file: {s} is not valid UTF-8 text", .{path});
    }
    const digest = std.hash.Wyhash.hash(0, content);

    // A trailing newline ends the last line; it is not an empty line after it.
    const body = if (content.len > 0 and content[content.len - 1] == '\n') content[0 .. content.len - 1] else content;
    const total = if (content.len == 0) 0 else std.mem.count(u8, body, "\n") + 1;
    if (syntax) |kind| return outline(alloc, body, total, kind, if (byte_truncated) size else null);

    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    var line: usize = 1;
    var taken: usize = 0;
    var line_truncated = false;
    var it = std.mem.splitScalar(u8, body, '\n');
    while (it.next()) |text| {
        if (content.len == 0) break;
        if (line < offset) {
            line += 1;
            continue;
        }
        if (taken == count) {
            line_truncated = true;
            break;
        }
        if (taken != 0) try out.append(alloc, '\n');
        try out.appendSlice(alloc, text);
        taken += 1;
        line += 1;
    }
    const file_size: ?usize = if (byte_truncated) size else null;
    // What the model reads after the lines: nothing for a whole file.
    if (taken == 0 or offset > 1 or line_truncated or file_size != null) {
        if (taken != 0) try out.append(alloc, '\n');
        try out.append(alloc, '[');
        try describe(alloc, &out, offset, taken, total, line_truncated, file_size);
        try out.append(alloc, ']');
    }
    var summary: std.ArrayList(u8) = .empty;
    errdefer summary.deinit(alloc);
    try describe(alloc, &summary, offset, taken, total, line_truncated, file_size);
    const owned_summary = try summary.toOwnedSlice(alloc);
    errdefer alloc.free(owned_summary);
    return .{
        .text = try out.toOwnedSlice(alloc),
        .truncated = byte_truncated or line_truncated,
        .summary = owned_summary,
        .lines = .{ .first = offset, .count = taken, .total = total, .digest = digest },
    };
}

/// The range shown, the file's length, and, when the read stopped early,
/// the offset that continues it: the transcript's detail row, and the note
/// the model reads after a partial page.
fn describe(alloc: std.mem.Allocator, out: *std.ArrayList(u8), offset: usize, taken: usize, total: usize, line_truncated: bool, file_size: ?usize) std.mem.Allocator.Error!void {
    if (taken == 0) {
        try out.print(alloc, "no lines at offset {d}; the file has {d}", .{ offset, total });
    } else {
        try out.print(alloc, "lines {d} to {d} of {d}", .{ offset, offset + taken - 1, total });
    }
    if (line_truncated) try out.print(alloc, " · truncated, continue with offset={d}", .{offset + taken});
    if (file_size) |size| try out.print(alloc, " · first {d} of {d} bytes only; use bash (head, wc, grep) for the rest", .{ max_bytes, size });
}

/// What an outline lists: markdown headings, or one family's definitions.
const Syntax = enum { markdown, python, zig, script, go, rust, c, shell };

fn syntaxOf(path: []const u8) ?Syntax {
    const table = [_]struct { ext: []const u8, syntax: Syntax }{
        .{ .ext = ".md", .syntax = .markdown },  .{ .ext = ".markdown", .syntax = .markdown },
        .{ .ext = ".mdx", .syntax = .markdown }, .{ .ext = ".py", .syntax = .python },
        .{ .ext = ".zig", .syntax = .zig },      .{ .ext = ".js", .syntax = .script },
        .{ .ext = ".mjs", .syntax = .script },   .{ .ext = ".cjs", .syntax = .script },
        .{ .ext = ".jsx", .syntax = .script },   .{ .ext = ".ts", .syntax = .script },
        .{ .ext = ".tsx", .syntax = .script },   .{ .ext = ".mts", .syntax = .script },
        .{ .ext = ".go", .syntax = .go },        .{ .ext = ".rs", .syntax = .rust },
        .{ .ext = ".c", .syntax = .c },          .{ .ext = ".h", .syntax = .c },
        .{ .ext = ".cc", .syntax = .c },         .{ .ext = ".cpp", .syntax = .c },
        .{ .ext = ".hpp", .syntax = .c },        .{ .ext = ".m", .syntax = .c },
        .{ .ext = ".mm", .syntax = .c },         .{ .ext = ".metal", .syntax = .c },
        .{ .ext = ".sh", .syntax = .shell },     .{ .ext = ".bash", .syntax = .shell },
    };
    const ext = std.fs.path.extension(path);
    for (table) |entry| if (std.ascii.eqlIgnoreCase(ext, entry.ext)) return entry.syntax;
    return null;
}

/// The outline as `line: text`, each entry's own indentation kept so nesting
/// shows, followed by a note of what it covers. Not a line-addressed read.
fn outline(alloc: std.mem.Allocator, body: []const u8, total: usize, syntax: Syntax, file_size: ?usize) std.mem.Allocator.Error!root.Result {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    var entries: usize = 0;
    var shown: usize = 0;
    var fence: ?u8 = null;
    var number: usize = 0;
    var it = std.mem.splitScalar(u8, body, '\n');
    while (it.next()) |raw| {
        number += 1;
        if (body.len == 0) break;
        const text = std.mem.trimEnd(u8, raw, " \t\r");
        if (syntax == .markdown) {
            // A heading inside a fenced block is code, not structure.
            const lead = std.mem.trimStart(u8, text, " ");
            if (std.mem.startsWith(u8, lead, "```") or std.mem.startsWith(u8, lead, "~~~")) {
                if (fence == null) fence = lead[0] else if (fence.? == lead[0]) fence = null;
                continue;
            }
            if (fence != null) continue;
        }
        if (!isEntry(syntax, text)) continue;
        entries += 1;
        if (shown == max_outline) continue;
        shown += 1;
        var kept = text[0..@min(text.len, max_outline_line)];
        while (!std.unicode.utf8ValidateSlice(kept)) kept = kept[0 .. kept.len - 1];
        try out.print(alloc, "{d}: {s}\n", .{ number, kept });
    }
    const noun = if (syntax == .markdown) "heading" else "definition";
    var note: std.ArrayList(u8) = .empty;
    defer note.deinit(alloc);
    try note.print(alloc, "outline: {d} {s}{s} in {d} lines", .{ entries, noun, if (entries == 1) "" else "s", total });
    if (shown < entries) try note.print(alloc, " · first {d} shown", .{shown});
    if (file_size) |size| try note.print(alloc, " · first {d} of {d} bytes only", .{ max_bytes, size });
    try out.print(alloc, "[{s}; read a part with offset and count]", .{note.items});
    const summary = try alloc.dupe(u8, note.items);
    errdefer alloc.free(summary);
    return .{ .text = try out.toOwnedSlice(alloc), .truncated = shown < entries or file_size != null, .summary = summary };
}

/// Whether one line opens a heading or a definition. Deliberately lexical:
/// a line-start keyword is right for the conventional layout of each family
/// and costs no parser.
fn isEntry(syntax: Syntax, line: []const u8) bool {
    const lead = std.mem.trimStart(u8, line, " \t");
    if (lead.len == 0) return false;
    return switch (syntax) {
        .markdown => line.len - lead.len <= 3 and lead[0] == '#' and blk: {
            const hashes = std.mem.indexOfNone(u8, lead, "#") orelse lead.len;
            break :blk hashes <= 6 and (hashes == lead.len or lead[hashes] == ' ');
        },
        .python => startsWithAny(lead, &.{ "def ", "async def ", "class " }),
        .zig => startsWithAny(lead, &.{ "fn ", "pub fn ", "export fn ", "inline fn ", "pub inline fn ", "test " }) or
            (startsWithAny(lead, &.{ "const ", "pub const " }) and zigContainer(lead)),
        .script => startsWithAny(lead, &.{ "function ", "async function ", "export function ", "export async function ", "export default ", "class ", "export class ", "interface ", "export interface ", "type ", "export type " }),
        .go => startsWithAny(lead, &.{ "func ", "type " }),
        .rust => startsWithAny(lead, &.{ "fn ", "pub fn ", "pub(crate) fn ", "async fn ", "pub async fn ", "struct ", "pub struct ", "enum ", "pub enum ", "trait ", "pub trait ", "impl ", "impl<", "mod ", "pub mod " }),
        // A definition head starts at column 0: a type, or a signature that
        // is not a declaration (no `;`) and not a preprocessor line.
        .c => line.len == lead.len and (startsWithAny(lead, &.{ "struct ", "typedef ", "enum ", "class ", "namespace ", "@interface", "@implementation", "kernel " }) or
            (std.mem.indexOfScalar(u8, lead, '(') != null and lead[lead.len - 1] != ';' and std.mem.indexOfScalar(u8, "#/*}{ \t", lead[0]) == null)),
        .shell => startsWithAny(lead, &.{"function "}) or (line.len == lead.len and std.mem.indexOf(u8, lead, "()") != null and lead[lead.len - 1] == '{'),
    };
}

fn startsWithAny(text: []const u8, prefixes: []const []const u8) bool {
    for (prefixes) |prefix| if (std.mem.startsWith(u8, text, prefix)) return true;
    return false;
}

/// `const Name = struct {` and its kin: a declaration that opens a type.
fn zigContainer(lead: []const u8) bool {
    const eq = std.mem.indexOf(u8, lead, " = ") orelse return false;
    const value = lead[eq + 3 ..];
    return startsWithAny(value, &.{ "struct", "enum", "union", "opaque", "extern struct", "packed struct", "extern union", "packed union" });
}

// ----- tests -----

const testing = std.testing;

fn testWorkspace() struct { tmp: testing.TmpDir, ws: root.Workspace, root_path: [:0]u8 } {
    var tmp = testing.tmpDir(.{});
    const root_path = tmp.dir.realPathFileAlloc(testing.io, ".", testing.allocator) catch @panic("root");
    return .{ .tmp = tmp, .ws = .{ .io = testing.io, .dir = tmp.dir, .root = root_path }, .root_path = root_path };
}

test "read_file returns the addressed lines and marks truncation" {
    var w = testWorkspace();
    defer w.tmp.cleanup();
    defer testing.allocator.free(w.root_path);
    try w.tmp.dir.writeFile(testing.io, .{ .sub_path = "a.txt", .data = "one\ntwo\nthree\nfour" });
    try w.tmp.dir.writeFile(testing.io, .{ .sub_path = "b.txt", .data = "one\ntwo\n" });
    try w.tmp.dir.writeFile(testing.io, .{ .sub_path = "empty.txt", .data = "" });

    var all = try run(w.ws, testing.allocator, "{\"path\":\"a.txt\"}");
    defer all.deinit(testing.allocator);
    try testing.expectEqualStrings("one\ntwo\nthree\nfour", all.text);
    try testing.expect(!all.truncated);
    try testing.expect(!all.is_error);
    try testing.expectEqualStrings("lines 1 to 4 of 4", all.summary.?);

    var middle = try run(w.ws, testing.allocator, "{\"path\":\"a.txt\",\"offset\":2,\"count\":2}");
    defer middle.deinit(testing.allocator);
    // A partial page tells the model where it stands; the transcript gets
    // the same words without the brackets.
    try testing.expectEqualStrings("two\nthree\n[lines 2 to 3 of 4 · truncated, continue with offset=4]", middle.text);
    try testing.expect(middle.truncated); // "four" was not shown
    try testing.expectEqualStrings("lines 2 to 3 of 4 · truncated, continue with offset=4", middle.summary.?);

    var past = try run(w.ws, testing.allocator, "{\"path\":\"a.txt\",\"offset\":9}");
    defer past.deinit(testing.allocator);
    try testing.expectEqualStrings("[no lines at offset 9; the file has 4]", past.text);
    try testing.expectEqualStrings("no lines at offset 9; the file has 4", past.summary.?);

    // A trailing newline is the end of the last line, not a line of its own:
    // an exact read of the whole file is complete, not truncated.
    var exact = try run(w.ws, testing.allocator, "{\"path\":\"b.txt\",\"count\":2}");
    defer exact.deinit(testing.allocator);
    try testing.expectEqualStrings("one\ntwo", exact.text);
    try testing.expect(!exact.truncated);
    try testing.expectEqualStrings("lines 1 to 2 of 2", exact.summary.?);

    var empty = try run(w.ws, testing.allocator, "{\"path\":\"empty.txt\"}");
    defer empty.deinit(testing.allocator);
    try testing.expectEqualStrings("[no lines at offset 1; the file has 0]", empty.text);
    try testing.expectEqualStrings("no lines at offset 1; the file has 0", empty.summary.?);

    try testing.expectError(error.OutsideWorkspace, @as(root.Workspace, w.ws).resolve(testing.allocator, "/"));
}

test "a file over the byte bound serves its first MiB with the size in the summary" {
    const alloc = testing.allocator;
    var f = testWorkspace();
    defer alloc.free(f.root_path);
    defer f.tmp.cleanup();
    // One byte over the bound: 32-byte lines, the last one cut by the bound.
    const line = "0123456789abcdef0123456789abcde\n";
    var data: std.ArrayList(u8) = .empty;
    defer data.deinit(alloc);
    while (data.items.len < max_bytes + 1) try data.appendSlice(alloc, line);
    try f.tmp.dir.writeFile(testing.io, .{ .sub_path = "big.txt", .data = data.items });

    var result = try tool.run(f.ws, alloc, "{\"path\":\"big.txt\",\"count\":2}");
    defer result.deinit(alloc);
    try testing.expect(!result.is_error);
    try testing.expect(result.truncated);
    try testing.expect(std.mem.startsWith(u8, result.text, line[0 .. line.len - 1]));
    const expected = try alloc.print("first {d} of {d} bytes only", .{ max_bytes, data.items.len });
    defer alloc.free(expected);
    try testing.expect(std.mem.indexOf(u8, result.summary.?, expected) != null);
    try testing.expect(std.mem.indexOf(u8, result.summary.?, "use bash") != null);
}

test "read_file reports bad arguments, missing files, and non-UTF-8 as results" {
    var w = testWorkspace();
    defer w.tmp.cleanup();
    defer testing.allocator.free(w.root_path);
    try w.tmp.dir.writeFile(testing.io, .{ .sub_path = "bin", .data = "\xff\xfe" });

    var bad = try run(w.ws, testing.allocator, "not json");
    defer bad.deinit(testing.allocator);
    try testing.expect(bad.is_error);
    try testing.expect(bad.summary == null);

    var missing = try run(w.ws, testing.allocator, "{\"path\":\"nope.txt\"}");
    defer missing.deinit(testing.allocator);
    try testing.expect(missing.is_error);

    var binary = try run(w.ws, testing.allocator, "{\"path\":\"bin\"}");
    defer binary.deinit(testing.allocator);
    try testing.expect(binary.is_error);
    try testing.expect(std.mem.indexOf(u8, binary.text, "UTF-8") != null);

    var zero = try run(w.ws, testing.allocator, "{\"path\":\"bin\",\"offset\":0}");
    defer zero.deinit(testing.allocator);
    try testing.expect(zero.is_error);
}

test "the default page is 120 lines and its note gives the total and the continuation" {
    const alloc = testing.allocator;
    var w = testWorkspace();
    defer w.tmp.cleanup();
    defer alloc.free(w.root_path);
    var data: std.Io.Writer.Allocating = .init(alloc);
    defer data.deinit();
    for (1..432) |i| try data.writer.print("line {d}\n", .{i});
    try w.tmp.dir.writeFile(testing.io, .{ .sub_path = "long.md", .data = data.written() });

    var page = try run(w.ws, alloc, "{\"path\":\"long.md\"}");
    defer page.deinit(alloc);
    try testing.expect(std.mem.endsWith(u8, page.text, "line 120\n[lines 1 to 120 of 431 · truncated, continue with offset=121]"));
    try testing.expectEqual(@as(usize, 120), page.lines.?.count);

    // The last page reaches the end: its note says where it is, nothing more.
    var last = try run(w.ws, alloc, "{\"path\":\"long.md\",\"offset\":400}");
    defer last.deinit(alloc);
    try testing.expect(std.mem.endsWith(u8, last.text, "line 431\n[lines 400 to 431 of 431]"));
}

test "outline lists markdown headings outside fences and code definitions, with line numbers" {
    const alloc = testing.allocator;
    var w = testWorkspace();
    defer w.tmp.cleanup();
    defer alloc.free(w.root_path);
    try w.tmp.dir.writeFile(testing.io, .{ .sub_path = "doc.md", .data = "# Title\n\ntext\n\n```sh\n# not a heading\n```\n## Part\n#hashtag\n### Deep\n" });
    try w.tmp.dir.writeFile(testing.io, .{ .sub_path = "m.py", .data = "import os\n\nclass Shape:\n    def area(self):\n        return 0\n\nasync def fetch():\n    pass\n" });
    try w.tmp.dir.writeFile(testing.io, .{ .sub_path = "m.zig", .data = "const std = @import(\"std\");\npub const Pair = struct {\n    pub fn sum(self: Pair) u32 {\n        return 0;\n    }\n};\nfn helper() void {}\ntest \"pair\" {}\n" });
    try w.tmp.dir.writeFile(testing.io, .{ .sub_path = "notes.txt", .data = "plain\n" });

    var md = try run(w.ws, alloc, "{\"path\":\"doc.md\",\"outline\":true}");
    defer md.deinit(alloc);
    try testing.expectEqualStrings("1: # Title\n8: ## Part\n10: ### Deep\n[outline: 3 headings in 10 lines; read a part with offset and count]", md.text);
    try testing.expect(md.lines == null);

    var py = try run(w.ws, alloc, "{\"path\":\"m.py\",\"outline\":true}");
    defer py.deinit(alloc);
    try testing.expectEqualStrings("3: class Shape:\n4:     def area(self):\n7: async def fetch():\n[outline: 3 definitions in 8 lines; read a part with offset and count]", py.text);

    var zig = try run(w.ws, alloc, "{\"path\":\"m.zig\",\"outline\":true}");
    defer zig.deinit(alloc);
    try testing.expectEqualStrings("2: pub const Pair = struct {\n3:     pub fn sum(self: Pair) u32 {\n7: fn helper() void {}\n8: test \"pair\" {}\n[outline: 4 definitions in 8 lines; read a part with offset and count]", zig.text);

    var none = try run(w.ws, alloc, "{\"path\":\"notes.txt\",\"outline\":true}");
    defer none.deinit(alloc);
    try testing.expect(none.is_error);
    try testing.expect(std.mem.indexOf(u8, none.text, "use grep") != null);
}
