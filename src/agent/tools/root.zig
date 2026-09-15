//! The agent's tool registry. A tool validates its own arguments and returns a
//! typed result; it never panics on bad input. Expected failures (a missing
//! file, a bad argument, a workspace escape) are *results* with `is_error`,
//! because the model is meant to read them; only an allocation failure is a
//! Zig error. Every limit is a host constant, never model-supplied.
//!
//! The workspace is the process's working directory, canonicalized once at
//! startup. `Workspace.resolve` canonicalizes each requested path (following
//! symlinks) and refuses anything that does not stay under the root, so
//! `..`, an absolute path, or a symlink cannot escape it.
const std = @import("std");
const tui = @import("../../tui/root.zig");
const read_file = @import("read_file.zig");
const glob = @import("glob.zig");
const grep = @import("grep.zig");
const bash = @import("bash.zig");
const write_file = @import("write_file.zig");
const edit_file = @import("edit_file.zig");

pub const Allocator = std.mem.Allocator;

/// A cooperative beat during a long-running tool: the agent's chance to read
/// the keyboard and repaint while a blocking call holds the loop. Only tools
/// that already poll invoke it (`bash`, once per iteration). `context` is the
/// agent's driver; print mode installs none, and the fast read tools pay
/// nothing. It is `void` on purpose: a tick that cannot read the terminal is
/// not a reason for the tool to fail.
pub const Tick = struct {
    context: *anyopaque,
    call: *const fn (context: *anyopaque) void,
};

pub const Workspace = struct {
    io: std.Io,
    dir: std.Io.Dir,
    /// Canonical, absolute. No requested path may resolve outside it.
    root: []const u8,
    /// The parent environment, for tools that start a child. Null in tests and
    /// in a caller that has none; `bash` then runs with a fixed PATH.
    environ: ?*const std.process.Environ.Map = null,
    /// Invoked by a polling tool while it runs. Null in print mode and tests.
    tick: ?Tick = null,

    /// Canonicalizes `path` and rejects anything outside the workspace. The
    /// caller owns the returned sentinel-terminated absolute path.
    pub fn resolve(self: Workspace, alloc: Allocator, path: []const u8) ![:0]u8 {
        const abs = try self.dir.realPathFileAlloc(self.io, path, alloc);
        errdefer alloc.free(abs);
        if (!within(self.root, abs)) return error.OutsideWorkspace;
        return abs;
    }

    /// Resolves a path for a mutation, where the target may not exist yet.
    /// Unlike `resolve`, a missing target is allowed: the parent directory
    /// must canonicalize inside the workspace and the final component must be
    /// a plain name. An existing path that canonicalizes outside — including
    /// through a symlink — is refused, which is what stops a write from
    /// following a link out of the tree. The caller owns `abs`.
    pub fn resolveTarget(self: Workspace, alloc: Allocator, path: []const u8) !ResolvedTarget {
        if (self.dir.realPathFileAlloc(self.io, path, alloc)) |abs| {
            if (!within(self.root, abs)) {
                alloc.free(abs);
                return error.OutsideWorkspace;
            }
            return .{ .abs = abs, .exists = true };
        } else |err| switch (err) {
            error.FileNotFound => {},
            else => return err,
        }
        const parent = std.fs.path.dirname(path) orelse ".";
        const name = std.fs.path.basename(path);
        if (name.len == 0 or std.mem.eql(u8, name, ".") or std.mem.eql(u8, name, "..")) return error.InvalidPath;
        const parent_abs = try self.dir.realPathFileAlloc(self.io, if (parent.len == 0) "." else parent, alloc);
        defer alloc.free(parent_abs);
        if (!within(self.root, parent_abs)) return error.OutsideWorkspace;
        return .{ .abs = try std.fs.path.joinZ(alloc, &.{ parent_abs, name }), .exists = false };
    }

    /// Replaces `target` with `content` atomically: a temp file beside it is
    /// written and renamed over it, so a crash leaves either the old file or
    /// the new one, never a half-written one. On any failure the temp file is
    /// removed and the target is untouched. `target` is a canonical absolute
    /// path from `resolveTarget`.
    pub fn replaceFile(self: Workspace, alloc: Allocator, target: []const u8, content: []const u8) !void {
        var random: [8]u8 = undefined;
        self.io.randomSecure(&random) catch self.io.random(&random);
        const temp = try std.fmt.allocPrintSentinel(alloc, "{s}.nuclis-{x}.tmp", .{ target, &random }, 0);
        defer alloc.free(temp);
        {
            const file = try std.Io.Dir.createFileAbsolute(self.io, temp, .{});
            errdefer std.Io.Dir.deleteFileAbsolute(self.io, temp) catch {};
            defer file.close(self.io);
            try file.writeStreamingAll(self.io, content);
        }
        std.Io.Dir.renameAbsolute(temp, target, self.io) catch |err| {
            std.Io.Dir.deleteFileAbsolute(self.io, temp) catch {};
            return err;
        };
    }
};

pub const ResolvedTarget = struct { abs: [:0]u8, exists: bool };

/// Whether `abs` is `root` itself or a path beneath it. `abs` and `root` are
/// both canonical, so a prefix comparison plus a separator is exact.
fn within(root: []const u8, abs: []const u8) bool {
    if (!std.mem.startsWith(u8, abs, root)) return false;
    return abs.len == root.len or abs[root.len] == std.fs.path.sep;
}

/// A mutation's before and after, for the diff event. The path is owned, and
/// the diff carries both the structured rows the transcript renders and the
/// unified text the model and session record.
pub const Change = struct {
    path: []u8,
    diff: tui.diff.Diff,

    pub fn deinit(self: *Change, alloc: Allocator) void {
        alloc.free(self.path);
        self.diff.deinit(alloc);
    }
};

/// The detail row of a mutation: added and removed line counts from the
/// structured diff (`+12 −3`).
pub fn changeSummary(alloc: Allocator, rows: []const tui.diff.Row) ![]u8 {
    var added: usize = 0;
    var removed: usize = 0;
    for (rows) |row| switch (row.kind) {
        .add => added += 1,
        .remove => removed += 1,
        .context => {},
    };
    return std.fmt.allocPrint(alloc, "+{d} −{d}", .{ added, removed });
}

pub const Result = struct {
    /// What the model reads. Owned by the caller.
    text: []u8,
    truncated: bool = false,
    is_error: bool = false,
    /// One sentence for the transcript's detail row (`lines 1 to 40 of 96`),
    /// owned; null when there is nothing to say (a clean `bash` run) and on
    /// every `fail`, whose `text` is the message.
    summary: ?[]u8 = null,
    /// A line-addressed read: where its text starts in the file and how long
    /// the file is, so a later cut can restate the range it actually kept.
    lines: ?LineRange = null,
    /// Set by a mutation tool; null for reads.
    change: ?Change = null,

    pub fn deinit(self: *Result, alloc: Allocator) void {
        alloc.free(self.text);
        if (self.summary) |summary| alloc.free(summary);
        if (self.change) |*change| change.deinit(alloc);
        self.* = undefined;
    }
};

pub const LineRange = struct { first: usize, total: usize };

pub const Tool = struct {
    name: []const u8,
    description: []const u8,
    /// A JSON object: the parameter schema the model is shown.
    parameters: []const u8,
    /// The gerund the call row starts with (`Reading`, `Running`, …).
    /// Presentation only; the model still sees `name`.
    label: []const u8,
    /// The primary parameter: the first property of `parameters`, shown
    /// after the label, or on the detail row when `detail_prefix` is set.
    subject: []const u8,
    /// When set, the subject moves to its own detail row behind this prefix
    /// (`bash`: `$ `), so a command never crowds the call row.
    detail_prefix: ?[]const u8 = null,
    run: *const fn (workspace: Workspace, alloc: Allocator, arguments: []const u8) Allocator.Error!Result,
};

pub const all = [_]Tool{ read_file.tool, glob.tool, grep.tool, write_file.tool, edit_file.tool, bash.tool };

pub fn find(name: []const u8) ?Tool {
    for (all) |tool| if (std.mem.eql(u8, tool.name, name)) return tool;
    return null;
}

/// An expected failure, as a result the model reads.
pub fn fail(alloc: Allocator, comptime format: []const u8, args: anytype) Allocator.Error!Result {
    return .{ .text = try std.fmt.allocPrint(alloc, format, args), .is_error = true };
}

/// A tool call as the transcript shows it: the call row (`Reading TODO.md`)
/// and, for a tool with a `detail_prefix`, the detail row (`$ make test`).
/// Both owned.
pub const Described = struct {
    summary: []u8,
    detail: ?[]u8 = null,

    pub fn deinit(self: *Described, alloc: Allocator) void {
        alloc.free(self.summary);
        if (self.detail) |detail| alloc.free(detail);
        self.* = undefined;
    }
};

/// Humanizes one call for the transcript: the label and the subject, the
/// other arguments left to the result's detail row. The model-facing name
/// and raw JSON are untouched. Anything unparseable falls back to
/// `label <raw arguments>`.
pub fn describe(alloc: Allocator, name: []const u8, arguments: []const u8) !Described {
    const tool = find(name) orelse return .{ .summary = try std.fmt.allocPrint(alloc, "{s} {s}", .{ name, arguments }) };
    const parsed = std.json.parseFromSlice(std.json.Value, alloc, arguments, .{}) catch
        return .{ .summary = try std.fmt.allocPrint(alloc, "{s} {s}", .{ tool.label, arguments }) };
    defer parsed.deinit();
    if (parsed.value != .object) return .{ .summary = try std.fmt.allocPrint(alloc, "{s} {s}", .{ tool.label, arguments }) };
    const subject = parsed.value.object.get(tool.subject);

    var out: std.Io.Writer.Allocating = .init(alloc);
    errdefer out.deinit();
    try out.writer.writeAll(tool.label);
    if (tool.detail_prefix) |prefix| {
        const summary = try out.toOwnedSlice();
        errdefer alloc.free(summary);
        const value = subject orelse return .{ .summary = summary };
        var detail: std.Io.Writer.Allocating = .init(alloc);
        errdefer detail.deinit();
        try detail.writer.writeAll(prefix);
        try writeDisplayValue(&detail.writer, value);
        return .{ .summary = summary, .detail = try detail.toOwnedSlice() };
    }
    if (subject) |value| {
        try out.writer.writeByte(' ');
        try writeDisplayValue(&out.writer, value);
    }
    return .{ .summary = try out.toOwnedSlice() };
}

/// A parameter value on the one-line display: strings are literal, everything
/// else is JSON. Newlines and tabs become spaces so a multi-line command
/// cannot break the call line.
fn writeDisplayValue(w: *std.Io.Writer, value: std.json.Value) !void {
    switch (value) {
        .string => |text| for (text) |byte| {
            if (byte == '\n' or byte == '\r' or byte == '\t') try w.writeByte(' ') else try w.writeByte(byte);
        },
        .integer => |n| try w.print("{d}", .{n}),
        .float => |x| try w.print("{d}", .{x}),
        .bool => |b| try w.writeAll(if (b) "true" else "false"),
        .null => try w.writeAll("null"),
        else => {
            var json: std.json.Stringify = .{ .writer = w };
            try json.write(value);
        },
    }
}

test "the registry finds tools by name" {
    try std.testing.expect(find("read_file") != null);
    try std.testing.expect(find("glob") != null);
    try std.testing.expect(find("grep") != null);
    try std.testing.expect(find("write_file") != null);
    try std.testing.expect(find("edit_file") != null);
    try std.testing.expect(find("bash") != null);
    try std.testing.expect(find("nonexistent") == null);
}

test "every tool's subject is the first property of its schema" {
    const alloc = std.testing.allocator;
    for (all) |tool| {
        const parsed = try std.json.parseFromSlice(std.json.Value, alloc, tool.parameters, .{});
        defer parsed.deinit();
        const properties = parsed.value.object.get("properties").?.object;
        var it = properties.iterator();
        const first = it.next() orelse return error.TestUnexpectedResult;
        try std.testing.expectEqualStrings(first.key_ptr.*, tool.subject);
    }
}

test "describe humanizes a call: label and subject, a command on its own row" {
    const alloc = std.testing.allocator;
    var read = try describe(alloc, "read_file", "{\"path\":\"TODO.md\",\"count\":24,\"offset\":126}");
    defer read.deinit(alloc);
    try std.testing.expectEqualStrings("Reading TODO.md", read.summary);
    try std.testing.expect(read.detail == null);

    var bash_line = try describe(alloc, "bash", "{\"command\":\"ls -la\"}");
    defer bash_line.deinit(alloc);
    try std.testing.expectEqualStrings("Running command", bash_line.summary);
    try std.testing.expectEqualStrings("$ ls -la", bash_line.detail.?);

    // A multi-line command stays on one display row.
    var multiline = try describe(alloc, "bash", "{\"command\":\"echo a\\necho b\"}");
    defer multiline.deinit(alloc);
    try std.testing.expectEqualStrings("$ echo a echo b", multiline.detail.?);

    var bad = try describe(alloc, "read_file", "not json");
    defer bad.deinit(alloc);
    try std.testing.expectEqualStrings("Reading not json", bad.summary);

    var unknown = try describe(alloc, "nope", "{\"x\":1}");
    defer unknown.deinit(alloc);
    try std.testing.expectEqualStrings("nope {\"x\":1}", unknown.summary);
}

test "resolve accepts inside the workspace and refuses escapes" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const root = try tmp.dir.realPathFileAlloc(std.testing.io, ".", std.testing.allocator);
    defer std.testing.allocator.free(root);
    const workspace: Workspace = .{ .io = std.testing.io, .dir = tmp.dir, .root = root };
    try tmp.dir.writeFile(std.testing.io, .{ .sub_path = "inside.txt", .data = "hi" });

    const inside = try workspace.resolve(std.testing.allocator, "inside.txt");
    defer std.testing.allocator.free(inside);
    try std.testing.expect(std.mem.endsWith(u8, inside, "inside.txt"));

    try std.testing.expectError(error.OutsideWorkspace, workspace.resolve(std.testing.allocator, "/"));
}

test "resolve refuses dot-dot, absolute paths, and a symlink out of the tree" {
    const alloc = std.testing.allocator;
    var inside = std.testing.tmpDir(.{});
    defer inside.cleanup();
    var outside = std.testing.tmpDir(.{});
    defer outside.cleanup();
    const root = try inside.dir.realPathFileAlloc(std.testing.io, ".", alloc);
    defer alloc.free(root);
    const outside_path = try outside.dir.realPathFileAlloc(std.testing.io, ".", alloc);
    defer alloc.free(outside_path);
    try outside.dir.writeFile(std.testing.io, .{ .sub_path = "secret.txt", .data = "no" });
    try inside.dir.writeFile(std.testing.io, .{ .sub_path = "ok.txt", .data = "yes" });
    // A link inside the workspace that canonicalizes to a directory outside it.
    try inside.dir.symLink(std.testing.io, outside_path, "link", .{});

    const workspace: Workspace = .{ .io = std.testing.io, .dir = inside.dir, .root = root };
    try std.testing.expectError(error.OutsideWorkspace, workspace.resolve(alloc, ".."));
    try std.testing.expectError(error.OutsideWorkspace, workspace.resolve(alloc, "/etc/hosts"));
    try std.testing.expectError(error.OutsideWorkspace, workspace.resolve(alloc, "link/secret.txt"));

    const ok = try workspace.resolve(alloc, "ok.txt");
    defer alloc.free(ok);
    try std.testing.expect(std.mem.endsWith(u8, ok, "ok.txt"));
}
