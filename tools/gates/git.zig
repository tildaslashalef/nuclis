//! The questions the selector asks git: what differs from a revision, and
//! what a file held there. Paths are repository-relative; `cwd` null is the
//! process's directory.
const std = @import("std");
const Allocator = std.mem.Allocator;

/// `git show` of a large fixture stays in memory; this bounds it.
const max_output_bytes = 512 << 20;

pub const Error = Allocator.Error || std.process.RunError;

/// Standard output of a `git` command, or null when it exits non-zero.
fn output(arena: Allocator, io: std.Io, cwd: ?[]const u8, args: []const []const u8) Error!?[]const u8 {
    const argv = try std.mem.concat(arena, []const u8, &.{ &.{"git"}, args });
    const result = try std.process.run(arena, io, .{
        .argv = argv,
        .cwd = if (cwd) |path| .{ .path = path } else .inherit,
        .stdout_limit = .limited(max_output_bytes),
        .stderr_limit = .limited(1 << 20),
    });
    return switch (result.term) {
        .exited => |code| if (code == 0) result.stdout else null,
        else => null,
    };
}

fn lines(arena: Allocator, list: *std.ArrayList([]const u8), text: []const u8) Allocator.Error!void {
    var it = std.mem.tokenizeScalar(u8, text, 0);
    while (it.next()) |path| try list.append(arena, path);
}

fn sortUnique(items: *std.ArrayList([]const u8)) void {
    std.mem.sort([]const u8, items.items, {}, struct {
        fn lessThan(_: void, a: []const u8, b: []const u8) bool {
            return std.mem.lessThan(u8, a, b);
        }
    }.lessThan);
    var kept: usize = 0;
    for (items.items) |path| {
        if (kept > 0 and std.mem.eql(u8, items.items[kept - 1], path)) continue;
        items.items[kept] = path;
        kept += 1;
    }
    items.items.len = kept;
}

/// Files that differ from `rev` in the working tree, untracked ones included,
/// sorted. Null when git refuses `rev`.
pub fn changedFiles(arena: Allocator, io: std.Io, rev: []const u8) Error!?[]const []const u8 {
    const diff = try output(arena, io, null, &.{ "diff", "--name-only", "--no-renames", "-z", rev }) orelse return null;
    const untracked = try output(arena, io, null, &.{ "ls-files", "--others", "--exclude-standard", "-z" }) orelse return null;
    var out: std.ArrayList([]const u8) = .empty;
    try lines(arena, &out, diff);
    try lines(arena, &out, untracked);
    sortUnique(&out);
    return out.items;
}

/// Files that differ between two revisions, sorted. Null when git refuses one.
pub fn changedBetween(arena: Allocator, io: std.Io, from: []const u8, to: []const u8) Error!?[]const []const u8 {
    const diff = try output(arena, io, null, &.{ "diff", "--name-only", "--no-renames", "-z", from, to }) orelse return null;
    var out: std.ArrayList([]const u8) = .empty;
    try lines(arena, &out, diff);
    sortUnique(&out);
    return out.items;
}

/// `path` as it was at `rev`; null when it did not exist there.
pub fn show(arena: Allocator, io: std.Io, rev: []const u8, path: []const u8) Error!?[]const u8 {
    const object = try arena.print("{s}:{s}", .{ rev, path });
    return output(arena, io, null, &.{ "show", object });
}

/// The commit this branch left `origin/main` at, or null without that ref.
pub fn unitBase(arena: Allocator, io: std.Io, cwd: ?[]const u8) Error!?[]const u8 {
    const text = try output(arena, io, cwd, &.{ "merge-base", "origin/main", "HEAD" }) orelse return null;
    const trimmed = std.mem.trim(u8, text, " \r\n");
    return if (trimmed.len == 0) null else trimmed;
}

/// `git rev-parse --short HEAD`, or "unknown".
pub fn revision(arena: Allocator, io: std.Io) Error![]const u8 {
    const text = try output(arena, io, null, &.{ "rev-parse", "--short", "HEAD" }) orelse return "unknown";
    return std.mem.trim(u8, text, " \r\n");
}

const testing = std.testing;

fn runGit(arena: Allocator, cwd: []const u8, args: []const []const u8) ![]const u8 {
    const identity = [_][]const u8{ "-c", "user.name=t", "-c", "user.email=t@t" };
    const text = try output(arena, testing.io, cwd, try std.mem.concat(arena, []const u8, &.{ &identity, args }));
    return std.mem.trim(u8, text orelse return error.GitFailed, " \r\n");
}

test "the unit base is where the branch left origin/main" {
    var state: std.heap.ArenaAllocator = .init(testing.allocator);
    defer state.deinit();
    const a = state.allocator();
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    const cwd = try a.print(".zig-cache/tmp/{s}", .{tmp.sub_path});
    _ = try runGit(a, cwd, &.{ "init", "-q" });
    _ = try runGit(a, cwd, &.{ "commit", "-q", "--allow-empty", "-m", "base" });
    const base = try runGit(a, cwd, &.{ "rev-parse", "HEAD" });
    try testing.expect(try unitBase(a, testing.io, cwd) == null);
    _ = try runGit(a, cwd, &.{ "update-ref", "refs/remotes/origin/main", base });
    _ = try runGit(a, cwd, &.{ "commit", "-q", "--allow-empty", "-m", "on the branch" });
    try testing.expectEqualStrings(base, (try unitBase(a, testing.io, cwd)).?);
}
