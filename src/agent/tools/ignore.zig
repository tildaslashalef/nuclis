//! The `.gitignore` rules a tree walk honours: the file in each directory it
//! enters, from the workspace root down, last match winning. Native rather
//! than git's own: the walk needs no repository and no `git` binary, and one
//! rule set on every machine keeps the agent's results reproducible.
//!
//! The subset: blank lines and `#` comments, `!` negation, a trailing `/` for
//! directories only, a leading or inner `/` anchoring the pattern to the
//! file's directory (otherwise it matches a name at any depth), and the
//! `glob` tool's wildcards (`*`, `?`, `[...]`, `**`). Escapes other than a
//! leading `\` are not interpreted. `.git/info/exclude` and the global
//! excludes file are not read.
const std = @import("std");
const glob = @import("glob.zig");

const Allocator = std.mem.Allocator;

/// The bytes read of one `.gitignore`; a larger file's tail is ignored.
const max_file_bytes: usize = 64 * 1024;

const Pattern = struct {
    /// Borrowed from the owning `Set.text`.
    glob: []const u8,
    negate: bool,
    dir_only: bool,
    anchored: bool,
};

const Set = struct {
    /// The directory the file sits in, workspace-relative ("" for the root).
    base: []u8,
    text: []u8,
    patterns: []Pattern,
};

/// A stack of rule sets, one per directory on the walk's current path.
pub const Rules = struct {
    alloc: Allocator,
    sets: std.ArrayList(Set) = .empty,

    pub fn deinit(self: *Rules) void {
        while (self.sets.items.len > 0) self.pop();
        self.sets.deinit(self.alloc);
    }

    /// Loads `<dir_path>/.gitignore` (workspace-relative, "" for the root)
    /// if it exists. Returns whether a set was pushed, so the caller pops
    /// exactly what it pushed. An unreadable file is no rules, not an error.
    pub fn push(self: *Rules, io: std.Io, dir: std.Io.Dir, dir_path: []const u8) Allocator.Error!bool {
        const name = try std.fs.path.join(self.alloc, &.{ dir_path, ".gitignore" });
        defer self.alloc.free(name);
        const text = dir.readFileAlloc(io, name, self.alloc, .limited(max_file_bytes)) catch |err| switch (err) {
            error.OutOfMemory => return error.OutOfMemory,
            error.StreamTooLong => return false,
            else => return false,
        };
        errdefer self.alloc.free(text);
        const patterns = try parse(self.alloc, text);
        errdefer self.alloc.free(patterns);
        const base = try self.alloc.dupe(u8, dir_path);
        errdefer self.alloc.free(base);
        try self.sets.append(self.alloc, .{ .base = base, .text = text, .patterns = patterns });
        return true;
    }

    pub fn pop(self: *Rules) void {
        const set = self.sets.pop().?;
        self.alloc.free(set.base);
        self.alloc.free(set.text);
        self.alloc.free(set.patterns);
    }

    /// Whether the workspace-relative `path` is ignored by the loaded sets.
    pub fn ignored(self: *const Rules, path: []const u8, is_dir: bool) bool {
        var result = false;
        for (self.sets.items) |set| {
            const local = relative(set.base, path) orelse continue;
            const name = std.fs.path.basename(local);
            for (set.patterns) |pattern| {
                if (pattern.dir_only and !is_dir) continue;
                const subject = if (pattern.anchored) local else name;
                if (glob.matchPattern(pattern.glob, subject)) result = !pattern.negate;
            }
        }
        return result;
    }
};

/// `path` relative to `base`, or null when it is not beneath it.
fn relative(base: []const u8, path: []const u8) ?[]const u8 {
    if (base.len == 0) return path;
    if (!std.mem.startsWith(u8, path, base) or path.len <= base.len or path[base.len] != '/') return null;
    return path[base.len + 1 ..];
}

/// The patterns of one file, borrowing `text`. Owned slice.
fn parse(alloc: Allocator, text: []const u8) Allocator.Error![]Pattern {
    var patterns: std.ArrayList(Pattern) = .empty;
    errdefer patterns.deinit(alloc);
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |raw| {
        var line = std.mem.trimEnd(u8, raw, " \t\r");
        if (line.len == 0 or line[0] == '#') continue;
        var pattern: Pattern = .{ .glob = undefined, .negate = false, .dir_only = false, .anchored = false };
        if (line[0] == '!') {
            pattern.negate = true;
            line = line[1..];
        } else if (line[0] == '\\') line = line[1..];
        if (line.len > 0 and line[line.len - 1] == '/') {
            pattern.dir_only = true;
            line = line[0 .. line.len - 1];
        }
        // A slash anywhere but the end ties the pattern to this directory.
        pattern.anchored = std.mem.indexOfScalar(u8, line, '/') != null;
        if (line.len > 0 and line[0] == '/') line = line[1..];
        if (line.len == 0) continue;
        pattern.glob = line;
        try patterns.append(alloc, pattern);
    }
    return patterns.toOwnedSlice(alloc);
}

// ----- tests -----

const testing = std.testing;

test "names match at any depth, anchored patterns from their directory, last match wins" {
    const alloc = testing.allocator;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(testing.io, .{ .sub_path = ".gitignore", .data = "# build output\n*.log\nbuild/\n/top.txt\ndocs/*.tmp\n!keep.log\n" });
    try tmp.dir.createDirPath(testing.io, "sub");
    try tmp.dir.writeFile(testing.io, .{ .sub_path = "sub/.gitignore", .data = "local.txt\n" });

    var rules: Rules = .{ .alloc = alloc };
    defer rules.deinit();
    try testing.expect(try rules.push(testing.io, tmp.dir, ""));
    try testing.expect(rules.ignored("a.log", false));
    try testing.expect(rules.ignored("deep/x/a.log", false));
    try testing.expect(!rules.ignored("keep.log", false));
    try testing.expect(rules.ignored("build", true));
    try testing.expect(!rules.ignored("build", false)); // a file named build
    try testing.expect(rules.ignored("x/build", true));
    try testing.expect(rules.ignored("top.txt", false));
    try testing.expect(!rules.ignored("sub/top.txt", false));
    try testing.expect(rules.ignored("docs/a.tmp", false));
    try testing.expect(!rules.ignored("other/docs/a.tmp", false));
    try testing.expect(!rules.ignored("sub/local.txt", false));

    try testing.expect(try rules.push(testing.io, tmp.dir, "sub"));
    try testing.expect(rules.ignored("sub/local.txt", false));
    try testing.expect(!rules.ignored("local.txt", false));
    rules.pop();
    try testing.expect(!rules.ignored("sub/local.txt", false));
    // A directory without the file pushes nothing.
    try testing.expect(!try rules.push(testing.io, tmp.dir, "missing"));
}
