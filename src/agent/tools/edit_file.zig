//! `edit_file` — replace unique text in a workspace file, or up to
//! `max_edits` such replacements applied in order to the running content,
//! all or none. Nothing is written on any failure.
//!
//! A match must be unique at the first tier that finds one: exact bytes;
//! whole lines with trailing whitespace and CRLF ignored; whole lines with
//! indentation ignored too, `new_string` shifted by the one indentation
//! delta found (or refused). `line` picks among several matches. A miss
//! names the most similar region of the file and what differs; the search is
//! bounded by the `similar_*` constants. The target must resolve inside the
//! workspace and be UTF-8 within 1 MiB before and after; the write is the
//! atomic replacement `write_file` uses, and the diff is on the result.
const std = @import("std");
const root = @import("root.zig");
const tui = @import("../../tui/root.zig");

const Allocator = std.mem.Allocator;

pub const tool: root.Tool = .{
    .name = "edit_file",
    .description = "Replace unique text in a workspace file: `old_string` with `new_string` (empty deletes), or `edits`, up to 16 pairs applied in order, all or none. Tolerates trailing whitespace, CRLF, indentation; `line` picks among several matches. A miss shows the closest region.",
    .parameters = "{\"type\":\"object\",\"properties\":{\"path\":{\"type\":\"string\"},\"old_string\":{\"type\":\"string\"},\"new_string\":{\"type\":\"string\"},\"line\":{\"type\":\"integer\"},\"edits\":{\"type\":\"array\",\"items\":{\"type\":\"object\",\"properties\":{\"old_string\":{\"type\":\"string\"},\"new_string\":{\"type\":\"string\"}},\"required\":[\"old_string\",\"new_string\"]}}},\"required\":[\"path\"]}",
    .label = "Editing",
    .display = "Edit",
    .subject = "path",
    .run = run,
};

const max_bytes: usize = 1024 * 1024;
/// Edits one call may carry.
pub const max_edits: usize = 16;
/// The most of a region a miss shows.
const region_lines: usize = 40;
const region_bytes: usize = 4096;
/// The line-similarity score, 0 to 1, a region needs to be shown.
const similar_threshold: f64 = 0.6;
/// The similarity search's bounds: the file's lines, windows times
/// `old_string` lines for the cheap upper bound, the bytes of each trimmed
/// line compared, and the LCS cells spent on exact scores. Past either of
/// the first two the search is skipped and the result says so.
const similar_max_lines: usize = 65_536;
const similar_max_pairs: usize = 4 * 1024 * 1024;
const similar_line_bytes: usize = 255;
const similar_max_cells: usize = 16 * 1024 * 1024;
/// The work one tier's match count may take: hash comparisons and the bytes
/// of confirmed lines, or the bytes of overlapping exact matches.
const match_max_work: usize = 16 * 1024 * 1024;

const whitespace = " \t\r";

/// One replacement; `line` picks among several matches, else the call's.
const Edit = struct { old_string: []const u8, new_string: []const u8, line: ?i64 = null };

const Args = struct {
    path: []const u8,
    old_string: ?[]const u8 = null,
    new_string: ?[]const u8 = null,
    edits: ?[]const Edit = null,
    line: ?i64 = null,
};

const usage = "edit_file: arguments must be {\"path\", \"old_string\", \"new_string\"} or {\"path\", \"edits\": [{\"old_string\", \"new_string\"}, ...]}, with an optional integer \"line\"";

fn run(workspace: root.Workspace, alloc: Allocator, arguments: []const u8) Allocator.Error!root.Result {
    const parsed = std.json.parseFromSlice(Args, alloc, arguments, .{ .ignore_unknown_fields = true }) catch |err| switch (err) {
        error.OutOfMemory => return error.OutOfMemory,
        else => return root.fail(alloc, "{s}", .{usage}),
    };
    defer parsed.deinit();
    const args = parsed.value;
    const path = args.path;
    if (path.len == 0) return root.fail(alloc, "edit_file: path is empty", .{});

    var single: [1]Edit = undefined;
    const edits: []const Edit = if (args.edits) |list| blk: {
        if (args.old_string != null or args.new_string != null) return root.fail(alloc, "edit_file: give old_string and new_string, or edits, not both", .{});
        if (list.len == 0) return root.fail(alloc, "edit_file: edits is empty", .{});
        if (list.len > max_edits) return root.fail(alloc, "edit_file: {d} edits; at most {d} in one call", .{ list.len, max_edits });
        break :blk list;
    } else blk: {
        const old = args.old_string orelse return root.fail(alloc, "edit_file: old_string is missing (or give edits)", .{});
        const new = args.new_string orelse return root.fail(alloc, "edit_file: new_string is missing", .{});
        single[0] = .{ .old_string = old, .new_string = new };
        break :blk &single;
    };
    for (edits, 1..) |edit, i| if (edit.old_string.len == 0) {
        if (edits.len == 1) return root.fail(alloc, "edit_file: old_string is empty", .{});
        return root.fail(alloc, "edit_file: edit {d} has an empty old_string", .{i});
    };
    // A call's `line` is every edit's hint unless the edit names its own;
    // a later edit's lines are of the file after the edits before it.
    var hints: [max_edits]?usize = undefined;
    for (edits, 0..) |edit, i| hints[i] = if (edit.line orelse args.line) |line|
        (if (line >= 1) std.math.cast(usize, line) orelse std.math.maxInt(usize) else return root.fail(alloc, "edit_file: line must be 1 or more", .{}))
    else
        null;

    const target = workspace.resolveTarget(alloc, path) catch |err| switch (err) {
        error.OutsideWorkspace => return root.fail(alloc, "edit_file: {s} is outside the workspace", .{path}),
        else => return root.fail(alloc, "edit_file: {s}: {s}", .{ path, @errorName(err) }),
    };
    defer alloc.free(target.abs);
    if (!target.exists) return root.fail(alloc, "edit_file: {s} does not exist", .{path});

    const stat = workspace.dir.statFile(workspace.io, target.abs, .{}) catch |err|
        return root.fail(alloc, "edit_file: {s}: {s}", .{ path, @errorName(err) });
    if (stat.kind == .directory) return root.fail(alloc, "edit_file: {s} is a directory", .{path});

    const content = workspace.dir.readFileAlloc(workspace.io, target.abs, alloc, .limited(max_bytes + 1)) catch |err|
        return root.fail(alloc, "edit_file: {s}: {s}", .{ path, @errorName(err) });
    defer alloc.free(content);
    if (content.len > max_bytes) return root.fail(alloc, "edit_file: {s} is larger than {d} bytes", .{ path, max_bytes });
    if (!std.unicode.utf8ValidateSlice(content)) return root.fail(alloc, "edit_file: {s} is not valid UTF-8", .{path});

    // A batch that repeats an old_string with no line takes its matches in
    // file order: each repeat matches only past the previous one's
    // replacement, tracked through the edits between them.
    var floors: [max_edits]?usize = @splat(null);
    var previous: [max_edits]?usize = @splat(null);
    for (edits, 0..) |edit, i| if (hints[i] == null) {
        for (0..edits.len) |j| if (j != i and hints[j] == null and std.mem.eql(u8, edits[j].old_string, edit.old_string)) {
            if (j < i) previous[i] = j;
            floors[i] = 0;
        };
    };

    var running = try alloc.dupe(u8, content);
    defer alloc.free(running);
    var applied: [max_edits]Applied = undefined;
    var ends: [max_edits]usize = undefined;
    for (edits, 0..) |edit, i| switch (try apply(alloc, running, path, edit.old_string, edit.new_string, hints[i], if (previous[i]) |j| ends[j] else floors[i])) {
        .failed => |message| {
            defer alloc.free(message);
            if (edits.len == 1) return root.fail(alloc, "edit_file: {s}", .{message});
            const after = if (i > 0) " (its line numbers are of the file after the edits before it)" else "";
            return root.fail(alloc, "edit_file: edit {d} of {d} failed, so none was applied{s}: {s}", .{ i + 1, edits.len, after, message });
        },
        .replaced => |replaced| {
            alloc.free(running);
            running = replaced.content;
            applied[i] = replaced.applied;
            applied[i].hint = hints[i];
            const done = replaced.applied;
            for (ends[0..i]) |*end| if (done.start < end.*) {
                end.* = (end.* -| done.removed) + done.inserted;
            };
            ends[i] = done.start + done.inserted;
        },
    };

    const path_owned = try alloc.dupe(u8, path);
    var change: root.Change = .{ .path = path_owned, .diff = undefined };
    change.diff = tui.diff.compute(alloc, content, running) catch |err| {
        alloc.free(path_owned);
        return err;
    };
    var keep = false;
    defer if (!keep) change.deinit(alloc);

    workspace.replaceFile(alloc, target.abs, running) catch |err|
        return root.fail(alloc, "edit_file: {s}: {s}", .{ path, @errorName(err) });

    const text = try resultText(alloc, path, applied[0..edits.len]);
    errdefer alloc.free(text);
    const counts = try root.changeSummary(alloc, change.diff.rows);
    defer alloc.free(counts);
    const summary = try alloc.print("Edited {s}: {s} lines", .{ path, counts });
    keep = true;
    return .{ .text = text, .change = change, .summary = summary };
}

/// `edited x.ts: 1 replacement`, with the chosen line when `line` picked
/// among several matches and the tier when it was not exact; a batch lists
/// every edit's line.
fn resultText(alloc: Allocator, path: []const u8, applied: []const Applied) Allocator.Error![]u8 {
    var out: std.ArrayList(u8) = .empty;
    errdefer out.deinit(alloc);
    if (applied.len == 1) {
        const one = applied[0];
        try out.print(alloc, "edited {s}: 1 replacement", .{path});
        if (one.matches > 1) if (one.hint) |near|
            try out.print(alloc, " at line {d}, the nearest to line {d} of {d} matches", .{ one.line, near, one.matches });
        if (one.tier != .exact) try out.print(alloc, " (matched {s})", .{one.tier.phrase()});
    } else {
        try out.print(alloc, "edited {s}: {d} replacements:", .{ path, applied.len });
        for (applied, 1..) |one, i| {
            try out.print(alloc, "{s} edit {d} at line {d}", .{ if (i > 1) ";" else "", i, one.line });
            if (one.matches > 1) if (one.hint) |near| try out.print(alloc, " (the nearest to line {d} of {d} matches)", .{ near, one.matches });
            if (one.tier != .exact) try out.print(alloc, " (matched {s})", .{one.tier.phrase()});
        }
    }
    return out.toOwnedSlice(alloc);
}

const Tier = enum {
    exact,
    /// Whole lines, trailing whitespace and `\r` ignored.
    trailing,
    /// Whole lines, leading whitespace ignored too.
    indentation,

    fn phrase(tier: Tier) []const u8 {
        return switch (tier) {
            .exact => "exactly",
            .trailing => "ignoring trailing whitespace and line endings",
            .indentation => "ignoring indentation",
        };
    }
};

/// One applied edit: where it replaced `removed` bytes with `inserted`.
const Applied = struct { line: usize, tier: Tier, matches: usize, hint: ?usize = null, start: usize = 0, removed: usize = 0, inserted: usize = 0 };

/// The running content after one edit (owned), or the reason it failed
/// (owned, without the `edit_file: ` prefix).
const Outcome = union(enum) {
    replaced: struct { content: []u8, applied: Applied },
    failed: []u8,
};

fn failed(alloc: Allocator, comptime format: []const u8, args: anytype) Allocator.Error!Outcome {
    return .{ .failed = try alloc.print(format, args) };
}

/// `old_string` split at `\n`; a final newline is a flag, not an empty line.
/// Each line is trimmed once: `ends` without trailing whitespace, `trims`
/// without either end's.
const OldLines = struct {
    items: [][]const u8,
    ends: [][]const u8,
    trims: [][]const u8,
    trailing_newline: bool,

    fn init(alloc: Allocator, old: []const u8) Allocator.Error!OldLines {
        const trailing_newline = old.len > 1 and old[old.len - 1] == '\n';
        const body = if (trailing_newline) old[0 .. old.len - 1] else old;
        const count = std.mem.countScalar(u8, body, '\n') + 1;
        const items = try alloc.alloc([]const u8, count);
        errdefer alloc.free(items);
        const ends = try alloc.alloc([]const u8, count);
        errdefer alloc.free(ends);
        const trims = try alloc.alloc([]const u8, count);
        var it = std.mem.splitScalar(u8, body, '\n');
        for (items, ends, trims) |*item, *end, *trim| {
            item.* = it.next() orelse "";
            end.* = std.mem.trimEnd(u8, item.*, whitespace);
            trim.* = std.mem.trimStart(u8, end.*, whitespace);
        }
        return .{ .items = items, .ends = ends, .trims = trims, .trailing_newline = trailing_newline };
    }

    fn deinit(self: OldLines, alloc: Allocator) void {
        alloc.free(self.items);
        alloc.free(self.ends);
        alloc.free(self.trims);
    }
};

/// One match: the bytes it replaces and the 1-based line it starts on.
const Match = struct { start: usize, end: usize, line: usize };

/// Every match of one tier, overlapping ones included: how many, the first,
/// and the one starting nearest the hint (the earlier on a tie). `cut` when
/// the count stopped at `match_max_work`.
const Found = struct {
    /// Matches starting before this byte are not counted.
    floor: usize = 0,
    count: usize = 0,
    first: Match = undefined,
    nearest: Match = undefined,
    cut: bool = false,

    fn add(self: *Found, match: Match, hint: ?usize) void {
        if (match.start < self.floor) return;
        self.count += 1;
        if (self.count == 1) {
            self.first = match;
            self.nearest = match;
        } else if (hint) |near| {
            if (distance(match.line, near) < distance(self.nearest.line, near)) self.nearest = match;
        }
    }

    fn distance(a: usize, b: usize) usize {
        return if (a > b) a - b else b - a;
    }
};

/// `floor`, for an old_string the batch repeats, is where its matches may
/// start; the first of them is taken.
fn apply(alloc: Allocator, content: []const u8, path: []const u8, old: []const u8, new: []const u8, hint: ?usize, floor: ?usize) Allocator.Error!Outcome {
    const lines = try OldLines.init(alloc, old);
    defer lines.deinit(alloc);
    // A whitespace-only old_string would match blank lines once whitespace is ignored.
    const tolerant = std.mem.trim(u8, old, whitespace ++ "\n").len > 0;
    for ([_]Tier{ .exact, .trailing, .indentation }) |tier| {
        if (tier != .exact and !tolerant) break;
        const found = if (tier == .exact)
            findExact(content, old, hint, floor orelse 0, match_max_work)
        else
            try findLines(alloc, content, lines, tier, hint, floor orelse 0, match_max_work);
        if (found.cut) {
            if (tier == .exact)
                return failed(alloc, "old_string appears at least {d} times in {s}, overlapping; it must be unique (add surrounding context)", .{ found.count, path });
            return failed(alloc, "the file is too long to match ignoring whitespace; give old_string exactly", .{});
        }
        if (found.count == 0) continue;
        if (std.mem.eql(u8, old, new))
            return failed(alloc, "nothing to change: the file already has this text at line {d}; the problem is elsewhere", .{found.first.line});
        if (found.count > 1 and hint == null and floor == null) {
            if (tier == .exact)
                return failed(alloc, "old_string appears {d} times in {s}; it must be unique (add surrounding context, or give line)", .{ found.count, path });
            return failed(alloc, "old_string appears {d} times in {s} ({s}); it must be unique (add surrounding context, or give line)", .{ found.count, path, tier.phrase() });
        }
        const pick = if (found.count > 1) found.nearest else found.first;
        return replace(alloc, content, path, pick, tier, lines, new, found.count);
    }
    if (floor) |at| if (at > 0)
        return failed(alloc, "old_string repeats an earlier edit's and has no match after that one's in {s} (a repeated old_string takes its matches in file order)", .{path});
    return .{ .failed = try notFound(alloc, content, path, lines) };
}

/// Every byte offset where `old` starts, so overlapping matches count; each
/// match charges its length to `max_work`.
fn findExact(content: []const u8, old: []const u8, hint: ?usize, floor: usize, max_work: usize) Found {
    var found: Found = .{ .floor = floor };
    var search: usize = 0;
    var line: usize = 1;
    var counted: usize = 0;
    var work: usize = 0;
    while (std.mem.indexOfPos(u8, content, search, old)) |at| {
        line += std.mem.countScalar(u8, content[counted..at], '\n');
        counted = at;
        found.add(.{ .start = at, .end = at + old.len, .line = line }, hint);
        work += old.len;
        if (work > max_work) {
            found.cut = true;
            break;
        }
        search = at + 1;
    }
    return found;
}

fn normalized(line: []const u8, tier: Tier) []const u8 {
    return if (tier == .trailing) std.mem.trimEnd(u8, line, whitespace) else std.mem.trim(u8, line, whitespace);
}

/// Whole-line matches at a tolerant tier, from every start line. Lines are
/// compared by hash, a full window confirmed byte for byte; a hash
/// comparison costs one unit of `max_work` and a confirmed line one plus its
/// length, and past the budget the result is `cut`.
fn findLines(alloc: Allocator, content: []const u8, lines: OldLines, tier: Tier, hint: ?usize, floor: usize, max_work: usize) Allocator.Error!Found {
    var found: Found = .{ .floor = floor };
    const old = if (tier == .trailing) lines.ends else lines.trims;
    const k = old.len;
    var file: std.ArrayList(FileLine) = .empty;
    defer file.deinit(alloc);
    var pos: usize = 0;
    while (pos < content.len) {
        const eol = std.mem.indexOfScalarPos(u8, content, pos, '\n') orelse content.len;
        try file.append(alloc, .{ .start = pos, .end = eol });
        pos = eol + 1;
    }
    const n = file.items.len;
    if (k > n) return found;
    const old_hash = try alloc.alloc(u64, k);
    defer alloc.free(old_hash);
    for (old, old_hash) |text, *h| h.* = std.hash.Wyhash.hash(0, text);
    const file_hash = try alloc.alloc(u64, n);
    defer alloc.free(file_hash);
    for (file.items, file_hash) |line, *h| h.* = std.hash.Wyhash.hash(0, normalized(content[line.start..line.end], tier));

    var work: usize = 0;
    for (0..n - k + 1) |i| {
        var j: usize = 0;
        while (j < k) : (j += 1) {
            work += 1;
            if (file_hash[i + j] != old_hash[j]) break;
        }
        if (j == k) {
            const same = for (file.items[i..][0..k], old) |line, text| {
                const mine = normalized(content[line.start..line.end], tier);
                work += 1 + mine.len;
                if (!std.mem.eql(u8, mine, text)) break false;
            } else true;
            if (same) {
                const last = file.items[i + k - 1];
                const next = @min(last.end + 1, content.len);
                const end = if (lines.trailing_newline) next else if (last.end > last.start and content[last.end - 1] == '\r') last.end - 1 else last.end;
                found.add(.{ .start = file.items[i].start, .end = end, .line = i + 1 }, hint);
            }
        }
        if (work > max_work) {
            found.cut = true;
            return found;
        }
    }
    return found;
}

fn leading(line: []const u8) []const u8 {
    const body = std.mem.trimStart(u8, line, " \t");
    return line[0 .. line.len - body.len];
}

fn replace(alloc: Allocator, content: []const u8, path: []const u8, pick: Match, tier: Tier, lines: OldLines, new: []const u8, matches: usize) Allocator.Error!Outcome {
    var replacement: []const u8 = new;
    var owned: std.ArrayList(u8) = .empty;
    defer owned.deinit(alloc);
    if (tier == .indentation) {
        const window = try windowAt(alloc, content, pick.start, lines.items.len);
        defer alloc.free(window);
        switch (shift(content, window, lines.items)) {
            .mixed => |at| {
                var out: std.ArrayList(u8) = .empty;
                errdefer out.deinit(alloc);
                const last = pick.line + window.len - 1;
                try out.print(alloc, "old_string matches lines {d} to {d} of {s} only ignoring indentation, and not by one shift on every line: ", .{ pick.line, last, path });
                try writeDiagnosis(&out, alloc, .{ .indentation = .{ .line = pick.line + at.index, .file = at.file, .yours = at.yours } });
                try out.appendSlice(alloc, "; copy the indentation as the file has it. Those lines are:\n");
                try writeRegion(&out, alloc, content, window, pick.line);
                return .{ .failed = try out.toOwnedSlice(alloc) };
            },
            .uniform => |delta| {
                var it = std.mem.splitScalar(u8, new, '\n');
                var index: usize = 0;
                while (it.next()) |piece| : (index += 1) {
                    if (index > 0) try owned.append(alloc, '\n');
                    if (std.mem.trim(u8, piece, whitespace).len == 0) {
                        try owned.appendSlice(alloc, piece);
                    } else if (delta.add) {
                        try owned.appendSlice(alloc, delta.text);
                        try owned.appendSlice(alloc, piece);
                    } else if (std.mem.startsWith(u8, piece, delta.text)) {
                        try owned.appendSlice(alloc, piece[delta.text.len..]);
                    } else {
                        var out: std.ArrayList(u8) = .empty;
                        errdefer out.deinit(alloc);
                        try out.print(alloc, "old_string matches lines {d} to {d} of {s} only ignoring indentation, the file's being ", .{ pick.line, pick.line + window.len - 1, path });
                        try writeIndent(&out, alloc, delta.text);
                        try out.print(alloc, " shallower, but line {d} of new_string has less indentation than that to remove; copy the indentation as the file has it", .{index + 1});
                        return .{ .failed = try out.toOwnedSlice(alloc) };
                    }
                }
                replacement = owned.items;
            },
        }
    }
    var crlf: ?[]u8 = null;
    defer if (crlf) |bytes| alloc.free(bytes);
    if (tier != .exact and usesCrlf(content, pick.start) and std.mem.indexOfScalar(u8, replacement, '\r') == null) {
        const converted = try std.mem.replaceOwned(u8, alloc, replacement, "\n", "\r\n");
        crlf = converted;
        replacement = converted;
    }

    // The tools read and write at most `max_bytes`: an edit past it would leave a file they refuse.
    const new_len = content.len - (pick.end - pick.start) + replacement.len;
    if (new_len > max_bytes) return failed(alloc, "the edit would make {s} {d} bytes, over the {d} the file tools handle; split the file or use bash", .{ path, new_len, max_bytes });
    const updated = try alloc.alloc(u8, new_len);
    @memcpy(updated[0..pick.start], content[0..pick.start]);
    @memcpy(updated[pick.start..][0..replacement.len], replacement);
    @memcpy(updated[pick.start + replacement.len ..], content[pick.end..]);
    return .{ .replaced = .{ .content = updated, .applied = .{ .line = pick.line, .tier = tier, .matches = matches, .start = pick.start, .removed = pick.end - pick.start, .inserted = replacement.len } } };
}

/// Whether the match's first line ends in CRLF; a file mixing endings gets the first line's.
fn usesCrlf(content: []const u8, start: usize) bool {
    const eol = std.mem.indexOfScalarPos(u8, content, start, '\n') orelse return false;
    return eol > start and content[eol - 1] == '\r';
}

/// A line of the file: `end` excludes its newline, not a `\r` before it.
const FileLine = struct { start: usize, end: usize };

/// Up to `count` lines from `start`, which begins a line. Caller owns.
fn windowAt(alloc: Allocator, content: []const u8, start: usize, count: usize) Allocator.Error![]FileLine {
    var list: std.ArrayList(FileLine) = .empty;
    errdefer list.deinit(alloc);
    var cursor = start;
    while (list.items.len < count and cursor < content.len) {
        const eol = std.mem.indexOfScalarPos(u8, content, cursor, '\n') orelse content.len;
        try list.append(alloc, .{ .start = cursor, .end = eol });
        cursor = eol + 1;
    }
    return list.toOwnedSlice(alloc);
}

/// The indentation `new_string` must gain (`add`) or lose to sit where the
/// match is: the same on every non-blank line, or the first line it differs on.
const Shift = union(enum) {
    uniform: struct { add: bool, text: []const u8 },
    mixed: struct { index: usize, file: []const u8, yours: []const u8 },
};

fn shift(content: []const u8, window: []const FileLine, old_lines: []const []const u8) Shift {
    var found: ?@FieldType(Shift, "uniform") = null;
    for (window, old_lines[0..window.len], 0..) |line, old, index| {
        if (std.mem.trim(u8, old, whitespace).len == 0) continue;
        const file_indent = leading(content[line.start..line.end]);
        const old_indent = leading(old);
        const this: @FieldType(Shift, "uniform") = if (std.mem.endsWith(u8, file_indent, old_indent))
            .{ .add = true, .text = file_indent[0 .. file_indent.len - old_indent.len] }
        else if (std.mem.endsWith(u8, old_indent, file_indent))
            .{ .add = false, .text = old_indent[0 .. old_indent.len - file_indent.len] }
        else
            return .{ .mixed = .{ .index = index, .file = file_indent, .yours = old_indent } };
        if (found) |first| {
            const same = std.mem.eql(u8, first.text, this.text) and (this.text.len == 0 or first.add == this.add);
            if (!same) return .{ .mixed = .{ .index = index, .file = file_indent, .yours = old_indent } };
        } else found = this;
    }
    return .{ .uniform = found orelse .{ .add = true, .text = "" } };
}

const Closest = union(enum) {
    none,
    too_long,
    /// The cell budget ran out before any window reached the threshold.
    cut_short,
    /// `complete` is false when the budget ran out first: similar, not
    /// necessarily the closest.
    found: struct { first: usize, score: f64, complete: bool },
};

const Histogram = [32]u8;

/// The trimmed prefix of a line the similarity compares.
fn key(line: []const u8) []const u8 {
    const trimmed = std.mem.trim(u8, line, whitespace);
    return trimmed[0..@min(trimmed.len, similar_line_bytes)];
}

fn histogram(text: []const u8) Histogram {
    var counts: Histogram = @splat(0);
    // At most `similar_line_bytes` (255) bytes, so no bucket overflows.
    for (text) |byte| counts[byte & 31] += 1;
    return counts;
}

/// Dice ratio of two counts against the lengths; equal empty lines are 1.
fn ratio(common: usize, a: usize, b: usize) f64 {
    if (a + b == 0) return 1;
    return @as(f64, @floatFromInt(2 * common)) / @as(f64, @floatFromInt(a + b));
}

/// The longest common subsequence of two keys, at most 255 bytes each.
fn lcs(a: []const u8, b: []const u8) usize {
    var row: [similar_line_bytes + 1]u16 = @splat(0);
    for (a) |ca| {
        var diagonal: u16 = 0;
        for (b, 1..) |cb, j| {
            const up = row[j];
            row[j] = if (ca == cb) diagonal + 1 else @max(up, row[j - 1]);
            diagonal = up;
        }
    }
    return row[b.len];
}

/// The window of `old_lines.len` file lines whose mean per-line LCS ratio
/// (lines trimmed) is highest. Each pair's bucketed byte-count overlap is an
/// upper bound on its LCS, so windows are scored exactly in order of their
/// bound until no remaining bound can win or the cell budget is spent.
fn closest(alloc: Allocator, file: []const FileLine, content: []const u8, lines: OldLines, max_cells: usize) Allocator.Error!Closest {
    const k = lines.trims.len;
    const n = file.len;
    if (k == 0 or k > n) return .none;
    const windows = n - k + 1;
    if (windows > similar_max_pairs / k) return .too_long;

    const file_hist = try alloc.alloc(Histogram, n);
    defer alloc.free(file_hist);
    const file_len = try alloc.alloc(u16, n);
    defer alloc.free(file_len);
    for (file, file_hist, file_len) |line, *h, *len| {
        const text = key(content[line.start..line.end]);
        h.* = histogram(text);
        len.* = @truncate(text.len);
    }
    const old_keys = try alloc.alloc([]const u8, k);
    defer alloc.free(old_keys);
    const old_hist = try alloc.alloc(Histogram, k);
    defer alloc.free(old_hist);
    for (lines.trims, old_keys, old_hist) |trimmed, *old_key, *h| {
        old_key.* = trimmed[0..@min(trimmed.len, similar_line_bytes)];
        h.* = histogram(old_key.*);
    }

    const bounds = try alloc.alloc(f64, windows);
    defer alloc.free(bounds);
    const order = try alloc.alloc(u32, windows);
    defer alloc.free(order);
    for (bounds, order, 0..) |*bound, *slot, i| {
        var sum: f64 = 0;
        for (old_hist, old_keys, 0..) |oh, old_key, j| {
            const fh: @Vector(32, u8) = file_hist[i + j];
            const common = @reduce(.Add, @as(@Vector(32, u16), @min(fh, @as(@Vector(32, u8), oh))));
            sum += ratio(common, file_len[i + j], old_key.len);
        }
        bound.* = sum / @as(f64, @floatFromInt(k));
        slot.* = @truncate(i);
    }
    const Order = struct {
        bounds: []const f64,
        fn before(self: @This(), a: u32, b: u32) bool {
            return self.bounds[a] > self.bounds[b] or (self.bounds[a] == self.bounds[b] and a < b);
        }
    };
    std.mem.sortUnstable(u32, order, Order{ .bounds = bounds }, Order.before);

    var best: f64 = -1;
    var best_first: usize = 0;
    var cells: usize = 0;
    var complete = true;
    for (order) |slot| {
        const i: usize = slot;
        if (bounds[i] < similar_threshold or bounds[i] <= best) break;
        var need: usize = 0;
        for (old_keys, 0..) |old_key, j| need += @as(usize, file_len[i + j]) * old_key.len;
        if (cells + need > max_cells) {
            complete = false;
            break;
        }
        cells += need;
        var sum: f64 = 0;
        for (old_keys, 0..) |b, j| {
            const a = key(content[file[i + j].start..file[i + j].end]);
            sum += ratio(lcs(a, b), a.len, b.len);
        }
        const score = sum / @as(f64, @floatFromInt(k));
        if (score > best) {
            best = score;
            best_first = i;
        }
    }
    if (best < similar_threshold) return if (complete) .none else .cut_short;
    return .{ .found = .{ .first = best_first, .score = best, .complete = complete } };
}

/// What differs between a region and `old_string`, line by line: the first
/// line whose trimmed text differs, else the first whitespace difference.
const Diagnosis = union(enum) {
    text: Where,
    backslashes: Where,
    indentation: struct { line: usize, file: []const u8, yours: []const u8 },
    line_endings: usize,
    trailing_whitespace: usize,
    none,

    const Where = struct { line: usize, yours: usize };
};

fn diagnose(file: []const []const u8, old: []const []const u8, first_line: usize) Diagnosis {
    for (file, old[0..file.len], 0..) |f, o, j| {
        const ft = std.mem.trim(u8, f, whitespace);
        const ot = std.mem.trim(u8, o, whitespace);
        if (std.mem.eql(u8, ft, ot)) continue;
        const where: Diagnosis.Where = .{ .line = first_line + j, .yours = j + 1 };
        return if (eqlWithoutBackslashes(ft, ot)) .{ .backslashes = where } else .{ .text = where };
    }
    for (file, old[0..file.len], 0..) |f, o, j| {
        const line = first_line + j;
        if (!std.mem.eql(u8, leading(f), leading(o))) return .{ .indentation = .{ .line = line, .file = leading(f), .yours = leading(o) } };
        if (std.mem.endsWith(u8, f, "\r") != std.mem.endsWith(u8, o, "\r")) return .{ .line_endings = line };
        if (!std.mem.eql(u8, f, o)) return .{ .trailing_whitespace = line };
    }
    return .none;
}

fn eqlWithoutBackslashes(a: []const u8, b: []const u8) bool {
    var i: usize = 0;
    var j: usize = 0;
    while (true) {
        while (i < a.len and a[i] == '\\') i += 1;
        while (j < b.len and b[j] == '\\') j += 1;
        if (i == a.len or j == b.len) return i == a.len and j == b.len;
        if (a[i] != b[j]) return false;
        i += 1;
        j += 1;
    }
}

fn writeDiagnosis(out: *std.ArrayList(u8), alloc: Allocator, diagnosis: Diagnosis) Allocator.Error!void {
    switch (diagnosis) {
        .text => |at| try out.print(alloc, "text differs at line {d} (your line {d})", .{ at.line, at.yours }),
        .backslashes => |at| try out.print(alloc, "backslashes differ at line {d} (your line {d})", .{ at.line, at.yours }),
        .indentation => |at| {
            try out.print(alloc, "indentation differs at line {d} (file: ", .{at.line});
            try writeIndent(out, alloc, at.file);
            try out.appendSlice(alloc, ", yours: ");
            try writeIndent(out, alloc, at.yours);
            try out.append(alloc, ')');
        },
        .line_endings => |line| try out.print(alloc, "line endings (CRLF) differ at line {d}", .{line}),
        .trailing_whitespace => |line| try out.print(alloc, "trailing whitespace differs at line {d}", .{line}),
        .none => try out.appendSlice(alloc, "the lines match only in part; old_string must cover them exactly"),
    }
}

fn writeIndent(out: *std.ArrayList(u8), alloc: Allocator, indent: []const u8) Allocator.Error!void {
    const tabs = std.mem.countScalar(u8, indent, '\t');
    const spaces = indent.len - tabs;
    if (tabs == 0) return out.print(alloc, "{d} space{s}", .{ spaces, if (spaces == 1) "" else "s" });
    try out.print(alloc, "{d} tab{s}", .{ tabs, if (tabs == 1) "" else "s" });
    if (spaces > 0) try out.print(alloc, " and {d} space{s}", .{ spaces, if (spaces == 1) "" else "s" });
}

/// The region's text as the file has it, at most `region_lines` lines and
/// `region_bytes` bytes, the cut marked with where to read on.
fn writeRegion(out: *std.ArrayList(u8), alloc: Allocator, content: []const u8, window: []const FileLine, first_line: usize) Allocator.Error!void {
    var bytes: usize = 0;
    for (window, 0..) |line, j| {
        const text = content[line.start..line.end];
        const over_bytes = bytes + text.len + 1 > region_bytes;
        if (j == region_lines or over_bytes) {
            if (j == 0) {
                // `text` is at least `region_bytes` long here, so `keep` indexes it.
                var keep = region_bytes - 1;
                while (keep > 0 and (text[keep] & 0xC0) == 0x80) keep -= 1;
                try out.appendSlice(alloc, text[0..keep]);
                try out.print(alloc, "\n[line {d} cut at {d} bytes]", .{ first_line, keep });
                if (window.len > 1) try out.print(alloc, "\n[cut: lines {d} to {d} not shown; read_file offset={d}]", .{ first_line + 1, first_line + window.len - 1, first_line + 1 });
                return;
            }
            try out.print(alloc, "[cut: lines {d} to {d} not shown; read_file offset={d}]", .{ first_line + j, first_line + window.len - 1, first_line + j });
            return;
        }
        try out.appendSlice(alloc, text);
        try out.append(alloc, '\n');
        bytes += text.len + 1;
    }
}

/// The not-found message: the closest region with its diagnosis and text,
/// or why there is none.
fn notFound(alloc: Allocator, content: []const u8, path: []const u8, lines: OldLines) Allocator.Error![]u8 {
    const old_lines = lines.items;
    var file: std.ArrayList(FileLine) = .empty;
    defer file.deinit(alloc);
    var pos: usize = 0;
    var too_long = false;
    while (pos < content.len) {
        if (file.items.len == similar_max_lines) {
            too_long = true;
            break;
        }
        const eol = std.mem.indexOfScalarPos(u8, content, pos, '\n') orelse content.len;
        try file.append(alloc, .{ .start = pos, .end = eol });
        pos = eol + 1;
    }
    const best: Closest = if (too_long) .too_long else try closest(alloc, file.items, content, lines, similar_max_cells);
    switch (best) {
        .none => return alloc.print("old_string was not found in {s}; no similar region; read the file before editing", .{path}),
        .too_long => return alloc.print("old_string was not found in {s}; the file is too long to search for a similar region; read the file before editing", .{path}),
        .cut_short => return alloc.print("old_string was not found in {s}; the search for a similar region was cut short; read the file before editing", .{path}),
        .found => |region| {
            const window = file.items[region.first..][0..old_lines.len];
            const texts = try alloc.alloc([]const u8, window.len);
            defer alloc.free(texts);
            for (window, texts) |line, *text| text.* = content[line.start..line.end];
            const first_line = region.first + 1;
            var out: std.ArrayList(u8) = .empty;
            errdefer out.deinit(alloc);
            const which = if (region.complete) "the closest region is" else "the search was cut short; a similar region is";
            try out.print(alloc, "old_string was not found in {s}; {s} lines {d} to {d} (similarity {d:.2}): ", .{ path, which, first_line, first_line + window.len - 1, region.score });
            try writeDiagnosis(&out, alloc, diagnose(texts, old_lines, first_line));
            try out.appendSlice(alloc, ". Those lines are:\n");
            try writeRegion(&out, alloc, content, window, first_line);
            return out.toOwnedSlice(alloc);
        },
    }
}

// ----- tests -----

const testing = std.testing;

const Fixture = struct {
    tmp: testing.TmpDir,
    ws: root.Workspace,
    root_path: [:0]u8,

    fn init(alloc: std.mem.Allocator) !Fixture {
        var tmp = testing.tmpDir(.{});
        const root_path = try tmp.dir.realPathFileAlloc(testing.io, ".", alloc);
        return .{ .tmp = tmp, .ws = .{ .io = testing.io, .dir = tmp.dir, .root = root_path }, .root_path = root_path };
    }
    fn deinit(self: *Fixture, alloc: std.mem.Allocator) void {
        self.tmp.cleanup();
        alloc.free(self.root_path);
    }
};

test "edit_file replaces one unique match and diffs the change" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "a.txt", .data = "one\ntwo\nthree\n" });
    var result = try run(fixture.ws, alloc, "{\"path\":\"a.txt\",\"old_string\":\"two\",\"new_string\":\"TWO\"}");
    defer result.deinit(alloc);
    try testing.expect(!result.is_error);
    try testing.expect(std.mem.indexOf(u8, result.text, "1 replacement") != null);
    try testing.expect(std.mem.indexOf(u8, result.change.?.diff.unified, "-two") != null);
    try testing.expect(std.mem.indexOf(u8, result.change.?.diff.unified, "+TWO") != null);
    try testing.expect(std.mem.endsWith(u8, result.summary.?, ": +1 −1 lines"));
    const contents = try fixture.tmp.dir.readFileAlloc(testing.io, "a.txt", alloc, .limited(1024));
    defer alloc.free(contents);
    try testing.expectEqualStrings("one\nTWO\nthree\n", contents);
}

test "edit_file refuses an edit past the bound, and keeps the file's mode" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    const big: [max_bytes - 4]u8 = @splat('a');
    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "big.txt", .data = &big ++ "MARK" });
    var refused = try run(fixture.ws, alloc, "{\"path\":\"big.txt\",\"old_string\":\"MARK\",\"new_string\":\"MARKMARK\"}");
    defer refused.deinit(alloc);
    try testing.expect(refused.is_error);
    try testing.expect(std.mem.indexOf(u8, refused.text, "over the") != null);

    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "run.sh", .data = "echo one\n" });
    {
        const file = try fixture.tmp.dir.openFile(testing.io, "run.sh", .{});
        defer file.close(testing.io);
        try file.setPermissions(testing.io, .fromMode(0o750));
    }
    var edited = try run(fixture.ws, alloc, "{\"path\":\"run.sh\",\"old_string\":\"one\",\"new_string\":\"two\"}");
    defer edited.deinit(alloc);
    try testing.expect(!edited.is_error);
    const st = try fixture.tmp.dir.statFile(testing.io, "run.sh", .{});
    try testing.expectEqual(@as(u32, 0o750), @as(u32, @intCast(st.permissions.toMode())) & 0o777);
}

test "edit_file with two matches changes nothing and returns the count" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "a.txt", .data = "x\nx\n" });
    var result = try run(fixture.ws, alloc, "{\"path\":\"a.txt\",\"old_string\":\"x\",\"new_string\":\"y\"}");
    defer result.deinit(alloc);
    try testing.expect(result.is_error);
    try testing.expect(std.mem.indexOf(u8, result.text, "2 times") != null);
    try testing.expect(result.change == null);
    const contents = try fixture.tmp.dir.readFileAlloc(testing.io, "a.txt", alloc, .limited(1024));
    defer alloc.free(contents);
    try testing.expectEqualStrings("x\nx\n", contents);
}

test "edit_file rejects a missing match, an empty old_string, and a no-op" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "a.txt", .data = "hello\n" });

    var missing = try run(fixture.ws, alloc, "{\"path\":\"a.txt\",\"old_string\":\"nope\",\"new_string\":\"y\"}");
    defer missing.deinit(alloc);
    try testing.expect(missing.is_error);
    try testing.expect(std.mem.indexOf(u8, missing.text, "not found") != null);

    var empty = try run(fixture.ws, alloc, "{\"path\":\"a.txt\",\"old_string\":\"\",\"new_string\":\"y\"}");
    defer empty.deinit(alloc);
    try testing.expect(empty.is_error);

    var noop = try run(fixture.ws, alloc, "{\"path\":\"a.txt\",\"old_string\":\"hello\",\"new_string\":\"hello\"}");
    defer noop.deinit(alloc);
    try testing.expect(noop.is_error);

    const contents = try fixture.tmp.dir.readFileAlloc(testing.io, "a.txt", alloc, .limited(1024));
    defer alloc.free(contents);
    try testing.expectEqualStrings("hello\n", contents);
}

fn expectFile(fixture: *Fixture, name: []const u8, want: []const u8) !void {
    const contents = try fixture.tmp.dir.readFileAlloc(testing.io, name, testing.allocator, .limited(max_bytes + 1));
    defer testing.allocator.free(contents);
    try testing.expectEqualStrings(want, contents);
}

fn expectContains(haystack: []const u8, needle: []const u8) !void {
    if (std.mem.indexOf(u8, haystack, needle) == null) {
        std.debug.print("expected to find\n  {s}\nin\n  {s}\n", .{ needle, haystack });
        return error.TestUnexpectedResult;
    }
}

test "edit_file's second tier ignores trailing whitespace and keeps the file's CRLF" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "a.txt", .data = "a  \nb\t\nc\n" });
    var trailing = try run(fixture.ws, alloc, "{\"path\":\"a.txt\",\"old_string\":\"a\\nb\",\"new_string\":\"A\\nB\"}");
    defer trailing.deinit(alloc);
    try testing.expect(!trailing.is_error);
    try testing.expectEqualStrings("edited a.txt: 1 replacement (matched ignoring trailing whitespace and line endings)", trailing.text);
    try expectFile(&fixture, "a.txt", "A\nB\nc\n");

    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "w.txt", .data = "one\r\ntwo\r\nthree\r\n" });
    var crlf = try run(fixture.ws, alloc, "{\"path\":\"w.txt\",\"old_string\":\"one\\ntwo\\n\",\"new_string\":\"1\\n2\\n\"}");
    defer crlf.deinit(alloc);
    try testing.expect(!crlf.is_error);
    try expectFile(&fixture, "w.txt", "1\r\n2\r\nthree\r\n");
}

test "edit_file's third tier re-indents new_string by the one delta found" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "a.zig", .data = "fn f() {\n        if (x) {\n            y();\n\n        }\n}\n" });
    var deeper = try run(fixture.ws, alloc, "{\"path\":\"a.zig\",\"old_string\":\"    if (x) {\\n        y();\\n\\n    }\",\"new_string\":\"    if (z) {\\n        w();\\n\\n    }\"}");
    defer deeper.deinit(alloc);
    try testing.expectEqualStrings("edited a.zig: 1 replacement (matched ignoring indentation)", deeper.text);
    try expectFile(&fixture, "a.zig", "fn f() {\n        if (z) {\n            w();\n\n        }\n}\n");

    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "b.txt", .data = "  a\n    b\n" });
    var shallower = try run(fixture.ws, alloc, "{\"path\":\"b.txt\",\"old_string\":\"    a\\n      b\",\"new_string\":\"    c\\n      d\"}");
    defer shallower.deinit(alloc);
    try testing.expect(!shallower.is_error);
    try expectFile(&fixture, "b.txt", "  c\n    d\n");
}

test "edit_file refuses an indentation that does not shift by one delta, and a new_string it cannot shift" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "a.txt", .data = "    a\n        b\n" });
    var mixed = try run(fixture.ws, alloc, "{\"path\":\"a.txt\",\"old_string\":\"  a\\n  b\",\"new_string\":\"x\"}");
    defer mixed.deinit(alloc);
    try testing.expect(mixed.is_error);
    try expectContains(mixed.text, "matches lines 1 to 2 of a.txt only ignoring indentation");
    try expectContains(mixed.text, "indentation differs at line 2 (file: 8 spaces, yours: 2 spaces)");
    try expectContains(mixed.text, "Those lines are:\n    a\n        b\n");
    try expectFile(&fixture, "a.txt", "    a\n        b\n");

    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "b.txt", .data = "a\nb\n" });
    var shallow = try run(fixture.ws, alloc, "{\"path\":\"b.txt\",\"old_string\":\"  a\\n  b\",\"new_string\":\"  x\\ny\"}");
    defer shallow.deinit(alloc);
    try testing.expect(shallow.is_error);
    try expectContains(shallow.text, "line 2 of new_string has less indentation");
    try expectFile(&fixture, "b.txt", "a\nb\n");
}

test "edit_file takes the first tier that matches, and several matches at a tier still fail" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    // Exact once, and twice ignoring indentation: the exact match wins.
    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "a.txt", .data = "  x\nx\n" });
    var exact = try run(fixture.ws, alloc, "{\"path\":\"a.txt\",\"old_string\":\"  x\",\"new_string\":\"  y\"}");
    defer exact.deinit(alloc);
    try testing.expectEqualStrings("edited a.txt: 1 replacement", exact.text);
    try expectFile(&fixture, "a.txt", "  y\nx\n");

    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "b.txt", .data = "  q\n    q\n" });
    var twice = try run(fixture.ws, alloc, "{\"path\":\"b.txt\",\"old_string\":\"\\tq\",\"new_string\":\"r\"}");
    defer twice.deinit(alloc);
    try testing.expect(twice.is_error);
    try expectContains(twice.text, "appears 2 times in b.txt (ignoring indentation)");
    try expectFile(&fixture, "b.txt", "  q\n    q\n");
}

test "edit_file's line hint picks the nearest of several matches and says which" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "a.txt", .data = "x\ny\nx\ny\nx\n" });
    var hinted = try run(fixture.ws, alloc, "{\"path\":\"a.txt\",\"old_string\":\"x\",\"new_string\":\"Z\",\"line\":4}");
    defer hinted.deinit(alloc);
    try testing.expectEqualStrings("edited a.txt: 1 replacement at line 3, the nearest to line 4 of 3 matches", hinted.text);
    try expectFile(&fixture, "a.txt", "x\ny\nZ\ny\nx\n");

    var zero = try run(fixture.ws, alloc, "{\"path\":\"a.txt\",\"old_string\":\"x\",\"new_string\":\"Z\",\"line\":0}");
    defer zero.deinit(alloc);
    try testing.expect(zero.is_error);
}

test "edit_file with identical strings says where the file already has the text" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "a.txt", .data = "a\nhello\n" });
    var same = try run(fixture.ws, alloc, "{\"path\":\"a.txt\",\"old_string\":\"hello\",\"new_string\":\"hello\"}");
    defer same.deinit(alloc);
    try testing.expect(same.is_error);
    try testing.expectEqualStrings("edit_file: nothing to change: the file already has this text at line 2; the problem is elsewhere", same.text);
}

test "edit_file applies a batch in order, and a failing edit leaves the file untouched" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "a.txt", .data = "a\nb\nc\n" });
    var broken = try run(fixture.ws, alloc, "{\"path\":\"a.txt\",\"edits\":[{\"old_string\":\"a\",\"new_string\":\"A\"},{\"old_string\":\"zzzz\",\"new_string\":\"Z\"},{\"old_string\":\"c\",\"new_string\":\"C\"}]}");
    defer broken.deinit(alloc);
    try testing.expect(broken.is_error);
    try testing.expect(broken.change == null);
    try expectContains(broken.text, "edit_file: edit 2 of 3 failed, so none was applied (its line numbers are of the file after the edits before it): old_string was not found");
    try expectFile(&fixture, "a.txt", "a\nb\nc\n");

    // The second edit sees the first's result.
    var chained = try run(fixture.ws, alloc, "{\"path\":\"a.txt\",\"edits\":[{\"old_string\":\"a\",\"new_string\":\"X\"},{\"old_string\":\"X\\nb\",\"new_string\":\"Y\"},{\"old_string\":\"c\",\"new_string\":\"C\"}]}");
    defer chained.deinit(alloc);
    try testing.expectEqualStrings("edited a.txt: 3 replacements: edit 1 at line 1; edit 2 at line 1; edit 3 at line 2", chained.text);
    try expectFile(&fixture, "a.txt", "Y\nC\n");
}

test "edit_file takes exactly one form, and at most 16 edits" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "a.txt", .data = "a\n" });
    const cases = [_][]const u8{
        "{\"path\":\"a.txt\",\"old_string\":\"a\",\"new_string\":\"b\",\"edits\":[{\"old_string\":\"a\",\"new_string\":\"b\"}]}",
        "{\"path\":\"a.txt\"}",
        "{\"path\":\"a.txt\",\"old_string\":\"a\"}",
        "{\"path\":\"a.txt\",\"edits\":[]}",
        "{\"path\":\"a.txt\",\"edits\":[{\"old_string\":\"\",\"new_string\":\"b\"}]}",
        "{\"path\":\"a.txt\",\"edits\":[{\"old_string\":1}]}",
    };
    for (cases) |arguments| {
        var result = try run(fixture.ws, alloc, arguments);
        defer result.deinit(alloc);
        try testing.expect(result.is_error);
    }
    var many: std.ArrayList(u8) = .empty;
    defer many.deinit(alloc);
    try many.appendSlice(alloc, "{\"path\":\"a.txt\",\"edits\":[");
    for (0..max_edits + 1) |i| try many.print(alloc, "{s}{{\"old_string\":\"a\",\"new_string\":\"a\"}}", .{if (i > 0) "," else ""});
    try many.appendSlice(alloc, "]}");
    var refused = try run(fixture.ws, alloc, many.items);
    defer refused.deinit(alloc);
    try testing.expectEqualStrings("edit_file: 17 edits; at most 16 in one call", refused.text);
    try expectFile(&fixture, "a.txt", "a\n");
}

test "a batch that repeats an old_string takes its matches in file order" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "a.py", .data = "def f(w):\n    if w < 0:\n        pass\ndef g(w):\n    if w < 0:\n        pass\n" });
    // The first repeat's new_string keeps its old_string, and an edit between moves the second.
    var result = try run(fixture.ws, alloc, "{\"path\":\"a.py\",\"edits\":[{\"old_string\":\"    if w < 0:\",\"new_string\":\"    check(w)\\n    if w < 0:\"},{\"old_string\":\"def g(w):\",\"new_string\":\"def g(v):\\n    w = v\"},{\"old_string\":\"    if w < 0:\",\"new_string\":\"    if w <= 0:\"}]}");
    defer result.deinit(alloc);
    try testing.expect(!result.is_error);
    try testing.expectEqualStrings("edited a.py: 3 replacements: edit 1 at line 2; edit 2 at line 5; edit 3 at line 7", result.text);
    try expectFile(&fixture, "a.py", "def f(w):\n    check(w)\n    if w < 0:\n        pass\ndef g(v):\n    w = v\n    if w <= 0:\n        pass\n");

    var past = try run(fixture.ws, alloc, "{\"path\":\"a.py\",\"edits\":[{\"old_string\":\"pass\",\"new_string\":\"return\"},{\"old_string\":\"pass\",\"new_string\":\"return\"},{\"old_string\":\"pass\",\"new_string\":\"return\"}]}");
    defer past.deinit(alloc);
    try testing.expect(past.is_error);
    try expectContains(past.text, "edit 3 of 3 failed, so none was applied (its line numbers are of the file after the edits before it): old_string repeats an earlier edit's and has no match after that one's in a.py");
}

test "a call's line hints every edit, and an edit's own line wins" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "a.txt", .data = "x\nx\nx\nz\nz\n" });
    var result = try run(fixture.ws, alloc, "{\"path\":\"a.txt\",\"line\":3,\"edits\":[{\"old_string\":\"x\",\"new_string\":\"y\"},{\"old_string\":\"z\",\"new_string\":\"w\",\"line\":5}]}");
    defer result.deinit(alloc);
    try testing.expect(!result.is_error);
    try testing.expectEqualStrings("edited a.txt: 2 replacements: edit 1 at line 3 (the nearest to line 3 of 3 matches); edit 2 at line 5 (the nearest to line 5 of 2 matches)", result.text);
    try expectFile(&fixture, "a.txt", "x\nx\ny\nz\nw\n");
}

test "a miss shows the closest region and what differs" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "c.ts", .data = "import x;\nconst s = `${a}`;\nconst t = 1;\nexport {};\n" });
    // The model escaped `${` as it would inside a template literal.
    var escaped = try run(fixture.ws, alloc, "{\"path\":\"c.ts\",\"old_string\":\"const s = `\\\\${a}`;\\nconst t = 1;\",\"new_string\":\"x\"}");
    defer escaped.deinit(alloc);
    try testing.expect(escaped.is_error);
    try expectContains(escaped.text, "old_string was not found in c.ts; the closest region is lines 2 to 3 (similarity 0.99): backslashes differ at line 2 (your line 1). Those lines are:\nconst s = `${a}`;\nconst t = 1;\n");

    var typo = try run(fixture.ws, alloc, "{\"path\":\"c.ts\",\"old_string\":\"const s = `${a}`;\\nconst t = 2;\",\"new_string\":\"x\"}");
    defer typo.deinit(alloc);
    try expectContains(typo.text, "text differs at line 3 (your line 2)");

    var unlike = try run(fixture.ws, alloc, "{\"path\":\"c.ts\",\"old_string\":\"qqqq zzzz wwww\",\"new_string\":\"x\"}");
    defer unlike.deinit(alloc);
    try testing.expectEqualStrings("edit_file: old_string was not found in c.ts; no similar region; read the file before editing", unlike.text);
    try expectFile(&fixture, "c.ts", "import x;\nconst s = `${a}`;\nconst t = 1;\nexport {};\n");
}

test "the whitespace diagnoses name the line and the counts" {
    const file = [_][]const u8{ "a", "        b\r", "c  " };
    const indent = diagnose(&file, &.{ "a", "    b\r", "c  " }, 10);
    try testing.expectEqual(@as(usize, 11), indent.indentation.line);
    try testing.expectEqualStrings("        ", indent.indentation.file);
    try testing.expectEqual(Diagnosis{ .line_endings = 11 }, diagnose(&file, &.{ "a", "        b", "c  " }, 10));
    try testing.expectEqual(Diagnosis{ .trailing_whitespace = 12 }, diagnose(&file, &.{ "a", "        b\r", "c" }, 10));

    var out: std.ArrayList(u8) = .empty;
    defer out.deinit(testing.allocator);
    try writeDiagnosis(&out, testing.allocator, .{ .indentation = .{ .line = 7, .file = "\t\t  ", .yours = "    " } });
    try testing.expectEqualStrings("indentation differs at line 7 (file: 2 tabs and 2 spaces, yours: 4 spaces)", out.items);
}

test "a miss shows at most 40 lines and 4 KiB of the region, the cut marked" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    var file: std.ArrayList(u8) = .empty;
    defer file.deinit(alloc);
    var old: std.ArrayList(u8) = .empty;
    defer old.deinit(alloc);
    for (1..61) |i| try file.print(alloc, "line {d} holds a value\n", .{i});
    for (11..61) |i| try old.print(alloc, "line {d} holds a valve\\n", .{i});
    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "a.txt", .data = file.items });
    const arguments = try std.fmt.allocPrint(alloc, "{{\"path\":\"a.txt\",\"old_string\":\"{s}\",\"new_string\":\"x\"}}", .{old.items});
    defer alloc.free(arguments);
    var lines = try run(fixture.ws, alloc, arguments);
    defer lines.deinit(alloc);
    try expectContains(lines.text, "the closest region is lines 11 to 60");
    try expectContains(lines.text, "text differs at line 11 (your line 1)");
    try expectContains(lines.text, "line 50 holds a value\n[cut: lines 51 to 60 not shown; read_file offset=51]");
    try testing.expect(std.mem.indexOf(u8, lines.text, "line 51 holds") == null);

    // Long lines: the byte bound cuts first.
    file.clearRetainingCapacity();
    old.clearRetainingCapacity();
    const long: [600]u8 = @splat('k');
    for (0..12) |i| try file.print(alloc, "{d} {s}\n", .{ i, &long });
    for (0..12) |i| try old.print(alloc, "{d} {s}!\\n", .{ i, &long });
    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "b.txt", .data = file.items });
    const wide_arguments = try std.fmt.allocPrint(alloc, "{{\"path\":\"b.txt\",\"old_string\":\"{s}\",\"new_string\":\"x\"}}", .{old.items});
    defer alloc.free(wide_arguments);
    var wide = try run(fixture.ws, alloc, wide_arguments);
    defer wide.deinit(alloc);
    try expectContains(wide.text, "[cut: lines 7 to 12 not shown; read_file offset=7]");
    const region = wide.text[std.mem.indexOf(u8, wide.text, "Those lines are:\n").? + 17 ..];
    try testing.expect(std.mem.indexOf(u8, region, "[cut").? <= region_bytes);
}

test "the closest-region search finds a 40-line region in a 1 MiB file" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    var file: std.ArrayList(u8) = .empty;
    defer file.deinit(alloc);
    var i: usize = 0;
    while (file.items.len < max_bytes - 64) : (i += 1) try file.print(alloc, "    const value_{d} = compute({d}, {d});\n", .{ i, i *% 7919 % 1000, i % 13 });
    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "big.zig", .data = file.items });
    var old: std.ArrayList(u8) = .empty;
    defer old.deinit(alloc);
    for (20_000..20_040) |j| try old.print(alloc, "    const value_{d} = compute({d}, {d}) ;\\n", .{ j, j *% 7919 % 1000, j % 13 });
    const arguments = try std.fmt.allocPrint(alloc, "{{\"path\":\"big.zig\",\"old_string\":\"{s}\",\"new_string\":\"x\"}}", .{old.items});
    defer alloc.free(arguments);
    var result = try run(fixture.ws, alloc, arguments);
    defer result.deinit(alloc);
    try expectContains(result.text, "the closest region is lines 20001 to 20040");
}

test "every profile renders edit_file's schema, the edits array included" {
    const profiles = @import("inference").profiles;
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const user: profiles.Message = .{ .role = .user, .content = "x" };
    // No description, so every `old_string` past the top-level property is
    // the nested items schema: its property and its required list.
    const definition: profiles.ToolDefinition = .{ .name = tool.name, .description = "", .parameters = tool.parameters };
    for (std.enums.values(profiles.Profile)) |profile| {
        const text = try profile.render(arena.allocator(), &.{user}, &.{definition}, .low, .{});
        try testing.expect(std.mem.count(u8, text, "old_string") >= 3);
    }
}

test "overlapping matches count as several at every tier" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "a.txt", .data = "  }\n  }\n  }\n" });
    var lines = try run(fixture.ws, alloc, "{\"path\":\"a.txt\",\"old_string\":\"}\\n}\",\"new_string\":\"x\"}");
    defer lines.deinit(alloc);
    try testing.expect(lines.is_error);
    try expectContains(lines.text, "appears 2 times in a.txt (ignoring indentation)");
    try expectFile(&fixture, "a.txt", "  }\n  }\n  }\n");

    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "b.txt", .data = "aaa" });
    var bytes = try run(fixture.ws, alloc, "{\"path\":\"b.txt\",\"old_string\":\"aa\",\"new_string\":\"b\"}");
    defer bytes.deinit(alloc);
    try expectContains(bytes.text, "appears 2 times in b.txt;");
    try expectFile(&fixture, "b.txt", "aaa");
}

test "match counting stops at its work budget and says so" {
    const alloc = testing.allocator;
    var content: std.ArrayList(u8) = .empty;
    defer content.deinit(alloc);
    for (0..1000) |_| try content.appendSlice(alloc, "x\n");
    var old: std.ArrayList(u8) = .empty;
    defer old.deinit(alloc);
    for (0..100) |_| try old.appendSlice(alloc, "x\n");
    try old.append(alloc, 'y');
    const lines = try OldLines.init(alloc, old.items);
    defer lines.deinit(alloc);
    // About 900 starts of 100 hash comparisons each: past a 10,000 budget.
    const cut = try findLines(alloc, content.items, lines, .indentation, null, 0, 10_000);
    try testing.expect(cut.cut);
    const whole = try findLines(alloc, content.items, lines, .indentation, null, 0, match_max_work);
    try testing.expect(!whole.cut);
    try testing.expectEqual(@as(usize, 0), whole.count);

    try testing.expect(findExact(content.items, "x\nx\n", null, 0, 64).cut);
    try testing.expectEqual(@as(usize, 999), findExact(content.items, "x\nx\n", null, 0, match_max_work).count);
}

test "a similarity search cut short claims no closest region" {
    const alloc = testing.allocator;
    const content = "alpha beta\ngamma delta\n";
    const file = [_]FileLine{ .{ .start = 0, .end = 10 }, .{ .start = 11, .end = 22 } };
    const lines = try OldLines.init(alloc, "alpha beta\ngamma delt0");
    defer lines.deinit(alloc);
    try testing.expectEqual(Closest.cut_short, try closest(alloc, &file, content, lines, 1));
    const whole = try closest(alloc, &file, content, lines, similar_max_cells);
    try testing.expect(whole.found.complete);
}
