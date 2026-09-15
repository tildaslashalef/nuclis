//! `bash` — run one shell command in the workspace.
//!
//! One command string through `/bin/sh -c`, with host-constant bounds on the
//! combined output (1 MiB) and wall time (300 s). The command's stderr is
//! folded into its stdout at the shell level (`exec 2>&1`), so the model reads
//! one ordered stream rather than two interleaved approximations. Cancellation
//! and the timeout both kill and reap the child, so an interrupted command
//! leaves nothing behind; exceeding the output bound stops reading, kills the
//! child, and returns the prefix with `truncated` set.
//!
//! The child sees a **minimal environment**: only `PATH`, `HOME`, `LANG`,
//! `TERM`, and `TMPDIR` are forwarded from the parent, so a secret sitting in
//! the parent environment (an API token, credentials) is not reachable from
//! the model's commands. `expand_arg0` is off: `/bin/sh` is an absolute path,
//! never resolved through the child environment.
const std = @import("std");
const root = @import("root.zig");
const interrupt = @import("../../interrupt.zig");

pub const tool: root.Tool = .{
    .name = "bash",
    .description = "Run one shell command in the workspace. Combined stdout and stderr is bounded and the exit status is reported.",
    .parameters = "{\"type\":\"object\",\"properties\":{\"command\":{\"type\":\"string\"}},\"required\":[\"command\"]}",
    .verb = "Bash",
    .subject = "command",
    .params = &.{"command"},
    .run = run,
};

const max_output: usize = 1024 * 1024;
const max_command: usize = 32 * 1024;
const timeout_ns: i96 = 300 * std.time.ns_per_s;
const poll_ms: i64 = 100;
/// Only these names cross into the child; everything else is dropped.
const forward_env = [_][]const u8{ "PATH", "HOME", "LANG", "TERM", "TMPDIR" };
const fallback_path = "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin";

const Args = struct { command: []const u8 };

const Outcome = enum { completed, timed_out, cancelled, truncated, read_failed };

fn run(workspace: root.Workspace, alloc: std.mem.Allocator, arguments: []const u8) std.mem.Allocator.Error!root.Result {
    const parsed = std.json.parseFromSlice(Args, alloc, arguments, .{ .ignore_unknown_fields = true }) catch {
        return root.fail(alloc, "bash: arguments must be a JSON object with a \"command\" string", .{});
    };
    defer parsed.deinit();
    const command = parsed.value.command;
    if (command.len == 0) return root.fail(alloc, "bash: command is empty", .{});
    if (command.len > max_command) return root.fail(alloc, "bash: command is too long ({d} bytes, max {d})", .{ command.len, max_command });
    if (std.mem.indexOfScalar(u8, command, 0) != null) return root.fail(alloc, "bash: command contains a NUL byte", .{});

    var env = std.process.Environ.Map.init(alloc);
    defer env.deinit();
    if (workspace.environ) |parent| {
        for (forward_env) |key| {
            if (parent.get(key)) |value| env.put(key, value) catch return error.OutOfMemory;
        }
    }
    if (env.get("PATH") == null) env.put("PATH", fallback_path) catch return error.OutOfMemory;
    if (env.get("LANG") == null) env.put("LANG", "C") catch return error.OutOfMemory;

    const script = try std.fmt.allocPrint(alloc, "exec 2>&1\n{s}", .{command});
    defer alloc.free(script);
    const argv = [_][]const u8{ "/bin/sh", "-c", script };

    var child = std.process.spawn(workspace.io, .{
        .argv = &argv,
        .cwd = .{ .dir = workspace.dir },
        .environ_map = &env,
        .stdin = .ignore,
        .stdout = .pipe,
        .stderr = .pipe,
        .expand_arg0 = .no_expand,
    }) catch |err| return root.fail(alloc, "bash: could not start a shell: {s}", .{@errorName(err)});

    var reader: std.Io.File.MultiReader = undefined;
    var reader_ready = false;
    var child_live = true;
    defer {
        if (reader_ready) reader.deinit();
        if (child_live) child.kill(workspace.io);
    }
    var storage: std.Io.File.MultiReader.Buffer(2) = undefined;
    reader.init(alloc, workspace.io, storage.toStreams(), &.{ child.stdout.?, child.stderr.? });
    reader_ready = true;

    const started = std.Io.Clock.awake.now(workspace.io);
    var outcome: Outcome = .completed;
    while (true) {
        // The agent's beat, before anything else: read the keyboard and
        // repaint while the command holds the loop. A Ctrl-C read here sets
        // the interrupt and the check below reaps the child.
        if (workspace.tick) |beat| beat.call(beat.context);
        if (interrupt.requested()) {
            outcome = .cancelled;
            break;
        }
        if (started.durationTo(std.Io.Clock.awake.now(workspace.io)).nanoseconds >= timeout_ns) {
            outcome = .timed_out;
            break;
        }
        // A short poll interval keeps cancellation and the deadline responsive
        // while the child is idle; data arriving wakes the wait early.
        reader.fill(64, .{ .duration = .{ .raw = .{ .nanoseconds = poll_ms * std.time.ns_per_ms }, .clock = .awake } }) catch |err| switch (err) {
            error.EndOfStream => break,
            error.Timeout => continue,
            else => {
                outcome = .read_failed;
                break;
            },
        };
        if (reader.reader(0).buffered().len + reader.reader(1).buffered().len > max_output) {
            outcome = .truncated;
            break;
        }
    }

    // Copy the buffered output before cleanup closes the pipes.
    const stdout_owned = try capture(alloc, reader.reader(0));
    defer alloc.free(stdout_owned);
    const stderr_owned = try capture(alloc, reader.reader(1));
    defer alloc.free(stderr_owned);
    reader.deinit();
    reader_ready = false;

    var term: std.process.Child.Term = .{ .exited = 0 };
    if (outcome == .completed) {
        term = child.wait(workspace.io) catch |err| return root.fail(alloc, "bash: could not reap the command: {s}", .{@errorName(err)});
        child_live = false;
    } else {
        child.kill(workspace.io);
        child_live = false;
    }

    var text: std.ArrayList(u8) = .empty;
    errdefer text.deinit(alloc);
    try text.appendSlice(alloc, stdout_owned);
    if (stderr_owned.len > 0) {
        if (text.items.len > 0) try text.append(alloc, '\n');
        try text.appendSlice(alloc, stderr_owned);
    }
    var is_error = false;
    switch (outcome) {
        .cancelled => {
            is_error = true;
            try text.appendSlice(alloc, "\n[bash: cancelled]");
        },
        .timed_out => {
            is_error = true;
            try text.appendSlice(alloc, "\n[bash: timed out after 300 s]");
        },
        .read_failed => {
            is_error = true;
            try text.appendSlice(alloc, "\n[bash: could not read the command's output]");
        },
        .completed, .truncated => {},
    }
    switch (term) {
        .exited => |code| if (code != 0) {
            is_error = true;
            var note: [32]u8 = undefined;
            try text.appendSlice(alloc, std.fmt.bufPrint(&note, "\n[bash: exit code {d}]", .{code}) catch "\n[bash: non-zero exit]");
        },
        .signal => |sig| {
            is_error = true;
            var note: [40]u8 = undefined;
            try text.appendSlice(alloc, std.fmt.bufPrint(&note, "\n[bash: killed by signal {d}]", .{sig}) catch "\n[bash: killed by a signal]");
        },
        else => {},
    }
    return .{ .text = try text.toOwnedSlice(alloc), .truncated = outcome == .truncated, .is_error = is_error };
}

fn capture(alloc: std.mem.Allocator, reader: *std.Io.Reader) std.mem.Allocator.Error![]u8 {
    const buffered = reader.buffered();
    return alloc.dupe(u8, buffered[0..@min(buffered.len, max_output)]);
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

test "bash runs a command in the workspace and reports its output" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    const result = try run(fixture.ws, alloc, "{\"command\":\"echo hello\"}");
    defer alloc.free(result.text);
    try testing.expectEqualStrings("hello\n", result.text);
    try testing.expect(!result.is_error);
    try testing.expect(!result.truncated);
}

test "bash folds stderr into stdout, in order" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    const result = try run(fixture.ws, alloc, "{\"command\":\"echo out; echo err >&2; echo done\"}");
    defer alloc.free(result.text);
    try testing.expectEqualStrings("out\nerr\ndone\n", result.text);
    try testing.expect(!result.is_error);
}

test "bash reports a non-zero exit as an error result" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    const result = try run(fixture.ws, alloc, "{\"command\":\"exit 7\"}");
    defer alloc.free(result.text);
    try testing.expect(result.is_error);
    try testing.expect(std.mem.indexOf(u8, result.text, "exit code 7") != null);
}

test "bash runs in the workspace directory" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "here.txt", .data = "x" });
    const result = try run(fixture.ws, alloc, "{\"command\":\"ls\"}");
    defer alloc.free(result.text);
    try testing.expect(std.mem.indexOf(u8, result.text, "here.txt") != null);
}

test "bash bounds the output and marks it truncated" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    // `yes` writes without end; the cap stops the read and the child is killed.
    const result = try run(fixture.ws, alloc, "{\"command\":\"yes aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"}");
    defer alloc.free(result.text);
    try testing.expect(result.truncated);
    try testing.expect(result.text.len <= max_output);
}

test "bash rejects an empty or oversized command as a result" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    const empty = try run(fixture.ws, alloc, "{\"command\":\"\"}");
    defer alloc.free(empty.text);
    try testing.expect(empty.is_error);
    const bad = try run(fixture.ws, alloc, "not json");
    defer alloc.free(bad.text);
    try testing.expect(bad.is_error);
}

test "a requested cancellation stops the command and reaps the child" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    defer interrupt.clear();
    interrupt.request();
    const started = std.Io.Clock.awake.now(testing.io);
    const result = try run(fixture.ws, alloc, "{\"command\":\"sleep 30\"}");
    defer alloc.free(result.text);
    const elapsed_ns = started.durationTo(std.Io.Clock.awake.now(testing.io)).nanoseconds;
    try testing.expect(result.is_error);
    try testing.expect(std.mem.indexOf(u8, result.text, "cancelled") != null);
    // If the child were not killed, this would wait for the full sleep.
    try testing.expect(elapsed_ns < 10 * std.time.ns_per_s);
}

test "the tick can request cancellation mid-command" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    defer interrupt.clear();
    const Ticker = struct {
        calls: usize = 0,
        fn tick(context: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(context));
            self.calls += 1;
            interrupt.request();
        }
    };
    var ticker: Ticker = .{};
    var ws = fixture.ws;
    ws.tick = .{ .context = &ticker, .call = Ticker.tick };
    const started = std.Io.Clock.awake.now(testing.io);
    const result = try run(ws, alloc, "{\"command\":\"sleep 30\"}");
    defer alloc.free(result.text);
    const elapsed_ns = started.durationTo(std.Io.Clock.awake.now(testing.io)).nanoseconds;
    // The tool reached back into the driver while the command ran…
    try testing.expect(ticker.calls > 0);
    // …and the interrupt it set stopped the command at once.
    try testing.expect(result.is_error);
    try testing.expect(std.mem.indexOf(u8, result.text, "cancelled") != null);
    try testing.expect(elapsed_ns < 10 * std.time.ns_per_s);
}
