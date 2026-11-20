//! `bash` — run one shell command in the workspace.
//!
//! One command string through `/bin/sh -c`, with host-constant bounds on the
//! combined output (1 MiB) and wall time (300 s). The command's stderr is
//! folded into its stdout at the shell level (`exec 2>&1`), so the model reads
//! one ordered stream rather than two interleaved approximations. The shell
//! leads its own process group: cancellation, the timeout, and the output
//! bound end the whole group (`stop`: TERM, a grace, then KILL) and reap the
//! shell, so a command that ignores TERM, a pipeline, or a `&` job cannot
//! outlive them. Exceeding the output bound returns the prefix with
//! `truncated` set. A command quiet 15 s with a process of its group
//! listening on TCP is a server in the foreground and is stopped the same
//! way (`listening`). An output longer than
//! `head_lines + tail_lines` reaches the model as its head and tail with the
//! omitted count marked between them.
//!
//! The child sees a **minimal environment**: only `PATH`, `HOME`, `LANG`,
//! `TERM`, and `TMPDIR` are forwarded from the parent, so a secret sitting in
//! the parent environment (an API token, credentials) is not reachable from
//! the model's commands. `expand_arg0` is off: `/bin/sh` is an absolute path,
//! never resolved through the child environment.
const std = @import("std");
const builtin = @import("builtin");
const root = @import("root.zig");
const interrupt = @import("../../interrupt.zig");

pub const tool: root.Tool = .{
    .name = "bash",
    .description = "Run one shell command in the workspace. Combined stdout and stderr is bounded and the exit status is reported.",
    .parameters = "{\"type\":\"object\",\"properties\":{\"command\":{\"type\":\"string\"}},\"required\":[\"command\"]}",
    .label = "Running command",
    .display = "Bash",
    .subject = "command",
    .detail_prefix = "$ ",
    .run = run,
};

const max_output: usize = 1024 * 1024;
/// Lines of a long output the model reads: its head and, since a failure
/// reports at the end, a longer tail, with the count omitted between them.
/// The loop's context cut keeps the same proportion (`Agent.fit`).
pub const head_lines: usize = 40;
pub const tail_lines: usize = 80;
/// What every note the tool writes starts with.
pub const note_prefix = "[bash: ";
/// How the elision marker, `note_prefix` and a count, ends.
pub const omitted_suffix = " lines omitted]";
const max_command: usize = 32 * 1024;
const timeout_ns: i96 = 300 * std.time.ns_per_s;
const poll_ms: i64 = 100;
/// How long a stopped command's group has to exit on TERM before KILL.
const grace_ms: i64 = 2000;
/// Only these names cross into the child; everything else is dropped.
const forward_env = [_][]const u8{ "PATH", "HOME", "LANG", "TERM", "TMPDIR" };
const fallback_path = "/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin";

const Args = struct { command: []const u8 };

/// How long the output is still read after the shell exits while a process
/// it left in the background holds the output open.
const drain_ms: i64 = 1000;

/// A command still running with its output quiet this long is checked
/// for a server: a process of its group listening on TCP (`listening`),
/// every `server_check_ms` after.
const server_quiet_ms: i64 = if (builtin.is_test) 300 else 15_000;
const server_check_ms: i64 = if (builtin.is_test) 200 else 5_000;
/// The most of `lsof`'s output read, and how long it may take.
const lsof_max: usize = 4096;
const lsof_timeout_ms: i64 = 2000;

const Outcome = enum { completed, timed_out, cancelled, truncated, read_failed, left_running, serving };

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

    const script = try alloc.print("exec 2>&1\n{s}", .{command});
    defer alloc.free(script);
    const argv = [_][]const u8{ "/bin/sh", "-c", script };

    var child = std.process.spawn(workspace.io, .{
        .argv = &argv,
        // The agent's workspace is the process's own directory (`Dir.cwd()`,
        // `AT_FDCWD`), which Darwin's `posix_spawn_file_actions_addfchdir_np`
        // refuses as a bad descriptor: the child inherits it instead.
        .cwd = if (workspace.dir.handle == std.Io.Dir.cwd().handle) .inherit else .{ .dir = workspace.dir },
        .environ_map = &env,
        .stdin = .ignore,
        .stdout = .pipe,
        .stderr = .pipe,
        .expand_arg0 = .no_expand,
        // Its own group, so `stop` reaches every process the command starts.
        .pgid = 0,
    }) catch |err| return root.fail(alloc, "bash: could not start a shell: {s}", .{@errorName(err)});

    var reader: std.Io.File.MultiReader = undefined;
    var reader_ready = false;
    var child_live = true;
    defer {
        if (reader_ready) reader.deinit();
        if (child_live) stop(&child, workspace.io);
    }
    var storage: std.Io.File.MultiReader.Buffer(2) = undefined;
    reader.init(alloc, workspace.io, storage.toStreams(), &.{ child.stdout.?, child.stderr.? }); // checked: both spawned as pipes
    reader_ready = true;

    const started = std.Io.Clock.awake.now(workspace.io);
    var outcome: Outcome = .completed;
    // When the shell exited with the output still open: a `&` job holds it.
    var shell_gone: ?std.Io.Timestamp = null;
    // The output's length when it last grew, and when; the last server check.
    var heard: usize = 0;
    var heard_at = started;
    var checked_at: ?std.Io.Timestamp = null;
    var ports: Ports = .{};
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
            error.Timeout => {},
            else => {
                outcome = .read_failed;
                break;
            },
        };
        if (reader.reader(0).buffered().len + reader.reader(1).buffered().len > max_output) {
            outcome = .truncated;
            break;
        }
        // Checked on every pass: a job that writes often never lets a read
        // time out.
        if (shell_gone) |gone| {
            if (gone.durationTo(std.Io.Clock.awake.now(workspace.io)).nanoseconds >= drain_ms * std.time.ns_per_ms) {
                outcome = .left_running;
                break;
            }
        } else if (exited(child.id.?)) { // checked: spawned
            shell_gone = std.Io.Clock.awake.now(workspace.io);
        } else {
            const now = std.Io.Clock.awake.now(workspace.io);
            const size = reader.reader(0).buffered().len + reader.reader(1).buffered().len;
            if (size != heard) {
                heard = size;
                heard_at = now;
            } else if (heard_at.durationTo(now).nanoseconds >= server_quiet_ms * std.time.ns_per_ms and
                (if (checked_at) |at| at.durationTo(now).nanoseconds >= server_check_ms * std.time.ns_per_ms else true))
            {
                checked_at = now;
                if (listening(alloc, workspace.io, child.id.?, &ports)) { // checked: spawned
                    outcome = .serving;
                    break;
                }
            }
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
    var stopped_left = false;
    if (outcome == .left_running) {
        // The shell is done: end what it left in its group while the
        // unreaped shell still holds the group's id, then reap it.
        stopped_left = endGroup(child.id.?, workspace.io); // checked: spawned
        term = child.wait(workspace.io) catch |err| return root.fail(alloc, "bash: could not reap the command: {s}", .{@errorName(err)});
        child_live = false;
    } else if (outcome == .completed) {
        term = child.wait(workspace.io) catch |err| return root.fail(alloc, "bash: could not reap the command: {s}", .{@errorName(err)});
        child_live = false;
    } else {
        stop(&child, workspace.io);
        child_live = false;
    }

    var text: std.ArrayList(u8) = .empty;
    errdefer text.deinit(alloc);
    try text.appendSlice(alloc, stdout_owned);
    if (stderr_owned.len > 0) {
        if (text.items.len > 0) try text.append(alloc, '\n');
        try text.appendSlice(alloc, stderr_owned);
    }
    const lines = countLines(text.items);
    try elide(alloc, &text, lines);
    // The detail row says only what went wrong; a clean run shows nothing
    // beyond its command.
    var summary: std.ArrayList(u8) = .empty;
    errdefer summary.deinit(alloc);
    var is_error = false;
    switch (outcome) {
        // The loop marks the call cancelled and the row says so with its
        // time; the summary keeps only what the command had printed.
        .cancelled => {
            is_error = true;
            try text.appendSlice(alloc, "\n[bash: cancelled]");
            if (lines > 0) try summary.print(alloc, "{d} line{s} of output", .{ lines, if (lines == 1) "" else "s" });
        },
        .timed_out => {
            is_error = true;
            try text.appendSlice(alloc, "\n[bash: timed out after 300 s; a command that does not exit, such as a server, runs in the background: `cmd > cmd.log 2>&1 & sleep 2; …; kill $!`]");
            try summary.appendSlice(alloc, "timed out after 300 s");
        },
        .read_failed => {
            is_error = true;
            try text.appendSlice(alloc, "\n[bash: could not read the command's output]");
            try summary.appendSlice(alloc, "could not read the output");
        },
        .left_running => {
            try text.print(alloc, "\n[bash: the command exited, but a process it started in the background still held its output{s}; give a background process its own output (`cmd > cmd.log 2>&1 &`) and stop it with `kill $!`]", .{if (stopped_left) ", and was stopped" else " outside the command's process group, and could not be stopped"});
            try summary.appendSlice(alloc, if (stopped_left) "a background process was stopped" else "a background process holds the output");
        },
        .serving => {
            is_error = true;
            const seconds = @divFloor(started.durationTo(std.Io.Clock.awake.now(workspace.io)).nanoseconds, std.time.ns_per_s);
            try text.print(alloc, "\n[bash: the command was still running after {d} s, listening on {s} with no output for {d} s, so it was stopped as a server in the foreground; run a server in the background: `cmd > cmd.log 2>&1 & sleep 2; …; kill $!`]", .{ seconds, ports.slice(), @divFloor(server_quiet_ms, 1000) });
            try summary.print(alloc, "a server in the foreground, stopped after {d} s", .{seconds});
        },
        .truncated => {
            try text.print(alloc, "\n[bash: output stopped at {d} bytes and the command was killed]", .{max_output});
            try summary.print(alloc, "output truncated at {d} bytes", .{max_output});
        },
        .completed => {},
    }
    switch (term) {
        .exited => |code| if (code != 0) {
            is_error = true;
            var note: [32]u8 = undefined;
            try text.appendSlice(alloc, std.fmt.bufPrint(&note, "\n[bash: exit code {d}]", .{code}) catch "\n[bash: non-zero exit]");
            try summary.print(alloc, "exit {d}", .{code});
        },
        .signal => |sig| {
            is_error = true;
            var note: [40]u8 = undefined;
            try text.appendSlice(alloc, std.fmt.bufPrint(&note, "\n[bash: killed by signal {d}]", .{sig}) catch "\n[bash: killed by a signal]");
            try summary.print(alloc, "killed by signal {d}", .{sig});
        },
        else => {},
    }
    // Blank output from a clean run reads to a model like a broken tool, not
    // like its own script matching nothing; say which it was.
    const clean = outcome == .completed and term == .exited and term.exited == 0;
    if (clean and std.mem.trim(u8, text.items, " \t\r\n").len == 0) {
        try text.appendSlice(alloc, "[bash: no output, exit 0]");
        try summary.appendSlice(alloc, "no output");
    } else if (summary.items.len > 0 and outcome != .cancelled) try summary.print(alloc, " · {d} line{s}", .{ lines, if (lines == 1) "" else "s" });
    const owned_summary: ?[]u8 = if (summary.items.len > 0) try summary.toOwnedSlice(alloc) else null;
    errdefer if (owned_summary) |s| alloc.free(s);
    return .{ .text = try text.toOwnedSlice(alloc), .truncated = outcome == .truncated, .is_error = is_error, .summary = owned_summary };
}

/// Ends the command's process group: TERM to all of it, up to `grace_ms`
/// for the shell to exit, KILL to whatever is left, then the shell's reap.
fn stop(child: *std.process.Child, io: std.Io) void {
    const pid = child.id orelse return;
    std.posix.kill(-pid, .TERM) catch {};
    // The grace is read off the clock: a loaded machine stretches each sleep.
    const termed = std.Io.Clock.awake.now(io);
    while (!exited(pid) and termed.durationTo(std.Io.Clock.awake.now(io)).nanoseconds < grace_ms * std.time.ns_per_ms) {
        io.sleep(.fromMilliseconds(10), .awake) catch break;
    }
    std.posix.kill(-pid, .KILL) catch {};
    _ = child.wait(io) catch {};
}

/// Ends what an exited, unreaped shell left in its group: TERM, a grace,
/// then KILL. Whether anything was signalled; the shell's zombie counts as
/// a member, so the grace is spent whole.
fn endGroup(pid: std.posix.pid_t, io: std.Io) bool {
    std.posix.kill(-pid, .TERM) catch return false;
    io.sleep(.fromMilliseconds(grace_ms), .awake) catch {};
    std.posix.kill(-pid, .KILL) catch {};
    return true;
}

/// The addresses `listening` found, as `lsof` names them (`*:3000`).
const Ports = struct {
    buf: [128]u8 = undefined,
    len: usize = 0,

    fn slice(self: *const Ports) []const u8 {
        return if (self.len == 0) "a port" else self.buf[0..self.len];
    }
};

/// Whether a process in group `pgid` listens on TCP, asked of `lsof`
/// (bounded in time and output); false when it cannot tell, as where
/// `lsof` is missing. Fills `ports` with the addresses.
fn listening(alloc: std.mem.Allocator, io: std.Io, pgid: std.posix.pid_t, ports: *Ports) bool {
    var number: [24]u8 = undefined;
    const group = std.fmt.bufPrint(&number, "{d}", .{pgid}) catch return false;
    const result = std.process.run(alloc, io, .{
        .argv = &.{ "lsof", "-nP", "-b", "-w", "-a", "-g", group, "-iTCP", "-sTCP:LISTEN", "-Fn" },
        .stdout_limit = .limited(lsof_max),
        .stderr_limit = .limited(lsof_max),
        .timeout = .{ .duration = .{ .raw = .{ .nanoseconds = lsof_timeout_ms * std.time.ns_per_ms }, .clock = .awake } },
    }) catch return false;
    defer alloc.free(result.stdout);
    defer alloc.free(result.stderr);
    if (result.term != .exited or result.term.exited != 0) return false;
    ports.len = 0;
    var lines = std.mem.splitScalar(u8, result.stdout, '\n');
    var found = false;
    while (lines.next()) |line| {
        if (line.len < 2 or line[0] != 'n') continue;
        found = true;
        const name = line[1..];
        const sep: []const u8 = if (ports.len > 0) ", " else "";
        if (ports.len + sep.len + name.len > ports.buf.len) continue;
        if (std.mem.indexOf(u8, ports.buf[0..ports.len], name) != null) continue;
        @memcpy(ports.buf[ports.len..][0..sep.len], sep);
        @memcpy(ports.buf[ports.len + sep.len ..][0..name.len], name);
        ports.len += sep.len + name.len;
    }
    return found;
}

extern "c" fn waitid(idtype: c_int, id: c_uint, info: *std.c.siginfo_t, options: c_int) c_int;

/// Whether `pid` has exited, without reaping it (`WNOWAIT`): `Child.wait`
/// still reaps it and frees its pipes.
fn exited(pid: std.posix.pid_t) bool {
    const p_pid: c_int = 1; // P_PID on Linux and Darwin
    var info = std.mem.zeroes(std.c.siginfo_t);
    // <sys/wait.h>'s values, which std.c names only for Linux.
    const nowait: c_int = if (builtin.os.tag.isDarwin()) 0x20 else 0x01000000;
    const options: c_int = 4 | 1 | nowait; // WEXITED | WNOHANG | WNOWAIT
    if (waitid(p_pid, @intCast(pid), &info, options) != 0) return true; // checked: a child's pid, positive
    // With WNOHANG and nothing to report the kernel fills nothing; `si_signo` leads on both.
    const signo: *const c_int = @ptrCast(@alignCast(&info));
    return signo.* == @backingInt(std.posix.SIG.CHLD);
}

/// Keeps the first `head_lines` and last `tail_lines` of an output longer
/// than both, with a marked count of the lines between.
fn elide(alloc: std.mem.Allocator, text: *std.ArrayList(u8), lines: usize) std.mem.Allocator.Error!void {
    if (lines <= head_lines + tail_lines) return;
    var head_end: usize = 0;
    for (0..head_lines) |_| head_end = std.mem.indexOfScalarPos(u8, text.items, head_end, '\n').? + 1; // checked: `lines` counts this text's lines, more than head and tail together
    // The tail starts after the newline that ends line `lines - tail_lines`.
    var tail_start = text.items.len;
    if (text.items[tail_start - 1] == '\n') tail_start -= 1;
    for (0..tail_lines) |_| tail_start = std.mem.lastIndexOfScalar(u8, text.items[0..tail_start], '\n').?; // checked: `lines` counts this text's lines, more than head and tail together
    tail_start += 1;
    var note: [48]u8 = undefined;
    const marker = std.fmt.bufPrint(&note, note_prefix ++ "{d}" ++ omitted_suffix ++ "\n", .{lines - head_lines - tail_lines}) catch unreachable; // checked: 48 bytes hold the note for any count
    try text.replaceRange(alloc, head_end, tail_start - head_end, marker);
}

fn countLines(text: []const u8) usize {
    if (text.len == 0) return 0;
    const trailing: usize = if (text[text.len - 1] == '\n') 1 else 0;
    return std.mem.count(u8, text, "\n") + 1 - trailing;
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
    var result = try run(fixture.ws, alloc, "{\"command\":\"echo hello\"}");
    defer result.deinit(alloc);
    try testing.expectEqualStrings("hello\n", result.text);
    try testing.expect(!result.is_error);
    try testing.expect(!result.truncated);
    // A clean run has nothing to add under its command.
    try testing.expect(result.summary == null);
}

test "bash runs from the process's own directory, the agent's workspace" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    var ws = fixture.ws;
    ws.dir = .cwd();
    var result = try run(ws, alloc, "{\"command\":\"echo here\"}");
    defer result.deinit(alloc);
    try testing.expectEqualStrings("here\n", result.text);
    try testing.expect(!result.is_error);
}

test "bash folds stderr into stdout, in order" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    var result = try run(fixture.ws, alloc, "{\"command\":\"echo out; echo err >&2; echo done\"}");
    defer result.deinit(alloc);
    try testing.expectEqualStrings("out\nerr\ndone\n", result.text);
    try testing.expect(!result.is_error);
}

test "bash reports a non-zero exit as an error result" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    var result = try run(fixture.ws, alloc, "{\"command\":\"exit 7\"}");
    defer result.deinit(alloc);
    try testing.expect(result.is_error);
    try testing.expect(std.mem.indexOf(u8, result.text, "exit code 7") != null);
    try testing.expectEqualStrings("exit 7 · 0 lines", result.summary.?);
}

test "bash says when a clean run printed nothing but blank lines" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    var result = try run(fixture.ws, alloc, "{\"command\":\"printf '\\\\n\\\\n'\"}");
    defer result.deinit(alloc);
    try testing.expectEqualStrings("\n\n[bash: no output, exit 0]", result.text);
    try testing.expect(!result.is_error);
    try testing.expectEqualStrings("no output", result.summary.?);

    // A failing command with no output keeps its exit note alone.
    var failed = try run(fixture.ws, alloc, "{\"command\":\"false\"}");
    defer failed.deinit(alloc);
    try testing.expect(std.mem.indexOf(u8, failed.text, "no output") == null);
    try testing.expectEqualStrings("exit 1 · 0 lines", failed.summary.?);
}

test "bash runs in the workspace directory" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    try fixture.tmp.dir.writeFile(testing.io, .{ .sub_path = "here.txt", .data = "x" });
    var result = try run(fixture.ws, alloc, "{\"command\":\"ls\"}");
    defer result.deinit(alloc);
    try testing.expect(std.mem.indexOf(u8, result.text, "here.txt") != null);
}

test "bash bounds the output and marks it truncated" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    // `yes` writes without end; the cap stops the read and the child is killed.
    var result = try run(fixture.ws, alloc, "{\"command\":\"yes aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\"}");
    defer result.deinit(alloc);
    try testing.expect(result.truncated);
    try testing.expect(result.text.len <= max_output);
    try testing.expect(std.mem.startsWith(u8, result.summary.?, "output truncated at "));
}

test "a long output keeps its head and its tail with the omitted count between" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    var result = try run(fixture.ws, alloc, "{\"command\":\"seq 1 500\"}");
    defer result.deinit(alloc);
    try testing.expect(std.mem.startsWith(u8, result.text, "1\n2\n"));
    try testing.expect(std.mem.indexOf(u8, result.text, "\n40\n[bash: 380 lines omitted]\n421\n") != null);
    try testing.expect(std.mem.endsWith(u8, result.text, "\n500\n"));

    // At the bound nothing is omitted.
    var whole = try run(fixture.ws, alloc, "{\"command\":\"seq 1 120\"}");
    defer whole.deinit(alloc);
    try testing.expect(std.mem.indexOf(u8, whole.text, "omitted") == null);
}

test "bash rejects an empty or oversized command as a result" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    var empty = try run(fixture.ws, alloc, "{\"command\":\"\"}");
    defer empty.deinit(alloc);
    try testing.expect(empty.is_error);
    var bad = try run(fixture.ws, alloc, "not json");
    defer bad.deinit(alloc);
    try testing.expect(bad.is_error);
}

test "a requested cancellation stops the command and reaps the child" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    defer interrupt.clear();
    interrupt.request();
    const started = std.Io.Clock.awake.now(testing.io);
    var result = try run(fixture.ws, alloc, "{\"command\":\"sleep 30\"}");
    defer result.deinit(alloc);
    const elapsed_ns = started.durationTo(std.Io.Clock.awake.now(testing.io)).nanoseconds;
    try testing.expect(result.is_error);
    try testing.expect(std.mem.indexOf(u8, result.text, "cancelled") != null);
    // If the child were not killed, this would wait for the full sleep.
    try testing.expect(elapsed_ns < 10 * std.time.ns_per_s);
}

test "a cancelled command's text ends in the note; its summary counts only the output" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    defer interrupt.clear();
    interrupt.request();
    var silent = try run(fixture.ws, alloc, "{\"command\":\"sleep 30\"}");
    defer silent.deinit(alloc);
    try testing.expectEqualStrings("\n[bash: cancelled]", silent.text);
    try testing.expect(silent.summary == null);
    interrupt.clear();

    // Cancels a beat after the command has printed, so the wait in between
    // has read the pipe, as a Ctrl-C after its output would.
    const Ticker = struct {
        dir: std.Io.Dir,
        seen: bool = false,
        fn tick(context: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(context));
            if (self.seen) return interrupt.request();
            _ = self.dir.statFile(testing.io, "printed", .{}) catch return;
            self.seen = true;
        }
    };
    var ticker: Ticker = .{ .dir = fixture.tmp.dir };
    var ws = fixture.ws;
    ws.tick = .{ .context = &ticker, .call = Ticker.tick };
    var printed = try run(ws, alloc, "{\"command\":\"echo one; echo two; touch printed; sleep 30\"}");
    defer printed.deinit(alloc);
    try testing.expect(printed.is_error);
    try testing.expectEqualStrings("one\ntwo\n\n[bash: cancelled]", printed.text);
    try testing.expectEqualStrings("2 lines of output", printed.summary.?);
}

test "a cancelled command that ignores TERM, and its background job, are gone within the grace" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    defer interrupt.clear();
    // Cancels once the background job's pid is on disk, so the job exists,
    // and notes when: the grace runs from the cancel, not from the shell's
    // start, which a loaded machine can delay by seconds.
    const Ticker = struct {
        dir: std.Io.Dir,
        cancelled: ?std.Io.Timestamp = null,
        fn tick(context: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(context));
            if (self.cancelled != null) return;
            const st = self.dir.statFile(testing.io, "bg.pid", .{}) catch return;
            if (st.size == 0) return;
            self.cancelled = std.Io.Clock.awake.now(testing.io);
            interrupt.request();
        }
    };
    var ticker: Ticker = .{ .dir = fixture.tmp.dir };
    var ws = fixture.ws;
    ws.tick = .{ .context = &ticker, .call = Ticker.tick };
    var result = try run(ws, alloc, "{\"command\":\"trap '' TERM; sleep 100 & echo $! > bg.pid; sleep 1000\"}");
    defer result.deinit(alloc);
    try testing.expect(std.mem.indexOf(u8, result.text, "cancelled") != null);
    const cancelled = ticker.cancelled orelse return error.TestUnexpectedResult;
    const elapsed_ns = cancelled.durationTo(std.Io.Clock.awake.now(testing.io)).nanoseconds;
    try testing.expect(elapsed_ns < (grace_ms + 3000) * std.time.ns_per_ms);
    const text = try fixture.tmp.dir.readFileAlloc(testing.io, "bg.pid", alloc, .limited(32));
    defer alloc.free(text);
    const job = try std.fmt.parseInt(std.posix.pid_t, std.mem.trim(u8, text, " \n"), 10);
    // KILL is delivered at once; its reap by init may lag, so wait for it, bounded.
    var gone = false;
    for (0..200) |_| {
        std.posix.kill(job, @fromBackingInt(@intCast(0))) catch {
            gone = true;
            break;
        };
        testing.io.sleep(.fromMilliseconds(10), .awake) catch {};
    }
    try testing.expect(gone);
}

test "a background job left holding the output is stopped once the shell exits, and the run stands" {
    const alloc = testing.allocator;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    const started = std.Io.Clock.awake.now(testing.io);
    var result = try run(fixture.ws, alloc, "{\"command\":\"(sleep 100; echo late) & echo $! > bg.pid; echo done\"}");
    defer result.deinit(alloc);
    const elapsed_ns = started.durationTo(std.Io.Clock.awake.now(testing.io)).nanoseconds;
    try testing.expect(elapsed_ns < (drain_ms + grace_ms + 3000) * std.time.ns_per_ms);
    try testing.expect(std.mem.startsWith(u8, result.text, "done\n"));
    try testing.expect(std.mem.indexOf(u8, result.text, "still held its output, and was stopped") != null);
    try testing.expect(!result.is_error);
    const text = try fixture.tmp.dir.readFileAlloc(testing.io, "bg.pid", alloc, .limited(32));
    defer alloc.free(text);
    const job = try std.fmt.parseInt(std.posix.pid_t, std.mem.trim(u8, text, " \n"), 10);
    var gone = false;
    for (0..200) |_| {
        std.posix.kill(job, @fromBackingInt(@intCast(0))) catch {
            gone = true;
            break;
        };
        testing.io.sleep(.fromMilliseconds(10), .awake) catch {};
    }
    try testing.expect(gone);
}

test "a quiet command listening on TCP is stopped as a server in the foreground" {
    const alloc = testing.allocator;
    const tools = std.process.run(alloc, testing.io, .{ .argv = &.{ "/bin/sh", "-c", "command -v lsof && command -v python3" } }) catch return error.SkipZigTest;
    defer alloc.free(tools.stdout);
    defer alloc.free(tools.stderr);
    if (tools.term != .exited or tools.term.exited != 0) return error.SkipZigTest;
    var fixture = try Fixture.init(alloc);
    defer fixture.deinit(alloc);
    const started = std.Io.Clock.awake.now(testing.io);
    var result = try run(fixture.ws, alloc, "{\"command\":\"echo up; python3 -c \\\"import socket, time; s = socket.socket(); s.bind(('127.0.0.1', 0)); s.listen(); time.sleep(60)\\\"\"}");
    defer result.deinit(alloc);
    const elapsed_ns = started.durationTo(std.Io.Clock.awake.now(testing.io)).nanoseconds;
    try testing.expect(elapsed_ns < 20 * std.time.ns_per_s);
    try testing.expect(result.is_error);
    try testing.expect(std.mem.startsWith(u8, result.text, "up\n"));
    try testing.expect(std.mem.indexOf(u8, result.text, "listening on 127.0.0.1:") != null);
    try testing.expect(std.mem.indexOf(u8, result.text, "so it was stopped as a server in the foreground") != null);
    try testing.expect(std.mem.startsWith(u8, result.summary.?, "a server in the foreground, stopped after "));
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
    var result = try run(ws, alloc, "{\"command\":\"sleep 30\"}");
    defer result.deinit(alloc);
    const elapsed_ns = started.durationTo(std.Io.Clock.awake.now(testing.io)).nanoseconds;
    // The tool reached back into the driver while the command ran…
    try testing.expect(ticker.calls > 0);
    // …and the interrupt it set stopped the command at once.
    try testing.expect(result.is_error);
    try testing.expect(std.mem.indexOf(u8, result.text, "cancelled") != null);
    try testing.expect(elapsed_ns < 10 * std.time.ns_per_s);
}
