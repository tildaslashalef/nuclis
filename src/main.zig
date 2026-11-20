const std = @import("std");
const cli = @import("cli.zig");
const style = @import("tui/style.zig");
const terminal = @import("tui/terminal.zig");
const request_log = @import("api/request_log.zig");
const kit = @import("ui.zig");
const errors = @import("errors.zig");

/// A panic puts a leased terminal back and, under `serve`, writes the
/// request log's last line before the default report: the shell is usable
/// to read it, and the crash names its request.
pub const panic = std.debug.FullPanic(onPanic);

fn onPanic(message: []const u8, first_trace_address: ?usize) noreturn {
    terminal.emergencyRestore();
    request_log.writePanic(message);
    std.debug.defaultPanic(message, first_trace_address);
}

pub fn main(init: std.process.Init) void {
    log_ui = kit.Ui.detect(init.environ_map, init.io, std.Io.File.stderr());
    var diag: cli.Diagnostic = .{};
    run(init, &diag) catch |err| {
        // Report expected failures without a Zig stack trace; all owned
        // resources have already unwound through defer. The line is
        // `nuclis: <message>` on stderr, the message the failure's
        // diagnostic or else `errors.describe`; a usage error adds its hint
        // and exits 2 (docs/app/command-output.md).
        var buffer: [1024]u8 = undefined;
        var stderr = std.Io.File.stderr().writerStreaming(init.io, &buffer);
        reportError(&stderr.interface, log_ui, err, &diag) catch {};
        stderr.interface.flush() catch {};
        std.process.exit(errors.exitCode(err, diag.hint_len > 0));
    };
}

/// `nuclis: <message>` and, for a usage error, its hint on the next line.
fn reportError(w: *std.Io.Writer, ui: kit.Ui, err: anyerror, diag: *const cli.Diagnostic) !void {
    try w.print("{s}nuclis:{s} ", .{ ui.on(.fail), ui.off(.fail) });
    if (diag.len > 0) {
        try w.writeAll(diag.message());
    } else {
        const described = errors.describe(err);
        var words: [96]u8 = undefined;
        try w.writeAll(if (described.len > 0) described else errors.words(&words, @errorName(err)));
    }
    try w.writeByte('\n');
    if (diag.hint_len > 0) try w.print("{s}{s}{s}\n", .{ ui.on(.muted), diag.hint(), ui.off(.muted) });
}

/// `std.log` in the same voice as the error line: `nuclis: warning: …`,
/// `nuclis: …`, the prefix in the warning or failure colour; debug messages
/// are dropped outside debug builds as usual.
pub const std_options: std.Options = .{ .logFn = log };

/// stderr's settings, detected once at startup; plain until then (tests).
var log_ui: kit.Ui = .plain;

fn log(comptime level: std.log.Level, comptime scope: @EnumLiteral(), comptime format: []const u8, args: anytype) void {
    _ = scope;
    const prefix, const role: kit.style.Role = switch (level) {
        .err => .{ "nuclis:", .fail },
        .info, .debug => .{ "nuclis:", .muted },
        .warn => .{ "nuclis: warning:", .warn },
    };
    const io = std.Options.debug_io;
    const prev = io.swapCancelProtection(.blocked);
    defer _ = io.swapCancelProtection(prev);
    var buffer: [512]u8 = undefined;
    const stderr = std.debug.lockStderr(&buffer).terminal();
    defer std.debug.unlockStderr();
    stderr.writer.print("{s}" ++ prefix ++ "{s} " ++ format ++ "\n", .{ log_ui.on(role), log_ui.off(role) } ++ args) catch {};
}

fn run(init: std.process.Init, diag: *cli.Diagnostic) !void {
    // The CPU kernels size their own fan-out to the core count. The default
    // async limit (cores - 1) also counts tasks parked in `concurrent` (a
    // server's accept loop, its executor, each connection), and past it
    // `async` runs inline: `serve` on 4 cores ran every kernel on one.
    // The juicy main's `io` is a `std.Io.Threaded`.
    const threaded: *std.Io.Threaded = @ptrCast(@alignCast(init.io.userdata.?));
    threaded.setAsyncLimit(.unlimited);
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const options = try cli.parse(args[1..], diag);
    var buffer: [4096]u8 = undefined;
    // Streaming, not positional: the default writer tracks its own offset
    // from zero, which overwrites a file opened with `>>` (seen while
    // appending bench reports to one log).
    var output = std.Io.File.stdout().writerStreaming(init.io, &buffer);
    // Flushed on every exit: a command that reports and then fails with a
    // typed error (`model pull` listing the choices before
    // `SelectionRequired`) keeps what it printed.
    defer output.interface.flush() catch {};
    try cli.run(init.gpa, init.io, init.environ_map, options, &output.interface, diag);
}

test {
    _ = cli;
    _ = kit;
    _ = errors;
}

test "the error line names no error, and a usage error carries its hint" {
    var buf: std.Io.Writer.Allocating = .init(std.testing.allocator);
    defer buf.deinit();
    var diag: cli.Diagnostic = .{};
    try reportError(&buf.writer, .plain, error.ContextFull, &diag);
    try std.testing.expectEqualStrings("nuclis: the context is full; raise --ctx-size or shorten the prompt\n", buf.written());
    buf.clearRetainingCapacity();
    try reportError(&buf.writer, .plain, error.SomethingRare, &diag);
    try std.testing.expectEqualStrings("nuclis: something rare\n", buf.written());
    buf.clearRetainingCapacity();
    _ = cli.parse(&.{ "model", "pull", "--frob" }, &diag) catch {};
    try reportError(&buf.writer, .plain, error.UnknownOption, &diag);
    try std.testing.expectEqualStrings("nuclis: 'nuclis model pull' needs a name: a catalogue name, a registry entry, owner/repo, or --catalogue\nRun 'nuclis model --help' for usage.\n", buf.written());
}
