const std = @import("std");
const cli = @import("cli.zig");
const style = @import("tui/style.zig");

pub fn main(init: std.process.Init) void {
    var diag: cli.Diagnostic = .{};
    run(init, &diag) catch |err| {
        // Report expected failures without a Zig stack trace; all owned resources
        // have already unwound through defer before we set the exit status.
        // A configuration error also carries its reason (the key and rule).
        // The line is `error: <Name>: <reason>` on stderr, the prefix red on
        // a terminal; write failures at this point have nowhere to go.
        const sty = style.Style.detect(init.environ_map, init.io, std.Io.File.stderr());
        var buffer: [1024]u8 = undefined;
        var stderr = std.Io.File.stderr().writerStreaming(init.io, &buffer);
        const w = &stderr.interface;
        w.print("{s}error:{s} {s}{s}{s}", .{ sty.on(.error_text), sty.off(), sty.on(.bold), @errorName(err), sty.off() }) catch {};
        if (diag.len > 0) w.print(": {s}", .{diag.message()}) catch {};
        w.writeByte('\n') catch {};
        w.flush() catch {};
        std.process.exit(1);
    };
}

fn run(init: std.process.Init, diag: *cli.Diagnostic) !void {
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const options = try cli.parseArgs(args[1..]);
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
}
