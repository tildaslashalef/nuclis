//! Cooperative Ctrl-C handling. The signal handler only sets a flag; the
//! generation loop observes it between layers and stops with a `cancelled`
//! reason, so partial output is flushed and resources unwind normally.
//! The handler resets itself after the first signal: a second Ctrl-C kills
//! the process the ordinary way if the loop cannot reach a checkpoint.
const std = @import("std");

var requested_flag: std.atomic.Value(bool) = .init(false);

fn handle(_: std.posix.SIG) callconv(.c) void {
    requested_flag.store(true, .seq_cst);
}

/// Installs the SIGINT handler for the rest of the process. Safe to call once;
/// there is no process state to restore because the handler is one-shot.
pub fn install() void {
    const act: std.posix.Sigaction = .{
        .handler = .{ .handler = handle },
        .mask = std.posix.sigemptyset(),
        .flags = std.posix.SA.RESETHAND,
    };
    std.posix.sigaction(.INT, &act, null);
}

/// Requests cancellation from ordinary code, for callers that read Ctrl-C as
/// a key in raw terminal mode (ISIG off) instead of receiving SIGINT.
pub fn request() void {
    requested_flag.store(true, .seq_cst);
}

pub fn requested() bool {
    return requested_flag.load(.seq_cst);
}

/// Consumes a pending request so a later operation starts clean. Because the
/// handler is one-shot, callers that want to survive a second Ctrl-C must
/// `install()` again after clearing.
pub fn clear() void {
    requested_flag.store(false, .seq_cst);
}

/// Error checkpoint used by observers: returns `error.Cancelled` when
/// interruption was requested.
pub fn check() error{Cancelled}!void {
    if (requested()) return error.Cancelled;
}

test "flag round trip without signals" {
    try std.testing.expect(!requested());
    request();
    try std.testing.expectError(error.Cancelled, check());
    clear();
    try check();
}
