//! A test build's deliberate panics (`zig build -Dpanic-hook=true`), so a
//! check can watch the panic handler restore a leased terminal and write the
//! request log's last line (docs/guide/code.md § Hostile input). Every other
//! build compiles the hooks out.
const std = @import("std");
const build_options = @import("build_options");

pub const enabled = build_options.panic_hook;

/// Panics when this build has the hooks and `NUCLIS_PANIC_HOOK` is `where`.
pub fn at(comptime where: []const u8) void {
    if (!enabled) return;
    const value = std.c.getenv("NUCLIS_PANIC_HOOK") orelse return;
    if (std.mem.eql(u8, std.mem.span(value), where)) @panic("panic hook: " ++ where);
}
