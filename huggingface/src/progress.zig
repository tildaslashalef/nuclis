//! Presentation-neutral progress. Events borrow the filename only for the call;
//! a TUI retaining an event must copy the name. No terminal or clock is implicit.
pub const Event = struct {
    phase: enum { resolving, selection_required, downloading, verifying, complete },
    filename: ?[]const u8 = null,
    /// Zero-based file index; file_count is zero until selection is complete.
    file_index: usize = 0,
    file_count: usize = 0,
    file_completed: u64 = 0,
    file_total: u64 = 0,
    /// Overall validated/reconstructed bytes, including verified local reuse.
    completed_bytes: u64 = 0,
    total_bytes: u64 = 0,
    reused: bool = false,
    /// Bytes of this file continued from a kept partial download, counted in
    /// `file_completed` and `completed_bytes` from the first downloading event.
    resumed_bytes: u64 = 0,

    /// null means indeterminate (metadata/selection); 1 still needs phase==complete
    /// before the TUI announces success, since hashing and publication may fail.
    pub fn fraction(e: Event) ?f64 {
        if (e.total_bytes == 0) return null;
        return @min(@as(f64, 1), @as(f64, @floatFromInt(e.completed_bytes)) / @as(f64, @floatFromInt(e.total_bytes)));
    }
};

pub const Sink = struct {
    context: ?*anyopaque = null,
    /// Synchronous, serialized on the caller task, never called by download
    /// workers. Keep it short; enqueue an event for a TUI running elsewhere.
    /// Canceled stops the operation and removes the current file's partial
    /// download; Paused stops and keeps it (`<path>.partial` and its
    /// `.partial.json`) for the next download to continue. Already published
    /// files remain valid either way.
    update: *const fn (?*anyopaque, Event) Error!void,

    pub const Error = error{ Canceled, Paused };

    pub fn report(s: Sink, event: Event) Error!void {
        try s.update(s.context, event);
    }
};

test "progress is indeterminate before selection and bounded afterwards" {
    const std = @import("std");
    try std.testing.expect((Event{ .phase = .resolving }).fraction() == null);
    try std.testing.expectEqual(@as(f64, 0.5), (Event{ .phase = .downloading, .total_bytes = 10, .completed_bytes = 5 }).fraction().?);
    try std.testing.expectEqual(@as(f64, 1), (Event{ .phase = .complete, .total_bytes = 10, .completed_bytes = 11 }).fraction().?);
}
