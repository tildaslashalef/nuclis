//! Per-dispatch GPU timing (`Backend.enableProfiling`), the commit tick, and
//! pipeline statistics.
const std = @import("std");
const Kernel = @import("kernels.zig").Kernel;

/// Per-dispatch GPU time accumulated by kernel and shape. Only dispatches the
/// GPU actually timestamped are added to `totals`; the rest are counted in
/// `unsampled` so a report can never attribute time it did not measure.
pub const Profile = struct {
    /// Matrix kernels carry their encoding and dimensions so one kernel's time
    /// splits by tensor shape (a 48-row projection is not a 17,408-row one);
    /// other kernels aggregate by name with `encoding = null` and zero dims.
    pub const Key = struct { kernel: Kernel, encoding: ?u32, rows: u32, columns: u32 };
    pub const Total = struct { dispatches: u64 = 0, seconds: f64 = 0, bytes: u64 = 0 };
    const Pending = struct { key: Key, bytes: u64 };

    totals: std.AutoHashMapUnmanaged(Key, Total) = .empty,
    /// Kernels recorded into the open command buffer, in dispatch order; paired
    /// with the bridge's durations at `commit`.
    pending: std.ArrayListUnmanaged(Pending) = .empty,
    durations: []f64 = &.{},
    /// Committed command buffers that contained at least one dispatch.
    command_buffers: u64 = 0,
    unsampled: u64 = 0,
    /// Whole-command-buffer GPU time while profiling. Exceeds the sum of the
    /// totals by the cost of the per-dispatch encoder boundaries.
    gpu_seconds: f64 = 0,
    /// While set, committed command buffers are timed but not accounted: a
    /// caller narrows the profile to the work it brackets.
    paused: bool = false,

    pub fn deinit(self: *Profile, alloc: std.mem.Allocator) void {
        self.totals.deinit(alloc);
        self.pending.deinit(alloc);
        alloc.free(self.durations);
        self.* = undefined;
    }
    /// Discards accumulated totals (for example after warm-up runs); the
    /// bridge's sample buffers stay in place.
    pub fn clear(self: *Profile) void {
        self.totals.clearRetainingCapacity();
        self.pending.clearRetainingCapacity();
        self.command_buffers = 0;
        self.unsampled = 0;
        self.gpu_seconds = 0;
    }
};

/// What a dispatch is about, for profiling: the weight encoding and matrix
/// dimensions of a matrix kernel and the bytes it must read. Elementwise
/// kernels pass `.{}`.
pub const Shape = struct { encoding: ?u32 = null, rows: u32 = 0, columns: u32 = 0, bytes: u64 = 0 };

/// A cooperative beat during a GPU wait longer than ~100 ms: an interactive
/// caller reads its keyboard and repaints here. It must not touch the backend
/// or the model — the command buffer it interrupts is still in flight.
pub const Tick = struct {
    context: *anyopaque,
    call: *const fn (context: *anyopaque) void,
};

/// A compiled pipeline's limits: `max_threads` per threadgroup drops below
/// the device's 1024 when the kernel's registers limit occupancy.
pub const PipelineStats = struct {
    name: []const u8,
    max_threads: u32,
    execution_width: u32,
    /// Bytes of `threadgroup` arrays the kernel declares statically.
    threadgroup_memory: u32,
};
