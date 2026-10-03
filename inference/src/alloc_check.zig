//! Exhaustive allocation-failure checks that stay deterministic over
//! allocators that grow blocks in place only sometimes.

const std = @import("std");
const Allocator = std.mem.Allocator;

/// `std.testing.checkAllAllocationFailures` over `backing` with in-place
/// growth refused. `std.testing.allocator` (and `std.process.Init.gpa` in
/// Debug) grows a block in place only while it ends its bucket, which
/// depends on what earlier runs allocated, so growing code would make a
/// different number of allocations from run to run and the check would
/// fail with `NondeterministicMemoryUsage`. Leaks and double frees are
/// still reported by `backing`.
pub fn checkAll(backing: Allocator, comptime test_fn: anytype, extra_args: anytype) !void {
    var inner = backing;
    try std.testing.checkAllAllocationFailures(noGrowth(&inner), test_fn, extra_args);
}

/// `backing` with `resize` and `remap` always refused, so every growth is a
/// fresh allocation. Borrows `backing`, which must outlive the result.
pub fn noGrowth(backing: *const Allocator) Allocator {
    return .{ .ptr = @constCast(backing), .vtable = &.{
        .alloc = alloc,
        .resize = Allocator.noResize,
        .remap = Allocator.noRemap,
        .free = free,
    } };
}

fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ra: usize) ?[*]u8 {
    const backing: *const Allocator = @ptrCast(@alignCast(ctx));
    return backing.rawAlloc(len, alignment, ra);
}

fn free(ctx: *anyopaque, memory: []u8, alignment: std.mem.Alignment, ra: usize) void {
    const backing: *const Allocator = @ptrCast(@alignCast(ctx));
    backing.rawFree(memory, alignment, ra);
}

test "a growing list allocates the same count on every run" {
    try checkAll(std.testing.allocator, struct {
        fn run(gpa: Allocator, n: usize) !void {
            var list: std.ArrayList(u32) = .empty;
            defer list.deinit(gpa);
            for (0..n) |i| try list.append(gpa, @intCast(i));
            const copy = try gpa.dupe(u32, list.items);
            gpa.free(copy);
        }
    }.run, .{100});
}
