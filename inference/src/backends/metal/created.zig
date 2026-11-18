//! An init-local record of the buffers a plan's `init` creates, so a failure
//! part way releases every one of them. A plan declares one on its `init`'s
//! stack (`var created: metal.Created = .init(alloc, backend); defer
//! created.deinit(); errdefer created.releaseAll();`) and passes it to the
//! helpers that create; nothing keeps a pointer to it after `init` returns,
//! and buffers created later belong to the backend as before.
//!
//! It knows `Backend` and `Buffer` only: any plan may use it.
const std = @import("std");
const root = @import("root.zig");
const Backend = root.Backend;
const Buffer = root.Buffer;

pub const Created = struct {
    alloc: std.mem.Allocator,
    backend: *Backend,
    buffers: std.ArrayList(Buffer) = .empty,

    pub fn init(alloc: std.mem.Allocator, backend: *Backend) Created {
        return .{ .alloc = alloc, .backend = backend };
    }

    /// Frees the record, not the buffers: on success they are the plan's.
    pub fn deinit(self: *Created) void {
        self.buffers.deinit(self.alloc);
        self.* = undefined;
    }

    /// Releases every recorded buffer, newest first; for `init`'s errdefer.
    pub fn releaseAll(self: *Created) void {
        while (self.buffers.pop()) |buffer| self.backend.release(buffer) catch {};
    }

    /// Room for `n` more records, so the creations after it cannot fail to
    /// be recorded.
    pub fn reserve(self: *Created, n: usize) !void {
        try self.buffers.ensureUnusedCapacity(self.alloc, n);
    }

    /// A zero-filled buffer of `len` bytes, recorded.
    pub fn create(self: *Created, len: usize) !Buffer {
        try self.reserve(1);
        const buffer = try self.backend.create(len);
        self.buffers.appendAssumeCapacity(buffer);
        return buffer;
    }

    /// Records a buffer made elsewhere (a plan's own buffer sets) into room a
    /// `reserve` before its creation made, so recording it cannot fail.
    pub fn adopt(self: *Created, buffer: Buffer) void {
        self.buffers.appendAssumeCapacity(buffer);
    }

    /// `Backend.topkBuffers`' six buffers for `k` entries, each recorded as
    /// it is made, so a failure part way leaks none.
    pub fn topk(self: *Created, k: usize) !Backend.TopKBuffers {
        const partials = Backend.topk_partials;
        return .{
            .partial_values = try self.create(partials * k * 4),
            .partial_indices = try self.create(partials * k * 4),
            .values = try self.create(k * 4),
            .indices = try self.create(k * 4),
            .sums = try self.create(partials * 4),
            .flags = try self.create(partials * 4),
        };
    }
};
