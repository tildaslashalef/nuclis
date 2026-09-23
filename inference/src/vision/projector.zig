//! One loaded companion projector, whatever its family: the file's
//! projector type picks the adapter (Qwen3-VL, either of Gemma 4's, or Muse
//! Glimmer's), and
//! the executor picks its CPU reference or Metal plan. The engine and the
//! checks hold this value instead of naming an adapter; its methods are
//! the contract of docs/reference/vision.md § The contract.
const std = @import("std");
const gguf = @import("../formats/gguf.zig");
const weights = @import("../runtime/weights.zig");
const metal = @import("../backends/metal/root.zig");
const image = @import("image.zig");
const preprocess = @import("preprocess.zig");
const qwen3vl = @import("qwen3vl.zig");
const gemma4 = @import("gemma4.zig");
const muse = @import("muse_glimmer.zig");

/// An image's placeholder-token grid.
pub const Grid = struct {
    width_tokens: u32,
    height_tokens: u32,
    pub fn tokens(self: Grid) usize {
        return @as(usize, self.width_tokens) * self.height_tokens;
    }
};

pub const Projector = struct {
    family: union(enum) {
        qwen3vl: qwen3vl.Binding,
        gemma4: gemma4.Binding,
        muse: muse.Binding,
    },
    exec: union(enum) {
        qwen3vl_cpu: qwen3vl.Runtime,
        qwen3vl_metal: qwen3vl.Plan,
        gemma4_cpu: gemma4.Runtime,
        gemma4_metal: gemma4.Plan,
        muse_cpu: muse.Runtime,
        muse_metal: muse.Plan,
    },
    /// The token minimum an image is scaled up to; the reference's, unless
    /// a check pins a smaller fixture (`gemma4` only).
    min_tokens: u32,
    /// The most tokens one image becomes: the family's maximum unless
    /// `limitTokens` lowered it.
    max_tokens: u32,

    /// Binds `doc` and builds the executor in place: `self` must not move
    /// afterwards (the executor borrows the binding). `backend` selects
    /// Metal; the mapped file must outlive the projector.
    pub fn init(self: *Projector, alloc: std.mem.Allocator, doc: *const gguf.Document, view: weights.View, backend: ?*metal.Backend) !void {
        if (muse.matches(doc)) {
            self.family = .{ .muse = try muse.bind(alloc, doc) };
            self.min_tokens = muse.min_tokens;
            self.max_tokens = muse.max_tokens;
            const binding = &self.family.muse;
            self.exec = if (backend) |b| .{ .muse_metal = try muse.Plan.init(alloc, b, view, binding) } else .{ .muse_cpu = try muse.Runtime.init(alloc, view, binding) };
        } else if (gemma4.kindOf(doc) != null) {
            self.family = .{ .gemma4 = try gemma4.bind(alloc, doc) };
            self.min_tokens = gemma4.min_tokens;
            self.max_tokens = gemma4.max_tokens;
            const binding = &self.family.gemma4;
            self.exec = if (backend) |b| .{ .gemma4_metal = try gemma4.Plan.init(alloc, b, view, binding) } else .{ .gemma4_cpu = try gemma4.Runtime.init(alloc, view, binding) };
        } else {
            self.family = .{ .qwen3vl = try qwen3vl.bind(alloc, doc) };
            self.min_tokens = qwen3vl.min_tokens;
            self.max_tokens = qwen3vl.max_tokens;
            const binding = &self.family.qwen3vl;
            self.exec = if (backend) |b| .{ .qwen3vl_metal = try qwen3vl.Plan.init(alloc, b, view, binding) } else .{ .qwen3vl_cpu = try qwen3vl.Runtime.init(alloc, view, binding) };
        }
    }
    pub fn deinit(self: *Projector) void {
        switch (self.exec) {
            inline else => |*e| e.deinit(),
        }
        self.* = undefined;
    }

    /// The width of a feature row: the language model's embedding width.
    pub fn outputWidth(self: *const Projector) usize {
        return switch (self.family) {
            .qwen3vl => qwen3vl.output_width,
            .gemma4 => |b| b.output_width,
            .muse => muse.output_width,
        };
    }
    /// The most tokens one image becomes (the cap in effect).
    pub fn maxTokens(self: *const Projector) usize {
        return self.max_tokens;
    }
    /// The family's token range: the reference's bounds, which the plans'
    /// buffers are sized for.
    pub fn tokenRange(self: *const Projector) struct { min: u32, max: u32 } {
        return switch (self.family) {
            .qwen3vl => .{ .min = qwen3vl.min_tokens, .max = qwen3vl.max_tokens },
            .gemma4 => .{ .min = gemma4.min_tokens, .max = gemma4.max_tokens },
            .muse => .{ .min = muse.min_tokens, .max = muse.max_tokens },
        };
    }
    /// Sets the cap: `null` is the family's maximum, a number is clamped to
    /// the family's range (a setting may name one value for every family).
    pub fn limitTokens(self: *Projector, requested: ?usize) void {
        const range = self.tokenRange();
        self.max_tokens = capWithin(range.min, range.max, requested);
    }
    /// Whether the language model attends bidirectionally inside a span
    /// (Gemma 4 on its sliding layers), which needs the span in one chunk.
    pub fn bidirectional(self: *const Projector) bool {
        return self.family == .gemma4;
    }

    /// The token grid of a decoded image of `size`.
    pub fn grid(self: *const Projector, size: preprocess.Size) Grid {
        return switch (self.family) {
            .qwen3vl => blk: {
                const g = qwen3vl.gridFor(size, self.max_tokens);
                break :blk .{ .width_tokens = g.widthTokens(), .height_tokens = g.heightTokens() };
            },
            .gemma4 => blk: {
                const g = gemma4.gridFor(size, self.min_tokens, self.max_tokens);
                break :blk .{ .width_tokens = g.width_tokens, .height_tokens = g.height_tokens };
            },
            .muse => blk: {
                const g = muse.gridFor(size, self.max_tokens);
                break :blk .{ .width_tokens = g.widthTokens(), .height_tokens = g.heightTokens() };
            },
        };
    }

    /// Resizes `source` to `g` and lays out the projector's patch rows.
    /// Caller owns the result.
    pub fn prepare(self: *const Projector, alloc: std.mem.Allocator, source: image.Rgb8, g: Grid) !preprocess.Patches {
        switch (self.family) {
            .qwen3vl => |b| {
                const target: preprocess.Size = .{ .width = g.width_tokens * qwen3vl.patch * qwen3vl.merge, .height = g.height_tokens * qwen3vl.patch * qwen3vl.merge };
                const resized = try preprocess.resizeLetterbox(alloc, source, target);
                defer alloc.free(resized);
                return preprocess.patches(alloc, resized, target, .{ .patch = qwen3vl.patch, .merge = qwen3vl.merge, .mean = b.mean, .std = b.std });
            },
            .gemma4 => |b| {
                const target: preprocess.Size = .{ .width = g.width_tokens * gemma4.token_side, .height = g.height_tokens * gemma4.token_side };
                const resized = try preprocess.resizeLetterbox(alloc, source, target);
                defer alloc.free(resized);
                return preprocess.patches(alloc, resized, target, gemma4.patchOptions(b.kind, b.mean, b.std));
            },
            .muse => |b| {
                // A stretch to the grid, no letterbox.
                const target: preprocess.Size = .{ .width = g.width_tokens * muse.token_side, .height = g.height_tokens * muse.token_side };
                const resized = try preprocess.resizeLanczos(alloc, source, target);
                defer alloc.free(resized);
                return preprocess.patches(alloc, resized, target, muse.patchOptions(b.mean, b.std));
            },
        }
    }

    /// Muse Glimmer only, for the checks: the encoder's residual rows (window
    /// order) after the first `count` blocks.
    pub fn encodeResidual(self: *Projector, patches: preprocess.Patches, g: Grid, count: usize, rows: []f32) !void {
        const muse_grid: muse.Grid = .{ .width_patches = g.width_tokens * muse.merge, .height_patches = g.height_tokens * muse.merge };
        switch (self.exec) {
            .muse_cpu => |*e| try e.encodeResidual(patches, muse_grid, count, rows),
            .muse_metal => |*e| try e.encodeResidual(patches, muse_grid, count, rows),
            else => return error.Unsupported,
        }
    }

    /// Runs the projector over `patches` of an image of `g` into `out`
    /// (`g.tokens() × outputWidth()`).
    pub fn encode(self: *Projector, patches: preprocess.Patches, g: Grid, out: []f32) !void {
        const gemma_grid: gemma4.Grid = .{ .width_tokens = g.width_tokens, .height_tokens = g.height_tokens };
        const muse_grid: muse.Grid = .{ .width_patches = g.width_tokens * muse.merge, .height_patches = g.height_tokens * muse.merge };
        switch (self.exec) {
            .qwen3vl_cpu => |*e| try e.encode(patches, out),
            .qwen3vl_metal => |*e| try e.encode(patches, out),
            .gemma4_cpu => |*e| try e.encode(patches, gemma_grid, out),
            .gemma4_metal => |*e| try e.encode(patches, gemma_grid, out),
            .muse_cpu => |*e| try e.encode(patches, muse_grid, out),
            .muse_metal => |*e| try e.encode(patches, muse_grid, out),
        }
    }
};

/// The cap for a family whose range is `[min, max]`: `max` for null, else
/// the request clamped into the range.
fn capWithin(min: u32, max: u32, requested: ?usize) u32 {
    return @intCast(std.math.clamp(requested orelse max, min, max));
}

test "a requested cap is clamped into the family's range" {
    try std.testing.expectEqual(@as(u32, 1120), capWithin(70, 1120, null));
    try std.testing.expectEqual(@as(u32, 1120), capWithin(70, 1120, 4096));
    try std.testing.expectEqual(@as(u32, 70), capWithin(70, 1120, 16));
    try std.testing.expectEqual(@as(u32, 512), capWithin(8, 1024, 512));
}
