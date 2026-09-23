//! One loaded companion projector, whatever its family: the file's
//! projector type picks the adapter (Qwen3-VL, or either of Gemma 4's), and
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
    },
    exec: union(enum) {
        qwen3vl_cpu: qwen3vl.Runtime,
        qwen3vl_metal: qwen3vl.Plan,
        gemma4_cpu: gemma4.Runtime,
        gemma4_metal: gemma4.Plan,
    },
    /// The token minimum an image is scaled up to; the reference's, unless
    /// a check pins a smaller fixture (`gemma4` only).
    min_tokens: u32,

    /// Binds `doc` and builds the executor in place: `self` must not move
    /// afterwards (the executor borrows the binding). `backend` selects
    /// Metal; the mapped file must outlive the projector.
    pub fn init(self: *Projector, alloc: std.mem.Allocator, doc: *const gguf.Document, view: weights.View, backend: ?*metal.Backend) !void {
        if (gemma4.kindOf(doc) != null) {
            self.family = .{ .gemma4 = try gemma4.bind(alloc, doc) };
            self.min_tokens = gemma4.min_tokens;
            const binding = &self.family.gemma4;
            self.exec = if (backend) |b| .{ .gemma4_metal = try gemma4.Plan.init(alloc, b, view, binding) } else .{ .gemma4_cpu = try gemma4.Runtime.init(alloc, view, binding) };
        } else {
            self.family = .{ .qwen3vl = try qwen3vl.bind(alloc, doc) };
            self.min_tokens = qwen3vl.min_tokens;
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
        };
    }
    /// The most tokens one image becomes.
    pub fn maxTokens(self: *const Projector) usize {
        return switch (self.family) {
            .qwen3vl => qwen3vl.max_tokens,
            .gemma4 => gemma4.max_tokens,
        };
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
                const g = qwen3vl.gridFor(size);
                break :blk .{ .width_tokens = g.widthTokens(), .height_tokens = g.heightTokens() };
            },
            .gemma4 => blk: {
                const g = gemma4.gridFor(size, self.min_tokens);
                break :blk .{ .width_tokens = g.width_tokens, .height_tokens = g.height_tokens };
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
        }
    }

    /// Runs the projector over `patches` of an image of `g` into `out`
    /// (`g.tokens() × outputWidth()`).
    pub fn encode(self: *Projector, patches: preprocess.Patches, g: Grid, out: []f32) !void {
        const gemma_grid: gemma4.Grid = .{ .width_tokens = g.width_tokens, .height_tokens = g.height_tokens };
        switch (self.exec) {
            .qwen3vl_cpu => |*e| try e.encode(patches, out),
            .qwen3vl_metal => |*e| try e.encode(patches, out),
            .gemma4_cpu => |*e| try e.encode(patches, gemma_grid, out),
            .gemma4_metal => |*e| try e.encode(patches, gemma_grid, out),
        }
    }
};
