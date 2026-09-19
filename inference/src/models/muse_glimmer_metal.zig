//! The Metal plan for Muse Glimmer does not exist yet. The registry and the
//! engine's executor union require every family to expose a `Plan`, so this
//! placeholder keeps the family registrable and makes `--backend metal` a
//! typed failure at open (`MetalPlanUnavailable`); no method past `init`
//! can run. The real plan composes the backend's kernels over the schedule
//! of `muse_glimmer_runtime.zig` (docs/reference/muse-glimmer.md).
const std = @import("std");
const model = @import("muse_glimmer.zig");
const weights = @import("../runtime/weights.zig");
const session = @import("../runtime/session.zig");
const sampling = @import("../sampling/root.zig");
const metal = @import("../backends/metal/root.zig");

pub const Observer = @import("../runtime/observer.zig").Observer;

pub const Plan = struct {
    state: session.Session,

    pub fn init(alloc: std.mem.Allocator, backend: *metal.Backend, view: weights.View, binding: model.Binding, capacity: usize, chunk: usize, kv: session.Precision) !Plan {
        _ = .{ alloc, backend, view, binding, capacity, chunk, kv };
        return error.MetalPlanUnavailable;
    }
    pub fn deinit(self: *Plan) void {
        self.* = undefined;
    }
    pub fn reset(self: *Plan) void {
        _ = self;
    }
    pub fn step(self: *Plan, token: u32, logits: ?[]f32, greedy: ?*u32, topk: ?*sampling.TopK, observer: ?Observer) !void {
        _ = .{ self, token, logits, greedy, topk, observer };
        return error.MetalPlanUnavailable;
    }
    pub fn prefill(self: *Plan, tokens: []const u32, logits: ?[]f32, greedy: ?*u32, topk: ?*sampling.TopK, observer: ?Observer) !void {
        _ = .{ self, tokens, logits, greedy, topk, observer };
        return error.MetalPlanUnavailable;
    }
    pub fn readLogits(self: *Plan, out: []f32) !void {
        _ = .{ self, out };
        return error.MetalPlanUnavailable;
    }
};
