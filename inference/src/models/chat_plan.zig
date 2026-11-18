//! The output readout the chat Metal plans share (Qwen3.8's and Gemma 4's):
//! the logits buffer, the penalties' history upload, the argmax and top-k
//! selections over it, their readbacks, and the verify batch's per-row
//! readbacks. The head matmul that fills `logits` and any shaping of it (a
//! soft-cap, the verify head's) stay with each family's plan, as does every
//! schedule; this module holds no layer, weight, or prompt knowledge.
//!
//! A `Readout` holds buffer handles and scalars only and takes the backend per
//! call: plans are moved by value, so it keeps no pointer to its plan.
//! Its argmax and top-k scratch is free between steps, and a drafter may
//! borrow it for its own selection (Qwen's prediction block does).
const std = @import("std");
const sampling = @import("../sampling/root.zig");
const metal = @import("../backends/metal/root.zig");
const Buffer = metal.Buffer;
const TopKBuffers = metal.Backend.TopKBuffers;

/// The readout of a vocabulary of `vocabulary` logits, with `verify_rows`
/// rows of verify scratch when `verify` (a drafter is loaded).
pub fn Readout(comptime vocabulary: usize, comptime verify_rows: usize) type {
    return struct {
        const Self = @This();

        /// The last step's vocabulary scores, as the family's head shaped them.
        logits: Buffer,
        argmax_values: Buffer,
        argmax_indices: Buffer,
        argmax_result: Buffer,
        /// Partial top-k scratch for sampled decoding; `sampling.TopK.capacity` entries.
        topk: TopKBuffers,
        /// The history bit set as device words for `nu_penalize`, uploaded when
        /// the history's revision changed (one 31 KB copy per change, not per
        /// step). `penalty_revision` is the last uploaded revision.
        penalty_history: Buffer,
        penalty_revision: u64,
        /// The verify scratch exists: `verify_logits` (`matmulPadded(verify_rows)`
        /// vocabulary rows, since `matmul` writes whole tiles), the per-row
        /// argmax results, and one top-k set per row for the sampled
        /// acceptance path. Without it the two buffers are 4-byte stubs.
        verify: bool,
        verify_logits: Buffer,
        verify_argmax: Buffer,
        topk_rows: [verify_rows]TopKBuffers,

        /// Creates every buffer through `created`, the verify scratch only
        /// when `verify`.
        pub fn init(created: *metal.Created, verify: bool) !Self {
            var self: Self = undefined;
            self.logits = try created.create(vocabulary * 4);
            self.verify = verify;
            if (verify) {
                self.verify_logits = try created.create(metal.Backend.matmulPadded(verify_rows) * vocabulary * 4);
                self.verify_argmax = try created.create(verify_rows * 4);
                for (&self.topk_rows) |*scratch| scratch.* = try created.topk(sampling.TopK.capacity);
            } else {
                self.verify_logits = try created.create(4);
                self.verify_argmax = try created.create(4);
            }
            self.argmax_values = try created.create(metal.Backend.argmax_partials * 4);
            self.argmax_indices = try created.create(metal.Backend.argmax_partials * 4);
            self.argmax_result = try created.create(4);
            self.topk = try created.topk(sampling.TopK.capacity);
            self.penalty_history = try created.create(@divCeil(vocabulary, 32) * 4);
            self.penalty_revision = std.math.maxInt(u64);
            return self;
        }

        /// Uploads the history words to the device when the set changed since
        /// the last upload; valid before the command buffer's first dispatch
        /// (the shared-storage buffer is only read by the GPU at `commit`).
        pub fn syncPenalties(self: *Self, p: sampling.Penalties) !void {
            if (p.history.revision == self.penalty_revision) return;
            const words = @divCeil(vocabulary, 32);
            const out = @as([*]u32, @ptrCast(@alignCast(self.penalty_history.host)))[0..words];
            if (p.history.writeWords(out) != words) return error.InvalidShape;
            self.penalty_revision = p.history.revision;
        }

        /// Records the selections over `logits`: `penalties` applies
        /// `nu_penalize` in place first, so both selections see the vector
        /// the CPU sampler would sort; then the argmax when `greedy`, the
        /// top-k when `topk`.
        pub fn select(self: *Self, b: *metal.Backend, greedy: bool, topk: ?*sampling.TopK, penalties: ?sampling.Penalties) !void {
            // In place, so only a selection readback may request it: a raw
            // `logits` readback must stay unpenalized for a CPU sampler.
            if (penalties) |p| if (greedy or topk != null) try b.penalize(self.logits, vocabulary, self.penalty_history, p.repetition, p.presence);
            if (greedy) try b.argmax(self.logits, vocabulary, self.argmax_values, self.argmax_indices, self.argmax_result);
            if (topk) |top| try b.topk(self.logits, vocabulary, sampling.TopK.capacity, top.temperature, self.topk);
        }

        /// Copies the requested readbacks out after `commit`. `penalized`
        /// marks a `topk` readback taken after the device penalties, which is
        /// what lets `Sampler.selectFrom` decide with penalties active.
        pub fn readOutputs(self: *Self, logits: ?[]f32, greedy: ?*u32, topk: ?*sampling.TopK, penalized: bool) !void {
            if (logits) |out| try self.readLogits(out);
            if (greedy) |out| {
                const id = @as(*const u32, @ptrCast(@alignCast(self.argmax_result.host))).*;
                if (id >= vocabulary) return error.NonFiniteResult;
                out.* = id;
            }
            if (topk) |top| {
                top.penalized = penalized;
                const ids = @as([*]const u32, @ptrCast(@alignCast(self.topk.indices.host)))[0..sampling.TopK.capacity];
                for (ids) |id| if (id >= vocabulary) return error.NonFiniteResult;
                @memcpy(top.ids[0..], ids);
                @memcpy(top.values[0..], self.topk.values.floats()[0..sampling.TopK.capacity]);
                top.count = sampling.TopK.capacity;
                // Partition order is fixed, so this F64 sum is deterministic run to run.
                var total: f64 = 0;
                for (self.topk.sums.floats()[0..metal.Backend.topk_partials]) |partial| total += partial;
                top.total = total;
                top.finite = true;
                for (@as([*]const u32, @ptrCast(@alignCast(self.topk.flags.host)))[0..metal.Backend.topk_partials]) |flag| if (flag != 0) {
                    top.finite = false;
                };
            }
        }

        /// Copies the last step's logits out of the shared buffer. Valid after
        /// a step that computed them and until the next step; the sampler's
        /// fallback path uses it. When a selection was requested with
        /// penalties, the vector is the penalized one (the raw vector is not
        /// kept), so the sampler must not penalize it again.
        pub fn readLogits(self: *Self, out: []f32) !void {
            if (out.len != vocabulary) return error.InvalidShape;
            @memcpy(out, self.logits.floats());
            for (out) |v| if (!std.math.isFinite(v)) return error.NonFiniteResult;
        }

        /// One verify row's partial top-k readback from its own scratch set.
        /// The values are the raw logits (the penalties, when active, are the
        /// sampler's on the host and `selectFromHistory` checks the bound).
        pub fn readVerifyTopK(self: *Self, row: usize, top: *sampling.TopK) !void {
            const scratch = self.topk_rows[row];
            const ids = @as([*]const u32, @ptrCast(@alignCast(scratch.indices.host)))[0..sampling.TopK.capacity];
            for (ids) |id| if (id >= vocabulary) return error.NonFiniteResult;
            @memcpy(top.ids[0..], ids);
            @memcpy(top.values[0..], scratch.values.floats()[0..sampling.TopK.capacity]);
            top.count = sampling.TopK.capacity;
            var total: f64 = 0;
            for (scratch.sums.floats()[0..metal.Backend.topk_partials]) |partial| total += partial;
            top.total = total;
            top.finite = true;
            for (@as([*]const u32, @ptrCast(@alignCast(scratch.flags.host)))[0..metal.Backend.topk_partials]) |flag| if (flag != 0) {
                top.finite = false;
            };
            top.penalized = false;
        }

        /// Copies one row of the last verify batch's logits out of the shared
        /// buffer: the fallback for a row whose readback could not decide.
        /// Valid until the next verify; `NoDraftBlock` without the scratch.
        pub fn readVerifyRow(self: *Self, row: usize, out: []f32) !void {
            if (!self.verify) return error.NoDraftBlock;
            if (row >= verify_rows or out.len != vocabulary) return error.InvalidShape;
            @memcpy(out, self.verify_logits.floats()[row * vocabulary ..][0..vocabulary]);
            for (out) |v| if (!std.math.isFinite(v)) return error.NonFiniteResult;
        }
    };
}

test "a readout without verify scratch refuses a verify row" {
    var readout: Readout(8, 2) = undefined;
    readout.verify = false;
    var out: [8]f32 = undefined;
    try std.testing.expectError(error.NoDraftBlock, readout.readVerifyRow(0, &out));
}
