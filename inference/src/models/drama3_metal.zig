//! Drama 3's backbone and codebook tower on Metal: the CPU reference's
//! forward (drama3_runtime.zig) with every matrix wrapped in place from the
//! mapping (Q4_K and F32, nothing requantized), the reference `Model`'s F32
//! constants in device buffers, and the input rows (the backbone's
//! embeddings, the tower's code embeddings) decoded on the host exactly as
//! the reference decodes them. One command buffer per prefill chunk and per
//! backbone row; a tower step commits after each codebook head, since the
//! caller's `Pick` samples on the host (docs/models/drama3.md § On Metal).
//!
//! Ownership: the plan borrows the `Backend` and the `runtime.Model` (and
//! through it the mapping), which must outlive it. It owns a
//! `runtime.State`, whose session memory is wrapped as shared buffers, and
//! the device buffers it creates, released in `deinit`. The backend commits
//! synchronously, so host writes between commits never race the GPU.
const std = @import("std");
const model = @import("drama3.zig");
const runtime = @import("drama3_runtime.zig");
const session = @import("../runtime/session.zig");
const metal = @import("../backends/metal/root.zig");
const cpu = @import("../backends/cpu/root.zig");
const Tensor = @import("../formats/gguf.zig").Tensor;

const Backend = metal.Backend;
const Buffer = metal.Buffer;
pub const Frame = runtime.Frame;
pub const Pick = runtime.Pick;

/// Prompt rows one prefill command buffer runs by default: enough for a
/// reference clip's prompt in one chunk, and for its expert slot rows to
/// fill the gathered tiles.
pub const default_chunk = 512;

const hidden = model.hidden;
const tower_layers = model.shared_layers + model.residual_layers;

const AttentionConstants = struct { query_norm: Buffer, key_norm: Buffer };
const DeltaConstants = struct { convolution: Buffer, a: Buffer, time_bias: Buffer, norm: Buffer };
const LayerConstants = struct {
    attention_norm: Buffer,
    post_attention_norm: Buffer,
    /// `ffn_gate_inp_shexp` as a vector: the shared expert's sigmoid gate.
    shared_gate: Buffer,
    mixer: union(enum) { full_attention: AttentionConstants, delta_net: DeltaConstants },
};
const TowerConstants = struct {
    enorm: Buffer,
    hnorm: Buffer,
    final_norm: Buffer,
    attention_norm: Buffer,
    ffn_norm: Buffer,
    attention: AttentionConstants,
};

const Weight = struct { buffer: Buffer, matrix: cpu.Matrix };
const ExpertWeight = struct { buffer: Buffer, tensor: cpu.ExpertMatrix };

pub const Plan = struct {
    alloc: std.mem.Allocator,
    backend: *Backend,
    model: *const runtime.Model,
    state: runtime.State,
    backbone_memory: Buffer,
    tower_memory: Buffer,
    /// Every buffer `create` made, released by `deinit`.
    created: std.ArrayList(Buffer),
    layers: [model.layer_count]LayerConstants,
    shared: [model.shared_layers]TowerConstants,
    residual: [model.residual_layers]TowerConstants,
    output_norm: Buffer,
    /// (cos, sin) of the 32 rotary pairs per position, base 1e7: the
    /// backbone's and the tower's angles are the same, only the pairing differs.
    rope_table: Buffer,
    chunk: usize,
    padded: usize,
    // Rows of one chunk, `padded` of each (row 0 serves a decode row).
    x: Buffer,
    normalized: Buffer,
    projected: Buffer,
    /// Every row's post-`output_norm` hidden.
    normed: Buffer,
    /// The last backbone row's post-norm hidden; `h` is its host view.
    last: Buffer,
    q: Buffer,
    gate: Buffer,
    k: Buffer,
    v: Buffer,
    attended: Buffer,
    mixed: Buffer,
    convolved: Buffer,
    z: Buffer,
    alpha: Buffer,
    beta: Buffer,
    partials: Buffer,
    router: Buffer,
    indices: Buffer,
    route_weights: Buffer,
    lists: Buffer,
    expert_gate: Buffer, // padded · experts_used slot rows
    expert_up: Buffer,
    expert_out: Buffer,
    shared_gate: Buffer,
    shared_up: Buffer,
    shared_out: Buffer,
    logits: Buffer,
    // The tower's rows.
    tower_h: Buffer,
    /// One slab of `padded` code-embedding rows per tower layer, host-written.
    embeddings: Buffer,
    concat: Buffer,
    tower_x: Buffer,
    tower_gate: Buffer,
    tower_up: Buffer,
    head_logits: Buffer,
    /// The tower's input: the last backbone row's hidden, valid until the
    /// next backbone call (the CPU runtime's `h`).
    h: []f32,
    scratch: []f32,
    shared_input: []f32,

    /// A plan for chunks of up to `rows` backbone rows, prefilled `chunk`
    /// rows per command buffer.
    pub fn init(alloc: std.mem.Allocator, backend: *Backend, m: *const runtime.Model, rows: usize, chunk: usize) !Plan {
        if (chunk == 0 or chunk > 4096) return error.InvalidShape;
        var state = try runtime.State.init(alloc, rows);
        errdefer state.deinit();
        var self: Plan = undefined;
        self.alloc = alloc;
        self.backend = backend;
        self.model = m;
        self.created = .empty;
        errdefer self.releaseBuffers();
        self.state = state;
        self.backbone_memory = try backend.wrap(state.backbone.memory);
        errdefer backend.unwrap(state.backbone.memory);
        self.tower_memory = try backend.wrap(state.tower.memory);
        errdefer backend.unwrap(state.tower.memory);
        self.scratch = try alloc.alloc(f32, 2 * hidden);
        errdefer alloc.free(self.scratch);
        self.shared_input = self.scratch[hidden..];
        self.scratch = self.scratch[0..hidden];

        const view = m.view;
        self.output_norm = try self.upload(m.output_norm);
        for (&self.layers, m.layers, m.binding.layers) |*c, constants, layer| {
            const gate_values = try view.vector(alloc, layer.moe.shared_router);
            defer alloc.free(gate_values);
            if (gate_values.len != hidden) return error.InvalidShape;
            c.* = .{
                .attention_norm = try self.upload(constants.attention_norm),
                .post_attention_norm = try self.upload(constants.post_attention_norm),
                .shared_gate = try self.upload(gate_values),
                .mixer = switch (constants.mixer) {
                    .full_attention => |attn| .{ .full_attention = try self.attentionConstants(attn.query_norm, attn.key_norm) },
                    .delta_net => |d| .{ .delta_net = .{ .convolution = try self.upload(d.convolution), .a = try self.upload(d.a), .time_bias = try self.upload(d.time_bias), .norm = try self.upload(d.norm) } },
                },
            };
        }
        for (&self.shared, m.shared) |*c, constants| c.* = try self.towerConstants(constants);
        for (&self.residual, m.residual) |*c, constants| c.* = try self.towerConstants(constants);
        self.rope_table = try self.create(rows * (model.rotary_dims / 2) * 8);
        try Backend.ropeTable(self.rope_table, rows, model.rotary_dims, model.rope_base, null);

        self.chunk = chunk;
        self.padded = Backend.matmulPadded(chunk);
        const n = self.padded;
        const slots = n * model.experts_used;
        inline for (.{ "x", "normalized", "projected", "normed", "tower_h", "tower_x", "shared_out" }) |field| @field(self, field) = try self.create(n * hidden * 4);
        self.last = try self.create(hidden * 4);
        inline for (.{ "q", "gate", "attended" }) |field| @field(self, field) = try self.create(n * model.query_width * 4);
        inline for (.{ "k", "v" }) |field| @field(self, field) = try self.create(n * model.kv_width * 4);
        inline for (.{ "mixed", "convolved" }) |field| @field(self, field) = try self.create(n * model.qkv_width * 4);
        self.z = try self.create(n * model.inner_width * 4);
        inline for (.{ "alpha", "beta" }) |field| @field(self, field) = try self.create(n * model.value_heads * 4);
        self.partials = try self.create(Backend.attentionVerifyPartials(Backend.verify_max_rows, model.heads, model.head_width) * 4);
        self.router = try self.create(n * model.experts * 4);
        self.indices = try self.create(slots * 4);
        self.route_weights = try self.create(slots * 4);
        self.lists = try self.create(Backend.expertListsLayout(slots, model.experts).words * 4);
        inline for (.{ "expert_gate", "expert_up" }) |field| @field(self, field) = try self.create(slots * model.expert_ffn * 4);
        self.expert_out = try self.create(slots * hidden * 4);
        inline for (.{ "shared_gate", "shared_up" }) |field| @field(self, field) = try self.create(n * model.expert_ffn * 4);
        self.logits = try self.create(model.output_rows * 4);
        self.embeddings = try self.create(tower_layers * n * hidden * 4);
        self.concat = try self.create(n * 2 * hidden * 4);
        inline for (.{ "tower_gate", "tower_up" }) |field| @field(self, field) = try self.create(n * model.tower_ffn * 4);
        self.head_logits = try self.create(model.codebook_size * 4);
        self.h = self.last.floats()[0..hidden];
        return self;
    }

    pub fn deinit(self: *Plan) void {
        self.alloc.free(self.scratch.ptr[0 .. 2 * hidden]);
        self.releaseBuffers();
        self.backend.unwrap(self.state.backbone.memory);
        self.backend.unwrap(self.state.tower.memory);
        self.state.deinit();
        self.* = undefined;
    }

    fn releaseBuffers(self: *Plan) void {
        for (self.created.items) |buffer| self.backend.release(buffer) catch {};
        self.created.deinit(self.alloc);
    }

    /// Clears the state for a fresh chunk of speech (`runtime.State.reset`).
    pub fn reset(self: *Plan) void {
        self.state.reset();
    }

    /// Device memory: the created buffers and the wrapped session.
    pub fn bytes(self: *const Plan) usize {
        var total: usize = self.state.bytes();
        for (self.created.items) |buffer| total += buffer.len;
        return total;
    }

    fn create(self: *Plan, len: usize) !Buffer {
        try self.created.ensureUnusedCapacity(self.alloc, 1);
        const buffer = try self.backend.create(len);
        self.created.appendAssumeCapacity(buffer);
        return buffer;
    }
    fn upload(self: *Plan, values: []const f32) !Buffer {
        if (values.len == 0) return error.InvalidShape;
        const buffer = try self.create(values.len * 4);
        @memcpy(buffer.floats(), values);
        return buffer;
    }
    fn attentionConstants(self: *Plan, query_norm: []const f32, key_norm: []const f32) !AttentionConstants {
        return .{ .query_norm = try self.upload(query_norm), .key_norm = try self.upload(key_norm) };
    }
    fn towerConstants(self: *Plan, c: anytype) !TowerConstants {
        return .{
            .enorm = try self.upload(c.enorm),
            .hnorm = try self.upload(c.hnorm),
            .final_norm = try self.upload(c.final_norm),
            .attention_norm = try self.upload(c.attention_norm),
            .ffn_norm = try self.upload(c.ffn_norm),
            .attention = try self.attentionConstants(c.attention.query_norm, c.attention.key_norm),
        };
    }

    /// Tensors are wrapped one at a time (the backend caches them by address).
    fn weight(self: *Plan, tensor: *const Tensor) !Weight {
        const matrix = try self.model.view.matrix(tensor);
        return .{ .buffer = try self.backend.wrap(matrix.bytes), .matrix = matrix };
    }
    fn expertWeight(self: *Plan, tensor: *const Tensor) !ExpertWeight {
        const matrix = try self.model.view.expertMatrix(tensor);
        return .{ .buffer = try self.backend.wrap(matrix.bytes), .tensor = matrix };
    }
    /// `count` rows through `tensor`: a matvec for one row, else the F32
    /// tile. The half-operand tiles move router logits past near-ties, and
    /// a 290-row prompt's hidden rows then drift by 39 % relative RMS; the
    /// F32 tile costs about a tenth more prefill (docs/models/drama3.md § On Metal).
    fn project(self: *Plan, tensor: *const Tensor, input: Buffer, in_stride: usize, output: Buffer, out_stride: usize, count: usize) !void {
        const w = try self.weight(tensor);
        if (count == 1) return self.backend.matvec(w.buffer, w.matrix, input, output);
        try self.backend.matmulKernel(.matmul, w.buffer, w.matrix, input, in_stride, output, out_stride, count);
    }
    fn region(memory: Buffer, s: *const session.Session, bytes_range: []const u8) Buffer {
        return memory.slice(s.offsetOf(bytes_range), bytes_range.len);
    }
    fn rowsOf(buffer: Buffer, width: usize, first: usize, count: usize) Buffer {
        return buffer.slice(first * width * 4, count * width * 4);
    }

    // --- the backbone ---------------------------------------------------

    /// `runtime.Runtime.prefill` on the plan's state: `frames` at the next
    /// backbone rows, `hidden_rows` (`frames.len × hidden`) receiving every
    /// row's post-norm hidden, `logits` (4,097) the last row's. The last
    /// row's hidden stays in `h`. The whole run is admitted before the first
    /// write.
    pub fn prefill(self: *Plan, frames: []const Frame, hidden_rows: ?[]f32, logits: ?[]f32) !void {
        const s = &self.state.backbone;
        if (frames.len == 0) return error.InvalidShape;
        if (hidden_rows) |rows| if (rows.len != frames.len * hidden) return error.InvalidShape;
        if (logits) |out| if (out.len != model.output_rows) return error.InvalidShape;
        if (s.status != .ready) return error.SessionNotReady;
        if (frames.len > s.capacity - s.position) return error.ContextFull;
        for (frames) |frame| try runtime.validFrame(frame);
        var offset: usize = 0;
        while (offset < frames.len) {
            const count = @min(self.chunk, frames.len - offset);
            const last = offset + count == frames.len;
            try self.backboneChunk(frames[offset..][0..count], if (hidden_rows) |rows| rows[offset * hidden ..][0 .. count * hidden] else null, if (last) logits else null);
            offset += count;
        }
    }

    /// One generated frame at the next backbone row; `prefill` of one.
    pub fn decode(self: *Plan, frame: Frame, logits: ?[]f32) !void {
        return self.prefill(&.{frame}, null, logits);
    }

    /// The 4,097 main-token logits of a post-norm hidden `h`.
    pub fn mainLogits(self: *Plan, h: []const f32, out: []f32) !void {
        if (h.len != hidden or out.len != model.output_rows) return error.InvalidShape;
        const staged = self.normed.floats()[0..hidden];
        @memcpy(staged, h);
        const b = self.backend;
        try b.begin();
        errdefer b.discard();
        try self.project(self.model.binding.output, self.normed, hidden, self.logits, model.output_rows, 1);
        try b.commit();
        try self.readLogits(out);
    }

    fn readLogits(self: *Plan, out: []f32) !void {
        @memcpy(out, self.logits.floats()[0..model.output_rows]);
        for (out) |x| if (!std.math.isFinite(x)) return error.NonFiniteResult;
    }

    fn backboneChunk(self: *Plan, frames: []const Frame, hidden_rows: ?[]f32, logits: ?[]f32) !void {
        const count = frames.len;
        const s = &self.state.backbone;
        const x = self.x.floats();
        for (frames, 0..) |frame, i| try self.model.embed(frame, x[i * hidden ..][0..hidden], self.scratch);
        try s.beginChunk(count);
        errdefer s.fail();
        const b = self.backend;
        try b.begin();
        errdefer b.discard();
        const norm: Backend.Norm = .{ .rows = count, .width = hidden, .in_stride = hidden, .out_stride = hidden, .eps = model.norm_eps };
        for (self.model.binding.layers, self.layers, 0..) |layer, c, il| {
            try b.rmsNorm(self.x, c.attention_norm, self.normalized, norm);
            switch (layer.mixer) {
                .full_attention => |attn| try self.attention(s, self.backbone_memory, il, attn, c.mixer.full_attention, count, .split_half),
                .delta_net => |d| try self.deltaNet(s, il, d, c.mixer.delta_net, count),
            }
            try b.addRmsNorm(self.x, self.projected, c.post_attention_norm, self.normalized, norm);
            try self.moe(layer.moe, c.shared_gate, count);
            try b.add(self.x, self.projected, count * hidden);
        }
        try b.rmsNorm(self.x, self.output_norm, self.normed, norm);
        try b.copy(self.last, rowsOf(self.normed, hidden, count - 1, 1), hidden);
        if (logits != null) try self.project(self.model.binding.output, self.last, hidden, self.logits, model.output_rows, 1);
        try b.commit();
        for (self.h) |v| if (!std.math.isFinite(v)) return error.NonFiniteResult;
        if (hidden_rows) |rows| @memcpy(rows, self.normed.floats()[0 .. count * hidden]);
        if (logits) |out| try self.readLogits(out);
        try s.commitChunk(count);
    }

    /// Gated full attention over `count` rows of `normalized` at the
    /// session's position, into `projected`: the backbone's (NeoX RoPE) and
    /// the tower's (interleaved RoPE) differ only in `pairing`.
    fn attention(self: *Plan, s: *const session.Session, memory: Buffer, il: usize, attn: model.FullAttention, c: AttentionConstants, count: usize, pairing: Backend.Pairing) !void {
        const b = self.backend;
        const position = s.position;
        const cache = s.layers[il].attention;
        try self.project(attn.query, self.normalized, hidden, self.q, model.query_width, count);
        try self.project(attn.gate, self.normalized, hidden, self.gate, model.query_width, count);
        try self.project(attn.key, self.normalized, hidden, self.k, model.kv_width, count);
        try self.project(attn.value, self.normalized, hidden, self.v, model.kv_width, count);
        const head: Backend.Norm = .{ .rows = count * model.heads, .width = model.head_width, .in_stride = model.head_width, .out_stride = model.head_width, .eps = model.norm_eps };
        try b.rmsNorm(self.q, c.query_norm, self.q, head);
        var key_head = head;
        key_head.rows = count * model.kv_heads;
        try b.rmsNorm(self.k, c.key_norm, self.k, key_head);
        try b.ropeRows(self.q, self.rope_table, model.heads, model.head_width, model.rotary_dims, position, count, model.query_width, pairing);
        try b.ropeRows(self.k, self.rope_table, model.kv_heads, model.head_width, model.rotary_dims, position, count, model.kv_width, pairing);
        try b.copy(region(memory, s, cache.keys.range(position, count)), self.k, count * model.kv_width);
        try b.copy(region(memory, s, cache.values.range(position, count)), self.v, count * model.kv_width);
        const keys = region(memory, s, cache.keys.range(0, position + count));
        const values = region(memory, s, cache.values.range(0, position + count));
        const shape: Backend.AttentionChunkShape = .{ .query_heads = model.heads, .kv_heads = model.kv_heads, .key_width = model.head_width, .value_width = model.head_width, .position = position, .count = count, .q_stride = model.query_width, .out_stride = model.query_width, .scale = 1.0 / @sqrt(@as(f32, model.head_width)) };
        if (count <= Backend.verify_max_rows) try b.attentionVerify(keys, values, self.q, self.partials, self.attended, shape) else try b.attentionChunk(keys, values, self.q, self.attended, shape);
        try b.sigmoidGate(self.attended, self.gate, count * model.heads, model.head_width, model.head_width, 0);
        try self.project(attn.output, self.attended, model.query_width, self.projected, hidden, count);
    }

    /// The gated delta rule over `count` rows of `normalized`, into
    /// `projected`. One row takes the sequential kernels, a chunk the
    /// chunkwise form; both read the value-head order from `Options`.
    fn deltaNet(self: *Plan, s: *const session.Session, il: usize, d: model.DeltaNet, c: DeltaConstants, count: usize) !void {
        const b = self.backend;
        const state = s.layers[il].recurrent;
        const qkv = model.qkv_width;
        const vh = model.value_heads;
        const grouped = self.model.options.value_order == .grouped;
        try self.project(d.qkv, self.normalized, hidden, self.mixed, qkv, count);
        try self.project(d.z, self.normalized, hidden, self.z, model.inner_width, count);
        try self.project(d.beta, self.normalized, hidden, self.beta, vh, count);
        try self.project(d.alpha, self.normalized, hidden, self.alpha, vh, count);
        const history = region(self.backbone_memory, s, std.mem.sliceAsBytes(state.history));
        const matrix = region(self.backbone_memory, s, std.mem.sliceAsBytes(state.matrix));
        const scale = 1.0 / @sqrt(@as(f32, model.state_width));
        if (count == 1) {
            try b.convolution(history, self.mixed, c.convolution, self.convolved, qkv, model.conv_kernel);
            try b.silu(self.convolved, qkv);
            try b.l2NormRows(self.convolved, 1, 2 * model.key_heads, model.state_width, model.state_width, qkv, model.norm_eps);
            try b.deltaGates(self.alpha, self.beta, c.a, c.time_bias, vh);
            try b.delta(matrix, self.convolved, self.alpha, self.beta, self.attended, .{ .qheads = model.key_heads, .vheads = vh, .keys = model.state_width, .values = model.state_width, .scale = scale, .grouped = grouped });
        } else {
            try b.convolutionRows(history, self.mixed, c.convolution, self.convolved, qkv, model.conv_kernel, count, qkv);
            try b.convolutionHistory(history, self.mixed, qkv, model.conv_kernel, count, qkv);
            try b.silu(self.convolved, count * qkv);
            try b.l2NormRows(self.convolved, count, 2 * model.key_heads, model.state_width, model.state_width, qkv, model.norm_eps);
            try b.deltaGatesRows(self.alpha, self.beta, c.a, c.time_bias, vh, count);
            try b.deltaChunk(matrix, self.convolved, self.alpha, self.beta, self.attended, .{ .qheads = model.key_heads, .vheads = vh, .keys = model.state_width, .values = model.state_width, .count = count, .in_stride = qkv, .gate_stride = vh, .out_stride = model.inner_width, .scale = scale, .grouped = grouped });
        }
        // GatedRMSNorm: weight ⊙ rmsnorm(o) ⊙ silu(z), per value head.
        try b.rmsNorm(self.attended, c.norm, self.attended, .{ .rows = count * vh, .width = model.state_width, .in_stride = model.state_width, .out_stride = model.state_width, .eps = model.norm_eps, .silu_multiplier = .{ .buffer = self.z, .stride = model.state_width } });
        try self.project(d.output, self.attended, model.inner_width, self.projected, hidden, count);
    }

    /// The routed experts and the gated shared expert over `count` rows of
    /// `normalized`, into `projected`: the gathered matvecs for one row,
    /// the expert lists and gathered tiles for a chunk.
    fn moe(self: *Plan, e: model.Moe, shared_gate: Buffer, count: usize) !void {
        const b = self.backend;
        const k = model.experts_used;
        const ff = model.expert_ffn;
        try self.project(e.router, self.normalized, hidden, self.router, model.experts, count);
        try b.route(self.router, model.experts, k, count, model.experts, self.indices, self.route_weights);
        const gate = try self.expertWeight(e.gate);
        const up = try self.expertWeight(e.up);
        const down = try self.expertWeight(e.down);
        if (count == 1) {
            try b.matvecExperts(gate.buffer, gate.tensor, self.indices, k, self.normalized, 0, self.expert_gate, ff);
            try b.matvecExperts(up.buffer, up.tensor, self.indices, k, self.normalized, 0, self.expert_up, ff);
            try b.siluMul(self.expert_gate, self.expert_up, k * ff);
            try b.matvecExperts(down.buffer, down.tensor, self.indices, k, self.expert_gate, ff, self.expert_out, hidden);
        } else {
            try b.expertLists(self.indices, count, k, model.experts, self.lists);
            try b.matmulExperts(gate.buffer, gate.tensor, self.lists, count, k, self.normalized, hidden, k, self.expert_gate, ff);
            try b.matmulExperts(up.buffer, up.tensor, self.lists, count, k, self.normalized, hidden, k, self.expert_up, ff);
            try b.siluMul(self.expert_gate, self.expert_up, count * k * ff);
            try b.matmulExperts(down.buffer, down.tensor, self.lists, count, k, self.expert_gate, ff, 1, self.expert_out, hidden);
        }
        try b.combineExperts(self.expert_out, self.route_weights, self.indices, null, self.projected, .{ .columns = hidden, .slots = k, .rows = count, .experts = model.experts, .in_stride = hidden, .out_stride = hidden });
        try self.project(e.shared_gate, self.normalized, hidden, self.shared_gate, ff, count);
        try self.project(e.shared_up, self.normalized, hidden, self.shared_up, ff, count);
        try b.siluMul(self.shared_gate, self.shared_up, count * ff);
        try self.project(e.shared_down, self.shared_gate, ff, self.shared_out, hidden, count);
        try b.sharedExpertAdd(self.projected, self.shared_out, shared_gate, self.normalized, hidden, count, hidden, hidden);
    }

    // --- the tower ------------------------------------------------------

    /// `runtime.Runtime.towerPrefill` on the plan's state: teacher-forces
    /// the tower over the next `hidden_rows.len / hidden` rows, row i
    /// labelled with `frames[i + 1]`'s codes and enriched by `frames[i]`'s.
    pub fn towerPrefill(self: *Plan, hidden_rows: []const f32, frames: []const Frame) !void {
        const s = &self.state.tower;
        const rows = hidden_rows.len / hidden;
        if (rows == 0 or hidden_rows.len != rows * hidden or frames.len != rows + 1) return error.InvalidShape;
        if (s.position + rows > self.state.backbone.position) return error.InvalidShape;
        if (s.status != .ready) return error.SessionNotReady;
        for (frames) |frame| try runtime.validFrame(frame);
        var offset: usize = 0;
        while (offset < rows) {
            const count = @min(self.chunk, rows - offset);
            @memcpy(self.tower_h.floats()[0 .. count * hidden], hidden_rows[offset * hidden ..][0 .. count * hidden]);
            for (0..count) |i| {
                const input = frames[offset + i];
                const label = frames[offset + i + 1];
                try self.sharedInput(input);
                for (0..tower_layers) |il| try self.towerEmbedding(il, i, label, if (il == 0) self.shared_input else null);
            }
            try s.beginChunk(count);
            errdefer s.fail();
            const b = self.backend;
            try b.begin();
            errdefer b.discard();
            for (0..tower_layers) |il| try self.towerLayer(il, count);
            try b.commit();
            for (self.tower_h.floats()[0 .. count * hidden]) |v| if (!std.math.isFinite(v)) return error.NonFiniteResult;
            try s.commitChunk(count);
            offset += count;
        }
    }

    /// `runtime.Runtime.step`: the tower at the backbone's last row, whose
    /// hidden is `h` (usually `self.h`) and input frame `input`; `main`
    /// gives code 0, then `pick` chooses codes 1–9 from each head's logits,
    /// one commit per head. Returns the new frame.
    pub fn step(self: *Plan, h: []const f32, input: Frame, main: u32, pick: Pick) !Frame {
        const s = &self.state.tower;
        if (h.len != hidden) return error.InvalidShape;
        if (s.position + 1 != self.state.backbone.position) return error.InvalidShape;
        try runtime.validFrame(input);
        if (main >= model.vocabulary) return error.InvalidTokenId;
        var codes: Frame = @splat(0);
        codes[0] = main;
        if (model.isSemantic(main)) codes[1] = main - model.semantic_start;
        @memcpy(self.tower_h.floats()[0..hidden], h);
        try self.sharedInput(input);
        for (0..model.shared_layers + 1) |il| try self.towerEmbedding(il, 0, codes, if (il == 0) self.shared_input else null);
        try s.begin();
        errdefer s.fail();
        const b = self.backend;
        errdefer b.discard();
        try b.begin();
        for (0..model.shared_layers) |il| try self.towerLayer(il, 1);
        for (self.model.binding.heads, 0..) |head, j| {
            const il = model.shared_layers + j;
            if (j > 0) {
                try self.towerEmbedding(il, 0, codes, null);
                try b.begin();
            }
            try self.towerLayer(il, 1);
            try self.project(head, self.tower_h, hidden, self.head_logits, model.codebook_size, 1);
            try b.commit();
            const logits = self.head_logits.floats()[0..model.codebook_size];
            for (logits) |x| if (!std.math.isFinite(x)) return error.NonFiniteResult;
            const code = try pick.pick(pick.context, j + 1, logits);
            if (code >= model.codebook_size) return error.InvalidCode;
            codes[2 + j] = code;
        }
        try s.commit();
        return codes;
    }

    /// The sum of `input`'s codebook rows for codes 2–9 into `shared_input`
    /// (zero for a text row), as the reference sums them.
    fn sharedInput(self: *Plan, input: Frame) !void {
        const m = self.model;
        @memset(self.shared_input, 0);
        if (!model.isSemantic(input[0])) return;
        for (2..model.codebooks) |k| {
            try m.view.row(m.binding.codebook_embedding, (k - 1) * model.codebook_size + input[1 + k], self.scratch);
            for (self.shared_input, self.scratch) |*x, e| x.* += e;
        }
    }

    /// Tower layer `il`'s code embedding of `label` into row `row` of its
    /// slab: the layer's code (code 0 for the shared layers, code j for
    /// residual layer j), zero on a non-semantic row, plus `extra`.
    fn towerEmbedding(self: *Plan, il: usize, row: usize, label: Frame, extra: ?[]const f32) !void {
        const layer = if (il < model.shared_layers) self.model.binding.shared[il] else self.model.binding.residual[il - model.shared_layers];
        const code = label[1 + (if (il < model.shared_layers) 0 else il - model.shared_layers)];
        const out = self.embeddings.floats()[(il * self.padded + row) * hidden ..][0..hidden];
        if (model.isSemantic(label[0])) try self.model.view.row(layer.embedding, @min(code, model.codebook_size - 1), out) else @memset(out, 0);
        if (extra) |values| for (out, values) |*e, v| {
            e.* += v;
        };
    }

    /// `TowerLayer.forward` over `count` rows of `tower_h` in place, at
    /// tower cache layer `il`, reading its embedding slab.
    fn towerLayer(self: *Plan, il: usize, count: usize) !void {
        const b = self.backend;
        const layer = if (il < model.shared_layers) self.model.binding.shared[il] else self.model.binding.residual[il - model.shared_layers];
        const c = if (il < model.shared_layers) self.shared[il] else self.residual[il - model.shared_layers];
        const norm: Backend.Norm = .{ .rows = count, .width = hidden, .in_stride = hidden, .out_stride = hidden, .eps = model.norm_eps };
        var into_concat = norm;
        into_concat.out_stride = 2 * hidden;
        const slab = self.embeddings.slice(il * self.padded * hidden * 4, self.padded * hidden * 4);
        try b.rmsNorm(slab, c.enorm, self.concat, into_concat);
        try b.rmsNorm(self.tower_h, c.hnorm, self.concat.slice(hidden * 4, self.concat.len - hidden * 4), into_concat);
        try self.project(layer.eh_proj, self.concat, 2 * hidden, self.tower_x, hidden, count);
        try b.rmsNorm(self.tower_x, c.attention_norm, self.normalized, norm);
        try self.attention(&self.state.tower, self.tower_memory, il, layer.attention, c.attention, count, .adjacent);
        try b.addRmsNorm(self.tower_x, self.projected, c.ffn_norm, self.normalized, norm);
        try self.project(layer.ffn_gate, self.normalized, hidden, self.tower_gate, model.tower_ffn, count);
        try self.project(layer.ffn_up, self.normalized, hidden, self.tower_up, model.tower_ffn, count);
        try b.siluMul(self.tower_gate, self.tower_up, count * model.tower_ffn);
        try self.project(layer.ffn_down, self.tower_gate, model.tower_ffn, self.projected, hidden, count);
        try b.add(self.tower_x, self.projected, count * hidden);
        try b.rmsNorm(self.tower_x, c.final_norm, self.tower_h, norm);
    }
};

test {
    std.testing.refAllDecls(Plan);
}
