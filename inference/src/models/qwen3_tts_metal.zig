//! Qwen3-TTS's talker and code predictor on Metal: the CPU reference's
//! forward (qwen3_tts_runtime.zig) with the checkpoint's BF16 matrices
//! wrapped in place (the generic matvec and matmul read them; nothing is
//! widened), norms and biases decoded to F32 buffers, and embedding rows
//! decoded on the host per code. The talker's cache is F32 or F16
//! (`Precision`), grown on demand; the predictor's 16 rows are F32.
//! Sampling stays on the host: a talker step and each of a frame's 15
//! predictor steps end in a commit whose logits the CPU sampler reads
//! (docs/models/qwen3-tts.md § On Metal). The plan borrows its `Backend`
//! and the checkpoint, which must outlive it.
const std = @import("std");
const metal = @import("../backends/metal/root.zig");
const cpu = @import("../backends/cpu/root.zig");
const safetensors = @import("../formats/safetensors.zig");
const tts = @import("qwen3_tts.zig");
const runtime = @import("qwen3_tts_runtime.zig");

const Allocator = std.mem.Allocator;
const Backend = metal.Backend;
const Buffer = metal.Buffer;
pub const Precision = metal.Precision;
const groups = tts.groups;
pub const Frame = runtime.Frame;

/// Prompt rows one prefill command buffer runs.
pub const chunk_rows = 128;
/// Talker rows the cache may grow to: a prompt and `max_new_tokens` frames.
pub const max_capacity = 16384;
/// Rows per SIMD group of the BF16 matvec.
const matvec_rows = 2;

const Weight = struct { buffer: Buffer, matrix: cpu.Matrix };
const Table = struct { buffer: Buffer, rows: usize };

/// The predictor's sampling options, as the device sampler takes them.
pub const SampleOptions = struct { temperature: f32, top_k: usize, top_p: f32 };
const Layer = struct {
    input_norm: Buffer,
    post_norm: Buffer,
    q_norm: Buffer,
    k_norm: Buffer,
    q: Weight,
    k: Weight,
    v: Weight,
    o: Weight,
    gate: Weight,
    up: Weight,
    down: Weight,
};
/// One stack's keys (after RoPE) and values per layer, `capacity` rows.
const Cache = struct {
    keys: []Buffer,
    values: []Buffer,
    capacity: usize = 0,
    precision: Precision,
    rope: ?Buffer = null,
};

pub const Plan = struct {
    gpa: Allocator,
    backend: *Backend,
    config: *const tts.Config,
    buffers: std.ArrayList(Buffer) = .empty,
    text_embedding: safetensors.Ref,
    fc1: Weight,
    fc1_b: Buffer,
    fc2: Weight,
    fc2_b: Buffer,
    codec_embedding: safetensors.Ref,
    talker: []Layer,
    norm: Buffer,
    codec_head: Weight,
    projection: ?Weight,
    projection_b: ?Buffer,
    predictor: []Layer,
    embeddings: [groups - 1]safetensors.Ref,
    heads: [groups - 1]Weight,
    predictor_norm: Buffer,
    talker_cache: Cache,
    predictor_cache: Cache,
    /// Talker rows run so far: the next row's position.
    rows: usize = 0,
    // Activations, `chunk_rows` rows (padded for the tile) of the widest stack.
    x: Buffer,
    h: Buffer,
    q: Buffer,
    q_half: Buffer,
    k: Buffer,
    v: Buffer,
    att: Buffer,
    o: Buffer,
    gate: Buffer,
    up: Buffer,
    partials: Buffer,
    /// The talker's last post-norm row, its logits, the predictor's logits.
    hidden: Buffer,
    logits: Buffer,
    predictor_logits: Buffer,
    /// The predictor's input row at talker width, before the projection.
    input: Buffer,
    text_in: Buffer,
    text_wide: Buffer,
    text_out: Buffer,
    /// Sees the talker's residual rows after each layer and its post-norm
    /// rows (a commit per layer: traces only).
    observer: ?runtime.Observer = null,
    /// The BF16 tables of the device-side frame: the talker's codec table,
    /// then group g's predictor table at g − 1 (wrapped in place).
    codec_table: Table,
    tables: [groups - 1]Table,
    /// A device frame's codes (u32), its draws, every group's logits, the text track.
    codes: Buffer,
    draws: Buffer,
    frame_logits: Buffer,
    pad: Buffer,
    /// With `split_timing`, `frameOnDevice` commits between the predictor and
    /// the talker and records each one's device time here.
    split_timing: bool = false,
    predictor_gpu: f64 = 0,
    talker_gpu: f64 = 0,
    /// Groups where the host sampler overruled the device's draw.
    disagreements: usize = 0,

    /// Wraps `binding`'s matrices and uploads its vectors; `kv` is the
    /// talker cache's precision (F32 or F16).
    pub fn init(gpa: Allocator, backend: *Backend, binding: *const tts.Binding, config: *const tts.Config, kv: Precision) !Plan {
        if (kv == .q8_0) return error.UnsupportedConfiguration;
        for ([_]tts.Stack{ config.talker, config.predictor }) |s| {
            if (s.heads * s.head_dim > 4096 or s.head_dim > 512 or s.heads % s.kv_heads != 0) return error.UnsupportedConfiguration;
        }
        var self: Plan = undefined;
        self.gpa = gpa;
        self.backend = backend;
        self.config = config;
        self.buffers = .empty;
        self.rows = 0;
        self.observer = null;
        self.split_timing = false;
        self.predictor_gpu = 0;
        self.talker_gpu = 0;
        self.disagreements = 0;
        self.talker = &.{};
        self.predictor = &.{};
        self.talker_cache = .{ .keys = &.{}, .values = &.{}, .precision = kv };
        self.predictor_cache = .{ .keys = &.{}, .values = &.{}, .precision = .f32 };
        errdefer self.deinit();
        self.text_embedding = binding.text_embedding;
        self.fc1 = try self.weight(binding.text_fc1.weight);
        self.fc1_b = try self.vector(binding.text_fc1.bias.?);
        self.fc2 = try self.weight(binding.text_fc2.weight);
        self.fc2_b = try self.vector(binding.text_fc2.bias.?);
        self.codec_embedding = binding.codec_embedding;
        self.talker = try self.layers(binding.talker);
        self.norm = try self.vector(binding.norm);
        self.codec_head = try self.weight(binding.codec_head);
        self.projection = if (binding.projection) |p| try self.weight(p.weight) else null;
        self.projection_b = if (binding.projection) |p| try self.vector(p.bias.?) else null;
        self.predictor = try self.layers(binding.predictor);
        self.embeddings = binding.embeddings;
        for (&self.heads, binding.heads) |*h, ref| h.* = try self.weight(ref);
        self.predictor_norm = try self.vector(binding.predictor_norm);
        self.talker_cache.keys = try gpa.alloc(Buffer, config.talker.layers);
        self.talker_cache.values = try gpa.alloc(Buffer, config.talker.layers);
        self.predictor_cache.keys = try gpa.alloc(Buffer, config.predictor.layers);
        self.predictor_cache.values = try gpa.alloc(Buffer, config.predictor.layers);
        try self.grow(&self.predictor_cache, config.predictor, groups);

        const t = Backend.matmulPadded(chunk_rows);
        const w = @max(config.talker.hidden, config.predictor.hidden);
        const qw = @max(config.talker.heads * config.talker.head_dim, config.predictor.heads * config.predictor.head_dim);
        const kvw = @max(config.talker.kv_heads * config.talker.head_dim, config.predictor.kv_heads * config.predictor.head_dim);
        const inter = @max(config.talker.intermediate, config.predictor.intermediate);
        self.x = try self.floats(t * w);
        self.h = try self.floats(t * w);
        self.q = try self.floats(t * qw);
        self.q_half = try self.floats(t * qw / 2);
        self.k = try self.floats(t * kvw);
        self.v = try self.floats(t * kvw);
        self.att = try self.floats(t * qw);
        self.o = try self.floats(t * w);
        self.gate = try self.floats(t * inter);
        self.up = try self.floats(t * inter);
        const heads = @max(config.talker.heads, config.predictor.heads);
        const head_dim = @max(config.talker.head_dim, config.predictor.head_dim);
        self.partials = try self.floats(Backend.attentionVerifyPartials(Backend.verify_max_rows, heads, head_dim));
        self.hidden = try self.floats(config.talker.hidden);
        self.logits = try self.floats(config.codec_vocab);
        self.predictor_logits = try self.floats(tts.codebook_size);
        self.input = try self.floats(config.talker.hidden);
        self.text_in = try self.floats(t * config.text_hidden);
        self.text_wide = try self.floats(t * config.text_hidden);
        self.text_out = try self.floats(t * config.talker.hidden);
        self.codec_table = try self.wrapTable(binding.codec_embedding);
        for (&self.tables, binding.embeddings) |*tab, ref| tab.* = try self.wrapTable(ref);
        self.codes = try self.floats(groups);
        self.draws = try self.floats(groups);
        self.frame_logits = try self.floats((groups - 1) * tts.codebook_size);
        self.pad = try self.floats(config.talker.hidden);
        return self;
    }

    pub fn deinit(self: *Plan) void {
        for (self.buffers.items) |b| self.backend.release(b) catch {};
        self.buffers.deinit(self.gpa);
        for ([_]*Cache{ &self.talker_cache, &self.predictor_cache }) |c| {
            self.releaseCache(c);
            self.gpa.free(c.keys);
            self.gpa.free(c.values);
        }
        self.gpa.free(self.talker);
        self.gpa.free(self.predictor);
        self.* = undefined;
    }

    fn floats(self: *Plan, count: usize) !Buffer {
        try self.buffers.ensureUnusedCapacity(self.gpa, 1);
        const b = try self.backend.create(count * 4);
        self.buffers.appendAssumeCapacity(b);
        return b;
    }
    fn vector(self: *Plan, ref: safetensors.Ref) !Buffer {
        const n: usize = @intCast(ref.tensor.elements);
        const b = try self.floats(n);
        try ref.decode(0, b.floats()[0..n]);
        return b;
    }
    fn weight(self: *Plan, ref: safetensors.Ref) !Weight {
        const encoding: u32 = switch (ref.tensor.dtype) {
            .f32 => 0,
            .f16 => 1,
            .bf16 => 30,
            else => return error.UnsupportedDtype,
        };
        const shape = ref.tensor.shape;
        const m: cpu.Matrix = .{ .encoding = encoding, .rows = @intCast(shape[0]), .columns = @intCast(shape[1]), .bytes = ref.bytes };
        // The matmul tile's constraints; every published matrix meets them.
        if (m.rows % 8 != 0 or m.columns % 64 != 0) return error.UnsupportedConfiguration;
        return .{ .buffer = try self.backend.wrap(ref.bytes), .matrix = m };
    }
    fn wrapTable(self: *Plan, ref: safetensors.Ref) !Table {
        if (ref.tensor.dtype != .bf16 or ref.tensor.shape.len != 2) return error.UnsupportedDtype;
        return .{ .buffer = try self.backend.wrap(ref.bytes), .rows = @intCast(ref.tensor.shape[0]) };
    }
    fn layers(self: *Plan, refs: []const tts.Layer) ![]Layer {
        const out = try self.gpa.alloc(Layer, refs.len);
        errdefer self.gpa.free(out);
        for (out, refs) |*l, r| l.* = .{
            .input_norm = try self.vector(r.input_norm),
            .post_norm = try self.vector(r.post_norm),
            .q_norm = try self.vector(r.q_norm),
            .k_norm = try self.vector(r.k_norm),
            .q = try self.weight(r.q),
            .k = try self.weight(r.k),
            .v = try self.weight(r.v),
            .o = try self.weight(r.o),
            .gate = try self.weight(r.gate),
            .up = try self.weight(r.up),
            .down = try self.weight(r.down),
        };
        return out;
    }

    fn releaseCache(self: *Plan, c: *Cache) void {
        if (c.capacity == 0) return;
        for (c.keys, c.values) |k, v| {
            self.backend.release(k) catch {};
            self.backend.release(v) catch {};
        }
        if (c.rope) |r| self.backend.release(r) catch {};
        c.rope = null;
        c.capacity = 0;
    }

    /// Grows `c` to hold at least `rows` rows, keeping the rows written so
    /// far (a host copy: nothing is in flight between commits).
    fn grow(self: *Plan, c: *Cache, s: tts.Stack, rows: usize) !void {
        if (rows <= c.capacity) return;
        if (rows > max_capacity) return error.ContextFull;
        const capacity = @min(max_capacity, @max(rows, 2 * c.capacity, 256));
        const row_bytes = c.precision.bytes(s.kv_heads * s.head_dim);
        const keys = try self.gpa.alloc(Buffer, c.keys.len);
        defer self.gpa.free(keys);
        const values = try self.gpa.alloc(Buffer, c.values.len);
        defer self.gpa.free(values);
        var made: usize = 0;
        errdefer for (keys[0..made], values[0..made]) |k, v| {
            self.backend.release(k) catch {};
            self.backend.release(v) catch {};
        };
        const rope = try self.backend.create(capacity * s.head_dim * 4);
        errdefer self.backend.release(rope) catch {};
        try Backend.ropeTable(rope, capacity, s.head_dim, s.rope_theta, null);
        for (keys, values, 0..) |*k, *v, i| {
            k.* = try self.backend.create(capacity * row_bytes);
            v.* = self.backend.create(capacity * row_bytes) catch |err| {
                self.backend.release(k.*) catch {};
                return err;
            };
            made += 1;
            if (c.capacity > 0) {
                const used = c.capacity * row_bytes;
                @memcpy(k.host[0..used], c.keys[i].host[0..used]);
                @memcpy(v.host[0..used], c.values[i].host[0..used]);
            }
        }
        self.releaseCache(c);
        @memcpy(c.keys, keys);
        @memcpy(c.values, values);
        c.rope = rope;
        c.capacity = capacity;
    }

    /// The talker's input rows for `prompt` into `out` (`[rows.len][width]`):
    /// the projected text row plus the codec row, as `Model.embed`.
    pub fn embed(self: *Plan, prompt: []const tts.Row, out: []f32) !void {
        const c = self.config;
        const w = c.talker.hidden;
        const th = c.text_hidden;
        if (out.len != prompt.len * w) return error.InvalidShape;
        var at: usize = 0;
        while (at < prompt.len) {
            // A piece of at most `chunk_rows` rows carrying text, projected in one command buffer.
            var count: usize = 0;
            var end = at;
            const tin = self.text_in.floats();
            while (end < prompt.len and count < chunk_rows) : (end += 1) if (prompt[end].text) |id| {
                if (id >= c.text_vocab) return error.InvalidTokenId;
                try self.text_embedding.decode(@as(u64, id) * th, tin[count * th ..][0..th]);
                count += 1;
            };
            if (count > 0) try self.project(count);
            const projected = self.text_out.floats();
            var t: usize = 0;
            for (prompt[at..end], at..) |row, i| {
                const o = out[i * w ..][0..w];
                if (row.text != null) {
                    @memcpy(o, projected[t * w ..][0..w]);
                    t += 1;
                } else @memset(o, 0);
                if (row.codec) |id| {
                    if (id >= c.codec_vocab) return error.InvalidCode;
                    try self.addRow(self.codec_embedding, id, o);
                }
            }
            at = end;
        }
    }

    /// `text_out = fc2 · silu(fc1 · text_in + b1) + b2` over `count` rows.
    fn project(self: *Plan, count: usize) !void {
        const b = self.backend;
        const c = self.config;
        try b.begin();
        errdefer b.discard();
        try self.mm(self.fc1, self.text_in, self.text_wide, count);
        try b.addBiasRows(self.text_wide, self.fc1_b, c.text_hidden, count, c.text_hidden);
        try b.silu(self.text_wide, count * c.text_hidden);
        try self.mm(self.fc2, self.text_wide, self.text_out, count);
        try b.addBiasRows(self.text_out, self.fc2_b, c.talker.hidden, count, c.talker.hidden);
        try b.commit();
    }

    /// `out += table[id]` (a BF16 row decoded on the host).
    fn addRow(self: *Plan, table: safetensors.Ref, id: usize, out: []f32) !void {
        _ = self;
        var row: [4096]f32 = undefined;
        const r = row[0..out.len];
        try table.decode(@as(u64, id) * out.len, r);
        for (out, r) |*a, x| a.* += x;
    }

    /// The next talker row (§ (e)), as `Model.nextRow`: the frame's 16 code
    /// embeddings summed in order, then the text track `text`.
    pub fn nextRow(self: *Plan, codes: Frame, text: []const f32, out: []f32) !void {
        const w = self.config.talker.hidden;
        try self.codec_embedding.decode(@as(u64, codes[0]) * w, out);
        for (self.embeddings, codes[1..]) |table, code| try self.addRow(table, code, out);
        for (out, text) |*a, b| a.* += b;
    }

    /// Runs the prompt's input rows `x` (`[n][width]`) from position 0; the
    /// first frame's logits and hidden row are ready after.
    pub fn prefill(self: *Plan, x: []const f32, n: usize) !void {
        const c = self.config;
        const w = c.talker.hidden;
        if (n == 0 or x.len != n * w) return error.InvalidShape;
        try self.grow(&self.talker_cache, c.talker, n);
        self.rows = 0;
        var at: usize = 0;
        while (at < n) {
            const m = @min(chunk_rows, n - at);
            @memcpy(self.x.floats()[0 .. m * w], x[at * w ..][0 .. m * w]);
            try self.talkerRows(m, at + m == n);
            at += m;
        }
    }

    /// One talker row (`[width]`) at the next position; the next logits and
    /// hidden row are ready after.
    pub fn step(self: *Plan, row: []const f32) !void {
        const w = self.config.talker.hidden;
        if (row.len != w) return error.InvalidShape;
        try self.grow(&self.talker_cache, self.config.talker, self.rows + 1);
        @memcpy(self.x.floats()[0..w], row);
        try self.talkerRows(1, true);
    }

    /// The talker's last group-0 logits (`codec_vocab`), valid after `prefill` or `step`.
    pub fn talkerLogits(self: *const Plan) []const f32 {
        return self.logits.floats()[0..self.config.codec_vocab];
    }
    /// The talker's last post-norm row, valid after `prefill` or `step`.
    pub fn hiddenRow(self: *const Plan) []const f32 {
        return self.hidden.floats()[0..self.config.talker.hidden];
    }

    /// `m` rows in `x` through the talker at `rows`; with `head`, the last
    /// row's post-norm state into `hidden` and its logits.
    fn talkerRows(self: *Plan, m: usize, head: bool) !void {
        const b = self.backend;
        const c = self.config;
        const s = c.talker;
        const w = s.hidden;
        try b.begin();
        errdefer b.discard();
        try self.stack(self.talker, s, &self.talker_cache, m, self.rows, self.observer);
        try b.rmsNorm(self.x, self.norm, self.h, .{ .rows = m, .width = w, .in_stride = w, .out_stride = w, .eps = s.rms_eps });
        if (head) {
            try b.copy(self.hidden, self.h.slice((m - 1) * w * 4, w * 4), w);
            try self.mv(self.codec_head, self.hidden, self.logits);
        }
        try b.commit();
        if (self.observer) |o| o.record(o.context, null, self.h.floats()[0 .. m * w]);
        self.rows += m;
    }

    /// Codes 1–15 of the frame whose group-0 code `c0` the last talker row
    /// gave (§ (d)), a command buffer per group; `pick` sees each group's
    /// logits in turn.
    pub fn frame(self: *Plan, c0: u16, pick: runtime.Pick) !Frame {
        if (c0 >= tts.codebook_size) return error.InvalidCode;
        var codes: Frame = undefined;
        codes[0] = c0;
        try self.groupsFrom(&codes, 1, pick);
        return codes;
    }

    /// Groups `first`…15 of `codes` (the earlier ones set, the predictor's
    /// cache holding their rows), a command buffer per group.
    fn groupsFrom(self: *Plan, codes: *Frame, first: usize, pick: runtime.Pick) !void {
        const c = self.config;
        const b = self.backend;
        const w = c.talker.hidden;
        for (first..groups) |g| {
            // The input row: the talker's table at c0, then group g − 1's table at its code.
            if (g == 1)
                try self.codec_embedding.decode(@as(u64, codes[0]) * w, self.input.floats()[0..w])
            else
                try self.embeddings[g - 2].decode(@as(u64, codes[g - 1]) * w, self.input.floats()[0..w]);
            try b.begin();
            errdefer b.discard();
            if (g == 1) try self.predictorRow(self.hidden, 0);
            try self.predictorRow(self.input, g);
            const p = c.predictor;
            try b.rmsNorm(self.x, self.predictor_norm, self.h, .{ .rows = 1, .width = p.hidden, .in_stride = p.hidden, .out_stride = p.hidden, .eps = p.rms_eps });
            try self.mv(self.heads[g - 1], self.h, self.predictor_logits);
            try b.commit();
            const code = try pick.pick(pick.context, g, self.predictor_logits.floats()[0..tts.codebook_size]);
            if (code >= tts.codebook_size) return error.InvalidCode;
            codes[g] = code;
        }
    }

    /// The whole frame on the device in one command buffer: groups 1–15
    /// sampled by the GPU with `draws` (the host sampler's next 15 uniforms,
    /// `options` its predictor options), the next talker row (§ (e), text
    /// track `pad`), and the talker's step, whose logits are ready after.
    /// Then `pick` (the host sampler) decides every group again from the
    /// same logits; where it disagrees with the device (a draw within F32
    /// rounding of a boundary), its code stands and the rest of the frame
    /// and the talker step are run again a group at a time. So the codes are
    /// always the host sampler's.
    pub fn frameOnDevice(self: *Plan, c0: u16, draws: *const [groups - 1]f64, options: SampleOptions, pad: []const f32, pick: runtime.Pick) !Frame {
        const c = self.config;
        const b = self.backend;
        const w = c.talker.hidden;
        const p = c.predictor;
        const size = tts.codebook_size;
        if (c0 >= size) return error.InvalidCode;
        if (pad.len != w) return error.InvalidShape;
        try self.grow(&self.talker_cache, c.talker, self.rows + 1);
        const device_codes: []u32 = @as([*]u32, @ptrCast(@alignCast(self.codes.host)))[0..groups];
        device_codes[0] = c0;
        for (self.draws.floats()[0 .. groups - 1], draws) |*d, x| d.* = @floatCast(x);
        @memcpy(self.pad.floats()[0..w], pad);
        const split = self.split_timing;
        const gpu_start = b.gpuSeconds();
        try b.begin();
        {
            errdefer b.discard();
            try self.predictorRow(self.hidden, 0);
            for (1..groups) |g| {
                const source = if (g == 1) self.codec_table else self.tables[g - 2];
                try b.ttsEmbedRow(source.buffer, source.rows, w, self.codes, g - 1, self.input, false);
                try self.predictorRow(self.input, g);
                try b.rmsNorm(self.x, self.predictor_norm, self.h, .{ .rows = 1, .width = p.hidden, .in_stride = p.hidden, .out_stride = p.hidden, .eps = p.rms_eps });
                const logits = self.frame_logits.slice((g - 1) * size * 4, size * 4);
                try self.mv(self.heads[g - 1], self.h, logits);
                try b.ttsSample(logits, size, self.draws, g - 1, self.codes, g, options.top_k, options.top_p, options.temperature);
            }
            if (split) {
                try b.commit();
                self.predictor_gpu = b.gpuSeconds() - gpu_start;
                try b.begin();
            }
            // The next talker row: the 16 code rows in order, then the text track.
            try b.ttsEmbedRow(self.codec_table.buffer, self.codec_table.rows, w, self.codes, 0, self.x, false);
            for (self.tables, 1..) |source, g| try b.ttsEmbedRow(source.buffer, source.rows, w, self.codes, g, self.x, true);
            try b.add(self.x, self.pad, w);
            try self.talkerLayers();
        }
        const talker_start = b.gpuSeconds();
        try b.commit();
        if (split) self.talker_gpu = b.gpuSeconds() - talker_start;
        self.rows += 1;

        var codes: Frame = undefined;
        codes[0] = c0;
        for (codes[1..], device_codes[1..]) |*code, d| code.* = @intCast(d);
        for (1..groups) |g| {
            const code = try pick.pick(pick.context, g, self.frame_logits.floats()[(g - 1) * size ..][0..size]);
            if (code >= size) return error.InvalidCode;
            if (code == codes[g]) continue;
            // The device's draw fell the other side of a boundary: redo from here.
            self.disagreements += 1;
            codes[g] = code;
            self.rows -= 1;
            try self.groupsFrom(&codes, g + 1, pick);
            const row = self.text_out.floats()[0..w];
            try self.nextRow(codes, pad, row);
            try self.step(row);
            break;
        }
        return codes;
    }

    /// The talker over the row in `x` at `rows`: its post-norm row into
    /// `hidden` and its logits (recorded, not committed).
    fn talkerLayers(self: *Plan) !void {
        const b = self.backend;
        const s = self.config.talker;
        const w = s.hidden;
        try self.stack(self.talker, s, &self.talker_cache, 1, self.rows, null);
        try b.rmsNorm(self.x, self.norm, self.hidden, .{ .rows = 1, .width = w, .in_stride = w, .out_stride = w, .eps = s.rms_eps });
        try self.mv(self.codec_head, self.hidden, self.logits);
    }

    /// One predictor row from `source` (talker width) at `position`.
    fn predictorRow(self: *Plan, source: Buffer, position: usize) !void {
        const b = self.backend;
        const c = self.config;
        const p = c.predictor;
        if (self.projection) |proj| {
            try self.mv(proj, source, self.x);
            try b.add(self.x, self.projection_b.?, p.hidden);
        } else try b.copy(self.x, source, p.hidden);
        try self.stack(self.predictor, p, &self.predictor_cache, 1, position, null);
    }

    /// Whether the fused BF16 kernels take these matrices (BF16, rows a
    /// multiple of 8, one width).
    fn bf16(weights: []const Weight) bool {
        for (weights) |w| if (w.matrix.encoding != 30 or w.matrix.rows % 8 != 0 or w.matrix.columns != weights[0].matrix.columns) return false;
        return true;
    }

    /// `output = w · input` for one row: BF16 through the TTS kernel.
    fn mv(self: *Plan, w: Weight, input: Buffer, output: Buffer) !void {
        if (w.matrix.encoding == 30 and input.offset % 16 == 0) return self.backend.ttsMatvecBf16(w.buffer, w.matrix, input, output, matvec_rows);
        try self.backend.matvec(w.buffer, w.matrix, input, output);
    }

    /// `output = w · input` over `count` rows (padded buffers).
    fn mm(self: *Plan, w: Weight, input: Buffer, output: Buffer, count: usize) !void {
        const b = self.backend;
        if (count == 1) return self.mv(w, input, output);
        if (w.matrix.encoding == 30) return b.ttsMatmulBf16(w.buffer, w.matrix, input, w.matrix.columns, output, w.matrix.rows, count);
        try b.matmul(w.buffer, w.matrix, input, w.matrix.columns, output, w.matrix.rows, count);
    }

    /// `n` rows of `x` through every layer of a stack at `position`…,
    /// writing their keys and values to `cache`.
    fn stack(self: *Plan, blocks: []const Layer, s: tts.Stack, cache: *Cache, n: usize, position: usize, observer: ?runtime.Observer) !void {
        const b = self.backend;
        const w = s.hidden;
        const hd = s.head_dim;
        const qw = s.heads * hd;
        const kvw = s.kv_heads * hd;
        const precision = cache.precision;
        const row_bytes = precision.bytes(kvw);
        const rope = cache.rope.?;
        const scale: f32 = @floatCast(1 / @sqrt(@as(f64, @floatFromInt(hd))));
        const norm: Backend.Norm = .{ .rows = n, .width = w, .in_stride = w, .out_stride = w, .eps = s.rms_eps };
        const total = position + n;
        // The input norm of every layer but the first is fused into the
        // previous layer's last residual add (`addRmsNorm`).
        try b.rmsNorm(self.x, blocks[0].input_norm, self.h, norm);
        for (blocks, cache.keys, cache.values, 0..) |l, keys, values, layer| {
            const k_rows = keys.slice(position * row_bytes, n * row_bytes);
            const v_rows = values.slice(position * row_bytes, n * row_bytes);
            // A row into an F32 cache: the values projected into their slot,
            // the keys normed and rotated into theirs.
            const direct = n == 1 and precision == .f32;
            const v_out = if (direct) v_rows else self.v;
            if (n == 1 and bf16(&.{ l.q, l.k, l.v }))
                try b.ttsMatvec3Bf16(.{ l.q.buffer, l.k.buffer, l.v.buffer }, .{ l.q.matrix, l.k.matrix, l.v.matrix }, self.h, .{ self.q, self.k, v_out })
            else {
                try self.mm(l.q, self.h, self.q, n);
                try self.mm(l.k, self.h, self.k, n);
                try self.mm(l.v, self.h, v_out, n);
            }
            if (n == 1) {
                try b.rmsNormRope(self.q, l.q_norm, rope, self.q, .{ .rows = s.heads, .width = hd, .in_stride = hd, .out_stride = hd, .eps = s.rms_eps }, hd, position, .split_half);
                try b.rmsNormRope(self.k, l.k_norm, rope, if (direct) k_rows else self.k, .{ .rows = s.kv_heads, .width = hd, .in_stride = hd, .out_stride = hd, .eps = s.rms_eps }, hd, position, .split_half);
            } else {
                try b.rmsNorm(self.q, l.q_norm, self.q, .{ .rows = n * s.heads, .width = hd, .in_stride = hd, .out_stride = hd, .eps = s.rms_eps });
                try b.ropeRows(self.q, rope, s.heads, hd, hd, position, n, qw, .split_half);
                try b.rmsNorm(self.k, l.k_norm, self.k, .{ .rows = n * s.kv_heads, .width = hd, .in_stride = hd, .out_stride = hd, .eps = s.rms_eps });
                try b.ropeRows(self.k, rope, s.kv_heads, hd, hd, position, n, kvw, .split_half);
            }
            if (!direct) try b.storeRows(&.{ .{ .dst = k_rows, .src = self.k, .count = n * kvw }, .{ .dst = v_rows, .src = self.v, .count = n * kvw } }, precision);
            const all_keys = keys.slice(0, total * row_bytes);
            const all_values = values.slice(0, total * row_bytes);
            if (n == 1) {
                try b.attentionDecode(all_keys, all_values, self.q, self.partials, self.att, .{ .query_heads = s.heads, .kv_heads = s.kv_heads, .key_width = hd, .value_width = hd, .visible = total, .scale = scale, .precision = precision });
            } else {
                const few = n <= Backend.verify_max_rows;
                var queries = self.q;
                if (!few and precision == .f16) {
                    try b.packHalf(&.{.{ .dst = self.q_half, .src = self.q, .count = Backend.attentionChunkRows(n) * qw }});
                    queries = self.q_half;
                }
                const shape: Backend.AttentionChunkShape = .{ .query_heads = s.heads, .kv_heads = s.kv_heads, .key_width = hd, .value_width = hd, .position = position, .count = n, .q_stride = qw, .out_stride = qw, .scale = scale, .precision = precision };
                if (few) try b.attentionVerify(all_keys, all_values, queries, self.partials, self.att, shape) else try b.attentionChunk(all_keys, all_values, queries, self.att, shape);
            }
            try self.mm(l.o, self.att, self.o, n);
            try b.addRmsNorm(self.x, self.o, l.post_norm, self.h, norm);
            if (n == 1 and bf16(&.{ l.gate, l.up }) and l.gate.matrix.rows == l.up.matrix.rows)
                try b.ttsGateUpBf16(l.gate.buffer, l.up.buffer, l.gate.matrix, self.h, self.gate)
            else {
                try self.mm(l.gate, self.h, self.gate, n);
                try self.mm(l.up, self.h, self.up, n);
                try b.siluMul(self.gate, self.up, n * s.intermediate);
            }
            try self.mm(l.down, self.gate, self.o, n);
            if (layer + 1 < blocks.len) try b.addRmsNorm(self.x, self.o, blocks[layer + 1].input_norm, self.h, norm) else try b.add(self.x, self.o, n * w);
            if (observer) |obs| {
                try b.commit();
                obs.record(obs.context, layer, self.x.floats()[0 .. n * w]);
                try b.begin();
            }
        }
    }
};
