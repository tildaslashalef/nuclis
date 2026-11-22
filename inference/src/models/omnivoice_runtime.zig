//! OmniVoice / VoiceTut-TTS masked-diffusion CPU reference runtime:
//! 28-layer bidirectional Qwen3-0.6B backbone (hidden 1024, FFN 3072 SiLU,
//! 16Q / 8KV heads × 128, Q/K per-head RMSNorm eps=1e-6, NeoX RoPE theta=1e6,
//! no causal mask, no KV cache) and deterministic seeded iterative unmasking
//! with log-probability Classifier-Free Guidance.
//!
//! Quantized matrices (`Q8_0`, `F16`) are read directly from the mapped GGUF
//! via `cpu.matvec` / `quant.row` (never expanded into F32 copies).
//! Facts: docs/models/omnivoice.md, docs/benchmarks/record/omnivoice.md.
const std = @import("std");
const gguf = @import("../formats/gguf.zig");
const weights = @import("../runtime/weights.zig");
const quant = @import("../quant/decode.zig");
const cpu = @import("../backends/cpu/root.zig");
const model = @import("omnivoice.zig");
const hc = @import("../audio/higgs_codec.zig");

const Allocator = std.mem.Allocator;
const dense = cpu.dense;

pub const hidden = model.hidden_size;
pub const intermediate = model.intermediate_size;
pub const layer_count = model.layer_count;
pub const heads = model.num_attention_heads;
pub const kv_heads = model.num_key_value_heads;
pub const head_dim = model.head_dim;
pub const query_width = model.query_width;
pub const kv_width = model.kv_width;
pub const codebooks = model.num_audio_codebooks;
pub const audio_vocab = model.audio_vocab_size;
pub const audio_mask_id: u16 = model.audio_mask_id;
pub const audio_table_rows = model.audio_table_rows;

/// Hard ceiling on sequence length (`prompt + ref + target`).
pub const max_sequence_rows: usize = 4096;

/// One position in the OmniVoice backbone input sequence.
pub const Row = union(enum) {
    text: u32,
    audio: hc.Frame,
};

pub const Stage = enum {
    embed,
    layer,
    norm,
    logits,
};

pub const Observer = struct {
    context: *anyopaque,
    record: *const fn (context: *anyopaque, stage: Stage, layer_index: ?usize, rows: []const f32) void,

    fn see(self: ?Observer, stage: Stage, layer_index: ?usize, rows: []const f32) void {
        if (self) |o| o.record(o.context, stage, layer_index, rows);
    }
};

/// Masked-diffusion sampling parameters matching `k2-fsa/OmniVoice`.
pub const SamplingConfig = struct {
    num_step: u32 = 32,
    guidance_scale: f32 = 2.0,
    t_shift: f32 = 0.1,
    layer_penalty_factor: f32 = 5.0,
    position_temperature: f32 = 5.0,
    class_temperature: f32 = 0.0,
    seed: u64 = 0,
};

pub const LayerWeights = struct {
    input_norm: []const f32,
    post_norm: []const f32,
    q_norm: []const f32,
    k_norm: []const f32,
    q_proj: cpu.Matrix,
    k_proj: cpu.Matrix,
    v_proj: cpu.Matrix,
    o_proj: cpu.Matrix,
    gate_proj: cpu.Matrix,
    up_proj: cpu.Matrix,
    down_proj: cpu.Matrix,
};

/// Mapped OmniVoice generator weights. Matrix views borrow the `Mapped` file;
/// small F32 norms live in `storage`.
pub const Model = struct {
    io: std.Io,
    storage: std.heap.ArenaAllocator,
    binding: model.Binding,
    view: weights.View,
    output_norm: []const f32,
    audio_heads: cpu.Matrix,
    layers: [layer_count]LayerWeights,

    pub fn init(gpa: Allocator, io: std.Io, doc: *const gguf.Document, view: weights.View) !Model {
        const binding = try model.bind(gpa, doc);
        const data = view.file[@intCast(view.data_offset)..];
        try model.verifyCodebookOffsets(&binding, data);

        var arena: std.heap.ArenaAllocator = .init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();

        const output_norm = try view.vector(a, binding.output_norm);
        const audio_heads = try view.matrix(binding.audio_heads);

        var layers: [layer_count]LayerWeights = undefined;
        for (&layers, binding.layers) |*dst, src| {
            dst.* = .{
                .input_norm = try view.vector(a, src.input_layernorm),
                .post_norm = try view.vector(a, src.post_attention_layernorm),
                .q_norm = try view.vector(a, src.q_norm),
                .k_norm = try view.vector(a, src.k_norm),
                .q_proj = try view.matrix(src.q_proj),
                .k_proj = try view.matrix(src.k_proj),
                .v_proj = try view.matrix(src.v_proj),
                .o_proj = try view.matrix(src.o_proj),
                .gate_proj = try view.matrix(src.gate_proj),
                .up_proj = try view.matrix(src.up_proj),
                .down_proj = try view.matrix(src.down_proj),
            };
        }

        return .{
            .io = io,
            .storage = arena,
            .binding = binding,
            .view = view,
            .output_norm = output_norm,
            .audio_heads = audio_heads,
            .layers = layers,
        };
    }

    pub fn deinit(self: *Model) void {
        self.storage.deinit();
        self.* = undefined;
    }

    /// Runs the 28-layer bidirectional Qwen3-0.6B backbone on `rows` (`[S]`)
    /// and projects the last `target_tail` rows through `audio_heads` into
    /// `logits_out` (`[target_tail * 8200]`).
    pub fn forward(
        self: *const Model,
        gpa: Allocator,
        rows: []const Row,
        target_tail: usize,
        logits_out: []f32,
        observer: ?Observer,
    ) !void {
        const s_len = rows.len;
        if (s_len == 0 or s_len > max_sequence_rows or target_tail == 0 or target_tail > s_len)
            return error.InvalidShape;
        if (logits_out.len != target_tail * audio_table_rows) return error.InvalidShape;

        var scratch_arena: std.heap.ArenaAllocator = .init(gpa);
        defer scratch_arena.deinit();
        const ws = scratch_arena.allocator();

        // 1. Build input embeddings [S, 1024].
        const h = try ws.alloc(f32, s_len * hidden);
        try self.embedRows(rows, h);
        Observer.see(observer, .embed, null, h);

        // Scratch buffers reused across all 28 layers.
        const norm_buf = try ws.alloc(f32, s_len * hidden);
        const q_buf = try ws.alloc(f32, s_len * query_width);
        const k_buf = try ws.alloc(f32, s_len * kv_width);
        const v_buf = try ws.alloc(f32, s_len * kv_width);
        const attn_out = try ws.alloc(f32, s_len * query_width);
        const attn_scratch = try ws.alloc(f32, heads * s_len);
        const o_buf = try ws.alloc(f32, hidden);
        const gate_buf = try ws.alloc(f32, intermediate);
        const up_buf = try ws.alloc(f32, intermediate);
        const mv_scratch = try ws.alloc(f32, cpu.matvecScratch(intermediate));

        const attn_scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(head_dim)));

        // 2. 28 bidirectional Qwen3 layers.
        for (self.layers, 0..) |layer, li| {
            // Input RMSNorm + Q/K/V projections + per-head Q/K RMSNorm + NeoX RoPE.
            for (0..s_len) |t| {
                const h_row = h[t * hidden ..][0..hidden];
                const n_row = norm_buf[t * hidden ..][0..hidden];
                try rmsNormWeighted(h_row, layer.input_norm, n_row, model.rms_norm_eps);

                const q_row = q_buf[t * query_width ..][0..query_width];
                const k_row = k_buf[t * kv_width ..][0..kv_width];
                const v_row = v_buf[t * kv_width ..][0..kv_width];
                try cpu.matvec(self.io, layer.q_proj, n_row, q_row, mv_scratch);
                try cpu.matvec(self.io, layer.k_proj, n_row, k_row, mv_scratch);
                try cpu.matvec(self.io, layer.v_proj, n_row, v_row, mv_scratch);

                const pos: i32 = @intCast(t);
                for (0..heads) |hd| {
                    const qh = q_row[hd * head_dim ..][0..head_dim];
                    try rmsNormWeighted(qh, layer.q_norm, qh, model.rms_norm_eps);
                    try cpu.rope.apply(qh, qh, .{ .dimensions = head_dim, .base = model.rope_theta, .position = pos, .pairing = .split_half });
                }
                for (0..kv_heads) |hd| {
                    const kh = k_row[hd * head_dim ..][0..head_dim];
                    try rmsNormWeighted(kh, layer.k_norm, kh, model.rms_norm_eps);
                    try cpu.rope.apply(kh, kh, .{ .dimensions = head_dim, .base = model.rope_theta, .position = pos, .pairing = .split_half });
                }
            }

            // Full bidirectional GQA attention (`window = null`).
            try dense.attention(self.io, .{
                .tokens = s_len,
                .heads = heads,
                .kv_heads = kv_heads,
                .head_dim = head_dim,
                .window = null,
                .scale = attn_scale,
                .stride = query_width,
                .kv_stride = kv_width,
                .q = q_buf,
                .k = k_buf,
                .v = v_buf,
            }, attn_out, attn_scratch);

            // Output projection + residual + post-attention RMSNorm + SwiGLU MLP + residual.
            for (0..s_len) |t| {
                const h_row = h[t * hidden ..][0..hidden];
                const a_row = attn_out[t * query_width ..][0..query_width];
                try cpu.matvec(self.io, layer.o_proj, a_row, o_buf, mv_scratch);
                for (h_row, o_buf) |*dst, d| dst.* += d;

                const n_row = norm_buf[t * hidden ..][0..hidden];
                try rmsNormWeighted(h_row, layer.post_norm, n_row, model.rms_norm_eps);
                try cpu.matvec(self.io, layer.gate_proj, n_row, gate_buf, mv_scratch);
                try cpu.matvec(self.io, layer.up_proj, n_row, up_buf, mv_scratch);
                for (gate_buf, up_buf) |*g, u| g.* = cpu.silu(g.*) * u;
                try cpu.matvec(self.io, layer.down_proj, gate_buf, o_buf, mv_scratch);
                for (h_row, o_buf) |*dst, d| dst.* += d;
            }

            Observer.see(observer, .layer, li, h);
        }

        // 3. Final RMSNorm across all rows.
        for (0..s_len) |t| {
            const h_row = h[t * hidden ..][0..hidden];
            try rmsNormWeighted(h_row, self.output_norm, h_row, model.rms_norm_eps);
        }
        Observer.see(observer, .norm, null, h);

        // 4. Audio heads on the last `target_tail` rows only.
        const tail_start = s_len - target_tail;
        for (0..target_tail) |t| {
            const h_row = h[(tail_start + t) * hidden ..][0..hidden];
            const out_row = logits_out[t * audio_table_rows ..][0..audio_table_rows];
            try cpu.matvec(self.io, self.audio_heads, h_row, out_row, mv_scratch);
        }
        Observer.see(observer, .logits, null, logits_out);
    }

    /// Decodes input rows into `[rows.len * hidden]` F32 embeddings.
    /// Reusable by both CPU `forward` and Metal `omnivoice_metal`.
    pub fn embedRows(self: *const Model, rows: []const Row, out: []f32) !void {
        if (out.len != rows.len * hidden) return error.InvalidShape;
        var tmp: [hidden]f32 = undefined;
        for (rows, 0..) |row, t| {
            const dst = out[t * hidden ..][0..hidden];
            switch (row) {
                .text => |id| {
                    if (id >= model.vocab_size) return error.InvalidShape;
                    try self.view.row(self.binding.token_embedding, id, dst);
                },
                .audio => |frame| {
                    @memset(dst, 0);
                    for (frame, 0..) |code, c| {
                        if (code >= audio_vocab) return error.InvalidShape;
                        const idx: usize = @as(usize, @intCast(model.expected_codebook_offsets[c])) + code;
                        try self.view.row(self.binding.audio_embeddings, idx, &tmp);
                        for (dst, tmp) |*acc, v| acc.* += v;
                    }
                },
            }
        }
    }

    /// Runs iterative masked-diffusion generation on the CPU for one utterance
    /// chunk (`prompt_ids`, optional `ref_codes`, `target_frames`).
    /// Returns caller-owned `[]hc.Frame` (`[target_frames][8]u16`, all codes in `0..1023`).
    pub fn generate(
        self: *const Model,
        gpa: Allocator,
        prompt_ids: []const u32,
        ref_codes: ?[]const hc.Frame,
        target_frames: usize,
        cfg: SamplingConfig,
        cancel: ?*const std.atomic.Value(bool),
    ) ![]hc.Frame {
        if (target_frames == 0 or cfg.num_step == 0) return error.InvalidShape;
        const ref_len = if (ref_codes) |rc| rc.len else 0;
        const c_len = prompt_ids.len + ref_len + target_frames;
        if (c_len > max_sequence_rows) return error.InvalidShape;

        const schedule = try buildSchedule(gpa, target_frames, cfg.num_step, cfg.t_shift);
        defer gpa.free(schedule);

        const cond_rows = try gpa.alloc(Row, c_len);
        defer gpa.free(cond_rows);
        for (prompt_ids, 0..) |id, i| cond_rows[i] = .{ .text = id };
        if (ref_codes) |rc| {
            for (rc, 0..) |frame, i| cond_rows[prompt_ids.len + i] = .{ .audio = frame };
        }
        const masked_frame: hc.Frame = @splat(audio_mask_id);
        for (0..target_frames) |t| cond_rows[prompt_ids.len + ref_len + t] = .{ .audio = masked_frame };

        const uncond_rows = try gpa.alloc(Row, target_frames);
        defer gpa.free(uncond_rows);
        for (uncond_rows) |*r| r.* = .{ .audio = masked_frame };

        const tokens = try gpa.alloc(hc.Frame, target_frames);
        errdefer gpa.free(tokens);
        for (tokens) |*f| f.* = masked_frame;

        const c_logits = try gpa.alloc(f32, target_frames * audio_table_rows);
        defer gpa.free(c_logits);
        const u_logits = try gpa.alloc(f32, target_frames * audio_table_rows);
        defer gpa.free(u_logits);

        var prng = std.Random.DefaultPrng.init(cfg.seed);
        const rng = prng.random();

        for (schedule) |k| {
            if (cancel) |flag| if (flag.load(.acquire)) return error.Canceled;
            if (k == 0) continue;

            try self.forward(gpa, cond_rows, target_frames, c_logits, null);
            if (cfg.guidance_scale != 0.0) {
                try self.forward(gpa, uncond_rows, target_frames, u_logits, null);
            }
            try unmaskStep(gpa, c_logits, if (cfg.guidance_scale != 0.0) u_logits else null, tokens, k, cfg, rng);

            for (0..target_frames) |t| {
                cond_rows[prompt_ids.len + ref_len + t] = .{ .audio = tokens[t] };
                uncond_rows[t] = .{ .audio = tokens[t] };
            }
        }
        return tokens;
    }
};

// ---- shared diffusion schedule & unmasking helpers (reused by Metal) -------

/// Computes per-step unmask counts for `target_frames * 8` slots using
/// OmniVoice's shifted schedule `t_i = t_shift * u / (1 + (t_shift - 1) * u)`.
/// Caller owns returned slice (`len == num_step`), whose sum equals `target_frames * 8`.
pub fn buildSchedule(alloc: Allocator, target_frames: usize, num_step: u32, t_shift: f32) Allocator.Error![]usize {
    const steps: usize = @intCast(num_step);
    const sched = try alloc.alloc(usize, steps);
    const total_mask: usize = target_frames * codebooks;
    var rem: usize = total_mask;
    const shift: f64 = @floatCast(t_shift);
    const denom: f64 = @floatFromInt(steps);

    for (0..steps) |s| {
        if (s + 1 == steps) {
            sched[s] = rem;
            rem = 0;
        } else {
            const p0 = @as(f64, @floatFromInt(s)) / denom;
            const p1 = @as(f64, @floatFromInt(s + 1)) / denom;
            const t0 = shift * p0 / (1.0 + (shift - 1.0) * p0);
            const t1 = shift * p1 / (1.0 + (shift - 1.0) * p1);
            const want: usize = @intFromFloat(@ceil(@as(f64, @floatFromInt(total_mask)) * (t1 - t0)));
            const take = @min(want, rem);
            sched[s] = take;
            rem -= take;
        }
    }
    return sched;
}

const ScoredSlot = struct {
    frame: usize,
    codebook: usize,
    token: u16,
    score: f32,
};

/// Executes one masked-diffusion unmasking step on `[target_frames][8][1025]`
/// logits (`c_logits` and optional `u_logits` when `guidance_scale != 0`),
/// updating `k` masked positions in `tokens`.
pub fn unmaskStep(
    alloc: Allocator,
    c_logits: []const f32,
    u_logits: ?[]const f32,
    tokens: []hc.Frame,
    k: usize,
    cfg: SamplingConfig,
    rng: std.Random,
) Allocator.Error!void {
    const t_len = tokens.len;
    const total = t_len * codebooks;
    const slots = try alloc.alloc(ScoredSlot, total);
    defer alloc.free(slots);

    var c_lp: [audio_vocab]f32 = undefined;
    var u_lp: [audio_vocab]f32 = undefined;
    var guided: [audio_vocab]f32 = undefined;

    for (0..t_len) |t| {
        for (0..codebooks) |c| {
            const idx = t * codebooks + c;
            if (tokens[t][c] != audio_mask_id) {
                slots[idx] = .{ .frame = t, .codebook = c, .token = tokens[t][c], .score = -std.math.inf(f32) };
                continue;
            }
            const base = t * audio_table_rows + c * audio_vocab;
            logSoftmax(c_logits[base..][0..audio_vocab], &c_lp);
            if (u_logits) |ul| {
                logSoftmax(ul[base..][0..audio_vocab], &u_lp);
                for (0..audio_vocab) |v| {
                    guided[v] = c_lp[v] + cfg.guidance_scale * (c_lp[v] - u_lp[v]);
                }
                logSoftmax(&guided, &c_lp);
            }
            // Never predict `audio_mask_id` (1024).
            c_lp[audio_mask_id] = -std.math.inf(f32);

            // Confidence is max(log_probs) over valid codes 0..1023.
            var max_lp: f32 = -std.math.inf(f32);
            var argmax_id: u16 = 0;
            for (0..audio_mask_id) |v| {
                if (c_lp[v] > max_lp) {
                    max_lp = c_lp[v];
                    argmax_id = @intCast(v);
                }
            }

            const pred: u16 = if (cfg.class_temperature > 0.0)
                sampleTopRatio(&c_lp, 0.1, cfg.class_temperature, rng)
            else
                argmax_id;

            var score: f32 = max_lp - @as(f32, @floatFromInt(c)) * cfg.layer_penalty_factor;
            if (cfg.position_temperature > 0.0) {
                score = score / cfg.position_temperature + gumbelNoise(rng);
            }
            slots[idx] = .{ .frame = t, .codebook = c, .token = pred, .score = score };
        }
    }

    // Sort descending by score and unmask top `k` masked positions.
    std.mem.sort(ScoredSlot, slots, {}, struct {
        fn gt(_: void, a: ScoredSlot, b: ScoredSlot) bool {
            return a.score > b.score;
        }
    }.gt);

    const take = @min(k, total);
    for (slots[0..take]) |s| {
        if (tokens[s.frame][s.codebook] == audio_mask_id) {
            tokens[s.frame][s.codebook] = s.token;
        }
    }
}

fn logSoftmax(in: []const f32, out: []f32) void {
    var max_v: f64 = -std.math.inf(f64);
    for (in) |x| max_v = @max(max_v, @as(f64, x));
    var sum_exp: f64 = 0;
    for (in) |x| sum_exp += @exp(@as(f64, x) - max_v);
    const lse = max_v + @log(sum_exp);
    for (in, out) |x, *y| y.* = @floatCast(@as(f64, x) - lse);
}

fn gumbelNoise(rng: std.Random) f32 {
    const u: f64 = rng.float(f64);
    return @floatCast(-@log(-@log(u + 1e-10) + 1e-10));
}

fn sampleTopRatio(log_probs: *const [audio_vocab]f32, ratio: f32, temp: f32, rng: std.Random) u16 {
    const k: usize = @intFromFloat(@ceil(ratio * @as(f32, @floatFromInt(audio_vocab))));
    const Item = struct { id: u16, lp: f32 };
    var items: [audio_mask_id]Item = undefined;
    for (0..audio_mask_id) |v| items[v] = .{ .id = @intCast(v), .lp = log_probs[v] };
    std.mem.sort(Item, &items, {}, struct {
        fn gt(_: void, a: Item, b: Item) bool {
            return a.lp > b.lp;
        }
    }.gt);
    const limit = @min(k, audio_mask_id);
    var best_id: u16 = items[0].id;
    var best_val: f32 = -std.math.inf(f32);
    for (items[0..limit]) |it| {
        const s = it.lp / temp + gumbelNoise(rng);
        if (s > best_val) {
            best_val = s;
            best_id = it.id;
        }
    }
    return best_id;
}

fn rmsNormWeighted(in: []const f32, weight: []const f32, out: []f32, eps: f32) cpu.Error!void {
    try cpu.rmsNorm(in, out, eps);
    for (out, weight) |*y, w| y.* *= w;
}

// ---- tests -----------------------------------------------------------------

test "buildSchedule sums to target_frames * 8 and front-loads with t_shift < 1" {
    const gpa = std.testing.allocator;
    const sched = try buildSchedule(gpa, 12, 32, 0.1);
    defer gpa.free(sched);
    try std.testing.expectEqual(@as(usize, 32), sched.len);
    var sum: usize = 0;
    for (sched) |k| sum += k;
    try std.testing.expectEqual(@as(usize, 12 * codebooks), sum);
    // With t_shift=0.1, later steps unmask more slots than early steps (low-SNR focus).
    try std.testing.expect(sched[31] >= sched[0]);
}

test "unmaskStep CFG log-softmax, layer penalty, termination, and seed determinism" {
    const gpa = std.testing.allocator;
    const t_len: usize = 4;
    const c_logits = try gpa.alloc(f32, t_len * audio_table_rows);
    defer gpa.free(c_logits);
    const u_logits = try gpa.alloc(f32, t_len * audio_table_rows);
    defer gpa.free(u_logits);
    @memset(c_logits, 0);
    @memset(u_logits, 0);

    // Put distinct peaks in c_logits and make audio_mask_id=1024 huge to verify masking out 1024.
    for (0..t_len) |t| {
        for (0..codebooks) |c| {
            const base = t * audio_table_rows + c * audio_vocab;
            c_logits[base + audio_mask_id] = 100.0; // must be ignored!
            const target_code: usize = (t * 17 + c * 31) % 1024;
            c_logits[base + target_code] = 5.0;
            u_logits[base + target_code] = 1.0;
        }
    }

    const cfg: SamplingConfig = .{ .num_step = 8, .guidance_scale = 2.0, .seed = 42 };
    const sched = try buildSchedule(gpa, t_len, cfg.num_step, cfg.t_shift);
    defer gpa.free(sched);

    const tok1 = try gpa.alloc(hc.Frame, t_len);
    defer gpa.free(tok1);
    const tok2 = try gpa.alloc(hc.Frame, t_len);
    defer gpa.free(tok2);
    for (tok1) |*f| f.* = @splat(audio_mask_id);
    for (tok2) |*f| f.* = @splat(audio_mask_id);

    var prng1 = std.Random.DefaultPrng.init(cfg.seed);
    var prng2 = std.Random.DefaultPrng.init(cfg.seed);
    for (sched) |k| {
        try unmaskStep(gpa, c_logits, u_logits, tok1, k, cfg, prng1.random());
        try unmaskStep(gpa, c_logits, u_logits, tok2, k, cfg, prng2.random());
    }

    // All positions unmasked to exact target codes, never 1024, deterministic across identical seeds.
    for (0..t_len) |t| {
        for (0..codebooks) |c| {
            const want: u16 = @intCast((t * 17 + c * 31) % 1024);
            try std.testing.expectEqual(want, tok1[t][c]);
            try std.testing.expectEqual(tok1[t][c], tok2[t][c]);
        }
    }
}

test "bidirectional GQA attention is symmetric across sequence order" {
    const io = std.testing.io;
    // 2 tokens, 2 heads, 1 kv_head, head_dim=8 (identical q/k/v swapped -> outputs swapped).
    var q = [_]f32{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0 };
    var k = [_]f32{ 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0 };
    var v = [_]f32{ 10, 0, 0, 0, 0, 0, 0, 0, 0, 20, 0, 0, 0, 0, 0, 0 };
    var out: [32]f32 = undefined;
    var scratch: [4]f32 = undefined;
    try dense.attention(io, .{
        .tokens = 2,
        .heads = 2,
        .kv_heads = 1,
        .head_dim = 8,
        .window = null,
        .scale = 1.0,
        .stride = 16,
        .kv_stride = 8,
        .q = &q,
        .k = &k,
        .v = &v,
    }, &out, &scratch);
    // Token 0 attends to token 1 (bidirectional!) so head 0 sees non-zero v[1] channel 1.
    try std.testing.expect(out[1] > 1.0);
    // Head 0 of token 0 equals head 1 of token 1 by symmetry.
    try std.testing.expectApproxEqAbs(out[0], out[24], 1e-6);
    try std.testing.expectApproxEqAbs(out[1], out[25], 1e-6);
}
