//! Higgs Audio V2 codec CPU encoder (`HiggsAudioV2TokenizerModel`): turns
//! 24 kHz mono F32 reference clips into `[frames][8]` RVQ codes at 25 fps.
//!
//! Combines two branches:
//! - Semantic branch: Hann-windowed sinc resample (24 kHz → 16 kHz, matching
//!   `torchaudio.functional.resample`), pad `(160, 160)`, HuBERT-Base (7 convs,
//!   layer-0 GroupNorm, feature projection, grouped weight-normed pos conv
//!   embedding, 12 post-norm bidirectional transformer layers, mean of all 13
//!   hidden states, `::2` subsampling) → `encoder_semantic` (ELU residual units).
//! - Acoustic branch: 5-block DAC encoder (`downsampling_ratios = [8, 5, 4, 2, 3]`,
//!   Snake1d activations) → `conv2` (`2048 → 256`).
//! Concatenates `[acoustic (256) || semantic (768)]` → `fc` (`1024 → 1024`) →
//! 8-codebook Euclidean RVQ.
const std = @import("std");
const gguf = @import("../formats/gguf.zig");
const weights = @import("../runtime/weights.zig");
const cpu = @import("../backends/cpu/root.zig");
const hc = @import("higgs_codec.zig");

const Allocator = std.mem.Allocator;
const dense = cpu.dense;

pub const max_samples: usize = 30 * hc.sample_rate; // 30 s at 24 kHz
pub const semantic_sample_rate: usize = 16_000;
const semantic_dim: usize = 768;
const hubert_conv_dim: usize = 512;
const hubert_layers: usize = 12;
const hubert_heads: usize = 12;
const hubert_head_dim: usize = 64;
const hubert_ffn_dim: usize = 3072;
const hubert_ln_eps: f32 = 1e-5;
const pos_conv_kernel: usize = 128;
const pos_conv_groups: usize = 16;
const pos_conv_group_ch: usize = semantic_dim / pos_conv_groups; // 48

const hubert_kernels = [7]usize{ 10, 3, 3, 3, 3, 2, 2 };
const hubert_strides = [7]usize{ 5, 2, 2, 2, 2, 2, 2 };

pub const Error = hc.Error || error{InvalidAudio};

/// Stages exposed to an optional `Observer`.
pub const Stage = enum {
    resampled_16k,
    hubert,
    semantic,
    acoustic,
    fused,
};

pub const Observer = struct {
    context: *anyopaque,
    record: *const fn (context: *anyopaque, stage: Stage, rows: []const f32) void,

    fn see(self: ?Observer, stage: Stage, rows: []const f32) void {
        if (self) |o| o.record(o.context, stage, rows);
    }
};

const LayerNorm = struct {
    w: []const f32,
    b: []const f32,
};

const HubertLayer = struct {
    q: hc.Linear,
    k: hc.Linear,
    v: hc.Linear,
    out: hc.Linear,
    ln1: LayerNorm,
    fc1: hc.Linear,
    fc2: hc.Linear,
    ln2: LayerNorm,
};

const SemanticResUnit = struct {
    conv1: hc.Conv1d,
    conv2: hc.Conv1d,
};

const SemanticBlock = struct {
    units: [2]SemanticResUnit,
    conv: hc.Conv1d,
};

const AcousticBlock = struct {
    units: [3]hc.ResUnit,
    snake1: []const f32,
    conv1: hc.Conv1d,
};

const Quantizer = struct {
    project_in: hc.Linear,
    project_out: hc.Linear,
    /// `[codebook_size][codebook_dim]`.
    embed: []const f32,
    /// Precomputed `||embed[k]||^2` for Euclidean nearest-neighbor search.
    embed_sq: []const f32,
};

pub const Encoder = struct {
    io: std.Io,
    storage: std.heap.ArenaAllocator,
    // HuBERT-Base semantic model
    feat_convs: [7]hc.Conv1d,
    feat_gn0: LayerNorm,
    feat_proj_ln: LayerNorm,
    feat_proj: hc.Linear,
    /// Grouped pos conv weights, folded from weight_norm(dim=2), `[16][48][128 * 48]` (tap-major per group).
    pos_conv_w: [pos_conv_groups][]const f32,
    pos_conv_b: []const f32,
    enc_ln: LayerNorm,
    hubert: [hubert_layers]HubertLayer,
    // Semantic encoder head
    sem_conv: hc.Conv1d,
    sem_blocks: [2]SemanticBlock,
    // DAC acoustic encoder
    aco_conv1: hc.Conv1d,
    aco_blocks: [5]AcousticBlock,
    aco_snake_out: []const f32,
    aco_conv2: hc.Conv1d,
    // Fusion & 8-codebook RVQ
    fc: hc.Linear,
    quantizers: [hc.codebooks]Quantizer,

    pub fn init(gpa: Allocator, io: std.Io, doc: gguf.Document, directory: []const u8, view: weights.View) !Encoder {
        var arena: std.heap.ArenaAllocator = .init(gpa);
        errdefer arena.deinit();
        const a = arena.allocator();
        var loader = try hc.NativeLoader.init(gpa, a, doc, directory, view);
        defer loader.deinit();

        var b1: [192]u8 = undefined;
        var b2: [192]u8 = undefined;

        // 1. HuBERT feature extractor (7 convs, layer 0 has GroupNorm(512, 512)).
        var feat_convs: [7]hc.Conv1d = undefined;
        for (0..7) |i| {
            const in_ch: usize = if (i == 0) 1 else hubert_conv_dim;
            feat_convs[i] = try loader.conv1d(
                fmt(&b1, "audio_tokenizer_weights/semantic_model.feature_extractor.conv_layers.{d}.conv.weight", .{i}),
                null,
                in_ch,
                hubert_conv_dim,
                hubert_kernels[i],
                hubert_strides[i],
                0,
                1,
            );
        }
        const feat_gn0: LayerNorm = .{
            .w = try loader.floats("audio_tokenizer_weights/semantic_model.feature_extractor.conv_layers.0.layer_norm.weight", hubert_conv_dim),
            .b = try loader.floats("audio_tokenizer_weights/semantic_model.feature_extractor.conv_layers.0.layer_norm.bias", hubert_conv_dim),
        };
        const feat_proj_ln: LayerNorm = .{
            .w = try loader.floats("audio_tokenizer_weights/semantic_model.feature_projection.layer_norm.weight", hubert_conv_dim),
            .b = try loader.floats("audio_tokenizer_weights/semantic_model.feature_projection.layer_norm.bias", hubert_conv_dim),
        };
        const feat_proj = try loader.linear(
            "audio_tokenizer_weights/semantic_model.feature_projection.projection.weight",
            "audio_tokenizer_weights/semantic_model.feature_projection.projection.bias",
            hubert_conv_dim,
            semantic_dim,
        );

        // 2. HuBERT positional conv embedding: fold weight_norm(dim=2) across kernel axis K=128.
        const pos_g = try loader.floats("audio_tokenizer_weights/semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original0", pos_conv_kernel);
        const pos_v = try loader.floats("audio_tokenizer_weights/semantic_model.encoder.pos_conv_embed.conv.parametrizations.weight.original1", semantic_dim * pos_conv_group_ch * pos_conv_kernel);
        const pos_conv_b = try loader.floats("audio_tokenizer_weights/semantic_model.encoder.pos_conv_embed.conv.bias", semantic_dim);

        var norm_inv: [pos_conv_kernel]f64 = undefined;
        for (0..pos_conv_kernel) |k| {
            var sq: f64 = 0;
            for (0..semantic_dim * pos_conv_group_ch) |oc| {
                const v = @as(f64, pos_v[oc * pos_conv_kernel + k]);
                sq += v * v;
            }
            norm_inv[k] = if (sq > 0) @as(f64, pos_g[k]) / @sqrt(sq) else 0;
        }
        var pos_conv_w: [pos_conv_groups][]const f32 = undefined;
        for (0..pos_conv_groups) |g| {
            const gw = try a.alloc(f32, pos_conv_group_ch * pos_conv_kernel * pos_conv_group_ch);
            for (0..pos_conv_group_ch) |o_local| {
                const o = g * pos_conv_group_ch + o_local;
                for (0..pos_conv_group_ch) |c_local| {
                    for (0..pos_conv_kernel) |k| {
                        const raw_v = @as(f64, pos_v[(o * pos_conv_group_ch + c_local) * pos_conv_kernel + k]);
                        gw[(o_local * pos_conv_kernel + k) * pos_conv_group_ch + c_local] = @floatCast(raw_v * norm_inv[k]);
                    }
                }
            }
            pos_conv_w[g] = gw;
        }

        const enc_ln: LayerNorm = .{
            .w = try loader.floats("audio_tokenizer_weights/semantic_model.encoder.layer_norm.weight", semantic_dim),
            .b = try loader.floats("audio_tokenizer_weights/semantic_model.encoder.layer_norm.bias", semantic_dim),
        };

        var hubert: [hubert_layers]HubertLayer = undefined;
        for (&hubert, 0..) |*l, i| {
            l.* = .{
                .q = try loader.linear(fmt(&b1, "audio_tokenizer_weights/semantic_model.encoder.layers.{d}.attention.q_proj.weight", .{i}), fmt(&b2, "audio_tokenizer_weights/semantic_model.encoder.layers.{d}.attention.q_proj.bias", .{i}), semantic_dim, semantic_dim),
                .k = try loader.linear(fmt(&b1, "audio_tokenizer_weights/semantic_model.encoder.layers.{d}.attention.k_proj.weight", .{i}), fmt(&b2, "audio_tokenizer_weights/semantic_model.encoder.layers.{d}.attention.k_proj.bias", .{i}), semantic_dim, semantic_dim),
                .v = try loader.linear(fmt(&b1, "audio_tokenizer_weights/semantic_model.encoder.layers.{d}.attention.v_proj.weight", .{i}), fmt(&b2, "audio_tokenizer_weights/semantic_model.encoder.layers.{d}.attention.v_proj.bias", .{i}), semantic_dim, semantic_dim),
                .out = try loader.linear(fmt(&b1, "audio_tokenizer_weights/semantic_model.encoder.layers.{d}.attention.out_proj.weight", .{i}), fmt(&b2, "audio_tokenizer_weights/semantic_model.encoder.layers.{d}.attention.out_proj.bias", .{i}), semantic_dim, semantic_dim),
                .ln1 = .{
                    .w = try loader.floats(fmt(&b1, "audio_tokenizer_weights/semantic_model.encoder.layers.{d}.layer_norm.weight", .{i}), semantic_dim),
                    .b = try loader.floats(fmt(&b2, "audio_tokenizer_weights/semantic_model.encoder.layers.{d}.layer_norm.bias", .{i}), semantic_dim),
                },
                .fc1 = try loader.linear(fmt(&b1, "audio_tokenizer_weights/semantic_model.encoder.layers.{d}.feed_forward.intermediate_dense.weight", .{i}), fmt(&b2, "audio_tokenizer_weights/semantic_model.encoder.layers.{d}.feed_forward.intermediate_dense.bias", .{i}), semantic_dim, hubert_ffn_dim),
                .fc2 = try loader.linear(fmt(&b1, "audio_tokenizer_weights/semantic_model.encoder.layers.{d}.feed_forward.output_dense.weight", .{i}), fmt(&b2, "audio_tokenizer_weights/semantic_model.encoder.layers.{d}.feed_forward.output_dense.bias", .{i}), hubert_ffn_dim, semantic_dim),
                .ln2 = .{
                    .w = try loader.floats(fmt(&b1, "audio_tokenizer_weights/semantic_model.encoder.layers.{d}.final_layer_norm.weight", .{i}), semantic_dim),
                    .b = try loader.floats(fmt(&b2, "audio_tokenizer_weights/semantic_model.encoder.layers.{d}.final_layer_norm.bias", .{i}), semantic_dim),
                },
            };
        }

        // 3. SemanticEncoder (strides=[1, 1], dilations=[1, 1], kernel=3).
        const sem_conv = try loader.conv1d("audio_tokenizer_weights/encoder_semantic.conv.weight", null, semantic_dim, semantic_dim, 3, 1, 1, 1);
        var sem_blocks: [2]SemanticBlock = undefined;
        for (&sem_blocks, 0..) |*blk, i| {
            for (&blk.units, 0..) |*u, j| {
                u.* = .{
                    .conv1 = try loader.conv1d(fmt(&b1, "audio_tokenizer_weights/encoder_semantic.conv_blocks.{d}.res_units.{d}.conv1.weight", .{ i, j }), null, semantic_dim, semantic_dim, 3, 1, 1, 1),
                    .conv2 = try loader.conv1d(fmt(&b1, "audio_tokenizer_weights/encoder_semantic.conv_blocks.{d}.res_units.{d}.conv2.weight", .{ i, j }), null, semantic_dim, semantic_dim, 1, 1, 0, 1),
                };
            }
            blk.conv = try loader.conv1d(
                fmt(&b1, "audio_tokenizer_weights/encoder_semantic.conv_blocks.{d}.conv.weight", .{i}),
                fmt(&b2, "audio_tokenizer_weights/encoder_semantic.conv_blocks.{d}.conv.bias", .{i}),
                semantic_dim,
                semantic_dim,
                3,
                1,
                1,
                1,
            );
        }

        // 4. DAC acoustic encoder (encoder_hidden_size=64, ratios=[8, 5, 4, 2, 3]).
        const aco_conv1 = try loader.conv1d("audio_tokenizer_weights/acoustic_encoder.conv1.weight", "audio_tokenizer_weights/acoustic_encoder.conv1.bias", 1, 64, 7, 1, 3, 1);
        var aco_blocks: [5]AcousticBlock = undefined;
        var ch: usize = 64;
        for (&aco_blocks, hc.rates, 0..) |*blk, stride, i| {
            for (&blk.units, hc.dilations, 0..) |*u, d, j| {
                var pbuf: [192]u8 = undefined;
                const prefix = fmt(&pbuf, "audio_tokenizer_weights/acoustic_encoder.block.{d}.res_unit{d}", .{ i, j + 1 });
                u.* = try loader.resUnit(prefix, ch, d);
            }
            blk.snake1 = try loader.floats(fmt(&b1, "audio_tokenizer_weights/acoustic_encoder.block.{d}.snake1.alpha", .{i}), ch);
            const out_ch = ch * 2;
            blk.conv1 = try loader.conv1d(
                fmt(&b1, "audio_tokenizer_weights/acoustic_encoder.block.{d}.conv1.weight", .{i}),
                fmt(&b2, "audio_tokenizer_weights/acoustic_encoder.block.{d}.conv1.bias", .{i}),
                ch,
                out_ch,
                2 * stride,
                stride,
                (stride + 1) / 2,
                1,
            );
            ch = out_ch;
        }
        const aco_snake_out = try loader.floats("audio_tokenizer_weights/acoustic_encoder.snake1.alpha", ch);
        const aco_conv2 = try loader.conv1d("audio_tokenizer_weights/acoustic_encoder.conv2.weight", "audio_tokenizer_weights/acoustic_encoder.conv2.bias", ch, hc.acoustic_dim, 3, 1, 1, 1);

        // 5. Fusion fc and 8 RVQ quantizers.
        const fc = try loader.linear("audio_tokenizer_weights/fc.weight", "audio_tokenizer_weights/fc.bias", hc.latent_dim, hc.latent_dim);
        var quantizers: [hc.codebooks]Quantizer = undefined;
        for (&quantizers, 0..) |*q, c| {
            const embed = try loader.floats(fmt(&b1, "audio_tokenizer_weights/quantizer.quantizers.{d}.codebook.embed", .{c}), hc.codebook_size * hc.codebook_dim);
            const embed_sq = try a.alloc(f32, hc.codebook_size);
            for (0..hc.codebook_size) |k| {
                const row = embed[k * hc.codebook_dim ..][0..hc.codebook_dim];
                var sq: f64 = 0;
                for (row) |v| sq += @as(f64, v) * @as(f64, v);
                embed_sq[k] = @floatCast(sq);
            }
            q.* = .{
                .project_in = try loader.linear(
                    fmt(&b1, "audio_tokenizer_weights/quantizer.quantizers.{d}.project_in.weight", .{c}),
                    fmt(&b2, "audio_tokenizer_weights/quantizer.quantizers.{d}.project_in.bias", .{c}),
                    hc.latent_dim,
                    hc.codebook_dim,
                ),
                .project_out = try loader.linear(
                    fmt(&b1, "audio_tokenizer_weights/quantizer.quantizers.{d}.project_out.weight", .{c}),
                    fmt(&b2, "audio_tokenizer_weights/quantizer.quantizers.{d}.project_out.bias", .{c}),
                    hc.codebook_dim,
                    hc.latent_dim,
                ),
                .embed = embed,
                .embed_sq = embed_sq,
            };
        }

        return .{
            .io = io,
            .storage = arena,
            .feat_convs = feat_convs,
            .feat_gn0 = feat_gn0,
            .feat_proj_ln = feat_proj_ln,
            .feat_proj = feat_proj,
            .pos_conv_w = pos_conv_w,
            .pos_conv_b = pos_conv_b,
            .enc_ln = enc_ln,
            .hubert = hubert,
            .sem_conv = sem_conv,
            .sem_blocks = sem_blocks,
            .aco_conv1 = aco_conv1,
            .aco_blocks = aco_blocks,
            .aco_snake_out = aco_snake_out,
            .aco_conv2 = aco_conv2,
            .fc = fc,
            .quantizers = quantizers,
        };
    }

    pub fn deinit(self: *Encoder) void {
        self.storage.deinit();
        self.* = undefined;
    }

    /// Encodes 24 kHz mono F32 samples (right-padded to a multiple of 960) into
    /// `[frames][8]` RVQ codes. Caller owns returned slice.
    pub fn encode(self: *const Encoder, gpa: Allocator, samples: []const f32) Error![]hc.Frame {
        return self.encodeObserved(gpa, samples, null);
    }

    pub fn encodeObserved(self: *const Encoder, gpa: Allocator, samples: []const f32, observer: ?Observer) Error![]hc.Frame {
        if (samples.len == 0 or samples.len > max_samples) return error.InvalidLength;
        for (samples) |s| if (!std.math.isFinite(s)) return error.InvalidAudio;

        var scratch: std.heap.ArenaAllocator = .init(gpa);
        defer scratch.deinit();
        const ws = scratch.allocator();

        // Right-pad to a whole number of 960-sample frames (DacFeatureExtractor).
        const padded_len = (std.math.divCeil(usize, samples.len, hc.samples_per_frame) catch unreachable) * hc.samples_per_frame;
        const input = try ws.alloc(f32, padded_len);
        @memset(input, 0);
        @memcpy(input[0..samples.len], samples);

        // 1. Semantic branch: resample 24k -> 16k, pad (160, 160), HuBERT, mean of 13 layers, ::2.
        const x_16k = try resample24kTo16k(ws, input);
        Observer.see(observer, .resampled_16k, x_16k);

        const x_16k_pad = try ws.alloc(f32, x_16k.len + 320);
        @memset(x_16k_pad, 0);
        @memcpy(x_16k_pad[160..][0..x_16k.len], x_16k);

        const hubert_res = try self.runHubert(ws, x_16k_pad);
        const t_frames = hubert_res.t;
        Observer.see(observer, .hubert, hubert_res.data);

        // 2. SemanticEncoder: [t_frames, 768] -> [t_frames, 768]
        const e_sem = try self.runSemanticEncoder(ws, hubert_res.data, t_frames);
        Observer.see(observer, .semantic, e_sem);

        // 3. DAC acoustic encoder (pad (480, 480) only if output length != t_frames).
        const aco_t_unpadded = acousticOutLen(input.len);
        const aco_in = if (aco_t_unpadded != t_frames) blk: {
            const pad = hc.samples_per_frame / 2;
            const p = try ws.alloc(f32, input.len + 2 * pad);
            @memset(p, 0);
            @memcpy(p[pad..][0..input.len], input);
            break :blk p;
        } else input;
        const e_aco_res = try self.runAcousticEncoder(ws, aco_in);
        if (e_aco_res.t != t_frames) return error.InvalidLength;
        Observer.see(observer, .acoustic, e_aco_res.data);

        // 4. Concatenate [acoustic (256) || semantic (768)] -> fc (1024 -> 1024).
        const cat = try ws.alloc(f32, t_frames * hc.latent_dim);
        for (0..t_frames) |t| {
            @memcpy(cat[t * hc.latent_dim ..][0..hc.acoustic_dim], e_aco_res.data[t * hc.acoustic_dim ..][0..hc.acoustic_dim]);
            @memcpy(cat[t * hc.latent_dim + hc.acoustic_dim ..][0..semantic_dim], e_sem[t * semantic_dim ..][0..semantic_dim]);
        }
        const residual = try ws.alloc(f32, t_frames * hc.latent_dim);
        try hc.runLinear(self.io, self.fc, cat, t_frames, residual);
        Observer.see(observer, .fused, residual);

        // 5. 8-codebook Euclidean RVQ.
        const out_frames = try gpa.alloc(hc.Frame, t_frames);
        errdefer gpa.free(out_frames);
        const z_e = try ws.alloc(f32, t_frames * hc.codebook_dim);
        const picked = try ws.alloc(f32, t_frames * hc.codebook_dim);
        const quant_out = try ws.alloc(f32, t_frames * hc.latent_dim);

        for (self.quantizers, 0..) |q, c| {
            try hc.runLinear(self.io, q.project_in, residual, t_frames, z_e);
            for (0..t_frames) |t| {
                const zv = z_e[t * hc.codebook_dim ..][0..hc.codebook_dim];
                const code = nearestCode(zv, q.embed, q.embed_sq);
                out_frames[t][c] = code;
                const src = q.embed[@as(usize, code) * hc.codebook_dim ..][0..hc.codebook_dim];
                @memcpy(picked[t * hc.codebook_dim ..][0..hc.codebook_dim], src);
            }
            if (c + 1 < hc.codebooks) {
                try hc.runLinear(self.io, q.project_out, picked, t_frames, quant_out);
                for (residual, quant_out) |*r, qv| r.* -= qv;
            }
        }
        return out_frames;
    }

    fn runHubert(self: *const Encoder, ws: Allocator, x_16k_pad: []const f32) (Allocator.Error || dense.Error)!struct { data: []f32, t: usize } {
        // 7-layer conv feature extractor.
        var cur: []f32 = try ws.dupe(f32, x_16k_pad);
        var cur_t: usize = x_16k_pad.len;
        for (self.feat_convs, 0..) |conv, i| {
            cur = try hc.runConv1d(self.io, ws, conv, cur, cur_t);
            cur_t = hc.conv1dOutLen(cur_t, conv.k, conv.stride, conv.padding, conv.dilation);
            if (i == 0) runGroupNormPerChannel(cur, cur_t, hubert_conv_dim, self.feat_gn0.w, self.feat_gn0.b, hubert_ln_eps);
            for (cur) |*v| v.* = cpu.geluErf(v.*);
        }

        // Feature projection: LayerNorm(512) -> Linear(512 -> 768).
        for (0..cur_t) |t| {
            const row = cur[t * hubert_conv_dim ..][0..hubert_conv_dim];
            try dense.layerNorm(row, self.feat_proj_ln.w, self.feat_proj_ln.b, hubert_ln_eps, row);
        }
        const h = try ws.alloc(f32, cur_t * semantic_dim);
        try hc.runLinear(self.io, self.feat_proj, cur, cur_t, h);

        // Positional conv embedding (16 groups of 48, k=128, pad=64, drop last 1, GELU).
        const pos = try self.runPosConv(ws, h, cur_t);
        for (h, pos) |*dst, p| dst.* += p;
        for (0..cur_t) |t| {
            const row = h[t * semantic_dim ..][0..semantic_dim];
            try dense.layerNorm(row, self.enc_ln.w, self.enc_ln.b, hubert_ln_eps, row);
        }

        // Accumulate mean over all 13 hidden states: state 0 (after pos_conv + enc_ln) + 12 layers.
        const acc = try ws.dupe(f32, h);

        const q = try ws.alloc(f32, cur_t * semantic_dim);
        const k = try ws.alloc(f32, cur_t * semantic_dim);
        const v = try ws.alloc(f32, cur_t * semantic_dim);
        const attn_out = try ws.alloc(f32, cur_t * semantic_dim);
        const attn_proj = try ws.alloc(f32, cur_t * semantic_dim);
        const attn_scratch = try ws.alloc(f32, hubert_heads * cur_t);
        const ffn_mid = try ws.alloc(f32, cur_t * hubert_ffn_dim);
        const ffn_out = try ws.alloc(f32, cur_t * semantic_dim);
        const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(hubert_head_dim)));

        for (self.hubert) |layer| {
            try hc.runLinear(self.io, layer.q, h, cur_t, q);
            try hc.runLinear(self.io, layer.k, h, cur_t, k);
            try hc.runLinear(self.io, layer.v, h, cur_t, v);
            try dense.attention(self.io, .{
                .tokens = cur_t,
                .heads = hubert_heads,
                .head_dim = hubert_head_dim,
                .window = null,
                .scale = scale,
                .stride = semantic_dim,
                .q = q,
                .k = k,
                .v = v,
            }, attn_out, attn_scratch);
            try hc.runLinear(self.io, layer.out, attn_out, cur_t, attn_proj);
            for (h, attn_proj) |*dst, d| dst.* += d;
            for (0..cur_t) |t| {
                const row = h[t * semantic_dim ..][0..semantic_dim];
                try dense.layerNorm(row, layer.ln1.w, layer.ln1.b, hubert_ln_eps, row);
            }

            try hc.runLinear(self.io, layer.fc1, h, cur_t, ffn_mid);
            for (ffn_mid) |*x| x.* = cpu.geluErf(x.*);
            try hc.runLinear(self.io, layer.fc2, ffn_mid, cur_t, ffn_out);
            for (h, ffn_out) |*dst, d| dst.* += d;
            for (0..cur_t) |t| {
                const row = h[t * semantic_dim ..][0..semantic_dim];
                try dense.layerNorm(row, layer.ln2.w, layer.ln2.b, hubert_ln_eps, row);
            }

            for (acc, h) |*sum, val| sum.* += val;
        }

        // Mean over 13 states and subsample by 2 (`[:, ::2, :]`).
        const t_sub = (cur_t + 1) / 2;
        const out = try ws.alloc(f32, t_sub * semantic_dim);
        const inv: f32 = 1.0 / @as(f32, @floatFromInt(hubert_layers + 1));
        for (0..t_sub) |i| {
            const src = acc[(i * 2) * semantic_dim ..][0..semantic_dim];
            const dst = out[i * semantic_dim ..][0..semantic_dim];
            for (dst, src) |*d, s| d.* = s * inv;
        }
        return .{ .data = out, .t = t_sub };
    }

    fn runPosConv(self: *const Encoder, ws: Allocator, x: []const f32, t_len: usize) (Allocator.Error || dense.Error)![]f32 {
        // Grouped Conv1d: 16 groups of 48 ch, k=128, pad=64 -> length t_len+1, SamePad drops last 1 -> t_len.
        const out = try ws.alloc(f32, t_len * semantic_dim);
        const inner = pos_conv_kernel * pos_conv_group_ch;
        const cols = try ws.alloc(f32, t_len * inner);
        const g_out = try ws.alloc(f32, t_len * pos_conv_group_ch);
        const pad_i: isize = pos_conv_kernel / 2;
        const t_i: isize = @intCast(t_len);

        for (0..pos_conv_groups) |g| {
            const ch_off = g * pos_conv_group_ch;
            @memset(cols, 0);
            for (0..t_len) |to| {
                const base: isize = @as(isize, @intCast(to)) - pad_i;
                const dst_row = cols[to * inner ..][0..inner];
                for (0..pos_conv_kernel) |tap| {
                    const ti = base + @as(isize, @intCast(tap));
                    if (ti >= 0 and ti < t_i) {
                        const src = x[@as(usize, @intCast(ti)) * semantic_dim + ch_off ..][0..pos_conv_group_ch];
                        @memcpy(dst_row[tap * pos_conv_group_ch ..][0..pos_conv_group_ch], src);
                    }
                }
            }
            const bias = self.pos_conv_b[ch_off..][0..pos_conv_group_ch];
            try dense.matmul(self.io, .{ .rows = t_len, .inner = inner, .outputs = pos_conv_group_ch, .x = cols, .w = self.pos_conv_w[g], .bias = bias }, g_out);
            for (0..t_len) |to| {
                const dst = out[to * semantic_dim + ch_off ..][0..pos_conv_group_ch];
                const src = g_out[to * pos_conv_group_ch ..][0..pos_conv_group_ch];
                for (dst, src) |*d, s| d.* = cpu.geluErf(s);
            }
        }
        return out;
    }

    fn runSemanticEncoder(self: *const Encoder, ws: Allocator, x: []const f32, t_len: usize) (Allocator.Error || dense.Error)![]f32 {
        var cur = try hc.runConv1d(self.io, ws, self.sem_conv, x, t_len);
        for (self.sem_blocks) |blk| {
            for (blk.units) |u| {
                const h = try ws.dupe(f32, cur);
                for (h) |*v| v.* = elu(v.*);
                const h1 = try hc.runConv1d(self.io, ws, u.conv1, h, t_len);
                for (h1) |*v| v.* = elu(v.*);
                const h2 = try hc.runConv1d(self.io, ws, u.conv2, h1, t_len);
                for (cur, h2) |*dst, d| dst.* += d;
            }
            cur = try hc.runConv1d(self.io, ws, blk.conv, cur, t_len);
        }
        return cur;
    }

    fn runAcousticEncoder(self: *const Encoder, ws: Allocator, x: []const f32) (Allocator.Error || dense.Error)!struct { data: []f32, t: usize } {
        var cur = try hc.runConv1d(self.io, ws, self.aco_conv1, x, x.len);
        var cur_t = x.len;
        for (self.aco_blocks) |blk| {
            for (blk.units) |u| try hc.runResUnit(self.io, ws, u, cur, cur_t);
            hc.runSnake(cur, cur_t, blk.snake1);
            cur = try hc.runConv1d(self.io, ws, blk.conv1, cur, cur_t);
            cur_t = hc.conv1dOutLen(cur_t, blk.conv1.k, blk.conv1.stride, blk.conv1.padding, blk.conv1.dilation);
        }
        hc.runSnake(cur, cur_t, self.aco_snake_out);
        cur = try hc.runConv1d(self.io, ws, self.aco_conv2, cur, cur_t);
        return .{ .data = cur, .t = cur_t };
    }
};

fn acousticOutLen(samples: usize) usize {
    var len = samples;
    for (hc.rates) |s| len = hc.conv1dOutLen(len, 2 * s, s, (s + 1) / 2, 1);
    return len;
}

fn elu(x: f32) f32 {
    return if (x > 0.0) x else @exp(x) - 1.0;
}

/// GroupNorm with `num_groups == num_channels` (InstanceNorm per channel over time).
fn runGroupNormPerChannel(x: []f32, t_len: usize, channels: usize, w: []const f32, b: []const f32, eps: f32) void {
    const n: f64 = @floatFromInt(t_len);
    for (0..channels) |c| {
        var sum: f64 = 0;
        for (0..t_len) |t| sum += x[t * channels + c];
        const mean = sum / n;
        var sq: f64 = 0;
        for (0..t_len) |t| {
            const d = @as(f64, x[t * channels + c]) - mean;
            sq += d * d;
        }
        const inv = 1.0 / @sqrt(sq / n + @as(f64, eps));
        const wc = w[c];
        const bc = b[c];
        for (0..t_len) |t| {
            const norm: f32 = @floatCast((@as(f64, x[t * channels + c]) - mean) * inv);
            x[t * channels + c] = norm * wc + bc;
        }
    }
}

/// Euclidean nearest codeword: `argmax_k (2 * z·e_k - ||e_k||^2)`.
fn nearestCode(z: []const f32, embed: []const f32, embed_sq: []const f32) u16 {
    var best_idx: u16 = 0;
    var best_score: f32 = -std.math.inf(f32);
    for (0..hc.codebook_size) |k| {
        const e = embed[k * hc.codebook_dim ..][0..hc.codebook_dim];
        var dot: f32 = 0;
        for (z, e) |zv, ev| dot += zv * ev;
        const score = 2.0 * dot - embed_sq[k];
        if (score > best_score) {
            best_score = score;
            best_idx = @intCast(k);
        }
    }
    return best_idx;
}

/// Hann-windowed sinc resampling from 24 kHz to 16 kHz, matching
/// `torchaudio.functional.resample(x, 24000, 16000)` (`lowpass_filter_width=6`, `rolloff=0.99`).
pub fn resample24kTo16k(alloc: Allocator, input: []const f32) Allocator.Error![]f32 {
    const orig_freq: usize = 3; // 24000 / gcd(24000, 16000)
    const new_freq: usize = 2; // 16000 / gcd(24000, 16000)
    const lowpass_width: f64 = 6.0;
    const rolloff: f64 = 0.99;
    const base_freq: f64 = @as(f64, new_freq) * rolloff; // 1.98
    const width: usize = @intFromFloat(@ceil(lowpass_width * @as(f64, orig_freq) / base_freq)); // 10
    const k_len: usize = 2 * width + orig_freq; // 23
    const scale: f64 = base_freq / @as(f64, orig_freq);

    var kernels: [new_freq][k_len]f32 = undefined;
    for (0..new_freq) |phase| {
        const p_f: f64 = -@as(f64, @floatFromInt(phase)) / @as(f64, new_freq);
        for (0..k_len) |j| {
            const idx: f64 = (@as(f64, @floatFromInt(j)) - @as(f64, @floatFromInt(width))) / @as(f64, orig_freq);
            var t: f64 = (p_f + idx) * base_freq;
            t = std.math.clamp(t, -lowpass_width, lowpass_width);
            const c = @cos(t * std.math.pi / lowpass_width / 2.0);
            const win = c * c;
            const tp = t * std.math.pi;
            const sinc = if (tp == 0.0) 1.0 else @sin(tp) / tp;
            kernels[phase][j] = @floatCast(sinc * win * scale);
        }
    }

    const target_len: usize = std.math.divCeil(usize, input.len * new_freq, orig_freq) catch unreachable;
    const out = try alloc.alloc(f32, target_len);
    const pad_left: isize = @intCast(width);
    const in_len_i: isize = @intCast(input.len);
    const blocks = (input.len + 2 * width + orig_freq - k_len) / orig_freq + 1;

    var written: usize = 0;
    for (0..blocks) |blk| {
        const base: isize = @as(isize, @intCast(blk * orig_freq)) - pad_left;
        for (0..new_freq) |phase| {
            if (written >= target_len) break;
            var sum: f32 = 0;
            for (0..k_len) |j| {
                const pos = base + @as(isize, @intCast(j));
                if (pos >= 0 and pos < in_len_i) {
                    sum += input[@intCast(pos)] * kernels[phase][j];
                }
            }
            out[written] = sum;
            written += 1;
        }
    }
    return out;
}

fn fmt(buf: *[192]u8, comptime format: []const u8, args: anytype) []const u8 {
    return std.fmt.bufPrint(buf, format, args) catch unreachable;
}

// ---- tests -----------------------------------------------------------------

test "resample24kTo16k preserves DC gain and exact length" {
    const gpa = std.testing.allocator;
    const dc = try gpa.alloc(f32, 960);
    defer gpa.free(dc);
    @memset(dc, 1.0);
    const out = try resample24kTo16k(gpa, dc);
    defer gpa.free(out);
    try std.testing.expectEqual(@as(usize, 640), out.len);
    // Interior samples preserve DC = 1.0 within the truncated Hann-sinc passband ripple (~4.2e-4).
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), out[320], 1e-3);
}

test "GroupNorm per channel and ELU match reference definitions" {
    var x = [_]f32{ 1.0, 10.0, 3.0, 20.0 }; // T=2, C=2
    runGroupNormPerChannel(&x, 2, 2, &.{ 1.0, 2.0 }, &.{ 0.0, 1.0 }, 1e-5);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), x[0], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 1.0), x[2], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, -1.0), x[1], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 3.0), x[3], 1e-4);
    try std.testing.expectApproxEqAbs(@as(f32, 2.5), elu(2.5), 1e-6);
    try std.testing.expectApproxEqAbs(@as(f32, @exp(-1.0) - 1.0), elu(-1.0), 1e-6);
}

test "Encoder input bounds reject empty, overlong, and non-finite clips" {
    var dummy: Encoder = undefined;
    try std.testing.expectError(error.InvalidLength, dummy.encode(std.testing.allocator, &.{}));
    try std.testing.expectError(error.InvalidAudio, dummy.encode(std.testing.allocator, &.{std.math.nan(f32)}));
    try std.testing.expectError(error.InvalidAudio, dummy.encode(std.testing.allocator, &.{std.math.inf(f32)}));
}
