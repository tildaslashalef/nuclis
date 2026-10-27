//! The Metal library's source and its kernel table: the codebooks as MSL
//! constants, `dequant.metal`, and the `kernels/` files joined in order, and
//! `Kernel`, whose tags name the entry points.
const std = @import("std");

/// The IQ codebooks are emitted into the shader source at compile time from
/// their single Zig definitions, so the CPU and GPU decoders share one table.
fn codebookSource(comptime msl_type: []const u8, comptime suffix: []const u8, comptime name: []const u8, comptime values: anytype) []const u8 {
    @setEvalBranchQuota(20_000_000);
    var text: []const u8 = std.fmt.comptimePrint("constant {s} {s}[{d}] = {{\n", .{ msl_type, name, values.len });
    for (values, 0..) |value, i| {
        text = text ++ std.fmt.comptimePrint("0x{x}{s},", .{ value, suffix });
        if (i % 16 == 15) text = text ++ "\n";
    }
    return text ++ "};\n";
}
/// An IQ2 grid as 2-bit codes: magnitude 8, 25, 43 of component j as 0, 1, 2 at bits 2j.
fn iq2Codes(comptime grid: anytype) [grid.len]u16 {
    @setEvalBranchQuota(20_000_000);
    var codes: [grid.len]u16 = undefined;
    for (&codes, grid) |*code, entry| {
        code.* = 0;
        for (0..8) |j| {
            const magnitude: u8 = @truncate(entry >> @intCast(8 * j));
            const c: u16 = switch (magnitude) {
                8 => 0,
                25 => 1,
                43 => 2,
                else => @compileError("IQ2 magnitude outside {8, 25, 43}"),
            };
            code.* |= c << @intCast(2 * j);
        }
    }
    return codes;
}
/// The IQ1 grid as 2-bit codes: component j + 1 (0, 1, 2) at bits 2j.
fn iq1Codes() [2048]u16 {
    @setEvalBranchQuota(20_000_000);
    var codes: [2048]u16 = undefined;
    for (&codes, codebooks.iq1_s) |*code, entry| {
        code.* = 0;
        for (0..8) |j| {
            const component: i8 = @bitCast(@as(u8, @truncate(entry >> @intCast(8 * j))));
            code.* |= @as(u16, @intCast(component + 1)) << @intCast(2 * j);
        }
    }
    return codes;
}
const codebooks = @import("../../quant/iq-codebooks.zig");
const codebook_source = codebookSource("uint", "u", "nu_iq3_grid", @import("../../quant/iq3-grid.zig").values) ++
    codebookSource("ulong", "ul", "nu_iq2_xxs_grid", codebooks.iq2_xxs) ++
    codebookSource("ulong", "ul", "nu_iq2_xs_grid", codebooks.iq2_xs) ++
    codebookSource("ulong", "ul", "nu_iq2_s_grid", codebooks.iq2_s) ++
    codebookSource("uint", "u", "nu_iq3_xxs_grid", codebooks.iq3_xxs) ++
    codebookSource("ulong", "ul", "nu_iq1_s_grid", codebooks.iq1_s) ++
    codebookSource("ushort", "", "nu_iq1_codes", iq1Codes()) ++
    codebookSource("ushort", "", "nu_iq2_s_codes", iq2Codes(codebooks.iq2_s));
/// The kernels, one file per kind of operation, joined in this order into one
/// library: later files call helpers defined in earlier ones (`common.metal`
/// first), so the order is part of the source.
const kernel_files = [_][]const u8{
    "kernels/common.metal",       "kernels/matvec.metal",          "kernels/matvec_codebook.metal",
    "kernels/matvec_split.metal", "kernels/matvec_rows.metal",     "kernels/experts.metal",
    "kernels/matmul.metal",       "kernels/rowwise.metal",         "kernels/recurrent.metal",
    "kernels/attention.metal",    "kernels/recurrent_chunk.metal", "kernels/sampling.metal",
    "kernels/encoders.metal",     "kernels/audio.metal",           "kernels/asr.metal",
};
const kernels_source = blk: {
    var joined: []const u8 = "";
    for (kernel_files) |file| joined = joined ++ @embedFile(file);
    break :blk joined;
};
pub const source = "#include <metal_stdlib>\nusing namespace metal;\n" ++ codebook_source ++ @embedFile("dequant.metal") ++ "\n" ++ kernels_source;

/// Each `Kernel` names its MSL entry point as `nu_` plus the tag; pipelines
/// are indexed by the tag's value, so the two cannot drift apart.
pub const kernel_names = blk: {
    const info = @typeInfo(Kernel).@"enum";
    var names: [info.field_names.len][:0]const u8 = undefined;
    for (info.field_names, info.field_values, &names, 0..) |tag, value, *name, i| {
        if (value != i) @compileError("Kernel values must be 0..n-1 in declaration order");
        name.* = "nu_" ++ tag;
    }
    break :blk names;
};
pub const Kernel = enum(u32) {
    matvec,
    matvec_q4_k,
    matvec_q5_k,
    matvec_q6_k,
    matvec_iq4_xs,
    embed,
    rmsnorm,
    l2norm,
    rope,
    add,
    silu_mul,
    dflash_conv,
    silu_inplace,
    delta_gates,
    sigmoid_gate,
    delta,
    convolution,
    attention_scores,
    attention_softmax,
    attention_values,
    penalize,
    argmax_partial,
    argmax_final,
    matvec_q3_k,
    matvec_iq3_s,
    matvec_segments,
    topk_partial,
    topk_final,
    expsum_partial,
    matmul,
    rope_rows,
    copy,
    convolution_rows,
    convolution_history,
    attention_chunk,
    delta_chunk,
    matmul_q3_k,
    matmul_q4_k,
    matmul_q5_k,
    matmul_q6_k,
    matmul_iq3_s,
    matmul_iq4_xs,
    matmul_q3_k_32,
    matmul_q4_k_32,
    matmul_q5_k_32,
    matmul_q6_k_32,
    matmul_iq3_s_32,
    matmul_iq4_xs_32,
    attention_scores_h,
    attention_values_h,
    attention_chunk_h,
    pack_half,
    attention_decode,
    attention_decode_h,
    attention_merge,
    gelu_mul,
    scale,
    add_scale,
    softcap,
    attention_decode_w,
    attention_decode_wh,
    matvec_q4_0,
    matmul_q4_0,
    matmul_q4_0_32,
    matvec_experts,
    route,
    combine_experts,
    gelu_mul_rows,
    expert_lists,
    matmul_experts,
    matmul_experts_q4_0,
    matvec_pq2_0,
    matvec_ptq1_0,
    matmul_pq2_0,
    matmul_ptq1_0,
    matmul_pq2_0_32,
    matmul_ptq1_0_32,
    hadamard,
    gather_rows,
    matmul_q3_k_8,
    matmul_q4_k_8,
    matmul_q5_k_8,
    matmul_q6_k_8,
    matmul_iq3_s_8,
    matmul_iq4_xs_8,
    matmul_q4_0_8,
    matmul_pq2_0_8,
    matmul_ptq1_0_8,
    matvec_rows,
    matvec_rows_q4_k_t2,
    matvec_rows_q4_k_t3,
    matvec_rows_q4_k_t4,
    matvec_rows_q4_k_t5,
    matvec_rows_q4_k_t6,
    matvec_rows_q4_k_t7,
    matvec_rows_q4_k_t8,
    matvec_rows_q5_k_t2,
    matvec_rows_q5_k_t3,
    matvec_rows_q5_k_t4,
    matvec_rows_q5_k_t5,
    matvec_rows_q5_k_t6,
    matvec_rows_q5_k_t7,
    matvec_rows_q5_k_t8,
    matvec_rows_q6_k_t2,
    matvec_rows_q6_k_t3,
    matvec_rows_q6_k_t4,
    matvec_rows_q6_k_t5,
    matvec_rows_q6_k_t6,
    matvec_rows_q6_k_t7,
    matvec_rows_q6_k_t8,
    matvec_rows_iq4_xs_t2,
    matvec_rows_iq4_xs_t3,
    matvec_rows_iq4_xs_t4,
    matvec_rows_iq4_xs_t5,
    matvec_rows_iq4_xs_t6,
    matvec_rows_iq4_xs_t7,
    matvec_rows_iq4_xs_t8,
    matvec_rows_iq2_xxs_t2,
    matvec_rows_iq2_xxs_t3,
    matvec_rows_iq2_xxs_t4,
    matvec_rows_iq2_xxs_t5,
    matvec_rows_iq2_xxs_t6,
    matvec_rows_iq2_xxs_t7,
    matvec_rows_iq2_xxs_t8,
    matmul_q3_k_w8,
    matmul_q4_k_w8,
    matmul_q5_k_w8,
    matmul_q6_k_w8,
    matmul_iq3_s_w8,
    matmul_iq4_xs_w8,
    matmul_q4_0_w8,
    matmul_pq2_0_w8,
    matmul_ptq1_0_w8,
    matvec_q4_k_split,
    matvec_q5_k_split,
    reduce_splits,
    matvec_segments_split,
    segment_reduce_splits,
    attention_chunk_reuse,
    attention_chunk_reuse_h,
    rmsnorm_add,
    add_rmsnorm,
    rmsnorm_rope,
    layernorm,
    add_bias_rows,
    gelu_inplace,
    attention_full,
    gelu_quick_mul,
    gelu_erf_inplace,
    clamp,
    attention_segments,
    gelu_erf_mul_rows,
    fragment_layout,
    matmul_q4_k_f2,
    matmul_q4_k_f2hh,
    matmul_q4_k_f4,
    matmul_q5_k_f2,
    matmul_q6_k_f2,
    matmul_iq4_xs_f2,
    matmul_q3_k_f2,
    matmul_iq3_s_f2,
    matmul_q4_0_f2,
    matmul_q8_0_f2,
    matmul_pq2_0_f2,
    matmul_ptq1_0_f2,
    matmul_iq4_nl_f2,
    delta_rows,
    delta_replay,
    segments_scores,
    segments_softmax,
    segments_values,
    audio_attention,
    glu_conv,
    matvec_iq2_xxs,
    matvec_iq2_xs,
    matvec_iq2_s,
    matvec_iq3_xxs,
    matvec_q2_k,
    matvec_iq1_s,
    matvec_iq1_m,
    matmul_q2_k,
    matmul_q2_k_32,
    matmul_q2_k_8,
    matmul_q2_k_f2,
    matmul_iq2_xxs,
    matmul_iq2_xxs_32,
    matmul_iq2_xxs_8,
    matmul_iq2_xxs_f2,
    matmul_iq2_xs,
    matmul_iq2_xs_32,
    matmul_iq2_xs_8,
    matmul_iq2_xs_f2,
    matmul_iq3_xxs,
    matmul_iq3_xxs_32,
    matmul_iq3_xxs_8,
    matmul_iq3_xxs_f2,
    matmul_iq1_s,
    matmul_iq1_s_32,
    matmul_iq1_s_8,
    matmul_iq1_s_f2,
    matmul_iq2_s,
    matmul_iq2_s_32,
    matmul_iq2_s_8,
    matmul_iq2_s_f2,
    matmul_iq1_m,
    matmul_iq1_m_32,
    matmul_iq1_m_8,
    matmul_iq1_m_f2,
    pack_q8_0,
    unpack_q8_0,
    attention_decode_q8,
    attention_decode_wq8,
    attention_heads_f32_128,
    attention_heads_f32_256,
    attention_heads_f32_512,
    attention_heads_f16_128,
    attention_heads_f16_256,
    attention_heads_f16_512,
    attention_heads_q8_128,
    attention_heads_q8_256,
    attention_heads_q8_512,
    asr_conv2d,
    asr_bias_relu,
    asr_shift,
    asr_attention,
    asr_glu,
    asr_depthwise,
    asr_joint,
    asr_add_scaled,
};
