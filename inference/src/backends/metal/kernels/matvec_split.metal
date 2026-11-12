// Merged projections and the specialized matvec instantiations:
// `nu_matvec_segments` runs up to four projections of one input in one
// dispatch (plain, or a SiLU / GELU gate pair), each segment through its
// encoding's decode body (`nu_segment_sums`). The forced split-K forms of the
// matvec and of the merge live in matvec_split_k.metal, compiled only into
// benchmark builds; `nu_segment_sums` keeps the K-range bounds they pass.

// A segment owns whole 16-row threadgroups, so encoding and binding switches
// are uniform within a SIMD group. Bit 31 requests generic arithmetic when
// alignment prevents specialized vector loads (or for a generic-only run).
struct MatvecSegment { uint rows, weight_slot, weight_offset, encoding, stride, output_slot, output_offset; };
// `splits` > 1 selects the benchmark-only split path, which needs
// `partials_slot` to name a free binding slot for its row-partial buffer and
// `total_rows` its row count; `nu_matvec_segments` ignores all four.
struct MatvecSegments { uint columns, blocks, count, mode, splits, split_blocks, total_rows, partials_slot; MatvecSegment segments[4]; };
inline device uchar * nu_segment_buffer(uint slot, device uchar * b1, device uchar * b2, device uchar * b3, device uchar * b4, device uchar * b5, device uchar * b6) {
    switch (slot) { case 1: return b1; case 2: return b2; case 3: return b3; case 4: return b4; case 5: return b5; default: return b6; }
}
inline void nu_segment_sums(device const uchar * w, device const float * x, MatvecBlockParams p, uint encoding, uint group, uint sg, uint lane, thread float * acc, uint first, uint last, threadgroup const float * iq4) {
    // Bodies produce the same lane partials as their standalone kernels.
    // The generic branch retains one lane's original 16-value segment order.
    // `first`/`last` carry the split's K-range bounds into the two encodings
    // that have a split body; the others have no K-range support (the host
    // refuses to force a split for them).
    switch (encoding) {
        case 2: nu_matvec_q4_0_body<4>(w, x, p, group, sg, lane, acc); return;
        case 6: nu_matvec_five_body<4, false>(w, x, p, group, sg, lane, acc); return;
        case 7: nu_matvec_five_body<4, true>(w, x, p, group, sg, lane, acc); return;
        case 11: nu_matvec_three_body<4>(w, x, p, group, sg, lane, acc); return;
        case 12: nu_matvec_k_body<4, false>(w, x, p, group, sg, lane, acc, first, last); return;
        case 13: nu_matvec_k_body<4, true>(w, x, p, group, sg, lane, acc, first, last); return;
        case 14: nu_matvec_q6_k_body<4>(w, x, p, group, sg, lane, acc); return;
        case 21: nu_matvec_iq3_s_body<4>(w, x, p, group, sg, lane, acc); return;
        case 23: nu_matvec_iq4_xs_body<4>(w, x, p, group, sg, lane, acc, iq4); return;
        // IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, Q2_K, IQ1_S, and IQ1_M never
        // arrive specialized: the host declines their merges (`mergeDeclined`
        // in ops/matvec.zig), so their segments take the generic branch.
        case 142: nu_matvec_pq2_0_body<4>(w, x, p, group, sg, lane, acc); return;
        case 143: nu_matvec_ptq1_0_body<4>(w, x, p, group, sg, lane, acc); return;
        default:
            uint row0 = (group * NU_MATVEC_SIMDGROUPS + sg) * 4;
            for (uint r = 0; r < 4; ++r) {
                float sum = 0, values[16];
                for (uint segment = lane; segment < p.columns / 16; segment += 32) {
                    nu_segment(w + ulong(min(row0 + r, p.rows - 1)) * p.stride, encoding & 0x7fffffffu, segment, values);
                    for (uint j = 0; j < 16; ++j) sum += values[j] * x[segment * 16 + j];
                }
                acc[r] = sum;
            }
    }
}
kernel void nu_matvec_segments(device const float * input [[buffer(0)]],
    device uchar * b1 [[buffer(1)]], device uchar * b2 [[buffer(2)]], device uchar * b3 [[buffer(3)]],
    device uchar * b4 [[buffer(4)]], device uchar * b5 [[buffer(5)]], device uchar * b6 [[buffer(6)]],
    constant MatvecSegments & p [[buffer(7)]], uint group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint index = 0, local = group;
    if (p.mode == 0) {
        while (index + 1 < p.count && local >= p.segments[index].rows / 16) {
            local -= p.segments[index].rows / 16;
            ++index;
        }
    }
    // Pair modes (1: silu, 2: gelu) give two SIMD groups to gate and two to
    // up. Each group computes eight output rows, retaining only one
    // projection's lane partials. The virtual group/sg pair preserves each
    // body's original row arithmetic.
    threadgroup float pair_values[16];
    const bool pair = p.mode != 0;
    if (pair) { index = sg / 2; local = group / 2; }
    uint body_sg = pair ? (group % 2) * 2 + sg % 2 : sg;
    MatvecSegment s = p.segments[index];
    MatvecBlockParams shape = { p.columns, s.stride, s.rows, p.blocks };
    device const uchar * w = nu_segment_buffer(s.weight_slot, b1,b2,b3,b4,b5,b6) + s.weight_offset;
    device float * out = (device float *)(nu_segment_buffer(s.output_slot, b1,b2,b3,b4,b5,b6) + s.output_offset);
    threadgroup float iq4[16];
    nu_iq4_table(iq4, sg, lane);
    float a[4];
    nu_segment_sums(w, input, shape, s.encoding, local, body_sg, lane, a, 0u, p.blocks, iq4);
    uint row0 = (local * NU_MATVEC_SIMDGROUPS + sg) * 4;
    if (pair) {
        for (uint r = 0; r < 4; ++r) {
            float value = simd_sum(a[r]);
            if (lane == 0) pair_values[(sg / 2) * 8 + (sg % 2) * 4 + r] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0 && sg < 2) for (uint r = 0; r < 4; ++r) {
            uint i = sg * 4 + r;
            const float gate = p.mode == 1 ? nu_silu(pair_values[i]) : nu_gelu(pair_values[i]);
            out[group * 8 + i] = gate * pair_values[8 + i];
        }
    } else nu_store_rows<4>(a, out, row0, s.rows, lane);
}

// Host-visible instantiations. ROWS must match `rows_per_simdgroup` (4) or
// `matvecRowsPerSimdgroup` (2) in ops/matvec.zig.
#define NU_MATVEC_ARGS device const uchar *, device const float *, device float *, constant MatvecBlockParams &, uint, uint, uint
template [[host_name("nu_matvec_q3_k")]] kernel void nu_matvec_three<4>(NU_MATVEC_ARGS);
template [[host_name("nu_matvec_iq3_s")]] kernel void nu_matvec_iq3_s<4>(NU_MATVEC_ARGS);
template [[host_name("nu_matvec_q4_k")]] kernel void nu_matvec_q4_k<4>(NU_MATVEC_ARGS);
template [[host_name("nu_matvec_q5_k")]] kernel void nu_matvec_q5_k<4>(NU_MATVEC_ARGS);
template [[host_name("nu_matvec_q6_k")]] kernel void nu_matvec_q6_k<4>(NU_MATVEC_ARGS);
template [[host_name("nu_matvec_iq4_xs")]] kernel void nu_matvec_iq4_xs<4>(NU_MATVEC_ARGS);
template [[host_name("nu_matvec_iq2_xxs")]] kernel void nu_matvec_iq2_xxs<2>(NU_MATVEC_ARGS);
template [[host_name("nu_matvec_iq2_xs")]] kernel void nu_matvec_iq2_xs<2>(NU_MATVEC_ARGS);
template [[host_name("nu_matvec_iq2_s")]] kernel void nu_matvec_iq2_s<2>(NU_MATVEC_ARGS);
template [[host_name("nu_matvec_iq3_xxs")]] kernel void nu_matvec_iq3_xxs<4>(NU_MATVEC_ARGS);
template [[host_name("nu_matvec_q2_k")]] kernel void nu_matvec_q2_k<2>(NU_MATVEC_ARGS);
template [[host_name("nu_matvec_iq1_s")]] kernel void nu_matvec_iq1_s<4>(NU_MATVEC_ARGS);
template [[host_name("nu_matvec_iq1_m")]] kernel void nu_matvec_iq1_m<2>(NU_MATVEC_ARGS);
template [[host_name("nu_matvec_q4_0")]] kernel void nu_matvec_q4_0<4>(NU_MATVEC_ARGS);
template [[host_name("nu_matvec_q5_0")]] kernel void nu_matvec_five<4, false>(NU_MATVEC_ARGS);
template [[host_name("nu_matvec_q5_1")]] kernel void nu_matvec_five<4, true>(NU_MATVEC_ARGS);
template [[host_name("nu_matvec_pq2_0")]] kernel void nu_matvec_pq2_0<4>(NU_MATVEC_ARGS);
template [[host_name("nu_matvec_ptq1_0")]] kernel void nu_matvec_ptq1_0<4>(NU_MATVEC_ARGS);

