// Split-K decode matvec: `splits` threadgroups per 16-row block,
// each summing only its share of the K blocks into a row-major partial
// buffer `[split][rows]`; `nu_reduce_splits` then sums the splits into the
// output. The K-range bounds passed to the shared decode body keep its
// per-lane loops identical to the non-split form, so the partials differ from
// a single-pass sum only by F32 rounding order. Measured behind the
// single-pass kernel on every row bucket (2026-09-21), so nothing routes here;
// the sweep and the fixture force it through the host's `matvecSplits` /
// `matvecSegmentsSplits`.
struct MatvecSplitParams { uint columns; uint stride; uint rows; uint blocks; uint splits; uint split_blocks; };
template <bool FIFTH_BIT>
inline void nu_matvec_k_split(device const uchar * weights, device const float * input, device float * partials, constant MatvecSplitParams & p,
                              uint group, uint sg, uint lane) {
    const uint split = group % p.splits, row_group = group / p.splits;
    MatvecBlockParams shape = { p.columns, p.stride, p.rows, p.blocks };
    float acc[4];
    nu_matvec_k_body<4, FIFTH_BIT>(weights, input, shape, row_group, sg, lane, acc, split * p.split_blocks, min((split + 1) * p.split_blocks, p.blocks));
    const uint row0 = (row_group * NU_MATVEC_SIMDGROUPS + sg) * 4;
    for (uint r = 0; r < 4; ++r) {
        float total = simd_sum(acc[r]);
        if (lane == 0 && row0 + r < p.rows) partials[split * p.rows + row0 + r] = total;
    }
}
kernel void nu_matvec_q4_k_split(device const uchar * weights [[buffer(0)]], device const float * input [[buffer(1)]], device float * partials [[buffer(2)]],
                                 constant MatvecSplitParams & p [[buffer(7)]], uint group [[threadgroup_position_in_grid]],
                                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    nu_matvec_k_split<false>(weights, input, partials, p, group, sg, lane);
}
kernel void nu_matvec_q5_k_split(device const uchar * weights [[buffer(0)]], device const float * input [[buffer(1)]], device float * partials [[buffer(2)]],
                                 constant MatvecSplitParams & p [[buffer(7)]], uint group [[threadgroup_position_in_grid]],
                                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    nu_matvec_k_split<true>(weights, input, partials, p, group, sg, lane);
}
struct ReduceSplitsParams { uint rows; uint splits; };
kernel void nu_reduce_splits(device const float * partials [[buffer(0)]], device float * output [[buffer(1)]],
                             constant ReduceSplitsParams & p [[buffer(7)]], uint row [[thread_position_in_grid]]) {
    if (row >= p.rows) return;
    float total = 0;
    for (uint s = 0; s < p.splits; ++s) total += partials[s * p.rows + row];
    output[row] = total;
}


// A segment owns whole 16-row threadgroups, so encoding and binding switches
// are uniform within a SIMD group. Bit 31 requests generic arithmetic when
// alignment prevents specialized vector loads (or for a generic-only run).
struct MatvecSegment { uint rows, weight_slot, weight_offset, encoding, stride, output_slot, output_offset; };
// `splits` > 1 selects the split path, which needs `partials_slot` to name a
// free binding slot for its row-partial buffer and `total_rows` its row
// count; the non-split kernel ignores all four.
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
        case 11: nu_matvec_three_body<4>(w, x, p, group, sg, lane, acc); return;
        case 12: nu_matvec_k_body<4, false>(w, x, p, group, sg, lane, acc, first, last); return;
        case 13: nu_matvec_k_body<4, true>(w, x, p, group, sg, lane, acc, first, last); return;
        case 14: nu_matvec_q6_k_body<4>(w, x, p, group, sg, lane, acc); return;
        case 21: nu_matvec_iq3_s_body<4>(w, x, p, group, sg, lane, acc); return;
        case 23: nu_matvec_iq4_xs_body<4>(w, x, p, group, sg, lane, acc, iq4); return;
        // The seven 2-bit encodings never arrive specialized: the host
        // declines their merges (`mergeDeclined` in root.zig).
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

// Plain-mode segment split: each group sums its K range into the row-partial
// buffer `[split][global row]` (the same row-to-segment walk as the kernel
// above); `nu_segment_reduce_splits` sums the splits and scatters every row
// back into its segment's output. Only specialized segments are eligible:
// the generic branch reads the whole K regardless of the range.
kernel void nu_matvec_segments_split(device const float * input [[buffer(0)]],
    device uchar * b1 [[buffer(1)]], device uchar * b2 [[buffer(2)]], device uchar * b3 [[buffer(3)]],
    device uchar * b4 [[buffer(4)]], device uchar * b5 [[buffer(5)]], device uchar * b6 [[buffer(6)]],
    constant MatvecSegments & p [[buffer(7)]], uint group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint split = group % p.splits, row_group = group / p.splits;
    const uint first = split * p.split_blocks;
    uint index = 0, local = row_group;
    while (index + 1 < p.count && local >= p.segments[index].rows / 16) {
        local -= p.segments[index].rows / 16;
        ++index;
    }
    MatvecSegment s = p.segments[index];
    MatvecBlockParams shape = { p.columns, s.stride, s.rows, p.blocks };
    device const uchar * w = nu_segment_buffer(s.weight_slot, b1,b2,b3,b4,b5,b6) + s.weight_offset;
    device float * partials = (device float *)nu_segment_buffer(p.partials_slot, b1,b2,b3,b4,b5,b6);
    threadgroup float iq4[16];
    nu_iq4_table(iq4, sg, lane);
    float a[4];
    nu_segment_sums(w, input, shape, s.encoding, local, sg, lane, a, first, min(first + p.split_blocks, p.blocks), iq4);
    uint base = 0;
    for (uint i = 0; i < index; ++i) base += p.segments[i].rows;
    const uint row0 = (local * NU_MATVEC_SIMDGROUPS + sg) * 4;
    for (uint r = 0; r < 4; ++r) {
        float total = simd_sum(a[r]);
        if (lane == 0 && row0 + r < s.rows) partials[split * p.total_rows + base + row0 + r] = total;
    }
}
kernel void nu_segment_reduce_splits(
    device uchar * b1 [[buffer(1)]], device uchar * b2 [[buffer(2)]], device uchar * b3 [[buffer(3)]],
    device uchar * b4 [[buffer(4)]], device uchar * b5 [[buffer(5)]], device uchar * b6 [[buffer(6)]],
    constant MatvecSegments & p [[buffer(7)]], uint row [[thread_position_in_grid]]) {
    if (row >= p.total_rows) return;
    uint index = 0, local = row;
    while (index + 1 < p.count && local >= p.segments[index].rows) {
        local -= p.segments[index].rows;
        ++index;
    }
    device const float * partials = (device const float *)nu_segment_buffer(p.partials_slot, b1,b2,b3,b4,b5,b6);
    float total = 0;
    for (uint s = 0; s < p.splits; ++s) total += partials[s * p.total_rows + row];
    MatvecSegment s = p.segments[index];
    device float * out = (device float *)(nu_segment_buffer(s.output_slot, b1,b2,b3,b4,b5,b6) + s.output_offset);
    out[local] = total;
}

// Host-visible instantiations. ROWS must match `rows_per_simdgroup` in root.zig.
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
template [[host_name("nu_matvec_pq2_0")]] kernel void nu_matvec_pq2_0<4>(NU_MATVEC_ARGS);
template [[host_name("nu_matvec_ptq1_0")]] kernel void nu_matvec_ptq1_0<4>(NU_MATVEC_ARGS);

