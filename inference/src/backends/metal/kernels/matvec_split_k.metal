// Benchmark-only kernels, compiled only into the `bench_kernels` module
// instance that `metal-check` imports; the application's library holds none
// of them. Joined after every production file, so the helpers of
// `matvec.metal`, `matvec_split.metal`, and `matmul.metal` are in scope.
//
// Split-K decode matvec: `splits` threadgroups per 16-row block, each summing
// only its share of the K blocks into a row-major partial buffer
// `[split][rows]`; `nu_reduce_splits` then sums the splits into the output.
// The K-range bounds passed to the shared decode body keep its per-lane loops
// identical to the non-split form, so the partials differ from a single-pass
// sum only by F32 rounding order. Measured behind the single-pass kernel on
// every row bucket (2026-09-21), so nothing routes here; the sweep and the
// fixture force it through the host's `matvecSplits` / `matvecSegmentsSplits`.
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

// Plain-mode segment split: each group sums its K range into the row-partial
// buffer `[split][global row]` (the same row-to-segment walk as
// `nu_matvec_segments`); `nu_segment_reduce_splits` sums the splits and
// scatters every row back into its segment's output. Only specialized
// segments are eligible: the generic branch reads the whole K regardless of
// the range.
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

// Writes element (r, c) of an 8×8 `simdgroup_matrix` filled through
// `thread_elements()` as 2·lane + i into `out[8r + c]`: `metal-check` pins
// the lane layout (`nu_fragment_row`/`nu_fragment_col` in matmul.metal) that
// the production register-fragment tiles assume.
kernel void nu_fragment_layout(device float * out [[buffer(0)]], uint lane [[thread_index_in_simdgroup]]) {
    simdgroup_float8x8 m;
    m.thread_elements()[0] = float(2 * lane);
    m.thread_elements()[1] = float(2 * lane + 1);
    simdgroup_store(m, out, 8);
}
