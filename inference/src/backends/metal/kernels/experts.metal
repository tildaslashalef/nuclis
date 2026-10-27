// ---------------------------------------------------------------------------
// Mixture of experts. A 3-D tensor holds `experts` contiguous row-major
// matrices; a token reads only the `slots` experts its router selected.
//
// Gathered matvec: threadgroup `group` serves slot `group / row_groups` and
// the 16-row block `group % row_groups` of that slot's expert, whose bytes
// start at expert · rows · stride. The lane arithmetic is the segment
// kernel's (`nu_segment_sums`: the specialized body of the encoding, or the
// generic decoder when bit 31 asks for it), so a selected expert costs the
// bytes of a dense matrix of its size. `in_stride` 0 shares one input
// vector across the slots (the gate-up projection); the down projection
// gives each slot its own hidden row. Expert indices are clamped so a
// corrupt routing buffer can never read past the tensor.
struct ExpertMatvecParams { uint columns, stride, rows, blocks, encoding, experts, slots, in_stride, out_stride, row_groups; };
kernel void nu_matvec_experts(device const uchar * weights [[buffer(0)]],
                              device const float * input [[buffer(1)]],
                              device float * output [[buffer(2)]],
                              device const uint * indices [[buffer(3)]],
                              constant ExpertMatvecParams & p [[buffer(7)]],
                              uint group [[threadgroup_position_in_grid]],
                              uint sg [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    uint slot = group / p.row_groups, local = group % p.row_groups;
    uint expert = min(indices[slot], p.experts - 1);
    device const uchar * w = weights + ulong(expert) * ulong(p.rows) * ulong(p.stride);
    device const float * x = input + ulong(slot) * p.in_stride;
    device float * out = output + ulong(slot) * p.out_stride;
    MatvecBlockParams shape = { p.columns, p.stride, p.rows, p.blocks };
    threadgroup float iq4[16];
    nu_iq4_table(iq4, sg, lane);
    float a[4];
    nu_segment_sums(w, x, shape, p.encoding, local, sg, lane, a, 0u, p.blocks, iq4);
    uint row0 = (local * NU_MATVEC_SIMDGROUPS + sg) * 4;
    if (row0 < p.rows) nu_store_rows<4>(a, out, row0, p.rows, lane);
}

// Router: one 256-thread group per row of logits selects `k` experts by
// (logit desc, index asc) — the comparison is on the logits, not the
// probabilities, so rounding in `exp` cannot reorder near-ties — and writes
// their softmax probabilities renormalized to sum one, the sum clamped
// below at the smallest F16 normal as the CPU reference does. `experts`
// is at most 256 (one logit per thread) and `k` at most NU_ROUTE_MAX_K.
#define NU_ROUTE_MAX_K 64
struct RouteParams { uint experts, k, rows, in_stride; };
kernel void nu_route(device const float * logits [[buffer(0)]],
                     device uint * indices [[buffer(1)]],
                     device float * weights [[buffer(2)]],
                     constant RouteParams & p [[buffer(7)]],
                     uint row [[threadgroup_position_in_grid]],
                     uint tid [[thread_position_in_threadgroup]],
                     uint lane [[thread_index_in_simdgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[8];
    threadgroup uint partial_index[8];
    threadgroup float selected[NU_ROUTE_MAX_K];
    const bool live = tid < p.experts;
    device const float * x = logits + ulong(row) * p.in_stride;
    float logit = live ? x[tid] : -INFINITY;
    float m = simd_max(logit);
    if (lane == 0) partial[sg] = m;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = 0; i < 8; ++i) m = max(m, partial[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float e = live ? exp(logit - m) : 0.0f;
    float total = simd_sum(e);
    if (lane == 0) partial[sg] = total;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    total = 0.0f;
    for (uint i = 0; i < 8; ++i) total += partial[i];
    const float probability = e / total;
    bool taken = !live;
    float sum = 0.0f;
    for (uint r = 0; r < p.k; ++r) {
        threadgroup_barrier(mem_flags::mem_threadgroup); // partials free for reuse
        float v = taken ? -INFINITY : logit; uint idx = taken ? 0xffffffffu : tid;
        for (uint offset = 16; offset > 0; offset >>= 1) {
            float ov = simd_shuffle_down(v, offset); uint oi = simd_shuffle_down(idx, offset);
            if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
        }
        if (lane == 0) { partial[sg] = v; partial_index[sg] = idx; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        v = partial[0]; idx = partial_index[0];
        for (uint i = 1; i < 8; ++i) if (partial[i] > v || (partial[i] == v && partial_index[i] < idx)) { v = partial[i]; idx = partial_index[i]; }
        if (tid == idx) { taken = true; selected[r] = probability; indices[ulong(row) * p.k + r] = idx; }
        sum += (tid == idx) ? probability : 0.0f;
    }
    // Every thread adds the same winners in the same order once summed
    // across the group: reduce the per-thread contributions (at most one
    // nonzero per selected expert) so the total matches a serial sum.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float group_sum = simd_sum(sum);
    if (lane == 0) partial[sg] = group_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    group_sum = 0.0f;
    for (uint i = 0; i < 8; ++i) group_sum += partial[i];
    if (tid < p.k) weights[ulong(row) * p.k + tid] = selected[tid] / max(group_sum, 6.103515625e-5f);
}

// Weighted sum of the slots' projections: out[row][c] = Σ_s w[row][s] ·
// scale[e_s] · y[row·slots + s][c], the per-expert scale only with flag 1.
struct CombineParams { uint columns, slots, rows, in_stride, out_stride, experts, flags; };
kernel void nu_combine_experts(device const float * values [[buffer(0)]],
                               device const float * weights [[buffer(1)]],
                               device const uint * indices [[buffer(2)]],
                               device const float * scales [[buffer(3)]],
                               device float * output [[buffer(4)]],
                               constant CombineParams & p [[buffer(7)]],
                               uint id [[thread_position_in_grid]]) {
    if (id >= p.columns * p.rows) return;
    uint row = id / p.columns, c = id % p.columns;
    float acc = 0.0f;
    for (uint s = 0; s < p.slots; ++s) {
        uint slot = row * p.slots + s;
        float w = weights[slot];
        if (p.flags & 1) w *= scales[min(indices[slot], p.experts - 1)];
        acc = fma(w, values[ulong(slot) * p.in_stride + c], acc);
    }
    output[ulong(row) * p.out_stride + c] = acc;
}

// Gated GELU over `rows` strided rows: out[r][i] = gelu(gate[r][i]) · up[r][i]
// (the up half of a fused gate-up row is `up` bound at its offset).
struct GeluRowsParams { uint width, rows, gate_stride, up_stride, out_stride; };
kernel void nu_gelu_mul_rows(device const float * gate [[buffer(0)]],
                             device const float * up [[buffer(1)]],
                             device float * output [[buffer(2)]],
                             constant GeluRowsParams & p [[buffer(7)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= p.width * p.rows) return;
    uint row = id / p.width, i = id % p.width;
    output[ulong(row) * p.out_stride + i] = nu_gelu(gate[ulong(row) * p.gate_stride + i]) * up[ulong(row) * p.up_stride + i];
}

// Row lists for the prefill path: one 256-thread group turns the routing of
// a chunk (`n` = tokens · k slot rows, `indices[i]` the expert of slot row
// i) into the `lists` buffer of u32 words the gathered matmul reads:
//   [0] tile count, [1] n,
//   [2 .. 2 + experts] exclusive prefix sum of the per-expert counts,
//   [experts + 3 ..) `max_tiles` tiles of (expert, first, count): the 32-row
//       tiles of each expert's rows, `first` an index into the row list,
//   [experts + 3 + 3 · max_tiles ..) the row list: the slot rows grouped by
//       expert (count[e] of them from offset[e]).
// Sum of ceil(count/32) is at most n/32 + experts, which is how the host
// bounds the matmul grid before the counts exist; tiles past the count exit.
// The order of rows within an expert comes from atomics and is not
// deterministic, but every row's result is computed independently, so the
// output is. Indices are clamped so a corrupt routing buffer stays in bounds.
#define NU_LISTS_MAX_EXPERTS 256
struct ExpertListsParams { uint experts, n, max_tiles; };
kernel void nu_expert_lists(device const uint * indices [[buffer(0)]],
                            device uint * lists [[buffer(1)]],
                            constant ExpertListsParams & p [[buffer(7)]],
                            uint tid [[thread_position_in_threadgroup]]) {
    threadgroup atomic_uint cursor[NU_LISTS_MAX_EXPERTS];
    threadgroup uint offset[NU_LISTS_MAX_EXPERTS];
    for (uint e = tid; e < p.experts; e += 256) atomic_store_explicit(&cursor[e], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < p.n; i += 256) atomic_fetch_add_explicit(&cursor[min(indices[i], p.experts - 1)], 1u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        device uint * tile = lists + p.experts + 3;
        uint start = 0, tiles = 0;
        for (uint e = 0; e < p.experts; ++e) {
            uint count = atomic_load_explicit(&cursor[e], memory_order_relaxed);
            offset[e] = start;
            lists[2 + e] = start;
            for (uint t = 0; t < count && tiles < p.max_tiles; t += 32, ++tiles) {
                tile[3 * tiles] = e; tile[3 * tiles + 1] = start + t; tile[3 * tiles + 2] = min(32u, count - t);
            }
            start += count;
        }
        lists[2 + p.experts] = start;
        lists[0] = tiles;
        lists[1] = start;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < p.experts; e += 256) atomic_store_explicit(&cursor[e], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device uint * rows = lists + p.experts + 3 + 3 * p.max_tiles;
    for (uint i = tid; i < p.n; i += 256) {
        uint e = min(indices[i], p.experts - 1);
        rows[offset[e] + atomic_fetch_add_explicit(&cursor[e], 1u, memory_order_relaxed)] = i;
    }
}

