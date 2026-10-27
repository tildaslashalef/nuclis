// History penalties in place, one thread per logit: repetition first (divide
// positive logits, multiply negative ones), then presence, for every token in
// the history bit set. Bit `id` lives in word `id / 32`, bit `id % 32`; the
// host uploads the words when the history changed. Matches the CPU sampler's
// `penalize` operation for operation.
struct PenalizeParams { uint count; uint history_words; float repetition; float presence; };
kernel void nu_penalize(device float * logits [[buffer(0)]],
                        device const uint * history [[buffer(1)]],
                        constant PenalizeParams & p [[buffer(7)]],
                        uint tid [[thread_position_in_grid]]) {
    if (tid >= p.count) return;
    if (!(history[tid >> 5] & (1u << (tid & 31u)))) return;
    float x = logits[tid];
    float scaled = x > 0.0f ? x / p.repetition : x * p.repetition;
    logits[tid] = scaled - p.presence;
}

// Greedy selection: one threadgroup reduces a strided slice to (value, index);
// a second pass with one group merges partials. Ties select the lowest index.
struct ArgmaxParams { uint count; uint partials; };
kernel void nu_argmax_partial(device const float * logits [[buffer(0)]],
                              device float * best_value [[buffer(1)]],
                              device uint * best_index [[buffer(2)]],
                              constant ArgmaxParams & p [[buffer(7)]],
                              uint group [[threadgroup_position_in_grid]],
                              uint tid [[thread_position_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float values[8];
    threadgroup uint indices[8];
    float v = -INFINITY; uint idx = 0xffffffffu;
    for (uint i = group * 256 + tid; i < p.count; i += p.partials * 256) {
        float x = logits[i];
        if (x > v || (x == v && i < idx)) { v = x; idx = i; }
    }
    for (uint offset = 16; offset > 0; offset >>= 1) {
        float ov = simd_shuffle_down(v, offset); uint oi = simd_shuffle_down(idx, offset);
        if (ov > v || (ov == v && oi < idx)) { v = ov; idx = oi; }
    }
    if (lane == 0) { values[sg] = v; indices[sg] = idx; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        for (uint i = 1; i < 8; ++i) if (values[i] > v || (values[i] == v && indices[i] < idx)) { v = values[i]; idx = indices[i]; }
        best_value[group] = v; best_index[group] = idx;
    }
}
// Partial top-k for sampled decoding: the k best logits by (value desc,
// index asc) — the CPU sampler's sort order — without reading the vocabulary
// back. Pass 1: each of `partials` threadgroups owns a strided slice; every
// thread keeps its ≤ NU_TOPK_LOCAL values in registers and a taken mask, and
// the group runs k rounds of "best untaken" (SIMD shuffle reduction, then an
// 8-way threadgroup pick), emitting its slice's sorted top k. Pass 2: one
// threadgroup merges the `partials` sorted lists k-way, one cursor per list.
// Pass 3: Σ exp((l − max) / T) over every logit as F32 partial sums (the CPU
// adds them in F64) plus a non-finite flag per partition. NaN never wins a
// comparison, so it cannot enter the list; the flag makes the CPU fall back.
struct TopKParams { uint count; uint partials; uint k; float temperature; };
#define NU_TOPK_LOCAL 16 // values per thread: count <= partials * 256 * 16
inline bool nu_topk_better(float v, uint i, float best_v, uint best_i) { return v > best_v || (v == best_v && i < best_i); }
kernel void nu_topk_partial(device const float * logits [[buffer(0)]],
                            device float * values [[buffer(1)]],
                            device uint * indices [[buffer(2)]],
                            constant TopKParams & p [[buffer(7)]],
                            uint group [[threadgroup_position_in_grid]],
                            uint tid [[thread_position_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float best_v[8];
    threadgroup uint best_i[8];
    threadgroup uint winner;
    const uint stride = p.partials * 256, first = group * 256 + tid;
    float local[NU_TOPK_LOCAL];
    uint taken = 0;
    for (uint j = 0; j < NU_TOPK_LOCAL; ++j) {
        uint i = first + j * stride;
        local[j] = i < p.count ? logits[i] : -INFINITY;
        if (i >= p.count) taken |= 1u << j;
    }
    for (uint round = 0; round < p.k; ++round) {
        float v = -INFINITY; uint idx = 0xffffffffu;
        for (uint j = 0; j < NU_TOPK_LOCAL; ++j) {
            uint i = first + j * stride;
            if (!(taken & (1u << j)) && nu_topk_better(local[j], i, v, idx)) { v = local[j]; idx = i; }
        }
        for (uint offset = 16; offset > 0; offset >>= 1) {
            float ov = simd_shuffle_down(v, offset); uint oi = simd_shuffle_down(idx, offset);
            if (nu_topk_better(ov, oi, v, idx)) { v = ov; idx = oi; }
        }
        if (lane == 0) { best_v[sg] = v; best_i[sg] = idx; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            for (uint i = 1; i < 8; ++i) if (nu_topk_better(best_v[i], best_i[i], v, idx)) { v = best_v[i]; idx = best_i[i]; }
            values[group * p.k + round] = v; indices[group * p.k + round] = idx;
            winner = idx;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint w = winner;
        // The owning thread retires the winner; reads of `winner` complete
        // before the next round's first barrier, after which tid 0 rewrites it.
        if (w != 0xffffffffu && w >= first && (w - first) % stride == 0) taken |= 1u << ((w - first) / stride);
    }
}
kernel void nu_topk_final(device const float * values [[buffer(0)]],
                          device const uint * indices [[buffer(1)]],
                          device float * out_values [[buffer(2)]],
                          device uint * out_indices [[buffer(3)]],
                          constant TopKParams & p [[buffer(7)]],
                          uint tid [[thread_position_in_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float best_v[8];
    threadgroup uint best_i[8];
    threadgroup uint best_o[8];
    threadgroup uint winner;
    uint cursor = 0;
    const bool active = tid < p.partials;
    for (uint round = 0; round < p.k; ++round) {
        float v = -INFINITY; uint idx = 0xffffffffu, owner = 0xffffffffu;
        if (active && cursor < p.k) { v = values[tid * p.k + cursor]; idx = indices[tid * p.k + cursor]; owner = tid; }
        for (uint offset = 16; offset > 0; offset >>= 1) {
            float ov = simd_shuffle_down(v, offset); uint oi = simd_shuffle_down(idx, offset); uint oo = simd_shuffle_down(owner, offset);
            if (nu_topk_better(ov, oi, v, idx)) { v = ov; idx = oi; owner = oo; }
        }
        if (lane == 0) { best_v[sg] = v; best_i[sg] = idx; best_o[sg] = owner; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            for (uint i = 1; i < 8; ++i) if (nu_topk_better(best_v[i], best_i[i], v, idx)) { v = best_v[i]; idx = best_i[i]; owner = best_o[i]; }
            out_values[round] = v; out_indices[round] = idx;
            winner = owner;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == winner) ++cursor;
    }
}
kernel void nu_expsum_partial(device const float * logits [[buffer(0)]],
                              device const float * top_values [[buffer(1)]],
                              device float * sums [[buffer(2)]],
                              device uint * flags [[buffer(3)]],
                              constant TopKParams & p [[buffer(7)]],
                              uint group [[threadgroup_position_in_grid]],
                              uint tid [[thread_position_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float part[8];
    threadgroup uint bad[8];
    const float maximum = top_values[0];
    float sum = 0; uint nonfinite = 0;
    for (uint i = group * 256 + tid; i < p.count; i += p.partials * 256) {
        float x = logits[i];
        if (!isfinite(x)) nonfinite = 1;
        sum += exp((x - maximum) / p.temperature);
    }
    sum = simd_sum(sum); nonfinite = simd_max(nonfinite);
    if (lane == 0) { part[sg] = sum; bad[sg] = nonfinite; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        float total = 0; uint flag = 0;
        for (uint i = 0; i < 8; ++i) { total += part[i]; flag |= bad[i]; }
        sums[group] = total; flags[group] = flag;
    }
}

kernel void nu_argmax_final(device const float * values [[buffer(0)]],
                            device const uint * indices [[buffer(1)]],
                            device uint * result [[buffer(2)]],
                            constant ArgmaxParams & p [[buffer(7)]],
                            uint tid [[thread_position_in_threadgroup]]) {
    if (tid != 0) return;
    float v = -INFINITY; uint idx = 0xffffffffu;
    for (uint i = 0; i < p.partials; ++i) if (values[i] > v || (values[i] == v && indices[i] < idx)) { v = values[i]; idx = indices[i]; }
    result[0] = idx;
}

