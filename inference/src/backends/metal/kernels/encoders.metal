// ---- The vision projectors' kernels (docs/engine/vision.md) ----------
// LayerNorm over `rows` rows of `width`: y = (x - mean) / sqrt(var + eps) · w
// + b, the population variance in F32 over a two-pass reduction. One
// 256-thread group per row; `output` may alias `input` exactly.
struct LayerNormParams { uint width; uint in_stride; uint out_stride; float eps; };
kernel void nu_layernorm(device const float * input [[buffer(0)]],
                         device const float * weight [[buffer(1)]],
                         device const float * bias [[buffer(2)]],
                         device float * output [[buffer(3)]],
                         constant LayerNormParams & p [[buffer(7)]],
                         uint row [[threadgroup_position_in_grid]],
                         uint tid [[thread_position_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[8];
    device const float * x = input + ulong(row) * p.in_stride;
    float sum = 0;
    for (uint i = tid; i < p.width; i += 256) sum += x[i];
    sum = simd_sum(sum);
    if (lane == 0) partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0;
    for (uint i = 0; i < 8; ++i) total += partial[i];
    const float mean = total / float(p.width);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sq = 0;
    for (uint i = tid; i < p.width; i += 256) { float d = x[i] - mean; sq += d * d; }
    sq = simd_sum(sq);
    if (lane == 0) partial[sg] = sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float variance = 0;
    for (uint i = 0; i < 8; ++i) variance += partial[i];
    const float scale = rsqrt(variance / float(p.width) + p.eps);
    device float * y = output + ulong(row) * p.out_stride;
    // Every lane finished reading x before any write: rows may alias in place.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < p.width; i += 256) y[i] = (x[i] - mean) * scale * weight[i] + bias[i];
}

// x[row · stride + i] += bias[i] over `rows` rows of `width`.
struct BiasRowsParams { uint width, rows, stride; };
kernel void nu_add_bias_rows(device float * x [[buffer(0)]], device const float * bias [[buffer(1)]],
                             constant BiasRowsParams & p [[buffer(7)]], uint id [[thread_position_in_grid]]) {
    if (id >= p.width * p.rows) return;
    uint row = id / p.width, i = id % p.width;
    x[ulong(row) * p.stride + i] += bias[i];
}
kernel void nu_gelu_inplace(device float * x [[buffer(0)]],
                            constant CountParams & p [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    if (i < p.count) x[i] = nu_gelu(x[i]);
}

// Bidirectional attention of one (row, head) over the keys [lo, hi), at most
// NU_FULL_MAX_ROWS of them, by one 256-thread group: the scores land in
// threadgroup memory, the softmax reduces there, and the value pass splits
// the keys across the threads that share a dimension.
#define NU_FULL_MAX_ROWS 4096
inline void nu_attention_range(device const float * q, device const float * k, device const float * v,
                               device float * out, uint width, uint kv_stride, float scale, uint lo, uint hi,
                               threadgroup float * scores, threadgroup float * partial, threadgroup float * acc,
                               uint tid, uint lane, uint sg) {
    float local_max = -INFINITY;
    for (uint j = lo + tid; j < hi; j += 256) {
        device const float * kj = k + ulong(j) * kv_stride;
        float dot = 0;
        for (uint d = 0; d < width; ++d) dot += q[d] * kj[d];
        dot *= scale;
        scores[j - lo] = dot;
        local_max = max(local_max, dot);
    }
    local_max = simd_max(local_max);
    if (lane == 0) partial[sg] = local_max;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = partial[0];
    for (uint i = 1; i < 8; ++i) m = max(m, partial[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float local_sum = 0;
    for (uint j = lo + tid; j < hi; j += 256) { float e = exp(scores[j - lo] - m); scores[j - lo] = e; local_sum += e; }
    local_sum = simd_sum(local_sum);
    if (lane == 0) partial[sg] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float denominator = 0;
    for (uint i = 0; i < 8; ++i) denominator += partial[i];
    const float inv = 1.0f / denominator;
    // Value pass: thread t owns dimension t % width for the keys j ≡ t / width
    // modulo the number of key groups that fit in 256 threads.
    const uint key_groups = 256 / width;
    const uint d = tid % width, kg = tid / width;
    float sum = 0;
    if (kg < key_groups) {
        for (uint j = lo + kg; j < hi; j += key_groups) sum += scores[j - lo] * v[ulong(j) * kv_stride + d];
    }
    acc[tid] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < width) {
        float total = 0;
        for (uint g = 0; g < key_groups; ++g) total += acc[g * width + d];
        out[d] = total * inv;
    }
}

// Bidirectional attention over `rows` rows (an image's patches): every row
// attends to every row. Queries, keys, and values are `heads` heads of
// `width` at `q_stride`/`kv_stride` per row; the output is `heads · width`
// wide at `out_stride`. One 256-thread group per (row, head).
struct AttentionFullParams { uint heads; uint width; uint rows; uint q_stride; uint kv_stride; uint out_stride; float scale; };
kernel void nu_attention_full(device const float * queries [[buffer(0)]],
                              device const float * keys [[buffer(1)]],
                              device const float * values [[buffer(2)]],
                              device float * output [[buffer(3)]],
                              constant AttentionFullParams & p [[buffer(7)]],
                              uint group [[threadgroup_position_in_grid]],
                              uint tid [[thread_position_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scores[NU_FULL_MAX_ROWS];
    threadgroup float partial[8];
    threadgroup float acc[256];
    const uint row = group / p.heads, head = group % p.heads;
    nu_attention_range(queries + ulong(row) * p.q_stride + ulong(head) * p.width, keys + ulong(head) * p.width,
                       values + ulong(head) * p.width, output + ulong(row) * p.out_stride + ulong(head) * p.width,
                       p.width, p.kv_stride, p.scale, 0, p.rows, scores, partial, acc, tid, lane, sg);
}

// Bidirectional attention over packed sequences: row i attends to the keys
// of its own sequence, `bounds[i]` = [begin, end), within `window` of it
// (|i − j| ≤ window; ~0u for none). The same layout and grid as
// nu_attention_full; the bounds are clamped so a bad entry stays in range.
struct AttentionSegmentsParams { uint heads; uint width; uint rows; uint q_stride; uint kv_stride; uint out_stride; float scale; uint window; };
kernel void nu_attention_segments(device const float * queries [[buffer(0)]],
                                  device const float * keys [[buffer(1)]],
                                  device const float * values [[buffer(2)]],
                                  device float * output [[buffer(3)]],
                                  device const uint2 * bounds [[buffer(4)]],
                                  constant AttentionSegmentsParams & p [[buffer(7)]],
                                  uint group [[threadgroup_position_in_grid]],
                                  uint tid [[thread_position_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]],
                                  uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float scores[NU_FULL_MAX_ROWS];
    threadgroup float partial[8];
    threadgroup float acc[256];
    const uint row = group / p.heads, head = group % p.heads;
    const uint2 b = bounds[row];
    uint lo = min(b.x, row);
    uint hi = clamp(b.y, row + 1, p.rows);
    if (p.window != ~0u) {
        lo = max(lo, row - min(p.window, row));
        hi = min(hi, row + min(p.window, p.rows) + 1);
    }
    hi = min(hi, lo + NU_FULL_MAX_ROWS);
    nu_attention_range(queries + ulong(row) * p.q_stride + ulong(head) * p.width, keys + ulong(head) * p.width,
                       values + ulong(head) * p.width, output + ulong(row) * p.out_stride + ulong(head) * p.width,
                       p.width, p.kv_stride, p.scale, lo, hi, scores, partial, acc, tid, lane, sg);
}

// Grouped-query attention over packed sequences with materialized scores:
// the masking contract of nu_attention_segments (row i sees the keys of
// `bounds[i]` within `window`), F32 throughout, in three dispatches over a
// list of query tiles. A tile is at most 32 rows (a multiple of 8) with the
// key range [key_lo, key_lo + span) its rows can see (span a multiple of
// 32) and its score strip at `offset` (`rows × span` floats; head h's strips
// start at h · head_stride). The two products are F32 matmul tiles with
// both operands staged in threadgroup memory, so the matrix unit is fed the
// way the projections' tiles feed it; the softmax between them is the exact
// two-pass one. A tile's arithmetic depends on its rows and keys only.
struct AttentionTile { uint first; uint rows; uint key_lo; uint span; uint offset; };
struct AttentionGemmParams { uint query_heads; uint kv_heads; uint width; uint rows; uint q_stride; uint kv_stride; uint out_stride; uint head_stride; float scale; uint window; };

// Scores: one threadgroup per (tile, head × 32-key block), S[32][32] =
// scale · Q · Kᵀ over the head width (a multiple of 8) in 64-wide steps, the
// last step's missing columns staged as zeros; four SIMD groups own a
// 16 × 16 quarter each. Invisible keys are written −∞, key rows past `rows`
// stage as zeros.
kernel void nu_segments_scores(device const float * queries [[buffer(0)]],
                                device const float * keys [[buffer(1)]],
                                device const uint2 * bounds [[buffer(2)]],
                                device const AttentionTile * tiles [[buffer(3)]],
                                device float * scores [[buffer(4)]],
                                constant AttentionGemmParams & p [[buffer(7)]],
                                uint2 group [[threadgroup_position_in_grid]],
                                uint2 local [[thread_position_in_threadgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint tid = local.x;
    threadgroup float qs[32 * 64];   // [row][d]; reused for the 32 × 32 result
    threadgroup float kt[64 * 32];   // [d][key]
    threadgroup uint lo[32], hi[32];
    const AttentionTile t = tiles[group.x];
    const uint head = group.y % p.query_heads, block = group.y / p.query_heads;
    if (block * 32 >= t.span) return; // uniform over the threadgroup
    const uint kv = head / (p.query_heads / p.kv_heads);
    const uint key0 = t.key_lo + block * 32;
    if (tid < 32) {
        const uint r = t.first + tid;
        uint a = 0, b = 0;
        if (tid < t.rows && r < p.rows) {
            const uint2 bb = bounds[r];
            a = min(bb.x, r);
            b = clamp(bb.y, r + 1, p.rows);
            if (p.window != ~0u) {
                a = max(a, r - min(p.window, r));
                b = min(b, r + min(p.window, p.rows) + 1);
            }
        }
        lo[tid] = a;
        hi[tid] = b;
    }
    const uint r = tid >> 2, seg = tid & 3;
    const bool q_ok = r < t.rows, k_ok = key0 + r < p.rows;
    device const float * qrow = queries + ulong(t.first + r) * p.q_stride + ulong(head) * p.width;
    device const float * krow = keys + ulong(key0 + r) * p.kv_stride + ulong(kv) * p.width;
    const uint sub_row = (sg & 1) * 16, sub_key = (sg >> 1) * 16;
    simdgroup_float8x8 acc[2][2];
    for (uint i = 0; i < 2; ++i) for (uint j = 0; j < 2; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
    for (uint d0 = 0; d0 < p.width; d0 += 64) {
        const uint dn = min(64u, p.width - d0); // a multiple of 8, uniform
        for (uint c = 0; c < 4; ++c) {
            const bool d_ok = seg * 16 + 4 * c < dn;
            const float4 x = q_ok && d_ok ? ((device const float4 *)(qrow + d0 + seg * 16))[c] : float4(0.0f);
            ((threadgroup float4 *)(qs + r * 64 + seg * 16))[c] = x;
            const float4 y = k_ok && d_ok ? ((device const float4 *)(krow + d0 + seg * 16))[c] : float4(0.0f);
            threadgroup float * column = kt + (seg * 16 + 4 * c) * 32 + r;
            column[0] = y.x; column[32] = y.y; column[64] = y.z; column[96] = y.w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k8 = 0; k8 < dn; k8 += 8) {
            simdgroup_float8x8 a[2], b[2];
            for (uint i = 0; i < 2; ++i) simdgroup_load(a[i], qs + (sub_row + 8 * i) * 64 + k8, 64);
            for (uint j = 0; j < 2; ++j) simdgroup_load(b[j], kt + k8 * 32 + sub_key + 8 * j, 32);
            for (uint i = 0; i < 2; ++i) for (uint j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    threadgroup float * result = qs; // [row][key], free after the last barrier
    for (uint i = 0; i < 2; ++i) for (uint j = 0; j < 2; ++j)
        simdgroup_store(acc[i][j], result + (sub_row + 8 * i) * 32 + sub_key + 8 * j, 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float * strip = scores + ulong(head) * p.head_stride + t.offset + block * 32;
    for (uint e = tid; e < t.rows * 32; e += 128) {
        const uint row = e >> 5, col = e & 31, key = key0 + col;
        strip[ulong(row) * t.span + col] = (key >= lo[row] && key < hi[row]) ? result[e] * p.scale : -INFINITY;
    }
}

// The softmax statistics of each strip row, read once: one SIMD group per
// row, eight rows at a time, one threadgroup per (tile, head); each lane
// keeps an online (max, sum), merged across the group. Row r of a tile in
// head h stores (max, 1 / sum) at float2 index h · head_stride / 32 +
// (offset + r · span) / 32, past the strips (`nu_segments_values` applies
// them while staging P). A row that sees no key (a padding row past `rows`)
// stores (0, 0), so its weights and output are zeros rather than NaN.
kernel void nu_segments_stats(device const AttentionTile * tiles [[buffer(0)]],
                              device float * scores [[buffer(1)]],
                              constant AttentionGemmParams & p [[buffer(7)]],
                              uint2 group [[threadgroup_position_in_grid]],
                              uint sg [[simdgroup_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]]) {
    const AttentionTile t = tiles[group.x];
    device const float * strip = scores + ulong(group.y) * p.head_stride + t.offset;
    device float2 * stats = (device float2 *)(scores + ulong(p.query_heads) * p.head_stride) + ulong(group.y) * (p.head_stride / 32);
    for (uint row = sg; row < t.rows; row += 8) {
        device const float * x = strip + ulong(row) * t.span;
        float m = -INFINITY, sum = 0.0f;
        for (uint i = lane; i < t.span; i += 32) {
            const float v = x[i];
            if (v == -INFINITY) continue;
            if (v > m) { sum = sum * exp(m - v) + 1.0f; m = v; }
            else sum += exp(v - m);
        }
        const float mx = simd_max(m);
        const float total = simd_sum(m == -INFINITY ? 0.0f : sum * exp(m - mx));
        if (lane == 0) stats[(t.offset + row * t.span) / 32] = mx == -INFINITY ? float2(0.0f) : float2(mx, 1.0f / total);
    }
}

// Values: one threadgroup per (tile, head × 32 value columns), O[32][32] =
// P · V over the span in 32-key steps, P = exp(S − max) / sum staged from
// the scores and `nu_segments_stats`' row statistics; key rows past `rows` stage as zeros
// (their weights are 0, and padding rows may hold anything). The last block
// of a width that is not a multiple of 32 stages and stores only its own
// 8-column blocks.
kernel void nu_segments_values(device const float * scores [[buffer(0)]],
                                device const float * values [[buffer(1)]],
                                device const AttentionTile * tiles [[buffer(2)]],
                                device float * output [[buffer(3)]],
                                constant AttentionGemmParams & p [[buffer(7)]],
                                uint2 group [[threadgroup_position_in_grid]],
                                uint2 local [[thread_position_in_threadgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]]) {
    const uint tid = local.x;
    threadgroup float ps[32 * 32]; // [row][key]
    threadgroup float vs[32 * 32]; // [key][column]
    const AttentionTile t = tiles[group.x];
    const uint head = group.y % p.query_heads, cb = group.y / p.query_heads;
    const uint kv = head / (p.query_heads / p.kv_heads);
    const uint c0 = cb * 32;
    const uint cn = min(32u, p.width - c0); // a multiple of 8, uniform
    const uint r = tid >> 2, part = (tid & 3) * 8;
    const bool c_ok = part < cn;
    const bool p_ok = r < t.rows;
    device const float * prow = scores + ulong(head) * p.head_stride + t.offset + ulong(r) * t.span + part;
    device const float2 * stats = (device const float2 *)(scores + ulong(p.query_heads) * p.head_stride) + ulong(head) * (p.head_stride / 32);
    const float2 st = p_ok ? stats[(t.offset + r * t.span) / 32] : float2(0.0f);
    device const float * vbase = values + ulong(kv) * p.width + c0 + part;
    const uint sub_row = (sg & 1) * 16, sub_col = (sg >> 1) * 16;
    simdgroup_float8x8 acc[2][2];
    for (uint i = 0; i < 2; ++i) for (uint j = 0; j < 2; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
    for (uint k0 = 0; k0 < t.span; k0 += 32) {
        const uint key = t.key_lo + k0 + r;
        for (uint c = 0; c < 2; ++c) {
            ((threadgroup float4 *)(ps + r * 32 + part))[c] = p_ok ? exp(((device const float4 *)(prow + k0))[c] - st.x) * st.y : float4(0.0f);
            ((threadgroup float4 *)(vs + r * 32 + part))[c] = key < p.rows && c_ok ? ((device const float4 *)(vbase + ulong(key) * p.kv_stride))[c] : float4(0.0f);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k8 = 0; k8 < 32; k8 += 8) {
            simdgroup_float8x8 a[2], b[2];
            for (uint i = 0; i < 2; ++i) simdgroup_load(a[i], ps + (sub_row + 8 * i) * 32 + k8, 32);
            for (uint j = 0; j < 2; ++j) simdgroup_load(b[j], vs + k8 * 32 + sub_col + 8 * j, 32);
            for (uint i = 0; i < 2; ++i) for (uint j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    device float * out = output + ulong(t.first) * p.out_stride + ulong(head) * p.width + c0;
    for (uint i = 0; i < 2; ++i) {
        if (sub_row + 8 * i >= t.rows) continue;
        for (uint j = 0; j < 2; ++j)
            if (sub_col + 8 * j < cn) simdgroup_store(acc[i][j], out + ulong(sub_row + 8 * i) * p.out_stride + sub_col + 8 * j, p.out_stride);
    }
}

