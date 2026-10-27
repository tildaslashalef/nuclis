// ---------------------------------------------------------------------------
// Single-position grouped-query attention in three passes over a scores buffer
// of [query_head][visible] floats. Keys/values are [token][kv_head][channel],
// stored as F32 or F16 (`KV` is the cache element type; queries, scores,
// and the output stay F32, so an F16 cache changes only what is read).
struct AttentionParams { uint query_heads; uint kv_heads; uint key_width; uint value_width; uint visible; float scale; };
template <typename KV>
kernel void nu_attention_scores_t(device const KV * keys [[buffer(0)]],
                                  device const float * query [[buffer(1)]],
                                  device float * scores [[buffer(2)]],
                                  constant AttentionParams & p [[buffer(7)]],
                                  uint row [[threadgroup_position_in_grid]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    uint head = row / p.visible, token = row % p.visible, kv = head / (p.query_heads / p.kv_heads);
    float dot = 0;
    for (uint k = lane; k < p.key_width; k += 32) dot += query[head*p.key_width + k] * float(keys[(ulong(token)*p.kv_heads + kv)*p.key_width + k]);
    dot = simd_sum(dot) * p.scale;
    if (lane == 0) scores[row] = dot;
}
kernel void nu_attention_softmax(device float * scores [[buffer(0)]],
                                 constant AttentionParams & p [[buffer(7)]],
                                 uint head [[threadgroup_position_in_grid]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    device float * s = scores + ulong(head) * p.visible;
    float maximum = -INFINITY;
    for (uint t = lane; t < p.visible; t += 32) maximum = max(maximum, s[t]);
    maximum = simd_max(maximum);
    float total = 0;
    for (uint t = lane; t < p.visible; t += 32) total += exp(s[t] - maximum);
    total = simd_sum(total);
    for (uint t = lane; t < p.visible; t += 32) s[t] = exp(s[t] - maximum) / total;
}
template <typename KV>
kernel void nu_attention_values_t(device const KV * values [[buffer(0)]],
                                  device const float * scores [[buffer(1)]],
                                  device float * output [[buffer(2)]],
                                  constant AttentionParams & p [[buffer(7)]],
                                  uint row [[threadgroup_position_in_grid]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    uint head = row / p.value_width, channel = row % p.value_width, kv = head / (p.query_heads / p.kv_heads);
    float sum = 0;
    for (uint t = lane; t < p.visible; t += 32) sum += scores[ulong(head)*p.visible + t] * float(values[(ulong(t)*p.kv_heads + kv)*p.value_width + channel]);
    sum = simd_sum(sum);
    if (lane == 0) output[row] = sum;
}
#define NU_ATTN_SCORES_ARGS device const float *, device float *, constant AttentionParams &, uint, uint
template [[host_name("nu_attention_scores")]] kernel void nu_attention_scores_t<float>(device const float *, NU_ATTN_SCORES_ARGS);
template [[host_name("nu_attention_scores_h")]] kernel void nu_attention_scores_t<half>(device const half *, NU_ATTN_SCORES_ARGS);
template [[host_name("nu_attention_values")]] kernel void nu_attention_values_t<float>(device const float *, NU_ATTN_SCORES_ARGS);
template [[host_name("nu_attention_values_h")]] kernel void nu_attention_values_t<half>(device const half *, NU_ATTN_SCORES_ARGS);

// Rounds two F32 vectors to F16: the attention projections compute
// keys and values in F32 and this writes the cache slot. `count0` elements
// of pair 0 then `count1` of pair 1, one thread per element; a single
// vector passes `count1 = 0`. Round-to-nearest-even, as the CPU's
// `@floatCast`, so a host conversion of the same floats is bit-identical.
struct PackParams { uint count0; uint count1; };
kernel void nu_pack_half(device half * dst0 [[buffer(0)]], device const float * src0 [[buffer(1)]],
                         device half * dst1 [[buffer(2)]], device const float * src1 [[buffer(3)]],
                         constant PackParams & p [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    if (i < p.count0) dst0[i] = half(src0[i]);
    else if (i - p.count0 < p.count1) dst1[i - p.count0] = half(src1[i - p.count0]);
}

// GGML's Q8_0 block: an F16 scale, then 32 signed codes; value = d * q.
// 34 bytes, 2-byte aligned, the `q8_0` attention cache's storage.
struct nu_q8_0 { half d; char qs[32]; };
// Quantizes two F32 vectors to Q8_0 (the `PackParams` contract; both
// counts are multiples of 32), one SIMD group per block: d = half(amax /
// 127), q = clamp(round(x / d), -127, 127), zero codes when d is zero or
// not finite, NaN coded as zero. The arithmetic of quant/encode.zig
// `q8_0Row`, bit-identical with it (precise division, half conversion to
// nearest even, rounding half away from zero).
kernel void nu_pack_q8_0(device nu_q8_0 * dst0 [[buffer(0)]], device const float * src0 [[buffer(1)]],
                         device nu_q8_0 * dst1 [[buffer(2)]], device const float * src1 [[buffer(3)]],
                         constant PackParams & p [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    // Block boundaries are multiples of 32, so a SIMD group returns or stays whole.
    const bool first = i < p.count0;
    const uint e = first ? i : i - p.count0;
    if (!first && e >= p.count1) return;
    const float x = first ? src0[e] : src1[e];
    const float amax = simd_max(isnan(x) ? 0.0f : abs(x));
    const half d = half(precise::divide(amax, 127.0f));
    const float scale = float(d);
    const bool zero = scale == 0.0f || !isfinite(scale) || isnan(x);
    const float q = zero ? 0.0f : clamp(round(precise::divide(x, scale)), -127.0f, 127.0f);
    device nu_q8_0 & block = (first ? dst0 : dst1)[e / 32];
    block.qs[e % 32] = char(q);
    if (e % 32 == 0) block.d = d;
}
// Expands two Q8_0 vectors to F16, one thread per value (the `PackParams`
// contract): the prefill chunk's half view of a `q8_0` cache. `d * q` is
// exact in F32 and rounds once to half.
kernel void nu_unpack_q8_0(device half * dst0 [[buffer(0)]], device const nu_q8_0 * src0 [[buffer(1)]],
                           device half * dst1 [[buffer(2)]], device const nu_q8_0 * src1 [[buffer(3)]],
                           constant PackParams & p [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    if (i < p.count0) dst0[i] = half(float(src0[i / 32].d) * float(src0[i / 32].qs[i % 32]));
    else if (i - p.count0 < p.count1) { const uint e = i - p.count0; dst1[e] = half(float(src1[e / 32].d) * float(src1[e / 32].qs[e % 32])); }
}
// A cache row of F32, F16, or Q8_0 blocks: `nu_kv_row` finds the row
// starting at element `offset` (a block boundary for Q8_0) once, outside
// the per-channel loop. `nu_kv_channel` is the channel lane `lane` owns in
// slot `i` of its CH: `lane + 32 i` for F32 and F16 (one element per
// load), four consecutive channels per 128-wide stripe for Q8_0, so a lane
// reads one 4-byte code word and one scale per stripe. `nu_kv_load` fills
// a lane's CH slots (zero past `width`; CH a multiple of 4).
inline device const float * nu_kv_row(device const float * p, ulong offset) { return p + offset; }
inline device const half * nu_kv_row(device const half * p, ulong offset) { return p + offset; }
inline device const nu_q8_0 * nu_kv_row(device const nu_q8_0 * p, ulong offset) { return p + offset / 32; }
inline uint nu_kv_channel(device const float *, uint lane, uint i) { return lane + 32 * i; }
inline uint nu_kv_channel(device const half *, uint lane, uint i) { return lane + 32 * i; }
inline uint nu_kv_channel(device const nu_q8_0 *, uint lane, uint i) { return 4 * lane + 128 * (i >> 2) + (i & 3); }
template <uint CH>
inline void nu_kv_load(device const float * row, uint lane, uint width, thread float * out) {
    for (uint i = 0; i < CH; ++i) { const uint c = lane + 32 * i; out[i] = c < width ? row[c] : 0.0f; }
}
template <uint CH>
inline void nu_kv_load(device const half * row, uint lane, uint width, thread float * out) {
    for (uint i = 0; i < CH; ++i) { const uint c = lane + 32 * i; out[i] = c < width ? float(row[c]) : 0.0f; }
}
template <uint CH>
inline void nu_kv_load(device const nu_q8_0 * row, uint lane, uint width, thread float * out) {
    for (uint j = 0; j < CH / 4; ++j) {
        const uint c = 4 * lane + 128 * j;
        if (c < width) {
            device const nu_q8_0 & b = row[c >> 5];
            const float d = float(b.d);
            // Codes sit 2 bytes into a 34-byte block: two aligned half-words.
            device const ushort * w = (device const ushort *)(b.qs + (c & 31));
            const char2 lo = as_type<char2>(w[0]), hi = as_type<char2>(w[1]);
            out[4 * j] = d * float(lo.x);
            out[4 * j + 1] = d * float(lo.y);
            out[4 * j + 2] = d * float(hi.x);
            out[4 * j + 3] = d * float(hi.y);
        } else {
            for (uint r = 0; r < 4; ++r) out[4 * j + r] = 0.0f;
        }
    }
}

// ---------------------------------------------------------------------------
// Flash-decoding attention: one query position over `visible` cache
// rows in one pass, no score buffer. Grid = (kv_head, split): a threadgroup
// owns the query heads that share one KV head (GQA, at most NU_DECODE_GROUP)
// over one slice of the visible range, so every cache row is read once for
// the whole group instead of once per query head. Each of its four SIMD
// groups walks every fourth row of the slice (the four rows a threadgroup
// touches per step are adjacent in memory); lane l owns channels l, l+32, …
// of the key and value rows and keeps, per head, a running max `m`, sum `l`,
// and accumulator over its channels (online softmax, all F32; F16 and Q8_0
// rows are converted on load through `nu_kv_load`, queries stay F32; a
// Q8_0 lane owns four consecutive channels per stripe, `nu_kv_channel`). The four SIMD groups merge into the
// first through threadgroup memory with the log-sum-exp rescale, and the
// threadgroup writes one partial (m, l, acc[value_width]) per head;
// `nu_attention_merge` combines the splits per head the same way. An empty
// slice or SIMD group leaves m = -inf and l = 0, which both merges skip.
// `rows` consecutive query positions (a verify batch) add a grid dimension:
// row r sees the first `visible - (rows - 1 - r)` cache rows, the split
// boundaries are the last row's, and a split past a row's end stays empty.
// A nonzero `window` also bounds each row from below (its last `window`
// rows). Queries and outputs are `[row][q_stride | out_stride]`; decode is
// rows = 1, window = 0 (its callers slice the window off the cache).
// Keys and values of one width 128, 256, or 512 run in
// `nu_attention_heads_t` below; this kernel takes every other shape.
struct AttentionDecodeParams { uint query_heads; uint kv_heads; uint key_width; uint value_width; uint visible; uint splits; float scale; uint rows; uint q_stride; uint out_stride; uint window; };
// GROUP query heads per threadgroup and CH channels per lane (widths up to
// 32 * CH); the register budget is GROUP * CH * 2 floats per lane in both
// instantiation pairs (8 × 8 for widths up to 256, 4 × 16 up to 512). A KV
// head with more query heads than GROUP is covered by several threadgroups
// per split (`head_groups`), each reading the slice once (Gemma's
// global layers put 16 query heads of 512 channels over one KV head).
template <typename KV, uint GROUP, uint CH>
kernel void nu_attention_decode_t(device const KV * keys [[buffer(0)]],
                                  device const KV * values [[buffer(1)]],
                                  device const float * queries [[buffer(2)]],
                                  device float * partials [[buffer(3)]],
                                  constant AttentionDecodeParams & p [[buffer(7)]],
                                  uint group [[threadgroup_position_in_grid]],
                                  uint sg [[simdgroup_index_in_threadgroup]],
                                  uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float stage_acc[GROUP * CH * 32];
    threadgroup float stage_m[GROUP];
    threadgroup float stage_l[GROUP];
    const uint g = p.query_heads / p.kv_heads;
    const uint head_groups = (g + GROUP - 1) / GROUP;
    const uint kv = group % p.kv_heads, rest = group / p.kv_heads;
    const uint hg = rest % head_groups, split = (rest / head_groups) % p.splits, row = rest / head_groups / p.splits;
    const uint h0 = hg * GROUP, hn = min(GROUP, g - h0); // this threadgroup's query heads: kv * g + h0 .. + hn
    const uint per = (p.visible + p.splits - 1) / p.splits;
    const uint seen = p.visible - (p.rows - 1 - row);
    const uint low = (p.window != 0 && seen > p.window) ? seen - p.window : 0;
    const uint start = max(split * per, low), end = min(seen, split * per + per);
    queries += ulong(row) * p.q_stride;
    float q[GROUP][CH];
    for (uint h = 0; h < GROUP; ++h) for (uint i = 0; i < CH; ++i) {
        const uint c = nu_kv_channel(keys, lane, i);
        q[h][i] = (h < hn && c < p.key_width) ? queries[(kv * g + h0 + h) * p.key_width + c] * p.scale : 0.0f;
    }
    float m[GROUP], l[GROUP], acc[GROUP][CH];
    for (uint h = 0; h < GROUP; ++h) { m[h] = -INFINITY; l[h] = 0.0f; for (uint i = 0; i < CH; ++i) acc[h][i] = 0.0f; }
    const ulong krow = ulong(p.kv_heads) * p.key_width, vrow = ulong(p.kv_heads) * p.value_width;
    for (uint t = start + sg; t < end; t += 4) {
        device const KV * k = nu_kv_row(keys, ulong(t) * krow + ulong(kv) * p.key_width);
        device const KV * v = nu_kv_row(values, ulong(t) * vrow + ulong(kv) * p.value_width);
        float kr[CH], vr[CH];
        nu_kv_load<CH>(k, lane, p.key_width, kr);
        nu_kv_load<CH>(v, lane, p.value_width, vr);
        for (uint h = 0; h < GROUP; ++h) {
            if (h < hn) {
                float partial = 0.0f;
                for (uint i = 0; i < CH; ++i) partial += q[h][i] * kr[i];
                const float sc = simd_sum(partial);
                const float m_new = max(m[h], sc);
                const float alpha = exp(m[h] - m_new), pr = exp(sc - m_new);
                l[h] = l[h] * alpha + pr;
                m[h] = m_new;
                for (uint i = 0; i < CH; ++i) acc[h][i] = acc[h][i] * alpha + pr * vr[i];
            }
        }
    }
    // Merge SIMD groups 1..3 into group 0, one at a time through the stage.
    for (uint r = 1; r < 4; ++r) {
        if (sg == r) {
            for (uint h = 0; h < GROUP; ++h) {
                if (lane == 0) { stage_m[h] = m[h]; stage_l[h] = l[h]; }
                for (uint i = 0; i < CH; ++i) stage_acc[(h * CH + i) * 32 + lane] = acc[h][i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            for (uint h = 0; h < GROUP; ++h) {
                const float mr = stage_m[h];
                if (mr != -INFINITY) {
                    const float m_new = max(m[h], mr);
                    const float a0 = exp(m[h] - m_new), ar = exp(mr - m_new);
                    l[h] = l[h] * a0 + stage_l[h] * ar;
                    m[h] = m_new;
                    for (uint i = 0; i < CH; ++i) acc[h][i] = acc[h][i] * a0 + stage_acc[(h * CH + i) * 32 + lane] * ar;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (sg != 0) return;
    const uint stride = 2 + p.value_width;
    for (uint h = 0; h < GROUP; ++h) {
        if (h >= hn) break;
        device float * out = partials + ((ulong(row) * p.query_heads + kv * g + h0 + h) * p.splits + split) * stride;
        if (lane == 0) { out[0] = m[h]; out[1] = l[h]; }
        for (uint i = 0; i < CH; ++i) { const uint c = nu_kv_channel(values, lane, i); if (c < p.value_width) out[2 + c] = acc[h][i]; }
    }
}
#define NU_ATTN_DECODE_ARGS device const float *, device float *, constant AttentionDecodeParams &, uint, uint, uint
template [[host_name("nu_attention_decode")]] kernel void nu_attention_decode_t<float, 8, 8>(device const float *, device const float *, NU_ATTN_DECODE_ARGS);
template [[host_name("nu_attention_decode_h")]] kernel void nu_attention_decode_t<half, 8, 8>(device const half *, device const half *, NU_ATTN_DECODE_ARGS);
template [[host_name("nu_attention_decode_w")]] kernel void nu_attention_decode_t<float, 4, 16>(device const float *, device const float *, NU_ATTN_DECODE_ARGS);
template [[host_name("nu_attention_decode_wh")]] kernel void nu_attention_decode_t<half, 4, 16>(device const half *, device const half *, NU_ATTN_DECODE_ARGS);
template [[host_name("nu_attention_decode_q8")]] kernel void nu_attention_decode_t<nu_q8_0, 8, 8>(device const nu_q8_0 *, device const nu_q8_0 *, NU_ATTN_DECODE_ARGS);
template [[host_name("nu_attention_decode_wq8")]] kernel void nu_attention_decode_t<nu_q8_0, 4, 16>(device const nu_q8_0 *, device const nu_q8_0 *, NU_ATTN_DECODE_ARGS);
// The split pass for one head width known at compile time, 32 × CH for
// keys and values alike (CH 4, 8, or 16): `nu_attention_decode_t`'s
// contract and partial layout, built for occupancy, since registers and
// not bytes or ALU bound the general kernel. A threadgroup is (KV head,
// split, row) and holds HG × RL SIMD groups: SIMD group s owns GROUP
// query heads from GROUP × (s % HG) and walks rows start + s / HG, step
// RL, so the head groups read each row together (once from memory, the
// copies from cache). Lane l owns channels CH·l … CH·l + CH − 1, read
// without guards. The RL row lanes merge through `stage` as in the
// general kernel; the host keeps HG × GROUP × 32 × CH within it when
// RL > 1. Design and measurements: docs/engine/metal-backend.md
// § The split pass at the memory roof.
template <uint CH>
inline void nu_heads_load(device const float * row, uint lane, thread float * out) {
    device const packed_float4 * r = (device const packed_float4 *)(row + CH * lane);
    for (uint j = 0; j < CH / 4; ++j) { const float4 x = float4(r[j]); out[4 * j] = x.x; out[4 * j + 1] = x.y; out[4 * j + 2] = x.z; out[4 * j + 3] = x.w; }
}
template <uint CH>
inline void nu_heads_load(device const half * row, uint lane, thread float * out) {
    device const packed_half4 * r = (device const packed_half4 *)(row + CH * lane);
    for (uint j = 0; j < CH / 4; ++j) { const float4 x = float4(half4(r[j])); out[4 * j] = x.x; out[4 * j + 1] = x.y; out[4 * j + 2] = x.z; out[4 * j + 3] = x.w; }
}
// One cache row for a SIMD group's GROUP heads: the dot, the online
// softmax, and the accumulation, per head.
template <uint CH, uint GROUP, typename T>
inline void nu_heads_row(device const T * k, device const T * v, uint lane, thread const float (&q)[GROUP][CH],
                         thread float * m, thread float * l, thread float (&acc)[GROUP][CH]) {
    float kr[CH], vr[CH];
    nu_heads_load<CH>(k, lane, kr);
    nu_heads_load<CH>(v, lane, vr);
    for (uint h = 0; h < GROUP; ++h) {
        float partial = 0.0f;
        for (uint i = 0; i < CH; ++i) partial += q[h][i] * kr[i];
        const float sc = simd_sum(partial);
        const float m_new = max(m[h], sc);
        const float alpha = exp(m[h] - m_new), pr = exp(sc - m_new);
        l[h] = l[h] * alpha + pr;
        m[h] = m_new;
        for (uint i = 0; i < CH; ++i) acc[h][i] = acc[h][i] * alpha + pr * vr[i];
    }
}
// U consecutive codes of one Q8_0 block from code `at`: the codes sit 2
// bytes into a 34-byte block, so they are read as aligned half-words.
template <uint U>
inline void nu_heads_codes(device const nu_q8_0 & b, uint at, thread float * x) {
    device const ushort * w = (device const ushort *)(b.qs + at);
    for (uint u = 0; u < U / 2; ++u) { const char2 e = as_type<char2>(w[u]); x[2 * u] = e.x; x[2 * u + 1] = e.y; }
}
// A Q8_0 row in three phases (every head's dot, the softmax, the V
// accumulation) so K and V codes are never live together, with each
// block's scale outside the per-code products: dot = d · Σ q·code,
// acc += (p · d) · code.
template <uint CH, uint GROUP>
inline void nu_heads_row(device const nu_q8_0 * k, device const nu_q8_0 * v, uint lane, thread const float (&q)[GROUP][CH],
                         thread float * m, thread float * l, thread float (&acc)[GROUP][CH]) {
    constexpr uint U = CH < 8 ? CH : 8;
    float partial[GROUP], pr[GROUP], alpha[GROUP];
    for (uint h = 0; h < GROUP; ++h) partial[h] = 0.0f;
    for (uint j = 0; j < CH / U; ++j) {
        const uint c = CH * lane + U * j;
        const float d = float(k[c >> 5].d);
        float x[U];
        nu_heads_codes<U>(k[c >> 5], c & 31, x);
        for (uint h = 0; h < GROUP; ++h) {
            float s = 0.0f;
            for (uint u = 0; u < U; ++u) s += q[h][U * j + u] * x[u];
            partial[h] += d * s;
        }
    }
    for (uint h = 0; h < GROUP; ++h) {
        const float sc = simd_sum(partial[h]);
        const float m_new = max(m[h], sc);
        alpha[h] = exp(m[h] - m_new);
        pr[h] = exp(sc - m_new);
        l[h] = l[h] * alpha[h] + pr[h];
        m[h] = m_new;
    }
    for (uint j = 0; j < CH / U; ++j) {
        const uint c = CH * lane + U * j;
        const float d = float(v[c >> 5].d);
        float x[U];
        nu_heads_codes<U>(v[c >> 5], c & 31, x);
        for (uint h = 0; h < GROUP; ++h) {
            const float pd = pr[h] * d;
            for (uint u = 0; u < U; ++u) acc[h][U * j + u] = acc[h][U * j + u] * alpha[h] + pd * x[u];
        }
    }
}
template <typename KV, uint GROUP, uint CH>
kernel void nu_attention_heads_t(device const KV * keys [[buffer(0)]],
                                 device const KV * values [[buffer(1)]],
                                 device const float * queries [[buffer(2)]],
                                 device float * partials [[buffer(3)]],
                                 constant AttentionDecodeParams & p [[buffer(7)]],
                                 uint group [[threadgroup_position_in_grid]],
                                 uint sg [[simdgroup_index_in_threadgroup]],
                                 uint simdgroups [[simdgroups_per_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float stage_acc[4096];
    threadgroup float stage_m[128];
    threadgroup float stage_l[128];
    constexpr uint W = 32 * CH;
    const uint g = p.query_heads / p.kv_heads;
    const uint head_groups = (g + GROUP - 1) / GROUP, lanes = simdgroups / head_groups;
    const uint kv = group % p.kv_heads, rest = group / p.kv_heads;
    const uint split = rest % p.splits, row = rest / p.splits;
    const uint hg = sg % head_groups, rl = sg / head_groups;
    const uint h0 = hg * GROUP, hn = min(GROUP, g - h0); // this SIMD group's query heads: kv * g + h0 .. + hn
    const uint per = (p.visible + p.splits - 1) / p.splits;
    const uint seen = p.visible - (p.rows - 1 - row);
    const uint low = (p.window != 0 && seen > p.window) ? seen - p.window : 0;
    const uint start = max(split * per, low), end = min(seen, split * per + per);
    queries += ulong(row) * p.q_stride;
    float q[GROUP][CH];
    for (uint h = 0; h < GROUP; ++h) for (uint i = 0; i < CH; ++i)
        q[h][i] = h < hn ? queries[(kv * g + h0 + h) * W + CH * lane + i] * p.scale : 0.0f;
    float m[GROUP], l[GROUP], acc[GROUP][CH];
    for (uint h = 0; h < GROUP; ++h) { m[h] = -INFINITY; l[h] = 0.0f; for (uint i = 0; i < CH; ++i) acc[h][i] = 0.0f; }
    const ulong stride_row = ulong(p.kv_heads) * W;
    for (uint t = start + rl; t < end; t += lanes)
        nu_heads_row<CH, GROUP>(nu_kv_row(keys, ulong(t) * stride_row + ulong(kv) * W), nu_kv_row(values, ulong(t) * stride_row + ulong(kv) * W), lane, q, m, l, acc);
    // Merge row lanes 1.. into lane 0, one at a time; every head group at once.
    const uint slot = hg * GROUP;
    for (uint r = 1; r < lanes; ++r) {
        if (rl == r) {
            for (uint h = 0; h < GROUP; ++h) {
                if (lane == 0) { stage_m[slot + h] = m[h]; stage_l[slot + h] = l[h]; }
                for (uint i = 0; i < CH; ++i) stage_acc[((slot + h) * CH + i) * 32 + lane] = acc[h][i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (rl == 0) {
            for (uint h = 0; h < GROUP; ++h) {
                const float mr = stage_m[slot + h];
                if (mr != -INFINITY) {
                    const float m_new = max(m[h], mr);
                    const float a0 = exp(m[h] - m_new), ar = exp(mr - m_new);
                    l[h] = l[h] * a0 + stage_l[slot + h] * ar;
                    m[h] = m_new;
                    for (uint i = 0; i < CH; ++i) acc[h][i] = acc[h][i] * a0 + stage_acc[((slot + h) * CH + i) * 32 + lane] * ar;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (rl != 0) return;
    for (uint h = 0; h < GROUP; ++h) {
        if (h >= hn) break;
        device float * out = partials + ((ulong(row) * p.query_heads + kv * g + h0 + h) * p.splits + split) * (2 + W);
        if (lane == 0) { out[0] = m[h]; out[1] = l[h]; }
        for (uint i = 0; i < CH; ++i) out[2 + CH * lane + i] = acc[h][i];
    }
}
// GROUP per (precision, width) must match `Backend.attentionHeadsGroup`.
#define NU_ATTN_HEADS(NAME, KV, GROUP, CH) \
    template [[host_name("nu_attention_heads_" NAME)]] kernel void nu_attention_heads_t<KV, GROUP, CH>(device const KV *, device const KV *, device const float *, device float *, constant AttentionDecodeParams &, uint, uint, uint, uint);
NU_ATTN_HEADS("f32_128", float, 4, 4) NU_ATTN_HEADS("f32_256", float, 2, 8) NU_ATTN_HEADS("f32_512", float, 1, 16)
NU_ATTN_HEADS("f16_128", half, 4, 4) NU_ATTN_HEADS("f16_256", half, 2, 8) NU_ATTN_HEADS("f16_512", half, 1, 16)
NU_ATTN_HEADS("q8_128", nu_q8_0, 4, 4) NU_ATTN_HEADS("q8_256", nu_q8_0, 3, 8) NU_ATTN_HEADS("q8_512", nu_q8_0, 1, 16)
// One threadgroup per (row, query head), 256 threads striding the value
// channels: rescale every split's partial to the global max, divide by the
// merged sum.
kernel void nu_attention_merge(device const float * partials [[buffer(0)]],
                               device float * output [[buffer(1)]],
                               constant AttentionDecodeParams & p [[buffer(7)]],
                               uint head [[threadgroup_position_in_grid]],
                               uint channel [[thread_position_in_threadgroup]]) {
    const uint stride = 2 + p.value_width;
    device const float * part = partials + ulong(head) * p.splits * stride;
    float global_max = -INFINITY;
    for (uint s = 0; s < p.splits; ++s) global_max = max(global_max, part[s * stride]);
    for (uint c = channel; c < p.value_width; c += 256) {
        float total = 0.0f, sum = 0.0f;
        for (uint s = 0; s < p.splits; ++s) {
            const float ms = part[s * stride];
            if (ms == -INFINITY) continue;
            const float a = exp(ms - global_max);
            total += part[s * stride + 1] * a;
            sum += part[s * stride + 2 + c] * a;
        }
        output[ulong(head / p.query_heads) * p.out_stride + (head % p.query_heads) * p.value_width + c] = sum / total;
    }
}

// ---------------------------------------------------------------------------
// Causal tiled attention for a prefill chunk: `count` query rows at
// positions `position..position+count` attend over the cache rows
// `[0, position + t]` in one dispatch, without a query × visible score
// buffer. Flash-attention forward: one threadgroup per (query head, 32-query
// tile); each of its four SIMD groups owns 8 query rows and walks the key
// tiles of 32 positions up to its own causal limit with an online softmax
// (running max `m` and sum `l` per row), using simdgroup 8×8 matrices for
// both Q·Kᵀ and P·V. The 8×32 score tile round-trips through threadgroup
// memory so the row-wise max/exp/sum can run on plain threads (lane l owns
// row l>>2, eight columns); the rescale of the running output by exp(m_old −
// m_new) is a multiply by a diagonal matrix, applied only when some row's
// max moved. Key sub-blocks that would cross the end of the visible range
// are staged through threadgroup memory with rows past the end zeroed, so
// the kernel never reads a cache row beyond `position + count`; masked
// keys get score −∞ (weight exactly 0), so a finite value in a masked row
// cannot leak. Query rows past `count` in the last tile are computed on the
// caller's padding (the encoder requires a multiple of 8 rows). Widths are
// multiples of 8; a `value_width` above 256 (32 accumulators per SIMD group)
// is covered by several threadgroups per (head, tile), each recomputing the
// scores and owning 256 value columns (Gemma's 512-wide global heads).
// A nonzero `window` (sliding layers) hides keys at positions below
// `query − window + 1`: the key loop starts at the tile holding the SIMD
// group's first visible key, and a row whose keys are all hidden in a tile
// keeps its running max at −∞ (its weights are exactly 0) until one shows.
// A nonempty span `[span_begin, span_end)` of chunk rows (the row-split body
// only) lifts those rows' upper bound to the span's last row: they attend
// to each other bidirectionally, and the SIMD group's key loop runs to the
// span's end when any of its rows is inside.
//
// `T` is the operand type: `float` reads an F32 cache with F32
// queries; `half` reads an F16 cache with a half copy of the queries and
// rounds the probability tile to half before P·V, because the matrix unit
// multiplies operands of one type into F32 accumulators (the matmul's
// contract). Scores, the online softmax, the accumulators, and the output
// stay F32 in both instantiations.
struct AttentionChunkParams { uint query_heads; uint kv_heads; uint key_width; uint value_width; uint position; uint count; uint q_stride; uint out_stride; float scale; uint window; uint span_begin; uint span_end; };
#define NU_ATTN_KEYS 32 // keys per tile; the score tile is 8 × NU_ATTN_KEYS per SIMD group
template <typename T>
inline void nu_load_rows_masked(thread simdgroup_matrix<T, 8, 8> & m, device const T * src, ulong stride, uint valid_rows, threadgroup T * stage, uint lane, bool transpose) {
    for (uint i = lane; i < 64; i += 32) { uint r = i >> 3, c = i & 7; stage[i] = r < valid_rows ? src[r * stride + c] : T(0.0f); }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_load(m, stage, 8, ulong2(0, 0), transpose);
    simdgroup_barrier(mem_flags::mem_threadgroup);
}
template <typename T>
kernel void nu_attention_chunk_t(device const T * keys [[buffer(0)]],
                                 device const T * values [[buffer(1)]],
                                 device const T * queries [[buffer(2)]],
                                 device float * output [[buffer(3)]],
                                 constant AttentionChunkParams & p [[buffer(7)]],
                                 uint group [[threadgroup_position_in_grid]],
                                 uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    // Per SIMD group: the F32 score tile and the diagonal, then the T-typed
    // masked-load staging and (half only) the probability tile; in F32 the
    // probabilities overwrite the score tile in place.
    constexpr uint TSTAGE = 64 + (sizeof(T) == 2 ? 8 * NU_ATTN_KEYS : 0);
    threadgroup float scratch[4][8 * NU_ATTN_KEYS + 64];
    threadgroup T tscratch[4][TSTAGE];
    threadgroup float * s = scratch[sg];            // score tile [8][NU_ATTN_KEYS]
    threadgroup float * diag = s + 8 * NU_ATTN_KEYS; // diagonal rescale matrix [8][8]
    threadgroup T * stage = tscratch[sg];           // masked-load staging [8][8]
    threadgroup T * pt = sizeof(T) == 2 ? stage + 64 : (threadgroup T *)s; // probability tile [8][NU_ATTN_KEYS]
    const uint vsplits = (p.value_width + 255) / 256;
    const uint head = group % p.query_heads, rest = group / p.query_heads;
    const uint vs = rest % vsplits, tile = rest / vsplits;
    const uint q0 = tile * 32 + sg * 8;
    if (q0 >= p.count) return; // whole SIMD group past the chunk; no threadgroup barriers follow
    const uint kv = head / (p.query_heads / p.kv_heads);
    const uint total = p.position + p.count;
    const bool group_in_span = q0 < p.span_end && q0 + 8 > p.span_begin;
    const uint limit = max(min(total, p.position + q0 + 8), group_in_span ? p.position + p.span_end : 0u); // exclusive key bound of this SIMD group's rows
    const ulong krow = ulong(p.kv_heads) * p.key_width, vrow = ulong(p.kv_heads) * p.value_width;
    const uint v0 = vs * 256; // this threadgroup's value columns: v0 .. v0 + 8 * vblocks
    device const T * q = queries + ulong(q0) * p.q_stride + ulong(head) * p.key_width;
    device const T * k = keys + ulong(kv) * p.key_width;
    device const T * v = values + ulong(kv) * p.value_width + v0;
    const uint row = lane >> 2, col0 = (lane & 3) * 8;
    const bool row_in_span = q0 + row >= p.span_begin && q0 + row < p.span_end;
    const uint row_limit = row_in_span ? p.position + p.span_end : min(limit, p.position + q0 + row + 1); // exclusive, this lane's query row
    // Sliding window: the first visible key of this lane's row, and of the
    // SIMD group's first row rounded down to a key tile (0 without a window).
    const uint row_lo = (p.window != 0 && p.position + q0 + row + 1 > p.window) ? p.position + q0 + row + 1 - p.window : 0;
    const uint k_begin = (p.window != 0 && p.position + q0 + 1 > p.window) ? (p.position + q0 + 1 - p.window) / NU_ATTN_KEYS * NU_ATTN_KEYS : 0;
    float m = -INFINITY, l = 0.0f;
    simdgroup_float8x8 o[32];
    const uint vblocks = min(p.value_width - v0, 256u) / 8;
    for (uint j = 0; j < 32; ++j) o[j] = simdgroup_float8x8(0.0f);
    for (uint k0 = k_begin; k0 < limit; k0 += NU_ATTN_KEYS) {
        // Scores: S[8][32] = Q[8][key_width] · K[k0..k0+32][key_width]ᵀ, sub-block b covers keys k0+8b.
        simdgroup_float8x8 acc[4];
        for (uint b = 0; b < 4; ++b) acc[b] = simdgroup_float8x8(0.0f);
        for (uint d = 0; d < p.key_width; d += 8) {
            simdgroup_matrix<T, 8, 8> a, kb;
            simdgroup_load(a, q + d, p.q_stride);
            for (uint b = 0; b < 4; ++b) {
                const uint key = k0 + b * 8;
                if (key >= limit) continue;
                if (key + 8 <= limit) simdgroup_load(kb, k + key * krow + d, krow, ulong2(0, 0), true);
                else nu_load_rows_masked(kb, k + key * krow + d, krow, limit - key, stage, lane, true);
                simdgroup_multiply_accumulate(acc[b], a, kb, acc[b]);
            }
        }
        for (uint b = 0; b < 4; ++b) if (k0 + b * 8 < limit) simdgroup_store(acc[b], s + b * 8, NU_ATTN_KEYS);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        // Online softmax on plain threads: row = lane>>2, eight columns per lane.
        float local[8];
        float tile_max = -INFINITY;
        for (uint c = 0; c < 8; ++c) {
            const uint key = k0 + col0 + c;
            local[c] = (key < row_limit && key >= row_lo) ? s[row * NU_ATTN_KEYS + col0 + c] * p.scale : -INFINITY;
            tile_max = max(tile_max, local[c]);
        }
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 1));
        tile_max = max(tile_max, simd_shuffle_xor(tile_max, 2));
        // −∞ only while a windowed row has seen no visible key yet (every row
        // sees its own key by its last tile): then nothing to rescale or add.
        const float m_new = max(m, tile_max);
        const bool empty = m_new == -INFINITY;
        const float alpha = empty ? 1.0f : exp(m - m_new);
        float sum = 0.0f;
        // The sum is taken over the stored (possibly half-rounded) weights, so
        // the normalization matches what P·V multiplies: a perturbed convex
        // combination rather than a mismatched denominator.
        for (uint c = 0; c < 8; ++c) { const T e = T(empty ? 0.0f : exp(local[c] - m_new)); pt[row * NU_ATTN_KEYS + col0 + c] = e; sum += float(e); }
        sum += simd_shuffle_xor(sum, 1);
        sum += simd_shuffle_xor(sum, 2);
        l = l * alpha + sum;
        m = m_new;
        if (simd_any(alpha != 1.0f)) {
            for (uint i = lane; i < 64; i += 32) diag[i] = ((i >> 3) == (i & 7)) ? simd_shuffle(alpha, (i >> 3) * 4) : 0.0f;
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_float8x8 dm;
            simdgroup_load(dm, diag, 8);
            for (uint j = 0; j < 32; ++j) if (j < vblocks) { simdgroup_float8x8 t; simdgroup_multiply(t, dm, o[j]); o[j] = t; }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        // Output: O[8][value_width] += P[8][32] · V[k0..k0+32][value_width].
        for (uint b = 0; b < 4; ++b) {
            const uint key = k0 + b * 8;
            if (key >= limit) continue;
            simdgroup_matrix<T, 8, 8> pm;
            simdgroup_load(pm, pt + b * 8, NU_ATTN_KEYS);
            for (uint j = 0; j < 32; ++j) {
                if (j >= vblocks) continue;
                simdgroup_matrix<T, 8, 8> vb;
                if (key + 8 <= limit) simdgroup_load(vb, v + key * vrow + j * 8, vrow);
                else nu_load_rows_masked(vb, v + key * vrow + j * 8, vrow, limit - key, stage, lane, false);
                simdgroup_multiply_accumulate(o[j], pm, vb, o[j]);
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup); // s / pt are rewritten by the next tile
    }
    // Normalize each row by its softmax sum and store.
    for (uint i = lane; i < 64; i += 32) diag[i] = ((i >> 3) == (i & 7)) ? simd_shuffle(1.0f / l, (i >> 3) * 4) : 0.0f;
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_float8x8 dm;
    simdgroup_load(dm, diag, 8);
    device float * out = output + ulong(q0) * p.out_stride + ulong(head) * p.value_width + v0;
    for (uint j = 0; j < 32; ++j) {
        if (j >= vblocks) continue;
        simdgroup_float8x8 t;
        simdgroup_multiply(t, dm, o[j]);
        simdgroup_store(t, out + j * 8, p.out_stride);
    }
}
#define NU_ATTN_CHUNK_ARGS device float *, constant AttentionChunkParams &, uint, uint, uint
template [[host_name("nu_attention_chunk")]] kernel void nu_attention_chunk_t<float>(device const float *, device const float *, device const float *, NU_ATTN_CHUNK_ARGS);
template [[host_name("nu_attention_chunk_h")]] kernel void nu_attention_chunk_t<half>(device const half *, device const half *, device const half *, NU_ATTN_CHUNK_ARGS);

// Register-reuse causal chunk attention: the same contract, tiling, and
// masking as `nu_attention_chunk_t`, with the four SIMD groups of a (head,
// 32-query) tile splitting the 256 value columns (64 each, every group
// covering all 32 query rows) instead of the query rows. Scores keep the row
// split — a group scores its own 8 rows — and the probabilities plus the
// per-row softmax rescale are published through threadgroup memory, so a V
// block serves four row-block multiplies and a published P block eight
// column-block multiplies instead of one each. Probabilities and rescales
// are double-buffered: a group publishing tile t+1 writes the other buffer
// while a slower group still multiplies tile t's, so the tile loop needs one
// threadgroup barrier per 32-key tile. A group whose 8 rows are past the
// caller's padded row count stays out of the score and softmax phases (it
// would read past the query buffer) but still publishes a zero row and joins
// the barriers.
//
// Live registers per lane at F16: 32 F32 output accumulators (64), the score
// tile (8), the Q and K blocks (2), the softmax row (8), four P blocks and
// one V block (5), the rescale diagonal (2), and the index arithmetic (~10)
// — about 48 live 8×8 matrices, inside the 128-register budget; F32 operands
// double the block costs.
#define NU_ATTN_P 32 // query rows per threadgroup, and the published P tile's row count
template <typename T>
kernel void nu_attention_chunk_reuse_t(device const T * keys [[buffer(0)]],
                                       device const T * values [[buffer(1)]],
                                       device const T * queries [[buffer(2)]],
                                       device float * output [[buffer(3)]],
                                       constant AttentionChunkParams & p [[buffer(7)]],
                                       uint group [[threadgroup_position_in_grid]],
                                       uint sg [[simdgroup_index_in_threadgroup]],
                                       uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float scores[4][8 * NU_ATTN_KEYS + 64];
    threadgroup T probabilities[2][NU_ATTN_P * NU_ATTN_KEYS];
    threadgroup float rescale[2][NU_ATTN_P];
    threadgroup float inverse_sum[NU_ATTN_P];
    threadgroup T stage[4][64];
    threadgroup float * s = scores[sg];                // this group's score tile [8][32]
    threadgroup float * diag = s + 8 * NU_ATTN_KEYS;   // rescale diagonal [8][8]
    const uint vsplits = (p.value_width + 255) / 256;
    const uint head = group % p.query_heads, rest = group / p.query_heads;
    const uint vs = rest % vsplits, tile = rest / vsplits;
    const uint q0 = tile * NU_ATTN_P;                  // first query row of the tile
    const uint kv = head / (p.query_heads / p.kv_heads);
    const uint total = p.position + p.count;
    const uint limit = min(total, p.position + q0 + NU_ATTN_P); // exclusive key bound of the tile
    const uint padded_rows = (p.count + 7) / 8 * 8;    // rows the caller allocated
    const ulong krow = ulong(p.kv_heads) * p.key_width, vrow = ulong(p.kv_heads) * p.value_width;
    const uint v0 = vs * 256;
    const uint vblocks = min(p.value_width - v0, 256u) / 8;
    const uint cb_begin = min(sg * 8, vblocks), cb_end = min(cb_begin + 8, vblocks), cb_cols = cb_end - cb_begin;
    const uint row = lane >> 2, col0 = (lane & 3) * 8;
    const uint grow = sg * 8 + row;                    // this lane's row within the 32-row tile
    const uint row_limit = min(limit, p.position + q0 + grow + 1);
    const uint row_lo = (p.window != 0 && p.position + q0 + grow + 1 > p.window) ? p.position + q0 + grow + 1 - p.window : 0;
    const uint k_begin = (p.window != 0 && p.position + q0 + 1 > p.window) ? (p.position + q0 + 1 - p.window) / NU_ATTN_KEYS * NU_ATTN_KEYS : 0;
    const bool active = q0 + sg * 8 < padded_rows;     // this group's rows exist in the caller's buffers
    device const T * q = queries + ulong(q0 + sg * 8) * p.q_stride + ulong(head) * p.key_width;
    device const T * k = keys + ulong(kv) * p.key_width;
    device const T * v = values + ulong(kv) * p.value_width + v0 + cb_begin * 8;
    float m = -INFINITY, l = 0.0f;
    simdgroup_float8x8 o[32];
    for (uint j = 0; j < 32; ++j) o[j] = simdgroup_float8x8(0.0f);
    uint tile_index = 0;
    for (uint k0 = k_begin; k0 < limit; k0 += NU_ATTN_KEYS, ++tile_index) {
        const uint buf = tile_index & 1u;
        if (active) {
            // Scores: S[8][32] = Q[8][key_width] · K[k0..k0+32][key_width]ᵀ.
            simdgroup_float8x8 acc[4];
            for (uint b = 0; b < 4; ++b) acc[b] = simdgroup_float8x8(0.0f);
            for (uint d = 0; d < p.key_width; d += 8) {
                simdgroup_matrix<T, 8, 8> a, kb;
                simdgroup_load(a, q + d, p.q_stride);
                for (uint b = 0; b < 4; ++b) {
                    const uint key = k0 + b * 8;
                    if (key >= limit) continue;
                    if (key + 8 <= limit) simdgroup_load(kb, k + key * krow + d, krow, ulong2(0, 0), true);
                    else nu_load_rows_masked(kb, k + key * krow + d, krow, limit - key, stage[sg], lane, true);
                    simdgroup_multiply_accumulate(acc[b], a, kb, acc[b]);
                }
            }
            for (uint b = 0; b < 4; ++b) if (k0 + b * 8 < limit) simdgroup_store(acc[b], s + b * 8, NU_ATTN_KEYS);
            simdgroup_barrier(mem_flags::mem_threadgroup);
            // Online softmax on plain threads; publish P and this row's rescale.
            float local[8];
            float tile_max = -INFINITY;
            for (uint c = 0; c < 8; ++c) {
                const uint key = k0 + col0 + c;
                local[c] = (key < row_limit && key >= row_lo) ? s[row * NU_ATTN_KEYS + col0 + c] * p.scale : -INFINITY;
                tile_max = max(tile_max, local[c]);
            }
            tile_max = max(tile_max, simd_shuffle_xor(tile_max, 1));
            tile_max = max(tile_max, simd_shuffle_xor(tile_max, 2));
            const float m_new = max(m, tile_max);
            const bool empty = m_new == -INFINITY;
            const float alpha = empty ? 1.0f : exp(m - m_new);
            float sum = 0.0f;
            for (uint c = 0; c < 8; ++c) { const T e = T(empty ? 0.0f : exp(local[c] - m_new)); probabilities[buf][grow * NU_ATTN_KEYS + col0 + c] = e; sum += float(e); }
            sum += simd_shuffle_xor(sum, 1);
            sum += simd_shuffle_xor(sum, 2);
            l = l * alpha + sum;
            m = m_new;
            if ((lane & 3) == 0) rescale[buf][grow] = alpha;
        } else {
            for (uint c = 0; c < 8; ++c) probabilities[buf][grow * NU_ATTN_KEYS + col0 + c] = T(0.0f);
            if ((lane & 3) == 0) rescale[buf][grow] = 1.0f;
        }
        // The published probabilities and rescales are read by every group;
        // the other buffer of the double buffer is still being multiplied, so
        // this one barrier also orders the previous tile's readers.
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simd_any(rescale[buf][lane] != 1.0f)) {
            for (uint rb = 0; rb < 4; ++rb) {
                for (uint i = lane; i < 64; i += 32) diag[i] = ((i >> 3) == (i & 7)) ? rescale[buf][rb * 8 + (i >> 3)] : 0.0f;
                simdgroup_barrier(mem_flags::mem_threadgroup);
                simdgroup_float8x8 dm;
                simdgroup_load(dm, diag, 8);
                for (uint j = 0; j < cb_cols; ++j) { simdgroup_float8x8 t; simdgroup_multiply(t, dm, o[rb * 8 + j]); o[rb * 8 + j] = t; }
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
        // O += P[32][32] · V[k0..k0+32][64], this group's eight column blocks.
        for (uint b = 0; b < 4; ++b) {
            const uint key = k0 + b * 8;
            if (key >= limit) continue;
            simdgroup_matrix<T, 8, 8> pm[4];
            for (uint rb = 0; rb < 4; ++rb) simdgroup_load(pm[rb], probabilities[buf] + rb * 8 * NU_ATTN_KEYS + b * 8, NU_ATTN_KEYS);
            for (uint j = 0; j < cb_cols; ++j) {
                simdgroup_matrix<T, 8, 8> vb;
                if (key + 8 <= limit) simdgroup_load(vb, v + key * vrow + j * 8, vrow);
                else nu_load_rows_masked(vb, v + key * vrow + j * 8, vrow, limit - key, stage[sg], lane, false);
                for (uint rb = 0; rb < 4; ++rb) simdgroup_multiply_accumulate(o[rb * 8 + j], pm[rb], vb, o[rb * 8 + j]);
            }
        }
    }
    if (active) if ((lane & 3) == 0) inverse_sum[grow] = 1.0f / l;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Normalize each row by its softmax sum and store this group's columns.
    device float * out = output + ulong(q0) * p.out_stride + ulong(head) * p.value_width + v0 + cb_begin * 8;
    for (uint rb = 0; rb < 4; ++rb) {
        if (q0 + rb * 8 >= padded_rows) continue;
        for (uint i = lane; i < 64; i += 32) diag[i] = ((i >> 3) == (i & 7)) ? inverse_sum[rb * 8 + (i >> 3)] : 0.0f;
        simdgroup_barrier(mem_flags::mem_threadgroup);
        simdgroup_float8x8 dm;
        simdgroup_load(dm, diag, 8);
        for (uint j = 0; j < cb_cols; ++j) {
            simdgroup_float8x8 t;
            simdgroup_multiply(t, dm, o[rb * 8 + j]);
            simdgroup_store(t, out + rb * 8 * p.out_stride + j * 8, p.out_stride);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}
template [[host_name("nu_attention_chunk_reuse")]] kernel void nu_attention_chunk_reuse_t<float>(device const float *, device const float *, device const float *, NU_ATTN_CHUNK_ARGS);
template [[host_name("nu_attention_chunk_reuse_h")]] kernel void nu_attention_chunk_reuse_t<half>(device const half *, device const half *, device const half *, NU_ATTN_CHUNK_ARGS);

