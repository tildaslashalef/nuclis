
// Qwen3-TTS's codec decoder (audio/qwen3_codec_metal.zig). Every output
// element is summed in an order fixed by its own index, never by how many
// rows a dispatch covers, so a stream pushed in pieces gives one decode's
// samples bit for bit (the CPU reference's contract, audio/qwen3_codec.zig).

// The split quantizer's lookup: first[t] = book 0 at code 0, rest[t] = the
// books 1…groups−1 at their codes summed in group order. books is
// [groups][size][dim], already divided by the cluster usage; codes [frames][groups].
struct TtsCodebookParams { uint frames; uint groups; uint size; uint dim; };
kernel void nu_tts_codebook_sum(device const float * books [[buffer(0)]],
                                device const uint * codes [[buffer(1)]],
                                device float * first [[buffer(2)]],
                                device float * rest [[buffer(3)]],
                                constant TtsCodebookParams & p [[buffer(7)]],
                                uint id [[thread_position_in_grid]]) {
    if (id >= p.frames * p.dim) return;
    const uint t = id / p.dim, c = id % p.dim;
    device const uint * frame = codes + ulong(t) * p.groups;
    first[id] = books[ulong(frame[0]) * p.dim + c];
    float sum = books[(ulong(p.size) + frame[1]) * p.dim + c];
    for (uint g = 2; g < p.groups; ++g) sum = sum + books[(ulong(g) * p.size + frame[g]) * p.dim + c];
    rest[id] = sum;
}

// A convolution's input rows gathered tap-major: out[r][j·width + i] =
// joined[r + base + j·step][i] for j < taps, zero up to out_stride (the
// matmul tile's 64-column multiple). A causal convolution reads base 0 and
// step = dilation; a transposed one base = taps − 1 and step −1.
struct TtsGatherParams { uint rows; uint width; uint taps; uint out_stride; uint base; int step; };
kernel void nu_tts_gather(device const float * joined [[buffer(0)]],
                          device float * out [[buffer(1)]],
                          constant TtsGatherParams & p [[buffer(7)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= p.rows * p.out_stride) return;
    const uint r = id / p.out_stride, column = id % p.out_stride;
    if (column >= p.taps * p.width) { out[id] = 0.0f; return; }
    const uint j = column / p.width, i = column % p.width;
    const int row = int(r + p.base) + int(j) * p.step;
    out[id] = joined[ulong(row) * p.width + i];
}

// ConvNeXt's causal depthwise convolution: out[t][c] = Σ_j w[c][j] ·
// joined[t + j][c] + b[c], joined holding the taps − 1 history rows first.
struct TtsDepthwiseParams { uint rows; uint channels; uint taps; };
kernel void nu_tts_depthwise(device const float * joined [[buffer(0)]],
                             device const float * weight [[buffer(1)]],
                             device const float * bias [[buffer(2)]],
                             device float * out [[buffer(3)]],
                             constant TtsDepthwiseParams & p [[buffer(7)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= p.rows * p.channels) return;
    const uint t = id / p.channels, c = id % p.channels;
    float acc = 0.0f;
    for (uint j = 0; j < p.taps; ++j) acc = acc + weight[c * p.taps + j] * joined[ulong(t + j) * p.channels + c];
    out[id] = acc + bias[c];
}

// SnakeBeta with its logarithms raised on the host: y = x + inv[c] ·
// sin²(scale[c] · x); y may alias x.
struct TtsSnakeParams { uint count; uint channels; };
kernel void nu_tts_snake(device const float * x [[buffer(0)]],
                         device const float * scale [[buffer(1)]],
                         device const float * inv [[buffer(2)]],
                         device float * y [[buffer(3)]],
                         constant TtsSnakeParams & p [[buffer(7)]],
                         uint i [[thread_position_in_grid]]) {
    if (i >= p.count) return;
    const uint c = i % p.channels;
    const float v = x[i];
    const float s = sin(v * scale[c]);
    y[i] = v + inv[c] * (s * s);
}

// x[i] += scale[i mod channels] · y[i]: the layer scales and ConvNeXt's γ.
struct TtsScaledAddParams { uint count; uint channels; };
kernel void nu_tts_scaled_add(device float * x [[buffer(0)]],
                              device const float * scale [[buffer(1)]],
                              device const float * y [[buffer(2)]],
                              constant TtsScaledAddParams & p [[buffer(7)]],
                              uint i [[thread_position_in_grid]]) {
    if (i < p.count) x[i] = x[i] + scale[i % p.channels] * y[i];
}

// Sliding-window attention of heads of 64 (the codec's transformer): one
// SIMD group per (row, head). Row t sits at key row i = past + t of keys and
// values (the cached rows first) and sees rows i − window + 1 … i. Lane l
// scores the keys lowest + l + 32·m, so a key's lane depends only on its
// distance from the window's start; the value pass walks the keys in order,
// two dimensions a lane.
struct TtsWindowParams { uint rows; uint past; uint heads; uint stride; uint window; float scale; };
kernel void nu_tts_window_attention(device const float * q [[buffer(0)]],
                                    device const float * keys [[buffer(1)]],
                                    device const float * values [[buffer(2)]],
                                    device float * out [[buffer(3)]],
                                    constant TtsWindowParams & p [[buffer(7)]],
                                    uint group [[threadgroup_position_in_grid]],
                                    uint lane [[thread_index_in_simdgroup]]) {
    const uint t = group / p.heads, head = group % p.heads;
    if (t >= p.rows) return;
    const uint i = p.past + t;
    const uint lowest = i + 1 > p.window ? i + 1 - p.window : 0;
    const uint count = i + 1 - lowest;
    device const float * qi = q + ulong(t) * p.stride + head * 64;
    float s[4];
    float top = -INFINITY;
    for (uint m = 0; m < 4; ++m) {
        const uint n = lane + 32 * m;
        if (n < count) {
            device const float * kj = keys + ulong(lowest + n) * p.stride + head * 64;
            float dot = 0.0f;
            for (uint d = 0; d < 64; ++d) dot = dot + qi[d] * kj[d];
            s[m] = dot * p.scale;
            top = max(top, s[m]);
        } else {
            s[m] = -INFINITY;
        }
    }
    top = simd_max(top);
    float part = 0.0f;
    for (uint m = 0; m < 4; ++m) {
        s[m] = lane + 32 * m < count ? exp(s[m] - top) : 0.0f;
        part = part + s[m];
    }
    const float total = simd_sum(part);
    float2 acc = float2(0.0f);
    for (uint n = 0; n < count; ++n) {
        const float w = simd_shuffle(s[n >> 5], ushort(n & 31));
        device const float * vj = values + ulong(lowest + n) * p.stride + head * 64 + 2 * lane;
        acc.x = acc.x + w * vj[0];
        acc.y = acc.y + w * vj[1];
    }
    device float * o = out + ulong(t) * p.stride + head * 64 + 2 * lane;
    o[0] = acc.x / total;
    o[1] = acc.y / total;
}

// The decoder's last convolution, to one channel: out[r] = clamp(b +
// Σ_j Σ_i w[j][i] · joined[r + j][i], −1, 1), joined holding the taps − 1
// history rows first.
struct TtsConvOutParams { uint rows; uint width; uint taps; };
kernel void nu_tts_conv_out(device const float * joined [[buffer(0)]],
                            device const float * weight [[buffer(1)]],
                            device const float * bias [[buffer(2)]],
                            device float * out [[buffer(3)]],
                            constant TtsConvOutParams & p [[buffer(7)]],
                            uint r [[thread_position_in_grid]]) {
    if (r >= p.rows) return;
    float acc = 0.0f;
    for (uint j = 0; j < p.taps; ++j) {
        device const float * row = joined + ulong(r + j) * p.width;
        device const float * w = weight + j * p.width;
        for (uint i = 0; i < p.width; ++i) acc = acc + w[i] * row[i];
    }
    out[r] = min(max(acc + bias[0], -1.0f), 1.0f);
}

// A BF16 matrix times an F32 vector, for the talker and predictor's decode
// steps: each SIMD group owns ROWS rows; a lane reads eight weights at a
// time (two 2-byte-aligned ushort4 loads, as the published tensors are only
// 2-byte aligned) and the eight inputs once for all ROWS rows. BF16 widens
// to F32 exactly (the high half of a single).
struct TtsMatvecParams { uint rows; uint columns; uint stride; };
inline float4 nu_bf16x4(ushort4 v) {
    return float4(as_type<float>(uint(v.x) << 16), as_type<float>(uint(v.y) << 16), as_type<float>(uint(v.z) << 16), as_type<float>(uint(v.w) << 16));
}
template <uint ROWS>
kernel void nu_tts_matvec_bf16_t(device const uchar * weights [[buffer(0)]],
                                 device const float * input [[buffer(1)]],
                                 device float * output [[buffer(2)]],
                                 constant TtsMatvecParams & p [[buffer(7)]],
                                 uint group [[threadgroup_position_in_grid]],
                                 uint sg [[simdgroup_index_in_threadgroup]],
                                 uint lane [[thread_index_in_simdgroup]]) {
    const uint first = (group * 4 + sg) * ROWS;
    if (first >= p.rows) return;
    float sum[ROWS];
    for (uint r = 0; r < ROWS; ++r) sum[r] = 0.0f;
    for (uint c = lane * 8; c < p.columns; c += 256) {
        const float4 x0 = *(device const float4 *)(input + c);
        const float4 x1 = *(device const float4 *)(input + c + 4);
        for (uint r = 0; r < ROWS; ++r) {
            if (first + r >= p.rows) break;
            device const packed_ushort4 * w = (device const packed_ushort4 *)(weights + ulong(first + r) * p.stride + c * 2);
            sum[r] += dot(nu_bf16x4(ushort4(w[0])), x0) + dot(nu_bf16x4(ushort4(w[1])), x1);
        }
    }
    for (uint r = 0; r < ROWS; ++r) {
        const float s = simd_sum(sum[r]);
        if (lane == 0 && first + r < p.rows) output[first + r] = s;
    }
}
template [[host_name("nu_tts_matvec_bf16")]] kernel void nu_tts_matvec_bf16_t<2>(device const uchar *, device const float *, device float *, constant TtsMatvecParams &, uint, uint, uint);
template [[host_name("nu_tts_matvec_bf16_r1")]] kernel void nu_tts_matvec_bf16_t<1>(device const uchar *, device const float *, device float *, constant TtsMatvecParams &, uint, uint, uint);
template [[host_name("nu_tts_matvec_bf16_r4")]] kernel void nu_tts_matvec_bf16_t<4>(device const uchar *, device const float *, device float *, constant TtsMatvecParams &, uint, uint, uint);

// The generic F32 matmul tile (kernels/matmul.metal) with a BF16 segment
// decode in place of `nu_segment`'s switch over every encoding: the same
// values in the same order, so the same sums, for the talker's prefill.
template <>
inline void nu_tile_segment<30>(device const uchar * row, uint encoding, uint segment, thread float4 * q) {
    device const packed_ushort4 * w = (device const packed_ushort4 *)(row + ulong(segment) * 32);
    for (uint i = 0; i < 4; ++i) q[i] = nu_bf16x4(ushort4(w[i]));
}
template [[host_name("nu_tts_matmul_bf16")]] kernel void nu_matmul_t<30, 32, 32, float, float, true>(NU_MATMUL_ARGS);

// A BF16 table's row at a code the device holds: out = row (or out += row)
// for codes[slot], so a frame's predictor inputs and the next talker row
// follow the codes sampled on the device. The row index is clamped to the
// table.
struct TtsEmbedRowParams { uint width; uint rows; uint slot; uint accumulate; };
kernel void nu_tts_embed_row(device const ushort * table [[buffer(0)]],
                             device const uint * codes [[buffer(1)]],
                             device float * out [[buffer(2)]],
                             constant TtsEmbedRowParams & p [[buffer(7)]],
                             uint i [[thread_position_in_grid]]) {
    if (i >= p.width) return;
    const uint row = min(codes[p.slot], p.rows - 1);
    const float v = as_type<float>(uint(table[ulong(row) * p.width + i]) << 16);
    out[i] = p.accumulate != 0 ? out[i] + v : v;
}

// One draw from top-k (top-p) of `count` ≤ 2048 logits at a temperature,
// the CPU sampler's chain (sampling/root.zig `select`) in F32: a bitonic sort
// to (value descending, id ascending) by one 1024-thread group, then thread
// 0 walks the exponentiated candidates with draws[draw] · retained and
// writes the id to codes[slot]. The host checks every such choice against
// the CPU sampler afterwards; F32 rounding can only move a draw that lands
// within rounding of a boundary.
struct TtsSampleParams { uint count; uint slot; uint top_k; uint draw; float top_p; float temperature; };
kernel void nu_tts_sample(device const float * logits [[buffer(0)]],
                          device const float * draws [[buffer(1)]],
                          device uint * codes [[buffer(2)]],
                          constant TtsSampleParams & p [[buffer(7)]],
                          uint tid [[thread_position_in_threadgroup]]) {
    threadgroup float values[2048];
    threadgroup uint ids[2048];
    for (uint i = tid; i < 2048; i += 1024) {
        values[i] = i < p.count ? logits[i] : -INFINITY;
        ids[i] = i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 2; k <= 2048; k <<= 1) {
        for (uint j = k >> 1; j > 0; j >>= 1) {
            const uint i = 2 * tid - (tid & (j - 1));
            const uint l = i + j;
            const float vi = values[i], vl = values[l];
            const uint ii = ids[i], il = ids[l];
            const bool ordered = vi > vl || (vi == vl && ii < il);
            if (((i & k) == 0) != ordered) {
                values[i] = vl;
                values[l] = vi;
                ids[i] = il;
                ids[l] = ii;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
    if (tid != 0) return;
    uint code = ids[0];
    if (p.temperature > 0.0f) {
        const uint k = p.top_k == 0 ? p.count : min(p.top_k, p.count);
        const float top = values[0];
        float sum = 0.0f;
        for (uint i = 0; i < k; ++i) {
            values[i] = exp((values[i] - top) / p.temperature);
            sum += values[i];
        }
        const float threshold = p.top_p * sum;
        uint kept = 0;
        float retained = 0.0f;
        while (kept < k) {
            retained += values[kept];
            kept += 1;
            if (retained >= threshold) break;
        }
        float draw = draws[p.draw] * retained;
        code = ids[kept - 1];
        for (uint i = 0; i < kept; ++i) {
            if (draw < values[i]) {
                code = ids[i];
                break;
            }
            draw -= values[i];
        }
    }
    codes[p.slot] = code;
}

// Three BF16 matrices over one input in one dispatch (a layer's q, k, v):
// the rows of the three are numbered in turn, each SIMD group owns two rows
// of one matrix (every row count a multiple of 8), and a row is summed as
// in `nu_tts_matvec_bf16`.
struct TtsMatvec3Params { uint rows0; uint rows1; uint rows2; uint columns; };
inline float nu_tts_row_bf16(device const uchar * row, device const float * input, uint columns, uint lane) {
    float sum = 0.0f;
    for (uint c = lane * 8; c < columns; c += 256) {
        device const packed_ushort4 * w = (device const packed_ushort4 *)(row + c * 2);
        sum += dot(nu_bf16x4(ushort4(w[0])), *(device const float4 *)(input + c)) + dot(nu_bf16x4(ushort4(w[1])), *(device const float4 *)(input + c + 4));
    }
    return simd_sum(sum);
}
kernel void nu_tts_matvec3_bf16(device const uchar * w0 [[buffer(0)]],
                                device const uchar * w1 [[buffer(1)]],
                                device const uchar * w2 [[buffer(2)]],
                                device const float * input [[buffer(3)]],
                                device float * out0 [[buffer(4)]],
                                device float * out1 [[buffer(5)]],
                                device float * out2 [[buffer(6)]],
                                constant TtsMatvec3Params & p [[buffer(7)]],
                                uint group [[threadgroup_position_in_grid]],
                                uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    uint first = (group * 4 + sg) * 2;
    device const uchar * w = w0;
    device float * out = out0;
    if (first >= p.rows0) {
        first -= p.rows0;
        w = w1;
        out = out1;
        if (first >= p.rows1) {
            first -= p.rows1;
            w = w2;
            out = out2;
            if (first >= p.rows2) return;
        }
    }
    const ulong stride = ulong(p.columns) * 2;
    for (uint r = 0; r < 2; ++r) {
        const float s = nu_tts_row_bf16(w + (first + r) * stride, input, p.columns, lane);
        if (lane == 0) out[first + r] = s;
    }
}

// The feed-forward's first half in one dispatch: out[r] = silu(gate[r] ·
// input) · (up[r] · input) for BF16 gate and up, two rows a SIMD group.
struct TtsGateUpParams { uint rows; uint columns; };
kernel void nu_tts_gate_up_bf16(device const uchar * gate [[buffer(0)]],
                                device const uchar * up [[buffer(1)]],
                                device const float * input [[buffer(2)]],
                                device float * out [[buffer(3)]],
                                constant TtsGateUpParams & p [[buffer(7)]],
                                uint group [[threadgroup_position_in_grid]],
                                uint sg [[simdgroup_index_in_threadgroup]],
                                uint lane [[thread_index_in_simdgroup]]) {
    const uint first = (group * 4 + sg) * 2;
    if (first >= p.rows) return;
    const ulong stride = ulong(p.columns) * 2;
    for (uint r = 0; r < 2 && first + r < p.rows; ++r) {
        const float g = nu_tts_row_bf16(gate + (first + r) * stride, input, p.columns, lane);
        const float u = nu_tts_row_bf16(up + (first + r) * stride, input, p.columns, lane);
        if (lane == 0) out[first + r] = nu_silu(g) * u;
    }
}
