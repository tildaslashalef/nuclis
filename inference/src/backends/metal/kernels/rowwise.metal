// Embedding lookup: dequantize row `token` into `output`; one thread per segment.
struct EmbedParams { uint columns; uint encoding; uint stride; uint token; };
kernel void nu_embed(device const uchar * weights [[buffer(0)]],
                     device float * output [[buffer(1)]],
                     constant EmbedParams & p [[buffer(7)]],
                     uint segment [[thread_position_in_grid]]) {
    if (segment >= p.columns/16) return;
    float values[16];
    nu_segment(weights + ulong(p.token) * p.stride, p.encoding, segment, values);
    for (uint i = 0; i < 16; ++i) output[segment*16+i] = values[i];
}

// ---------------------------------------------------------------------------
// The activation side of a folded Hadamard rotation (cpu/hadamard.zig,
// docs/models/bonsai.md): per block of 1,024 consecutive elements,
// forward x = H (s ⊙ x) and inverse x = s ⊙ (H x), H the normalized
// Sylvester Walsh-Hadamard matrix (scale 1/32 exactly). One 256-thread group
// per block of one row; each thread owns four consecutive values, does the
// first two butterfly stages in registers and the other eight through
// threadgroup memory, two pairs per stage. The signs are the width's vector.
struct HadamardParams { uint width; uint stride; uint rows; uint blocks; uint inverse; };
#define NU_HADAMARD_BLOCK 1024
kernel void nu_hadamard(device float * data [[buffer(0)]],
                        device const float * signs [[buffer(1)]],
                        constant HadamardParams & p [[buffer(7)]],
                        uint group [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float v[NU_HADAMARD_BLOCK];
    const uint row = group / p.blocks, block = group % p.blocks;
    if (row >= p.rows) return;
    device float * x = data + ulong(row) * p.stride + block * NU_HADAMARD_BLOCK;
    device const float * s = signs + block * NU_HADAMARD_BLOCK;
    const uint i = tid * 4;
    float4 a = *(device float4 *)(x + i);
    if (!p.inverse) a *= *(device const float4 *)(s + i);
    float4 b = float4(a.x + a.y, a.x - a.y, a.z + a.w, a.z - a.w);
    a = float4(b.x + b.z, b.y + b.w, b.x - b.z, b.y - b.w);
    v[i] = a.x; v[i + 1] = a.y; v[i + 2] = a.z; v[i + 3] = a.w;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint span = 4; span < NU_HADAMARD_BLOCK; span *= 2) {
        for (uint j = tid; j < NU_HADAMARD_BLOCK / 2; j += 256) {
            uint lo = (j / span) * 2 * span + (j % span), hi = lo + span;
            float u = v[lo], w = v[hi];
            v[lo] = u + w;
            v[hi] = u - w;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float4 out = float4(v[i], v[i + 1], v[i + 2], v[i + 3]) * (1.0f / 32.0f);
    if (p.inverse) out *= *(device const float4 *)(s + i);
    *(device float4 *)(x + i) = out;
}

// A fixed permutation of the `groups` vectors of `width` inside every row:
// dst[r][g] = src[r][map[g]]. One thread per element.
struct GatherParams { uint width; uint groups; uint rows; uint in_stride; uint out_stride; };
kernel void nu_gather_rows(device float * dst [[buffer(0)]],
                           device const float * src [[buffer(1)]],
                           device const uint * map [[buffer(2)]],
                           constant GatherParams & p [[buffer(7)]],
                           uint i [[thread_position_in_grid]]) {
    const uint per_row = p.groups * p.width;
    const uint row = i / per_row, rest = i % per_row;
    if (row >= p.rows) return;
    const uint g = rest / p.width, j = rest % p.width;
    dst[ulong(row) * p.out_stride + g * p.width + j] = src[ulong(row) * p.in_stride + map[g] * p.width + j];
}

// ---------------------------------------------------------------------------
// RMSNorm with learned weight: y = x * rsqrt(mean(x^2) + eps) * w, optionally
// multiplied by silu(multiplier) (flags & 1). One 256-thread group per row.
struct NormParams { uint width; uint in_stride; uint out_stride; uint mult_stride; float eps; uint flags; };
kernel void nu_rmsnorm(device const float * input [[buffer(0)]],
                       device const float * weight [[buffer(1)]],
                       device float * output [[buffer(2)]],
                       device const float * multiplier [[buffer(3)]],
                       constant NormParams & p [[buffer(7)]],
                       uint row [[threadgroup_position_in_grid]],
                       uint tid [[thread_position_in_threadgroup]],
                       uint lane [[thread_index_in_simdgroup]],
                       uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[8];
    device const float * x = input + ulong(row) * p.in_stride;
    float sum = 0;
    for (uint i = tid; i < p.width; i += 256) sum += x[i] * x[i];
    sum = simd_sum(sum);
    if (lane == 0) partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0;
    for (uint i = 0; i < 8; ++i) total += partial[i];
    float scale = rsqrt(total / float(p.width) + p.eps);
    device float * y = output + ulong(row) * p.out_stride;
    device const float * z = multiplier + ulong(row) * p.mult_stride;
    // Every lane finished reading x before any write: rows may alias in place.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < p.width; i += 256) {
        float v = x[i] * scale * weight[i];
        if (p.flags & 1) v *= nu_silu(z[i]);
        y[i] = v;
    }
}

// The post-norm fusion, norm first: `destination = (destination +
// norm(input)·w) · scale`, the exact pair `rmsNorm` then `addScale` (whose
// factor is applied after the sum). Same reduction and geometry as
// nu_rmsnorm; `input` and `destination` must not overlap (the unfused pair
// normalizes `input` in place).
struct NormAddParams { uint width; uint in_stride; uint out_stride; float eps; float scale; };
kernel void nu_rmsnorm_add(device const float * input [[buffer(0)]],
                           device const float * weight [[buffer(1)]],
                           device float * destination [[buffer(2)]],
                           constant NormAddParams & p [[buffer(7)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[8];
    device const float * x = input + ulong(row) * p.in_stride;
    float sum = 0;
    for (uint i = tid; i < p.width; i += 256) sum += x[i] * x[i];
    sum = simd_sum(sum);
    if (lane == 0) partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0;
    for (uint i = 0; i < 8; ++i) total += partial[i];
    float inv = rsqrt(total / float(p.width) + p.eps);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float * y = destination + ulong(row) * p.out_stride;
    for (uint i = tid; i < p.width; i += 256) y[i] = (y[i] + x[i] * inv * weight[i]) * p.scale;
}

// The post-norm fusion, add first: `residual += input` then
// `output = norm(residual)·w`, replacing the `add` + `rmsNorm` pair when the
// norm consumes the updated residual. The first pass stores the sum it
// accumulates, so the reduction divides by the mean square of the stored
// residual exactly as the unfused pair does. `output` must not alias
// `input`.
struct AddNormParams { uint width; uint in_stride; uint out_stride; float eps; };
kernel void nu_add_rmsnorm(device float * residual [[buffer(0)]],
                           device const float * input [[buffer(1)]],
                           device const float * weight [[buffer(2)]],
                           device float * output [[buffer(3)]],
                           constant AddNormParams & p [[buffer(7)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[8];
    device float * x = residual + ulong(row) * p.in_stride;
    device const float * s = input + ulong(row) * p.in_stride;
    float sum = 0;
    for (uint i = tid; i < p.width; i += 256) {
        const float t = x[i] + s[i];
        x[i] = t;
        sum += t * t;
    }
    sum = simd_sum(sum);
    if (lane == 0) partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0;
    for (uint i = 0; i < 8; ++i) total += partial[i];
    float inv = rsqrt(total / float(p.width) + p.eps);
    device float * y = output + ulong(row) * p.out_stride;
    for (uint i = tid; i < p.width; i += 256) y[i] = x[i] * inv * weight[i];
}

// L2 normalization in place: x / max(sqrt(sum(x^2)), eps). One SIMD group per
// row; rows are grouped `heads` per token row of `row_stride` floats (a
// single token passes heads = rows, row_stride = 0).
struct L2Params { uint width; uint stride; float eps; uint rows; uint heads; uint row_stride; };
kernel void nu_l2norm(device float * data [[buffer(0)]],
                      constant L2Params & p [[buffer(7)]],
                      uint row [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]]) {
    device float * x = data + ulong(row / p.heads) * p.row_stride + ulong(row % p.heads) * p.stride;
    float sum = 0;
    for (uint i = lane; i < p.width; i += 32) sum += x[i] * x[i];
    sum = simd_sum(sum);
    float scale = 1.0f / max(sqrt(sum), p.eps);
    for (uint i = lane; i < p.width; i += 32) x[i] *= scale;
}

// Rotary embedding from a precomputed table: table[position][i] = (cos, sin)
// for i < dims/2. Rotates the leading `dims` channels of each head in place.
// Pair i is (i, i + dims/2) with `pairing` 0 (split-half) and (2i, 2i + 1)
// with 1 (adjacent, the GGUF "normal" rope type): cpu.rope.Pairing.
struct RopeParams { uint heads; uint head_stride; uint dims; uint position; uint pairing; };
inline void nu_rotate_pair(device float * x, uint i, uint half_dims, uint pairing, float2 cs) {
    uint first = pairing ? 2 * i : i, second = pairing ? 2 * i + 1 : i + half_dims;
    float a = x[first], b = x[second];
    x[first] = a * cs.x - b * cs.y;
    x[second] = a * cs.y + b * cs.x;
}
kernel void nu_rope(device float * data [[buffer(0)]],
                    device const float2 * table [[buffer(1)]],
                    constant RopeParams & p [[buffer(7)]],
                    uint index [[thread_position_in_grid]]) {
    uint half_dims = p.dims / 2;
    if (index >= p.heads * half_dims) return;
    uint head = index / half_dims, i = index % half_dims;
    float2 cs = table[ulong(p.position) * half_dims + i];
    nu_rotate_pair(data + ulong(head) * p.head_stride, i, half_dims, p.pairing, cs);
}

// Rotary embedding over a chunk of token rows: row t holds `heads` heads at
// `row_stride` and sits at position `position + t`. Same arithmetic as nu_rope.
struct RopeRowsParams { uint heads; uint head_stride; uint dims; uint position; uint rows; uint row_stride; uint pairing; };
kernel void nu_rope_rows(device float * data [[buffer(0)]],
                         device const float2 * table [[buffer(1)]],
                         constant RopeRowsParams & p [[buffer(7)]],
                         uint index [[thread_position_in_grid]]) {
    uint half_dims = p.dims / 2, per_row = p.heads * half_dims;
    if (index >= p.rows * per_row) return;
    uint row = index / per_row, rest = index % per_row, head = rest / half_dims, i = rest % half_dims;
    float2 cs = table[ulong(p.position + row) * half_dims + i];
    nu_rotate_pair(data + ulong(row) * p.row_stride + ulong(head) * p.head_stride, i, half_dims, p.pairing, cs);
}

// The q/k fusion: `output = norm(input)·w` then the leading `dims` channels
// of every row are rotated at `position` with the same table, pairing, and
// arithmetic as nu_rope, replacing the `rmsNorm` + `rope` pair. One
// 256-thread group per row; the rotation runs after the normalized row is
// written and visible (`mem_device` barrier), so `output` may alias `input`.
struct NormRopeParams { uint width; uint in_stride; uint out_stride; uint dims; uint position; uint pairing; float eps; };
kernel void nu_rmsnorm_rope(device const float * input [[buffer(0)]],
                            device const float * weight [[buffer(1)]],
                            device const float2 * table [[buffer(2)]],
                            device float * output [[buffer(3)]],
                            constant NormRopeParams & p [[buffer(7)]],
                            uint row [[threadgroup_position_in_grid]],
                            uint tid [[thread_position_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[8];
    device const float * x = input + ulong(row) * p.in_stride;
    float sum = 0;
    for (uint i = tid; i < p.width; i += 256) sum += x[i] * x[i];
    sum = simd_sum(sum);
    if (lane == 0) partial[sg] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0;
    for (uint i = 0; i < 8; ++i) total += partial[i];
    float inv = rsqrt(total / float(p.width) + p.eps);
    device float * y = output + ulong(row) * p.out_stride;
    // Every lane finished reading x before any write: rows may alias in place.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < p.width; i += 256) y[i] = x[i] * inv * weight[i];
    threadgroup_barrier(mem_flags::mem_device);
    const uint half_dims = p.dims / 2;
    if (tid < half_dims) {
        const float2 cs = table[ulong(p.position) * half_dims + tid];
        nu_rotate_pair(y, tid, half_dims, p.pairing, cs);
    }
}

// Elementwise helpers.
struct CountParams { uint count; };
kernel void nu_copy(device float * dst [[buffer(0)]], device const float * src [[buffer(1)]],
                    constant CountParams & p [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    if (i < p.count) dst[i] = src[i];
}
kernel void nu_add(device float * x [[buffer(0)]], device const float * y [[buffer(1)]],
                   constant CountParams & p [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    if (i < p.count) x[i] += y[i];
}
kernel void nu_silu_mul(device float * gate [[buffer(0)]], device const float * up [[buffer(1)]],
                        constant CountParams & p [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    if (i < p.count) gate[i] = nu_silu(gate[i]) * up[i];
}
// DFlash2's two-tap dynamic convolution over a block's rows (`dflash2.convolve`):
// out[t][c] = Σ_{tap ≤ min(t, 1)} (base[half][tap][c] + taps[t][half][tap][c / group]) · in[t − tap][c].
struct DflashConvParams { uint rows; uint width; uint half_index; uint group; uint groups; uint taps_stride; };
kernel void nu_dflash_conv(device const float * base [[buffer(0)]], device const float * taps [[buffer(1)]],
                           device const float * input [[buffer(2)]], device float * output [[buffer(3)]],
                           constant DflashConvParams & p [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    if (i >= p.rows * p.width) return;
    uint t = i / p.width, c = i % p.width;
    float sum = 0.0f;
    for (uint tap = 0; tap < 2 && tap <= t; ++tap) {
        uint which = p.half_index * 2 + tap;
        float w = base[which * p.width + c] + taps[t * p.taps_stride + which * p.groups + c / p.group];
        sum += w * input[(t - tap) * p.width + c];
    }
    output[i] = sum;
}
kernel void nu_silu_inplace(device float * x [[buffer(0)]],
                            constant CountParams & p [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    if (i < p.count) x[i] = nu_silu(x[i]);
}
kernel void nu_gelu_mul(device float * gate [[buffer(0)]], device const float * up [[buffer(1)]],
                        constant CountParams & p [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    if (i < p.count) gate[i] = nu_gelu(gate[i]) * up[i];
}
// The quick GELU gate, x·σ(1.702x), of Gemma 4's vision encoder.
kernel void nu_gelu_quick_mul(device float * gate [[buffer(0)]], device const float * up [[buffer(1)]],
                              constant CountParams & p [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    if (i < p.count) { const float x = gate[i]; gate[i] = x / (1.0f + exp(-1.702f * x)) * up[i]; }
}
// The exact GELU, 0.5·x·(1 + erf(x/√2)), of Muse Glimmer's vision encoder.
// Metal has no erf: Abramowitz & Stegun 7.1.26, |error| ≤ 1.5e-7.
inline float nu_erf(float z) {
    const float a = fabs(z);
    const float t = 1.0f / (1.0f + 0.3275911f * a);
    const float poly = ((((1.061405429f * t - 1.453152027f) * t + 1.421413741f) * t - 0.284496736f) * t + 0.254829592f) * t;
    return copysign(1.0f - poly * exp(-a * a), z);
}
kernel void nu_gelu_erf_inplace(device float * x [[buffer(0)]],
                                constant CountParams & p [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    if (i < p.count) { const float v = x[i]; x[i] = 0.5f * v * (1.0f + nu_erf(v * 0.7071067811865476f)); }
}
// The exact-GELU gate over strided rows: out[r][i] = gelu_erf(gate[r][i]) ·
// up[r][i] (ModernBERT's GeGLU; the halves of one fused row bound at offsets).
kernel void nu_gelu_erf_mul_rows(device const float * gate [[buffer(0)]],
                                 device const float * up [[buffer(1)]],
                                 device float * output [[buffer(2)]],
                                 constant GeluRowsParams & p [[buffer(7)]],
                                 uint id [[thread_position_in_grid]]) {
    if (id >= p.width * p.rows) return;
    uint row = id / p.width, i = id % p.width;
    const float v = gate[ulong(row) * p.gate_stride + i];
    output[ulong(row) * p.out_stride + i] = 0.5f * v * (1.0f + nu_erf(v * 0.7071067811865476f)) * up[ulong(row) * p.up_stride + i];
}
// Scalar epilogues (Gemma 4): x *= factor (the embedding scale);
// x = (x + y) * factor (a residual add followed by a per-layer output
// scale, one rounding after the add as the CPU reference does it); and
// x = cap * tanh(x / cap), the final logit soft-cap (factor = cap).
struct ScaleParams { uint count; float factor; };
kernel void nu_scale(device float * x [[buffer(0)]],
                     constant ScaleParams & p [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    if (i < p.count) x[i] *= p.factor;
}
struct ClampParams { uint count; float low; float high; };
kernel void nu_clamp(device float * x [[buffer(0)]],
                     constant ClampParams & p [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    if (i < p.count) x[i] = min(max(x[i], p.low), p.high);
}
kernel void nu_add_scale(device float * x [[buffer(0)]], device const float * y [[buffer(1)]],
                         constant ScaleParams & p [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    if (i < p.count) x[i] = (x[i] + y[i]) * p.factor;
}
kernel void nu_softcap(device float * x [[buffer(0)]],
                       constant ScaleParams & p [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    if (i < p.count) x[i] = p.factor * nu_tanh(x[i] / p.factor);
}
// DeltaNet gates: alpha[h] = a[h] * softplus(alpha[h] + bias[h]); beta[h] = sigmoid(beta[h]),
// over `count` entries that repeat every `heads` (one token: count = heads).
struct DeltaGatesParams { uint count; uint heads; };
kernel void nu_delta_gates(device float * alpha [[buffer(0)]], device float * beta [[buffer(1)]],
                           device const float * a [[buffer(2)]], device const float * bias [[buffer(3)]],
                           constant DeltaGatesParams & p [[buffer(7)]], uint h [[thread_position_in_grid]]) {
    if (h >= p.count) return;
    uint head = h % p.heads;
    alpha[h] = a[head] * nu_softplus(alpha[h] + bias[head]);
    beta[h] = nu_sigmoid(beta[h]);
}
// Attention output gate: out[h*width+i] *= sigmoid(gates[h*gate_stride + gate_offset + i]).
struct GateParams { uint heads; uint width; uint gate_stride; uint gate_offset; };
kernel void nu_sigmoid_gate(device float * out [[buffer(0)]], device const float * gates [[buffer(1)]],
                            constant GateParams & p [[buffer(7)]], uint index [[thread_position_in_grid]]) {
    if (index >= p.heads * p.width) return;
    uint h = index / p.width, i = index % p.width;
    out[index] *= nu_sigmoid(gates[h * p.gate_stride + p.gate_offset + i]);
}

