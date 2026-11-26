
// ---------------------------------------------------------------------------
// Attention over the depth (`cpu.depth`): the mix of a token's block states
// by learned queries, and the message controller's norms and blend.

// `out[token][j] = Σ_s softmax_s(q_j · K_s / √width) · K_s` with K_s state
// s over its unweighted RMS (`eps` inside the root): each token's `count`
// states are rows of `width` from `states + token · state_stride`, the
// `queries` query rows of `width` are shared, and the output rows of
// `width` start at `output + token · out_stride`. One 256-thread group per
// (token, query); `count` is at most NU_DEPTH_MAX.
#define NU_DEPTH_MAX 32
struct DepthMixParams { uint width; uint count; uint queries; uint state_stride; uint out_stride; float eps; };
kernel void nu_depth_mix(device const float * states [[buffer(0)]],
                         device const float * queries [[buffer(1)]],
                         device float * output [[buffer(2)]],
                         constant DepthMixParams & p [[buffer(7)]],
                         uint group [[threadgroup_position_in_grid]],
                         uint tid [[thread_position_in_threadgroup]],
                         uint lane [[thread_index_in_simdgroup]],
                         uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float squares[8][NU_DEPTH_MAX];
    threadgroup float dots[8][NU_DEPTH_MAX];
    const uint token = group / p.queries, j = group % p.queries;
    device const float * st = states + ulong(token) * p.state_stride;
    device const float * q = queries + ulong(j) * p.width;
    for (uint s = 0; s < p.count; ++s) {
        float a = 0.0f, b = 0.0f;
        for (uint c = tid; c < p.width; c += 256) { const float v = st[s * p.width + c]; a += v * v; b += q[c] * v; }
        a = simd_sum(a);
        b = simd_sum(b);
        if (lane == 0) { squares[sg][s] = a; dots[sg][s] = b; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Every thread forms the same weights: p_s / (norm_s · Σ p).
    float weight[NU_DEPTH_MAX], inverse_norm[NU_DEPTH_MAX];
    float m = -INFINITY;
    const float scale = rsqrt(float(p.width));
    for (uint s = 0; s < p.count; ++s) {
        float a = 0.0f, b = 0.0f;
        for (uint i = 0; i < 8; ++i) { a += squares[i][s]; b += dots[i][s]; }
        inverse_norm[s] = 1.0f / sqrt(a / float(p.width) + p.eps);
        weight[s] = b * inverse_norm[s] * scale;
        m = max(m, weight[s]);
    }
    float total = 0.0f;
    for (uint s = 0; s < p.count; ++s) { weight[s] = exp(weight[s] - m); total += weight[s]; }
    for (uint s = 0; s < p.count; ++s) weight[s] *= inverse_norm[s] / total;
    device float * out = output + ulong(token) * p.out_stride + ulong(j) * p.width;
    for (uint c = tid; c < p.width; c += 256) {
        float sum = 0.0f;
        for (uint s = 0; s < p.count; ++s) sum += weight[s] * st[s * p.width + c];
        out[c] = sum;
    }
}

// The controller's input (`cpu.depth.control`): `joined[row]` =
// [rms(current[row]); rms(message[row])], each half over its own unweighted
// RMS (`eps` inside the root). One 256-thread group per (row, half).
struct ControlJoinParams { uint width; uint rows; uint current_stride; uint message_stride; uint joined_stride; float eps; };
kernel void nu_control_join(device const float * current [[buffer(0)]],
                            device const float * message [[buffer(1)]],
                            device float * joined [[buffer(2)]],
                            constant ControlJoinParams & p [[buffer(7)]],
                            uint group [[threadgroup_position_in_grid]],
                            uint tid [[thread_position_in_threadgroup]],
                            uint lane [[thread_index_in_simdgroup]],
                            uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partial[8];
    const uint row = group / 2, half_index = group % 2;
    device const float * x = half_index == 0 ? current + ulong(row) * p.current_stride : message + ulong(row) * p.message_stride;
    float a = 0.0f;
    for (uint c = tid; c < p.width; c += 256) a += x[c] * x[c];
    a = simd_sum(a);
    if (lane == 0) partial[sg] = a;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint i = 0; i < 8; ++i) total += partial[i];
    const float inverse = 1.0f / sqrt(total / float(p.width) + p.eps);
    device float * out = joined + ulong(row) * p.joined_stride + half_index * p.width;
    for (uint c = tid; c < p.width; c += 256) out[c] = x[c] * inverse;
}

// The controller's blend: out = (1 − tanh(forget)) ⊙ current + tanh(gain) ⊙
// message per row; `output` may alias `current` exactly.
struct ControlBlendParams { uint width; uint rows; uint current_stride; uint message_stride; uint gate_stride; uint out_stride; };
kernel void nu_control_blend(device const float * current [[buffer(0)]],
                             device const float * message [[buffer(1)]],
                             device const float * gain [[buffer(2)]],
                             device const float * forget [[buffer(3)]],
                             device float * output [[buffer(4)]],
                             constant ControlBlendParams & p [[buffer(7)]],
                             uint id [[thread_position_in_grid]]) {
    if (id >= p.width * p.rows) return;
    const uint row = id / p.width, c = id % p.width;
    const ulong g = ulong(row) * p.gate_stride + c;
    const float kept = (1.0f - nu_tanh(forget[g])) * current[ulong(row) * p.current_stride + c];
    output[ulong(row) * p.out_stride + c] = kept + nu_tanh(gain[g]) * message[ulong(row) * p.message_stride + c];
}
