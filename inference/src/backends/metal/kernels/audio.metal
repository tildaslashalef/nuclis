// Gemma 4's audio attention (audio/gemma4a.zig `attention`): one SIMD group
// per (row, head), 4 of the 128 dimensions a lane. Query i sees keys
// i − 11 … i of its clip, scored q·k + q·relk[12 − (i − j)] on scaled q and
// k, capped by tanh, softmaxed. relk is the block's 13 × width relative keys.
struct AudioAttentionParams { uint rows; uint width; float key_scale; float cap; };
kernel void nu_audio_attention(device const float * q [[buffer(0)]],
                               device const float * k [[buffer(1)]],
                               device const float * v [[buffer(2)]],
                               device const float * relk [[buffer(3)]],
                               device const float * qscale [[buffer(4)]],
                               device float * out [[buffer(5)]],
                               constant AudioAttentionParams & p [[buffer(7)]],
                               uint group [[threadgroup_position_in_grid]],
                               uint lane [[thread_index_in_simdgroup]]) {
    const uint i = group / 8, head = group % 8;
    if (i >= p.rows) return;
    const uint base = head * 128 + lane * 4;
    const float4 s = ((device const float4 *)(qscale + lane * 4))[0];
    const float4 qi = ((device const float4 *)(q + ulong(i) * p.width + base))[0] * s;
    const uint lowest = i >= 11 ? i - 11 : 0;
    float scores[12];
    float top = -INFINITY;
    for (uint j = lowest; j <= i; ++j) {
        const float4 kj = ((device const float4 *)(k + ulong(j) * p.width + base))[0] * p.key_scale;
        const float4 rj = ((device const float4 *)(relk + ulong(12 - (i - j)) * p.width + base))[0];
        const float raw = simd_sum(dot(qi, kj) + dot(qi, rj));
        const float capped = p.cap * nu_tanh(raw / p.cap);
        scores[j - lowest] = capped;
        top = max(top, capped);
    }
    float total = 0.0f;
    float4 acc = float4(0.0f);
    for (uint j = lowest; j <= i; ++j) {
        const float w = exp(scores[j - lowest] - top);
        total += w;
        acc += w * ((device const float4 *)(v + ulong(j) * p.width + base))[0];
    }
    ((device float4 *)(out + ulong(i) * p.width + base))[0] = acc / total;
}

// The light convolution's middle (audio/gemma4a.zig `gluConv`): the GLU of a
// `2·width` row (values, then gates) through a causal depthwise convolution
// of 5 taps; dw is `[width][5]`.
struct GluConvParams { uint rows; uint width; };
kernel void nu_glu_conv(device const float * start [[buffer(0)]],
                        device const float * dw [[buffer(1)]],
                        device float * out [[buffer(2)]],
                        constant GluConvParams & p [[buffer(7)]],
                        uint id [[thread_position_in_grid]]) {
    if (id >= p.rows * p.width) return;
    const uint t = id / p.width, c = id % p.width;
    float acc = 0.0f;
    for (uint tap = 0; tap < 5; ++tap) {
        if (t + tap < 4) continue;
        device const float * r = start + ulong(t + tap - 4) * 2 * p.width;
        const float glu = r[c] / (1.0f + exp(-r[p.width + c]));
        acc = fma(dw[c * 5 + tap], glu, acc);
    }
    out[id] = acc;
}
