// ---------------------------------------------------------------------------
// Scalar-gated DeltaNet: one SIMD group per (value head, value coordinate).
// State is [value_head][value_coordinate][key_coordinate], updated in place.
// Input layout: [Q: qheads*keys | K: qheads*keys | V: vheads*values].
// Value head h reads Q/K head h % qheads (tiled), or h / group when group
// is nonzero (grouped: each Q/K head over `group` consecutive value heads).
struct DeltaParams { uint qheads; uint vheads; uint keys; uint values; float scale; uint group; };
kernel void nu_delta(device float * state [[buffer(0)]],
                     device const float * input [[buffer(1)]],
                     device const float * decay_log [[buffer(2)]],
                     device const float * beta_gate [[buffer(3)]],
                     device float * output [[buffer(4)]],
                     constant DeltaParams & p [[buffer(7)]],
                     uint row [[threadgroup_position_in_grid]],
                     uint lane [[thread_index_in_simdgroup]]) {
    uint head = row / p.values, value = row % p.values, kh = p.group ? head / p.group : head % p.qheads;
    uint qsize = p.qheads * p.keys;
    float decay = exp(decay_log[head]);
    float beta = beta_gate[head];
    float prediction = 0;
    for (uint k = lane; k < p.keys; k += 32) prediction += (state[row*p.keys+k]*decay) * input[qsize + kh*p.keys + k];
    prediction = simd_sum(prediction);
    float correction = beta * (input[2*qsize + head*p.values + value] - prediction);
    float result = 0;
    for (uint k = lane; k < p.keys; k += 32) {
        float next = state[row*p.keys+k]*decay + input[qsize + kh*p.keys + k]*correction;
        state[row*p.keys+k] = next;
        result += next * input[kh*p.keys + k];
    }
    result = simd_sum(result) * p.scale;
    if (lane == 0) output[row] = result;
}

// A verify batch's DeltaNet rows, token by token: nu_delta's arithmetic with
// the state row in registers, read once and never written. Each row t also
// writes a replay tape that nu_delta_replay applies to the accepted prefix:
// row t's normalized key per Q/K head and decay per value head at
// tape[t·key_stride], its correction per (value head, value) at
// tape[corr_offset + t·vheads·values]. keys is a multiple of 32, at most 128.
struct DeltaRowsParams { uint qheads; uint vheads; uint keys; uint values; uint count; uint in_stride; uint gate_stride; uint out_stride; uint key_stride; uint corr_offset; float scale; };
#define NU_DELTA_LANE_KEYS 4
kernel void nu_delta_rows(device const float * state [[buffer(0)]],
                          device const float * input [[buffer(1)]],
                          device const float * decay_log [[buffer(2)]],
                          device const float * beta_gate [[buffer(3)]],
                          device float * output [[buffer(4)]],
                          device float * tape [[buffer(5)]],
                          constant DeltaRowsParams & p [[buffer(7)]],
                          uint row [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_simdgroup]]) {
    uint head = row / p.values, value = row % p.values, kh = head % p.qheads;
    uint qsize = p.qheads * p.keys, n = p.keys / 32;
    float s[NU_DELTA_LANE_KEYS];
    for (uint i = 0; i < n; ++i) s[i] = state[row*p.keys + lane + 32*i];
    for (uint t = 0; t < p.count; ++t) {
        device const float * in = input + ulong(t) * p.in_stride;
        device float * keys = tape + ulong(t) * p.key_stride;
        float decay = exp(decay_log[t*p.gate_stride + head]);
        float beta = beta_gate[t*p.gate_stride + head];
        float prediction = 0;
        for (uint i = 0; i < n; ++i) prediction += (s[i]*decay) * in[qsize + kh*p.keys + lane + 32*i];
        prediction = simd_sum(prediction);
        float correction = beta * (in[2*qsize + head*p.values + value] - prediction);
        float result = 0;
        for (uint i = 0; i < n; ++i) {
            float next = s[i]*decay + in[qsize + kh*p.keys + lane + 32*i]*correction;
            s[i] = next;
            result += next * in[kh*p.keys + lane + 32*i];
        }
        result = simd_sum(result) * p.scale;
        if (lane == 0) {
            output[ulong(t)*p.out_stride + row] = result;
            tape[p.corr_offset + ulong(t)*p.vheads*p.values + row] = correction;
        }
        // Value row 0 of the first qheads value heads owns the shared keys.
        if (value == 0) {
            if (head < p.qheads) for (uint i = 0; i < n; ++i) keys[head*p.keys + lane + 32*i] = in[qsize + head*p.keys + lane + 32*i];
            if (lane == 0) keys[qsize + head] = decay;
        }
    }
}

// Applies the first `count` rows of a nu_delta_rows tape to the state:
// nu_delta's update with the recorded decay, key, and correction, so the
// state equals `count` decode steps. One SIMD group per (value head, value).
struct DeltaReplayParams { uint qheads; uint vheads; uint keys; uint values; uint count; uint key_stride; uint corr_offset; };
kernel void nu_delta_replay(device float * state [[buffer(0)]],
                            device const float * tape [[buffer(1)]],
                            constant DeltaReplayParams & p [[buffer(7)]],
                            uint row [[threadgroup_position_in_grid]],
                            uint lane [[thread_index_in_simdgroup]]) {
    uint head = row / p.values, kh = head % p.qheads;
    uint qsize = p.qheads * p.keys, n = p.keys / 32;
    float s[NU_DELTA_LANE_KEYS];
    for (uint i = 0; i < n; ++i) s[i] = state[row*p.keys + lane + 32*i];
    for (uint t = 0; t < p.count; ++t) {
        device const float * keys = tape + ulong(t) * p.key_stride;
        float decay = keys[qsize + head];
        float correction = tape[p.corr_offset + ulong(t)*p.vheads*p.values + row];
        for (uint i = 0; i < n; ++i) s[i] = s[i]*decay + keys[kh*p.keys + lane + 32*i]*correction;
    }
    for (uint i = 0; i < n; ++i) state[row*p.keys + lane + 32*i] = s[i];
}

// Depthwise causal convolution with in-place history shift; one thread per channel.
struct ConvParams { uint channels; uint taps; };
kernel void nu_convolution(device float * history [[buffer(0)]],
                           device const float * input [[buffer(1)]],
                           device const float * weights [[buffer(2)]],
                           device float * output [[buffer(3)]],
                           constant ConvParams & p [[buffer(7)]],
                           uint channel [[thread_position_in_grid]]) {
    if (channel >= p.channels) return;
    uint taps = p.taps;
    float sum = input[channel] * weights[channel*taps + taps-1];
    for (uint tap = 0; tap+1 < taps; ++tap) sum += history[channel*(taps-1)+tap] * weights[channel*taps+tap];
    output[channel] = sum;
    for (uint tap = 0; tap+2 < taps; ++tap) history[channel*(taps-1)+tap] = history[channel*(taps-1)+tap+1];
    if (taps > 1) history[channel*(taps-1)+taps-2] = input[channel];
}

// Causal convolution over a chunk of `rows` token rows (stride `stride`):
// out[t][c] = Σ_tap w[c][tap] · in(t − (taps−1) + tap), where inputs before the
// chunk come from `history` (oldest first). The summation order matches
// nu_convolution exactly (current input first, then the older taps), so a
// chunk equals the sequential steps bit for bit. `nu_convolution_history`
// then shifts the last taps−1 inputs into the history, one thread per channel.
struct ConvRowsParams { uint channels; uint taps; uint rows; uint stride; };
inline float nu_conv_input(device const float * history, device const float * input, uint c, int s, uint taps, uint stride) {
    return s < 0 ? history[c * (taps - 1) + uint(int(taps - 1) + s)] : input[ulong(uint(s)) * stride + c];
}
kernel void nu_convolution_rows(device const float * history [[buffer(0)]],
                                device const float * input [[buffer(1)]],
                                device const float * weights [[buffer(2)]],
                                device float * output [[buffer(3)]],
                                constant ConvRowsParams & p [[buffer(7)]],
                                uint index [[thread_position_in_grid]]) {
    if (index >= p.rows * p.channels) return;
    uint t = index / p.channels, c = index % p.channels, taps = p.taps;
    float sum = input[ulong(t) * p.stride + c] * weights[c * taps + taps - 1];
    for (uint tap = 0; tap + 1 < taps; ++tap) sum += nu_conv_input(history, input, c, int(t) - int(taps - 1) + int(tap), taps, p.stride) * weights[c * taps + tap];
    output[ulong(t) * p.stride + c] = sum;
}
kernel void nu_convolution_history(device float * history [[buffer(0)]],
                                   device const float * input [[buffer(1)]],
                                   constant ConvRowsParams & p [[buffer(7)]],
                                   uint c [[thread_position_in_grid]]) {
    if (c >= p.channels) return;
    uint n = p.taps - 1;
    float next[31];
    for (uint j = 0; j < n; ++j) next[j] = nu_conv_input(history, input, c, int(p.rows) - int(n) + int(j), p.taps, p.stride);
    for (uint j = 0; j < n; ++j) history[c * n + j] = next[j];
}

