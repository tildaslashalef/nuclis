// Nemotron 3.5 ASR's encoder and joint (audio/nemotron_encoder.zig,
// nemotron_rnnt.zig): the subsampling's causal stride-2 convolutions, the
// row shifts that keep a stream's caches, the chunk-masked relative
// attention over cached keys, the convolution module's GLU and depthwise
// convolution, and the joint's input. Each mirrors a CPU function named in
// its comment.

// `nemotron_encoder.conv2d`: output (t, f, c) of a 3×3 stride-2 convolution
// causal on both axes, input rows `2t − 2 … 2t` (row y at `y − first`, rows
// outside [0, count) zero) and columns `2f − 2 … 2f`; `in_channels` is 1 (the
// stem) or `channels` (depthwise). Bias added; ReLU when `relu`.
struct AsrConvParams { uint rows; uint next; uint first; uint count; uint width_in; uint width_out; uint channels; uint in_channels; uint relu; };
kernel void nu_asr_conv2d(device const float * input [[buffer(0)]],
                          device const float * weight [[buffer(1)]],
                          device const float * bias [[buffer(2)]],
                          device float * out [[buffer(3)]],
                          constant AsrConvParams & p [[buffer(7)]],
                          uint id [[thread_position_in_grid]]) {
    if (id >= p.rows * p.width_out * p.channels) return;
    const uint c = id % p.channels;
    const uint f = (id / p.channels) % p.width_out;
    const uint t = p.next + id / (p.channels * p.width_out);
    const uint in_row = p.width_in * p.in_channels;
    float acc = 0.0f;
    for (uint kh = 0; kh < 3; ++kh) {
        const int y = int(2 * t + kh) - 2;
        if (y < 0 || uint(y) >= p.count) continue;
        device const float * row = input + ulong(uint(y) - p.first) * in_row;
        for (uint kw = 0; kw < 3; ++kw) {
            const int x = int(2 * f + kw) - 2;
            if (x < 0 || uint(x) >= p.width_in) continue;
            const float v = p.in_channels == 1 ? row[x] : row[uint(x) * p.channels + c];
            acc = fma(weight[(c * 3 + kh) * 3 + kw], v, acc);
        }
    }
    const float o = acc + bias[c];
    out[id] = p.relu != 0 ? max(o, 0.0f) : o;
}

// x[row][i] = max(x[row][i] + bias[i], 0) over `rows` rows of `width`.
struct AsrRowsParams { uint width; uint rows; };
kernel void nu_asr_bias_relu(device float * x [[buffer(0)]], device const float * bias [[buffer(1)]],
                             constant AsrRowsParams & p [[buffer(7)]], uint id [[thread_position_in_grid]]) {
    if (id >= p.width * p.rows) return;
    x[id] = max(x[id] + bias[id % p.width], 0.0f);
}

// Per plane, rows `from … from + keep − 1` move to rows `0 … keep − 1`. One
// thread per (plane, column) copies its column in increasing row order,
// so a move toward the start never overwrites a row it has yet to read.
struct AsrShiftParams { uint width; uint keep; uint from; uint planes; uint plane_stride; };
kernel void nu_asr_shift(device float * x [[buffer(0)]],
                         constant AsrShiftParams & p [[buffer(7)]], uint id [[thread_position_in_grid]]) {
    if (id >= p.planes * p.width) return;
    device float * plane = x + ulong(id / p.width) * p.plane_stride + id % p.width;
    for (uint i = 0; i < p.keep; ++i) plane[ulong(i) * p.width] = plane[ulong(p.from + i) * p.width];
}

// `nemotron_encoder.attention`: one SIMD group per (query, head), 4 of the
// 128 dimensions a lane. Query i is row `first + i`; key slot j is row
// `first − held + j` (the cache's `held` rows, then the chunk's). Query a
// sees rows of chunks ⌊a/chunk⌋ − reach … ⌊a/chunk⌋, scored
// ((q + u)·k + (q + v)·relk[a − b + offset]) · scale, softmaxed online.
struct AsrAttentionParams { uint rows; uint held; uint first; uint chunk; uint reach; uint width; uint offset; float scale; };
kernel void nu_asr_attention(device const float * q [[buffer(0)]],
                             device const float * k [[buffer(1)]],
                             device const float * v [[buffer(2)]],
                             device const float * relk [[buffer(3)]],
                             device const float * bias_u [[buffer(4)]],
                             device const float * bias_v [[buffer(5)]],
                             device float * out [[buffer(6)]],
                             constant AsrAttentionParams & p [[buffer(7)]],
                             uint group [[threadgroup_position_in_grid]],
                             uint lane [[thread_index_in_simdgroup]]) {
    const uint heads = p.width / 128;
    const uint i = group / heads, head = group % heads;
    if (i >= p.rows) return;
    const uint base = head * 128 + lane * 4;
    const float4 qi = ((device const float4 *)(q + ulong(i) * p.width + base))[0];
    const float4 qu = qi + ((device const float4 *)(bias_u + base))[0];
    const float4 qv = qi + ((device const float4 *)(bias_v + base))[0];
    const uint a = p.first + i;
    const uint key_first = p.first - p.held;
    const uint own = a / p.chunk;
    const uint lo = max(key_first, (own >= p.reach ? own - p.reach : 0) * p.chunk);
    const uint hi = min(key_first + p.held + p.rows, (own + 1) * p.chunk);
    float top = -INFINITY, total = 0.0f;
    float4 acc = float4(0.0f);
    for (uint b = lo; b < hi; ++b) {
        const ulong slot = ulong(b - key_first) * p.width + base;
        const float4 kj = ((device const float4 *)(k + slot))[0];
        const float4 rj = ((device const float4 *)(relk + ulong(a + p.offset - b) * p.width + base))[0];
        const float s = simd_sum(dot(qu, kj) + dot(qv, rj)) * p.scale;
        const float next = max(top, s);
        const float carry = exp(top - next), w = exp(s - next);
        total = total * carry + w;
        acc = acc * carry + w * ((device const float4 *)(v + slot))[0];
        top = next;
    }
    ((device float4 *)(out + ulong(i) * p.width + base))[0] = acc / total;
}

// `nemotron_encoder.glu`: out[r][c] = wide[r][c] · σ(wide[r][width + c]).
kernel void nu_asr_glu(device const float * wide [[buffer(0)]], device float * out [[buffer(1)]],
                       constant AsrRowsParams & p [[buffer(7)]], uint id [[thread_position_in_grid]]) {
    if (id >= p.width * p.rows) return;
    device const float * r = wide + ulong(id / p.width) * 2 * p.width;
    const uint c = id % p.width;
    out[id] = r[c] * nu_sigmoid(r[p.width + c]);
}

// `nemotron_encoder.depthwise`: out[t][c] = Σ_tap w[c][tap] · rows[t + tap][c],
// `rows` holding `kernel − 1` earlier rows before the `rows` new ones.
struct AsrDepthwiseParams { uint width; uint rows; uint kernel_size; };
kernel void nu_asr_depthwise(device const float * rows [[buffer(0)]], device const float * weight [[buffer(1)]],
                             device float * out [[buffer(2)]],
                             constant AsrDepthwiseParams & p [[buffer(7)]], uint id [[thread_position_in_grid]]) {
    if (id >= p.width * p.rows) return;
    const uint t = id / p.width, c = id % p.width;
    float acc = 0.0f;
    for (uint tap = 0; tap < p.kernel_size; ++tap) acc = fma(weight[c * p.kernel_size + tap], rows[ulong(t + tap) * p.width + c], acc);
    out[id] = acc;
}

// `nemotron_rnnt.jointInput` per row: out[r][c] = max(e[r][c] + g[c], 0).
kernel void nu_asr_joint(device const float * e [[buffer(0)]], device const float * g [[buffer(1)]],
                         device float * out [[buffer(2)]],
                         constant AsrRowsParams & p [[buffer(7)]], uint id [[thread_position_in_grid]]) {
    if (id >= p.width * p.rows) return;
    out[id] = max(e[id] + g[id % p.width], 0.0f);
}

// x[i] += factor · y[i]: the macaron feed-forward's half step.
struct AsrScaleParams { uint count; float factor; };
kernel void nu_asr_add_scaled(device float * x [[buffer(0)]], device const float * y [[buffer(1)]],
                              constant AsrScaleParams & p [[buffer(7)]], uint id [[thread_position_in_grid]]) {
    if (id < p.count) x[id] = fma(p.factor, y[id], x[id]);
}
