// nuclis compute kernels. Convention: data buffers bind at indices 0..6 in the
// order listed by each kernel; a small parameter struct binds at index 7.
// All arithmetic is F32; reductions use SIMD-group sums and, where a row is
// wider than one SIMD group serves well, a 256-thread threadgroup.

inline float nu_silu(float x) { return x / (1.0f + exp(-x)); }
inline float nu_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }
inline float nu_softplus(float x) { return max(x, 0.0f) + log(1.0f + exp(-abs(x))); }
// tanh saturating exactly: Metal's tanh goes through exp and returns NaN
// past about ±44 (exp overflow), where the true value is ±1 to F32
// precision already at ±20 (1 − 2e-17 rounds to 1). Clamping there changes
// no finite result and keeps large gate values and logits finite.
inline float nu_tanh(float x) { return tanh(clamp(x, -20.0f, 20.0f)); }
// GELU in the tanh approximation (`cpu.gelu`, the form GGUF checkpoints
// declare as gelu_pytorch_tanh; Gemma 4's FFN gate).
inline float nu_gelu(float x) { return 0.5f * x * (1.0f + nu_tanh(0.7978845608028654f * x * (1.0f + 0.044715f * x * x))); }

// ---------------------------------------------------------------------------
// Matrix-vector product: one SIMD group per output row. Each lane decodes
// consecutive 16-value segments directly from quantized storage.
struct MatvecParams { uint columns; uint encoding; uint stride; uint rows; };
kernel void nu_matvec(device const uchar * weights [[buffer(0)]],
                      device const float * input [[buffer(1)]],
                      device float * output [[buffer(2)]],
                      constant MatvecParams & p [[buffer(7)]],
                      uint row [[threadgroup_position_in_grid]],
                      uint lane [[thread_index_in_simdgroup]]) {
    if (row >= p.rows) return;
    device const uchar * bytes = weights + ulong(row) * p.stride;
    float sum = 0;
    float values[16];
    for (uint segment = lane; segment < p.columns/16; segment += 32) {
        nu_segment(bytes, p.encoding, segment, values);
        for (uint i = 0; i < 16; ++i) sum += values[i] * input[segment*16+i];
    }
    sum = simd_sum(sum);
    if (lane == 0) output[row] = sum;
}

// ---------------------------------------------------------------------------
// Specialized matrix-vector kernels for Q4_K, Q5_K, Q6_K, and IQ4_XS: the four
// encodings holding 96 % of the model's bytes, so these decide decode speed.
// KERN-03 added Q3_K and IQ3_S, MODL-08 Q4_0 (the QAT Gemma checkpoint's only
// weight encoding).
//
// Geometry: 128-thread groups of four SIMD groups. Each SIMD group owns ROWS
// consecutive output rows and walks their 256-value blocks four at a time,
// eight lanes per block (lane>>3 selects the block, lane&7 the slice). A lane
// decodes 32 values of its slice from one to three vector loads. Rather than
// forming every value, it accumulates per 32-value group the sums q·x and x
// and applies the group's scale and minimum once:
//     Σ (d·s·q − dmin·m)·x  =  d·s·Σ(q·x) − dmin·m·Σx
// With a one-hot input this reproduces the CPU decoder's expression exactly
// (Σq·x = q, Σx = 1), which is what the pinned fixture check relies on; for
// dense inputs it differs only by F32 rounding order. The 32 inputs are read as
// eight float4 from device memory (the vector stays hot in cache) once per lane
// per block and shared by the ROWS rows — that sharing is why several rows per
// SIMD group beat one.
//
// Vector-load widths follow each block's natural alignment: Q4_K/Q5_K blocks
// (144/176 B) are 16-byte aligned, IQ4_XS (136 B) 8-byte, Q6_K (210 B) and
// Q4_0 (18 B) only 2-byte. The Zig encoder verifies row offset/stride
// alignment before choosing a specialized kernel and otherwise records
// nu_matvec. `blocks` counts 256-value strides; the Q4_0 body walks its own
// 32-value blocks from `columns` instead.
struct MatvecBlockParams { uint columns; uint stride; uint rows; uint blocks; };
#define NU_MATVEC_SIMDGROUPS 4 // must match simdgroups_per_matvec_group in root.zig

inline float nu_half_low(uint w) { return float(as_type<half>(ushort(w & 0xffffu))); }
inline float nu_half_high(uint w) { return float(as_type<half>(ushort(w >> 16))); }
inline uint nu_byte(uint w, uint index) { return (w >> (8 * index)) & 255u; }
inline float nu_signed_byte(uint w, uint index) { return float(as_type<char>(uchar(nu_byte(w, index)))); }
// Codes are assembled in the packed byte domain (one integer op serves four
// values) and converted with a single uchar4 -> float4 cast; the GPU is scalar
// per lane, so building float4s element by element costs 3-4x more.
inline float4 nu_bytes(uint w) { return float4(as_type<uchar4>(w)); }
inline float4 nu_low_nibbles(uint w) { return nu_bytes(w & 0x0f0f0f0fu); }
inline float4 nu_high_nibbles(uint w) { return nu_bytes((w >> 4) & 0x0f0f0f0fu); }
// Q5_K: nibbles of `v` completed with bit `shift` of each byte of the plane word `h`.
inline float4 nu_low_fives(uint v, uint h, uint shift) { return nu_bytes((v & 0x0f0f0f0fu) | (((h >> shift) & 0x01010101u) << 4)); }
inline float4 nu_high_fives(uint v, uint h, uint shift) { return nu_bytes(((v >> 4) & 0x0f0f0f0fu) | (((h >> shift) & 0x01010101u) << 4)); }
// Q6_K: nibbles of `l` plus the two bits at `shift` of each byte of `hb`; the
// bias of 32 is applied by the caller through the input sum.
inline float4 nu_low_sixes(uint l, uint hb, uint shift) { return nu_bytes((l & 0x0f0f0f0fu) | (((hb >> shift) & 0x03030303u) << 4)); }
inline float4 nu_high_sixes(uint l, uint hb, uint shift) { return nu_bytes(((l >> 4) & 0x0f0f0f0fu) | (((hb >> shift) & 0x03030303u) << 4)); }
inline float4 nu_iq4_low(uint w) { uchar4 n = as_type<uchar4>(w & 0x0f0f0f0fu); return float4(nu_iq4_values_f[n.x], nu_iq4_values_f[n.y], nu_iq4_values_f[n.z], nu_iq4_values_f[n.w]); }
inline float4 nu_iq4_high(uint w) { uchar4 n = as_type<uchar4>((w >> 4) & 0x0f0f0f0fu); return float4(nu_iq4_values_f[n.x], nu_iq4_values_f[n.y], nu_iq4_values_f[n.z], nu_iq4_values_f[n.w]); }
// acc + a·b as four fused multiply-adds: MTLMathModeSafe never contracts on its
// own, and explicit fma halves the instruction count of a dot product.
inline float nu_dot(float4 a, float4 b, float acc) { return fma(a.w, b.w, fma(a.z, b.z, fma(a.y, b.y, fma(a.x, b.x, acc)))); }
inline float nu_sum4(float4 v) { return (v.x + v.y) + (v.z + v.w); }
inline uint nu_word(packed_ushort4 v, uint index) { return index == 0 ? (uint(v.x) | (uint(v.y) << 16)) : (uint(v.z) | (uint(v.w) << 16)); }
// Sixteen consecutive inputs as four float4 plus their sum (16-byte aligned).
struct NuInputs16 { float4 v[4]; float sum; };
inline NuInputs16 nu_inputs16(device const float * x) {
    NuInputs16 r;
    device const float4 * v = (device const float4 *)x;
    r.v[0] = v[0]; r.v[1] = v[1]; r.v[2] = v[2]; r.v[3] = v[3];
    float4 s = (r.v[0] + r.v[1]) + (r.v[2] + r.v[3]);
    r.sum = (s.x + s.y) + (s.z + s.w);
    return r;
}
// Scale and minimum of two adjacent 32-value groups (2*pair, 2*pair+1) from the
// twelve packed Q4_K/Q5_K scale bytes held in three words. Groups 0-3 store six
// bits directly in words 0 (scale) and 1 (minimum); groups 4-7 keep their low
// four bits in word 2 and their top two bits in the high bits of words 0/1 —
// the same rule as nu_k_scale_min, for a pair at once. Lanes of one SIMD group
// take both forms, so this is written with selects rather than a branch.
inline void nu_k_scales_pair(packed_uint3 s, uint pair, thread float & scale_a, thread float & min_a, thread float & scale_b, thread float & min_b) {
    bool direct = pair < 2;
    uint shift = direct ? 16 * pair : 16 * pair - 32; // bit offset of byte 2*pair (or 2*pair-4)
    uint x = s.x >> shift, y = s.y >> shift, z = s.z >> shift; // bytes 0 and 1 of each are groups a and b
    uint sa = direct ? (x & 63u) : ((z & 15u) | ((x >> 6) & 3u) << 4);
    uint ma = direct ? (y & 63u) : (((z >> 4) & 15u) | ((y >> 6) & 3u) << 4);
    uint sb = direct ? ((x >> 8) & 63u) : (((z >> 8) & 15u) | ((x >> 14) & 3u) << 4);
    uint mb = direct ? ((y >> 8) & 63u) : (((z >> 12) & 15u) | ((y >> 14) & 3u) << 4);
    scale_a = float(sa); min_a = float(ma); scale_b = float(sb); min_b = float(mb);
}
// Writes the ROWS reduced sums; every lane must reach this (simd_sum is collective).
template <uint ROWS>
inline void nu_store_rows(thread float * acc, device float * output, uint row0, uint rows, uint lane) {
    for (uint r = 0; r < ROWS; ++r) {
        float total = simd_sum(acc[r]);
        if (lane == 0 && row0 + r < rows) output[row0 + r] = total;
    }
}

// Q4_K (FIFTH_BIT false) and Q5_K (true). Lane slice: groups 2*pair and
// 2*pair+1, columns 16*half..16*half+15 of each — the low and high nibbles of
// the same sixteen bytes. Q5_K adds the fifth bit from the 32-byte plane, bit
// `group` of byte `column`.
template <uint ROWS, bool FIFTH_BIT>
inline void nu_matvec_k_body(device const uchar * weights, device const float * input, MatvecBlockParams p, uint group_index, uint sg, uint lane, thread float * acc) {
    const uint block_bytes = FIFTH_BIT ? 176 : 144;
    const uint nibble_offset = FIFTH_BIT ? 48 : 16;
    uint row0 = (group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS;
    if (row0 >= p.rows) return;
    uint pair = (lane & 7) >> 1, half_index = lane & 1;
    for (uint r = 0; r < ROWS; ++r) acc[r] = 0;
    for (uint kb = lane >> 3; kb < p.blocks; kb += 4) {
        device const float * x = input + kb * 256 + pair * 64 + half_index * 16;
        NuInputs16 xa = nu_inputs16(x), xb = nu_inputs16(x + 32);
        uint slice = kb * block_bytes + nibble_offset + pair * 32 + half_index * 16;
        uint plane = kb * block_bytes + 16 + half_index * 16;
        for (uint r = 0; r < ROWS; ++r) {
            device const uchar * row = weights + ulong(min(row0 + r, p.rows - 1)) * p.stride;
            uint dd = *(device const uint *)(row + kb * block_bytes);
            packed_uint3 s = *(device const packed_uint3 *)(row + kb * block_bytes + 4);
            uint4 v = *(device const uint4 *)(row + slice);
            float4 qa0, qa1, qa2, qa3, qb0, qb1, qb2, qb3;
            if (FIFTH_BIT) {
                uint4 h = *(device const uint4 *)(row + plane);
                uint bit_a = 2 * pair, bit_b = bit_a + 1;
                qa0 = nu_low_fives(v.x, h.x, bit_a); qa1 = nu_low_fives(v.y, h.y, bit_a); qa2 = nu_low_fives(v.z, h.z, bit_a); qa3 = nu_low_fives(v.w, h.w, bit_a);
                qb0 = nu_high_fives(v.x, h.x, bit_b); qb1 = nu_high_fives(v.y, h.y, bit_b); qb2 = nu_high_fives(v.z, h.z, bit_b); qb3 = nu_high_fives(v.w, h.w, bit_b);
            } else {
                qa0 = nu_low_nibbles(v.x); qa1 = nu_low_nibbles(v.y); qa2 = nu_low_nibbles(v.z); qa3 = nu_low_nibbles(v.w);
                qb0 = nu_high_nibbles(v.x); qb1 = nu_high_nibbles(v.y); qb2 = nu_high_nibbles(v.z); qb3 = nu_high_nibbles(v.w);
            }
            float sqa = nu_dot(qa3, xa.v[3], nu_dot(qa2, xa.v[2], nu_dot(qa1, xa.v[1], nu_dot(qa0, xa.v[0], 0.0f))));
            float sqb = nu_dot(qb3, xb.v[3], nu_dot(qb2, xb.v[2], nu_dot(qb1, xb.v[1], nu_dot(qb0, xb.v[0], 0.0f))));
            float d = nu_half_low(dd), dmin = nu_half_high(dd);
            float sa, ma, sb, mb;
            nu_k_scales_pair(s, pair, sa, ma, sb, mb);
            acc[r] += ((d * sa) * sqa - (dmin * ma) * xa.sum) + ((d * sb) * sqb - (dmin * mb) * xb.sum);
        }
    }
}
template <uint ROWS>
kernel void nu_matvec_q4_k(device const uchar * weights [[buffer(0)]], device const float * input [[buffer(1)]], device float * output [[buffer(2)]],
                           constant MatvecBlockParams & p [[buffer(7)]], uint group_index [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    float acc[ROWS];
    nu_matvec_k_body<ROWS, false>(weights, input, p, group_index, sg, lane, acc);
    if ((group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS < p.rows)
        nu_store_rows<ROWS>(acc, output, (group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS, p.rows, lane);
}
template <uint ROWS>
kernel void nu_matvec_q5_k(device const uchar * weights [[buffer(0)]], device const float * input [[buffer(1)]], device float * output [[buffer(2)]],
                           constant MatvecBlockParams & p [[buffer(7)]], uint group_index [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    float acc[ROWS];
    nu_matvec_k_body<ROWS, true>(weights, input, p, group_index, sg, lane, acc);
    if ((group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS < p.rows)
        nu_store_rows<ROWS>(acc, output, (group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS, p.rows, lane);
}

// Q6_K. Lane slice: half `h` (values h*128..), columns 8*c8..8*c8+7 of all
// four 32-value quarters. Byte (h*64 + column) holds quarters 0 and 2 in its
// nibbles, byte (h*64 + 32 + column) quarters 1 and 3, and byte
// (128 + h*32 + column) two high bits per quarter; codes are biased by 32 and
// each group of sixteen has a signed byte scale. Blocks are 2-byte aligned, so
// loads are packed_ushort4.
template <uint ROWS>
inline void nu_matvec_q6_k_body(device const uchar * weights, device const float * input, MatvecBlockParams p, uint group_index, uint sg, uint lane, thread float * acc) {
    uint row0 = (group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS;
    if (row0 >= p.rows) return;
    uint h = (lane & 7) >> 2, c8 = lane & 3, scale_byte = c8 >> 1;
    for (uint r = 0; r < ROWS; ++r) acc[r] = 0;
    for (uint kb = lane >> 3; kb < p.blocks; kb += 4) {
        device const float4 * x = (device const float4 *)(input + kb * 256 + h * 128 + c8 * 8);
        float4 x0a = x[0], x0b = x[1], x1a = x[8], x1b = x[9], x2a = x[16], x2b = x[17], x3a = x[24], x3b = x[25];
        // Σ(q−32)·x = Σq·x − 32·Σx: the per-quarter input sums are shared by all rows.
        float sx0 = nu_sum4(x0a + x0b), sx1 = nu_sum4(x1a + x1b), sx2 = nu_sum4(x2a + x2b), sx3 = nu_sum4(x3a + x3b);
        uint base = kb * 210;
        for (uint r = 0; r < ROWS; ++r) {
            device const uchar * b = weights + ulong(min(row0 + r, p.rows - 1)) * p.stride + base;
            packed_ushort4 l0 = *(device const packed_ushort4 *)(b + h * 64 + c8 * 8);
            packed_ushort4 l1 = *(device const packed_ushort4 *)(b + h * 64 + 32 + c8 * 8);
            packed_ushort4 hb = *(device const packed_ushort4 *)(b + 128 + h * 32 + c8 * 8);
            packed_ushort4 sc = *(device const packed_ushort4 *)(b + 192 + h * 8);
            float d = float(as_type<half>(*(device const ushort *)(b + 208)));
            uint l0a = nu_word(l0, 0), l0b = nu_word(l0, 1), l1a = nu_word(l1, 0), l1b = nu_word(l1, 1), hba = nu_word(hb, 0), hbb = nu_word(hb, 1);
            float s0 = nu_dot(nu_low_sixes(l0b, hbb, 0), x0b, nu_dot(nu_low_sixes(l0a, hba, 0), x0a, 0.0f));
            float s1 = nu_dot(nu_low_sixes(l1b, hbb, 2), x1b, nu_dot(nu_low_sixes(l1a, hba, 2), x1a, 0.0f));
            float s2 = nu_dot(nu_high_sixes(l0b, hbb, 4), x2b, nu_dot(nu_high_sixes(l0a, hba, 4), x2a, 0.0f));
            float s3 = nu_dot(nu_high_sixes(l1b, hbb, 6), x3b, nu_dot(nu_high_sixes(l1a, hba, 6), x3a, 0.0f));
            float c0 = nu_signed_byte(uint(sc.x), scale_byte), c1 = nu_signed_byte(uint(sc.y), scale_byte);
            float c2 = nu_signed_byte(uint(sc.z), scale_byte), c3 = nu_signed_byte(uint(sc.w), scale_byte);
            acc[r] += (d * c0) * fma(-32.0f, sx0, s0) + (d * c1) * fma(-32.0f, sx1, s1) + (d * c2) * fma(-32.0f, sx2, s2) + (d * c3) * fma(-32.0f, sx3, s3);
        }
    }
}
template <uint ROWS>
kernel void nu_matvec_q6_k(device const uchar * weights [[buffer(0)]], device const float * input [[buffer(1)]], device float * output [[buffer(2)]],
                            constant MatvecBlockParams & p [[buffer(7)]], uint group_index [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    float acc[ROWS];
    nu_matvec_q6_k_body<ROWS>(weights, input, p, group_index, sg, lane, acc);
    if ((group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS < p.rows)
        nu_store_rows<ROWS>(acc, output, (group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS, p.rows, lane);
}


// IQ4_XS. Lane slice: one 32-value group `g`: sixteen bytes at 8 + 16*g whose
// low nibbles are values 0-15 and high nibbles values 16-31, through the
// nonlinear table. The six-bit group scale is split between four low-nibble
// bytes and a 16-bit word of high pairs; blocks are 8-byte aligned.
template <uint ROWS>
inline void nu_matvec_iq4_xs_body(device const uchar * weights, device const float * input, MatvecBlockParams p, uint group_index, uint sg, uint lane, thread float * acc) {
    uint row0 = (group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS;
    if (row0 >= p.rows) return;
    uint g = lane & 7;
    for (uint r = 0; r < ROWS; ++r) acc[r] = 0;
    for (uint kb = lane >> 3; kb < p.blocks; kb += 4) {
        device const float4 * x = (device const float4 *)(input + kb * 256 + g * 32);
        float4 x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3], x4 = x[4], x5 = x[5], x6 = x[6], x7 = x[7];
        uint base = kb * 136;
        for (uint r = 0; r < ROWS; ++r) {
            device const uchar * b = weights + ulong(min(row0 + r, p.rows - 1)) * p.stride + base;
            uint2 header = *(device const uint2 *)b;
            uint2 qa = *(device const uint2 *)(b + 8 + g * 16);
            uint2 qb = *(device const uint2 *)(b + 16 + g * 16);
            float d = nu_half_low(header.x);
            uint low = (header.y >> (4 * g)) & 15u, high = (header.x >> (16 + 2 * g)) & 3u;
            float scale = float(int(low | (high << 4)) - 32);
            float sum = nu_dot(nu_iq4_low(qa.x), x0, nu_dot(nu_iq4_low(qa.y), x1, nu_dot(nu_iq4_low(qb.x), x2, nu_dot(nu_iq4_low(qb.y), x3, 0.0f))));
            sum = nu_dot(nu_iq4_high(qa.x), x4, nu_dot(nu_iq4_high(qa.y), x5, nu_dot(nu_iq4_high(qb.x), x6, nu_dot(nu_iq4_high(qb.y), x7, sum))));
            acc[r] += (d * scale) * sum;
        }
    }
}
template <uint ROWS>
kernel void nu_matvec_iq4_xs(device const uchar * weights [[buffer(0)]], device const float * input [[buffer(1)]], device float * output [[buffer(2)]],
                            constant MatvecBlockParams & p [[buffer(7)]], uint group_index [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    float acc[ROWS];
    nu_matvec_iq4_xs_body<ROWS>(weights, input, p, group_index, sg, lane, acc);
    if ((group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS < p.rows)
        nu_store_rows<ROWS>(acc, output, (group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS, p.rows, lane);
}


// Q4_0. Lane slice: whole 18-byte blocks of 32 values, block `lane + 32·i`
// (the row is walked in blocks, not 256-value strides, so any block count
// serves — the expert down projection has 22): its sixteen nibble bytes
// (low nibbles values 0-15, high nibbles 16-31) as four words from two
// packed_ushort4 loads, the code itself as the value. The bias of eight
// folds into the input sum like Q6_K's 32: Σ d·(q−8)·x = d·(Σq·x − 8·Σx),
// which with a one-hot input is the CPU decoder's `d * (q - 8)` exactly.
template <uint ROWS>
inline void nu_matvec_q4_0_body(device const uchar * weights, device const float * input, MatvecBlockParams p, uint group_index, uint sg, uint lane, thread float * acc) {
    uint row0 = (group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS;
    if (row0 >= p.rows) return;
    for (uint r = 0; r < ROWS; ++r) acc[r] = 0;
    for (uint block = lane; block < p.columns / 32; block += 32) {
        device const float4 * x = (device const float4 *)(input + block * 32);
        float4 x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3], x4 = x[4], x5 = x[5], x6 = x[6], x7 = x[7];
        float sx = nu_sum4(((x0 + x1) + (x2 + x3)) + ((x4 + x5) + (x6 + x7)));
        uint base = block * 18;
        for (uint r = 0; r < ROWS; ++r) {
            device const uchar * b = weights + ulong(min(row0 + r, p.rows - 1)) * p.stride + base;
            float d = float(as_type<half>(*(device const ushort *)b));
            packed_ushort4 l0 = *(device const packed_ushort4 *)(b + 2);
            packed_ushort4 l1 = *(device const packed_ushort4 *)(b + 10);
            uint w0 = nu_word(l0, 0), w1 = nu_word(l0, 1), w2 = nu_word(l1, 0), w3 = nu_word(l1, 1);
            float sum = nu_dot(nu_low_nibbles(w0), x0, nu_dot(nu_low_nibbles(w1), x1, nu_dot(nu_low_nibbles(w2), x2, nu_dot(nu_low_nibbles(w3), x3, 0.0f))));
            sum = nu_dot(nu_high_nibbles(w0), x4, nu_dot(nu_high_nibbles(w1), x5, nu_dot(nu_high_nibbles(w2), x6, nu_dot(nu_high_nibbles(w3), x7, sum))));
            acc[r] += d * fma(-8.0f, sx, sum);
        }
    }
}
template <uint ROWS>
kernel void nu_matvec_q4_0(device const uchar * weights [[buffer(0)]], device const float * input [[buffer(1)]], device float * output [[buffer(2)]],
                           constant MatvecBlockParams & p [[buffer(7)]], uint group_index [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    float acc[ROWS];
    nu_matvec_q4_0_body<ROWS>(weights, input, p, group_index, sg, lane, acc);
    if ((group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS < p.rows)
        nu_store_rows<ROWS>(acc, output, (group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS, p.rows, lane);
}

// ---------------------------------------------------------------------------
// Ternary blocks of 128 (PQ2_0 34 B, PTQ1_0 28 B; docs/reference/bonsai.md):
// w = d * t with t in {-1, 0, +1}, so per lane Σ d·(t−1)·x = d·(Σt·x − Σx)
// with the codes 0..2 (or 0..3 for PQ2_0's unused +2) as t. Four lanes per
// block, eight blocks per iteration (1,024 values), every lane running the
// same code on its quarter: the widths 5,120 / 6,144 / 17,408 divide evenly.
// PQ2_0's quarter is 32 consecutive values (eight bytes, a packed_ushort4 at
// two-byte alignment); PTQ1_0's is four bytes of the 16-byte run (one uint,
// four-byte alignment: 28-byte blocks keep it), two bytes of the 8-byte run,
// and one digit of the two tail bytes — 20 + 10 + 2 values in the strided
// order the digit-major layout gives them.
// Field `k` (bits 2k, 2k+1) of the four bytes of `w`: elements k, 4+k, 8+k,
// 12+k of the sixteen the word holds. One mask serves four values; the
// caller pairs the result with the inputs in the same strided order, which
// is a free re-labelling of registers (the GPU is scalar per lane).
inline float4 nu_two_bit_field(uint w, uint k) { return nu_bytes((w >> (2 * k)) & 0x03030303u); }
// Digit `digit` of four packed base-3 bytes, as consecutive values: the
// fixed-point extraction two bytes at a time in 16-bit slots (byte · 3ⁿ < 2¹⁵
// never crosses a slot), then the trit of each slot from bits 8-9 of q · 3.
inline float4 nu_trits(uint w, uint digit) {
    uint p = nu_pow3[digit];
    uint even = ((w & 0x00ff00ffu) * p) & 0x00ff00ffu;
    uint odd = (((w >> 8) & 0x00ff00ffu) * p) & 0x00ff00ffu;
    return nu_bytes((((even * 3u) & 0x03000300u) >> 8) | ((odd * 3u) & 0x03000300u));
}
template <uint ROWS>
inline void nu_matvec_pq2_0_body(device const uchar * weights, device const float * input, MatvecBlockParams p, uint group_index, uint sg, uint lane, thread float * acc) {
    uint row0 = (group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS;
    if (row0 >= p.rows) return;
    for (uint r = 0; r < ROWS; ++r) acc[r] = 0;
    const uint quarter = lane & 3;
    for (uint block = lane >> 2; block < p.columns / 128; block += 8) {
        device const float4 * x = (device const float4 *)(input + block * 128 + quarter * 32);
        float4 x0 = x[0], x1 = x[1], x2 = x[2], x3 = x[3], x4 = x[4], x5 = x[5], x6 = x[6], x7 = x[7];
        float sx = nu_sum4(((x0 + x1) + (x2 + x3)) + ((x4 + x5) + (x6 + x7)));
        uint base = block * 34;
        for (uint r = 0; r < ROWS; ++r) {
            device const uchar * b = weights + ulong(min(row0 + r, p.rows - 1)) * p.stride + base;
            float d = float(as_type<half>(*(device const ushort *)b));
            packed_ushort4 l = *(device const packed_ushort4 *)(b + 2 + quarter * 8);
            uint w0 = nu_word(l, 0), w1 = nu_word(l, 1);
            // Field k of a word pairs with inputs k, 4+k, 8+k, 12+k: the
            // columns of the four loaded float4.
            float sum = nu_dot(nu_two_bit_field(w0, 0), float4(x0.x, x1.x, x2.x, x3.x), nu_dot(nu_two_bit_field(w0, 1), float4(x0.y, x1.y, x2.y, x3.y), nu_dot(nu_two_bit_field(w0, 2), float4(x0.z, x1.z, x2.z, x3.z), nu_dot(nu_two_bit_field(w0, 3), float4(x0.w, x1.w, x2.w, x3.w), 0.0f))));
            sum = nu_dot(nu_two_bit_field(w1, 0), float4(x4.x, x5.x, x6.x, x7.x), nu_dot(nu_two_bit_field(w1, 1), float4(x4.y, x5.y, x6.y, x7.y), nu_dot(nu_two_bit_field(w1, 2), float4(x4.z, x5.z, x6.z, x7.z), nu_dot(nu_two_bit_field(w1, 3), float4(x4.w, x5.w, x6.w, x7.w), sum))));
            acc[r] += d * fma(-1.0f, sx, sum);
        }
    }
}
template <uint ROWS>
kernel void nu_matvec_pq2_0(device const uchar * weights [[buffer(0)]], device const float * input [[buffer(1)]], device float * output [[buffer(2)]],
                            constant MatvecBlockParams & p [[buffer(7)]], uint group_index [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    float acc[ROWS];
    nu_matvec_pq2_0_body<ROWS>(weights, input, p, group_index, sg, lane, acc);
    if ((group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS < p.rows)
        nu_store_rows<ROWS>(acc, output, (group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS, p.rows, lane);
}
template <uint ROWS>
inline void nu_matvec_ptq1_0_body(device const uchar * weights, device const float * input, MatvecBlockParams p, uint group_index, uint sg, uint lane, thread float * acc) {
    uint row0 = (group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS;
    if (row0 >= p.rows) return;
    for (uint r = 0; r < ROWS; ++r) acc[r] = 0;
    const uint quarter = lane & 3;
    for (uint block = lane >> 2; block < p.columns / 128; block += 8) {
        device const float * x = input + block * 128;
        // Run 1: digit n of bytes 4q..4q+3 are values 16n + 4q .. +3.
        float4 a0 = *(device const float4 *)(x + quarter * 4), a1 = *(device const float4 *)(x + 16 + quarter * 4), a2 = *(device const float4 *)(x + 32 + quarter * 4), a3 = *(device const float4 *)(x + 48 + quarter * 4), a4 = *(device const float4 *)(x + 64 + quarter * 4);
        // Run 2: digit n of bytes 16 + 2q, 17 + 2q are values 80 + 8n + 2q, +1.
        float2 c0 = *(device const float2 *)(x + 80 + quarter * 2), c1 = *(device const float2 *)(x + 88 + quarter * 2), c2 = *(device const float2 *)(x + 96 + quarter * 2), c3 = *(device const float2 *)(x + 104 + quarter * 2), c4 = *(device const float2 *)(x + 112 + quarter * 2);
        // Tail: digit q of bytes 24, 25 are values 120 + 2q, +1.
        float2 t = *(device const float2 *)(x + 120 + quarter * 2);
        float sx = nu_sum4(((a0 + a1) + (a2 + a3)) + a4) + ((c0.x + c0.y) + (c1.x + c1.y)) + ((c2.x + c2.y) + (c3.x + c3.y)) + ((c4.x + c4.y) + (t.x + t.y));
        uint base = block * 28;
        for (uint r = 0; r < ROWS; ++r) {
            device const uchar * b = weights + ulong(min(row0 + r, p.rows - 1)) * p.stride + base;
            float d = float(as_type<half>(*(device const ushort *)(b + 26)));
            uint w = *(device const uint *)(b + quarter * 4);
            uint v = uint(*(device const ushort *)(b + 16 + quarter * 2));
            uint h = uint(*(device const ushort *)(b + 24));
            float sum = nu_dot(nu_trits(w, 0), a0, nu_dot(nu_trits(w, 1), a1, nu_dot(nu_trits(w, 2), a2, nu_dot(nu_trits(w, 3), a3, nu_dot(nu_trits(w, 4), a4, 0.0f)))));
            for (uint n = 0; n < 5; ++n) {
                float2 tr = nu_trits(v, n).xy;
                float2 xc = n == 0 ? c0 : n == 1 ? c1 : n == 2 ? c2 : n == 3 ? c3 : c4;
                sum = fma(tr.y, xc.y, fma(tr.x, xc.x, sum));
            }
            float2 th = nu_trits(h, quarter).xy;
            sum = fma(th.y, t.y, fma(th.x, t.x, sum));
            acc[r] += d * fma(-1.0f, sx, sum);
        }
    }
}
template <uint ROWS>
kernel void nu_matvec_ptq1_0(device const uchar * weights [[buffer(0)]], device const float * input [[buffer(1)]], device float * output [[buffer(2)]],
                             constant MatvecBlockParams & p [[buffer(7)]], uint group_index [[threadgroup_position_in_grid]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    float acc[ROWS];
    nu_matvec_ptq1_0_body<ROWS>(weights, input, p, group_index, sg, lane, acc);
    if ((group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS < p.rows)
        nu_store_rows<ROWS>(acc, output, (group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS, p.rows, lane);
}

// Q3_K and IQ3_S both have 110-byte blocks: never assume more than
// two-byte alignment. Each lane owns one consecutive group of 32 values;
// its input is reused across ROWS output rows.
inline float nu_q3_scale(device const uchar * s, uint group) {
    uint nibble = (s[group & 7] >> (group < 8 ? 0 : 4)) & 15u;
    uint upper = (s[8 + group % 4] >> (2 * (group / 4))) & 3u;
    return float(int(nibble | (upper << 4)) - 32);
}
template <uint ROWS, bool IQ>
inline void nu_matvec_three_body(device const uchar * weights, device const float * input, MatvecBlockParams p, uint group_index, uint sg, uint lane, thread float * acc) {
    uint row0 = (group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS;
    if (row0 >= p.rows) return;
    uint g = lane & 7;
    for (uint r = 0; r < ROWS; ++r) acc[r] = 0;
    for (uint kb = lane >> 3; kb < p.blocks; kb += 4) {
        NuInputs16 xa = nu_inputs16(input + kb * 256 + g * 32);
        NuInputs16 xb = nu_inputs16(input + kb * 256 + g * 32 + 16);
        for (uint r = 0; r < ROWS; ++r) {
            device const uchar * b = weights + ulong(min(row0 + r, p.rows - 1)) * p.stride + kb * 110;
            float sa = 0, sb = 0;
            if (IQ) {
                packed_ushort4 indices = *(device const packed_ushort4 *)(b + 2 + g * 8);
                uint high = b[66 + g];
                uint signs = uint(*(device const ushort *)(b + 74 + g * 4)) | (uint(*(device const ushort *)(b + 76 + g * 4)) << 16);
                for (uint j = 0; j < 8; ++j) {
                    uint index = nu_byte(nu_word(indices, j / 4), j % 4) | (((high >> j) & 1u) << 8);
                    float4 magnitude = nu_bytes(nu_iq3_grid[index]);
                    uint bits = signs >> (4 * j);
                    float4 value = magnitude * float4((bits & 1) ? -1 : 1, (bits & 2) ? -1 : 1, (bits & 4) ? -1 : 1, (bits & 8) ? -1 : 1);
                    if (j < 4) sa = nu_dot(value, xa.v[j], sa);
                    else sb = nu_dot(value, xb.v[j - 4], sb);
                }
                float scale = float(1 + 2 * ((b[106 + g / 2] >> (4 * (g % 2))) & 15));
                acc[r] += (nu_half(b) * scale) * (sa + sb);
            } else {
                for (uint j = 0; j < 4; ++j) {
                    packed_ushort4 q = *(device const packed_ushort4 *)(b + 32 + (g / 4) * 32 + j * 8);
                    packed_ushort4 h = *(device const packed_ushort4 *)(b + j * 8);
                    for (uint k = 0; k < 2; ++k) {
                        uint codes = ((nu_word(q, k) >> (2 * (g % 4))) & 0x03030303u) | (((nu_word(h, k) >> g) & 0x01010101u) << 2);
                        uint v = j * 2 + k;
                        if (v < 4) sa = nu_dot(nu_bytes(codes), xa.v[v], sa);
                        else sb = nu_dot(nu_bytes(codes), xb.v[v - 4], sb);
                    }
                }
                float d = nu_half(b + 108);
                acc[r] += (d * nu_q3_scale(b + 96, 2 * g)) * fma(-4.0f, xa.sum, sa)
                        + (d * nu_q3_scale(b + 96, 2 * g + 1)) * fma(-4.0f, xb.sum, sb);
            }
        }
    }
}
template <uint ROWS, bool IQ>
kernel void nu_matvec_three(device const uchar * weights [[buffer(0)]], device const float * input [[buffer(1)]], device float * output [[buffer(2)]],
                            constant MatvecBlockParams & p [[buffer(7)]], uint group_index [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    float acc[ROWS];
    nu_matvec_three_body<ROWS, IQ>(weights, input, p, group_index, sg, lane, acc);
    if ((group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS < p.rows)
        nu_store_rows<ROWS>(acc, output, (group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS, p.rows, lane);
}


// A segment owns whole 16-row threadgroups, so encoding and binding switches
// are uniform within a SIMD group. Bit 31 requests generic arithmetic when
// alignment prevents specialized vector loads (or for a generic-only run).
struct MatvecSegment { uint rows, weight_slot, weight_offset, encoding, stride, output_slot, output_offset; };
struct MatvecSegments { uint columns, blocks, count, mode; MatvecSegment segments[4]; };
inline device uchar * nu_segment_buffer(uint slot, device uchar * b1, device uchar * b2, device uchar * b3, device uchar * b4, device uchar * b5, device uchar * b6) {
    switch (slot) { case 1: return b1; case 2: return b2; case 3: return b3; case 4: return b4; case 5: return b5; default: return b6; }
}
inline void nu_segment_sums(device const uchar * w, device const float * x, MatvecBlockParams p, uint encoding, uint group, uint sg, uint lane, thread float * acc) {
    // Bodies produce the same lane partials as their standalone kernels.
    // The generic branch retains one lane's original 16-value segment order.
    switch (encoding) {
        case 2: nu_matvec_q4_0_body<4>(w, x, p, group, sg, lane, acc); return;
        case 11: nu_matvec_three_body<4, false>(w, x, p, group, sg, lane, acc); return;
        case 12: nu_matvec_k_body<4, false>(w, x, p, group, sg, lane, acc); return;
        case 13: nu_matvec_k_body<4, true>(w, x, p, group, sg, lane, acc); return;
        case 14: nu_matvec_q6_k_body<4>(w, x, p, group, sg, lane, acc); return;
        case 21: nu_matvec_three_body<4, true>(w, x, p, group, sg, lane, acc); return;
        case 23: nu_matvec_iq4_xs_body<4>(w, x, p, group, sg, lane, acc); return;
        case 142: nu_matvec_pq2_0_body<4>(w, x, p, group, sg, lane, acc); return;
        case 143: nu_matvec_ptq1_0_body<4>(w, x, p, group, sg, lane, acc); return;
        default:
            uint row0 = (group * NU_MATVEC_SIMDGROUPS + sg) * 4;
            for (uint r = 0; r < 4; ++r) {
                float sum = 0, values[16];
                for (uint segment = lane; segment < p.columns / 16; segment += 32) {
                    nu_segment(w + ulong(min(row0 + r, p.rows - 1)) * p.stride, encoding & 0x7fffffffu, segment, values);
                    for (uint j = 0; j < 16; ++j) sum += values[j] * x[segment * 16 + j];
                }
                acc[r] = sum;
            }
    }
}
kernel void nu_matvec_segments(device const float * input [[buffer(0)]],
    device uchar * b1 [[buffer(1)]], device uchar * b2 [[buffer(2)]], device uchar * b3 [[buffer(3)]],
    device uchar * b4 [[buffer(4)]], device uchar * b5 [[buffer(5)]], device uchar * b6 [[buffer(6)]],
    constant MatvecSegments & p [[buffer(7)]], uint group [[threadgroup_position_in_grid]],
    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint index = 0, local = group;
    if (p.mode == 0) {
        while (index + 1 < p.count && local >= p.segments[index].rows / 16) {
            local -= p.segments[index].rows / 16;
            ++index;
        }
    }
    // Pair modes (1: silu, 2: gelu) give two SIMD groups to gate and two to
    // up. Each group computes eight output rows, retaining only one
    // projection's lane partials. The virtual group/sg pair preserves each
    // body's original row arithmetic.
    threadgroup float pair_values[16];
    const bool pair = p.mode != 0;
    if (pair) { index = sg / 2; local = group / 2; }
    uint body_sg = pair ? (group % 2) * 2 + sg % 2 : sg;
    MatvecSegment s = p.segments[index];
    MatvecBlockParams shape = { p.columns, s.stride, s.rows, p.blocks };
    device const uchar * w = nu_segment_buffer(s.weight_slot, b1,b2,b3,b4,b5,b6) + s.weight_offset;
    device float * out = (device float *)(nu_segment_buffer(s.output_slot, b1,b2,b3,b4,b5,b6) + s.output_offset);
    float a[4];
    nu_segment_sums(w, input, shape, s.encoding, local, body_sg, lane, a);
    uint row0 = (local * NU_MATVEC_SIMDGROUPS + sg) * 4;
    if (pair) {
        for (uint r = 0; r < 4; ++r) {
            float value = simd_sum(a[r]);
            if (lane == 0) pair_values[(sg / 2) * 8 + (sg % 2) * 4 + r] = value;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0 && sg < 2) for (uint r = 0; r < 4; ++r) {
            uint i = sg * 4 + r;
            const float gate = p.mode == 1 ? nu_silu(pair_values[i]) : nu_gelu(pair_values[i]);
            out[group * 8 + i] = gate * pair_values[8 + i];
        }
    } else nu_store_rows<4>(a, out, row0, s.rows, lane);
}

// Host-visible instantiations. ROWS must match `rows_per_simdgroup` in root.zig.
#define NU_MATVEC_ARGS device const uchar *, device const float *, device float *, constant MatvecBlockParams &, uint, uint, uint
template [[host_name("nu_matvec_q3_k")]] kernel void nu_matvec_three<4, false>(NU_MATVEC_ARGS);
template [[host_name("nu_matvec_iq3_s")]] kernel void nu_matvec_three<4, true>(NU_MATVEC_ARGS);
template [[host_name("nu_matvec_q4_k")]] kernel void nu_matvec_q4_k<4>(NU_MATVEC_ARGS);
template [[host_name("nu_matvec_q5_k")]] kernel void nu_matvec_q5_k<4>(NU_MATVEC_ARGS);
template [[host_name("nu_matvec_q6_k")]] kernel void nu_matvec_q6_k<4>(NU_MATVEC_ARGS);
template [[host_name("nu_matvec_iq4_xs")]] kernel void nu_matvec_iq4_xs<4>(NU_MATVEC_ARGS);
template [[host_name("nu_matvec_q4_0")]] kernel void nu_matvec_q4_0<4>(NU_MATVEC_ARGS);
template [[host_name("nu_matvec_pq2_0")]] kernel void nu_matvec_pq2_0<4>(NU_MATVEC_ARGS);
template [[host_name("nu_matvec_ptq1_0")]] kernel void nu_matvec_ptq1_0<4>(NU_MATVEC_ARGS);

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
    float a[4];
    nu_segment_sums(w, x, shape, p.encoding, local, sg, lane, a);
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

// ---------------------------------------------------------------------------
// Batched matrix product for prefill: out[t][r] = Σ_k W[r][k] · X[t][k]
// over a chunk of tokens. A 128-thread group (four SIMD groups) owns a 32-row
// × 32-token output tile; SIMD group sg owns the 16×16 quarter at rows
// (sg&1)·16 and tokens (sg>>1)·16 as four 8×8 accumulators. The K loop takes
// 64 columns per step: each thread decodes one 16-value segment of one row
// into a threadgroup tile [32 rows][64 k] (8 KB), then every SIMD group
// multiplies that tile by the activation tile loaded straight from device
// memory (transposed, since activations are [token][k]). Decode reads weights
// once per 32-token tile instead of once per token — that is the whole point
// of prefill in chunks. Rows are clamped for the decode and 8×8 blocks past
// `rows` are not stored; token rows past `tokens` are computed and stored on
// the caller's padding (the encoder requires the buffers to hold a multiple
// of 32 token rows).
//
// The kernel is a template on the encoding, the tile shape, and the
// operand types. NU_TILE_GENERIC decodes through the runtime switch of
// `nu_segment` (any encoding, any alignment) and keeps F32 operands; the
// specialized instantiations decode a segment with the vector loads and
// packed-byte helpers of the specialized matvecs and hold both operands as
// half (F32 accumulation). Each specialized decoder evaluates the generic
// decoder's expression for every value in the same F32 operation order
// (`multiplier * q - offset`, `scale * (q - 32)`, ...), so before the half
// rounding the two tiles are bit-identical; `test-metal` bounds the half
// tile against the generic F32 tile per encoding. The Zig encoder selects
// an instantiation under the matvec alignment rules and by chunk length
// (`specializedMatmul`, `matmulGeometry`) and otherwise records the generic one.
struct MatmulParams { uint columns; uint encoding; uint stride; uint rows; uint tokens; uint in_stride; uint out_stride; uint row_tiles; };
#define NU_TILE_GENERIC 0xffffffffu

// Q4_K / Q5_K segment: group `first/32` of the block, sixteen bytes whose low
// (even group) or high (odd group) nibbles are the codes; Q5_K adds bit
// `group` of the plane bytes at the same columns.
template <bool FIFTH_BIT>
inline void nu_tile_k(device const uchar * b, uint first, thread float4 * q) {
    const uint nibble_offset = FIFTH_BIT ? 48 : 16;
    uint group = first / 32, half_index = (first % 32) / 16;
    uint dd = *(device const uint *)b;
    packed_uint3 s = *(device const packed_uint3 *)(b + 4);
    float sa, ma, sb, mb;
    nu_k_scales_pair(s, group >> 1, sa, ma, sb, mb);
    float multiplier = nu_half_low(dd) * ((group & 1) ? sb : sa);
    float offset = nu_half_high(dd) * ((group & 1) ? mb : ma);
    uint4 v = *(device const uint4 *)(b + nibble_offset + (group / 2) * 32 + half_index * 16);
    if (FIFTH_BIT) {
        uint4 h = *(device const uint4 *)(b + 16 + half_index * 16);
        if (group & 1) { q[0] = nu_high_fives(v.x, h.x, group); q[1] = nu_high_fives(v.y, h.y, group); q[2] = nu_high_fives(v.z, h.z, group); q[3] = nu_high_fives(v.w, h.w, group); }
        else { q[0] = nu_low_fives(v.x, h.x, group); q[1] = nu_low_fives(v.y, h.y, group); q[2] = nu_low_fives(v.z, h.z, group); q[3] = nu_low_fives(v.w, h.w, group); }
    } else {
        if (group & 1) { q[0] = nu_high_nibbles(v.x); q[1] = nu_high_nibbles(v.y); q[2] = nu_high_nibbles(v.z); q[3] = nu_high_nibbles(v.w); }
        else { q[0] = nu_low_nibbles(v.x); q[1] = nu_low_nibbles(v.y); q[2] = nu_low_nibbles(v.z); q[3] = nu_low_nibbles(v.w); }
    }
    for (uint i = 0; i < 4; ++i) q[i] = multiplier * q[i] - offset;
}
// Q6_K segment: half `first/128`, quarter `(first%128)/32`, columns
// `first%32 ..+16`: sixteen nibble bytes and sixteen high-bit bytes as four
// packed_ushort4 (blocks are only 2-byte aligned).
inline void nu_tile_q6_k(device const uchar * b, uint first, thread float4 * q) {
    uint h = first / 128, quarter = (first % 128) / 32, column = first % 32;
    device const uchar * lo = b + h * 64 + (quarter % 2) * 32 + column;
    device const uchar * hi = b + 128 + h * 32 + column;
    packed_ushort4 l0 = *(device const packed_ushort4 *)lo, l1 = *(device const packed_ushort4 *)(lo + 8);
    packed_ushort4 h0 = *(device const packed_ushort4 *)hi, h1 = *(device const packed_ushort4 *)(hi + 8);
    float scale = nu_half(b + 208) * float(char(b[192 + first / 16]));
    uint shift = quarter * 2;
    if (quarter < 2) {
        q[0] = nu_low_sixes(nu_word(l0, 0), nu_word(h0, 0), shift); q[1] = nu_low_sixes(nu_word(l0, 1), nu_word(h0, 1), shift);
        q[2] = nu_low_sixes(nu_word(l1, 0), nu_word(h1, 0), shift); q[3] = nu_low_sixes(nu_word(l1, 1), nu_word(h1, 1), shift);
    } else {
        q[0] = nu_high_sixes(nu_word(l0, 0), nu_word(h0, 0), shift); q[1] = nu_high_sixes(nu_word(l0, 1), nu_word(h0, 1), shift);
        q[2] = nu_high_sixes(nu_word(l1, 0), nu_word(h1, 0), shift); q[3] = nu_high_sixes(nu_word(l1, 1), nu_word(h1, 1), shift);
    }
    for (uint i = 0; i < 4; ++i) q[i] = scale * (q[i] - 32.0f);
}
// Q3_K segment: two-bit codes of sixteen consecutive bytes plus the inverted
// sign bit `first/32` of the mask bytes at the same columns; value = code + 4·mask − 4.
inline void nu_tile_q3_k(device const uchar * b, uint first, thread float4 * q) {
    uint g = first / 32, column = first % 32, shift = 2 * ((first % 128) / 32);
    device const uchar * codes = b + 32 + (first / 128) * 32 + column;
    device const uchar * mask = b + column;
    packed_ushort4 c0 = *(device const packed_ushort4 *)codes, c1 = *(device const packed_ushort4 *)(codes + 8);
    packed_ushort4 m0 = *(device const packed_ushort4 *)mask, m1 = *(device const packed_ushort4 *)(mask + 8);
    float multiplier = nu_half(b + 108) * nu_q3_scale(b + 96, first / 16);
    q[0] = nu_bytes(((nu_word(c0, 0) >> shift) & 0x03030303u) | (((nu_word(m0, 0) >> g) & 0x01010101u) << 2));
    q[1] = nu_bytes(((nu_word(c0, 1) >> shift) & 0x03030303u) | (((nu_word(m0, 1) >> g) & 0x01010101u) << 2));
    q[2] = nu_bytes(((nu_word(c1, 0) >> shift) & 0x03030303u) | (((nu_word(m1, 0) >> g) & 0x01010101u) << 2));
    q[3] = nu_bytes(((nu_word(c1, 1) >> shift) & 0x03030303u) | (((nu_word(m1, 1) >> g) & 0x01010101u) << 2));
    for (uint i = 0; i < 4; ++i) q[i] = multiplier * (q[i] - 4.0f);
}
// IQ3_S segment: four grid entries (four magnitudes each) of group `first/32`
// and sixteen sign bits.
inline void nu_tile_iq3_s(device const uchar * b, uint first, thread float4 * q) {
    uint group = first / 32, entry0 = (first % 32) / 4;
    uint high = b[66 + group];
    uint signs = uint(b[74 + first / 8]) | (uint(b[75 + first / 8]) << 8);
    float multiplier = nu_half(b) * float(1 + 2 * ((b[106 + group / 2] >> (4 * (group % 2))) & 15));
    for (uint e = 0; e < 4; ++e) {
        uint entry = entry0 + e;
        uint index = uint(b[2 + group * 8 + entry]) | (((high >> entry) & 1u) << 8);
        float4 value = multiplier * nu_bytes(nu_iq3_grid[index]);
        uint bits = signs >> (4 * e);
        q[e] = select(value, -value, bool4(bits & 1, bits & 2, bits & 4, bits & 8));
    }
}
// IQ4_XS segment: the low (first%32 == 0) or high nibbles of the sixteen bytes
// of group `first/32` through the nonlinear table; 8-byte aligned blocks.
inline void nu_tile_iq4_xs(device const uchar * b, uint first, thread float4 * q) {
    uint group = first / 32;
    uint2 header = *(device const uint2 *)b;
    uint low = (header.y >> (4 * group)) & 15u, high = (header.x >> (16 + 2 * group)) & 3u;
    float multiplier = nu_half_low(header.x) * float(int(low | (high << 4)) - 32);
    uint2 qa = *(device const uint2 *)(b + 8 + group * 16), qb = *(device const uint2 *)(b + 16 + group * 16);
    if (first % 32) { q[0] = nu_iq4_high(qa.x); q[1] = nu_iq4_high(qa.y); q[2] = nu_iq4_high(qb.x); q[3] = nu_iq4_high(qb.y); }
    else { q[0] = nu_iq4_low(qa.x); q[1] = nu_iq4_low(qa.y); q[2] = nu_iq4_low(qb.x); q[3] = nu_iq4_low(qb.y); }
    for (uint i = 0; i < 4; ++i) q[i] = multiplier * q[i];
}
// Q4_0 segment: the low (first == 0) or high nibbles of the block's
// sixteen bytes, `d * (q - 8)` in the generic decoder's operation order;
// 2-byte aligned blocks, so two packed_ushort4 loads.
inline void nu_tile_q4_0(device const uchar * b, uint first, thread float4 * q) {
    float d = float(as_type<half>(*(device const ushort *)b));
    packed_ushort4 l0 = *(device const packed_ushort4 *)(b + 2), l1 = *(device const packed_ushort4 *)(b + 10);
    uint w0 = nu_word(l0, 0), w1 = nu_word(l0, 1), w2 = nu_word(l1, 0), w3 = nu_word(l1, 1);
    if (first) { q[0] = nu_high_nibbles(w0); q[1] = nu_high_nibbles(w1); q[2] = nu_high_nibbles(w2); q[3] = nu_high_nibbles(w3); }
    else { q[0] = nu_low_nibbles(w0); q[1] = nu_low_nibbles(w1); q[2] = nu_low_nibbles(w2); q[3] = nu_low_nibbles(w3); }
    for (uint i = 0; i < 4; ++i) q[i] = d * (q[i] - 8.0f);
}
// PQ2_0 segment: four bytes at 2 + first/4 (two-byte aligned), each four
// consecutive codes; PTQ1_0 segment: digit first/16 of the 16-byte run for
// the first five segments, then the 8-byte run's digit pairs (0,1), (2,3),
// and (4, tail). The expressions are the generic decoders' in the same order.
inline void nu_tile_pq2_0(device const uchar * b, uint first, thread float4 * q) {
    float d = float(as_type<half>(*(device const ushort *)b));
    packed_ushort2 l = *(device const packed_ushort2 *)(b + 2 + first / 4);
    uint w = uint(l.x) | (uint(l.y) << 16);
    // Field k holds elements k, 4+k, 8+k, 12+k; transpose back to element order.
    float4 c0 = nu_two_bit_field(w, 0), c1 = nu_two_bit_field(w, 1), c2 = nu_two_bit_field(w, 2), c3 = nu_two_bit_field(w, 3);
    q[0] = float4(c0.x, c1.x, c2.x, c3.x); q[1] = float4(c0.y, c1.y, c2.y, c3.y); q[2] = float4(c0.z, c1.z, c2.z, c3.z); q[3] = float4(c0.w, c1.w, c2.w, c3.w);
    for (uint i = 0; i < 4; ++i) q[i] = d * (q[i] - 1.0f);
}
inline void nu_tile_ptq1_0(device const uchar * b, uint first, thread float4 * q) {
    float d = float(as_type<half>(*(device const ushort *)(b + 26)));
    if (first < 80) {
        uint digit = first / 16;
        for (uint i = 0; i < 4; ++i) q[i] = nu_trits(*(device const uint *)(b + 4 * i), digit);
    } else {
        uint w0 = *(device const uint *)(b + 16), w1 = *(device const uint *)(b + 20);
        uint digit = (first - 80) / 8; // 0, 2, or 4
        q[0] = nu_trits(w0, digit); q[1] = nu_trits(w1, digit);
        if (digit < 4) { q[2] = nu_trits(w0, digit + 1); q[3] = nu_trits(w1, digit + 1); }
        else {
            uint h = uint(*(device const ushort *)(b + 24));
            float4 h0 = nu_trits(h, 0), h1 = nu_trits(h, 1), h2 = nu_trits(h, 2), h3 = nu_trits(h, 3);
            q[2] = float4(h0.x, h0.y, h1.x, h1.y); q[3] = float4(h2.x, h2.y, h3.x, h3.y);
        }
    }
    for (uint i = 0; i < 4; ++i) q[i] = d * (q[i] - 1.0f);
}
// Sixteen decoded values of `segment` of an encoded row, as four float4.
template <uint ENC>
inline void nu_tile_segment(device const uchar * row, uint encoding, uint segment, thread float4 * q) {
    if (ENC == NU_TILE_GENERIC) {
        float values[16];
        nu_segment(row, encoding, segment, values);
        for (uint i = 0; i < 4; ++i) q[i] = float4(values[4 * i], values[4 * i + 1], values[4 * i + 2], values[4 * i + 3]);
        return;
    }
    // 256-value blocks hold sixteen segments; Q4_0's 32-value block two; the
    // ternary 128-value blocks eight.
    const uint block_bytes = ENC == 2 ? 18 : ENC == 12 ? 144 : ENC == 13 ? 176 : ENC == 14 ? 210 : ENC == 23 ? 136 : ENC == 142 ? 34 : ENC == 143 ? 28 : 110;
    const uint segments_per_block = ENC == 2 ? 2 : (ENC == 142 || ENC == 143) ? 8 : 16;
    device const uchar * b = row + ulong(segment / segments_per_block) * block_bytes;
    uint first = (segment % segments_per_block) * 16;
    switch (ENC) {
        case 2: nu_tile_q4_0(b, first, q); break;
        case 11: nu_tile_q3_k(b, first, q); break;
        case 12: nu_tile_k<false>(b, first, q); break;
        case 13: nu_tile_k<true>(b, first, q); break;
        case 14: nu_tile_q6_k(b, first, q); break;
        case 21: nu_tile_iq3_s(b, first, q); break;
        case 142: nu_tile_pq2_0(b, first, q); break;
        case 143: nu_tile_ptq1_0(b, first, q); break;
        default: nu_tile_iq4_xs(b, first, q); break;
    }
}

// TR × TT output tile (rows × tokens) with operands of type TW (decoded
// weights) and TX (activations) in threadgroup memory and F32 accumulators.
// Each of the four SIMD groups owns a (TR/2) × (TT/2) quarter as (TR/16) ×
// (TT/16) `simdgroup_float8x8`. Per 64-column K step: decode covers TR rows
// × four segments (TR/32 segments per thread), staging copies TT token rows
// × 64 columns transposed to [k][token] (so the B loads are plain loads;
// TT/8 float4 per thread), then every SIMD group multiplies. Measured on
// the M4 Pro: threadgroup memory is the occupancy limit — 16 KB per
// group runs at full speed, 24 KB and 32 KB collapse five-fold — so the
// half instantiations use 64×64 (8 KB + 8 KB) and the F32 generic
// instantiation 32×32 (8 KB + 8 KB). Prefetching the next step's segments
// before the MACs measured slower (register pressure) and is not done.
// STAGE: stage the activation tile in threadgroup memory as TX (the large
// tiles); otherwise load B blocks straight from device memory as F32
// (transposed), which keeps the small tile at 4 KB of threadgroup memory
// so more groups fit per core — what a one-token-tile chunk needs.
// GATHER (the expert tiles, one 32-row token tile): the tile's activation
// rows are the `valid` slot rows `row_list[j]`, read from input row
// `row_list[j] / in_group` (the gate-up projection shares a token's input
// across its k slots; the down projection reads one hidden row per slot),
// rows past `valid` stage zeros, and the result goes through threadgroup
// memory (reusing `tile`, which holds TT × TR floats exactly) to scatter
// each valid row to output row `row_list[j]`; no padding rows are written.
template <uint ENC, uint TR, uint TT, typename TW, typename TX, bool STAGE, bool GATHER>
inline void nu_matmul_body(device const uchar * weights, device const float * input, device float * output, MatmulParams p,
                           uint row0, uint token0, device const uint * row_list, uint valid, uint in_group,
                           threadgroup TW * tile, threadgroup TX * xt, uint tid, uint sg) {
    const uint RI = TR / 16, RJ = TT / 16, NR = TR / 32, NT = TT / 32;
    static_assert(!GATHER || (STAGE && TT == 32 && TT * TR * sizeof(float) <= TR * 64 * sizeof(TW)), "gathered tiles stage one 32-row token tile and scatter through `tile`");
    const uint sub_row = (sg & 1) * (TR / 2), sub_token = (sg >> 1) * (TT / 2);
    simdgroup_float8x8 acc[RI][RJ];
    for (uint i = 0; i < RI; ++i) for (uint j = 0; j < RJ; ++j) acc[i][j] = simdgroup_float8x8(0.0f);
    const uint r = tid >> 2, seg = tid & 3;
    device const uchar * rows[NR];
    for (uint i = 0; i < NR; ++i) rows[i] = weights + ulong(min(row0 + r + 32 * i, p.rows - 1)) * p.stride;
    // The activation row this thread stages for tile row `r` (NT == 1 when gathering).
    const bool gathered = GATHER && r < valid;
    const uint source = GATHER ? (gathered ? row_list[r] / in_group : 0u) : token0 + r;
    for (uint k0 = 0; k0 < p.columns; k0 += 64) {
        for (uint i = 0; i < NR; ++i) {
            float4 q[4];
            nu_tile_segment<ENC>(rows[i], p.encoding, k0 / 16 + seg, q);
            threadgroup vec<TW, 4> * slot = (threadgroup vec<TW, 4> *)(tile + (r + 32 * i) * 64 + seg * 16);
            for (uint c = 0; c < 4; ++c) slot[c] = vec<TW, 4>(q[c]);
        }
        if (STAGE) for (uint t = 0; t < NT; ++t) {
            device const float4 * xrow = (device const float4 *)(input + ulong(source + 32 * t) * p.in_stride + k0 + seg * 16);
            for (uint c = 0; c < 4; ++c) {
                float4 x = (GATHER && !gathered) ? float4(0.0f) : xrow[c];
                threadgroup TX * column = xt + (seg * 16 + 4 * c) * TT + r + 32 * t;
                column[0] = TX(x.x); column[TT] = TX(x.y); column[2 * TT] = TX(x.z); column[3 * TT] = TX(x.w);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k8 = 0; k8 < 64; k8 += 8) {
            simdgroup_matrix<TW, 8, 8> a[RI];
            for (uint i = 0; i < RI; ++i) simdgroup_load(a[i], tile + (sub_row + 8 * i) * 64 + k8, 64);
            if (STAGE) {
                simdgroup_matrix<TX, 8, 8> b[RJ];
                for (uint j = 0; j < RJ; ++j) simdgroup_load(b[j], xt + k8 * TT + sub_token + 8 * j, TT);
                for (uint i = 0; i < RI; ++i) for (uint j = 0; j < RJ; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
            } else {
                simdgroup_float8x8 b[RJ];
                for (uint j = 0; j < RJ; ++j) simdgroup_load(b[j], input + ulong(token0 + sub_token + 8 * j) * p.in_stride + k0 + k8, p.in_stride, ulong2(0, 0), true);
                for (uint i = 0; i < RI; ++i) for (uint j = 0; j < RJ; ++j) simdgroup_multiply_accumulate(acc[i][j], a[i], b[j], acc[i][j]);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (GATHER) {
        threadgroup float * staged = (threadgroup float *)tile; // [token][row], free after the last barrier
        for (uint i = 0; i < RI; ++i) for (uint j = 0; j < RJ; ++j)
            simdgroup_store(acc[i][j], staged + (sub_token + 8 * j) * TR + sub_row + 8 * i, TR, ulong2(0, 0), true);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Thread: tile row tid / 4, a quarter of its TR outputs.
        const uint j = tid >> 2, c0 = (tid & 3) * (TR / 4);
        if (j < valid) {
            device float * out = output + ulong(row_list[j]) * p.out_stride + row0;
            for (uint c = c0; c < c0 + TR / 4; ++c) if (row0 + c < p.rows) out[c] = staged[j * TR + c];
        }
        return;
    }
    for (uint i = 0; i < RI; ++i) {
        if (row0 + sub_row + 8 * i + 8 > p.rows) continue;
        for (uint j = 0; j < RJ; ++j)
            simdgroup_store(acc[i][j], output + ulong(token0 + sub_token + 8 * j) * p.out_stride + row0 + sub_row + 8 * i, p.out_stride, ulong2(0, 0), true);
    }
}
template <uint ENC, uint TR, uint TT, typename TW, typename TX, bool STAGE>
kernel void nu_matmul_t(device const uchar * weights [[buffer(0)]],
                        device const float * input [[buffer(1)]],
                        device float * output [[buffer(2)]],
                        constant MatmulParams & p [[buffer(7)]],
                        uint group [[threadgroup_position_in_grid]],
                        uint tid [[thread_position_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup TW tile[TR * 64];
    threadgroup TX xt[STAGE ? 64 * TT : 1];
    const uint row0 = (group % p.row_tiles) * TR, token0 = (group / p.row_tiles) * TT;
    nu_matmul_body<ENC, TR, TT, TW, TX, STAGE, false>(weights, input, output, p, row0, token0, nullptr, 0, 1, tile, xt, tid, sg);
}
// Small-chunk tile: one 128-thread group owns 16 rows × 8 tokens of output.
// Two 8x8 accumulators per SIMD group share one activation block per step, so
// one B load serves two weight loads and the gathered activation traffic per
// weight byte halves. The four SIMD groups partition the 64-column K steps,
// each lane decodes two 16-value segments into its own group's 16×64 half
// tile, and only `simdgroup_barrier` orders the loop; the four K partials are
// summed once at the end in a fixed SIMD-group order.
template <uint ENC, typename TW>
inline void nu_matmul_split_body(device const uchar * weights, device const float * input, device float * output, MatmulParams p,
                                   uint row0, uint token0, threadgroup TW * tile, uint tid, uint sg) {
    const uint lane = tid & 31;
    const uint row = lane & 15, seg = lane >> 4; // two segments: `seg` and `seg + 2`
    threadgroup TW * own = tile + sg * 16 * 64;
    device const uchar * wrow = weights + ulong(min(row0 + row, p.rows - 1)) * p.stride;
    simdgroup_float8x8 acc0 = simdgroup_float8x8(0.0f), acc1 = simdgroup_float8x8(0.0f);
    const uint steps = p.columns / 64;
    for (uint step = sg; step < steps; step += 4) {
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const uint k0 = step * 64;
        float4 q[4];
        nu_tile_segment<ENC>(wrow, p.encoding, k0 / 16 + seg, q);
        threadgroup vec<TW, 4> * slot = (threadgroup vec<TW, 4> *)(own + row * 64 + seg * 16);
        for (uint c = 0; c < 4; ++c) slot[c] = vec<TW, 4>(q[c]);
        nu_tile_segment<ENC>(wrow, p.encoding, k0 / 16 + seg + 2, q);
        slot = (threadgroup vec<TW, 4> *)(own + row * 64 + (seg + 2) * 16);
        for (uint c = 0; c < 4; ++c) slot[c] = vec<TW, 4>(q[c]);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k8 = 0; k8 < 64; k8 += 8) {
            simdgroup_matrix<TW, 8, 8> a0, a1;
            simdgroup_load(a0, own + k8, 64);
            simdgroup_load(a1, own + 8 * 64 + k8, 64);
            simdgroup_float8x8 b;
            simdgroup_load(b, input + ulong(token0) * p.in_stride + k0 + k8, p.in_stride, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc0, a0, b, acc0);
            simdgroup_multiply_accumulate(acc1, a1, b, acc1);
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_store(acc0, (threadgroup float *)own, 8, ulong2(0, 0), false);
    simdgroup_store(acc1, (threadgroup float *)own + 64, 8, ulong2(0, 0), false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * part0 = (threadgroup float *)tile;
    threadgroup float * part1 = (threadgroup float *)(tile + 16 * 64);
    threadgroup float * part2 = (threadgroup float *)(tile + 2 * 16 * 64);
    threadgroup float * part3 = (threadgroup float *)(tile + 3 * 16 * 64);
    const uint i = tid & 63, hi = tid >> 6; // tid < 64 reduces acc0, else acc1
    const uint r = 8 * hi + (i >> 3), t = i & 7;
    const uint at = i + 64 * hi;
    const float total = ((part0[at] + part1[at]) + part2[at]) + part3[at];
    if (row0 + r < p.rows) output[ulong(token0 + t) * p.out_stride + row0 + r] = total;
}
template <uint ENC, typename TW>
kernel void nu_matmul_split_t(device const uchar * weights [[buffer(0)]],
                                device const float * input [[buffer(1)]],
                                device float * output [[buffer(2)]],
                                constant MatmulParams & p [[buffer(7)]],
                                uint group [[threadgroup_position_in_grid]],
                                uint tid [[thread_position_in_threadgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup TW tile[4 * 16 * 64];
    const uint row0 = (group % p.row_tiles) * 16, token0 = (group / p.row_tiles) * 8;
    nu_matmul_split_body<ENC, TW>(weights, input, output, p, row0, token0, tile, tid, sg);
}
// Gathered expert matmul over the row lists of `nu_expert_lists`: threadgroup
// `group` serves tile `group / row_tiles` of the list (exiting past the tile
// count) and row tile `group % row_tiles` of that tile's expert, whose bytes
// start at expert · rows · stride.
struct MatmulExpertsParams { uint columns, encoding, stride, rows, in_stride, out_stride, row_tiles, experts, in_group, tiles_at, rows_at; };
template <uint ENC, uint TR, typename TW, typename TX>
kernel void nu_matmul_experts_t(device const uchar * weights [[buffer(0)]],
                                device const float * input [[buffer(1)]],
                                device float * output [[buffer(2)]],
                                device const uint * lists [[buffer(3)]],
                                constant MatmulExpertsParams & p [[buffer(7)]],
                                uint group [[threadgroup_position_in_grid]],
                                uint tid [[thread_position_in_threadgroup]],
                                uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup TW tile[TR * 64];
    threadgroup TX xt[64 * 32];
    const uint index = group / p.row_tiles;
    if (index >= lists[0]) return;
    device const uint * tile_entry = lists + p.tiles_at + 3 * index;
    const uint expert = min(tile_entry[0], p.experts - 1);
    MatmulParams shape = { p.columns, p.encoding, p.stride, p.rows, 32, p.in_stride, p.out_stride, p.row_tiles };
    nu_matmul_body<ENC, TR, 32, TW, TX, true, true>(weights + ulong(expert) * ulong(p.rows) * ulong(p.stride), input, output, shape,
                                                    (group % p.row_tiles) * TR, 0, lists + p.rows_at + tile_entry[1], tile_entry[2], p.in_group, tile, xt, tid, sg);
}
// Host-visible instantiations; names and tile shapes must match `kernel_names`
// and `matmulGeometry` in root.zig. The generic tile keeps F32 operands
// (exact for dense F32/F16 rows of any magnitude); the specialized tiles
// round both operands to half. The `_32` set (32×32) serves chunks of at
// most 32 tokens, where the 64-row tiles leave a 5,120-row projection with
// only 80 threadgroups and the GPU mostly idle.
#define NU_MATMUL_ARGS device const uchar *, device const float *, device float *, constant MatmulParams &, uint, uint, uint
template [[host_name("nu_matmul")]] kernel void nu_matmul_t<NU_TILE_GENERIC, 32, 32, float, float, true>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q3_k")]] kernel void nu_matmul_t<11, 64, 64, half, half, true>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q4_k")]] kernel void nu_matmul_t<12, 64, 64, half, half, true>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q5_k")]] kernel void nu_matmul_t<13, 64, 64, half, half, true>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q6_k")]] kernel void nu_matmul_t<14, 64, 64, half, half, true>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_iq3_s")]] kernel void nu_matmul_t<21, 64, 64, half, half, true>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_iq4_xs")]] kernel void nu_matmul_t<23, 64, 64, half, half, true>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q4_0")]] kernel void nu_matmul_t<2, 64, 64, half, half, true>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q3_k_32")]] kernel void nu_matmul_t<11, 32, 32, half, float, false>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q4_k_32")]] kernel void nu_matmul_t<12, 32, 32, half, float, false>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q5_k_32")]] kernel void nu_matmul_t<13, 32, 32, half, float, false>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q6_k_32")]] kernel void nu_matmul_t<14, 32, 32, half, float, false>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_iq3_s_32")]] kernel void nu_matmul_t<21, 32, 32, half, float, false>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_iq4_xs_32")]] kernel void nu_matmul_t<23, 32, 32, half, float, false>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q4_0_32")]] kernel void nu_matmul_t<2, 32, 32, half, float, false>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_pq2_0")]] kernel void nu_matmul_t<142, 64, 64, half, half, true>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_ptq1_0")]] kernel void nu_matmul_t<143, 64, 64, half, half, true>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_pq2_0_32")]] kernel void nu_matmul_t<142, 32, 32, half, float, false>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_ptq1_0_32")]] kernel void nu_matmul_t<143, 32, 32, half, float, false>(NU_MATMUL_ARGS);
// The `_8` set (16 rows × 8 tokens) serves chunks of at most
// `small_chunk_tokens` tokens, where even the 32-row tile leaves too few
// threadgroups.
template [[host_name("nu_matmul_q3_k_8")]] kernel void nu_matmul_split_t<11, half>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q4_k_8")]] kernel void nu_matmul_split_t<12, half>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q5_k_8")]] kernel void nu_matmul_split_t<13, half>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q6_k_8")]] kernel void nu_matmul_split_t<14, half>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_iq3_s_8")]] kernel void nu_matmul_split_t<21, half>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_iq4_xs_8")]] kernel void nu_matmul_split_t<23, half>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q4_0_8")]] kernel void nu_matmul_split_t<2, half>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_pq2_0_8")]] kernel void nu_matmul_split_t<142, half>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_ptq1_0_8")]] kernel void nu_matmul_split_t<143, half>(NU_MATMUL_ARGS);
// Gathered expert tiles: 64 rows × 32 slot rows with half operands for Q4_0
// (an expert averages k · chunk / experts slot rows per chunk, 16 on the
// 26B-A4B at 256 tokens, so one token tile covers most experts), and the
// generic F32 32 × 32 tile for every other encoding or alignment.
#define NU_MATMUL_EXPERTS_ARGS device const uchar *, device const float *, device float *, device const uint *, constant MatmulExpertsParams &, uint, uint, uint
template [[host_name("nu_matmul_experts")]] kernel void nu_matmul_experts_t<NU_TILE_GENERIC, 32, float, float>(NU_MATMUL_EXPERTS_ARGS);
template [[host_name("nu_matmul_experts_q4_0")]] kernel void nu_matmul_experts_t<2, 64, half, half>(NU_MATMUL_EXPERTS_ARGS);

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
// docs/reference/bonsai.md): per block of 1,024 consecutive elements,
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
kernel void nu_silu_inplace(device float * x [[buffer(0)]],
                            constant CountParams & p [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    if (i < p.count) x[i] = nu_silu(x[i]);
}
kernel void nu_gelu_mul(device float * gate [[buffer(0)]], device const float * up [[buffer(1)]],
                        constant CountParams & p [[buffer(7)]], uint i [[thread_position_in_grid]]) {
    if (i < p.count) gate[i] = nu_gelu(gate[i]) * up[i];
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

// ---------------------------------------------------------------------------
// Scalar-gated DeltaNet: one SIMD group per (value head, value coordinate).
// State is [value_head][value_coordinate][key_coordinate], updated in place.
// Input layout: [Q: qheads*keys | K: qheads*keys | V: vheads*values].
struct DeltaParams { uint qheads; uint vheads; uint keys; uint values; float scale; };
kernel void nu_delta(device float * state [[buffer(0)]],
                     device const float * input [[buffer(1)]],
                     device const float * decay_log [[buffer(2)]],
                     device const float * beta_gate [[buffer(3)]],
                     device float * output [[buffer(4)]],
                     constant DeltaParams & p [[buffer(7)]],
                     uint row [[threadgroup_position_in_grid]],
                     uint lane [[thread_index_in_simdgroup]]) {
    uint head = row / p.values, value = row % p.values, kh = head % p.qheads;
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

// ---------------------------------------------------------------------------
// Flash-decoding attention: one query position over `visible` cache
// rows in one pass, no score buffer. Grid = (kv_head, split): a threadgroup
// owns the query heads that share one KV head (GQA, at most NU_DECODE_GROUP)
// over one slice of the visible range, so every cache row is read once for
// the whole group instead of once per query head. Each of its four SIMD
// groups walks every fourth row of the slice (the four rows a threadgroup
// touches per step are adjacent in memory); lane l owns channels l, l+32, …
// of the key and value rows and keeps, per head, a running max `m`, sum `l`,
// and accumulator over its channels (online softmax, all F32; F16 rows are
// converted on load, queries stay F32). The four SIMD groups merge into the
// first through threadgroup memory with the log-sum-exp rescale, and the
// threadgroup writes one partial (m, l, acc[value_width]) per head;
// `nu_attention_merge` combines the splits per head the same way. An empty
// slice or SIMD group leaves m = -inf and l = 0, which both merges skip.
struct AttentionDecodeParams { uint query_heads; uint kv_heads; uint key_width; uint value_width; uint visible; uint splits; float scale; };
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
    const uint hg = rest % head_groups, split = rest / head_groups;
    const uint h0 = hg * GROUP, hn = min(GROUP, g - h0); // this threadgroup's query heads: kv * g + h0 .. + hn
    const uint per = (p.visible + p.splits - 1) / p.splits;
    const uint start = split * per, end = min(p.visible, start + per);
    const uint kch = (p.key_width + 31) / 32, vch = (p.value_width + 31) / 32;
    float q[GROUP][CH];
    for (uint h = 0; h < GROUP; ++h) for (uint i = 0; i < CH; ++i) {
        const uint c = lane + 32 * i;
        q[h][i] = (h < hn && c < p.key_width) ? queries[(kv * g + h0 + h) * p.key_width + c] * p.scale : 0.0f;
    }
    float m[GROUP], l[GROUP], acc[GROUP][CH];
    for (uint h = 0; h < GROUP; ++h) { m[h] = -INFINITY; l[h] = 0.0f; for (uint i = 0; i < CH; ++i) acc[h][i] = 0.0f; }
    const ulong krow = ulong(p.kv_heads) * p.key_width, vrow = ulong(p.kv_heads) * p.value_width;
    for (uint t = start + sg; t < end; t += 4) {
        device const KV * k = keys + ulong(t) * krow + ulong(kv) * p.key_width;
        device const KV * v = values + ulong(t) * vrow + ulong(kv) * p.value_width;
        float kr[CH], vr[CH];
        for (uint i = 0; i < CH; ++i) {
            const uint c = lane + 32 * i;
            kr[i] = (i < kch && c < p.key_width) ? float(k[c]) : 0.0f;
            vr[i] = (i < vch && c < p.value_width) ? float(v[c]) : 0.0f;
        }
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
        device float * out = partials + (ulong(kv * g + h0 + h) * p.splits + split) * stride;
        if (lane == 0) { out[0] = m[h]; out[1] = l[h]; }
        for (uint i = 0; i < CH; ++i) { const uint c = lane + 32 * i; if (i < vch && c < p.value_width) out[2 + c] = acc[h][i]; }
    }
}
#define NU_ATTN_DECODE_ARGS device const float *, device float *, constant AttentionDecodeParams &, uint, uint, uint
template [[host_name("nu_attention_decode")]] kernel void nu_attention_decode_t<float, 8, 8>(device const float *, device const float *, NU_ATTN_DECODE_ARGS);
template [[host_name("nu_attention_decode_h")]] kernel void nu_attention_decode_t<half, 8, 8>(device const half *, device const half *, NU_ATTN_DECODE_ARGS);
template [[host_name("nu_attention_decode_w")]] kernel void nu_attention_decode_t<float, 4, 16>(device const float *, device const float *, NU_ATTN_DECODE_ARGS);
template [[host_name("nu_attention_decode_wh")]] kernel void nu_attention_decode_t<half, 4, 16>(device const half *, device const half *, NU_ATTN_DECODE_ARGS);
// One threadgroup per query head, 256 threads striding the value channels:
// rescale every split's partial to the global max, divide by the merged sum.
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
        output[ulong(head) * p.value_width + c] = sum / total;
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
//
// `T` is the operand type: `float` reads an F32 cache with F32
// queries; `half` reads an F16 cache with a half copy of the queries and
// rounds the probability tile to half before P·V, because the matrix unit
// multiplies operands of one type into F32 accumulators (the matmul's
// contract). Scores, the online softmax, the accumulators, and the output
// stay F32 in both instantiations.
struct AttentionChunkParams { uint query_heads; uint kv_heads; uint key_width; uint value_width; uint position; uint count; uint q_stride; uint out_stride; float scale; uint window; };
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
    const uint limit = min(total, p.position + q0 + 8); // exclusive key bound of this SIMD group's rows
    const ulong krow = ulong(p.kv_heads) * p.key_width, vrow = ulong(p.kv_heads) * p.value_width;
    const uint v0 = vs * 256; // this threadgroup's value columns: v0 .. v0 + 8 * vblocks
    device const T * q = queries + ulong(q0) * p.q_stride + ulong(head) * p.key_width;
    device const T * k = keys + ulong(kv) * p.key_width;
    device const T * v = values + ulong(kv) * p.value_width + v0;
    const uint row = lane >> 2, col0 = (lane & 3) * 8;
    const uint row_limit = min(limit, p.position + q0 + row + 1); // exclusive, this lane's query row
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

// ---------------------------------------------------------------------------
// Chunkwise DeltaNet: `count` tokens of one layer through the WY form
// of the recurrence, one threadgroup per (value head, block of 32 value
// rows), looping over sub-chunks of 32 tokens with the state carried in the
// session buffer between them. Per sub-chunk of n ≤ 32 tokens (γ_t =
// Π a_r as cumulative log decays, r(t,s) = exp(L_t − L_s)):
//   KK = K·Kᵀ, KQ = Q·Kᵀ                      (simdgroup products, tokens × tokens)
//   A[t][s] = β_t r(t,s) KK[t][s] for s < t   (strictly lower)
//   SK = S₀·Kᵀ, SQ = S₀·Qᵀ                    (this group's 32 state rows × tokens)
//   B[t][j] = β_t (v_t[j] − γ_t SK[j][t]);  U = (I + A)⁻¹ B by forward
//   substitution, one thread per value column;
//   O[t][j] = scale (γ_t SQ[j][t] + Σ_{s≤t} r(t,s) KQ[t][s] U[s][j]);
//   S_new = γ_n S₀ + Wᵀ K with W[s][j] = r(n−1, s) U[s][j].
// Value rows are independent given the shared tokens, which is what lets
// four threadgroups split a head and loop over sub-chunks without any
// cross-group synchronization: each reads and writes only its own state
// rows (device-coherent within the group after a mem_device barrier).
// Token rows past `count` in the last sub-chunk are loaded through the
// zero-filling masked loader (never read from the caller's padding) and
// their U rows are zero, so they cannot reach the outputs or the carry.
// Mirrors cpu.recurrent.deltaChunk; the Q/K head of value head h is
// h % qheads, as in nu_delta.
struct DeltaChunkParams { uint qheads; uint vheads; uint keys; uint values; uint count; uint in_stride; uint gate_stride; uint out_stride; float scale; };
#define NU_DELTA_SUB 32   // tokens per state carry
#define NU_DELTA_ROWS 32  // value rows per threadgroup
kernel void nu_delta_chunk(device float * state [[buffer(0)]],
                           device const float * input [[buffer(1)]],
                           device const float * decay_log [[buffer(2)]],
                           device const float * beta_gate [[buffer(3)]],
                           device float * output [[buffer(4)]],
                           constant DeltaChunkParams & p [[buffer(7)]],
                           uint group [[threadgroup_position_in_grid]],
                           uint tid [[thread_position_in_threadgroup]],
                           uint sg [[simdgroup_index_in_threadgroup]],
                           uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float kk[NU_DELTA_SUB * NU_DELTA_SUB];   // K·Kᵀ, then A
    threadgroup float kq[NU_DELTA_SUB * NU_DELTA_SUB];   // Q·Kᵀ, then r·KQ
    threadgroup float sk[NU_DELTA_ROWS * NU_DELTA_SUB];  // [j][t]; reused for the output product
    threadgroup float sq[NU_DELTA_ROWS * NU_DELTA_SUB];  // [j][t]
    threadgroup float u[NU_DELTA_SUB * NU_DELTA_ROWS];   // [t][j]: B, then U
    threadgroup float w[NU_DELTA_SUB * NU_DELTA_ROWS];   // [s][j]: r(n−1,s) U
    threadgroup float cum[NU_DELTA_SUB], beta[NU_DELTA_SUB], diag[64];
    threadgroup float stage[4][64];
    const uint blocks_per_head = p.values / NU_DELTA_ROWS;
    const uint head = group / blocks_per_head, r0 = (group % blocks_per_head) * NU_DELTA_ROWS;
    const uint kh = head % p.qheads, qsize = p.qheads * p.keys;
    device const float * q_base = input + kh * p.keys;
    device const float * k_base = input + qsize + kh * p.keys;
    device const float * v_base = input + 2 * qsize + head * p.values + r0;
    device float * s_base = state + (ulong(head) * p.values + r0) * p.keys;
    const uint kblocks = p.keys / 8;
    for (uint t0 = 0; t0 < p.count; t0 += NU_DELTA_SUB) {
        const uint n = min(uint(NU_DELTA_SUB), p.count - t0);
        device const float * q_rows = q_base + ulong(t0) * p.in_stride;
        device const float * k_rows = k_base + ulong(t0) * p.in_stride;
        device const float * v_rows = v_base + ulong(t0) * p.in_stride;
        // Phase 1: cumulative log decays and betas of this sub-chunk.
        if (tid == 0) {
            float running = 0;
            for (uint t = 0; t < NU_DELTA_SUB; ++t) {
                if (t < n) running += decay_log[(t0 + t) * p.gate_stride + head];
                cum[t] = running;
                beta[t] = t < n ? beta_gate[(t0 + t) * p.gate_stride + head] : 0.0f;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Phase 2: KK[t][s] = k_t·k_s and KQ[t][s] = q_t·k_s; SIMD group sg owns token block t = 8·sg.
        for (uint sb = 0; sb < 4; ++sb) {
            simdgroup_float8x8 akk = simdgroup_float8x8(0.0f), akq = simdgroup_float8x8(0.0f);
            const uint valid_t = min(8u, n - min(n, 8u * sg)), valid_s = min(8u, n - min(n, 8u * sb));
            for (uint d = 0; d < p.keys; d += 8) {
                simdgroup_float8x8 kt, qt, ks;
                if (valid_t == 8) { simdgroup_load(kt, k_rows + 8 * sg * p.in_stride + d, p.in_stride); simdgroup_load(qt, q_rows + 8 * sg * p.in_stride + d, p.in_stride); }
                else { nu_load_rows_masked(kt, k_rows + 8 * sg * p.in_stride + d, p.in_stride, valid_t, stage[sg], lane, false); nu_load_rows_masked(qt, q_rows + 8 * sg * p.in_stride + d, p.in_stride, valid_t, stage[sg], lane, false); }
                if (valid_s == 8) simdgroup_load(ks, k_rows + 8 * sb * p.in_stride + d, p.in_stride, ulong2(0, 0), true);
                else nu_load_rows_masked(ks, k_rows + 8 * sb * p.in_stride + d, p.in_stride, valid_s, stage[sg], lane, true);
                simdgroup_multiply_accumulate(akk, kt, ks, akk);
                simdgroup_multiply_accumulate(akq, qt, ks, akq);
            }
            simdgroup_store(akk, kk + 8 * sg * NU_DELTA_SUB + 8 * sb, NU_DELTA_SUB);
            simdgroup_store(akq, kq + 8 * sg * NU_DELTA_SUB + 8 * sb, NU_DELTA_SUB);
        }
        // Phase 3: SK[j][t] = Σ_i S₀[r0+j][i] k_t[i], SQ likewise; SIMD group sg owns value-row block j = 8·sg.
        for (uint tb = 0; tb < 4; ++tb) {
            simdgroup_float8x8 ask = simdgroup_float8x8(0.0f), asq = simdgroup_float8x8(0.0f);
            const uint valid_t = min(8u, n - min(n, 8u * tb));
            for (uint d = 0; d < p.keys; d += 8) {
                simdgroup_float8x8 s0, kt, qt;
                simdgroup_load(s0, s_base + 8 * sg * p.keys + d, p.keys);
                if (valid_t == 8) { simdgroup_load(kt, k_rows + 8 * tb * p.in_stride + d, p.in_stride, ulong2(0, 0), true); simdgroup_load(qt, q_rows + 8 * tb * p.in_stride + d, p.in_stride, ulong2(0, 0), true); }
                else { nu_load_rows_masked(kt, k_rows + 8 * tb * p.in_stride + d, p.in_stride, valid_t, stage[sg], lane, true); nu_load_rows_masked(qt, q_rows + 8 * tb * p.in_stride + d, p.in_stride, valid_t, stage[sg], lane, true); }
                simdgroup_multiply_accumulate(ask, s0, kt, ask);
                simdgroup_multiply_accumulate(asq, s0, qt, asq);
            }
            simdgroup_store(ask, sk + 8 * sg * NU_DELTA_SUB + 8 * tb, NU_DELTA_SUB);
            simdgroup_store(asq, sq + 8 * sg * NU_DELTA_SUB + 8 * tb, NU_DELTA_SUB);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Phase 4: decay ratios into A and KQ; B into u. 1,024 entries each, 8 per thread.
        for (uint i = tid; i < NU_DELTA_SUB * NU_DELTA_SUB; i += 128) {
            const uint t = i / NU_DELTA_SUB, s = i % NU_DELTA_SUB;
            const float ratio = (t < n && s <= t) ? exp(cum[t] - cum[s]) : 0.0f;
            kk[i] = (s < t) ? beta[t] * ratio * kk[i] : 0.0f;
            kq[i] = ratio * kq[i];
            const uint j = s; // reuse the same index space for u[t][j]
            u[i] = (t < n) ? beta[t] * (v_rows[t * p.in_stride + j] - exp(cum[t]) * sk[j * NU_DELTA_SUB + t]) : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Phase 5: forward substitution, thread j owns value column j (rows read only after they are final).
        if (tid < NU_DELTA_ROWS) {
            for (uint t = 1; t < n; ++t) {
                float acc = u[t * NU_DELTA_ROWS + tid];
                for (uint s = 0; s < t; ++s) acc -= kk[t * NU_DELTA_SUB + s] * u[s * NU_DELTA_ROWS + tid];
                u[t * NU_DELTA_ROWS + tid] = acc;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Phase 6: O = KQ·U into sk (reused), then scale and add the S₀ term; SIMD group sg owns token block 8·sg.
        for (uint jb = 0; jb < 4; ++jb) {
            simdgroup_float8x8 acc = simdgroup_float8x8(0.0f);
            for (uint sb = 0; sb < 4; ++sb) {
                simdgroup_float8x8 a, b;
                simdgroup_load(a, kq + 8 * sg * NU_DELTA_SUB + 8 * sb, NU_DELTA_SUB);
                simdgroup_load(b, u + 8 * sb * NU_DELTA_ROWS + 8 * jb, NU_DELTA_ROWS);
                simdgroup_multiply_accumulate(acc, a, b, acc);
            }
            simdgroup_store(acc, sk + 8 * sg * NU_DELTA_ROWS + 8 * jb, NU_DELTA_ROWS); // sk now [t][j]
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < NU_DELTA_SUB * NU_DELTA_ROWS; i += 128) {
            const uint t = i / NU_DELTA_ROWS, j = i % NU_DELTA_ROWS;
            if (t < n) output[ulong(t0 + t) * p.out_stride + head * p.values + r0 + j] = p.scale * (exp(cum[t]) * sq[j * NU_DELTA_SUB + t] + sk[i]);
            w[i] = (t < n) ? exp(cum[n - 1] - cum[t]) * u[i] : 0.0f;
        }
        for (uint i = tid; i < 64; i += 128) diag[i] = ((i >> 3) == (i & 7)) ? exp(cum[n - 1]) : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Phase 7: S_new = γ_n S₀ + Wᵀ K over this group's rows; SIMD group sg owns value-row block 8·sg.
        {
            simdgroup_float8x8 dg, wt[4];
            simdgroup_load(dg, diag, 8);
            for (uint sb = 0; sb < 4; ++sb) simdgroup_load(wt[sb], w + 8 * sb * NU_DELTA_ROWS + 8 * sg, NU_DELTA_ROWS, ulong2(0, 0), true);
            for (uint ib = 0; ib < kblocks; ++ib) {
                simdgroup_float8x8 s0, acc;
                simdgroup_load(s0, s_base + 8 * sg * p.keys + 8 * ib, p.keys);
                simdgroup_multiply(acc, dg, s0);
                for (uint sb = 0; sb < 4; ++sb) {
                    const uint valid_s = min(8u, n - min(n, 8u * sb));
                    if (valid_s == 0) break;
                    simdgroup_float8x8 kb;
                    if (valid_s == 8) simdgroup_load(kb, k_rows + 8 * sb * p.in_stride + 8 * ib, p.in_stride);
                    else nu_load_rows_masked(kb, k_rows + 8 * sb * p.in_stride + 8 * ib, p.in_stride, valid_s, stage[sg], lane, false);
                    simdgroup_multiply_accumulate(acc, wt[sb], kb, acc);
                }
                simdgroup_store(acc, s_base + 8 * sg * p.keys + 8 * ib, p.keys);
            }
        }
        threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup);
    }
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
