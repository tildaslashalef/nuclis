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
// encodings holding 96 % of Qwen's bytes, so these decide decode speed; then
// Q3_K, Q4_0 (the QAT Gemma checkpoint's only weight encoding), the ternary
// blocks, and the codebook encodings further down (Saluki's).
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
// IQ4 codes are looked up in a threadgroup copy of the table: divergent reads
// of the `constant` table cost the IQ4_XS matvec 3-11 % of its rate.
inline float4 nu_iq4_tg(threadgroup const float * tv, uint w) { uchar4 n = as_type<uchar4>(w & 0x0f0f0f0fu); return float4(tv[n.x], tv[n.y], tv[n.z], tv[n.w]); }
// Fills that copy; every thread of the group must reach it (it holds a barrier).
inline void nu_iq4_table(threadgroup float * tv, uint sg, uint lane) {
    if (sg == 0 && lane < 16) tv[lane] = nu_iq4_values_f[lane];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
inline float nu_dot(float4 a, float4 b, float acc) { return fma(a.w, b.w, fma(a.z, b.z, fma(a.y, b.y, fma(a.x, b.x, acc)))); }
inline float nu_sum4(float4 v) { return (v.x + v.y) + (v.z + v.w); }
// Two codes of at most ten bits OR'd into the mantissas of halves whose
// exponent is 1024's, then 1024 subtracted: exact, and on the FP16 pipe
// instead of an integer-to-float conversion per value.
inline float2 nu_magic(uint bits) { return float2(as_type<half2>(bits | 0x64006400u) - half2(1024.0h)); }
// The eight nibbles of `w` dotted into two sums: low nibbles with `xa`, high
// nibbles with `xb` scaled by 16 (they stay at bits 4-7; the caller divides).
inline void nu_magic_nibbles(uint w, float4 xa, float4 xb, thread float & sa, thread float & sb) {
    uint w8 = w >> 8;
    float2 lo02 = nu_magic(w & 0x000f000fu), lo13 = nu_magic(w8 & 0x000f000fu);
    float2 hi02 = nu_magic(w & 0x00f000f0u), hi13 = nu_magic(w8 & 0x00f000f0u);
    sa = fma(lo13.y, xa.w, fma(lo02.y, xa.z, fma(lo13.x, xa.y, fma(lo02.x, xa.x, sa))));
    sb = fma(hi13.y, xb.w, fma(hi02.y, xb.z, fma(hi13.x, xb.y, fma(hi02.x, xb.x, sb))));
}
// Q5_K's form of the above: bit `bit_a` (low) or `bit_b` (high) of each byte
// of the plane word `h` completes each code at bit 4 (low) or bit 8 (high, ×16).
inline void nu_magic_fives(uint v, uint h, uint bit_a, uint bit_b, float4 xa, float4 xb, thread float & sa, thread float & sb) {
    uint v8 = v >> 8, ha = h >> bit_a, hb = h >> bit_b;
    float2 lo02 = nu_magic((v & 0x000f000fu) | ((ha & 0x00010001u) << 4)), lo13 = nu_magic((v8 & 0x000f000fu) | ((ha >> 4) & 0x00100010u));
    float2 hi02 = nu_magic((v & 0x00f000f0u) | ((hb & 0x00010001u) << 8)), hi13 = nu_magic((v8 & 0x00f000f0u) | (hb & 0x01000100u));
    sa = fma(lo13.y, xa.w, fma(lo02.y, xa.z, fma(lo13.x, xa.y, fma(lo02.x, xa.x, sa))));
    sb = fma(hi13.y, xb.w, fma(hi02.y, xb.z, fma(hi13.x, xb.y, fma(hi02.x, xb.x, sb))));
}
// The same two decodes returning the codes in byte order: `lo` the low
// nibbles, `hi` the high ones at ×16 (Q5_K's fifth bit included when FIFTH).
template <bool FIFTH>
inline void nu_magic_word(uint v, uint h, uint bit_a, uint bit_b, thread float4 & lo, thread float4 & hi) {
    uint v8 = v >> 8, ha = h >> bit_a, hb = h >> bit_b;
    float2 lo02 = nu_magic((v & 0x000f000fu) | (FIFTH ? (ha & 0x00010001u) << 4 : 0u)), lo13 = nu_magic((v8 & 0x000f000fu) | (FIFTH ? (ha >> 4) & 0x00100010u : 0u));
    float2 hi02 = nu_magic((v & 0x00f000f0u) | (FIFTH ? (hb & 0x00010001u) << 8 : 0u)), hi13 = nu_magic((v8 & 0x00f000f0u) | (FIFTH ? hb & 0x01000100u : 0u));
    lo = float4(lo02.x, lo13.x, lo02.y, lo13.y);
    hi = float4(hi02.x, hi13.x, hi02.y, hi13.y);
}
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
// `group` of byte `column`. `first`/`last` bound the lane's 256-value K blocks
// to `[first, last)`: the whole row for the standalone kernels, the split's
// share for the split bodies.
template <uint ROWS, bool FIFTH_BIT>
inline void nu_matvec_k_body(device const uchar * weights, device const float * input, MatvecBlockParams p, uint group_index, uint sg, uint lane, thread float * acc, uint first, uint last) {
    const uint block_bytes = FIFTH_BIT ? 176 : 144;
    const uint nibble_offset = FIFTH_BIT ? 48 : 16;
    uint row0 = (group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS;
    if (row0 >= p.rows) return;
    uint pair = (lane & 7) >> 1, half_index = lane & 1;
    for (uint r = 0; r < ROWS; ++r) acc[r] = 0;
    for (uint kb = first + (lane >> 3); kb < last; kb += 4) {
        device const float * x = input + kb * 256 + pair * 64 + half_index * 16;
        NuInputs16 xa = nu_inputs16(x), xb = nu_inputs16(x + 32);
        uint slice = kb * block_bytes + nibble_offset + pair * 32 + half_index * 16;
        uint plane = kb * block_bytes + 16 + half_index * 16;
        for (uint r = 0; r < ROWS; ++r) {
            device const uchar * row = weights + ulong(min(row0 + r, p.rows - 1)) * p.stride;
            uint dd = *(device const uint *)(row + kb * block_bytes);
            packed_uint3 s = *(device const packed_uint3 *)(row + kb * block_bytes + 4);
            uint4 v = *(device const uint4 *)(row + slice);
            float sqa = 0.0f, sqb = 0.0f;
            if (FIFTH_BIT) {
                uint4 h = *(device const uint4 *)(row + plane);
                uint bit_a = 2 * pair, bit_b = bit_a + 1;
                nu_magic_fives(v.x, h.x, bit_a, bit_b, xa.v[0], xb.v[0], sqa, sqb);
                nu_magic_fives(v.y, h.y, bit_a, bit_b, xa.v[1], xb.v[1], sqa, sqb);
                nu_magic_fives(v.z, h.z, bit_a, bit_b, xa.v[2], xb.v[2], sqa, sqb);
                nu_magic_fives(v.w, h.w, bit_a, bit_b, xa.v[3], xb.v[3], sqa, sqb);
            } else {
                nu_magic_nibbles(v.x, xa.v[0], xb.v[0], sqa, sqb);
                nu_magic_nibbles(v.y, xa.v[1], xb.v[1], sqa, sqb);
                nu_magic_nibbles(v.z, xa.v[2], xb.v[2], sqa, sqb);
                nu_magic_nibbles(v.w, xa.v[3], xb.v[3], sqa, sqb);
            }
            sqb *= 0.0625f;
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
    nu_matvec_k_body<ROWS, false>(weights, input, p, group_index, sg, lane, acc, 0u, p.blocks);
    if ((group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS < p.rows)
        nu_store_rows<ROWS>(acc, output, (group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS, p.rows, lane);
}
template <uint ROWS>
kernel void nu_matvec_q5_k(device const uchar * weights [[buffer(0)]], device const float * input [[buffer(1)]], device float * output [[buffer(2)]],
                           constant MatvecBlockParams & p [[buffer(7)]], uint group_index [[threadgroup_position_in_grid]],
                           uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    float acc[ROWS];
    nu_matvec_k_body<ROWS, true>(weights, input, p, group_index, sg, lane, acc, 0u, p.blocks);
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
inline void nu_matvec_iq4_xs_body(device const uchar * weights, device const float * input, MatvecBlockParams p, uint group_index, uint sg, uint lane, thread float * acc, threadgroup const float * tv) {
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
            float sum = nu_dot(nu_iq4_tg(tv, qa.x), x0, nu_dot(nu_iq4_tg(tv, qa.y), x1, nu_dot(nu_iq4_tg(tv, qb.x), x2, nu_dot(nu_iq4_tg(tv, qb.y), x3, 0.0f))));
            sum = nu_dot(nu_iq4_tg(tv, qa.x >> 4), x4, nu_dot(nu_iq4_tg(tv, qa.y >> 4), x5, nu_dot(nu_iq4_tg(tv, qb.x >> 4), x6, nu_dot(nu_iq4_tg(tv, qb.y >> 4), x7, sum))));
            acc[r] += (d * scale) * sum;
        }
    }
}
template <uint ROWS>
kernel void nu_matvec_iq4_xs(device const uchar * weights [[buffer(0)]], device const float * input [[buffer(1)]], device float * output [[buffer(2)]],
                            constant MatvecBlockParams & p [[buffer(7)]], uint group_index [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float tv[16];
    nu_iq4_table(tv, sg, lane);
    float acc[ROWS];
    nu_matvec_iq4_xs_body<ROWS>(weights, input, p, group_index, sg, lane, acc, tv);
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
// Ternary blocks of 128 (PQ2_0 34 B, PTQ1_0 28 B; docs/models/bonsai.md):
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

// Q3_K: 110-byte blocks, so never more than two-byte alignment. Each lane
// owns one consecutive group of 32 values; its input is reused across ROWS
// output rows.
inline float nu_q3_scale(device const uchar * s, uint group) {
    uint nibble = (s[group & 7] >> (group < 8 ? 0 : 4)) & 15u;
    uint upper = (s[8 + group % 4] >> (2 * (group / 4))) & 3u;
    return float(int(nibble | (upper << 4)) - 32);
}
template <uint ROWS>
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
template <uint ROWS>
kernel void nu_matvec_three(device const uchar * weights [[buffer(0)]], device const float * input [[buffer(1)]], device float * output [[buffer(2)]],
                            constant MatvecBlockParams & p [[buffer(7)]], uint group_index [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    float acc[ROWS];
    nu_matvec_three_body<ROWS>(weights, input, p, group_index, sg, lane, acc);
    if ((group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS < p.rows)
        nu_store_rows<ROWS>(acc, output, (group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS, p.rows, lane);
}

