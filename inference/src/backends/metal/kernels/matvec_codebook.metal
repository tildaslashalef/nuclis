// ---------------------------------------------------------------------------
// The codebook encodings (IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS; block layouts in
// dequant.metal). A grid entry holds eight magnitudes below 64, one per byte.
// Two bytes masked into the low bits of a half pair are the F16 subnormals
// q·2^-24, exact, and the signs are XOR'd into the half sign bits; the 2^24
// joins the block scale. No integer-to-float conversion or integer multiply
// per value: those share the integer and complex pipe, the one that bounds
// these kernels (docs/engine/apple-gpu.md). An octet's values pair as (0,2),
// (1,3) from its low word and (4,6), (5,7) from its high word;
// `nu_octet_dot` pairs the inputs the same way. With a one-hot input the sum
// is ±q·2^-24 exactly, so the result is the CPU decoder's multiplier · q.
struct NuOctet { half2 v02, v13, v46, v57; };
inline half2 nu_signed_pair(uint bits, uint mask) { return as_type<half2>(bits ^ (mask & 0x80008000u)); }
// Bit j of `signs` negates value j. `u` holds bits j and j+2 sixteen apart,
// so one shift puts a pair's two signs on bits 15 and 31.
inline NuOctet nu_signed_octet(ulong entry, uint signs) {
    uint lo = uint(entry), hi = uint(entry >> 32), u = signs | (signs << 14);
    NuOctet o;
    o.v02 = nu_signed_pair(lo & 0x00ff00ffu, u << 15);
    o.v13 = nu_signed_pair((lo >> 8) & 0x00ff00ffu, u << 14);
    o.v46 = nu_signed_pair(hi & 0x00ff00ffu, u << 11);
    o.v57 = nu_signed_pair((hi >> 8) & 0x00ff00ffu, u << 10);
    return o;
}
// The octet dotted with eight consecutive inputs `xa`, `xb`.
inline float nu_octet_dot(NuOctet o, float4 xa, float4 xb, float sum) {
    sum = fma(float(o.v02.x), xa.x, fma(float(o.v13.x), xa.y, fma(float(o.v02.y), xa.z, fma(float(o.v13.y), xa.w, sum))));
    return fma(float(o.v46.x), xb.x, fma(float(o.v57.x), xb.y, fma(float(o.v46.y), xb.z, fma(float(o.v57.y), xb.w, sum))));
}

// IQ2_XXS. Lane slice: one 32-value group `g`, the eight bytes at 2 + 8g:
// four grid indices, then a word of four 7-bit sign indices with the
// four-bit scale on top. 66-byte blocks: two-byte alignment. The grid stays
// in `constant` memory: a threadgroup copy measured no faster.
template <uint ROWS>
inline void nu_matvec_iq2_xxs_body(device const uchar * weights, device const float * input, MatvecBlockParams p, uint group_index, uint sg, uint lane, thread float * acc) {
    uint row0 = (group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS;
    if (row0 >= p.rows) return;
    uint g = lane & 7;
    for (uint r = 0; r < ROWS; ++r) acc[r] = 0;
    for (uint kb = lane >> 3; kb < p.blocks; kb += 4) {
        device const float4 * x = (device const float4 *)(input + kb * 256 + g * 32);
        float4 xv[8] = { x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7] };
        for (uint r = 0; r < ROWS; ++r) {
            device const uchar * b = weights + ulong(min(row0 + r, p.rows - 1)) * p.stride + kb * 66;
            packed_ushort4 q = *(device const packed_ushort4 *)(b + 2 + g * 8);
            uint indices = nu_word(q, 0), word = nu_word(q, 1);
            float sum = 0.0f;
            for (uint l = 0; l < 4; ++l)
                sum = nu_octet_dot(nu_signed_octet(nu_iq2_xxs_grid[(indices >> (8 * l)) & 255u], nu_signs7(word >> (7 * l))), xv[2 * l], xv[2 * l + 1], sum);
            float d = float(as_type<half>(*(device const ushort *)b));
            acc[r] += ((d * ((0.5f + float(word >> 28)) * 0.25f)) * 16777216.0f) * sum;
        }
    }
}
// IQ2_XS. Lane slice: group `g`: four 16-bit codes at 2 + 8g (a 9-bit grid
// index under a 7-bit sign index), and a scale byte at 66 + g whose low
// nibble scales octets 0-1 and high nibble octets 2-3. 74-byte blocks.
template <uint ROWS, typename Grid>
inline void nu_matvec_iq2_xs_body(device const uchar * weights, device const float * input, MatvecBlockParams p, uint group_index, uint sg, uint lane, thread float * acc, Grid grid) {
    uint row0 = (group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS;
    if (row0 >= p.rows) return;
    uint g = lane & 7;
    for (uint r = 0; r < ROWS; ++r) acc[r] = 0;
    for (uint kb = lane >> 3; kb < p.blocks; kb += 4) {
        device const float4 * x = (device const float4 *)(input + kb * 256 + g * 32);
        float4 xv[8] = { x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7] };
        for (uint r = 0; r < ROWS; ++r) {
            device const uchar * b = weights + ulong(min(row0 + r, p.rows - 1)) * p.stride + kb * 74;
            packed_ushort4 q = *(device const packed_ushort4 *)(b + 2 + g * 8);
            float sums[2] = { 0.0f, 0.0f };
            for (uint l = 0; l < 4; ++l) {
                uint code = uint(q[l]);
                sums[l / 2] = nu_octet_dot(nu_signed_octet(grid[code & 511u], nu_signs7(code >> 9)), xv[2 * l], xv[2 * l + 1], sums[l / 2]);
            }
            float d = float(as_type<half>(*(device const ushort *)b)) * 16777216.0f;
            uint scales = b[66 + g];
            acc[r] += (d * ((0.5f + float(scales & 15u)) * 0.25f)) * sums[0] + (d * ((0.5f + float(scales >> 4)) * 0.25f)) * sums[1];
        }
    }
}

// IQ2_S. Lane slice: group `g`: four index bytes at 2 + 4g completed by two
// bits each from byte 66 + g, a full sign byte per octet at 34 + 4g, and a
// scale byte at 74 + g split as IQ2_XS's. 82-byte blocks.
template <uint ROWS, typename Grid>
inline void nu_matvec_iq2_s_body(device const uchar * weights, device const float * input, MatvecBlockParams p, uint group_index, uint sg, uint lane, thread float * acc, Grid grid) {
    uint row0 = (group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS;
    if (row0 >= p.rows) return;
    uint g = lane & 7;
    for (uint r = 0; r < ROWS; ++r) acc[r] = 0;
    for (uint kb = lane >> 3; kb < p.blocks; kb += 4) {
        device const float4 * x = (device const float4 *)(input + kb * 256 + g * 32);
        float4 xv[8] = { x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7] };
        for (uint r = 0; r < ROWS; ++r) {
            device const uchar * b = weights + ulong(min(row0 + r, p.rows - 1)) * p.stride + kb * 82;
            packed_ushort2 qi = *(device const packed_ushort2 *)(b + 2 + g * 4);
            packed_ushort2 qs = *(device const packed_ushort2 *)(b + 34 + g * 4);
            uint indices = uint(qi.x) | (uint(qi.y) << 16), signs = uint(qs.x) | (uint(qs.y) << 16), high = b[66 + g];
            float sums[2] = { 0.0f, 0.0f };
            for (uint l = 0; l < 4; ++l) {
                uint index = ((indices >> (8 * l)) & 255u) | (((high >> (2 * l)) & 3u) << 8);
                sums[l / 2] = nu_octet_dot(nu_signed_octet(grid[index], (signs >> (8 * l)) & 255u), xv[2 * l], xv[2 * l + 1], sums[l / 2]);
            }
            float d = float(as_type<half>(*(device const ushort *)b)) * 16777216.0f;
            uint scales = b[74 + g];
            acc[r] += (d * ((0.5f + float(scales & 15u)) * 0.25f)) * sums[0] + (d * ((0.5f + float(scales >> 4)) * 0.25f)) * sums[1];
        }
    }
}

// IQ3_XXS. Lane slice: group `g`: eight index bytes at 2 + 8g, two
// four-magnitude entries per octet, and the word at 66 + 4g of four 7-bit
// sign indices under the scale: d·(0.5 + s)/2. 98-byte blocks.
template <uint ROWS>
inline void nu_matvec_iq3_xxs_body(device const uchar * weights, device const float * input, MatvecBlockParams p, uint group_index, uint sg, uint lane, thread float * acc) {
    uint row0 = (group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS;
    if (row0 >= p.rows) return;
    uint g = lane & 7;
    for (uint r = 0; r < ROWS; ++r) acc[r] = 0;
    for (uint kb = lane >> 3; kb < p.blocks; kb += 4) {
        device const float4 * x = (device const float4 *)(input + kb * 256 + g * 32);
        float4 xv[8] = { x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7] };
        for (uint r = 0; r < ROWS; ++r) {
            device const uchar * b = weights + ulong(min(row0 + r, p.rows - 1)) * p.stride + kb * 98;
            packed_ushort4 q = *(device const packed_ushort4 *)(b + 2 + g * 8);
            packed_ushort2 w = *(device const packed_ushort2 *)(b + 66 + g * 4);
            uint word = uint(w.x) | (uint(w.y) << 16);
            float sum = 0.0f;
            for (uint l = 0; l < 4; ++l) {
                uint pair = uint(q[l]);
                ulong entry = ulong(nu_iq3_xxs_grid[pair & 255u]) | (ulong(nu_iq3_xxs_grid[pair >> 8]) << 32);
                sum = nu_octet_dot(nu_signed_octet(entry, nu_signs7(word >> (7 * l))), xv[2 * l], xv[2 * l + 1], sum);
            }
            float d = float(as_type<half>(*(device const ushort *)b));
            acc[r] += ((d * ((0.5f + float(word >> 28)) * 0.5f)) * 16777216.0f) * sum;
        }
    }
}

// IQ3_S. Lane slice: group `g`: eight index bytes at 2 + 8g completed by a
// ninth bit each from byte 66 + g, two four-magnitude entries per octet, a
// full sign byte per octet at 74 + 4g, and the odd scale 1 + 2s from the
// nibble of byte 106 + g/2. 110-byte blocks.
template <uint ROWS>
inline void nu_matvec_iq3_s_body(device const uchar * weights, device const float * input, MatvecBlockParams p, uint group_index, uint sg, uint lane, thread float * acc) {
    uint row0 = (group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS;
    if (row0 >= p.rows) return;
    uint g = lane & 7;
    for (uint r = 0; r < ROWS; ++r) acc[r] = 0;
    for (uint kb = lane >> 3; kb < p.blocks; kb += 4) {
        device const float4 * x = (device const float4 *)(input + kb * 256 + g * 32);
        float4 xv[8] = { x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7] };
        for (uint r = 0; r < ROWS; ++r) {
            device const uchar * b = weights + ulong(min(row0 + r, p.rows - 1)) * p.stride + kb * 110;
            packed_ushort4 q = *(device const packed_ushort4 *)(b + 2 + g * 8);
            packed_ushort2 sw = *(device const packed_ushort2 *)(b + 74 + g * 4);
            uint signs = uint(sw.x) | (uint(sw.y) << 16), high = b[66 + g];
            float sum = 0.0f;
            for (uint l = 0; l < 4; ++l) {
                uint pair = uint(q[l]), h = high >> (2 * l);
                ulong entry = ulong(nu_iq3_grid[(pair & 255u) | ((h & 1u) << 8)]) | (ulong(nu_iq3_grid[(pair >> 8) | ((h & 2u) << 7)]) << 32);
                sum = nu_octet_dot(nu_signed_octet(entry, (signs >> (8 * l)) & 255u), xv[2 * l], xv[2 * l + 1], sum);
            }
            float d = float(as_type<half>(*(device const ushort *)b));
            acc[r] += ((d * float(1 + 2 * ((b[106 + g / 2] >> (4 * (g & 1))) & 15u))) * 16777216.0f) * sum;
        }
    }
}

// Q2_K. Lane slice: half `H` (values 128H..) and byte columns 8c..8c+7 of
// its 32 code bytes at 16 + 32H, whose four bit pairs are the four 32-value
// runs `s` (values 128H + 32s + 8c ..); each run's eight values lie in one
// 16-value group with a scale (low nibble) and a minimum (high nibble) at
// byte 8H + 2s + c/2. Codes decode as subnormal pairs like the codebook
// encodings: Σ(d·sc·q − dmin·m)·x = d·sc·Σq·x − dmin·m·Σx. 84-byte blocks,
// four-byte alignment.
template <uint ROWS>
inline void nu_matvec_q2_k_body(device const uchar * weights, device const float * input, MatvecBlockParams p, uint group_index, uint sg, uint lane, thread float * acc) {
    uint row0 = (group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS;
    if (row0 >= p.rows) return;
    uint half_index = (lane & 7) >> 2, c = lane & 3;
    for (uint r = 0; r < ROWS; ++r) acc[r] = 0;
    for (uint kb = lane >> 3; kb < p.blocks; kb += 4) {
        device const float4 * x = (device const float4 *)(input + kb * 256 + half_index * 128 + c * 8);
        float4 xv[8] = { x[0], x[1], x[8], x[9], x[16], x[17], x[24], x[25] };
        float sx[4] = { nu_sum4(xv[0] + xv[1]), nu_sum4(xv[2] + xv[3]), nu_sum4(xv[4] + xv[5]), nu_sum4(xv[6] + xv[7]) };
        for (uint r = 0; r < ROWS; ++r) {
            device const uchar * b = weights + ulong(min(row0 + r, p.rows - 1)) * p.stride + kb * 84;
            uint w0 = *(device const uint *)(b + 16 + half_index * 32 + c * 8), w1 = *(device const uint *)(b + 20 + half_index * 32 + c * 8);
            uint sc0 = *(device const uint *)(b + half_index * 8), sc1 = *(device const uint *)(b + half_index * 8 + 4);
            uint dd = *(device const uint *)(b + 80);
            float d = nu_half_low(dd) * 16777216.0f, dmin = nu_half_high(dd);
            float total = 0.0f;
            for (uint s = 0; s < 4; ++s) {
                half2 q02 = as_type<half2>((w0 >> (2 * s)) & 0x00030003u), q13 = as_type<half2>((w0 >> (2 * s + 8)) & 0x00030003u);
                half2 q46 = as_type<half2>((w1 >> (2 * s)) & 0x00030003u), q57 = as_type<half2>((w1 >> (2 * s + 8)) & 0x00030003u);
                float4 xa = xv[2 * s], xb = xv[2 * s + 1];
                float sum = fma(float(q02.x), xa.x, fma(float(q13.x), xa.y, fma(float(q02.y), xa.z, float(q13.y) * xa.w)));
                sum = fma(float(q46.x), xb.x, fma(float(q57.x), xb.y, fma(float(q46.y), xb.z, fma(float(q57.y), xb.w, sum))));
                uint scale = nu_byte(s < 2 ? sc0 : sc1, 2 * (s & 1) + (c >> 1));
                total += (d * float(scale & 15u)) * sum - (dmin * float(scale >> 4)) * sx[s];
            }
            acc[r] += total;
        }
    }
}

// The IQ1 grid as 2-bit codes, component + 1 at bits 2j (`nu_iq1_codes`, a
// quarter of the byte table). `e` holds codes j and j+1 sixteen bits apart,
// so one shift and mask gives a subnormal pair; the +1 is taken back
// through the input sum by the caller.
inline float nu_ternary_octet_dot(uint codes, float4 xa, float4 xb, float sum) {
    uint e = codes | (codes << 14);
    half2 v01 = as_type<half2>(e & 0x00030003u), v23 = as_type<half2>((e >> 4) & 0x00030003u);
    half2 v45 = as_type<half2>((e >> 8) & 0x00030003u), v67 = as_type<half2>((e >> 12) & 0x00030003u);
    sum = fma(float(v01.x), xa.x, fma(float(v01.y), xa.y, fma(float(v23.x), xa.z, fma(float(v23.y), xa.w, sum))));
    return fma(float(v45.x), xb.x, fma(float(v45.y), xb.y, fma(float(v67.x), xb.z, fma(float(v67.y), xb.w, sum))));
}

// IQ1_S. Lane slice: group `g`: four index bytes at 2 + 4g, and the word at
// 34 + 2g of four 3-bit high index parts, a 3-bit scale, and the shift's
// sign. Each value is m·(t + δ) = m·((t + 1) + (δ − 1)), so the shift and
// the code bias fold into the group's input sum. 50-byte blocks.
template <uint ROWS>
inline void nu_matvec_iq1_s_body(device const uchar * weights, device const float * input, MatvecBlockParams p, uint group_index, uint sg, uint lane, thread float * acc) {
    uint row0 = (group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS;
    if (row0 >= p.rows) return;
    uint g = lane & 7;
    for (uint r = 0; r < ROWS; ++r) acc[r] = 0;
    for (uint kb = lane >> 3; kb < p.blocks; kb += 4) {
        device const float4 * x = (device const float4 *)(input + kb * 256 + g * 32);
        float4 xv[8] = { x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7] };
        float sx = nu_sum4(((xv[0] + xv[1]) + (xv[2] + xv[3])) + ((xv[4] + xv[5]) + (xv[6] + xv[7])));
        for (uint r = 0; r < ROWS; ++r) {
            device const uchar * b = weights + ulong(min(row0 + r, p.rows - 1)) * p.stride + kb * 50;
            packed_ushort2 qi = *(device const packed_ushort2 *)(b + 2 + g * 4);
            uint indices = uint(qi.x) | (uint(qi.y) << 16), word = *(device const ushort *)(b + 34 + g * 2);
            float sum = 0.0f;
            for (uint l = 0; l < 4; ++l)
                sum = nu_ternary_octet_dot(nu_iq1_codes[((indices >> (8 * l)) & 255u) | (((word >> (3 * l)) & 7u) << 8)], xv[2 * l], xv[2 * l + 1], sum);
            float m = float(as_type<half>(*(device const ushort *)b)) * float(2 * ((word >> 12) & 7u) + 1);
            acc[r] += m * fma((word & 0x8000u) ? -1.125f : -0.875f, sx, sum * 16777216.0f);
        }
    }
}

// IQ1_M. Lane slice: group `g`: four index bytes at 4g, two bytes at 32 + 2g
// whose nibble l holds octet l's three high index bits and its shift's sign,
// and the four scale words at 48, which carry each 16 values' 3-bit scale and
// the F16 block scale in their top nibbles. 56-byte blocks, eight-byte
// alignment.
template <uint ROWS>
inline void nu_matvec_iq1_m_body(device const uchar * weights, device const float * input, MatvecBlockParams p, uint group_index, uint sg, uint lane, thread float * acc) {
    uint row0 = (group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS;
    if (row0 >= p.rows) return;
    uint g = lane & 7;
    for (uint r = 0; r < ROWS; ++r) acc[r] = 0;
    for (uint kb = lane >> 3; kb < p.blocks; kb += 4) {
        device const float4 * x = (device const float4 *)(input + kb * 256 + g * 32);
        float4 xv[8] = { x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7] };
        float sx[4] = { nu_sum4(xv[0] + xv[1]), nu_sum4(xv[2] + xv[3]), nu_sum4(xv[4] + xv[5]), nu_sum4(xv[6] + xv[7]) };
        for (uint r = 0; r < ROWS; ++r) {
            device const uchar * b = weights + ulong(min(row0 + r, p.rows - 1)) * p.stride + kb * 56;
            uint indices = *(device const uint *)(b + g * 4), high = *(device const ushort *)(b + 32 + g * 2);
            uint2 words = *(device const uint2 *)(b + 48);
            uint bits = ((words.x >> 12) & 0x000fu) | ((words.x >> 24) & 0x00f0u) | ((words.y >> 4) & 0x0f00u) | ((words.y >> 16) & 0xf000u);
            float d = float(as_type<half>(ushort(bits)));
            uint word = ((g < 4 ? words.x : words.y) >> (16 * ((g >> 1) & 1))) >> (6 * (g & 1));
            float sums[2] = { 0.0f, 0.0f };
            for (uint l = 0; l < 4; ++l) {
                uint nibble = (high >> (4 * l)) & 15u;
                float s = nu_ternary_octet_dot(nu_iq1_codes[((indices >> (8 * l)) & 255u) | ((nibble & 7u) << 8)], xv[2 * l], xv[2 * l + 1], 0.0f);
                sums[l / 2] = fma((nibble & 8u) ? -1.125f : -0.875f, sx[l], fma(s, 16777216.0f, sums[l / 2]));
            }
            acc[r] += (d * float(2 * (word & 7u) + 1)) * sums[0] + (d * float(2 * ((word >> 3) & 7u) + 1)) * sums[1];
        }
    }
}

// Copies a codebook into threadgroup memory; every thread of the group must
// reach it (it holds a barrier).
template <uint N, uint SIMDGROUPS>
inline void nu_codebook_copy(threadgroup ulong * tg, constant ulong * table, uint sg, uint lane) {
    for (uint i = sg * 32 + lane; i < N; i += 32 * SIMDGROUPS) tg[i] = table[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// The standalone kernels of the bodies above.
#define NU_MATVEC_KERNEL(NAME) \
template <uint ROWS> \
kernel void nu_matvec_##NAME(device const uchar * weights [[buffer(0)]], device const float * input [[buffer(1)]], device float * output [[buffer(2)]], \
                             constant MatvecBlockParams & p [[buffer(7)]], uint group_index [[threadgroup_position_in_grid]], \
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) { \
    float acc[ROWS]; \
    nu_matvec_##NAME##_body<ROWS>(weights, input, p, group_index, sg, lane, acc); \
    if ((group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS < p.rows) \
        nu_store_rows<ROWS>(acc, output, (group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS, p.rows, lane); \
}
NU_MATVEC_KERNEL(iq2_xxs)
template <uint ROWS>
kernel void nu_matvec_iq2_xs(device const uchar * weights [[buffer(0)]], device const float * input [[buffer(1)]], device float * output [[buffer(2)]],
                             constant MatvecBlockParams & p [[buffer(7)]], uint group_index [[threadgroup_position_in_grid]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    float acc[ROWS];
    nu_matvec_iq2_xs_body<ROWS, constant ulong *>(weights, input, p, group_index, sg, lane, acc, nu_iq2_xs_grid);
    if ((group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS < p.rows)
        nu_store_rows<ROWS>(acc, output, (group_index * NU_MATVEC_SIMDGROUPS + sg) * ROWS, p.rows, lane);
}
// Sixteen SIMD groups share one grid copy (`matvecSimdgroups` in root.zig);
// the virtual group/sg pair keeps the body's four-group row arithmetic.
#define NU_IQ2_S_SIMDGROUPS 16
template <uint ROWS>
kernel void nu_matvec_iq2_s(device const uchar * weights [[buffer(0)]], device const float * input [[buffer(1)]], device float * output [[buffer(2)]],
                             constant MatvecBlockParams & p [[buffer(7)]], uint group_index [[threadgroup_position_in_grid]],
                             uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup ulong grid[1024];
    nu_codebook_copy<1024, NU_IQ2_S_SIMDGROUPS>(grid, nu_iq2_s_grid, sg, lane);
    uint global_sg = group_index * NU_IQ2_S_SIMDGROUPS + sg;
    float acc[ROWS];
    nu_matvec_iq2_s_body<ROWS, threadgroup const ulong *>(weights, input, p, global_sg / NU_MATVEC_SIMDGROUPS, global_sg % NU_MATVEC_SIMDGROUPS, lane, acc, grid);
    if (global_sg * ROWS < p.rows)
        nu_store_rows<ROWS>(acc, output, global_sg * ROWS, p.rows, lane);
}
NU_MATVEC_KERNEL(iq3_xxs)
NU_MATVEC_KERNEL(iq3_s)
NU_MATVEC_KERNEL(q2_k)
NU_MATVEC_KERNEL(iq1_s)
NU_MATVEC_KERNEL(iq1_m)

