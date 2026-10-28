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
// Q5_0 / Q5_1 (OFFSET) segment: Q4_0's nibbles completed by their plane
// bits, `d * (q - 16)` or `d * q + m` in the generic decoder's operation
// order; 2-byte aligned blocks.
template <bool OFFSET>
inline void nu_tile_five(device const uchar * b, uint first, thread float4 * q) {
    const uint plane_at = OFFSET ? 4 : 2;
    device const ushort * h = (device const ushort *)b;
    float d = float(as_type<half>(h[0])), m = OFFSET ? float(as_type<half>(h[1])) : 0.0f;
    uint qh = uint(h[plane_at / 2]) | (uint(h[plane_at / 2 + 1]) << 16);
    packed_ushort4 l0 = *(device const packed_ushort4 *)(b + plane_at + 4), l1 = *(device const packed_ushort4 *)(b + plane_at + 12);
    uint w[4] = { nu_word(l0, 0), nu_word(l0, 1), nu_word(l1, 0), nu_word(l1, 1) };
    for (uint i = 0; i < 4; ++i) {
        float4 code = nu_bytes(((first ? w[i] >> 4 : w[i]) & 0x0f0f0f0fu) | nu_plane_bits(qh, first + 4 * i));
        q[i] = OFFSET ? d * code + m : d * (code - 16.0f);
    }
}
// Q8_0 segment: sixteen of the block's 32 signed bytes, `d * q`; 34-byte
// blocks, 2-byte aligned, so the bytes are read as ushort pairs.
inline void nu_tile_q8_0(device const uchar * b, uint first, thread float4 * q) {
    float d = float(as_type<half>(*(device const ushort *)b));
    device const ushort * w = (device const ushort *)(b + 2 + first);
    for (uint i = 0; i < 4; ++i) {
        ushort lo = w[2 * i], hi = w[2 * i + 1];
        q[i] = d * float4(float(as_type<char>(uchar(lo & 0xff))), float(as_type<char>(uchar(lo >> 8))), float(as_type<char>(uchar(hi & 0xff))), float(as_type<char>(uchar(hi >> 8))));
    }
}
// IQ4_NL segment: Q4_0's block and nibble order through the nonlinear
// table, `d * value` as the generic decoder computes it.
inline void nu_tile_iq4_nl(device const uchar * b, uint first, thread float4 * q) {
    float d = float(as_type<half>(*(device const ushort *)b));
    packed_ushort4 l0 = *(device const packed_ushort4 *)(b + 2), l1 = *(device const packed_ushort4 *)(b + 10);
    uint w0 = nu_word(l0, 0), w1 = nu_word(l0, 1), w2 = nu_word(l1, 0), w3 = nu_word(l1, 1);
    if (first) { q[0] = nu_iq4_high(w0); q[1] = nu_iq4_high(w1); q[2] = nu_iq4_high(w2); q[3] = nu_iq4_high(w3); }
    else { q[0] = nu_iq4_low(w0); q[1] = nu_iq4_low(w1); q[2] = nu_iq4_low(w2); q[3] = nu_iq4_low(w3); }
    for (uint i = 0; i < 4; ++i) q[i] = d * q[i];
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
// The codebook segments: two octets, each `multiplier * magnitude` negated
// by its sign bits, as the generic decoders compute them.
inline void nu_tile_octet(ulong grid, uint signs, float multiplier, thread float4 * q) {
    float4 lo = multiplier * nu_bytes(uint(grid)), hi = multiplier * nu_bytes(uint(grid >> 32));
    q[0] = select(lo, -lo, bool4(signs & 1u, signs & 2u, signs & 4u, signs & 8u));
    q[1] = select(hi, -hi, bool4(signs & 16u, signs & 32u, signs & 64u, signs & 128u));
}
inline void nu_tile_iq2_xxs(device const uchar * b, uint first, thread float4 * q) {
    uint group = first / 32, l = (first % 32) / 8;
    device const uchar * data = b + 2 + group * 8;
    uint word = uint(*(device const ushort *)(data + 4)) | (uint(*(device const ushort *)(data + 6)) << 16);
    float multiplier = nu_half(b) * ((0.5f + float(word >> 28)) * 0.25f);
    nu_tile_octet(nu_iq2_xxs_grid[data[l]], nu_signs7(word >> (7 * l)), multiplier, q);
    nu_tile_octet(nu_iq2_xxs_grid[data[l + 1]], nu_signs7(word >> (7 * l + 7)), multiplier, q + 2);
}
inline void nu_tile_iq2_xs(device const uchar * b, uint first, thread float4 * q) {
    uint group = first / 32, half16 = (first % 32) / 16;
    uint scale = half16 == 0 ? (b[66 + group] & 15u) : (b[66 + group] >> 4);
    float multiplier = nu_half(b) * ((0.5f + float(scale)) * 0.25f);
    device const ushort * codes = (device const ushort *)(b + 2 + 8 * group + 4 * half16);
    uint c0 = codes[0], c1 = codes[1];
    nu_tile_octet(nu_iq2_xs_grid[c0 & 511u], nu_signs7(c0 >> 9), multiplier, q);
    nu_tile_octet(nu_iq2_xs_grid[c1 & 511u], nu_signs7(c1 >> 9), multiplier, q + 2);
}
inline void nu_tile_iq2_s(device const uchar * b, uint first, thread float4 * q) {
    uint group = first / 32, l = (first % 32) / 8;
    uint scale = l < 2 ? (b[74 + group] & 15u) : (b[74 + group] >> 4);
    float multiplier = nu_half(b) * ((0.5f + float(scale)) * 0.25f);
    uint high = b[66 + group];
    for (uint h = 0; h < 2; ++h) {
        uint index = uint(b[2 + group * 4 + l + h]) | (((high >> (2 * (l + h))) & 3u) << 8);
        nu_tile_octet(nu_iq2_s_grid[index], b[34 + group * 4 + l + h], multiplier, q + 2 * h);
    }
}
inline void nu_tile_iq3_xxs(device const uchar * b, uint first, thread float4 * q) {
    uint group = first / 32, l = (first % 32) / 8;
    uint word = uint(*(device const ushort *)(b + 66 + group * 4)) | (uint(*(device const ushort *)(b + 68 + group * 4)) << 16);
    float multiplier = nu_half(b) * ((0.5f + float(word >> 28)) * 0.5f);
    for (uint h = 0; h < 2; ++h) {
        device const uchar * pair = b + 2 + group * 8 + 2 * (l + h);
        nu_tile_octet(ulong(nu_iq3_xxs_grid[pair[0]]) | (ulong(nu_iq3_xxs_grid[pair[1]]) << 32), nu_signs7(word >> (7 * (l + h))), multiplier, q + 2 * h);
    }
}
// Q2_K segment: sixteen code bytes at the same shift (four-byte aligned).
inline void nu_tile_q2_k(device const uchar * b, uint first, thread float4 * q) {
    uint group = first / 16, shift = 2 * ((first % 128) / 32);
    float multiplier = nu_half(b + 80) * float(b[group] & 15u), offset = nu_half(b + 82) * float(b[group] >> 4);
    device const uint * w = (device const uint *)(b + 16 + (first / 128) * 32 + first % 32);
    for (uint i = 0; i < 4; ++i) q[i] = multiplier * nu_bytes((w[i] >> shift) & 0x03030303u) - offset;
}
// IQ1 segments: two shifted ternary octets, `multiplier * (t + delta)`.
inline void nu_tile_shifted(uint index, bool negative, float multiplier, thread float4 * q) {
    ulong grid = nu_iq1_s_grid[index];
    float delta = negative ? -0.125f : 0.125f;
    q[0] = multiplier * (float4(as_type<char4>(uint(grid))) + delta);
    q[1] = multiplier * (float4(as_type<char4>(uint(grid >> 32))) + delta);
}
inline void nu_tile_iq1_s(device const uchar * b, uint first, thread float4 * q) {
    uint group = first / 32, l = (first % 32) / 8;
    uint word = *(device const ushort *)(b + 34 + 2 * group);
    float multiplier = nu_half(b) * float(2 * ((word >> 12) & 7u) + 1);
    for (uint h = 0; h < 2; ++h)
        nu_tile_shifted(uint(b[2 + group * 4 + l + h]) | (((word >> (3 * (l + h))) & 7u) << 8), (word & 0x8000u) != 0, multiplier, q + 2 * h);
}
inline void nu_tile_iq1_m(device const uchar * b, uint first, thread float4 * q) {
    device const ushort * w = (device const ushort *)(b + 48);
    uint w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3];
    float d = float(as_type<half>(ushort((w0 >> 12) | ((w1 >> 8) & 0x00f0u) | ((w2 >> 4) & 0x0f00u) | (w3 & 0xf000u))));
    uint group = first / 32, half16 = (first % 32) / 16;
    uint word = group < 2 ? w0 : group < 4 ? w1 : group < 6 ? w2 : w3;
    float multiplier = d * float(2 * ((word >> (6 * (group % 2) + 3 * half16)) & 7u) + 1);
    for (uint h = 0; h < 2; ++h) {
        uint l = half16 * 2 + h;
        uint nibble = (b[32 + group * 2 + l / 2] >> (4 * (l % 2))) & 15u;
        nu_tile_shifted(uint(b[group * 4 + l]) | ((nibble & 7u) << 8), (nibble & 8u) != 0, multiplier, q + 2 * h);
    }
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
    // 256-value blocks hold sixteen segments; the 32-value blocks (Q4_0, Q5_0,
    // Q5_1, Q8_0, IQ4_NL) two; the ternary 128-value blocks eight.
    const uint block_bytes = ENC == 8 ? 34 : (ENC == 2 || ENC == 20) ? 18 : ENC == 6 ? 22 : ENC == 7 ? 24 : ENC == 12 ? 144 : ENC == 13 ? 176 : ENC == 14 ? 210 : ENC == 23 ? 136 : ENC == 142 ? 34 : ENC == 143 ? 28
        : ENC == 10 ? 84 : ENC == 16 ? 66 : ENC == 17 ? 74 : ENC == 18 ? 98 : ENC == 19 ? 50 : ENC == 22 ? 82 : ENC == 29 ? 56 : 110;
    const uint segments_per_block = (ENC == 2 || ENC == 6 || ENC == 7 || ENC == 20 || ENC == 8) ? 2 : (ENC == 142 || ENC == 143) ? 8 : 16;
    device const uchar * b = row + ulong(segment / segments_per_block) * block_bytes;
    uint first = (segment % segments_per_block) * 16;
    switch (ENC) {
        case 2: nu_tile_q4_0(b, first, q); break;
        case 6: nu_tile_five<false>(b, first, q); break;
        case 7: nu_tile_five<true>(b, first, q); break;
        case 8: nu_tile_q8_0(b, first, q); break;
        case 20: nu_tile_iq4_nl(b, first, q); break;
        case 11: nu_tile_q3_k(b, first, q); break;
        case 12: nu_tile_k<false>(b, first, q); break;
        case 13: nu_tile_k<true>(b, first, q); break;
        case 14: nu_tile_q6_k(b, first, q); break;
        case 21: nu_tile_iq3_s(b, first, q); break;
        case 142: nu_tile_pq2_0(b, first, q); break;
        case 143: nu_tile_ptq1_0(b, first, q); break;
        case 10: nu_tile_q2_k(b, first, q); break;
        case 16: nu_tile_iq2_xxs(b, first, q); break;
        case 17: nu_tile_iq2_xs(b, first, q); break;
        case 18: nu_tile_iq3_xxs(b, first, q); break;
        case 19: nu_tile_iq1_s(b, first, q); break;
        case 22: nu_tile_iq2_s(b, first, q); break;
        case 29: nu_tile_iq1_m(b, first, q); break;
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
// Wide small-chunk tile: the same 128-thread group, K split, and reduction
// pattern as the 16×8 tile above, but 32 output rows per group. Each lane
// owns one row and decodes four 16-value segments into the group's 32×64
// half tile; four 8-row accumulator blocks share one B load per K step, so
// the gathered activation traffic per weight byte halves. The four K
// partials land at each group's region start and are summed by 128 threads
// over two entries each.
template <uint ENC, typename TW>
inline void nu_matmul_wide_body(device const uchar * weights, device const float * input, device float * output, MatmulParams p,
                                 uint row0, uint token0, threadgroup TW * tile, uint tid, uint sg) {
    const uint lane = tid & 31;
    threadgroup TW * own = tile + sg * 32 * 64;
    device const uchar * wrow = weights + ulong(min(row0 + lane, p.rows - 1)) * p.stride;
    simdgroup_float8x8 acc0 = simdgroup_float8x8(0.0f), acc1 = simdgroup_float8x8(0.0f);
    simdgroup_float8x8 acc2 = simdgroup_float8x8(0.0f), acc3 = simdgroup_float8x8(0.0f);
    const uint steps = p.columns / 64;
    for (uint step = sg; step < steps; step += 4) {
        simdgroup_barrier(mem_flags::mem_threadgroup);
        const uint k0 = step * 64;
        for (uint seg = 0; seg < 4; ++seg) {
            float4 q[4];
            nu_tile_segment<ENC>(wrow, p.encoding, k0 / 16 + seg, q);
            threadgroup vec<TW, 4> * slot = (threadgroup vec<TW, 4> *)(own + lane * 64 + seg * 16);
            for (uint c = 0; c < 4; ++c) slot[c] = vec<TW, 4>(q[c]);
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint k8 = 0; k8 < 64; k8 += 8) {
            simdgroup_matrix<TW, 8, 8> a0, a1, a2, a3;
            simdgroup_load(a0, own + k8, 64);
            simdgroup_load(a1, own + 8 * 64 + k8, 64);
            simdgroup_load(a2, own + 16 * 64 + k8, 64);
            simdgroup_load(a3, own + 24 * 64 + k8, 64);
            simdgroup_float8x8 b;
            simdgroup_load(b, input + ulong(token0) * p.in_stride + k0 + k8, p.in_stride, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc0, a0, b, acc0);
            simdgroup_multiply_accumulate(acc1, a1, b, acc1);
            simdgroup_multiply_accumulate(acc2, a2, b, acc2);
            simdgroup_multiply_accumulate(acc3, a3, b, acc3);
        }
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_store(acc0, (threadgroup float *)own, 8, ulong2(0, 0), false);
    simdgroup_store(acc1, (threadgroup float *)own + 64, 8, ulong2(0, 0), false);
    simdgroup_store(acc2, (threadgroup float *)own + 128, 8, ulong2(0, 0), false);
    simdgroup_store(acc3, (threadgroup float *)own + 192, 8, ulong2(0, 0), false);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    threadgroup float * part0 = (threadgroup float *)tile;
    threadgroup float * part1 = (threadgroup float *)(tile + 32 * 64);
    threadgroup float * part2 = (threadgroup float *)(tile + 2 * 32 * 64);
    threadgroup float * part3 = (threadgroup float *)(tile + 3 * 32 * 64);
    const uint t = tid & 7, r = tid >> 3;
    const float total0 = ((part0[tid] + part1[tid]) + part2[tid]) + part3[tid];
    const float total1 = ((part0[tid + 128] + part1[tid + 128]) + part2[tid + 128]) + part3[tid + 128];
    if (row0 + r < p.rows) output[ulong(token0 + t) * p.out_stride + row0 + r] = total0;
    if (row0 + 16 + r < p.rows) output[ulong(token0 + t) * p.out_stride + row0 + 16 + r] = total1;
}
template <uint ENC, typename TW>
kernel void nu_matmul_wide_t(device const uchar * weights [[buffer(0)]],
                             device const float * input [[buffer(1)]],
                             device float * output [[buffer(2)]],
                             constant MatmulParams & p [[buffer(7)]],
                             uint group [[threadgroup_position_in_grid]],
                             uint tid [[thread_position_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup TW tile[4 * 32 * 64];
    const uint row0 = (group % p.row_tiles) * 32, token0 = (group / p.row_tiles) * 8;
    nu_matmul_wide_body<ENC, TW>(weights, input, output, p, row0, token0, tile, tid, sg);
}
// Lane `lane` holds elements (row, col) and (row, col + 1) of an 8×8
// `simdgroup_matrix` through `thread_elements()`; `nu_fragment_layout`
// pins this in metal-check.
inline uint nu_fragment_row(uint lane) { return ((lane >> 2) & 4) + ((lane >> 1) & 3); }
inline uint nu_fragment_col(uint lane) { return (((lane >> 2) & 2) + (lane & 1)) * 2; }
kernel void nu_fragment_layout(device float * out [[buffer(0)]], uint lane [[thread_index_in_simdgroup]]) {
    simdgroup_float8x8 m;
    m.thread_elements()[0] = float(2 * lane);
    m.thread_elements()[1] = float(2 * lane + 1);
    simdgroup_store(m, out, 8);
}
// Register-fragment small-batch tile: 8·RB rows × 8 tokens per 128-thread
// group, the four SIMD groups splitting the 64-column K steps as the 16×8
// tile does, with nothing staged in threadgroup memory. The K order inside
// a step is permuted, identically for both operands so each product is
// unchanged: fragment j, column c = 2q + e holds step column 16q + 8e + j.
// A lane (fragment row r, quarter q) then needs segment q of its weight row
// (values j and 8 + j for fragment j), decoded straight into its elements,
// and step columns 16(r/2) + 8(r%2) + 0..7 of its two token rows, two
// float4 loads each. Token rows past `tokens` contribute zeros.
template <uint ENC, uint RB, typename TA, typename TB>
inline void nu_matmul_frag_body(device const uchar * weights, device const float * input, device float * output, MatmulParams p,
                                uint row0, uint token0, threadgroup float * partials, uint tid, uint sg) {
    const uint lane = tid & 31;
    const uint fr = nu_fragment_row(lane), fc = nu_fragment_col(lane), quarter = fc / 2;
    device const uchar * wrow[RB];
    for (uint i = 0; i < RB; ++i) wrow[i] = weights + ulong(min(row0 + 8 * i + fr, p.rows - 1)) * p.stride;
    const bool live0 = token0 + fc < p.tokens, live1 = token0 + fc + 1 < p.tokens;
    const uint x_at = 16 * (fr >> 1) + 8 * (fr & 1);
    device const float * x0 = input + ulong(token0 + fc) * p.in_stride + x_at;
    device const float * x1 = x0 + p.in_stride;
    simdgroup_float8x8 acc[RB];
    for (uint i = 0; i < RB; ++i) acc[i] = simdgroup_float8x8(0.0f);
    const uint steps = p.columns / 64;
    for (uint step = sg; step < steps; step += 4) {
        const uint k0 = step * 64;
        float4 u0 = 0.0f, u1 = 0.0f, v0 = 0.0f, v1 = 0.0f;
        if (live0) { u0 = *(device const float4 *)(x0 + k0); u1 = *(device const float4 *)(x0 + k0 + 4); }
        if (live1) { v0 = *(device const float4 *)(x1 + k0); v1 = *(device const float4 *)(x1 + k0 + 4); }
        #pragma unroll
        for (uint i = 0; i < RB; ++i) {
            float4 w[4];
            nu_tile_segment<ENC>(wrow[i], p.encoding, k0 / 16 + quarter, w);
            #pragma unroll
            for (uint j = 0; j < 8; ++j) {
                simdgroup_matrix<TB, 8, 8> b;
                b.thread_elements()[0] = TB(j < 4 ? u0[j & 3] : u1[j & 3]);
                b.thread_elements()[1] = TB(j < 4 ? v0[j & 3] : v1[j & 3]);
                simdgroup_matrix<TA, 8, 8> a;
                a.thread_elements()[0] = TA(w[j >> 2][j & 3]);
                a.thread_elements()[1] = TA(w[2 + (j >> 2)][j & 3]);
                simdgroup_multiply_accumulate(acc[i], a, b, acc[i]);
            }
        }
    }
    for (uint i = 0; i < RB; ++i) simdgroup_store(acc[i], partials + (sg * RB + i) * 64, 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Element (row 8i + r, token t) of SIMD group s's partial sits at
    // (s·RB + i)·64 + 8r + t; the four partials are summed in a fixed order.
    for (uint o = tid; o < 64 * RB; o += 128) {
        const float total = ((partials[o] + partials[64 * RB + o]) + partials[128 * RB + o]) + partials[192 * RB + o];
        const uint i = o / 64, r = (o / 8) & 7, t = o & 7;
        if (row0 + 8 * i + r < p.rows) output[ulong(token0 + t) * p.out_stride + row0 + 8 * i + r] = total;
    }
}
template <uint ENC, uint RB, typename TA, typename TB>
kernel void nu_matmul_frag_t(device const uchar * weights [[buffer(0)]],
                             device const float * input [[buffer(1)]],
                             device float * output [[buffer(2)]],
                             constant MatmulParams & p [[buffer(7)]],
                             uint group [[threadgroup_position_in_grid]],
                             uint tid [[thread_position_in_threadgroup]],
                             uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float partials[4 * RB * 64];
    const uint row0 = (group % p.row_tiles) * (8 * RB), token0 = (group / p.row_tiles) * 8;
    nu_matmul_frag_body<ENC, RB, TA, TB>(weights, input, output, p, row0, token0, partials, tid, sg);
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
template [[host_name("nu_matmul_q5_0")]] kernel void nu_matmul_t<6, 64, 64, half, half, true>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q5_1")]] kernel void nu_matmul_t<7, 64, 64, half, half, true>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q3_k_32")]] kernel void nu_matmul_t<11, 32, 32, half, float, false>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q4_k_32")]] kernel void nu_matmul_t<12, 32, 32, half, float, false>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q5_k_32")]] kernel void nu_matmul_t<13, 32, 32, half, float, false>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q6_k_32")]] kernel void nu_matmul_t<14, 32, 32, half, float, false>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_iq3_s_32")]] kernel void nu_matmul_t<21, 32, 32, half, float, false>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_iq4_xs_32")]] kernel void nu_matmul_t<23, 32, 32, half, float, false>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q4_0_32")]] kernel void nu_matmul_t<2, 32, 32, half, float, false>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q5_0_32")]] kernel void nu_matmul_t<6, 32, 32, half, float, false>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q5_1_32")]] kernel void nu_matmul_t<7, 32, 32, half, float, false>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_pq2_0")]] kernel void nu_matmul_t<142, 64, 64, half, half, true>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_ptq1_0")]] kernel void nu_matmul_t<143, 64, 64, half, half, true>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_pq2_0_32")]] kernel void nu_matmul_t<142, 32, 32, half, float, false>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_ptq1_0_32")]] kernel void nu_matmul_t<143, 32, 32, half, float, false>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q2_k")]] kernel void nu_matmul_t<10, 64, 64, half, half, true>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q2_k_32")]] kernel void nu_matmul_t<10, 32, 32, half, float, false>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_iq2_xxs")]] kernel void nu_matmul_t<16, 64, 64, half, half, true>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_iq2_xxs_32")]] kernel void nu_matmul_t<16, 32, 32, half, float, false>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_iq2_xs")]] kernel void nu_matmul_t<17, 64, 64, half, half, true>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_iq2_xs_32")]] kernel void nu_matmul_t<17, 32, 32, half, float, false>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_iq3_xxs")]] kernel void nu_matmul_t<18, 64, 64, half, half, true>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_iq3_xxs_32")]] kernel void nu_matmul_t<18, 32, 32, half, float, false>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_iq1_s")]] kernel void nu_matmul_t<19, 64, 64, half, half, true>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_iq1_s_32")]] kernel void nu_matmul_t<19, 32, 32, half, float, false>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_iq2_s")]] kernel void nu_matmul_t<22, 64, 64, half, half, true>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_iq2_s_32")]] kernel void nu_matmul_t<22, 32, 32, half, float, false>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_iq1_m")]] kernel void nu_matmul_t<29, 64, 64, half, half, true>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_iq1_m_32")]] kernel void nu_matmul_t<29, 32, 32, half, float, false>(NU_MATMUL_ARGS);
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
template [[host_name("nu_matmul_q5_0_8")]] kernel void nu_matmul_split_t<6, half>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q5_1_8")]] kernel void nu_matmul_split_t<7, half>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_pq2_0_8")]] kernel void nu_matmul_split_t<142, half>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_ptq1_0_8")]] kernel void nu_matmul_split_t<143, half>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q2_k_8")]] kernel void nu_matmul_split_t<10, half>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_iq2_xxs_8")]] kernel void nu_matmul_split_t<16, half>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_iq2_xs_8")]] kernel void nu_matmul_split_t<17, half>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_iq3_xxs_8")]] kernel void nu_matmul_split_t<18, half>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_iq1_s_8")]] kernel void nu_matmul_split_t<19, half>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_iq2_s_8")]] kernel void nu_matmul_split_t<22, half>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_iq1_m_8")]] kernel void nu_matmul_split_t<29, half>(NU_MATMUL_ARGS);
// The wide (`_w8`) set: the same 8-token, at most `small_chunk_tokens`-row
// regime with 32 output rows per group (the production `matmul`
// does not route to it, `matmulTile32` measures it).
template [[host_name("nu_matmul_q3_k_w8")]] kernel void nu_matmul_wide_t<11, half>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q4_k_w8")]] kernel void nu_matmul_wide_t<12, half>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q5_k_w8")]] kernel void nu_matmul_wide_t<13, half>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q6_k_w8")]] kernel void nu_matmul_wide_t<14, half>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_iq3_s_w8")]] kernel void nu_matmul_wide_t<21, half>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_iq4_xs_w8")]] kernel void nu_matmul_wide_t<23, half>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q4_0_w8")]] kernel void nu_matmul_wide_t<2, half>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_pq2_0_w8")]] kernel void nu_matmul_wide_t<142, half>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_ptq1_0_w8")]] kernel void nu_matmul_wide_t<143, half>(NU_MATMUL_ARGS);
// Register-fragment tiles (`_f<RB>`: 8·RB rows × 8 tokens; half weights,
// F32 activations, half both with `hh`). `matmul` routes the `_f2` set
// (`specializedMatmulFrag`); `_f2hh` and `_f4` are measured variants.
template [[host_name("nu_matmul_q4_k_f2")]] kernel void nu_matmul_frag_t<12, 2, half, float>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q4_k_f2hh")]] kernel void nu_matmul_frag_t<12, 2, half, half>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q4_k_f4")]] kernel void nu_matmul_frag_t<12, 4, half, float>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q5_k_f2")]] kernel void nu_matmul_frag_t<13, 2, half, float>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q6_k_f2")]] kernel void nu_matmul_frag_t<14, 2, half, float>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_iq4_xs_f2")]] kernel void nu_matmul_frag_t<23, 2, half, float>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q3_k_f2")]] kernel void nu_matmul_frag_t<11, 2, half, float>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_iq3_s_f2")]] kernel void nu_matmul_frag_t<21, 2, half, float>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q4_0_f2")]] kernel void nu_matmul_frag_t<2, 2, half, float>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q5_0_f2")]] kernel void nu_matmul_frag_t<6, 2, half, float>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q5_1_f2")]] kernel void nu_matmul_frag_t<7, 2, half, float>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_pq2_0_f2")]] kernel void nu_matmul_frag_t<142, 2, half, float>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_ptq1_0_f2")]] kernel void nu_matmul_frag_t<143, 2, half, float>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_iq4_nl_f2")]] kernel void nu_matmul_frag_t<20, 2, half, float>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_q2_k_f2")]] kernel void nu_matmul_frag_t<10, 2, half, float>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_iq2_xxs_f2")]] kernel void nu_matmul_frag_t<16, 2, half, float>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_iq2_xs_f2")]] kernel void nu_matmul_frag_t<17, 2, half, float>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_iq3_xxs_f2")]] kernel void nu_matmul_frag_t<18, 2, half, float>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_iq1_s_f2")]] kernel void nu_matmul_frag_t<19, 2, half, float>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_iq2_s_f2")]] kernel void nu_matmul_frag_t<22, 2, half, float>(NU_MATMUL_ARGS);
template [[host_name("nu_matmul_iq1_m_f2")]] kernel void nu_matmul_frag_t<29, 2, half, float>(NU_MATMUL_ARGS);
// Q8_0's tile is called by name (the DFlash2 drafter), never routed by `matmul`.
template [[host_name("nu_matmul_q8_0_f2")]] kernel void nu_matmul_frag_t<8, 2, half, float>(NU_MATMUL_ARGS);
// Gathered expert tiles: 64 rows × 32 slot rows with half operands for Q4_0,
// Q5_0, and Q5_1 (an expert averages k · chunk / experts slot rows per chunk,
// 16 on the 26B-A4B at 256 tokens, so one token tile covers most experts), and
// the generic F32 32 × 32 tile for every other encoding or alignment.
#define NU_MATMUL_EXPERTS_ARGS device const uchar *, device const float *, device float *, device const uint *, constant MatmulExpertsParams &, uint, uint, uint
template [[host_name("nu_matmul_experts")]] kernel void nu_matmul_experts_t<NU_TILE_GENERIC, 32, float, float>(NU_MATMUL_EXPERTS_ARGS);
template [[host_name("nu_matmul_experts_q4_0")]] kernel void nu_matmul_experts_t<2, 64, half, half>(NU_MATMUL_EXPERTS_ARGS);
template [[host_name("nu_matmul_experts_q5_0")]] kernel void nu_matmul_experts_t<6, 64, half, half>(NU_MATMUL_EXPERTS_ARGS);
template [[host_name("nu_matmul_experts_q5_1")]] kernel void nu_matmul_experts_t<7, 64, half, half>(NU_MATMUL_EXPERTS_ARGS);

