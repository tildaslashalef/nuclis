// ---------------------------------------------------------------------------
// Multi-row matvec: the same per-lane decode as the matvec above, but each
// weight block is decoded once and multiplied against `tokens ≤ NU_ROWS_MAX`
// activation rows, so the weight bytes are read once per batch instead of once
// per token. That is the floor for a small batch (2–8 rows): one weight pass,
// where the 16×8 split-K tile reads them once per token tile. Each 8-lane
// group owns ROWS rows (below); the accumulator is `acc[row][token]`. Q4_0
// and the ternary encodings keep the generic path, which the sweep in
// metal-check measures.
struct MatvecRowsParams { uint columns; uint encoding; uint stride; uint rows; uint tokens; uint in_stride; uint out_stride; };
#define NU_MATVEC_ROWS_MAX 8

// The token count is a template parameter, not a runtime bound: each lane holds
// `ROWS × TOKENS` accumulators, and only a compile-time bound lets the compiler
// keep them in registers. A runtime `tokens` loop with a `break` spills them to
// thread-local memory, which is what made the first body fall to 4 GB/s at 8
// rows.
//
// Each 8-lane group of a SIMD group owns ROWS rows and the four groups walk
// the same column block together, so one input load serves 4·ROWS rows.
// Rows past the matrix are read clamped and never stored.
template <uint ROWS>
inline uint nu_rows_lane_row0(uint group_index, uint sg, uint lane) { return ((group_index * NU_MATVEC_SIMDGROUPS + sg) * 4 + (lane >> 3)) * ROWS; }

// The loop runs word-outer: for each 4-byte word of a lane's slice, every
// row's eight codes are dequantized once (scale and bias folded in), then each
// token's eight inputs are loaded once and dotted into every row. Decode is
// shared by the tokens and input loads by the rows; the live set is the
// accumulators plus 8·ROWS decoded values. The word loop stays rolled and
// reads its word from memory: unrolled, the nest grows past what the compiler
// unrolls, and a run-time index into `acc` compiles to select chains.
//
// Q4_K / Q5_K: the high nibbles come out at ×16, so their scale carries 1/16.
template <uint ROWS, uint TOKENS, bool FIFTH_BIT>
inline void nu_matvec_rows_k_body(device const uchar * weights, device const float * input, MatvecRowsParams p, uint group_index, uint sg, uint lane, thread float * acc) {
    const uint block_bytes = FIFTH_BIT ? 176 : 144;
    const uint nibble_offset = FIFTH_BIT ? 48 : 16;
    uint row0 = nu_rows_lane_row0<ROWS>(group_index, sg, lane);
    uint pair = (lane & 7) >> 1, half_index = lane & 1;
    uint bit_a = 2 * pair, bit_b = bit_a + 1;
    for (uint i = 0; i < ROWS * TOKENS; ++i) acc[i] = 0;
    const uint blocks = p.columns / 256;
    for (uint kb = 0; kb < blocks; ++kb) {
        uint slice = kb * block_bytes + nibble_offset + pair * 32 + half_index * 16;
        uint plane = kb * block_bytes + 16 + half_index * 16;
        device const uchar * rows[ROWS];
        float da[ROWS], ma[ROWS], db[ROWS], mb[ROWS];
#pragma unroll
        for (uint r = 0; r < ROWS; ++r) {
            device const uchar * row = weights + ulong(min(row0 + r, p.rows - 1)) * p.stride;
            rows[r] = row;
            uint dd = *(device const uint *)(row + kb * block_bytes);
            packed_uint3 s = *(device const packed_uint3 *)(row + kb * block_bytes + 4);
            float d = nu_half_low(dd), dmin = nu_half_high(dd);
            float sa, sm, sb, smb;
            nu_k_scales_pair(s, pair, sa, sm, sb, smb);
            da[r] = d * sa; ma[r] = -dmin * sm; db[r] = d * sb * (1.0f / 16.0f); mb[r] = -dmin * smb;
        }
        device const float * x = input + kb * 256 + pair * 64 + half_index * 16;
#pragma clang loop unroll(disable)
        for (uint i = 0; i < 4; ++i) {
            float4 wa[ROWS], wb[ROWS];
#pragma unroll
            for (uint r = 0; r < ROWS; ++r) {
                uint v = *(device const uint *)(rows[r] + slice + 4 * i);
                uint h = FIFTH_BIT ? *(device const uint *)(rows[r] + plane + 4 * i) : 0u;
                float4 lo, hi;
                nu_magic_word<FIFTH_BIT>(v, h, bit_a, bit_b, lo, hi);
                wa[r] = fma(lo, float4(da[r]), float4(ma[r]));
                wb[r] = fma(hi, float4(db[r]), float4(mb[r]));
            }
#pragma unroll
            for (uint t = 0; t < TOKENS; ++t) {
                device const float * xt = x + ulong(t) * p.in_stride + 4 * i;
                float4 xa = *(device const float4 *)xt, xb = *(device const float4 *)(xt + 32);
#pragma unroll
                for (uint r = 0; r < ROWS; ++r) acc[r * TOKENS + t] = nu_dot(wb[r], xb, nu_dot(wa[r], xa, acc[r * TOKENS + t]));
            }
        }
    }
}

// Q6_K: four groups of eight codes per lane, each with its own signed scale;
// the bias of 32 folds into the dequantized value.
template <uint ROWS, uint TOKENS>
inline void nu_matvec_rows_q6_k_body(device const uchar * weights, device const float * input, MatvecRowsParams p, uint group_index, uint sg, uint lane, thread float * acc) {
    uint row0 = nu_rows_lane_row0<ROWS>(group_index, sg, lane);
    uint h = (lane & 7) >> 2, c8 = lane & 3, scale_byte = c8 >> 1;
    for (uint i = 0; i < ROWS * TOKENS; ++i) acc[i] = 0;
    const uint blocks = p.columns / 256;
    for (uint kb = 0; kb < blocks; ++kb) {
        uint base = kb * 210;
        device const uchar * blocks_at[ROWS];
        uint hba[ROWS], hbb[ROWS];
        float d[ROWS];
#pragma unroll
        for (uint r = 0; r < ROWS; ++r) {
            device const uchar * b = weights + ulong(min(row0 + r, p.rows - 1)) * p.stride + base;
            blocks_at[r] = b;
            packed_ushort4 hb = *(device const packed_ushort4 *)(b + 128 + h * 32 + c8 * 8);
            hba[r] = nu_word(hb, 0); hbb[r] = nu_word(hb, 1);
            d[r] = float(as_type<half>(*(device const ushort *)(b + 208)));
        }
        device const float * x = input + kb * 256 + h * 128 + c8 * 8;
#pragma clang loop unroll(disable)
        for (uint k = 0; k < 4; ++k) {
            float4 wa[ROWS], wb[ROWS];
#pragma unroll
            for (uint r = 0; r < ROWS; ++r) {
                device const uchar * b = blocks_at[r];
                packed_ushort4 lw = *(device const packed_ushort4 *)(b + h * 64 + (k & 1) * 32 + c8 * 8);
                uint2 l = uint2(nu_word(lw, 0), nu_word(lw, 1));
                float dc = d[r] * nu_signed_byte(uint(*(device const ushort *)(b + 192 + h * 8 + 2 * k)), scale_byte);
                float4 qa = k < 2 ? nu_low_sixes(l.x, hba[r], 2 * k) : nu_high_sixes(l.x, hba[r], 2 * k);
                float4 qb = k < 2 ? nu_low_sixes(l.y, hbb[r], 2 * k) : nu_high_sixes(l.y, hbb[r], 2 * k);
                wa[r] = fma(qa, float4(dc), float4(-32.0f * dc));
                wb[r] = fma(qb, float4(dc), float4(-32.0f * dc));
            }
#pragma unroll
            for (uint t = 0; t < TOKENS; ++t) {
                device const float4 * xt = (device const float4 *)(x + ulong(t) * p.in_stride) + 8 * k;
                float4 xa = xt[0], xb = xt[1];
#pragma unroll
                for (uint r = 0; r < ROWS; ++r) acc[r * TOKENS + t] = nu_dot(wb[r], xb, nu_dot(wa[r], xa, acc[r * TOKENS + t]));
            }
        }
    }
}

// IQ4_XS: word j of the lane's 16 code bytes pairs with inputs j (low
// nibbles) and j + 4 (high), through the threadgroup copy of the table.
template <uint ROWS, uint TOKENS>
inline void nu_matvec_rows_iq4_xs_body(device const uchar * weights, device const float * input, MatvecRowsParams p, uint group_index, uint sg, uint lane, thread float * acc, threadgroup const float * tv) {
    uint row0 = nu_rows_lane_row0<ROWS>(group_index, sg, lane);
    uint g = lane & 7;
    for (uint i = 0; i < ROWS * TOKENS; ++i) acc[i] = 0;
    const uint blocks = p.columns / 256;
    for (uint kb = 0; kb < blocks; ++kb) {
        uint base = kb * 136;
        device const uchar * codes[ROWS];
        float ds[ROWS];
#pragma unroll
        for (uint r = 0; r < ROWS; ++r) {
            device const uchar * b = weights + ulong(min(row0 + r, p.rows - 1)) * p.stride + base;
            uint2 header = *(device const uint2 *)b;
            codes[r] = b + 8 + g * 16;
            uint low = (header.y >> (4 * g)) & 15u, high = (header.x >> (16 + 2 * g)) & 3u;
            ds[r] = nu_half_low(header.x) * float(int(low | (high << 4)) - 32);
        }
        device const float * x = input + kb * 256 + g * 32;
#pragma clang loop unroll(disable)
        for (uint j = 0; j < 4; ++j) {
            float4 wa[ROWS], wb[ROWS];
#pragma unroll
            for (uint r = 0; r < ROWS; ++r) {
                uint q = *(device const uint *)(codes[r] + 4 * j);
                wa[r] = nu_iq4_tg(tv, q) * ds[r];
                wb[r] = nu_iq4_tg(tv, q >> 4) * ds[r];
            }
#pragma unroll
            for (uint t = 0; t < TOKENS; ++t) {
                device const float4 * xt = (device const float4 *)(x + ulong(t) * p.in_stride);
                float4 xa = xt[j], xb = xt[j + 4];
#pragma unroll
                for (uint r = 0; r < ROWS; ++r) acc[r * TOKENS + t] = nu_dot(wb[r], xb, nu_dot(wa[r], xa, acc[r * TOKENS + t]));
            }
        }
    }
}

// IQ2_XXS: lane `g` owns its block's 32-value group, as in the matvec. Each
// octet is decoded once per row with the group's scale folded in, then dotted
// with every token's eight inputs. The grid values are half denormals at
// 2^-24 of the byte, hence the 2^24 in the scale.
template <uint ROWS, uint TOKENS>
inline void nu_matvec_rows_iq2_xxs_body(device const uchar * weights, device const float * input, MatvecRowsParams p, uint group_index, uint sg, uint lane, thread float * acc) {
    uint row0 = nu_rows_lane_row0<ROWS>(group_index, sg, lane);
    uint g = lane & 7;
    for (uint i = 0; i < ROWS * TOKENS; ++i) acc[i] = 0;
    const uint blocks = p.columns / 256;
    for (uint kb = 0; kb < blocks; ++kb) {
        uint indices[ROWS], words[ROWS];
        float scale[ROWS];
#pragma unroll
        for (uint r = 0; r < ROWS; ++r) {
            device const uchar * b = weights + ulong(min(row0 + r, p.rows - 1)) * p.stride + kb * 66;
            packed_ushort4 q = *(device const packed_ushort4 *)(b + 2 + g * 8);
            indices[r] = nu_word(q, 0);
            words[r] = nu_word(q, 1);
            float d = float(as_type<half>(*(device const ushort *)b));
            scale[r] = (d * ((0.5f + float(words[r] >> 28)) * 0.25f)) * 16777216.0f;
        }
        device const float * x = input + kb * 256 + g * 32;
#pragma clang loop unroll(disable)
        for (uint l = 0; l < 4; ++l) {
            float4 wa[ROWS], wb[ROWS];
#pragma unroll
            for (uint r = 0; r < ROWS; ++r) {
                NuOctet o = nu_signed_octet(nu_iq2_xxs_grid[(indices[r] >> (8 * l)) & 255u], nu_signs7(words[r] >> (7 * l)));
                wa[r] = float4(float(o.v02.x), float(o.v13.x), float(o.v02.y), float(o.v13.y)) * scale[r];
                wb[r] = float4(float(o.v46.x), float(o.v57.x), float(o.v46.y), float(o.v57.y)) * scale[r];
            }
#pragma unroll
            for (uint t = 0; t < TOKENS; ++t) {
                device const float4 * xt = (device const float4 *)(x + ulong(t) * p.in_stride) + 2 * l;
                float4 xa = xt[0], xb = xt[1];
#pragma unroll
                for (uint r = 0; r < ROWS; ++r) acc[r * TOKENS + t] = nu_dot(wb[r], xb, nu_dot(wa[r], xa, acc[r * TOKENS + t]));
            }
        }
    }
}

// Sums each accumulator over its 8-lane group; the group's first lane stores.
template <uint ROWS, uint TOKENS>
inline void nu_store_rows_multi(thread float * acc, device float * output, uint row0, uint rows, uint out_stride, uint lane) {
    for (uint t = 0; t < TOKENS; ++t)
        for (uint r = 0; r < ROWS; ++r) {
            float total = acc[r * TOKENS + t];
            total += simd_shuffle_xor(total, 1);
            total += simd_shuffle_xor(total, 2);
            total += simd_shuffle_xor(total, 4);
            if ((lane & 7) == 0 && row0 + r < rows) output[ulong(t) * out_stride + row0 + r] = total;
        }
}

template <uint ROWS, uint TOKENS>
kernel void nu_matvec_rows_q4_k(device const uchar * weights [[buffer(0)]], device const float * input [[buffer(1)]], device float * output [[buffer(2)]],
                                constant MatvecRowsParams & p [[buffer(7)]], uint group_index [[threadgroup_position_in_grid]],
                                uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    float acc[ROWS * TOKENS];
    nu_matvec_rows_k_body<ROWS, TOKENS, false>(weights, input, p, group_index, sg, lane, acc);
    nu_store_rows_multi<ROWS, TOKENS>(acc, output, nu_rows_lane_row0<ROWS>(group_index, sg, lane), p.rows, p.out_stride, lane);
}
template <uint ROWS, uint TOKENS>
kernel void nu_matvec_rows_q5_k(device const uchar * weights [[buffer(0)]], device const float * input [[buffer(1)]], device float * output [[buffer(2)]],
                                constant MatvecRowsParams & p [[buffer(7)]], uint group_index [[threadgroup_position_in_grid]],
                                uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    float acc[ROWS * TOKENS];
    nu_matvec_rows_k_body<ROWS, TOKENS, true>(weights, input, p, group_index, sg, lane, acc);
    nu_store_rows_multi<ROWS, TOKENS>(acc, output, nu_rows_lane_row0<ROWS>(group_index, sg, lane), p.rows, p.out_stride, lane);
}
template <uint ROWS, uint TOKENS>
kernel void nu_matvec_rows_q6_k(device const uchar * weights [[buffer(0)]], device const float * input [[buffer(1)]], device float * output [[buffer(2)]],
                                constant MatvecRowsParams & p [[buffer(7)]], uint group_index [[threadgroup_position_in_grid]],
                                uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    float acc[ROWS * TOKENS];
    nu_matvec_rows_q6_k_body<ROWS, TOKENS>(weights, input, p, group_index, sg, lane, acc);
    nu_store_rows_multi<ROWS, TOKENS>(acc, output, nu_rows_lane_row0<ROWS>(group_index, sg, lane), p.rows, p.out_stride, lane);
}
template <uint ROWS, uint TOKENS>
kernel void nu_matvec_rows_iq4_xs(device const uchar * weights [[buffer(0)]], device const float * input [[buffer(1)]], device float * output [[buffer(2)]],
                                  constant MatvecRowsParams & p [[buffer(7)]], uint group_index [[threadgroup_position_in_grid]],
                                  uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float tv[16];
    nu_iq4_table(tv, sg, lane);
    float acc[ROWS * TOKENS];
    nu_matvec_rows_iq4_xs_body<ROWS, TOKENS>(weights, input, p, group_index, sg, lane, acc, tv);
    nu_store_rows_multi<ROWS, TOKENS>(acc, output, nu_rows_lane_row0<ROWS>(group_index, sg, lane), p.rows, p.out_stride, lane);
}

template <uint ROWS, uint TOKENS>
kernel void nu_matvec_rows_iq2_xxs(device const uchar * weights [[buffer(0)]], device const float * input [[buffer(1)]], device float * output [[buffer(2)]],
                                   constant MatvecRowsParams & p [[buffer(7)]], uint group_index [[threadgroup_position_in_grid]],
                                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    float acc[ROWS * TOKENS];
    nu_matvec_rows_iq2_xxs_body<ROWS, TOKENS>(weights, input, p, group_index, sg, lane, acc);
    nu_store_rows_multi<ROWS, TOKENS>(acc, output, nu_rows_lane_row0<ROWS>(group_index, sg, lane), p.rows, p.out_stride, lane);
}

// Generic fallback for encodings without a specialized multi-row body
// (Q3_K, IQ3_S, IQ4_NL, Q8_0, F32, ...): one SIMD group per output row, the
// same segment decode as `nu_matvec`, one accumulator per token row.
kernel void nu_matvec_rows(device const uchar * weights [[buffer(0)]],
                           device const float * input [[buffer(1)]],
                           device float * output [[buffer(2)]],
                           constant MatvecRowsParams & p [[buffer(7)]],
                           uint row [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_simdgroup]]) {
    if (row >= p.rows) return;
    device const uchar * bytes = weights + ulong(row) * p.stride;
    float acc[NU_MATVEC_ROWS_MAX];
    for (uint t = 0; t < NU_MATVEC_ROWS_MAX; ++t) acc[t] = 0;
    float values[16];
    for (uint segment = lane; segment < p.columns / 16; segment += 32) {
        nu_segment(bytes, p.encoding, segment, values);
        for (uint t = 0; t < NU_MATVEC_ROWS_MAX; ++t) {
            if (t >= p.tokens) break;
            device const float * x = input + ulong(t) * p.in_stride + segment * 16;
            for (uint i = 0; i < 16; ++i) acc[t] += values[i] * x[i];
        }
    }
    for (uint t = 0; t < NU_MATVEC_ROWS_MAX; ++t) {
        if (t >= p.tokens) break;
        float total = simd_sum(acc[t]);
        if (lane == 0) output[ulong(t) * p.out_stride + row] = total;
    }
}

// One instantiation per token count so the accumulator loops are compile-time
// bound; `matvec_rows_max` in root.zig is the largest count. ROWS must match
// `rows_per_simdgroup` there.
#define NU_ROWS_PER_SIMDGROUP 2 // rows per 8-lane group; four groups give matvec_rows_per_simdgroup in root.zig
#define NU_MATVEC_ROWS_INSTANCE(NAME, TOK) \
    template [[host_name("nu_matvec_rows_" #NAME "_t" #TOK)]] kernel void nu_matvec_rows_##NAME<NU_ROWS_PER_SIMDGROUP, TOK>(device const uchar *, device const float *, device float *, constant MatvecRowsParams &, uint, uint, uint);
#define NU_MATVEC_ROWS_ALL(NAME) \
    NU_MATVEC_ROWS_INSTANCE(NAME, 2) NU_MATVEC_ROWS_INSTANCE(NAME, 3) NU_MATVEC_ROWS_INSTANCE(NAME, 4) NU_MATVEC_ROWS_INSTANCE(NAME, 5) \
    NU_MATVEC_ROWS_INSTANCE(NAME, 6) NU_MATVEC_ROWS_INSTANCE(NAME, 7) NU_MATVEC_ROWS_INSTANCE(NAME, 8)
NU_MATVEC_ROWS_ALL(q4_k)
NU_MATVEC_ROWS_ALL(q5_k)
NU_MATVEC_ROWS_ALL(q6_k)
NU_MATVEC_ROWS_ALL(iq4_xs)
NU_MATVEC_ROWS_ALL(iq2_xxs)

