// GGUF block decoders for the GPU, ported from nuclis's CPU reference
// (inference/src/quant/decode.zig). Every function decodes 16 consecutive
// values starting at element `first` (a multiple of 16) of one block, indexing
// individual little-endian bytes exactly like the Zig code, so the two can be
// compared bit for bit. Block layouts are facts of the GGML storage formats;
// see THIRD_PARTY_NOTICES.md for the provenance of the two lookup tables.
//
// Block sizes in bytes / elements per block:
//   F32 4/1  F16 2/1  Q8_0 34/32  IQ4_NL 18/32  Q4_0 18/32
//   Q3_K 110/256  Q4_K 144/256  Q5_K 176/256  Q6_K 210/256  IQ3_S 110/256  IQ4_XS 136/256

inline float nu_half(device const uchar * b) {
    ushort bits = ushort(b[0]) | (ushort(b[1]) << 8);
    return float(as_type<half>(bits));
}

inline void nu_dequant_f32(device const uchar * row, uint first, thread float * out) {
    device const float * x = (device const float *)row + first;
    for (uint j = 0; j < 16; ++j) out[j] = x[j];
}
inline void nu_dequant_f16(device const uchar * row, uint first, thread float * out) {
    device const half * x = (device const half *)row + first;
    for (uint j = 0; j < 16; ++j) out[j] = float(x[j]);
}

// Q8_0: half scale, then 32 signed bytes.
inline void nu_dequant_q8_0(device const uchar * b, uint first, thread float * out) {
    float d = nu_half(b);
    for (uint j = 0; j < 16; ++j) out[j] = d * float(char(b[2 + first + j]));
}

// IQ4_NL: half scale, then 16 bytes whose low nibbles are values 0..15 and high
// nibbles values 16..31. The 16 codes map to a fixed nonlinear table.
constant int nu_iq4_values[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };
// The same table as floats for the specialized matvec, which skips the int->float conversion.
constant float nu_iq4_values_f[16] = { -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113 };
inline void nu_dequant_iq4_nl(device const uchar * b, uint first, thread float * out) {
    float d = nu_half(b);
    for (uint j = 0; j < 16; ++j) {
        uchar byte = b[2 + j];
        out[j] = d * float(nu_iq4_values[first == 0 ? (byte & 15) : (byte >> 4)]);
    }
}

// Q4_0: IQ4_NL's block and nibble order with the code itself as the
// value, biased by eight: d * (q - 8).
inline void nu_dequant_q4_0(device const uchar * b, uint first, thread float * out) {
    float d = nu_half(b);
    for (uint j = 0; j < 16; ++j) {
        uchar byte = b[2 + j];
        out[j] = d * float(int(first == 0 ? (byte & 15) : (byte >> 4)) - 8);
    }
}

// Q4_K / Q5_K: d, dmin, twelve packed six-bit scale/min pairs for eight groups
// of 32, then nibbles (Q5_K adds a 32-byte fifth-bit plane before the nibbles).
inline void nu_k_scale_min(device const uchar * scales, uint group, thread uint & scale, thread uint & minimum) {
    if (group < 4) {
        scale = scales[group] & 63;
        minimum = scales[group + 4] & 63;
    } else {
        scale = (scales[group + 4] & 15) | ((scales[group - 4] >> 6) << 4);
        minimum = (scales[group + 4] >> 4) | ((scales[group] >> 6) << 4);
    }
}
inline void nu_dequant_k(device const uchar * b, uint first, bool fifth_bit, thread float * out) {
    float d = nu_half(b), dmin = nu_half(b + 2);
    device const uchar * low = b + (fifth_bit ? 48 : 16);
    uint group = first / 32;
    uint scale, minimum;
    nu_k_scale_min(b + 4, group, scale, minimum);
    float multiplier = d * float(scale);
    float offset = dmin * float(minimum);
    for (uint j = 0; j < 16; ++j) {
        uint column = (first % 32) + j;
        uchar byte = low[(group / 2) * 32 + column];
        uint value = (group % 2 == 0) ? (byte & 15) : (byte >> 4);
        if (fifth_bit) value |= ((b[16 + column] >> group) & 1) << 4;
        out[j] = multiplier * float(value) - offset;
    }
}
inline void nu_dequant_q4_k(device const uchar * b, uint first, thread float * out) { nu_dequant_k(b, first, false, out); }
inline void nu_dequant_q5_k(device const uchar * b, uint first, thread float * out) { nu_dequant_k(b, first, true, out); }

// Q3_K: 32-byte inverted sign mask, 64 bytes of two-bit codes, twelve bytes of
// packed six-bit biased scales for sixteen groups of 16, then d.
inline void nu_dequant_q3_k(device const uchar * b, uint first, thread float * out) {
    float d = nu_half(b + 108);
    device const uchar * scales = b + 96;
    uint group = first / 16;
    uint nibble = (group < 8) ? (scales[group] & 15) : (scales[group - 8] >> 4);
    uint upper = (scales[8 + group % 4] >> (2 * (group / 4))) & 3;
    int scale = int(nibble | (upper << 4)) - 32;
    float multiplier = d * float(scale);
    for (uint j = 0; j < 16; ++j) {
        uint i = first + j;
        uint low = (b[32 + (i / 128) * 32 + i % 32] >> (2 * ((i % 128) / 32))) & 3;
        uint nonnegative = (b[i % 32] >> (i / 32)) & 1;
        int value = int(low) - (nonnegative == 0 ? 4 : 0);
        out[j] = multiplier * float(value);
    }
}

// Q6_K: 128 bytes of low nibbles, 64 bytes of high two-bit pairs, sixteen
// signed byte scales for groups of 16, then d. Codes are biased by 32.
inline void nu_dequant_q6_k(device const uchar * b, uint first, thread float * out) {
    float d = nu_half(b + 208);
    float scale = d * float(char(b[192 + first / 16]));
    for (uint j = 0; j < 16; ++j) {
        uint i = first + j;
        uint quarter = (i % 128) / 32, column = i % 32;
        uint low = (b[(i / 128) * 64 + (quarter % 2) * 32 + column] >> ((quarter / 2) * 4)) & 15;
        uint high = (b[128 + (i / 128) * 32 + column] >> (quarter * 2)) & 3;
        int value = int(low | (high << 4)) - 32;
        out[j] = scale * float(value);
    }
}

// IQ4_XS: d, a 16-bit word of high scale bits, four bytes of low scale nibbles
// for eight groups of 32, then 128 bytes of IQ4_NL-style nibble pairs.
inline void nu_dequant_iq4_xs(device const uchar * b, uint first, thread float * out) {
    float d = nu_half(b);
    uint high = uint(b[2]) | (uint(b[3]) << 8);
    uint group = first / 32;
    uint low = (b[4 + group / 2] >> (4 * (group % 2))) & 15;
    int scale = int(low | (((high >> (2 * group)) & 3) << 4)) - 32;
    float multiplier = d * float(scale);
    bool upper_half = (first % 32) != 0;
    for (uint j = 0; j < 16; ++j) {
        uchar byte = b[8 + group * 16 + j];
        out[j] = multiplier * float(nu_iq4_values[upper_half ? (byte >> 4) : (byte & 15)]);
    }
}

// IQ3_S: d, 64 bytes of grid indices (eight groups of eight entries, each entry
// four magnitudes), eight bytes of ninth index bits, 32 bytes of per-value sign
// bits, then four bytes of odd scale codes. Component 0 is the grid's low byte.
inline void nu_dequant_iq3_s(device const uchar * b, uint first, thread float * out) {
    float d = nu_half(b);
    uint group = first / 32;
    uint scale = (b[106 + group / 2] >> (4 * (group % 2))) & 15;
    float multiplier = d * float(1 + 2 * scale);
    for (uint j = 0; j < 16; ++j) {
        uint i = first + j;
        uint entry = (i % 32) / 4, component = i % 4;
        uint index = uint(b[2 + group * 8 + entry]) | (((b[66 + group] >> entry) & 1) << 8);
        uint magnitude = (nu_iq3_grid[index] >> (8 * component)) & 255;
        uint negative = (b[74 + i / 8] >> (i % 8)) & 1;
        float value = multiplier * float(magnitude);
        out[j] = negative ? -value : value;
    }
}

// Block geometry per encoding id: (bytes per block, elements per block).
inline uint2 nu_block_geometry(uint encoding) {
    switch (encoding) {
        case 0: return uint2(4, 1);
        case 1: return uint2(2, 1);
        case 2: return uint2(18, 32);
        case 8: return uint2(34, 32);
        case 11: return uint2(110, 256);
        case 12: return uint2(144, 256);
        case 13: return uint2(176, 256);
        case 14: return uint2(210, 256);
        case 20: return uint2(18, 32);
        case 21: return uint2(110, 256);
        case 23: return uint2(136, 256);
        default: return uint2(0, 0);
    }
}

// Decodes elements [segment*16, segment*16+16) of one encoded row.
inline void nu_segment(device const uchar * row, uint encoding, uint segment, thread float * out) {
    uint element = segment * 16;
    if (encoding == 0) { nu_dequant_f32(row, element, out); return; }
    if (encoding == 1) { nu_dequant_f16(row, element, out); return; }
    uint2 g = nu_block_geometry(encoding);
    device const uchar * block = row + ulong(element / g.y) * g.x;
    uint first = element % g.y;
    switch (encoding) {
        case 2: nu_dequant_q4_0(block, first, out); break;
        case 8: nu_dequant_q8_0(block, first, out); break;
        case 11: nu_dequant_q3_k(block, first, out); break;
        case 12: nu_dequant_q4_k(block, first, out); break;
        case 13: nu_dequant_q5_k(block, first, out); break;
        case 14: nu_dequant_q6_k(block, first, out); break;
        case 20: nu_dequant_iq4_nl(block, first, out); break;
        case 21: nu_dequant_iq3_s(block, first, out); break;
        case 23: nu_dequant_iq4_xs(block, first, out); break;
        default: for (uint j = 0; j < 16; ++j) out[j] = 0; break;
    }
}
