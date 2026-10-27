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

