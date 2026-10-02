# The nuclis API

(Draft while APPS-19 is open; the client reference is written at its close.)

## Measured rates

Apple M4 Pro (48 GB), ReleaseFast, Zig 0.16.0, macOS 27.0; `nuclis serve
--model laya --model laya-multilingual` on Metal; ApacheBench 2.3 with
keep-alive (`ab -k`) on loopback; checkpoints `convaiinnovations/laya` at
`55cf4c4e`. The decision request is one state (93 characters, a billing
complaint) and two questions (a two-option choice and a noul),
`.zig-cache/apps19/two.json` in the session; latencies are ab's
percentiles.

### One GPU job per request (before batching), 2026-10-02, at `8c5d2e6` + the API layer

| Route | Model | Concurrency | Requests | Requests/s | p50 ms | p99 ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `GET /v1/health` | — | 1 | 1,000 | 25,905 | 0.019 | 0.082 |
| `GET /v1/health` | — | 16 | 20,000 | 146,801–166,589 | 0.096 | 0.187 |
| `POST /v1/decisions` | laya | 1 | 512 | 28.8 | 34.4 | 36.4 |
| `POST /v1/decisions` | laya | 4 | 512 | 28.9 | 136.4 | 153.0 |
| `POST /v1/decisions` | laya | 16 | 512 | 29.4 | 543.2 | 544.2 |
| `POST /v1/decisions` | laya-multilingual | 1 | 512 | 69.2 | 14.5 | 14.9 |
| `POST /v1/decisions` | laya-multilingual | 4 | 512 | 71.1 | 55.6 | 58.5 |
| `POST /v1/decisions` | laya-multilingual | 16 | 512 | 72.0 | 222.2 | 222.9 |

What the server adds: a warm request's HTTP p50 against the encode the
response reports (median of 40, `timings_ms.encode`), tokenizing 0.1 ms:
laya 34.05 ms against 34.5 ms, laya-multilingual 13.97 ms against
14.3 ms, so nothing measurable (the targets were within 2 ms). The same
request through `nuclis decide --request … --json`, a process per call:
0.09–0.10 s (laya), 0.20–0.21 s (laya-multilingual), wall clock.

### Batched across requests, 2026-10-02

Same machine, build mode, request, and protocol:

| Model | Concurrency | Requests/s before → after | p50 ms before → after | p99 ms before → after |
| --- | ---: | --- | --- | --- |
| laya | 1 | 28.8 → 29.2 | 34.4 → 34.2 | 36.4 → 34.7 |
| laya | 4 | 28.9 → 33.8 | 136.4 → 118.3 | 153.0 → 118.9 |
| laya | 16 | 29.4 → 36.4 (1.24×) | 543.2 → 439.3 | 544.2 → 444.3 |
| laya-multilingual | 1 | 69.2 → 71.2 | 14.5 → 14.0 | 14.9 → 14.5 |
| laya-multilingual | 4 | 71.1 → 85.1 | 55.6 → 46.9 | 58.5 → 47.5 |
| laya-multilingual | 16 | 72.0 → 93.4 (1.30×) | 222.2 → 171.0 | 222.9 → 172.1 |

At concurrency 16 a pass held 7.9 requests on average (512 in 65 passes,
`GET /v1/health` counters), about 820 rows. The gain is the model's: one
request already encodes at 3,020 rows/s (laya, 104 rows in 34.4 ms) and
7,160 (laya-multilingual, 101 in 14.1 ms), against the packed rates
[laya.md § Time per call](laya.md#time-per-call) measures, about 3,700
and 8,660, so packing can add about 1.2× here, and does. The planned 2×
would need a faster Laya encode, not a different server. p99 is about two
passes (laya 444 ms against 2 × 217 ms, laya-multilingual 172 against
2 × 85): a request that just missed a pass waits for it and runs in the
next.

Batching changes timing, never answers: 16 concurrent requests with
different questions (2 states each, choice, noul, and score questions)
returned the same bytes as each sent alone, timings aside, on both
checkpoints on Metal (16 in 3 and in 4 passes), and the unit test
asserts the same on the CPU.
