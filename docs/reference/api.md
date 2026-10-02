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
