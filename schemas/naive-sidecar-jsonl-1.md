# Naive sidecar JSONL

One JSON object per line. A producer may emit `span` and `metric` records.

```json
{"type":"span","span_id":1,"name":"infinilm:queue_wait","window_id":"window-0","start_ns":1730000000000000000,"end_ns":1730000000000250000,"tid":123,"rank":0,"iteration":17,"queue_depth":3}
{"type":"span","span_id":2,"name":"infinilm:admission","window_id":"window-0","duration_us":120,"rank":0,"iteration":17}
{"type":"span","span_id":3,"name":"infinilm:batch_build","window_id":"window-0","duration_us":180,"rank":0,"iteration":17,"batch_size":8,"tokens_per_iteration":32}
{"type":"span","span_id":4,"name":"infinilm:schedule","window_id":"window-0","duration_us":40,"rank":0,"iteration":17}
{"type":"span","span_id":5,"name":"infinilm:forward_step","window_id":"window-0","duration_us":900,"rank":0,"iteration":17}
{"type":"span","span_id":6,"name":"infinilm:sample","window_id":"window-0","duration_us":35,"rank":0,"iteration":17,"acceptance_count":4,"preempt_count":0}
{"type":"metric","name":"tokens_per_iteration","value":32,"window_id":"window-0","rank":0,"iteration":17}
```

Rules:

- Span names must be exactly one of `infinilm:queue_wait`,
  `infinilm:admission`, `infinilm:batch_build`, `infinilm:schedule`,
  `infinilm:forward_step`, or `infinilm:sample`.
- Every span and metric should carry the same `window_id`.
- A sidecar stream with no single accepted window cannot enter a P0 promotion
  request.
- Baseline and candidate sidecar streams must use the same window identity.
- Spans must come from real engine ranges. Do not derive them from hcTracer gaps.
- Use one file per role and one contiguous observation window.
- Add `rank` for every tensor-parallel worker when available.
- Optional numeric span fields are `iteration`, `queue_depth`, `batch_size`,
  `tokens_per_iteration`, `acceptance_count`, and `preempt_count`.
- Unknown scheduler-looking span names are preserved as parser warnings but do
  not become scheduler signals.
- The parser preserves malformed-line warnings instead of silently dropping the run.
- Use one file per role and one contiguous observation window.
- Add `rank` for every tensor-parallel worker when available.
- The parser preserves malformed-line warnings instead of silently dropping the run.
