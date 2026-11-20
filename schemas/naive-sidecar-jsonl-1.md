# Naive sidecar JSONL

One JSON object per line. A producer may emit `span` and `metric` records.

```json
{"type":"span","name":"infinilm.scheduler.step","start_ns":1730000000000000000,"end_ns":1730000000000250000,"tid":123,"rank":0}
{"type":"span","name":"infinilm.scheduler.batch_build","duration_us":120,"rank":0}
{"type":"metric","name":"tokens_per_iteration","value":32,"rank":0}
{"type":"metric","name":"queue_depth","value":3,"rank":0}
```

Rules:

- Spans must come from real engine ranges. Do not derive them from hcTracer gaps.
- Use one file per role and one contiguous observation window.
- Add `rank` for every tensor-parallel worker when available.
- The parser preserves malformed-line warnings instead of silently dropping the run.
