# Naive sidecar JSONL

One JSON object per line. A producer may emit `span` and `metric` records.
**Promotion ranking uses metrics only.** Span rows are provenance/warnings and
are not a `scheduler_signals` contract.

```json
{"type":"metric","name":"tokens_per_iteration","value":32,"window_id":"window-0","rank":0,"iteration":17}
{"type":"metric","name":"queue_depth","value":3,"window_id":"window-0","rank":0,"iteration":17}
{"type":"metric","name":"batch_size","value":8,"window_id":"window-0","rank":0,"iteration":17}
{"type":"span","span_id":1,"name":"optional-engine-marker","window_id":"window-0","duration_us":40,"rank":0,"iteration":17}
```

Rules:

- Metric rows are the promotion evidence for packing/queue hints.
- Span names are free-form; the parser records them as warnings and does not
  aggregate them into ranking signals.
- Every metric should carry the same `window_id`.
- A sidecar stream with no single accepted window cannot enter a P0 promotion
  request.
- Baseline and candidate sidecar streams must use the same window identity.
- Use one file per role and one contiguous observation window.
- Add `rank` for every tensor-parallel worker when available.
- The parser preserves malformed-line warnings instead of silently dropping the run.
