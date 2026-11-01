# Annotated operator intents (M04b, scored in M08b)

`annotated.jsonl` holds one intent per line. Every line is a JSON object with these fields:

| Field | Meaning |
|---|---|
| `id` | `h01`… for the held-out split, `t01`… for the tuning split |
| `split` | `heldout` (scored once, in M08b) or `tune` (used to choose the model and tune the prompt) |
| `lang` | `fr` or `en` |
| `text` | the operator's words, verbatim |
| `ambiguous` | true when a careful human would ask a question before acting |
| `expected` | the facts a correct parser extracts: a `ParsedIntent` (below) |
| `notes` | why this case is here, what makes it hard |

`expected` is a `ParsedIntent` (`agent/schema.py`), the parse step's output:

```json
{"understood": true, "clarification": null, "priority": "urllc",
 "min_thr_mbps": {"embb": null, "urllc": 10, "be": null},
 "max_latency_ms": 8, "duration_s": 600}
```

## Annotation rules

1. **Facts only.** Record only what the text states. Something not stated is `null`. The
   parser never decides weights; the plan step does.
2. **`priority`** is set only when the operator explicitly favours or protects one slice
   ("prioriser", "protéger", "favoriser", "prioritise", "protect", "favour"). Asking for a
   guarantee alone does not set a priority.
3. **Units:** throughput in Mb/s (Gb/s × 1000, kb/s ÷ 1000), latency in ms, duration in s.
4. **Slice words:** video, streaming, "haut débit" mean eMBB; critical, real-time, robots,
   "faible latence" mean URLLC; background, backups, "le reste" mean best effort.
5. **`understood: false`** means the request can't be written down with these fields
   without guessing: a number with no slice, a contradiction, a latency bound on a slice
   other than URLLC, something out of scope, or something too vague. Only `understood` is
   scored then; the other fields stay `null` and the clarification wording is free.
6. **Impossible values are still extracted.** A clear request for something the cell can't
   do ("60 Mb/s for URLLC") is `understood: true` with the value; the verifier rejects it
   later. The parser does not judge values.

## Who wrote it, in what order

Claude drafted every intent, and Souhail reviewed them (decision of 2026-11-01). To limit
the bias of having one author for both splits:
1. the held-out split was written and committed **before** the parsing prompt existed;
2. the prompt was then tuned only on the tuning split.

The M08b extraction accuracy should still be read as optimistic, because one author wrote
both splits.
