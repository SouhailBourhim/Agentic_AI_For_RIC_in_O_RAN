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
   parser never decides weights; the plan step does. A missing or vague duration ("for a
   while") is just `null`; it never makes a request `understood: false` on its own.
2. **`priority`** is set only when the operator explicitly favours or protects one slice:
   the verbs "prioriser", "protéger", "favoriser", "prioritise", "protect", "favour", or the
   noun "priorité / priority" given to one slice. Asking for a guarantee alone does not set
   a priority, and neither do "keep", "make sure", "assurer", "garantir" or "donner".
3. **Units:** throughput in Mb/s (Gb/s × 1000, kb/s ÷ 1000), latency in ms, duration in s.
   "Give / donne / assure / at least / au moins" N Mb/s to a slice is a minimum throughput.
4. **Slice words:** video, streaming, "haut débit" mean eMBB; critical, real-time, robots,
   low-latency, "faible latence", "slice critique" mean URLLC; background, backups,
   "trafic de fond", "le reste" mean best effort.
5. **`understood: false`** means the request can't be written down with these fields
   without guessing. The cases: a number with no slice or with two candidate slices; a
   contradiction (including two slices favoured at once, since `priority` holds one); a
   latency bound on a slice other than URLLC; something out of scope (switching a slice
   off, a throughput cap); or something too vague. Only `understood` is scored then; the
   other fields stay `null` and the clarification wording is free.
6. **Values the verifier will reject are still extracted.** A clear request outside the
   verifier's limits ("60 Mb/s for URLLC", above the 40 % training range) is
   `understood: true` with the value; the verifier rejects it later. The parser does not
   judge values.

`ambiguous` is true exactly when `understood` is false in this set; it is kept as a
separate field so a later set can hold requests that are understood but deserve a check.

## Who wrote it, in what order

Claude drafted every intent (decision of 2026-11-01). To limit the bias of having one
author for both splits:
1. the held-out split was written and committed **before** the parsing prompt existed;
2. the prompt was then tuned only on the tuning split.

**Review (2026-11-02).** Souhail delegated the review to four independent Claude
subagents: two annotated the 40 texts blind (from these rules only, without the labels)
and both agreed with every label, field by field; two checked the labels against the rules
and found no wrong label. Their wording findings were applied: in the held-out split h10,
h17 and h19 were reworded to remove a second reading (labels unchanged; no model was run
on the held-out split), and h05 and h20 got corrected notes; in the tuning split t08, t11,
t12, t13 and t18 were rewritten to stop sharing a sentence template with a held-out item or
to remove a second reading. Rules 1–6 above were made explicit where the reviewers found
gaps.

The M08b extraction accuracy should still be read as optimistic: one author (a Claude
model) wrote both splits and the reviewers are Claude models too, and most held-out case
types (units, no slice, contradiction, out of scope) also appear in the tuning split, so
the score measures familiar case types more than generalisation.
