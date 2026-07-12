# Hermes memory-provider benchmark

Head-to-head comparison of Hermes Agent memory providers, driven through the
**same `MemoryProvider` seams Hermes itself calls** (`prefetch` →
`on_turn_start` → `sync_turn` → `queue_prefetch` → `on_session_end` →
`shutdown`), with a full provider shutdown between sessions so recall must
come from durable storage — never from process state or chat history.

## Results (2026-07-12, Hermes 0.15.2 provider ABI)

Scenario: 4 sessions, 20 turns, 6 cold-start probes (including one
correction/supersession test and one hallucination-bait probe that was never
answered in the transcript).

| Provider | Packet recall | Stale leaks | Median packet tokens | Median recall ms | Net connections (ingest/recall) | Offline recall | Model-visible tools |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: |
| **cortext** | **8/14 facts** | 1 | **97** | **16** | **0/0** | **6/6 probes** | **0** |
| mem0 (60.5K★, most popular) | 8/14 facts | 1 | 122 | 505 | 9/1 | 0/6 probes | 3 |
| holographic (built-in default) | 0/14 facts | 0 | 0 | 0 | 0/0 | 0/6 probes | 2 |
| holographic-tools (steelman) | 0/14 facts | 0 | 0 | 0 | 0/0 | 0/6 probes | 2 |

Read: Cortext ties the most popular provider on recall quality while being
~30× faster at recall, fully offline, invisible to the model, and never
touching the network. Both leaked one superseded fact fragment on the
correction probe.

## Method

- Every provider replays the **identical scripted transcript**
  ([scenario.py](scenario.py)) — fixed user *and* assistant turns, so the only
  variable is the memory backend (standard replay methodology).
- Probes run in **fresh cold-start provider instances** on the same durable
  store, with natural-language questions passed to `prefetch()` — exactly
  what Hermes passes (the user's message).
- **Packet recall** counts expected fact groups present in the returned
  context packet. **Stale leaks** counts superseded facts (the moved vet
  appointment) still present.
- **Offline recall** repeats every probe with outbound sockets disabled.
- **Net connections** counts outbound TCP connections (keep-alive reuse means
  requests ≥ connections).
- An optional live phase (set `OPENAI_API_KEY`) has a model answer each probe
  from each packet, plus a no-memory control, then blind-judges anonymized
  answers with a separate judge prompt.

## Caveats — read before quoting

- **Holographic** stores facts primarily via model-invoked `fact_store`
  tools; a seam-level replay has no model, so it captures only its
  `auto_extract` regexes. The `holographic-tools` steelman simulates a
  perfectly diligent model storing **every** user turn via its tool — it
  still scored 0/14 because its stage-1 retrieval is FTS5 with implicit AND:
  natural-sentence prefetch queries match nothing (single keywords do). In
  real use, quality depends on the model distilling good keyword queries.
- **Mem0** extracts facts server-side (their hosted platform); the harness
  waits 10s between sessions and 5s after the final one for eventual
  consistency. Slower settle could improve its recall slightly.
- One scenario, one run, small N. This measures the automatic memory path
  under identical treatment; it is not a claim about every workload. The
  scenario was written before any provider was run and was not tuned
  afterward.

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install hermes-agent mem0ai certifi
echo 'MEM0_API_KEY=...' >> .env                 # only needed for mem0
.venv/bin/python -m bench.run_bench --providers cortext,holographic,holographic-tools,mem0
# optional live answer + blind-judge phase:
echo 'OPENAI_API_KEY=...' >> .env
```

Outputs land in `bench/results/`: raw per-probe packets in `results.json`,
summary in `REPORT.md`, blind-judge verdicts in `judgments.json` (live phase).
