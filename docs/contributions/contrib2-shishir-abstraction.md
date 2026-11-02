# MATP: Memory-Augmented Tool Policy
## Contribution 2: Background Abstraction, Causal DAG & Macro Distillation Engine

**Author:** Shishir Hegde  
**Department of Information Technology, NITK Surathkal**

This repository contains **Contribution 2 only**. Contributions 1 (temporal/state-aware vector store) and 3 (Bayesian policy ranking) are owned by teammates and are not implemented here.

---

## 1. What was delivered (end-semester)

| Objective | Status | Module |
| :--- | :--- | :--- |
| Atomic trajectory abstraction + episodic knowledge graph | Implemented | `abstraction.py`, `memory_store.py` |
| Zero-dependency hybrid retrieval | Implemented | `embedder.py`, `memory_store.py` |
| **Cross-session causal DAG invalidation** | Implemented | `resources.py`, `dag.py` |
| **Hierarchical macro-action distillation** | Implemented | `signatures.py`, `macros.py` |
| **Async Ollama/Qwen2.5 background worker** | Implemented | `queue.py`, `ollama_client.py`, `worker.py` |
| **Multi-agent / multi-session evaluation harnesses** | Implemented | `benchmarks/` |
| MCP stdio server | Implemented, 16 tools | `mcp_server.py` |
| SQLite durability (replacing per-index JSON rewrites) | Implemented | `storage.py` |

### Cross-session causal DAG

Every indexed trajectory records the resources it read, wrote or deleted. Resource identifiers are canonicalised (case-normalised paths, DB tables, API endpoints, git repositories). A write creates a new resource version and:

* **supersedes** earlier writers (retrieval weight `0`),
* marks earlier readers **stale** (weight `0.5 × taint`),
* transitively taints downstream `DEPENDS_ON` consumers as **suspect** (weight = taint, decayed per hop, bounded by depth and floor).

Validity is a soft state: invalid nodes remain in storage for provenance and can be retrieved with `include_stale=True`. A truth-maintenance pass (`recompute_validity`) heals nodes after re-execution, and `confirm_state` provides a manual override. Invalidation is scoped by `scope_id` (workspace snapshot), so two independent tasks on the same repository do not invalidate each other.

### Hierarchical macro-action distillation

The miner slices sessions into 10-step windows, canonicalises each action signature (tool + parameter *shape* + status, discarding literal values), clusters near-identical sequences with `difflib.SequenceMatcher`, parameterises varying literals into slots, and stores the composite skill. Level-2 skills are promoted at support ≥ 3 and success ≥ 0.8; chains of promoted skills can be mined as level-3 skills. Matching combines dense similarity, lexical evidence and precondition state; expansion is declarative (the agent executes the returned steps).

### Async neural worker

Indexing only performs one SQLite `INSERT` into a durable job queue. A separate worker process leases jobs (with lease recovery and idempotency keys) and calls a local Ollama `/api/chat` endpoint with JSON-schema structured output. Handlers cover trajectory summary refinement, macro synthesis and session digests. `MATP_NEURAL=auto` (default) enqueues jobs but never starts a model call in the hot path; `off` disables enrichment; `on` is used when a daemon is running.

---

## 2. Directory structure

```
├── demo.py                       # End-to-end CLI demo
├── src/matp/
│   ├── abstraction.py            # Multi-domain distillation + triples
│   ├── embedder.py               # Deterministic subword n-gram vectorizer
│   ├── resources.py              # Canonical file/DB/API/git/shell resources
│   ├── signatures.py             # Canonical action signatures
│   ├── storage.py                # SQLite schema, WAL, legacy JSON migration
│   ├── dag.py                    # Causal invalidation + truth maintenance
│   ├── macros.py                 # Hierarchical macro mining/matching/expansion
│   ├── queue.py                  # Durable leased job queue
│   ├── ollama_client.py          # Stdlib Ollama client (JSON schema)
│   ├── worker.py                 # Async worker daemon
│   ├── memory_store.py           # Public store API (backwards compatible)
│   ├── mcp_server.py             # MCP stdio server (16 tools)
│   ├── tokenizer.py              # Optional tiktoken, documented fallback
│   └── benchmark.py              # Measured compression/retrieval/latency suite
├── benchmarks/
│   ├── replay.py                 # DAG labels, macro replay, worker overhead
│   ├── mini_swe.py               # Controlled SWE-style transfer benchmark
│   ├── gaia_lite.py              # Local multi-hop QA + stale-fact traps
│   └── run_all.py                # Runs every suite, writes summary.json
├── tests/                        # 45 pytest tests
└── data/
    └── matp.db                   # SQLite store (created/migrated at runtime)
```

The legacy `data/matp.memory.json` is imported once into SQLite (tracked by `meta.legacy_imported`).

---

## 3. How to run

```bash
pip install -e ".[dev]"                 # or: pip install pytest
python -m pytest                        # 45 tests
python demo.py                          # end-to-end demonstration

python -m src.matp.benchmark            # compression / retrieval / latency
python -m benchmarks.replay             # DAG labels, macro replay, queue overhead
python -m benchmarks.mini_swe           # cross-session repair transfer
python -m benchmarks.gaia_lite          # multi-hop QA with stale facts
python -m benchmarks.run_all            # all suites -> benchmarks/results/summary.json
```

Async worker:

```bash
# terminal 1: local model
ollama pull qwen2.5:7b-instruct
ollama serve

# terminal 2: worker
python -m src.matp.worker --once          # one job
python -m src.matp.worker --daemon        # continuous
python -m src.matp.worker --daemon --max-jobs 20
```

MCP server:

```bash
MATP_DB=data/matp.db MATP_NEURAL=off python -m src.matp.mcp_server
```

### Configuration

| Variable | Default | Meaning |
| :--- | :--- | :--- |
| `MATP_DB` | `data/matp.db` | SQLite database path |
| `MATP_LEGACY_JSON` | `data/matp.memory.json` | One-time legacy import |
| `MATP_NEURAL` | `auto` | `off` / `auto` / `on` |
| `MATP_OLLAMA_URL` | `http://127.0.0.1:11434` | Ollama endpoint |
| `MATP_OLLAMA_MODEL` | `qwen2.5:7b-instruct` | Worker model |
| `MATP_EMBED_DIM` | `256` | Local vector dimension |
| `MATP_DAG_DECAY` | `0.8` | Per-hop taint decay |
| `MATP_TEMPORAL_DECAY` | `0.0` | Retrieval recency decay λ (per hour) |
| `MATP_MACRO_WINDOW` | `10` | Mining window size |
| `MATP_MACRO_MIN_SUPPORT` | `3` | Promotion support threshold |
| `MATP_TOKENIZER` | `auto` | Use `tiktoken` if installed, else heuristic |

---

## 4. Measured results

Environment: Python 3.14, SQLite 3.50, Windows, CPU only. All values below are produced by the scripts in this repository; results are written to `benchmarks/results/`.

### 4.1 Token compression and retrieval (`benchmark.py`)

| Tool domain | Raw tokens / call | Distilled tokens / call | Compression |
| :--- | ---: | ---: | ---: |
| SQL database | 3,441 | 234 | 93.2% |
| Runtime traceback | 1,808 | 107 | 94.1% |
| REST API response | 1,416 | 178 | 87.4% |
| File system / Bash | 2,185 | 131 | 94.0% |
| Document / PDF | 1,337 | 104 | 92.2% |
| **Overall (30 turns)** | **61,122** | **4,524** | **92.6% (13.5×)** |

* Context growth: raw accumulates 61,122 tokens by turn 30; the distilled buffer holds the full history in 4,524 tokens.
* Retrieval on 10 targeted queries:
  * **Domain P@1 / P@3:** MATP **100% / 100%** vs raw log store **60% / 60%**.
  * **Strict exact-step P@1 / P@3:** MATP **50% / 80%** vs raw **20% / 60%**. Strict matching is deliberately harsh here: the 30-turn synthetic workload repeats each of five domains six times, so several steps are content-identical and exact-step identity is ambiguous.
  * **Fact retention at equal token budget:** MATP **88.0%** vs raw **37.3%**.
* Median query latency at N=30: MATP 4.68 ms vs raw scan 1.48 ms. MATP pays for SQLite row decoding and validity weighting; latency scales linearly in both cases (N=240: 43.8 ms vs 10.0 ms).
* Token counts use `tiktoken/cl100k_base` when installed; otherwise the suite records `heuristic(words+punct)` and never reports a hidden `chars/4` constant.

### 4.2 Causal DAG replay (`benchmarks/replay.py`)

Scenarios cover direct staleness, supersession, transitive suspects, cross-session invalidation, scope isolation, healing after re-execution, two-hop chains, manual override and file/API writes.

* **Validity-label accuracy: 24/24 = 100%** (fresh / stale / suspect / superseded).

### 4.3 Macro-action replay

Two 10-step routines were mined from three training sessions each and replayed on held-out fourth sessions with unseen literals.

* **Held-out tool-sequence match: 100%.**
* **Held-out slot-binding accuracy: 100%.**
* Macro definition size: ~214–226 tokens vs ~461–480 tokens for the full compact trajectory (~53% smaller).

### 4.4 Async worker overhead

* Median indexing latency: **4.96 ms without queue** vs **5.22 ms with queue** (difference within run-to-run noise).
* Queue mode enqueued one job per trajectory (30 jobs) without any model call on the hot path.

### 4.5 Multi-agent-style evaluations

These are controlled, dependency-free analogues; they are **not** official SWE-bench or GAIA leaderboard scores.

| Suite | Result |
| :--- | :--- |
| `mini_swe.py` (10 unseen repair tasks, scripted fixer) | Cold heuristic **20%** solved vs MATP memory **100%**; retrieval accuracy **100%** |
| `gaia_lite.py` (6 multi-hop questions, 2 stale traps) | MATP **100%** vs naive **83.3%**; support recall **100%** both; stale answers **0/2 vs 1/2** |

The real GAIA dataset is gated and not bundled; `gaia_lite.py` uses a local corpus with the same multi-hop + superseded-fact structure. `mini_swe.py` measures cross-session repair-pattern transfer with a deterministic scripted fixer, isolating the memory contribution rather than claiming an end-to-end LLM SWE-bench result.

---

## 5. Metric definitions and measurement procedure

Every aggregate in §4 is computed at runtime from the scripts in this repository; no retention or latency value is hardcoded. Per-query and per-task records are available in `benchmarks/results/*.json`.

### 5.1 Shared primitives

**Token counting** (`src/matp/tokenizer.py`). When `tiktoken` is installed, counts come from `cl100k_base`. Otherwise the counter is `max(1, words+punct, len(text)//6)`, a documented heuristic rather than a hidden `chars/4` constant. The active counter is recorded in every result file, and both sides of every comparison use the same counter.

**Retrieval scoring.** MATP combines dense and lexical evidence with

```text
hybrid = α · cosine(query, tuple) + (1 − α) · |query_terms ∩ tuple_terms| / |query_terms|
score  = hybrid × validity_weight
```

with `α = 0.75` by default. The validity weight is:

| Validity | Weight |
| :--- | :--- |
| `fresh` | `1.0` |
| `stale` | `max(0.15, 0.5 × taint)` |
| `suspect` | `max(0.15, taint)` |
| `superseded` | `0.0` (excluded unless `include_stale=True`) |

An optional temporal factor `exp(−λ · age_hours)` is applied when `MATP_TEMPORAL_DECAY` is non-zero. The raw baseline embeds the full raw tool output with the same local n-gram embedder and ranks by cosine similarity.

**Fact recall.** The fraction of a step's ground-truth fact strings that occur as case-insensitive substrings in the candidate text. This is deliberately lexical; it does not use an LLM judge, so paraphrases count as misses.

### 5.2 Compression and context growth (`benchmark.py`)

| Metric | Definition | Procedure |
| :--- | :--- | :--- |
| Per-domain compression | `(1 − D/R) × 100` | Arithmetic mean over the six turns in that domain; `R` = raw output tokens, `D` = compact-tuple tokens |
| Reduction factor | `R/D` | Same per-domain values |
| Overall compression | `(1 − ΣD/ΣR) × 100` | Sums over all 30 rows in SQLite, not an average of percentages |
| Context at turn 30 | Raw `cumsum`, FIFO(3) last-3 sum, MATP distilled `cumsum` | Recomputed from the same 30-turn dataset |
| Token mass | `removed% = compression%`, `retained% = 100 − compression%` | Re-plot of the measured split |

The 30-turn workload contains five domains repeated six times. The 8k reference line in the context figure is a fixed budget marker, not a measured value.

### 5.3 Retrieval quality (`benchmark.py`)

Ten queries each declare a `target_step` and a `target_domain`. MATP returns top-3 after validity weighting; the raw baseline returns top-3 by cosine over the 30 raw texts.

| Metric | Definition |
| :--- | :--- |
| Strict P@1 / P@3 | rank-1 / any top-3 `step_id` exactly equals the target step |
| Domain P@1 / P@3 | rank-1 / any top-3 result belongs to the target domain |
| Fact retention | fraction of target-step facts present in the MATP tuple versus the raw output truncated to the **same token budget** |
| Latency (table) | median wall-clock over the 10 query calls |
| Latency (scaling) | one warm-up plus five timed repeats at N = 30/60/120/240; median reported |

Both strict and domain P@1 are reported because each of five domains repeats six times, so several steps are content-identical and exact-step identity is ambiguous. Strict matching is the harsher test; domain matching asks whether the right kind of evidence was retrieved. Fact retention isolates compression fidelity from ranking: both representations receive the same budget, and the raw text is truncated to the MATP tuple's token count before scoring.

### 5.4 Causal DAG replay (`benchmarks/replay.py`)

Nine scenarios cover direct staleness, supersession, transitive suspects, cross-session invalidation, scope isolation, healing after re-execution, two-hop chains, manual override, and file/API writes. The 24 nodes are labelled by construction, then compared with the DAG's computed `validity`.

* **Accuracy** = correct / 24.
* **Per-class precision / recall / F1** come from a one-vs-rest confusion matrix over `fresh`, `stale`, `suspect`, `superseded` (supports 13 / 7 / 3 / 1).

A 100% result shows consistency with the intended semantics; it is not a measurement of open-world generalization, because the scenarios and labels were authored alongside the DAG rules.

### 5.5 Macro-action replay (`benchmarks/replay.py`)

Two 10-step routines are mined from three training variants each (`window=10`, `stride=10`, `min_support=3`; promotion requires support ≥ 3 and success ≥ 0.8). The held-out fourth variant changes literals only (`auth_v4`, `q6`), never the routine structure.

* `tool_sequence_match` = position-wise agreement of canonical tool names (`zip`, so a length mismatch is penalized).
* `binding_accuracy` = exact bound-value matches / total expected parameters.
* Token reduction = `1 − tokens(macro JSON) / tokens(sum of compact trajectories)`.

The token-reduction path uses `len(text)//4` rather than the shared tokenizer, so the ~53% figure is indicative; the sequence and binding metrics are exact. Held-out variation is literal-only, so 100% supports parameterised macro reuse, not discovery of entirely new workflows.

### 5.6 Async worker overhead (`benchmarks/replay.py`)

The same 30 trajectories are indexed once with `enable_queue=False` and once with `enable_queue=True`; each individual `matp_index_trajectory` call is timed and the median reported. With the queue enabled, 30 durable job rows are inserted and **no worker or model runs** during measurement. This measures enqueue overhead only, not LLM enrichment latency or throughput.

### 5.7 mini-SWE and GAIA-lite (`benchmarks/mini_swe.py`, `benchmarks/gaia_lite.py`)

**mini-SWE.** Five bug types, one solved training example each, two unseen evaluation tasks per type (10 total). The cold baseline always applies the most common repair; the memory condition retrieves the top-1 stored trajectory, extracts its `{pattern, replacement}` rule, applies it, and runs the task's tests. A task counts as solved when the test subprocess exits 0; retrieval accuracy compares the retrieved rule with the correct rule for that bug type.

**GAIA-lite.** Thirteen local facts and six multi-hop questions, two of which include a superseded fact. The naive condition embeds all raw fact texts and takes cosine top-4; MATP takes top-4 after validity weighting. Answers are extracted by the question's regex. Support recall is the fraction of required fact IDs present in the top-4; stale answers count extractions that equal the superseded value.

Both are controlled local analogues. They isolate memory retrieval and state suppression; they are **not** official SWE-bench or GAIA scores.

### 5.8 Threats to validity

* **Small, synthetic, self-authored workloads.** Ten queries, ten repair tasks and six QA questions are point estimates without confidence intervals.
* **No live LLM in the main loop.** Compression and retrieval metrics do not by themselves prove downstream task success; mini-SWE uses a deterministic regex fixer and GAIA-lite uses regex answer extraction.
* **Heuristic tokenizer.** Absolute token counts would change with `tiktoken`, though both sides of each comparison use the same counter.
* **Lexical fact matching.** Paraphrased facts count as misses.
* **Authored ground truth.** DAG labels and macro hold-outs test intended semantics and literal transfer, not open-world generalization.
* **Latency is not optimized.** Measurements are single-threaded Python/SQLite on a Windows CPU; run-to-run variation is visible in the reported medians.

### 5.9 Reproducing individual records

```bash
python -m src.matp.benchmark     # -> benchmark_results.json
python -m benchmarks.replay      # -> replay_results.json
python -m benchmarks.mini_swe    # -> mini_swe_results.json
python -m benchmarks.gaia_lite   # -> gaia_lite_results.json
python -m benchmarks.run_all     # -> summary.json
```

Each JSON file retains the per-query or per-task records behind its aggregate, including expected labels, retrieved IDs, extracted answers, and timings.

---

## 6. Test suite

```
45 passed
```

Coverage includes resource canonicalisation, SQL/file/API/git/shell access extraction, direct and transitive invalidation, scope isolation, healing, overrides, lineage, macro mining/matching/expansion/demotion, queue leasing/retry/idempotency, worker handlers and revision guards, store retrieval filters, legacy migration, and the MCP server over real stdio subprocesses.

---

## 7. Known limitations

* **GAIA/SWE-bench are local analogues.** No gated dataset or Docker-based SWE-bench harness is bundled; access was unavailable/undecided.
* **No live LLM A/B run.** The neural worker is implemented and tested with a fake client; end-to-end quality with a running Qwen2.5 daemon is not measured here.
* **Linear scan latency.** SQLite retrieval is exact and `O(N)`; at very large N an ANN index / `sqlite-vec` table would be needed.
* **Macro matching is dependency-free.** The dense component is a hashed n-gram vectorizer; lexical evidence is used deliberately to reduce hash-collision noise. A real embedding model would be expected to improve semantic matching.
* **Scope must be supplied by the caller.** Cross-task isolation depends on using a distinct `scope_id` per workspace snapshot; the default is a single shared scope.
