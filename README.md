# MATP: Memory-Augmented Tool Policy Monorepo
### Unified 7th Semester Capstone Deliverable — NITK Surathkal

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://python.org/)
[![TypeScript 5.x](https://img.shields.io/badge/TypeScript-5.x-blue.svg)](https://www.typescriptlang.org/)
[![Model Context Protocol](https://img.shields.io/badge/MCP-1.30.0-purple.svg)](https://modelcontextprotocol.io/)
[![LaTeX: Tectonic](https://img.shields.io/badge/Rendered_with-Tectonic-brightgreen.svg)](https://tectonic-typesetting.github.io/)

**MATP (Memory-Augmented Tool Policy)** resolves the classic long-horizon failure modes of autonomous LLM agents:
1. **Catastrophic Context Eviction (FIFO Trap):** Invariants established early in a session are lost when prompts are truncated.
2. **Stale-State Hallucination (Naive RAG Trap):** Unversioned semantic similarity retrieves high-cosine superseded state vectors.
3. **Procedural Thrashing:** Repetitive schema violations and unguided tool choice loops.

This monorepo consolidates three complementary NITK Surathkal contributions into a single, cohesive, zero-external-dependency local MCP system:
- **Contribution 1 (Rahul Saravanan):** Version-aware temporal semantic memory with causal discrete logical clocks, exponential decay $R(q, v_i)$, and online adaptive feedback calibration $\lambda_{\text{task}}$.
- **Contribution 2 (Shishir Hegde):** Trajectory distillation ($13.5\times$ token reduction), cross-session causal Directed Acyclic Graph (DAG) state invalidation, hierarchical macro-action mining, and ACID SQLite durability.
- **Contribution 3 (Ashab Muhammed):** Dynamic Bayesian tool recommendation matrix with Laplace-smoothed Beta posterior confidence and self-healing parameter auto-correction hints.

---

## Repository Structure

```text
matp-monorepo/
├── demo.py                             # Interactive end-to-end unified demonstration script
├── pyproject.toml                      # Root Python package definition (pip install -e .)
│
├── src/matp/                           # Unified Python MCP Server & Core Engine
│   ├── abstraction.py                  # Trajectory tuple distillation & AST/regex fact triples
│   ├── config.py                       # Environment and path configurations
│   ├── dag.py                          # Cross-session causal DAG & invalidation BFS propagation
│   ├── decay.py                        # Adaptive temporal decay tuner (per-task lambda)
│   ├── embedder.py                     # Deterministic local subword n-gram vectorizer (no GPU)
│   ├── macros.py                       # Hierarchical macro skill miner & parameter slotting
│   ├── mcp_server.py                   # Unified stdio FastMCP server (20 tools, 4 resources)
│   ├── memory_store.py                 # Unified MATP persistent SQLite memory store
│   ├── policy.py                       # Dynamic candidate ranking & self-healing error hints
│   ├── queue.py                        # Non-blocking SQLite background job queue
│   ├── resources.py                    # Canonical resource resolver (db:*, file:*, api:*, git:*)
│   ├── signatures.py                   # Canonical action signatures
│   ├── storage.py                      # SQLite DDL, transactions, and binary vector blobs
│   ├── tokenizer.py                    # Token estimation utilities
│   └── worker.py                       # Background Ollama/neural synthesis worker
│
├── packages/matp-mcp-ts/               # TypeScript MCP Server Package
│   ├── src/core/policy.ts              # TypeScript policy implementation
│   ├── src/memory/memory.ts            # Brute-force JSON vector memory store
│   └── package.json
│
├── benchmarks/                         # Evaluation Harnesses & Validation Scripts
│   ├── benchmark_temporal.py           # 40-step long-horizon schema evolution benchmark
│   ├── gaia_lite.py                    # Multi-step complex question answering harness
│   ├── mini_swe.py                     # Automated software repair simulation harness
│   ├── replay.py                       # Causal DAG replay and macro accuracy replay
│   └── run_all.py                      # Master benchmark runner
│
├── report/                             # IEEE Computer Society Formal Research Paper
│   ├── report.tex                      # IEEEtran Computer Society LaTeX document
│   ├── report.pdf                      # Fully rendered publication-grade PDF (via Tectonic)
│   └── figures/                        # Generated figures and SVG/PDF metrics plots
│       ├── architecture.pdf / .png     # Mermaid architecture diagram
│       ├── dag_flowchart.pdf / .png    # Mermaid causal DAG lifecycle flowchart
│       ├── macro_policy_flowchart.pdf  # Mermaid macro & tool policy flowchart
│       ├── metric_retrieval_precision  # Long-horizon retrieval precision (SVG, PDF, PNG)
│       ├── metric_compression_fact     # Token compression vs. fact retention (SVG, PDF, PNG)
│       ├── metric_temporal_decay_curves# Temporal decay & adaptive lambda convergence
│       └── metric_latency_breakdown    # Microbenchmark latency profile across operations
│
└── presentation/                       # Academic Defense Beamer Deck
    ├── presentation.tex                # Professional Madrid/NITK 16:9 Beamer slides
    ├── presentation.pdf                # Fully rendered presentation PDF (via Tectonic)
    └── figures/                        # Presentation slide graphic assets
```

---

## Quick Start & Verification

### 1. Run the Unified End-to-End Demo
The unified demo exercises all three contributions in sequence:
```bash
cd matp-monorepo
python3 demo.py
```

### 2. Run the 40-Step Long-Horizon Benchmark
Compare FIFO vs. Naive RAG vs. Lineage-Only vs. Unified MATP:
```bash
python3 benchmarks/benchmark_temporal.py
```

### 3. Re-compile Report & Presentation with Tectonic
Both the IEEE report and the Beamer presentation compile with zero LaTeX dependencies:
```bash
# Compile the 8-page IEEE Computer Society Research Paper
cd report && tectonic report.tex

# Compile the 11-slide Academic Defense Beamer Presentation
cd ../presentation && tectonic presentation.tex
```

### 4. Regenerate Architecture Diagrams & Metric Plots
```bash
# Rebuild Mermaid diagrams
cd report
mmdc -p puppeteer-config.json -i figures/architecture.mmd -o figures/architecture.pdf
mmdc -p puppeteer-config.json -i figures/dag_flowchart.mmd -o figures/dag_flowchart.pdf
mmdc -p puppeteer-config.json -i figures/macro_policy_flowchart.mmd -o figures/macro_policy_flowchart.pdf

# Rebuild Python metric plots (SVG, PDF, PNG)
python3 figures/generate_metrics_plots.py
```

---

## Key Performance Results

| Metric | Baseline | MATP Performance | Impact |
| :--- | :--- | :--- | :--- |
| **Active State Precision** | 25.0% (FIFO) / 60.0% (Naive RAG) | **100.0%** | Complete elimination of obsolete state |
| **Stale State Leakage** | 40.0% (Naive RAG) | **0.0%** | Hard-zero retrieval weighting for superseded items |
| **Token Footprint** | 61,122 raw tokens | **4,524 distilled tokens** | **92.6% reduction (13.5$\times$ compression)** |
| **Fact Retention** | 37.3% (Truncated window) | **88.0%** | High-fidelity tuple fact triple preservation |
| **Macro Replay Accuracy**| N/A | **100.0%** | Zero-divergence sequence match and slot binding |
| **Operational Latency** | N/A | **< 5.0 ms** | Real-time stdio MCP response on commodity CPU |

---

## Authors & Acknowledgments

- **Rahul Saravanan** (Contribution 1: Temporal Versioning, Scoring Function & Adaptive Decay)
- **Shishir Hegde** (Contribution 2: Background Distillation, Causal DAG Invalidation & Macro Mining)
- **Ashab Muhammed** (Contribution 3: Tool Policy Matrix & Self-Healing Parameter Hints)

*Department of Information Technology, National Institute of Technology Karnataka (NITK), Surathkal.*
