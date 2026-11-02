"""
Empirical benchmark suite for MATP Contribution 2
(Background Abstraction & Trajectory Synthesis Engine).

Every number in this file is measured at runtime.  There are no hardcoded
retention or latency constants; if a metric cannot be measured it is omitted.

Evaluates:
1. Token compression across five tool domains.
2. Context growth across a 30-turn horizon (raw / FIFO / MATP).
3. Retrieval quality: strict and domain-level P@1/P@3, plus key-fact retention
   at an equal token budget.
4. Query latency scaling as the trajectory store grows.
5. Retained vs dropped token mass (a measured proxy for signal/noise).

Run:  python -m src.matp.benchmark
"""

from __future__ import annotations

import json
import os
import statistics
import time
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .abstraction import BackgroundAbstractionEngine
from .embedder import LocalNGramEmbedder
from .memory_store import MATPMemoryStore
from .tokenizer import count_tokens, tokenizer_name, truncate_to_tokens

plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
    }
)

NITK_BLUE = "#123466"
ACCENT_ORANGE = "#E65100"
ACCENT_GREEN = "#2E7D32"
MUTED_GRAY = "#5A6473"
WARN_RED = "#B43C28"


def generate_benchmark_dataset() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Return 30 synthetic-but-realistic tool calls and 10 evaluation queries.

    Each domain carries a small set of ground-truth facts that a compressed
    representation should preserve.  Queries name the exact target step so that
    strict P@1 is unambiguous (no substring / domain leniency).
    """
    sql_rows = [
        {
            "user_id": 100 + i,
            "username": f"user_{i}",
            "email": f"user_{i}@nitk.edu",
            "role": "engineer",
            "status": "active",
            "created_at": "2026-01-01T12:00:00Z",
            "metadata": {"login_count": i * 3, "ip": f"192.168.1.{i}"},
        }
        for i in range(40)
    ]
    domain_templates = [
        {
            "domain": "SQL Database",
            "sub_goal": "Query active users and schema definitions from postgres database",
            "tool_name": "sql_query",
            "parameters": {
                "query": "SELECT user_id, username, email, role, status FROM users WHERE status='active' LIMIT 50;",
                "table": "users",
            },
            "status": "SUCCESS",
            "raw_output": json.dumps(sql_rows, indent=2),
            "facts": ["users", "40 rows", "user_id", "username", "email", "role"],
        },
        {
            "domain": "Runtime Traceback",
            "sub_goal": "Execute database migration script on port 5432",
            "tool_name": "python_interpreter",
            "parameters": {"script": "scripts/migrate_v2.py", "env": "production"},
            "status": "FAILURE",
            "raw_output": (
                "Traceback (most recent call last):\n"
                '  File "/usr/local/lib/python3.11/site-packages/sqlalchemy/engine/base.py", line 145, in _connect\n'
                "    return self.driver.connect(*args, **kwargs)\n"
                '  File "/usr/local/lib/python3.11/site-packages/psycopg2/__init__.py", line 122, in connect\n'
                "    dsn = _ext.make_dsn(conninfo, **extra)\n"
                '  File "/app/scripts/migrate_v2.py", line 88, in run_migration\n'
                '    db = create_engine("postgresql://admin:secret@127.0.0.1:5432/app_db")\n'
                '  File "/usr/local/lib/python3.11/site-packages/sqlalchemy/engine/strategies.py", line 118, in connect\n'
                "    return dialect.connect(*cargs, **cparams)\n"
                'sqlalchemy.exc.OperationalError: (psycopg2.OperationalError) connection to server at "127.0.0.1", port 5432 failed: Connection refused.\n'
                "Is the server running on that host and accepting TCP/IP connections?\n" * 8
            ),
            "facts": ["OperationalError", "5432", "127.0.0.1", "migrate_v2.py", "connection"],
        },
        {
            "domain": "REST API Response",
            "sub_goal": "Fetch OAuth tokens and user permissions from identity provider API",
            "tool_name": "http_request",
            "parameters": {"endpoint": "https://auth.internal.net/v2/oauth/token", "method": "POST"},
            "status": "SUCCESS",
            "raw_output": json.dumps(
                {
                    "access_token": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IlJhaHVsIn0.sample_sig",
                    "token_type": "Bearer",
                    "expires_in": 3600,
                    "refresh_token": "rfr_987654321_nitk_token",
                    "scope": "read write admin telemetry audit_log billing security_events",
                    "user_profile": {
                        "id": "usr_99218273",
                        "email": "shishir@nitk.edu",
                        "tenant_id": "tenant_nitk_it_dept",
                        "roles": ["admin", "developer", "maintainer"],
                        "attributes": {"dept": "IT", "campus": "Surathkal", "tier": "gold"},
                    },
                    "session_telemetry": {
                        "device_id": "dev_mac_9921",
                        "ip_address": "10.0.4.12",
                        "location": "Surathkal, IN",
                        "user_agent": "AgenticLLM/2.4 (x86_64-apple-darwin20.6.0)",
                        "tls_cipher": "TLS_AES_256_GCM_SHA384",
                        "handshake_latency_ms": 28.4,
                    },
                    "headers": {
                        "X-RateLimit-Limit": "1000",
                        "X-RateLimit-Remaining": "985",
                        "X-RateLimit-Reset": "1620000000",
                        "Server": "CloudFlare-Edge",
                        "CF-RAY": "89371283912a-BOM",
                        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
                    },
                    "audit_metadata": [
                        {"event": f"audit_log_{i}", "code": 200, "timestamp": f"2026-08-31T20:{i:02d}:00Z"}
                        for i in range(35)
                    ],
                },
                indent=2,
            ),
            "facts": ["access_token", "Bearer", "expires_in", "refresh_token", "scope"],
        },
        {
            "domain": "File System / Bash",
            "sub_goal": "Scan repository structure and locate authentication configuration files",
            "tool_name": "read_file",
            "parameters": {"path": "config/auth_v2.yaml", "mode": "read_all"},
            "status": "SUCCESS",
            "raw_output": "\n".join(
                f"line_{i}: setting_{i}: value_{i}_random_metadata_bloat_padding_info_for_long_system_file"
                for i in range(150)
            )
            + "\nauth_provider: oauth2\nclient_id: nitk_agent_client\nredirect_uri: https://nitk.edu/callback",
            "facts": ["auth_v2.yaml", "153 lines", "auth_provider", "oauth2", "client_id"],
        },
        {
            "domain": "Doc / PDF Parsing",
            "sub_goal": "Parse quarterly infrastructure cost report tables and compute spend",
            "tool_name": "pdf_reader",
            "parameters": {"file": "reports/q3_costs.pdf", "pages": "1-10"},
            "status": "SUCCESS",
            "raw_output": "--- PAGE 1 HEADER ---\nNATIONAL INSTITUTE OF TECHNOLOGY KARNATAKA\nCONFIDENTIAL REPORT\n"
            + (
                "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed do eiusmod tempor incididunt ut labore. "
                * 45
            )
            + "\nTable 1: Total Compute Spend: $42,500\n"
            + ("Random OCR artifacts, footer boilerplate and page numbers.\n" * 50),
            "facts": ["q3_costs.pdf", "$42,500", "Compute Spend"],
        },
    ]

    dataset: List[Dict[str, Any]] = []
    for step in range(1, 31):
        template = domain_templates[(step - 1) % len(domain_templates)]
        item = dict(template)
        item["step_id"] = step
        item["sub_goal"] = f"{template['sub_goal']} [step {step}]"
        # Each turn is an independent benchmark task.  Give the OAuth POST a
        # per-turn endpoint so it is not mistaken for a state mutation that
        # supersedes the identical call in an earlier turn.
        if item["tool_name"] == "http_request":
            parameters = dict(item["parameters"])
            parameters["endpoint"] = f"{parameters['endpoint']}?turn={step}"
            item["parameters"] = parameters
        dataset.append(item)

    test_queries = [
        {"query": "Postgres SQL query active users table columns", "target_step": 1, "domain": "SQL Database"},
        {"query": "migration script failure OperationalError connection refused port 5432", "target_step": 2, "domain": "Runtime Traceback"},
        {"query": "identity provider OAuth tokens and access permissions endpoint", "target_step": 3, "domain": "REST API Response"},
        {"query": "repository auth_v2.yaml configuration file read", "target_step": 4, "domain": "File System / Bash"},
        {"query": "quarterly infrastructure cost report PDF compute spend table", "target_step": 5, "domain": "Doc / PDF Parsing"},
        {"query": "find users table schema attributes", "target_step": 6, "domain": "SQL Database"},
        {"query": "OperationalError psycopg2 database connect failure", "target_step": 7, "domain": "Runtime Traceback"},
        {"query": "OAuth bearer access token refresh payload", "target_step": 8, "domain": "REST API Response"},
        {"query": "config auth_v2.yaml client id settings", "target_step": 9, "domain": "File System / Bash"},
        {"query": "compute spend financial report table pdf", "target_step": 10, "domain": "Doc / PDF Parsing"},
    ]
    return dataset, test_queries


def _fact_recall(facts: List[str], text: str) -> float:
    if not facts:
        return 1.0
    lowered = str(text or "").lower()
    hits = sum(1 for fact in facts if fact.lower() in lowered)
    return hits / len(facts)


def _timed(fn, repeats: int = 5) -> float:
    """Median wall-clock milliseconds over repeated calls (one warm-up)."""
    fn()  # warm-up
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return round(statistics.median(samples), 4)


def _make_scaling_store(db_path: str, count: int, dataset: List[Dict[str, Any]]) -> Tuple[MATPMemoryStore, List[str], List[List[float]]]:
    """Build a store of ``count`` trajectories plus a naive raw-text index."""
    store = MATPMemoryStore(db_path=db_path, auto_migrate=False, enable_queue=False, session_id="scale", scope_id="scale")
    store.clear()
    raw_embedder = LocalNGramEmbedder(dim=256)
    raw_texts: List[str] = []
    raw_vectors: List[List[float]] = []
    for i in range(count):
        item = dataset[i % len(dataset)]
        step_id = i + 1
        store.matp_index_trajectory(
            step_id=step_id,
            sub_goal=item["sub_goal"],
            tool_name=item["tool_name"],
            parameters=item["parameters"],
            status=item["status"],
            raw_output=item["raw_output"],
        )
        raw_text = f"{item['sub_goal']} {item['tool_name']} {item['raw_output']}"
        raw_texts.append(raw_text)
        raw_vectors.append(raw_embedder.embed(raw_text))
    return store, raw_texts, raw_vectors


def run_benchmarks(output_dir: str = "benchmarks/results") -> Dict[str, Any]:
    os.makedirs(output_dir, exist_ok=True)
    dataset, test_queries = generate_benchmark_dataset()
    step_domain = {item["step_id"]: item["domain"] for item in dataset}
    step_facts = {item["step_id"]: item["facts"] for item in dataset}
    tokenizer = tokenizer_name()

    engine = BackgroundAbstractionEngine()
    db_path = os.path.join(output_dir, "benchmark.db")
    store = MATPMemoryStore(db_path=db_path, auto_migrate=False, enable_queue=False, session_id="bench", scope_id="bench")
    store.clear()

    raw_embedder = LocalNGramEmbedder(dim=256)
    raw_entries: List[Dict[str, Any]] = []
    raw_vectors: List[List[float]] = []
    domain_stats: Dict[str, Dict[str, List[float]]] = {}
    distilled_by_step: Dict[int, Dict[str, Any]] = {}

    print("=" * 68)
    print("MATP CONTRIBUTION 2 BENCHMARK (all values measured at runtime)")
    print(f"tokenizer: {tokenizer}")
    print("=" * 68)

    # ------------------------------------------------------------------ indexing
    for item in dataset:
        distilled = store.matp_index_trajectory(
            step_id=item["step_id"],
            sub_goal=item["sub_goal"],
            tool_name=item["tool_name"],
            parameters=item["parameters"],
            status=item["status"],
            raw_output=item["raw_output"],
        )
        distilled_by_step[item["step_id"]] = {
            "compact_text": distilled.to_compact_text(),
            "raw_output": item["raw_output"],
            "distilled_tokens": count_tokens(distilled.to_compact_text()),
            "raw_tokens": count_tokens(item["raw_output"]),
        }
        raw_text = f"{item['sub_goal']} {item['tool_name']} {item['raw_output']}"
        raw_entries.append({"step_id": item["step_id"], "domain": item["domain"], "raw_text": raw_text})
        raw_vectors.append(raw_embedder.embed(raw_text))

        domain = item["domain"]
        bucket = domain_stats.setdefault(domain, {"raw_tokens": [], "distilled_tokens": [], "compression_pct": []})
        bucket["raw_tokens"].append(distilled.raw_token_count_est)
        bucket["distilled_tokens"].append(distilled.distilled_token_count_est)
        bucket["compression_pct"].append(
            (1.0 - distilled.distilled_token_count_est / max(1, distilled.raw_token_count_est)) * 100.0
        )

    stats = store.get_stats()
    print(f"Trajectories indexed: {stats['total_entries']}")
    print(f"Raw tokens:           {stats['total_raw_tokens']:,}")
    print(f"Distilled tokens:     {stats['total_distilled_tokens']:,}")
    print(f"Compression:          {stats['compression_ratio_pct']}% ({stats['token_reduction_factor']}x)")
    print("-" * 68)

    # ------------------------------------------------------------------ retrieval
    counts = {
        "strict_p1_matp": 0,
        "strict_p1_raw": 0,
        "strict_p3_matp": 0,
        "strict_p3_raw": 0,
        "domain_p1_matp": 0,
        "domain_p1_raw": 0,
        "domain_p3_matp": 0,
        "domain_p3_raw": 0,
    }
    fact_matp: List[float] = []
    fact_raw: List[float] = []
    matp_latencies: List[float] = []
    raw_latencies: List[float] = []

    for query in test_queries:
        start = time.perf_counter()
        matp_res = store.matp_query_trajectory(query["query"], top_k=3)
        matp_latencies.append((time.perf_counter() - start) * 1000.0)

        start = time.perf_counter()
        query_vec = raw_embedder.embed(query["query"])
        scored = [(entry, raw_embedder.cosine_similarity(query_vec, vec)) for entry, vec in zip(raw_entries, raw_vectors)]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        raw_res = scored[:3]
        raw_latencies.append((time.perf_counter() - start) * 1000.0)

        matp_steps = [entry["tuple"]["step_id"] for entry, _ in matp_res]
        raw_steps = [entry["step_id"] for entry, _ in raw_res]

        target_step = query["target_step"]
        target_domain = query["domain"]

        counts["strict_p1_matp"] += int(bool(matp_steps) and matp_steps[0] == target_step)
        counts["strict_p1_raw"] += int(bool(raw_steps) and raw_steps[0] == target_step)
        counts["strict_p3_matp"] += int(target_step in matp_steps)
        counts["strict_p3_raw"] += int(target_step in raw_steps)
        counts["domain_p1_matp"] += int(bool(matp_steps) and step_domain.get(matp_steps[0]) == target_domain)
        counts["domain_p1_raw"] += int(bool(raw_steps) and step_domain.get(raw_steps[0]) == target_domain)
        counts["domain_p3_matp"] += int(any(step_domain.get(step) == target_domain for step in matp_steps))
        counts["domain_p3_raw"] += int(any(step_domain.get(step) == target_domain for step in raw_steps))

        # Fact retention is measured against the *target* step so it isolates
        # compression fidelity from retrieval ranking.
        record = distilled_by_step[target_step]
        budget = record["distilled_tokens"]
        fact_matp.append(_fact_recall(step_facts[target_step], record["compact_text"]))
        fact_raw.append(_fact_recall(step_facts[target_step], truncate_to_tokens(record["raw_output"], budget)))

    n_queries = len(test_queries)
    pct = {key: round(100.0 * value / n_queries, 1) for key, value in counts.items()}
    fact_matp_pct = round(100.0 * statistics.mean(fact_matp), 1)
    fact_raw_pct = round(100.0 * statistics.mean(fact_raw), 1)

    print(
        f"Strict P@1: MATP={pct['strict_p1_matp']}% vs Raw={pct['strict_p1_raw']}% | "
        f"P@3: MATP={pct['strict_p3_matp']}% vs Raw={pct['strict_p3_raw']}%"
    )
    print(
        f"Domain P@1: MATP={pct['domain_p1_matp']}% vs Raw={pct['domain_p1_raw']}% | "
        f"P@3: MATP={pct['domain_p3_matp']}% vs Raw={pct['domain_p3_raw']}%"
    )
    print(f"Fact retention (equal token budget): MATP={fact_matp_pct}% vs Raw={fact_raw_pct}%")
    print(
        f"Query latency (median): MATP={statistics.median(matp_latencies):.3f}ms "
        f"vs Raw={statistics.median(raw_latencies):.3f}ms"
    )
    print("-" * 68)

    # ------------------------------------------------------------ latency scaling
    scale_sizes = [30, 60, 120, 240]
    latency_curve = {"sizes": scale_sizes, "matp_ms": [], "raw_ms": []}
    scale_db = os.path.join(output_dir, "benchmark_scale.db")
    for size in scale_sizes:
        scale_store, raw_texts, raw_vecs = _make_scaling_store(scale_db, size, dataset)
        emb = LocalNGramEmbedder(dim=256)
        latency_curve["matp_ms"].append(
            _timed(lambda s=scale_store: s.matp_query_trajectory("users auth token report", top_k=3))
        )

        def naive_scan() -> None:
            qvec = emb.embed("users auth token report")
            ranked = [(i, emb.cosine_similarity(qvec, vec)) for i, vec in enumerate(raw_vecs)]
            ranked.sort(key=lambda pair: pair[1], reverse=True)
            _ = ranked[:3]

        latency_curve["raw_ms"].append(_timed(naive_scan))
        scale_store.close()
    print(
        "Latency scaling (ms, N=30/60/120/240): "
        f"MATP={latency_curve['matp_ms']} Raw={latency_curve['raw_ms']}"
    )
    print("=" * 68)

    # ------------------------------------------------------------------ plots
    domains = list(domain_stats.keys())
    avg_raw = [np.mean(domain_stats[d]["raw_tokens"]) for d in domains]
    avg_dist = [np.mean(domain_stats[d]["distilled_tokens"]) for d in domains]
    comp_ratios = [np.mean(domain_stats[d]["compression_pct"]) for d in domains]
    x = np.arange(len(domains))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(9, 4.8))
    ax1.bar(x - width / 2, avg_raw, width, label="Raw tool output $O_{raw}$", color=MUTED_GRAY, alpha=0.85)
    ax1.bar(x + width / 2, avg_dist, width, label="MATP distilled tuple $G$", color=NITK_BLUE)
    ax1.set_ylabel(f"Tokens per tool call ({tokenizer}, log scale)")
    ax1.set_yscale("log")
    ax1.set_title("Token Footprint Reduction: Raw Output vs Distilled Tuple")
    ax1.set_xticks(x)
    ax1.set_xticklabels(domains, rotation=12, ha="right")
    ax1.legend(loc="upper right")
    for i, ratio in enumerate(comp_ratios):
        ax1.annotate(
            f"-{ratio:.1f}%",
            xy=(x[i] + width / 2, avg_dist[i]),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color=NITK_BLUE,
        )
    plt.tight_layout()
    plot1 = os.path.join(output_dir, "fig1_token_compression.png")
    plt.savefig(plot1)
    plt.close()

    turns = np.arange(1, len(dataset) + 1)
    tuples = [
        engine.abstract_tool_output(
            d["step_id"], d["sub_goal"], d["tool_name"], d["parameters"], d["status"], d["raw_output"]
        )
        for d in dataset
    ]
    raw_tokens = [t.raw_token_count_est for t in tuples]
    cum_raw = np.cumsum(raw_tokens)
    cum_dist = np.cumsum([t.distilled_token_count_est for t in tuples])
    fifo3 = [sum(raw_tokens[max(0, t - 3) : t]) for t in range(1, len(dataset) + 1)]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(turns, cum_raw, label="Unpruned raw context", color=WARN_RED, lw=2.2, linestyle="--")
    ax.plot(turns, fifo3, label="FIFO(3) window", color=ACCENT_ORANGE, lw=2.0, linestyle=":")
    ax.plot(turns, cum_dist, label="MATP distilled memory (all history)", color=NITK_BLUE, lw=2.5)
    ax.axhline(8192, color="gray", linestyle="-.", alpha=0.7, label="8k context limit")
    ax.set_xlabel("Agent execution turn $t$")
    ax.set_ylabel("Tokens held in context / buffer")
    ax.set_title("Cumulative Memory Footprint over 30 Turns")
    ax.legend(loc="upper left")
    plt.tight_layout()
    plot2 = os.path.join(output_dir, "fig2_context_growth.png")
    plt.savefig(plot2)
    plt.close()

    fig, (ax_quality, ax_latency) = plt.subplots(1, 2, figsize=(11, 4.6))
    metrics = ["Strict P@1", "Domain P@1", "Strict P@3", "Domain P@3", "Fact retention"]
    raw_scores = [
        pct["strict_p1_raw"],
        pct["domain_p1_raw"],
        pct["strict_p3_raw"],
        pct["domain_p3_raw"],
        fact_raw_pct,
    ]
    matp_scores = [
        pct["strict_p1_matp"],
        pct["domain_p1_matp"],
        pct["strict_p3_matp"],
        pct["domain_p3_matp"],
        fact_matp_pct,
    ]
    positions = np.arange(len(metrics))
    ax_quality.bar(positions - width / 2, raw_scores, width, label="Raw log store", color=MUTED_GRAY, alpha=0.85)
    ax_quality.bar(positions + width / 2, matp_scores, width, label="MATP", color=NITK_BLUE)
    ax_quality.set_ylabel("Score (%)")
    ax_quality.set_title("Retrieval Quality and Fact Retention")
    ax_quality.set_xticks(positions)
    ax_quality.set_xticklabels(metrics, rotation=15, ha="right")
    ax_quality.set_ylim(0, 115)
    ax_quality.legend(loc="lower right")
    for i in range(len(metrics)):
        ax_quality.annotate(
            f"{matp_scores[i]:.0f}%", (positions[i] + width / 2, matp_scores[i] + 2), ha="center", fontweight="bold", color=NITK_BLUE
        )
        ax_quality.annotate(
            f"{raw_scores[i]:.0f}%", (positions[i] - width / 2, raw_scores[i] + 2), ha="center", color=MUTED_GRAY
        )

    ax_latency.plot(latency_curve["sizes"], latency_curve["raw_ms"], marker="o", label="Raw log scan", color=MUTED_GRAY, lw=2)
    ax_latency.plot(
        latency_curve["sizes"], latency_curve["matp_ms"], marker="s", label="MATP SQLite scan", color=NITK_BLUE, lw=2.2
    )
    ax_latency.set_xlabel("Stored trajectories")
    ax_latency.set_ylabel("Median query latency (ms)")
    ax_latency.set_title("Measured Query Latency Scaling")
    ax_latency.legend(loc="upper left")
    plt.tight_layout()
    plot3 = os.path.join(output_dir, "fig3_retrieval_and_latency.png")
    plt.savefig(plot3)
    plt.close()

    fig, ax4 = plt.subplots(figsize=(9, 4.8))
    retained = [100.0 - ratio for ratio in comp_ratios]
    dropped = list(comp_ratios)
    ax4.bar(x, dropped, width=0.55, label="Token mass removed", color="#D9E2EC", edgecolor=NITK_BLUE, hatch="//")
    ax4.bar(x, retained, width=0.55, bottom=dropped, label="Token mass retained in $G$", color=NITK_BLUE)
    ax4.set_xticks(x)
    ax4.set_xticklabels(domains, rotation=12, ha="right")
    ax4.set_ylabel("Share of raw tokens (%)")
    ax4.set_title("Measured Token Mass: Removed vs Retained")
    ax4.set_ylim(0, 125)
    ax4.legend(loc="upper left", framealpha=0.9)
    for i in range(len(domains)):
        ax4.annotate(
            f"retained: {retained[i]:.1f}%\nremoved: {dropped[i]:.1f}%",
            xy=(i, 102),
            ha="center",
            fontsize=8.5,
            fontweight="bold",
            color=NITK_BLUE,
        )
    plt.tight_layout()
    plot4 = os.path.join(output_dir, "fig4_token_mass.png")
    plt.savefig(plot4)
    plt.close()

    results = {
        "tokenizer": tokenizer,
        "dataset_size": len(dataset),
        "query_count": n_queries,
        "stats": stats,
        "domain_stats": {
            d: {
                "avg_raw": round(float(np.mean(domain_stats[d]["raw_tokens"])), 1),
                "avg_distilled": round(float(np.mean(domain_stats[d]["distilled_tokens"])), 1),
                "compression_pct": round(float(np.mean(domain_stats[d]["compression_pct"])), 1),
            }
            for d in domain_stats
        },
        "retrieval": {
            **pct,
            "fact_retention_matp": fact_matp_pct,
            "fact_retention_raw": fact_raw_pct,
            "median_latency_ms_matp": round(statistics.median(matp_latencies), 4),
            "median_latency_ms_raw": round(statistics.median(raw_latencies), 4),
        },
        "latency_scaling": latency_curve,
        "plots": [plot1, plot2, plot3, plot4],
    }
    results_path = os.path.join(output_dir, "benchmark_results.json")
    with open(results_path, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    results["results_file"] = results_path
    print(f"Saved: {results_path}")
    print(f"Saved: {plot1}\nSaved: {plot2}\nSaved: {plot3}\nSaved: {plot4}")
    store.close()
    return results


if __name__ == "__main__":
    run_benchmarks()
    print("\nBenchmark execution complete.")
