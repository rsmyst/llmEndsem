"""GAIA-lite: multi-hop QA over local observations with stale-fact traps.

GAIA itself is gated and requires external tooling, so this harness uses a
small local corpus with the same structure: multi-hop questions whose answer
requires two supporting observations, plus two facts that are superseded by a
later observation.  It measures answer accuracy, supporting-fact recall and
whether a naive log store returns stale values that MATP suppresses.

Run:  python -m benchmarks.gaia_lite
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.matp.embedder import LocalNGramEmbedder  # noqa: E402
from src.matp.memory_store import MATPMemoryStore  # noqa: E402

RESULTS = ROOT / "benchmarks" / "results"

# mode="write" facts share a resource path, so the later one supersedes the
# earlier one in the causal DAG.
FACTS: List[Dict[str, Any]] = [
    {
        "id": "owner",
        "text": "Project Falcon is owned by the IT department.",
        "source": "docs/falcon.md",
        "mode": "doc",
    },
    {
        "id": "hq",
        "text": "The IT department is headquartered in Surathkal.",
        "source": "docs/org.md",
        "mode": "doc",
    },
    {
        "id": "atlas_owner",
        "text": "Atlas is maintained by the Platform team.",
        "source": "docs/atlas.md",
        "mode": "doc",
    },
    {
        "id": "platform_cluster",
        "text": "The Platform team uses the Vega compute cluster.",
        "source": "docs/platform.md",
        "mode": "doc",
    },
    {
        "id": "dash_db",
        "text": "The analytics dashboard reads from the Orion database.",
        "source": "docs/dashboard.md",
        "mode": "doc",
    },
    {
        "id": "orion_region",
        "text": "The Orion database is hosted in the Mumbai region.",
        "source": "docs/orion.md",
        "mode": "doc",
    },
    {
        "id": "mobile_gateway",
        "text": "The mobile app is fronted by the Edge gateway.",
        "source": "docs/mobile.md",
        "mode": "doc",
    },
    {
        "id": "edge_tls",
        "text": "The Edge gateway requires TLS 1.3.",
        "source": "docs/edge.md",
        "mode": "doc",
    },
    {
        "id": "owner_office",
        "text": "The IT department occupies building A on the campus.",
        "source": "docs/office.md",
        "mode": "doc",
    },
    {
        "id": "key_old",
        "text": "The current API key for Project Falcon is KEY-OLD in the legacy configuration.",
        "source": "config/falcon.env",
        "mode": "write",
    },
    {
        "id": "key_new",
        "text": "The current API key for Project Falcon is KEY-NEW after the rotation.",
        "source": "config/falcon.env",
        "mode": "write",
    },
    {
        "id": "retry_old",
        "text": "The payments service retry limit is 5 attempts in the legacy configuration.",
        "source": "config/payments.yaml",
        "mode": "write",
    },
    {
        "id": "retry_new",
        "text": "The payments service retry limit is 8 attempts after the incident review.",
        "source": "config/payments.yaml",
        "mode": "write",
    },
]

QUESTIONS: List[Dict[str, Any]] = [
    {
        "id": "q_hq",
        "question": "Where is the department that owns Project Falcon headquartered?",
        "supports": ["owner", "hq"],
        "pattern": r"headquartered in ([A-Za-z]+)",
        "answer": "Surathkal",
    },
    {
        "id": "q_cluster",
        "question": "Which compute cluster does the team that maintains Atlas use?",
        "supports": ["atlas_owner", "platform_cluster"],
        "pattern": r"uses the ([A-Za-z0-9]+) compute cluster",
        "answer": "Vega",
    },
    {
        "id": "q_region",
        "question": "Which region hosts the database used by the analytics dashboard?",
        "supports": ["dash_db", "orion_region"],
        "pattern": r"hosted in the ([A-Za-z]+) region",
        "answer": "Mumbai",
    },
    {
        "id": "q_tls",
        "question": "What TLS version does the gateway that fronts the mobile app require?",
        "supports": ["mobile_gateway", "edge_tls"],
        "pattern": r"requires (TLS [0-9]+(?:\.[0-9]+)?)",
        "answer": "TLS 1.3",
    },
    {
        "id": "q_key",
        "question": "What is the current API key for Project Falcon?",
        "supports": ["key_new"],
        "pattern": r"(KEY-[A-Z]+)",
        "answer": "KEY-NEW",
        "stale_value": "KEY-OLD",
        "stale_support": "key_old",
    },
    {
        "id": "q_retry",
        "question": "What is the recommended retry limit for the payments service?",
        "supports": ["retry_new"],
        "pattern": r"retry limit is (\d+)",
        "answer": "8",
        "stale_value": "5",
        "stale_support": "retry_old",
    },
]


def _extract(text: str, pattern: str) -> Optional[str]:
    match = re.search(pattern, text or "", re.IGNORECASE)
    return match.group(1) if match else None


def _index_facts(store: MATPMemoryStore) -> Dict[str, str]:
    node_by_fact: Dict[str, str] = {}
    for fact in FACTS:
        session = f"gaia-{fact['id']}"
        tool = "write_file" if fact["mode"] == "write" else "doc_parser"
        parameters = {"path": fact["source"]} if fact["mode"] == "write" else {"file": fact["source"]}
        store.matp_index_trajectory(
            step_id=1,
            sub_goal=fact["text"],
            tool_name=tool,
            parameters=parameters,
            status="SUCCESS",
            raw_output=fact["text"],
            session_id=session,
            scope_id="gaia",
        )
        node_by_fact[fact["id"]] = f"{session}:1"
    return node_by_fact


def run(output_dir: Path = RESULTS) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    store = MATPMemoryStore(
        db_path=str(output_dir / "gaia_lite.db"), auto_migrate=False, enable_queue=False, session_id="gaia", scope_id="gaia"
    )
    store.clear()
    try:
        node_by_fact = _index_facts(store)
        embedder = LocalNGramEmbedder(dim=256)
        raw_texts = [fact["text"] for fact in FACTS]
        raw_vectors = [embedder.embed(text) for text in raw_texts]

        records = []
        for question in QUESTIONS:
            query_vec = embedder.embed(question["question"])
            raw_ranked = sorted(
                zip(FACTS, [embedder.cosine_similarity(query_vec, vec) for vec in raw_vectors]),
                key=lambda pair: pair[1],
                reverse=True,
            )
            raw_top = raw_ranked[:4]
            raw_answer = _extract(" ".join(fact["text"] for fact, _ in raw_top), question["pattern"])
            raw_support = {fact["id"] for fact, _ in raw_top}

            matp_results = store.matp_query_trajectory(question["question"], top_k=4, scope_id="gaia")
            matp_text = " ".join(entry["compact_text"] for entry, _ in matp_results)
            matp_answer = _extract(matp_text, question["pattern"])
            matp_support = {
                fact_id
                for fact_id, node in node_by_fact.items()
                if node in {entry["node_id"] for entry, _ in matp_results}
            }

            support = set(question["supports"])
            records.append(
                {
                    "id": question["id"],
                    "question": question["question"],
                    "expected": question["answer"],
                    "matp_answer": matp_answer,
                    "naive_answer": raw_answer,
                    "matp_correct": matp_answer == question["answer"],
                    "naive_correct": raw_answer == question["answer"],
                    "matp_support_recall": round(len(support & matp_support) / len(support), 4),
                    "naive_support_recall": round(len(support & raw_support) / len(support), 4),
                    "stale_expected": question.get("stale_value"),
                }
            )

        total = len(records)
        stale_questions = [r for r in records if r["stale_expected"]]
        return {
            "questions": total,
            "matp_correct": sum(r["matp_correct"] for r in records),
            "naive_correct": sum(r["naive_correct"] for r in records),
            "matp_accuracy": round(100.0 * sum(r["matp_correct"] for r in records) / total, 1),
            "naive_accuracy": round(100.0 * sum(r["naive_correct"] for r in records) / total, 1),
            "mean_matp_support_recall": round(100.0 * sum(r["matp_support_recall"] for r in records) / total, 1),
            "mean_naive_support_recall": round(100.0 * sum(r["naive_support_recall"] for r in records) / total, 1),
            "stale_questions": len(stale_questions),
            "stale_answers_naive": sum(1 for r in stale_questions if r["naive_answer"] == r["stale_expected"]),
            "stale_answers_matp": sum(1 for r in stale_questions if r["matp_answer"] == r["stale_expected"]),
            "records": records,
            "note": "Local GAIA-style analogue; the real GAIA dataset is gated and not bundled.",
        }
    finally:
        store.close()


def main() -> int:
    results = run()
    path = RESULTS / "gaia_lite_results.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("=" * 68)
    print("GAIA-LITE MULTI-HOP QA")
    print("=" * 68)
    print(f"Questions: {results['questions']}")
    print(f"MATP accuracy:    {results['matp_accuracy']}% ({results['matp_correct']}/{results['questions']})")
    print(f"Naive accuracy:   {results['naive_accuracy']}% ({results['naive_correct']}/{results['questions']})")
    print(f"Support recall:   MATP={results['mean_matp_support_recall']}% Naive={results['mean_naive_support_recall']}%")
    print(
        f"Stale answers:    MATP={results['stale_answers_matp']}/{results['stale_questions']} "
        f"vs Naive={results['stale_answers_naive']}/{results['stale_questions']}"
    )
    print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
