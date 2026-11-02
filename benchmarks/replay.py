"""Replay benchmarks for causal invalidation, macro reuse and worker overhead.

Unlike the synthetic metric plots in ``benchmark.py``, this harness builds
scenarios with known ground-truth validity labels and scores the DAG against
them.  It then mines macros from three training sessions and replays the learned
macro on a held-out fourth session.

Run:  python -m benchmarks.replay
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.matp.memory_store import MATPMemoryStore  # noqa: E402
from src.matp.signatures import canonical_tool  # noqa: E402

RESULTS = ROOT / "benchmarks" / "results"

READ_USERS = {"query": "SELECT * FROM users", "table": "users"}
WRITE_USERS = {"query": "ALTER TABLE users ADD COLUMN x TEXT", "table": "users"}
WRITE_USERS_2 = {"query": "CREATE INDEX idx_users_x ON users(x)", "table": "users"}


def _index(store, session, scope, step, tool, params, parent=None, scope_override=None):
    store.matp_index_trajectory(
        step_id=step,
        sub_goal=f"{session} step {step}",
        tool_name=tool,
        parameters=params,
        status="SUCCESS",
        raw_output="ok",
        parent_step_id=parent,
        session_id=session,
        scope_id=scope_override or scope,
    )


# ---------------------------------------------------------------------------
# Scenario construction: each entry is (session, step, tool, params, parent,
# expected validity).
# ---------------------------------------------------------------------------
def build_dag_scenarios() -> List[Tuple[str, List[Tuple[str, int, str, Dict[str, Any], Any, str]]]]:
    return [
        (
            "read_then_write",
            [
                ("rtw", 1, "sql_query", READ_USERS, None, "stale"),
                ("rtw", 2, "sql_query", WRITE_USERS, None, "fresh"),
            ],
        ),
        (
            "two_writes",
            [
                ("tw", 1, "sql_query", WRITE_USERS, None, "superseded"),
                ("tw", 2, "sql_query", WRITE_USERS_2, None, "fresh"),
            ],
        ),
        (
            "transitive_child",
            [
                ("tc", 1, "sql_query", READ_USERS, None, "stale"),
                ("tc", 2, "read_file", {"path": "logs/users.log"}, 1, "suspect"),
                ("tc", 3, "sql_query", WRITE_USERS, None, "fresh"),
            ],
        ),
        (
            "cross_session",
            [
                ("cs-a", 1, "sql_query", READ_USERS, None, "stale"),
                ("cs-b", 1, "sql_query", WRITE_USERS, None, "fresh"),
            ],
        ),
        (
            "scope_isolation",
            [
                ("iso-a", 1, "sql_query", READ_USERS, None, "fresh"),
                ("iso-b", 1, "sql_query", WRITE_USERS, None, "fresh"),
            ],
        ),
        (
            "heal_after_reexecution",
            [
                ("heal", 1, "sql_query", READ_USERS, None, "stale"),
                ("heal", 2, "sql_query", WRITE_USERS, None, "fresh"),
                ("heal", 3, "sql_query", READ_USERS, None, "fresh"),
            ],
        ),
        (
            "two_hop_chain",
            [
                ("chain", 1, "sql_query", READ_USERS, None, "stale"),
                ("chain", 2, "read_file", {"path": "auth/users.yaml"}, 1, "suspect"),
                ("chain", 3, "read_file", {"path": "logs/auth.log"}, 2, "suspect"),
                ("chain", 4, "sql_query", WRITE_USERS, None, "fresh"),
            ],
        ),
        (
            "manual_override",
            [
                ("ovr", 1, "sql_query", READ_USERS, None, "fresh"),  # overridden below
                ("ovr", 2, "sql_query", WRITE_USERS, None, "fresh"),
            ],
        ),
        (
            "file_and_api",
            [
                ("fa", 1, "read_file", {"path": "config/app.yaml"}, None, "stale"),
                ("fa", 2, "write_file", {"path": "config/app.yaml"}, None, "fresh"),
                ("fa", 3, "http_request", {"endpoint": "https://api.example.com/v1/users", "method": "GET"}, None, "stale"),
                ("fa", 4, "http_request", {"endpoint": "https://api.example.com/v1/users", "method": "POST"}, None, "fresh"),
            ],
        ),
    ]


def _scope_for(name: str) -> str:
    return f"scope-{name}"


def score_labels(expected: Dict[str, str], actual: Dict[str, str]) -> Dict[str, Any]:
    classes = ["fresh", "stale", "suspect", "superseded"]
    matrix = {c: {"tp": 0, "fp": 0, "fn": 0} for c in classes}
    correct = 0
    for node, wanted in expected.items():
        got = actual.get(node, "missing")
        if got == wanted:
            correct += 1
        for c in classes:
            if got == c and wanted == c:
                matrix[c]["tp"] += 1
            elif got == c and wanted != c:
                matrix[c]["fp"] += 1
            elif got != c and wanted == c:
                matrix[c]["fn"] += 1
    per_class = {}
    for c in classes:
        tp, fp, fn = matrix[c]["tp"], matrix[c]["fp"], matrix[c]["fn"]
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        f1 = 2 * precision * recall / max(1e-9, precision + recall)
        per_class[c] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": tp + fn,
        }
    return {
        "accuracy": round(correct / max(1, len(expected)), 4),
        "correct": correct,
        "total": len(expected),
        "per_class": per_class,
    }


def run_dag_replay() -> Dict[str, Any]:
    db_path = RESULTS / "replay_dag.db"
    store = MATPMemoryStore(db_path=str(db_path), auto_migrate=False, enable_queue=False, session_id="replay", scope_id="replay")
    store.clear()
    expected: Dict[str, str] = {}
    try:
        for name, nodes in build_dag_scenarios():
            scope = _scope_for(name)
            for session, step, tool, params, parent, wanted in nodes:
                if name == "scope_isolation":
                    node_scope = "scope-iso-A" if session == "iso-a" else "scope-iso-B"
                else:
                    node_scope = scope
                _index(store, session, node_scope, step, tool, params, parent=parent)
                expected[f"{session}:{step}"] = wanted
        # Manual override scenario.
        assert store.confirm_state("ovr:1", "verified by replay harness") is True
        store.recompute_validity(_scope_for("manual_override"))
        actual = {node: store.get_trajectory(node)["validity"] for node in expected}
        metrics = score_labels(expected, actual)
        metrics["misclassified"] = {node: {"expected": expected[node], "actual": actual[node]} for node in expected if expected[node] != actual[node]}
    finally:
        store.close()
    return metrics


# ---------------------------------------------------------------------------
# Macro replay on held-out sessions
# ---------------------------------------------------------------------------
ROUTINE_MIGRATE = [
    ("read_file", "path", "config/auth.yaml"),
    ("read_file", "path", "db/schema.sql"),
    ("sql_query", "table", "users"),
    ("write_file", "path", "scripts/migrate.py"),
    ("pytest", "path", "tests/test_migration.py"),
    ("read_file", "path", "logs/pytest.log"),
    ("patch_file", "path", "config/auth.yaml"),
    ("pytest", "path", "tests/test_auth.py"),
    ("git_command", "command", "git status"),
    ("git_command", "command", "git commit -am migrate"),
]

ROUTINE_REPORT = [
    ("http_request", "endpoint", "https://api.example.com/v1/token"),
    ("pdf_reader", "file", "reports/q3.pdf"),
    ("doc_parser", "file", "reports/q3.pdf"),
    ("write_file", "path", "reports/summary.md"),
    ("read_file", "path", "reports/summary.md"),
    ("pytest", "path", "tests/test_report.py"),
    ("git_command", "command", "git status"),
    ("read_file", "path", "logs/report.log"),
    ("write_file", "path", "reports/final.md"),
    ("git_command", "command", "git commit -am report"),
]

ROUTINES = {"migrate": ROUTINE_MIGRATE, "report": ROUTINE_REPORT}


def _routine_params(routine, variant: int) -> List[Dict[str, Any]]:
    params = []
    for _, key, value in routine:
        value = value.replace("auth", f"auth_v{variant + 1}").replace("q3", f"q{variant + 3}")
        params.append({key: value})
    return params


def _seed_routine(store, session: str, routine_name: str, variant: int) -> None:
    store.new_session(session, "scope-macro")
    routine = ROUTINES[routine_name]
    for step, ((tool, _, _), params) in enumerate(zip(routine, _routine_params(routine, variant)), start=1):
        store.matp_index_trajectory(
            step_id=step,
            sub_goal=f"{routine_name} routine step {step}",
            tool_name=tool,
            parameters=params,
            status="SUCCESS",
            raw_output="ok",
        )


def run_macro_replay() -> Dict[str, Any]:
    db_path = RESULTS / "replay_macro.db"
    store = MATPMemoryStore(db_path=str(db_path), auto_migrate=False, enable_queue=False, session_id="macro", scope_id="scope-macro")
    store.clear()
    try:
        for variant in range(3):
            _seed_routine(store, f"train-migrate-{variant}", "migrate", variant)
            _seed_routine(store, f"train-report-{variant}", "report", variant)
        skills = store.mine_macros(window=10, stride=10, min_support=3)
        promoted = [s for s in skills if s.get("status") == "promoted" and s.get("level") == 2]

        # Held-out variants: the miner has never seen these literals/sessions.
        heldout = [
            ("heldout-migrate", "migrate", 3, "migrate the auth schema and run the tests"),
            ("heldout-report", "report", 3, "parse the quarterly cost report and commit the summary"),
        ]
        records = []
        for session, routine_name, variant, goal in heldout:
            _seed_routine(store, session, routine_name, variant)
            expected_tools = [canonical_tool(tool) for tool, _, _ in ROUTINES[routine_name]]
            expected_params = _routine_params(ROUTINES[routine_name], variant)
            matches = store.match_macro(goal, top_k=3)
            if not matches:
                records.append({"session": session, "retrieved": False})
                continue
            skill = matches[0]
            expanded_tools = [canonical_tool(step["tool"]) for step in skill.get("steps", [])]
            # Bind each slot to the held-out literal for its step/key.
            bindings = {}
            for param in skill.get("params", []):
                index = param.get("step_index", 0)
                key = param.get("key")
                if 0 <= index < len(expected_params) and key in expected_params[index]:
                    bindings[param["name"]] = expected_params[index][key]
            try:
                expanded = store.expand_macro(skill["skill_id"], bindings)
            except ValueError as exc:
                records.append({"session": session, "retrieved": True, "error": str(exc)})
                continue
            actual_tools = [canonical_tool(step["tool"]) for step in expanded["steps"]]
            step_coverage = sum(
                1
                for expected, actual in zip(expected_tools, actual_tools)
                if expected == actual
            ) / max(1, len(expected_tools))
            param_matches = 0
            param_total = 0
            for index, step in enumerate(expanded["steps"]):
                if index >= len(expected_params):
                    break
                for key, expected_value in expected_params[index].items():
                    param_total += 1
                    if str(step["parameters"].get(key)) == str(expected_value):
                        param_matches += 1
            compact_tokens = sum(
                len(store.get_trajectory(f"{session}:{step}")["compact_text"]) // 4
                for step in range(1, len(expected_tools) + 1)
            )
            macro_tokens = len(json.dumps(skill.get("steps", []))) // 4
            records.append(
                {
                    "session": session,
                    "retrieved": True,
                    "correct_routine": actual_tools == expected_tools,
                    "tool_sequence_match": round(step_coverage, 4),
                    "binding_accuracy": round(param_matches / max(1, param_total), 4),
                    "match_score": skill.get("match_score"),
                    "trajectory_tokens": compact_tokens,
                    "macro_definition_tokens": macro_tokens,
                    "token_reduction_pct": round((1 - macro_tokens / max(1, compact_tokens)) * 100.0, 2),
                }
            )
        return {
            "skills_mined": len(skills),
            "promoted_level2": len(promoted),
            "heldout": records,
            "mean_tool_sequence_match": round(statistics.mean([r.get("tool_sequence_match", 0.0) for r in records]), 4),
            "mean_binding_accuracy": round(statistics.mean([r.get("binding_accuracy", 0.0) for r in records]), 4),
        }
    finally:
        store.close()


# ---------------------------------------------------------------------------
# Worker / queue overhead on the indexing hot path
# ---------------------------------------------------------------------------
def _worker_overhead(queue_enabled: bool, db_name: str) -> Dict[str, Any]:
    db_path = RESULTS / db_name
    store = MATPMemoryStore(
        db_path=str(db_path), auto_migrate=False, enable_queue=queue_enabled, session_id="overhead", scope_id="overhead"
    )
    store.clear()
    from src.matp.benchmark import generate_benchmark_dataset

    dataset, _ = generate_benchmark_dataset()
    durations = []
    try:
        for item in dataset:
            start = time.perf_counter()
            store.matp_index_trajectory(
                step_id=item["step_id"],
                sub_goal=item["sub_goal"],
                tool_name=item["tool_name"],
                parameters=item["parameters"],
                status=item["status"],
                raw_output=item["raw_output"],
            )
            durations.append((time.perf_counter() - start) * 1000.0)
        stats = store.queue.stats()
    finally:
        store.close()
    return {
        "queue_enabled": queue_enabled,
        "median_index_ms": round(statistics.median(durations), 4),
        "total_index_ms": round(sum(durations), 4),
        "queued_jobs": stats.get("queued", 0),
    }


def run(output_dir: Path = RESULTS) -> Dict[str, Any]:
    global RESULTS
    RESULTS = Path(output_dir)
    RESULTS.mkdir(parents=True, exist_ok=True)
    return {
        "dag_invalidation": run_dag_replay(),
        "macro_replay": run_macro_replay(),
        "worker_overhead": {
            "without_queue": _worker_overhead(False, "replay_overhead_off.db"),
            "with_queue": _worker_overhead(True, "replay_overhead_on.db"),
        },
    }


def main() -> int:
    results = run()
    path = RESULTS / "replay_results.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    dag = results["dag_invalidation"]
    macro = results["macro_replay"]
    print("=" * 68)
    print("REPLAY BENCHMARKS")
    print("=" * 68)
    print(f"DAG label accuracy: {dag['accuracy'] * 100:.1f}% ({dag['correct']}/{dag['total']})")
    if dag["misclassified"]:
        print("  misclassified:", json.dumps(dag["misclassified"]))
    print(f"Macro: {macro['skills_mined']} skills mined, {macro['promoted_level2']} promoted")
    print(f"  held-out tool-sequence match: {macro['mean_tool_sequence_match'] * 100:.1f}%")
    print(f"  held-out binding accuracy:     {macro['mean_binding_accuracy'] * 100:.1f}%")
    overhead = results["worker_overhead"]
    print(
        "Index latency: "
        f"{overhead['without_queue']['median_index_ms']:.3f} ms without queue vs "
        f"{overhead['with_queue']['median_index_ms']:.3f} ms with queue "
        f"({overhead['with_queue']['queued_jobs']} jobs queued)"
    )
    print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
