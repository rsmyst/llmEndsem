"""Mini-SWE: a dependency-free SWE-bench-style transfer benchmark.

This is a *controlled simulation*, not an official SWE-bench score.  It exists
to isolate the Contribution 2 claim: cross-session episodic memory lets a
scripted fixer reuse the right repair pattern on a new repository state.

Protocol
--------
1. One solved example per bug type is indexed as a past trajectory.
2. Two unseen tasks per bug type are generated with different literal values.
3. ``cold`` applies a single fixed heuristic (the most common bug type).
4. ``memory`` retrieves the top-1 past trajectory with MATP and applies its
   stored repair rule.
5. Every repaired workspace is validated by running ``unittest`` in a subprocess.

Run:  python -m benchmarks.mini_swe
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.matp.memory_store import MATPMemoryStore  # noqa: E402

RESULTS = ROOT / "benchmarks" / "results"

BUG_TYPES: Dict[str, Dict[str, str]] = {
    "off_by_one": {
        "description": "off-by-one loop skips the last element of a list",
        "pattern": r"range\(len\((\w+)\) - 1\)",
        "replacement": r"range(len(\1))",
        "train_body": "def total(scores):\n    acc = 0\n    for i in range(len(scores) - 1):\n        acc += scores[i]\n    return acc\n",
        "train_test": "from {module} import total\nassert total([1, 2, 3]) == 6\nassert total([5]) == 5\n",
    },
    "wrong_operator": {
        "description": "threshold comparison uses strict greater-than instead of greater-or-equal",
        "pattern": r"> (\d+)",
        "replacement": r">= \1",
        "train_body": "def passed(score):\n    return score > 50\n",
        "train_test": "from {module} import passed\nassert passed(50) is True\nassert passed(51) is True\nassert passed(49) is False\n",
    },
    "wrong_constant": {
        "description": "tax multiplier uses the outdated constant",
        "pattern": r"\* 1\.5",
        "replacement": "* 1.18",
        "train_body": "def price_with_tax(amount):\n    return amount * 1.5\n",
        "train_test": "from {module} import price_with_tax\nassert round(price_with_tax(100), 2) == 118.0\n",
    },
    "missing_sorted": {
        "description": "top three returns unsorted leading values instead of the largest",
        "pattern": r"return (\w+)\[:3\]",
        "replacement": r"return sorted(\1)[-3:]",
        "train_body": "def top_three(values):\n    return values[:3]\n",
        "train_test": "from {module} import top_three\nassert top_three([9, 1, 8, 2, 7]) == [7, 8, 9]\n",
    },
    "wrong_slice": {
        "description": "slice returns the tail instead of the head of a list",
        "pattern": r"\[1:\]",
        "replacement": "[:1]",
        "train_body": "def first(values):\n    return values[1:]\n",
        "train_test": "from {module} import first\nassert first([4, 5, 6]) == [4]\n",
    },
}

# Two held-out tasks per bug type; literals differ from the training example.
EVAL_TASKS: List[Dict[str, str]] = [
    {
        "id": "offbyone_a",
        "bug_type": "off_by_one",
        "goal": "Fix the off-by-one loop so every score is included when summing a list",
        "module": "scores_a",
        "body": "def total(scores):\n    acc = 0\n    for i in range(len(scores) - 1):\n        acc += scores[i]\n    return acc\n",
        "test": "from {module} import total\nassert total([2, 4, 6]) == 12\nassert total([]) == 0\n",
    },
    {
        "id": "offbyone_b",
        "bug_type": "off_by_one",
        "goal": "Fix the off-by-one iteration bug that drops the final list element",
        "module": "scores_b",
        "body": "def total(scores):\n    acc = 0\n    for i in range(len(scores) - 1):\n        acc += scores[i]\n    return acc\n",
        "test": "from {module} import total\nassert total([10, 20, 30, 40]) == 100\n",
    },
    {
        "id": "operator_a",
        "bug_type": "wrong_operator",
        "goal": "Fix the threshold comparison so a score exactly at the cut-off passes",
        "module": "threshold_a",
        "body": "def passed(score):\n    return score > 60\n",
        "test": "from {module} import passed\nassert passed(60) is True\nassert passed(59) is False\n",
    },
    {
        "id": "operator_b",
        "bug_type": "wrong_operator",
        "goal": "Fix the strict greater-than boundary in the pass/fail check",
        "module": "threshold_b",
        "body": "def passed(score):\n    return score > 75\n",
        "test": "from {module} import passed\nassert passed(75) is True\nassert passed(74) is False\n",
    },
    {
        "id": "constant_a",
        "bug_type": "wrong_constant",
        "goal": "Update the outdated tax multiplier so prices use the new rate",
        "module": "tax_a",
        "body": "def price_with_tax(amount):\n    return amount * 1.5\n",
        "test": "from {module} import price_with_tax\nassert round(price_with_tax(200), 2) == 236.0\n",
    },
    {
        "id": "constant_b",
        "bug_type": "wrong_constant",
        "goal": "Replace the stale multiplier constant in the billing calculation",
        "module": "tax_b",
        "body": "def price_with_tax(amount):\n    return amount * 1.5\n",
        "test": "from {module} import price_with_tax\nassert round(price_with_tax(50), 2) == 59.0\n",
    },
    {
        "id": "sorted_a",
        "bug_type": "missing_sorted",
        "goal": "Return the three largest values in ascending order instead of the first three",
        "module": "top_a",
        "body": "def top_three(values):\n    return values[:3]\n",
        "test": "from {module} import top_three\nassert top_three([5, 100, 2, 99, 1]) == [5, 99, 100]\n",
    },
    {
        "id": "sorted_b",
        "bug_type": "missing_sorted",
        "goal": "Fix the leaderboard helper so it sorts before taking the top three",
        "module": "top_b",
        "body": "def top_three(values):\n    return values[:3]\n",
        "test": "from {module} import top_three\nassert top_three([3, 2, 1]) == [1, 2, 3]\n",
    },
    {
        "id": "slice_a",
        "bug_type": "wrong_slice",
        "goal": "Fix the slice so it returns only the first element",
        "module": "slice_a",
        "body": "def first(values):\n    return values[1:]\n",
        "test": "from {module} import first\nassert first([7, 8, 9]) == [7]\n",
    },
    {
        "id": "slice_b",
        "bug_type": "wrong_slice",
        "goal": "The helper returns the tail; make it return the head of the list",
        "module": "slice_b",
        "body": "def first(values):\n    return values[1:]\n",
        "test": "from {module} import first\nassert first(['a', 'b']) == ['a']\n",
    },
]


def _apply_rule(code: str, rule: Dict[str, str]) -> str:
    return re.sub(rule["pattern"], rule["replacement"], code)


def _run_test(workspace: Path, module: str, body: str, test: str) -> Dict[str, Any]:
    (workspace / f"{module}.py").write_text(body, encoding="utf-8")
    (workspace / "test_case.py").write_text(test.format(module=module), encoding="utf-8")
    (workspace / "_runner.py").write_text("import test_case\n", encoding="utf-8")
    proc = subprocess.run(
        [sys.executable, "-W", "ignore", "_runner.py"],
        cwd=str(workspace),
        capture_output=True,
        text=True,
        timeout=30,
    )
    return {"solved": proc.returncode == 0, "stderr": (proc.stderr or "")[-400:]}


def _normalise_rule(value: Any) -> Dict[str, str]:
    if isinstance(value, str):
        value = json.loads(value)
    return {"pattern": value["pattern"], "replacement": value["replacement"]}


def train_memory(store: MATPMemoryStore) -> None:
    store.clear()
    for bug_type, spec in BUG_TYPES.items():
        module = f"train_{bug_type}"
        fixed = _apply_rule(spec["train_body"], spec)
        store.matp_index_trajectory(
            step_id=len(store.trajectories) + 1,
            sub_goal=f"Repair pattern for {spec['description']}",
            tool_name="write_file",
            parameters={
                "path": f"{module}.py",
                "rule": json.dumps({"pattern": spec["pattern"], "replacement": spec["replacement"]}),
            },
            status="SUCCESS",
            raw_output=fixed,
            session_id="training",
            scope_id="mini-swe",
        )
        # The held-out tests are only used to create the solved training state.
        assert _run_test(Path(tempfile.mkdtemp()), module, fixed, spec["train_test"])["solved"]


def run(output_dir: Path = RESULTS) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    store = MATPMemoryStore(
        db_path=str(output_dir / "mini_swe.db"), auto_migrate=False, enable_queue=False, session_id="eval", scope_id="mini-swe"
    )
    try:
        train_memory(store)
        cold_rule = {
            "pattern": BUG_TYPES["off_by_one"]["pattern"],
            "replacement": BUG_TYPES["off_by_one"]["replacement"],
        }
        records = []
        for task in EVAL_TASKS:
            correct_rule = {
                "pattern": BUG_TYPES[task["bug_type"]]["pattern"],
                "replacement": BUG_TYPES[task["bug_type"]]["replacement"],
            }

            cold_workspace = Path(tempfile.mkdtemp(prefix="mini-swe-cold-"))
            cold_result = _run_test(
                cold_workspace, task["module"], _apply_rule(task["body"], cold_rule), task["test"]
            )

            matches = store.matp_query_trajectory(task["goal"], top_k=1)
            if matches:
                retrieved = _normalise_rule(matches[0][0]["tuple"]["parameters"]["rule"])
                retrieval_correct = retrieved == correct_rule
            else:
                retrieved = None
                retrieval_correct = False
            memory_workspace = Path(tempfile.mkdtemp(prefix="mini-swe-mem-"))
            code = _apply_rule(task["body"], retrieved) if retrieved else task["body"]
            memory_result = _run_test(memory_workspace, task["module"], code, task["test"])

            records.append(
                {
                    "id": task["id"],
                    "bug_type": task["bug_type"],
                    "cold_solved": cold_result["solved"],
                    "memory_solved": memory_result["solved"],
                    "retrieval_correct": retrieval_correct,
                    "neural_summary": None,
                }
            )
        total = len(records)
        return {
            "tasks": total,
            "cold_solved": sum(r["cold_solved"] for r in records),
            "memory_solved": sum(r["memory_solved"] for r in records),
            "retrieval_correct": sum(r["retrieval_correct"] for r in records),
            "cold_solve_rate": round(100.0 * sum(r["cold_solved"] for r in records) / total, 1),
            "memory_solve_rate": round(100.0 * sum(r["memory_solved"] for r in records) / total, 1),
            "retrieval_accuracy": round(100.0 * sum(r["retrieval_correct"] for r in records) / total, 1),
            "records": records,
            "note": "Controlled scripted-fixer simulation; not an official SWE-bench evaluation.",
        }
    finally:
        store.close()


def main() -> int:
    results = run()
    path = RESULTS / "mini_swe_results.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("=" * 68)
    print("MINI-SWE CROSS-SESSION TRANSFER (controlled simulation)")
    print("=" * 68)
    print(f"Tasks: {results['tasks']}")
    print(f"Cold heuristic solve rate: {results['cold_solve_rate']}% ({results['cold_solved']}/{results['tasks']})")
    print(f"MATP memory solve rate:    {results['memory_solve_rate']}% ({results['memory_solved']}/{results['tasks']})")
    print(f"Retrieval accuracy:        {results['retrieval_accuracy']}% ({results['retrieval_correct']}/{results['tasks']})")
    print(f"Saved: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
