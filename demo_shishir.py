"""Interactive POC demo for MATP Contribution 2 (endsem).

Demonstrates:
1. Distillation + persistent indexing (SQL, traceback, tests).
2. Cross-session causal invalidation and validity-aware retrieval.
3. Hierarchical macro-action mining, matching and expansion.
4. Async worker status (queue is non-blocking).
"""

from __future__ import annotations

import json
import os
import re

from src.matp.memory_store import MATPMemoryStore

DB = os.environ.get("MATP_DEMO_DB", "data/matp_demo.db")


def routine_raw_output(tool: str, step_id: int) -> str:
    """Realistic verbose output so the demo's compression numbers are honest."""
    if tool == "read_file":
        return "\n".join(
            f"line_{i}: setting_{i}: value_{i}_metadata_padding_for_a_long_system_file"
            for i in range(60)
        )
    if tool == "pytest":
        return "collected 12 items\n" + "".join(
            f"tests/test_suite.py::test_case_{i} PASSED [ {i * 8:3d}%]\n" for i in range(12)
        ) + "12 passed in 0.41s\n"
    if tool == "git_command":
        return "On branch main\nChanges not staged for commit:\n" + "".join(
            f"\tmodified:   src/module_{i}.py\n" for i in range(20)
        )
    if tool == "sql_query":
        return json.dumps([{"user_id": i, "email": f"user_{i}@nitk.edu"} for i in range(25)], indent=2)
    return f"step {step_id} completed\n" + ("output padding line\n" * 40)


def banner(title: str) -> None:
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)


def run_demo() -> None:
    store = MATPMemoryStore(db_path=DB, session_id="refactor-A", scope_id="workspace", enable_queue=False)
    store.clear()

    banner("1. DISTILLATION + CAUSAL INDEXING")
    raw_sql = json.dumps(
        [{"user_id": 100 + i, "username": f"user_{i}", "email": f"user_{i}@nitk.edu", "role": "engineer"} for i in range(40)],
        indent=2,
    )
    first = store.matp_index_trajectory(
        step_id=1,
        sub_goal="Read active users table",
        tool_name="sql_query",
        parameters={"query": "SELECT user_id, username, email FROM users", "table": "users"},
        status="SUCCESS",
        raw_output=raw_sql,
    )
    print(f"  raw={first.raw_token_count_est} tokens  distilled={first.distilled_token_count_est} tokens")
    print(f"  entities={first.entities}")
    print(f"  triples={first.triples[:3]}")

    store.matp_index_trajectory(
        step_id=2,
        sub_goal="Migrate users table: add oauth_subject",
        tool_name="sql_query",
        parameters={"query": "ALTER TABLE users ADD COLUMN oauth_subject TEXT", "table": "users"},
        status="SUCCESS",
        raw_output="ALTER TABLE",
    )
    store.matp_index_trajectory(
        step_id=3,
        sub_goal="Run user integration tests",
        tool_name="pytest",
        parameters={"path": "tests/test_users.py"},
        status="SUCCESS",
        raw_output="12 passed in 0.45s",
        parent_step_id=2,
    )

    banner("2. CROSS-SESSION INVALIDATION")
    store.new_session("refactor-B", "workspace")
    store.matp_index_trajectory(
        step_id=1,
        sub_goal="Re-read users after migration",
        tool_name="sql_query",
        parameters={"query": "SELECT user_id, oauth_subject FROM users", "table": "users"},
        status="SUCCESS",
        raw_output=json.dumps([{"user_id": 1, "oauth_subject": "sub_1"}] * 10),
    )
    store.matp_index_trajectory(
        step_id=2,
        sub_goal="Add email index to users",
        tool_name="sql_query",
        parameters={"query": "CREATE INDEX idx_users_email ON users(email)", "table": "users"},
        status="SUCCESS",
        raw_output="CREATE INDEX",
    )
    # Now the original session's read (A:1) and the second read (B:1) are stale.
    for node in ("refactor-A:1", "refactor-B:1"):
        lineage = store.get_lineage(node)
        print(f"  {node}: {lineage['node']['validity']} -> {lineage['node']['reason']}")

    banner("3. VALIDITY-AWARE RETRIEVAL")
    for stale in (False, True):
        results = store.matp_query_trajectory("users table columns", top_k=3, include_stale=stale, explain=True)
        print(f"  include_stale={stale}")
        for entry, score in results:
            print(
                f"    {entry['node_id']:>14}  score={score:.3f}  validity={entry['validity']:<10} "
                f"weight={entry['explain']['weight']:.2f}  {entry['tuple']['sub_goal'][:40]}"
            )

    banner("4. HIERARCHICAL MACRO-ACTION MINING")
    routines = [
        ("read config/auth.yaml", "read_file", {"path": "config/auth.yaml"}),
        ("read db/schema.sql", "read_file", {"path": "db/schema.sql"}),
        ("apply schema migration", "sql_query", {"query": "ALTER TABLE users ADD COLUMN oauth_subject TEXT", "table": "users"}),
        ("write migration script", "write_file", {"path": "scripts/migrate_v3.py"}),
        ("run migration tests", "pytest", {"path": "tests/test_migration.py"}),
        ("read test log", "read_file", {"path": "logs/pytest.log"}),
        ("patch auth config", "patch_file", {"path": "config/auth.yaml"}),
        ("re-run auth tests", "pytest", {"path": "tests/test_auth.py"}),
        ("inspect git status", "git_command", {"command": "git status"}),
        ("commit migration", "git_command", {"command": "git commit -am migrate"}),
    ]
    for session_index in range(3):
        session = f"routine-{session_index}"
        store.new_session(session, "workspace")
        for step_id, (goal, tool, params) in enumerate(routines, start=1):
            params = dict(params)
            if "path" in params:
                params["path"] = params["path"].replace("auth", f"auth_v{session_index + 1}")
            store.matp_index_trajectory(
                step_id=step_id,
                sub_goal=f"{goal} [run {session_index}]",
                tool_name=tool,
                parameters=params,
                status="SUCCESS",
                raw_output=routine_raw_output(tool, step_id),
            )
    skills = store.mine_macros(window=10, min_support=3, hierarchical=True)
    print(f"  learned {len(skills)} skills; stats={store.macro_stats()}")
    for skill in skills[:5]:
        print(f"    {skill['skill_id']}  L{skill['level']}  support={skill['support']}  {skill['name']}")

    if skills:
        promoted = [s for s in skills if s.get("status") == "promoted"] or skills
        match = store.match_macro("migrate the auth schema and run tests", top_k=2)
        if match:
            print(f"  best match: {match[0]['name']} (score={match[0]['match_score']})")
            bindings = {
                p["name"]: (
                    re.sub(r"auth_v\d+", "auth_v9", str(p.get("example", "")))
                    if p.get("type") == "string"
                    else p.get("example")
                )
                for p in match[0].get("params", [])
            }
            expanded = store.expand_macro(match[0]["skill_id"], bindings)
            print(f"  expanded first step: {expanded['steps'][0]}")

    banner("5. SESSION SUMMARY + WORKER STATUS")
    print(store.get_session_summary("refactor-A"))
    print(json.dumps(store.worker_status(), indent=2))
    print(json.dumps(store.get_stats(), indent=2))


if __name__ == "__main__":
    run_demo()
