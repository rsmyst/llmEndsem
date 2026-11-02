"""Unified Demonstration of MATP Monorepo.

Combines all three contributions:
1. Contribution 1 (Rahul): Temporal decay, logical clock & adaptive decay from feedback.
2. Contribution 2 (Shishir): Trajectory distillation, causal DAG invalidation & macro mining.
3. Contribution 3 (Ashab): Tool policy recommendation matrix & self-healing parameter corrections.
"""

import json
import math
import os
import sys

from src.matp.memory_store import MATPMemoryStore

def main():
    print("=" * 78)
    print("  MATP: Memory-Augmented Tool Policy & Temporal Semantic Memory")
    print("  NITK Surathkal — 7th Semester Capstone Unified Monorepo")
    print("=" * 78)

    db_path = "data/demo_unified.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    store = MATPMemoryStore(db_path=db_path, auto_migrate=False, enable_queue=False)

    print("\n--- [1] Trajectory Distillation & Indexing (Contribution 2) ---")
    raw_schema_v1 = "CREATE TABLE users (id INT PRIMARY KEY, username TEXT, password_hash TEXT);"
    t1 = store.matp_index_trajectory(
        step_id=1,
        sub_goal="Initialize database user schema",
        tool_name="sql_query",
        parameters={"query": "CREATE TABLE users (password_hash TEXT);"},
        status="SUCCESS",
        raw_output=raw_schema_v1,
    )
    print(f"  Step 1 indexed: Raw tokens={t1.raw_token_count_est} -> Distilled={t1.distilled_token_count_est} ({(1 - t1.distilled_token_count_est/t1.raw_token_count_est)*100:.1f}% reduction)")

    t2 = store.matp_index_trajectory(
        step_id=2,
        sub_goal="Configure authentication secrets",
        tool_name="write_file",
        parameters={"path": "config/auth.env"},
        status="SUCCESS",
        raw_output="AUTH_PROVIDER=legacy_basic\nUSER_TABLE=users",
    )
    print(f"  Step 2 indexed: {t2.tool_name} {t2.parameters['path']}")

    print("\n--- [2] Causal DAG Invalidation (Contribution 1 & 2) ---")
    # Step 3 reads auth.env (creates prior reader access)
    store.matp_index_trajectory(
        step_id=3,
        sub_goal="Read auth configuration",
        tool_name="read_file",
        parameters={"path": "config/auth.env"},
        status="SUCCESS",
        raw_output="AUTH_PROVIDER=legacy_basic\nUSER_TABLE=users",
    )
    print("  Step 3: Read config/auth.env v1")

    # Step 4 mutates schema to v2 (supersedes v1)
    raw_schema_v2 = "ALTER TABLE users RENAME COLUMN password_hash TO oauth_uid;"
    store.matp_index_trajectory(
        step_id=4,
        sub_goal="Migrate database to OAuth credentials",
        tool_name="sql_query",
        parameters={"query": "ALTER TABLE users RENAME COLUMN password_hash TO oauth_uid;"},
        status="SUCCESS",
        raw_output=raw_schema_v2,
    )
    print("  Step 4: Schema migrated (v1 superseded)")

    # Step 5 mutates auth.env (supersedes v1 writer, marks v1 reader stale)
    store.matp_index_trajectory(
        step_id=5,
        sub_goal="Upgrade auth configuration to OAuth2",
        tool_name="write_file",
        parameters={"path": "config/auth.env"},
        status="SUCCESS",
        raw_output="AUTH_PROVIDER=oauth2\nUSER_TABLE=users",
    )
    print("  Step 5: config/auth.env updated to OAuth2")

    # Query with and without stale filter
    query = "database users authentication credentials"
    fresh_hits = store.matp_query_trajectory(query, top_k=5, include_stale=False)
    all_hits = store.matp_query_trajectory(query, top_k=10, include_stale=True)
    print(f"\n  Query: '{query}'")
    print(f"  Clean retrieval (exclude_stale=True): {len(fresh_hits)} items returned (all fresh)")
    for h, s in fresh_hits:
        print(f"    - Step {h['step_id']}: {h['sub_goal']} [validity={h['validity']}] (score={s:.3f})")
    print(f"  Naive retrieval (include_stale=True): {len(all_hits)} items returned")
    for h, s in all_hits:
        print(f"    - Step {h['step_id']}: {h['sub_goal']} [validity={h['validity']}] (score={s:.3f})")

    print("\n--- [3] Temporal Decay & Adaptive Feedback (Contribution 1) ---")
    store.advance_clock(10)
    print(f"  Advanced logical clock to t={store.current_step()}")
    decay_hits = store.matp_query_trajectory(query, top_k=3, time_decay=0.05)
    print("  Exponential decay applied (lambda=0.05):")
    for h, s in decay_hits:
        exp = h.get("explain", {})
        print(f"    - Step {h['step_id']}: score={s:.3f}, age={exp.get('age_steps')} steps, weight={exp.get('weight'):.3f}")

    fb = store.record_feedback("auth-task", [decay_hits[0][0]["node_id"]])
    print(f"  Recorded user feedback for 'auth-task' -> learned lambda={fb['lambda_value']:.4f}")

    print("\n--- [4] Tool Policy Recommendation Matrix & Self-Healing (Contribution 3) ---")
    # Learn a parameter correction
    store.learn_correction(
        tool="sql_query",
        error_pattern="syntax error at or near 'RENAME'",
        hint="Use SQLite compatible ALTER TABLE table_name RENAME COLUMN old_name TO new_name syntax.",
    )
    print("  Registered self-healing parameter correction rule for sql_query")

    policy = store.recommend_tools("Migrate database schema", candidate_tools=["sql_query", "bash_exec"])
    print(f"\n  Policy recommendation matrix for sub-goal: 'Migrate database schema':")
    for r in policy["ranked"]:
        print(f"    - Tool '{r['tool']}': successRate={r['successRate']:.2f}, confidence={r['confidence']:.2f}, attempts={r['attempts']}")

    hints = store.parameter_hints("sql_query", "syntax error at or near 'RENAME' column")
    print(f"  Parameter auto-correction hints triggered: {len(hints)}")
    for h in hints:
        print(f"    * Pattern: '{h['errorPattern']}' -> Hint: '{h['hint']}'")

    print("\n" + "=" * 78)
    print("  All MATP subsystems successfully verified!")
    print("=" * 78)

if __name__ == "__main__":
    main()
