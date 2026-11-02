from __future__ import annotations

import json

from src.matp.memory_store import MATPMemoryStore


def test_index_and_query_roundtrip(store, index, raw_sql):
    distilled = store.matp_index_trajectory(
        step_id=1,
        sub_goal="read users",
        tool_name="sql_query",
        parameters={"query": "SELECT * FROM users", "table": "users"},
        status="SUCCESS",
        raw_output=raw_sql,
    )
    assert distilled.raw_token_count_est > distilled.distilled_token_count_est
    assert "users" in distilled.entities

    results = store.matp_query_trajectory("users table", top_k=3)
    assert results
    entry, score = results[0]
    assert entry["node_id"] == "s1:1"
    assert entry["tuple"]["tool_name"] == "sql_query"
    assert score > 0

    stats = store.get_stats()
    assert stats["total_entries"] == 1
    assert stats["compression_ratio_pct"] > 0


def test_superseded_excluded_unless_include_stale(store, index):
    index(1, "write_file", {"path": "a.txt"}, "v1")
    index(2, "write_file", {"path": "a.txt"}, "v2")
    assert store.get_trajectory("s1:1")["validity"] == "superseded"

    current = [e["node_id"] for e, _ in store.matp_query_trajectory("a.txt", top_k=5)]
    assert "s1:1" not in current
    assert "s1:2" in current

    everything = [e["node_id"] for e, _ in store.matp_query_trajectory("a.txt", top_k=5, include_stale=True)]
    assert "s1:1" in everything


def test_tool_and_status_filters(store, index):
    index(1, "read_file", {"path": "a.txt"})
    index(2, "sql_query", {"query": "SELECT * FROM users", "table": "users"})
    index(3, "read_file", {"path": "b.txt"}, status="FAILURE")

    by_tool = store.matp_query_trajectory("file", filter_tool="read_file", top_k=5)
    assert {e["tuple"]["tool_name"] for e, _ in by_tool} == {"read_file"}

    by_status = store.matp_query_trajectory("file", filter_status="FAILURE", top_k=5)
    assert [e["node_id"] for e, _ in by_status] == ["s1:3"]


def test_entity_filter(store, index, raw_sql):
    store.matp_index_trajectory(
        1, "read users", "sql_query", {"query": "SELECT * FROM users", "table": "users"}, "SUCCESS", raw_sql
    )
    results = store.matp_query_trajectory("rows", filter_entity="users", top_k=3)
    assert results
    assert store.matp_query_trajectory("rows", filter_entity="nonexistent_entity", top_k=3) == []


def test_session_and_scope_filters(store, index):
    index(1, "read_file", {"path": "a.txt"}, session="other", scope="scope2")
    index(1, "read_file", {"path": "a.txt"})
    only_other = store.matp_query_trajectory("a.txt", session_id="other", top_k=5)
    assert [e["node_id"] for e, _ in only_other] == ["other:1"]
    only_scope2 = store.matp_query_trajectory("a.txt", scope_id="scope2", top_k=5)
    assert [e["node_id"] for e, _ in only_scope2] == ["other:1"]


def test_clear(store, index):
    index(1, "read_file", {"path": "a.txt"})
    assert store.get_stats()["total_entries"] == 1
    store.clear()
    assert store.get_stats()["total_entries"] == 0
    assert store.trajectories == []
    assert store.query_dag()["resources"] == []


def test_legacy_json_migration(tmp_path, raw_sql):
    legacy = tmp_path / "matp.memory.json"
    legacy.write_text(
        json.dumps(
            {
                "trajectories": [
                    {
                        "tuple": {"step_id": 1, "tool_name": "sql_query", "status": "SUCCESS", "sub_goal": "legacy", "parameters": {}},
                        "compact_text": "legacy compact text users",
                        "raw_output_tokens": 100,
                        "distilled_tokens": 10,
                    }
                ],
                "embeddings": [[0.0] * 8],
            }
        ),
        encoding="utf-8",
    )
    store = MATPMemoryStore(persistence_file=str(legacy), enable_queue=False, session_id="new", scope_id="scope")
    try:
        assert store.get_stats()["total_entries"] == 1
        assert store.trajectories[0]["tuple"]["sub_goal"] == "legacy"
        assert store.trajectories[0]["session_id"] == "legacy"
        # Import happens exactly once, even if the JSON still exists.
        store2 = MATPMemoryStore(persistence_file=str(legacy), enable_queue=False)
        assert store2.get_stats()["total_entries"] == 1
        store2.close()
    finally:
        store.close()
    assert (tmp_path / "matp.memory.db").exists()


def test_session_summary_marks_invalidated_steps(store, index):
    index(1, "sql_query", {"query": "SELECT * FROM users", "table": "users"})
    index(2, "sql_query", {"query": "ALTER TABLE users ADD COLUMN x TEXT", "table": "users"})
    summary = store.get_session_summary("s1")
    assert "Step 1" in summary and "STALE" in summary
    assert "Step 2" in summary and "fresh" not in summary.splitlines()[2].lower()


def test_neural_summary_revision_guard(store, index):
    index(1, "read_file", {"path": "a.txt"})
    assert store.update_neural_summary("s1:1", 1, "neural summary v1") is True
    assert store.get_trajectory("s1:1")["neural_summary"] == "neural summary v1"

    # Re-index bumps the revision, so the old neural summary must not apply.
    index(1, "read_file", {"path": "a.txt"})
    assert store.get_trajectory("s1:1")["neural_summary"] is None
    assert store.update_neural_summary("s1:1", 1, "stale write") is False


def test_knowledge_graph(store, raw_sql):
    store.matp_index_trajectory(
        1, "read users", "sql_query", {"query": "SELECT * FROM users", "table": "users"}, "SUCCESS", raw_sql
    )
    graph = store.get_knowledge_graph()
    assert graph["total_edges"] > 0
    assert "users" in graph["nodes"]
    related = store.query_knowledge_graph("users")
    assert related
    assert all("users" in (e["subject"] + e["object"]).lower() for e in related)


def test_scope_isolation_in_retrieval(store, index):
    index(1, "read_file", {"path": "a.txt"}, session="alpha", scope="alpha")
    index(1, "read_file", {"path": "b.txt"}, session="beta", scope="beta")
    alpha = store.matp_query_trajectory("file", scope_id="alpha", top_k=5)
    assert [e["node_id"] for e, _ in alpha] == ["alpha:1"]
    assert alpha[0][0]["tuple"]["parameters"]["path"].endswith("a.txt")
