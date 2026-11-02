from __future__ import annotations

READ_USERS = {"query": "SELECT * FROM users", "table": "users"}
WRITE_USERS = {"query": "ALTER TABLE users ADD COLUMN x TEXT", "table": "users"}
WRITE_USERS_2 = {"query": "CREATE INDEX idx ON users(x)", "table": "users"}


def validity(store, node_id):
    return store.get_trajectory(node_id)["validity"]


def test_reader_becomes_stale_after_write(store, index):
    index(1, "sql_query", READ_USERS)
    index(2, "sql_query", WRITE_USERS)
    assert validity(store, "s1:1") == "stale"
    assert validity(store, "s1:2") == "fresh"


def test_prior_writer_is_superseded(store, index):
    index(1, "sql_query", WRITE_USERS)
    index(2, "sql_query", WRITE_USERS_2)
    assert validity(store, "s1:1") == "superseded"
    assert validity(store, "s1:2") == "fresh"


def test_transitive_suspect_through_causal_parent(store, index):
    index(1, "sql_query", READ_USERS)
    index(2, "sql_query", WRITE_USERS)  # invalidates s1:1
    index(3, "read_file", {"path": "log.txt"}, parent=1)
    assert validity(store, "s1:1") == "stale"
    assert validity(store, "s1:3") == "suspect"
    assert store.get_trajectory("s1:3")["taint"] < 1.0


def test_cross_session_invalidation(store, index):
    index(1, "sql_query", READ_USERS, session="session-A")
    index(1, "sql_query", WRITE_USERS, session="session-B")
    assert validity(store, "session-A:1") == "stale"
    assert validity(store, "session-B:1") == "fresh"


def test_scope_isolation(store, index):
    index(1, "sql_query", READ_USERS, scope="workspace-A")
    index(1, "sql_query", WRITE_USERS, scope="workspace-B")
    assert validity(store, "s1:1") == "fresh"


def test_reexecution_heals_downstream(store, index):
    index(1, "sql_query", READ_USERS)
    index(2, "sql_query", WRITE_USERS)
    index(3, "read_file", {"path": "log.txt"}, parent=1)
    assert validity(store, "s1:1") == "stale"
    assert validity(store, "s1:3") == "suspect"

    index(1, "sql_query", READ_USERS)  # re-read current version
    assert validity(store, "s1:1") == "fresh"
    counts = store.recompute_validity("scope1")
    assert counts["fresh"] == 3
    assert validity(store, "s1:3") == "fresh"


def test_manual_override_and_later_write(store, index):
    index(1, "sql_query", READ_USERS)
    index(2, "sql_query", WRITE_USERS)
    assert store.confirm_state("s1:1", "verified manually") is True
    assert validity(store, "s1:1") == "fresh"
    assert store.get_trajectory("s1:1")["taint"] == 1.0

    index(3, "sql_query", WRITE_USERS_2)
    # A new write clears the override and invalidates the old read again.
    assert validity(store, "s1:1") == "stale"


def test_lineage_and_resource_history(store, index):
    index(1, "sql_query", READ_USERS)
    index(2, "sql_query", WRITE_USERS)
    index(3, "read_file", {"path": "log.txt"}, parent=2)
    lineage = store.get_lineage("s1:3")
    assert lineage["node"]["step_id"] == 3
    assert any(a["node_id"] == "s1:2" for a in lineage["ancestors"])

    history = store.query_dag("db:users", "scope1")
    assert history["resources"][0]["current_version"] == 1
    assert history["versions"][0]["node_id"] == "s1:2"
    assert any(d["node_id"] == "s1:1" for d in history["dependents"])


def test_reindex_same_node(store, index):
    index(1, "sql_query", READ_USERS)
    index(1, "sql_query", WRITE_USERS)  # same node now writes
    assert validity(store, "s1:1") == "fresh"
    history = store.query_dag("db:users", "scope1")
    assert history["resources"][0]["current_version"] == 1
