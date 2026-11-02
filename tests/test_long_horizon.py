"""Cross-repo long-horizon scenario (ported from contribution 1's demo):
naive retrieval floats superseded state; validity-weighted retrieval does not;
dependents of mutated resources are suppressed by the DAG."""

from __future__ import annotations

import pytest

QUERY = "user table columns and auth fields password_hash and oauth"

SCHEMA_V1 = "CREATE TABLE users (password_hash TEXT, legacy_pin TEXT);"
SCHEMA_V2 = "ALTER TABLE users RENAME COLUMN password_hash TO oauth_uid; DROP COLUMN legacy_pin;"
AUTHENV_V1 = "AUTH_PROVIDER=legacy_basic\nUSER_TABLE=password_hash"
AUTHENV_V2 = "AUTH_PROVIDER=oauth2\nUSER_TABLE=oauth_uid"


def _seed(store):
    store.matp_index_trajectory(1, "create schema", "sql_query", {"query": "1"}, "SUCCESS", SCHEMA_V1)
    store.matp_index_trajectory(2, "write auth config", "write_file", {"path": "config/auth.env"}, "SUCCESS", AUTHENV_V1)
    store.matp_index_trajectory(3, "read auth config v1", "read_file", {"path": "config/auth.env"}, "SUCCESS", AUTHENV_V1)
    # the ORM code depends on the v1 schema...
    store.matp_index_trajectory(
        4, "sync ORM", "write_file", {"path": "src/models/user.py"},
        "SUCCESS", "class User(Model): password_hash = Column(); legacy_pin = Column()",
    )
    # ...then the schema migrates and the config flips to oauth
    store.matp_index_trajectory(5, "migrate schema", "sql_query", {"query": "2"}, "SUCCESS", SCHEMA_V2)
    store.matp_index_trajectory(6, "update auth config", "write_file", {"path": "config/auth.env"}, "SUCCESS", AUTHENV_V2)


def test_naive_retrieval_floats_superseded_state(store):
    _seed(store)
    incl = store.matp_query_trajectory(QUERY, top_k=5, include_stale=True)
    stale = [e for e, _ in incl if e["validity"] != "fresh"]
    assert stale, "naive RAG must leak superseded versions"
    assert len(stale) >= 2  # schema v1 and auth.env v1 float up


def test_validity_filtered_retrieval_is_clean(store):
    _seed(store)
    hits = store.matp_query_trajectory(QUERY, top_k=5)
    assert all(e["validity"] == "fresh" for e, _ in hits)


def test_dependent_state_is_suppressed(store):
    _seed(store)
    store.matp_index_trajectory(7, "read auth config", "read_file", {"path": "config/auth.env"}, "SUCCESS", AUTHENV_V2)
    hits = store.matp_query_trajectory(QUERY, top_k=5)
    assert all(e["validity"] == "fresh" for e, _ in hits)
    # the superseded pair stays retrievable for provenance with include_stale
    incl = store.matp_query_trajectory(QUERY, top_k=10, include_stale=True)
    walk = [e for e, _ in incl if e["validity"] == "stale"]
    assert walk  # reads before the write are marked stale, not deleted
