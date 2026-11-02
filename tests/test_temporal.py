"""Unified-store tests for contribution-1 semantics: temporal decay, adaptive
per-task lambda from feedback, and the logical clock."""

from __future__ import annotations

import math

import pytest


SQL_V1 = "CREATE TABLE users (password_hash TEXT, legacy_pin TEXT);"
SQL_V2 = "ALTER TABLE users ADD COLUMN oauth_uid TEXT;"

QUERY_LEGACY = "user table columns password_hash and oauth fields"


def _seed(store):
    store.matp_index_trajectory(1, "create schema", "sql_query", {"query": "1"}, "SUCCESS", SQL_V1)
    store.matp_index_trajectory(2, "swap auth backend", "sql_query", {"query": "2"}, "SUCCESS", SQL_V2)
    store.advance_clock(9)  # logical clock moves to 11


def test_logical_clock_tracks_steps(store):
    _seed(store)
    assert store.current_step() == 11
    assert store.advance_clock(4) == 15


def test_temporal_decay_downweights_old_state(store):
    _seed(store)
    hits_flat = store.matp_query_trajectory(QUERY_LEGACY, top_k=5, explain=True)
    hits_decayed = store.matp_query_trajectory(QUERY_LEGACY, top_k=5, time_decay=0.05)

    # in both runs explain metadata is present
    assert all("explain" in e for e, _ in hits_flat)
    exp = hits_decayed[0][0]["explain"]
    assert exp["decay_lambda"] == pytest.approx(0.05)
    assert 0.0 < exp["weight"] <= 1.0

    # every decayed score must equal flat score * exp(-lambda * age)
    flat_by_node = {e["node_id"]: s for e, s in hits_flat}
    for e, s in hits_decayed:
        if e["node_id"] in flat_by_node and e["validity"] == "fresh":
            age = e["explain"]["age_steps"]
            assert s == pytest.approx(flat_by_node[e["node_id"]] * math.exp(-0.05 * age), abs=2e-4)


def test_invalid_time_decay_mode_raises(store):
    with pytest.raises(ValueError):
        store.matp_query_trajectory("q", time_decay="cranky")


def test_auto_decay_learns_from_feedback(store):
    _seed(store)
    hits = store.matp_query_trajectory(QUERY_LEGACY, top_k=2, time_decay="auto", task="oauth-refactor")
    # never seen before: default lambda
    assert store.decay_stats()["tasks"] == {}
    assert hits[0][0]["explain"]["decay_lambda"] == pytest.approx(0.05)

    store.record_feedback("oauth-refactor", [h[0]["node_id"] for h in hits])
    state = store.decay_stats()
    assert "oauth-refactor" in state["tasks"]
    lam = state["tasks"]["oauth-refactor"]["lambda_value"]
    # median useful age == 10 (clock=11, steps 1 & 2) -> target ln(2)/10 ~ 0.0693, clamped into [0.005, 0.5]
    assert 0.005 <= lam <= 0.5
    assert lam == pytest.approx(math.log(2.0) / 10.0, abs=0.01)

    # subsequent retrieval with "auto" uses the learned rate
    hits2 = store.matp_query_trajectory(QUERY_LEGACY, top_k=2, time_decay="auto", task="oauth-refactor")
    assert hits2[0][0]["explain"]["decay_lambda"] == pytest.approx(lam)

    # empty feedback cannot corrupt an existing estimate
    ret = store.record_feedback("oauth-refactor", [])
    assert ret["lambda_value"] == pytest.approx(lam)
