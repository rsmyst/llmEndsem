"""Tests for the unified policy + correction engine (ported from contribution 3)."""

from __future__ import annotations

import pytest


def test_policy_ranks_laplace_smoothed_tools(store):
    # sql_query: 3 attempts / 3 success; bash_exec: 1 failure.
    for i in range(3):
        store.matp_index_trajectory(
            step_id=i + 1,
            sub_goal="pull rows from the users table",
            tool_name="sql_query",
            parameters={"query": f"SELECT {i} FROM users"},
            status="SUCCESS",
            raw_output=f"[{{'id': 1}}] rows={i}",
        )
    store.matp_index_trajectory(
        step_id=10,
        sub_goal="grep the log file for misses",
        tool_name="bash_exec",
        parameters={"command": "grep misses app.log"},
        status="FAILURE",
        raw_output="grep: app.log: No such file or directory",
    )
    state = store.recommend_tools("pull rows from the users table", candidate_tools=["sql_query", "bash_exec"])
    ranked = {r["tool"]: r for r in state["ranked"]}
    sql = ranked["sql_query"]
    # Laplace smoothing: (3+1)/(3+2), confidence shrunk toward the 0.5 prior.
    assert sql["successRate"] == pytest.approx(4 / 5)
    expected_conf = (4 / 5) * (3 / 8) + 0.5 * (5 / 8)
    assert sql["confidence"] == pytest.approx(expected_conf, abs=1e-3)
    assert sql["confidence"] > ranked["bash_exec"]["confidence"]
    entry = [m for m in state["matrix"] if m["tool"] == "sql_query"][0]
    assert entry["bestColor"] == "green"
    assert entry["exampleParameters"] == {"query": "SELECT 0 FROM users"}


def test_policy_empty_state_renders_fallback(store):
    prompt = store.render_policy_prompt("an utterly novel sub-goal")
    assert "No historical tool policy" in prompt


def test_correction_exact_match_then_hint(store):
    rule = store.learn_correction(
        tool="sec_edgar_extractor",
        error_pattern="year must be an integer",
        hint="Pass the fiscal year as a 4-digit number (e.g. 2024), not a string.",
    )
    assert rule["id"]
    hints = store.parameter_hints(
        tool="sec_edgar_extractor",
        context="sec_edgar_extractor failed: year must be an integer",
    )
    assert hints, "exact tier should always win"
    assert hints[0]["similarity"] == 1.0
    assert hints[0]["hint"].startswith("Pass the fiscal year")
    # usage telemetry increments exactly once per injected hint
    listed = store.list_corrections(enabled_only=False)
    assert listed[0]["timesApplied"] == len(hints)


def test_correction_fuzzy_fallback_and_floor(store):
    store.learn_correction(
        tool="pdf_reader",
        error_pattern="context length exceeded",
        hint="Chunk the document and extract tables page by page.",
    )
    hits = store.parameter_hints(tool="pdf_reader", context="Error: context length exceeded while parsing resume.pdf")
    assert hits and hints_ok(hits)

    # a completely unrelated context must stay below the noise floor
    misses = store.parameter_hints(tool="pdf_reader", context="hello world foo bar")
    assert misses == []


def hints_ok(hits):
    return all(isinstance(h["similarity"], float) for h in hits)


def test_correction_wildcard_seen_once(store):
    store.learn_correction(tool="*", error_pattern="rate limit", hint="Back off and retry with exponential delay.")
    many = store.parameter_hints(tool="sql_query", context="Error: rate limit reached, retry blocked")
    assert len(many) == 1


def test_disable_correction(store):
    rule = store.learn_correction(tool="x", error_pattern="boom", hint="Do not press the red button.")
    assert store.disable_correction(rule["id"])["ok"] is True
    assert store.parameter_hints(tool="x", context="boom!") == []
