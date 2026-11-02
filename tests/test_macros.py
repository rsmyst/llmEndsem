from __future__ import annotations

ROUTINE_A = [
    ("sql_query", {"query": "SELECT * FROM users", "table": "users"}),
    ("read_file", {"path": "config/auth.yaml"}),
    ("write_file", {"path": "scripts/migrate.py"}),
    ("pytest", {"path": "tests/test_migration.py"}),
    ("git_command", {"command": "git status"}),
    ("read_file", {"path": "logs/pytest.log"}),
    ("patch_file", {"path": "config/auth.yaml"}),
    ("pytest", {"path": "tests/test_auth.py"}),
    ("git_command", {"command": "git diff"}),
    ("read_file", {"path": "README.md"}),
]

ROUTINE_B = [
    ("http_request", {"endpoint": "https://api.example.com/v1/token", "method": "GET"}),
    ("pdf_reader", {"file": "reports/q3.pdf"}),
    ("doc_parser", {"file": "reports/q3.pdf"}),
    ("write_file", {"path": "reports/summary.md"}),
    ("read_file", {"path": "reports/summary.md"}),
    ("pytest", {"path": "tests/test_report.py"}),
    ("git_command", {"command": "git status"}),
    ("read_file", {"path": "logs/report.log"}),
    ("write_file", {"path": "reports/final.md"}),
    ("git_command", {"command": "git commit -am report"}),
]


def seed_routine(store, session, routine, variant=0):
    store.new_session(session, "scope1")
    for step_id, (tool, params) in enumerate(routine, start=1):
        params = dict(params)
        for key, value in list(params.items()):
            if isinstance(value, str) and ("auth" in value or "q3" in value or "migrate" in value):
                params[key] = value.replace("q3", f"q{variant + 3}").replace("auth", f"auth_v{variant + 1}")
        store.matp_index_trajectory(
            step_id=step_id,
            sub_goal=f"{tool} step {step_id}",
            tool_name=tool,
            parameters=params,
            status="SUCCESS",
            raw_output="ok",
        )


def test_mine_repeated_routine(store):
    for i in range(3):
        seed_routine(store, f"routine-{i}", ROUTINE_A, variant=i)
    skills = store.mine_macros(window=10, min_support=3, threshold=0.8)
    assert skills
    skill = max(skills, key=lambda s: s["support"])
    assert skill["support"] >= 3
    assert skill["status"] == "promoted"
    assert skill["level"] == 2
    assert len(skill["steps"]) == 10
    assert skill["success_rate"] == 1.0
    assert any(p["name"].startswith("path") for p in skill["params"])
    stats = store.macro_stats()
    assert stats["skills"] >= 1
    assert stats["coverage_pct"] > 0


def test_match_and_expand_macro(store):
    for i in range(3):
        seed_routine(store, f"routine-{i}", ROUTINE_A, variant=i)
    store.mine_macros(window=10, min_support=3)
    matches = store.match_macro("migrate the auth schema and run tests", top_k=3)
    assert matches
    skill = matches[0]
    bindings = {p["name"]: p.get("example") for p in skill.get("params", [])}
    expanded = store.expand_macro(skill["skill_id"], bindings)
    assert len(expanded["steps"]) == 10
    assert all(
        "{" not in str(value)
        for step in expanded["steps"]
        for value in step["parameters"].values()
    )


def test_missing_binding_raises(store):
    for i in range(3):
        seed_routine(store, f"routine-{i}", ROUTINE_A, variant=i)
    skill = store.mine_macros(window=10, min_support=3)[0]
    try:
        store.expand_macro(skill["skill_id"], {})
    except ValueError as exc:
        assert "Missing macro bindings" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError")


def test_macro_demotion_after_repeated_failures(store):
    for i in range(3):
        seed_routine(store, f"routine-{i}", ROUTINE_A, variant=i)
    skill = store.mine_macros(window=10, min_support=3)[0]
    assert skill["status"] == "promoted"
    for _ in range(3):
        updated = store.record_macro_use(skill["skill_id"], success=False)
    assert updated["status"] == "deprecated"
    assert updated["failures"] == 3


def test_hierarchical_mining(store):
    # Two distinct 10-step routines per session, repeated three times.
    for i in range(3):
        store.new_session(f"combo-{i}", "scope1")
        for step_id, (tool, params) in enumerate(ROUTINE_A, start=1):
            store.matp_index_trajectory(step_id, f"a{step_id}", tool, dict(params), "SUCCESS", "ok")
        for step_id, (tool, params) in enumerate(ROUTINE_B, start=11):
            store.matp_index_trajectory(step_id, f"b{step_id}", tool, dict(params), "SUCCESS", "ok")
    skills = store.mine_macros(window=10, stride=10, min_support=3, hierarchical=True)
    level2 = [s for s in skills if s["level"] == 2]
    assert len(level2) >= 2
    assert any(s["level"] == 3 for s in skills)
