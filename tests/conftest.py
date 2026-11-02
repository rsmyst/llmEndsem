from __future__ import annotations

import json

import pytest

from src.matp.memory_store import MATPMemoryStore


@pytest.fixture
def store(tmp_path):
    s = MATPMemoryStore(
        db_path=str(tmp_path / "matp.db"),
        auto_migrate=False,
        enable_queue=False,
        session_id="s1",
        scope_id="scope1",
        workspace=str(tmp_path),
    )
    yield s
    s.close()


@pytest.fixture
def index(store):
    def _index(step_id, tool="read_file", params=None, raw="output", status="SUCCESS", parent=None, session=None, scope=None):
        return store.matp_index_trajectory(
            step_id=step_id,
            sub_goal=f"goal {step_id}",
            tool_name=tool,
            parameters=params or {"path": f"file_{step_id}.txt"},
            status=status,
            raw_output=raw,
            parent_step_id=parent,
            session_id=session,
            scope_id=scope,
        )

    return _index


@pytest.fixture
def raw_sql():
    return json.dumps([{"id": i, "name": f"user_{i}", "email": f"u{i}@x.com"} for i in range(40)], indent=2)
