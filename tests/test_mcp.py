from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


class MCPProcess:
    def __init__(self, db_path: Path):
        env = os.environ.copy()
        env.update(
            {
                "MATP_DB": str(db_path),
                "MATP_LEGACY_JSON": str(db_path.parent / "does-not-exist.json"),
                "MATP_NEURAL": "off",
                "PYTHONPATH": str(ROOT),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
        )
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "src.matp.mcp_server"],
            cwd=str(ROOT),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            env=env,
        )
        self._next_id = 0

    def request(self, method: str, params=None):
        self._next_id += 1
        payload = {"jsonrpc": "2.0", "id": self._next_id, "method": method}
        if params is not None:
            payload["params"] = params
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            stderr = self.proc.stderr.read()
            raise AssertionError(f"MCP server exited without a response. stderr:\n{stderr}")
        return json.loads(line)

    def call(self, name: str, arguments=None):
        response = self.request("tools/call", {"name": name, "arguments": arguments or {}})
        assert "error" not in response, response
        text = response["result"]["content"][0]["text"]
        return json.loads(text)

    def close(self):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=5)


@pytest.fixture
def mcp(tmp_path):
    process = MCPProcess(tmp_path / "matp.db")
    yield process
    process.close()


def test_initialize_and_tools_list(mcp):
    init = mcp.request("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "pytest"}})
    assert init["result"]["serverInfo"]["name"] == "matp"
    assert "tools" in init["result"]["capabilities"]

    listing = mcp.request("tools/list")
    names = {tool["name"] for tool in listing["result"]["tools"]}
    assert {
        "matp_index_trajectory",
        "matp_query_trajectory",
        "matp_get_lineage",
        "matp_mine_macros",
        "matp_worker_status",
    } <= names
    assert len(names) == 25


def test_index_query_and_invalidate_over_stdio(mcp):
    mcp.call(
        "matp_index_trajectory",
        {
            "step_id": 1,
            "sub_goal": "read users",
            "tool_name": "sql_query",
            "parameters": {"query": "SELECT * FROM users", "table": "users"},
            "status": "SUCCESS",
            "raw_output": json.dumps([{"id": i, "email": f"u{i}@x.com"} for i in range(10)]),
            "session_id": "mcp-test",
            "scope_id": "mcp-scope",
        },
    )
    mcp.call(
        "matp_index_trajectory",
        {
            "step_id": 2,
            "sub_goal": "migrate users",
            "tool_name": "sql_query",
            "parameters": {"query": "ALTER TABLE users ADD COLUMN sso TEXT", "table": "users"},
            "status": "SUCCESS",
            "raw_output": "ok",
            "session_id": "mcp-test",
            "scope_id": "mcp-scope",
        },
    )

    lineage = mcp.call("matp_get_lineage", {"node_id": "mcp-test:1"})
    assert lineage["node"]["validity"] == "stale"

    current = mcp.call(
        "matp_query_trajectory", {"query": "users", "top_k": 5, "scope_id": "mcp-scope", "explain": True}
    )
    by_id = {item["node_id"]: item for item in current}
    # Stale state stays visible but is down-weighted.
    assert by_id["mcp-test:2"]["explain"]["weight"] == 1.0
    assert by_id["mcp-test:1"]["explain"]["weight"] < 1.0

    # A second write supersedes the first; superseded nodes need include_stale.
    mcp.call(
        "matp_index_trajectory",
        {
            "step_id": 3,
            "sub_goal": "index users",
            "tool_name": "sql_query",
            "parameters": {"query": "CREATE INDEX idx ON users(sso)", "table": "users"},
            "status": "SUCCESS",
            "raw_output": "ok",
            "session_id": "mcp-test",
            "scope_id": "mcp-scope",
        },
    )
    lineage = mcp.call("matp_get_lineage", {"node_id": "mcp-test:2"})
    assert lineage["node"]["validity"] == "superseded"

    current = mcp.call("matp_query_trajectory", {"query": "users", "top_k": 5, "scope_id": "mcp-scope"})
    assert "mcp-test:2" not in {item["node_id"] for item in current}

    with_stale = mcp.call(
        "matp_query_trajectory", {"query": "users", "top_k": 5, "scope_id": "mcp-scope", "include_stale": True}
    )
    assert "mcp-test:2" in {item["node_id"] for item in with_stale}


def test_unknown_tool_returns_error(mcp):
    response = mcp.request("tools/call", {"name": "not_a_tool", "arguments": {}})
    assert response["error"]["code"] == -32601


def test_stats_and_worker_status(mcp):
    stats = mcp.call("matp_stats")
    assert stats["memory"]["total_entries"] == 0
    status = mcp.call("matp_worker_status")
    assert status["neural_mode"] == "off"
