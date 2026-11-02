"""SQLite storage for MATP Contribution 2.

The original prototype rewrote a single JSON file on every index call, which is
O(n^2) and unsafe when a background worker is reading concurrently.  SQLite is
stdlib, single-file and gives us durable transactions for the trajectory store,
the causal DAG, the macro skill library and the async job queue.
"""

from __future__ import annotations

import json
import sqlite3
import time
from array import array
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

SCHEMA_VERSION = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    scope_id TEXT NOT NULL,
    created_at REAL NOT NULL,
    meta TEXT
);

CREATE TABLE IF NOT EXISTS trajectories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT NOT NULL UNIQUE,
    session_id TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    step_id INTEGER NOT NULL,
    tool_name TEXT NOT NULL,
    status TEXT NOT NULL,
    sub_goal TEXT NOT NULL,
    tuple_json TEXT NOT NULL,
    compact_text TEXT NOT NULL,
    raw_excerpt TEXT,
    raw_tokens INTEGER NOT NULL,
    distilled_tokens INTEGER NOT NULL,
    signature TEXT,
    revision INTEGER NOT NULL DEFAULT 1,
    neural_summary TEXT,
    neural_revision INTEGER,
    validity TEXT NOT NULL DEFAULT 'fresh',
    taint REAL NOT NULL DEFAULT 1.0,
    invalidated_by TEXT,
    reason TEXT,
    validity_override TEXT,
    created_at REAL NOT NULL,
    UNIQUE(session_id, step_id)
);
CREATE INDEX IF NOT EXISTS idx_traj_session ON trajectories(session_id, step_id);
CREATE INDEX IF NOT EXISTS idx_traj_scope ON trajectories(scope_id);
CREATE INDEX IF NOT EXISTS idx_traj_validity ON trajectories(validity);

CREATE TABLE IF NOT EXISTS embeddings (
    trajectory_id INTEGER PRIMARY KEY REFERENCES trajectories(id) ON DELETE CASCADE,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS resources (
    resource_id TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    kind TEXT NOT NULL,
    current_version INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (resource_id, scope_id)
);

CREATE TABLE IF NOT EXISTS resource_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    resource_id TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    version INTEGER NOT NULL,
    node_id TEXT NOT NULL,
    ts REAL NOT NULL,
    digest TEXT,
    UNIQUE(resource_id, scope_id, version)
);
CREATE INDEX IF NOT EXISTS idx_versions_node ON resource_versions(node_id);

CREATE TABLE IF NOT EXISTS accesses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    node_id TEXT NOT NULL,
    trajectory_id INTEGER NOT NULL,
    resource_id TEXT NOT NULL,
    scope_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    version_at_access INTEGER,
    digest TEXT,
    UNIQUE(node_id, resource_id, mode)
);
CREATE INDEX IF NOT EXISTS idx_access_resource ON accesses(resource_id, scope_id);

CREATE TABLE IF NOT EXISTS dag_edges (
    src TEXT NOT NULL,
    dst TEXT NOT NULL,
    kind TEXT NOT NULL,
    meta TEXT,
    PRIMARY KEY (src, dst, kind)
);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON dag_edges(dst);
CREATE INDEX IF NOT EXISTS idx_edges_src ON dag_edges(src);

CREATE TABLE IF NOT EXISTS corrections (
    id TEXT PRIMARY KEY,
    tool TEXT NOT NULL,
    error_pattern TEXT NOT NULL,
    hint TEXT NOT NULL,
    times_applied INTEGER NOT NULL DEFAULT 0,
    enabled INTEGER NOT NULL DEFAULT 1,
    searchable TEXT NOT NULL,
    dim INTEGER NOT NULL,
    vector BLOB NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_corrections_tool ON corrections(tool, enabled);

CREATE TABLE IF NOT EXISTS skills (
    skill_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    version INTEGER NOT NULL DEFAULT 1,
    level INTEGER NOT NULL DEFAULT 2,
    definition_json TEXT NOT NULL,
    support INTEGER NOT NULL DEFAULT 0,
    successes INTEGER NOT NULL DEFAULT 0,
    failures INTEGER NOT NULL DEFAULT 0,
    consecutive_failures INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'candidate',
    created_at REAL NOT NULL,
    last_used REAL
);

CREATE TABLE IF NOT EXISTS skill_members (
    skill_id TEXT NOT NULL,
    trajectory_id INTEGER NOT NULL,
    PRIMARY KEY (skill_id, trajectory_id)
);

CREATE TABLE IF NOT EXISTS jobs (
    id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL,
    priority INTEGER NOT NULL DEFAULT 5,
    state TEXT NOT NULL DEFAULT 'queued',
    attempts INTEGER NOT NULL DEFAULT 0,
    max_attempts INTEGER NOT NULL DEFAULT 3,
    available_at REAL NOT NULL,
    started_at REAL,
    finished_at REAL,
    lease_until REAL,
    last_error TEXT,
    result TEXT,
    idempotency_key TEXT UNIQUE,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_jobs_state ON jobs(state, available_at, priority);
"""


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), timeout=30.0, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute("PRAGMA busy_timeout=30000")
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.execute(
        "INSERT OR IGNORE INTO meta(key, value) VALUES('schema_version', ?)",
        (str(SCHEMA_VERSION),),
    )


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        raise
    else:
        conn.execute("COMMIT")


def float_list_to_blob(vector: Iterable[float]) -> bytes:
    return array("f", (float(x) for x in vector)).tobytes()


def blob_to_float_list(blob: bytes) -> List[float]:
    vec = array("f")
    vec.frombytes(blob)
    return list(vec)


def get_meta(conn: sqlite3.Connection, key: str, default: Optional[str] = None) -> Optional[str]:
    row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
    return row["value"] if row else default


def set_meta(conn: sqlite3.Connection, key: str, value: Any) -> None:
    conn.execute(
        "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (key, str(value)),
    )


def migrate_legacy_json(conn: sqlite3.Connection, json_path: str | Path) -> int:
    """Import the pre-endsem ``matp.memory.json`` store exactly once."""
    if get_meta(conn, "legacy_imported") == "1":
        return 0
    path = Path(json_path)
    if not path.exists():
        set_meta(conn, "legacy_imported", "1")
        return 0

    data = json.loads(path.read_text(encoding="utf-8"))
    trajectories = data.get("trajectories", [])
    embeddings = data.get("embeddings", [])
    imported = 0
    now = time.time()

    with transaction(conn):
        for idx, entry in enumerate(trajectories):
            tpl = entry.get("tuple", {})
            step_id = int(tpl.get("step_id", idx + 1))
            node_id = f"legacy:{step_id}:{idx}"
            try:
                cur = conn.execute(
                    """
                    INSERT INTO trajectories(
                        node_id, session_id, scope_id, step_id, tool_name, status, sub_goal,
                        tuple_json, compact_text, raw_excerpt, raw_tokens, distilled_tokens, signature,
                        revision, validity, taint, created_at
                    ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """,
                    (
                        node_id,
                        "legacy",
                        "legacy",
                        step_id,
                        tpl.get("tool_name", "unknown"),
                        tpl.get("status", "SUCCESS"),
                        tpl.get("sub_goal", ""),
                        json.dumps(tpl),
                        entry.get("compact_text", ""),
                        None,
                        int(entry.get("raw_output_tokens", 0)),
                        int(entry.get("distilled_tokens", 0)),
                        None,
                        1,
                        "fresh",
                        1.0,
                        now,
                    ),
                )
                traj_id = cur.lastrowid
            except sqlite3.IntegrityError:
                continue

            if idx < len(embeddings) and embeddings[idx]:
                vec = embeddings[idx]
                conn.execute(
                    "INSERT OR REPLACE INTO embeddings(trajectory_id, dim, vector) VALUES(?,?,?)",
                    (traj_id, len(vec), float_list_to_blob(vec)),
                )
            imported += 1

        set_meta(conn, "legacy_imported", "1")
        set_meta(conn, "legacy_imported_at", now)
    return imported
