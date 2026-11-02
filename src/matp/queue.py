"""Durable async job queue used by the neural background worker.

The agent hot path only performs a single INSERT; every network call to Ollama
happens in a separate worker process.  Jobs are leased so a crashed worker's
work is automatically reclaimed instead of being lost.
"""

from __future__ import annotations

import json
import threading
import time
import uuid
from typing import Any, Dict, List, Optional

from . import config
from .storage import get_meta, set_meta, transaction


class JobQueue:
    def __init__(self, conn, lease_seconds: Optional[int] = None):
        self.conn = conn
        self.lease_seconds = config.WORKER_LEASE_SECONDS if lease_seconds is None else lease_seconds
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ write
    def enqueue(
        self,
        kind: str,
        payload: Dict[str, Any],
        priority: int = 5,
        idempotency_key: Optional[str] = None,
        max_attempts: int = 3,
        delay: float = 0.0,
    ) -> str:
        now = time.time()
        job_id = "job_" + uuid.uuid4().hex
        with self._lock:
            if idempotency_key:
                existing = self.conn.execute(
                    "SELECT id FROM jobs WHERE idempotency_key = ?", (idempotency_key,)
                ).fetchone()
                if existing:
                    return existing["id"]
            self.conn.execute(
                """INSERT INTO jobs(id, kind, payload, priority, state, attempts, max_attempts,
                                    available_at, created_at, idempotency_key)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    job_id,
                    kind,
                    json.dumps(payload),
                    int(priority),
                    "queued",
                    0,
                    int(max_attempts),
                    now + float(delay),
                    now,
                    idempotency_key,
                ),
            )
        return job_id

    def claim(self, worker_id: Optional[str] = None, lease_seconds: Optional[int] = None) -> Optional[Dict[str, Any]]:
        lease = self.lease_seconds if lease_seconds is None else lease_seconds
        with self._lock:
            self.recover_expired()
            now = time.time()
            row = self.conn.execute(
                """SELECT id FROM jobs
                   WHERE state='queued' AND available_at <= ?
                   ORDER BY priority ASC, created_at ASC LIMIT 1""",
                (now,),
            ).fetchone()
            if row is None:
                return None
            job_id = row["id"]
            try:
                with transaction(self.conn):
                    cur = self.conn.execute(
                        """UPDATE jobs SET state='running', started_at=?, lease_until=?,
                                  attempts=attempts+1, last_error=NULL
                           WHERE id=? AND state='queued'""",
                        (now, now + lease, job_id),
                    )
                    if cur.rowcount != 1:
                        return None
            except Exception:
                return None
            return self.get(job_id)

    def complete(self, job_id: str, result: Any = None) -> None:
        with self._lock:
            self.conn.execute(
                """UPDATE jobs SET state='done', finished_at=?, lease_until=NULL,
                          result=? WHERE id=?""",
                (time.time(), json.dumps(result) if result is not None else None, job_id),
            )

    def fail(self, job_id: str, error: str, retry: bool = True) -> None:
        with self._lock:
            row = self.get(job_id)
            if row is None:
                return
            attempts = int(row["attempts"])
            max_attempts = int(row["max_attempts"])
            now = time.time()
            if retry and attempts < max_attempts:
                backoff = min(300.0, 2.0 ** attempts)
                self.conn.execute(
                    """UPDATE jobs SET state='queued', available_at=?, lease_until=NULL, last_error=?
                       WHERE id=?""",
                    (now + backoff, str(error)[:2000], job_id),
                )
            else:
                self.conn.execute(
                    """UPDATE jobs SET state='failed', finished_at=?, lease_until=NULL, last_error=?
                       WHERE id=?""",
                    (now, str(error)[:2000], job_id),
                )

    def recover_expired(self) -> int:
        with self._lock:
            now = time.time()
            cur = self.conn.execute(
                """UPDATE jobs
                   SET state = CASE WHEN attempts >= max_attempts THEN 'failed' ELSE 'queued' END,
                       available_at = ?, lease_until = NULL,
                       last_error = COALESCE(last_error, 'lease expired')
                   WHERE state='running' AND lease_until IS NOT NULL AND lease_until < ?""",
                (now, now),
            )
            return cur.rowcount

    def cancel(self, job_id: str) -> bool:
        with self._lock:
            cur = self.conn.execute(
                "UPDATE jobs SET state='cancelled', finished_at=?, lease_until=NULL WHERE id=? AND state='queued'",
                (time.time(), job_id),
            )
            return cur.rowcount == 1

    def purge(self, states: List[str] = None) -> int:
        states = states or ["done", "failed", "cancelled"]
        marks = ",".join("?" for _ in states)
        cur = self.conn.execute(f"DELETE FROM jobs WHERE state IN ({marks})", tuple(states))
        return cur.rowcount

    # ------------------------------------------------------------------ read
    def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
        return self._decode(row) if row else None

    def _decode(self, row) -> Dict[str, Any]:
        item = dict(row)
        try:
            item["payload"] = json.loads(item["payload"])
        except (TypeError, json.JSONDecodeError):
            item["payload"] = {}
        if item.get("result"):
            try:
                item["result"] = json.loads(item["result"])
            except json.JSONDecodeError:
                pass
        return item

    def stats(self) -> Dict[str, Any]:
        rows = self.conn.execute(
            "SELECT state, COUNT(*) AS n FROM jobs GROUP BY state"
        ).fetchall()
        counts = {r["state"]: int(r["n"]) for r in rows}
        now = time.time()
        oldest = self.conn.execute(
            "SELECT MIN(available_at) AS t FROM jobs WHERE state='queued'"
        ).fetchone()
        return {
            "counts": counts,
            "queued": counts.get("queued", 0),
            "running": counts.get("running", 0),
            "done": counts.get("done", 0),
            "failed": counts.get("failed", 0),
            "oldest_queued_age_s": round(max(0.0, now - float(oldest["t"] or now)), 2),
        }

    # ------------------------------------------------------------------ heartbeat
    def heartbeat(self, worker_id: str, status: str = "ok", meta: Optional[Dict[str, Any]] = None) -> None:
        payload = {"worker_id": worker_id, "status": status, "ts": time.time(), "meta": meta or {}}
        set_meta(self.conn, f"worker:{worker_id}", json.dumps(payload))

    def worker_status(self) -> Dict[str, Any]:
        rows = self.conn.execute("SELECT key, value FROM meta WHERE key LIKE 'worker:%'").fetchall()
        workers = []
        for row in rows:
            try:
                workers.append(json.loads(row["value"]))
            except json.JSONDecodeError:
                continue
        return {"queue": self.stats(), "workers": workers}
