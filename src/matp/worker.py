"""Non-blocking neural background worker.

The worker is a separate process (``python -m src.matp.worker --daemon``) that
leases jobs from the SQLite queue and enriches deterministic macros/summaries
with a local Ollama model.  The agent hot path never waits for it.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import threading
import time
import traceback
from typing import Any, Dict, Optional

from . import config
from .memory_store import MATPMemoryStore
from .ollama_client import OllamaClient, OllamaError, OllamaUnavailable


class Worker:
    def __init__(
        self,
        store: Optional[MATPMemoryStore] = None,
        db_path: Optional[str] = None,
        client: Optional[OllamaClient] = None,
        poll_interval: Optional[float] = None,
        worker_id: Optional[str] = None,
    ):
        self.owns_store = store is None
        self.store = store or MATPMemoryStore(
            db_path=db_path or str(config.DB_PATH),
            auto_migrate=False,
            enable_queue=True,
            session_id="worker",
            scope_id="worker",
        )
        self.queue = self.store.queue
        self.client = client or OllamaClient()
        self.poll_interval = config.WORKER_POLL_SECONDS if poll_interval is None else poll_interval
        self.worker_id = worker_id or f"{socket.gethostname()}:{os.getpid()}"
        self._stop = threading.Event()

    # ------------------------------------------------------------------ lifecycle
    def stop(self) -> None:
        self._stop.set()

    def close(self) -> None:
        if self.owns_store:
            self.store.close()

    # ------------------------------------------------------------------ jobs
    def run_once(self) -> bool:
        job = self.queue.claim(worker_id=self.worker_id)
        if job is None:
            return False
        try:
            result = self.handle(job)
            self.queue.complete(job["id"], result)
            self.queue.heartbeat(self.worker_id, "ok", {"last_job": job["id"], "kind": job["kind"]})
        except OllamaUnavailable as exc:
            self.queue.fail(job["id"], str(exc), retry=True)
            self.queue.heartbeat(self.worker_id, "degraded", {"error": str(exc)[:300]})
        except (OllamaError, ValueError, KeyError) as exc:
            self.queue.fail(job["id"], f"{type(exc).__name__}: {exc}", retry=True)
            self.queue.heartbeat(self.worker_id, "error", {"error": str(exc)[:300]})
        except Exception as exc:  # pragma: no cover - defensive
            self.queue.fail(job["id"], traceback.format_exc()[:2000], retry=True)
            self.queue.heartbeat(self.worker_id, "error", {"error": str(exc)[:300]})
        return True

    def run_forever(self, max_jobs: Optional[int] = None) -> int:
        processed = 0
        available = self.client.available()
        last_check = time.time()
        while not self._stop.is_set():
            # Re-check Ollama at most once every 30 seconds.
            if not available and time.time() - last_check > 30:
                available = self.client.available()
                last_check = time.time()
            if not available:
                self.queue.heartbeat(self.worker_id, "offline", {"url": self.client.base_url})
                time.sleep(self.poll_interval)
                continue
            if self.run_once():
                processed += 1
                if max_jobs and processed >= max_jobs:
                    break
            else:
                time.sleep(self.poll_interval)
        self.queue.heartbeat(self.worker_id, "stopped", {"processed": processed})
        return processed

    # ------------------------------------------------------------------ handlers
    def handle(self, job: Dict[str, Any]) -> Dict[str, Any]:
        kind = job.get("kind")
        payload = job.get("payload", {})
        if kind == "refine_summary":
            return self._refine_summary(payload)
        if kind == "macro_synthesize":
            return self._macro_synthesize(payload)
        if kind == "session_digest":
            return self._session_digest(payload)
        raise ValueError(f"Unknown job kind: {kind}")

    def _refine_summary(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        node_id = payload["node_id"]
        revision = int(payload.get("revision", 0))
        row = self.store.conn.execute("SELECT * FROM trajectories WHERE node_id=?", (node_id,)).fetchone()
        if row is None:
            return {"skipped": "missing trajectory"}
        if int(row["revision"]) != revision:
            return {"skipped": "stale revision", "current": int(row["revision"])}
        tpl = json.loads(row["tuple_json"])
        result = self.client.refine_summary(
            sub_goal=tpl.get("sub_goal", ""),
            tool_name=tpl.get("tool_name", ""),
            status=tpl.get("status", ""),
            raw_output=row["raw_excerpt"] or "",
            base_summary=tpl.get("output_summary", ""),
        )
        summary = str(result.get("summary", "")).strip()
        if not summary:
            raise ValueError("empty summary")
        self.store.update_neural_summary(node_id, revision, summary)
        return {"node_id": node_id, "summary": summary, "key_facts": result.get("key_facts", [])}

    def _macro_synthesize(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        skill_id = payload["skill_id"]
        row = self.store.conn.execute("SELECT * FROM skills WHERE skill_id=?", (skill_id,)).fetchone()
        if row is None:
            return {"skipped": "missing skill"}
        definition = json.loads(row["definition_json"])
        examples = [
            r["compact_text"]
            for r in self.store.conn.execute(
                """SELECT t.compact_text FROM skill_members m
                   JOIN trajectories t ON t.id=m.trajectory_id
                   WHERE m.skill_id=? ORDER BY t.id LIMIT 3""",
                (skill_id,),
            ).fetchall()
        ]
        result = self.client.synthesize_macro(definition, examples)
        name = str(result.get("name", "")).strip()[:80]
        description = str(result.get("description", "")).strip()
        if not name or not description:
            raise ValueError("incomplete macro synthesis")
        definition["name"] = name
        definition["description"] = description
        definition["neural"] = {
            "name": name,
            "description": description,
            "parameters": result.get("parameters", []),
            "preconditions": result.get("preconditions", []),
            "postconditions": result.get("postconditions", []),
            "model": self.client.model,
            "synthesized_at": time.time(),
        }
        # Merge parameter descriptions without overwriting deterministic slots.
        by_name = {p.get("name"): p for p in result.get("parameters", []) if isinstance(p, dict)}
        for param in definition.get("params", []):
            extra = by_name.get(param.get("name"))
            if extra and extra.get("description"):
                param["description"] = extra["description"]
        self.store.conn.execute(
            "UPDATE skills SET name=?, definition_json=? WHERE skill_id=?",
            (name, json.dumps(definition), skill_id),
        )
        return {"skill_id": skill_id, "name": name, "description": description}

    def _session_digest(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        session_id = payload.get("session_id")
        rows = self.store.conn.execute(
            "SELECT * FROM trajectories WHERE session_id=? ORDER BY step_id", (session_id,)
        ).fetchall()
        if not rows:
            return {"skipped": "empty session"}
        steps = []
        for row in rows:
            tpl = json.loads(row["tuple_json"])
            steps.append(
                {
                    "step_id": row["step_id"],
                    "status": row["status"],
                    "tool_name": row["tool_name"],
                    "summary": tpl.get("output_summary", ""),
                }
            )
        result = self.client.session_digest(steps)
        summary = str(result.get("summary", "")).strip()
        if not summary:
            raise ValueError("empty digest")
        self.store.conn.execute(
            "INSERT INTO sessions(session_id, scope_id, created_at, meta) VALUES(?,?,?,?) "
            "ON CONFLICT(session_id) DO UPDATE SET meta=excluded.meta",
            (session_id, "worker", time.time(), json.dumps({"digest": result}, default=str)),
        )
        return {"session_id": session_id, "summary": summary}


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="MATP neural background worker")
    parser.add_argument("--db", default=str(config.DB_PATH))
    parser.add_argument("--model", default=config.OLLAMA_MODEL)
    parser.add_argument("--url", default=config.OLLAMA_URL)
    parser.add_argument("--once", action="store_true", help="process one job and exit")
    parser.add_argument("--daemon", action="store_true", help="run forever (default)")
    parser.add_argument("--poll", type=float, default=config.WORKER_POLL_SECONDS)
    parser.add_argument("--max-jobs", type=int, default=None)
    args = parser.parse_args(argv)

    worker = Worker(
        db_path=args.db,
        client=OllamaClient(base_url=args.url, model=args.model),
        poll_interval=args.poll,
    )

    def _stop(signum, frame):
        worker.stop()

    try:
        signal.signal(signal.SIGINT, _stop)
        signal.signal(signal.SIGTERM, _stop)
    except (AttributeError, ValueError):  # pragma: no cover - non-main thread
        pass

    try:
        if args.once:
            processed = 1 if worker.run_once() else 0
        else:
            processed = worker.run_forever(max_jobs=args.max_jobs)
        print(json.dumps({"processed": processed, "worker": worker.worker_id}))
        return 0
    finally:
        worker.close()


if __name__ == "__main__":
    raise SystemExit(main())
