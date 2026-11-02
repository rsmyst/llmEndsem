from __future__ import annotations

import time

from src.matp.queue import JobQueue


def test_enqueue_claim_complete(store):
    q = store.queue
    job_id = q.enqueue("refine_summary", {"node_id": "s1:1"}, priority=3)
    job = q.claim(worker_id="w1")
    assert job is not None
    assert job["id"] == job_id
    assert job["state"] == "running"
    assert job["attempts"] == 1
    q.complete(job_id, {"ok": True})
    done = q.get(job_id)
    assert done["state"] == "done"
    assert done["result"] == {"ok": True}


def test_idempotency_key(store):
    q = store.queue
    first = q.enqueue("x", {"a": 1}, idempotency_key="same-key")
    second = q.enqueue("x", {"a": 2}, idempotency_key="same-key")
    assert first == second
    assert q.stats()["queued"] == 1


def test_retry_until_failed(store):
    q = store.queue
    job_id = q.enqueue("x", {}, max_attempts=2)
    job = q.claim()
    q.fail(job["id"], "boom", retry=True)
    assert q.get(job_id)["state"] == "queued"

    # Skip the exponential backoff window.
    q.conn.execute("UPDATE jobs SET available_at = 0 WHERE id = ?", (job_id,))
    job = q.claim()
    q.fail(job["id"], "boom again", retry=True)
    failed = q.get(job_id)
    assert failed["state"] == "failed"
    assert "boom again" in failed["last_error"]


def test_lease_recovery(store):
    q = store.queue
    job_id = q.enqueue("x", {})
    claimed = q.claim(lease_seconds=-1)
    assert claimed is not None
    assert q.get(job_id)["state"] == "running"
    recovered = q.recover_expired()
    assert recovered == 1
    assert q.get(job_id)["state"] == "queued"


def test_heartbeat_and_status(store):
    q = store.queue
    q.heartbeat("worker-1", "ok", {"processed": 3})
    status = q.worker_status()
    assert status["workers"][0]["worker_id"] == "worker-1"
    assert status["workers"][0]["meta"]["processed"] == 3


def test_purge(store):
    q = store.queue
    job_id = q.enqueue("x", {})
    job = q.claim()
    q.complete(job["id"], {})
    assert q.purge(["done"]) == 1
    assert q.get(job_id) is None
