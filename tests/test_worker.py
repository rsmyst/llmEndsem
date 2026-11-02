from __future__ import annotations

from src.matp.ollama_client import OllamaUnavailable
from src.matp.worker import Worker


class FakeClient:
    model = "fake-qwen"
    base_url = "http://fake"

    def available(self):
        return True

    def refine_summary(self, sub_goal, tool_name, status, raw_output, base_summary):
        return {"summary": "neural: " + base_summary[:40], "key_facts": ["fact-1"], "confidence": 0.9}

    def synthesize_macro(self, definition, exemplars):
        return {
            "name": "neural_macro_name",
            "description": "A neural description of the mined routine.",
            "parameters": [{"name": p["name"], "description": "neural param description"} for p in definition.get("params", [])],
            "preconditions": ["resource exists"],
            "postconditions": ["macro completed"],
        }

    def session_digest(self, steps):
        return {"summary": "session digest", "open_issues": []}


class UnavailableClient(FakeClient):
    def refine_summary(self, *args, **kwargs):
        raise OllamaUnavailable("connection refused")


def index_one(store):
    return store.matp_index_trajectory(
        1, "read users", "sql_query", {"query": "SELECT * FROM users", "table": "users"}, "SUCCESS", "[]"
    )


def test_refine_summary_job(store):
    index_one(store)
    store.queue.enqueue("refine_summary", {"node_id": "s1:1", "revision": 1})
    worker = Worker(store=store, client=FakeClient())
    assert worker.run_once() is True
    trajectory = store.get_trajectory("s1:1")
    assert trajectory["neural_summary"].startswith("neural:")
    assert store.queue.stats()["done"] == 1


def test_stale_revision_is_skipped(store):
    index_one(store)
    store.queue.enqueue("refine_summary", {"node_id": "s1:1", "revision": 1})
    # Re-index the same node so the queued job refers to an obsolete revision.
    store.matp_index_trajectory(
        1, "read users again", "sql_query", {"query": "SELECT * FROM users", "table": "users"}, "SUCCESS", "[]"
    )
    worker = Worker(store=store, client=FakeClient())
    worker.run_once()
    assert store.get_trajectory("s1:1")["neural_summary"] is None


def test_macro_synthesis_job(store):
    for run in range(3):
        store.new_session(f"r{run}", "scope1")
        for step in range(1, 11):
            store.matp_index_trajectory(
                step,
                f"step {step}",
                "read_file",
                {"path": f"file_{step}.txt"},
                "SUCCESS",
                "ok",
            )
    skills = store.mine_macros(window=10, min_support=3)
    assert skills
    skill_id = skills[0]["skill_id"]
    store.queue.enqueue("macro_synthesize", {"skill_id": skill_id})
    worker = Worker(store=store, client=FakeClient())
    worker.run_once()
    updated = [s for s in store.list_macros() if s["skill_id"] == skill_id][0]
    assert updated["name"] == "neural_macro_name"
    assert updated["neural"]["description"].startswith("A neural description")


def test_unavailable_client_requeues(store):
    index_one(store)
    store.queue.enqueue("refine_summary", {"node_id": "s1:1", "revision": 1})
    worker = Worker(store=store, client=UnavailableClient())
    worker.run_once()
    job = store.queue.get(store.conn.execute("SELECT id FROM jobs").fetchone()["id"])
    assert job["state"] == "queued"
    assert job["attempts"] == 1
    assert "connection refused" in job["last_error"]


def test_index_is_non_blocking_and_enqueues(store):
    # Queue enabled: indexing only writes a job, it never calls the model.
    store.enable_queue = True
    index_one(store)
    assert store.queue.stats()["queued"] == 1
    assert store.get_trajectory("s1:1") is not None
