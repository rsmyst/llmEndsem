"""Persistent trajectory memory for MATP (unified monorepo stack).

Merges the three original contributions into one store:

* contribution 1 — version-aware retrieval with supersession and temporal
  decay, including per-task adaptively learned decay rates from feedback.
* contribution 2 — output distillation into trajectory tuples, a causal DAG
  for cross-session invalidation, macro skills and the async neural worker.
* contribution 3 — memory-derived tool policy ranking plus self-healing
  parameter-correction hints.

Public API is backwards compatible with each original implementation, storage
is SQLite, and every trajectory participates in the causal DAG.  Retrieval
multiplies the dense+sparse hybrid score by a validity weight so superseded
and stale state cannot outrank current state.
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from . import config
from .abstraction import BackgroundAbstractionEngine, TrajectoryTuple
from .dag import CausalDAG
from .decay import DecayTuner
from .embedder import LocalNGramEmbedder
from .macros import MacroMiner
from .policy import DEFAULT_TOP_K, MIN_HINT_SIMILARITY, ToolPolicyEngine
from .queue import JobQueue
from .resources import ResourceResolver
from .signatures import action_signature
from .storage import (
    blob_to_float_list,
    connect,
    float_list_to_blob,
    get_meta,
    init_db,
    migrate_legacy_json,
    set_meta,
    transaction,
)


class MATPMemoryStore:
    def __init__(
        self,
        persistence_file: Optional[str] = "data/matp.memory.json",
        embedder_dim: Optional[int] = None,
        db_path: Optional[str] = None,
        session_id: str = "default",
        scope_id: str = "default",
        enable_queue: bool = True,
        auto_migrate: bool = True,
        workspace: Optional[str] = None,
    ):
        if db_path is None:
            if persistence_file and str(persistence_file).lower().endswith(".json"):
                db_path = str(Path(persistence_file).with_suffix(".db"))
            else:
                db_path = str(config.DB_PATH)
        self.db_path = Path(db_path)
        self.persistence_file = persistence_file
        self.conn = connect(self.db_path)
        init_db(self.conn)
        self.embedder = LocalNGramEmbedder(dim=embedder_dim or config.EMBED_DIM)
        self.abstraction_engine = BackgroundAbstractionEngine()
        self.dag = CausalDAG(self.conn)
        self.decay = DecayTuner(
            self.conn,
            default_lambda=config.DECAY_DEFAULT_LAMBDA,
            alpha=config.DECAY_ALPHA,
        )
        self.queue = JobQueue(self.conn)
        self.session_id = session_id
        self.scope_id = scope_id
        self.workspace = workspace
        self.enable_queue = enable_queue
        self._lock = threading.RLock()

        if auto_migrate:
            default_legacy = "data/matp.memory.json"
            if persistence_file and str(persistence_file).lower().endswith(".json") and str(persistence_file) != default_legacy:
                legacy = str(persistence_file)
            else:
                legacy = str(config.LEGACY_JSON)
            migrate_legacy_json(self.conn, legacy)

        self.trajectories: List[Dict[str, Any]] = []
        self.embeddings: List[List[float]] = []
        self.policy = ToolPolicyEngine(self)
        self._ensure_session(self.session_id, self.scope_id)
        self.refresh_cache()

    # ------------------------------------------------------------------ sessions
    def _ensure_session(self, session_id: str, scope_id: str) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO sessions(session_id, scope_id, created_at) VALUES(?,?,?)",
            (session_id, scope_id, time.time()),
        )

    def new_session(self, session_id: str, scope_id: Optional[str] = None) -> None:
        self.session_id = session_id
        self.scope_id = scope_id or self.scope_id
        self._ensure_session(self.session_id, self.scope_id)

    # ------------------------------------------------------------------ clock
    def current_step(self) -> int:
        """Global logical clock: counts causal steps, not wall time."""
        clock = int(float(get_meta(self.conn, "logical_clock", "0") or 0))
        row = self.conn.execute("SELECT MAX(step_id) AS m FROM trajectories").fetchone()
        max_step = int(row["m"] or 0)
        return max(clock, max_step)

    def advance_clock(self, steps: int = 1) -> int:
        value = self.current_step() + max(1, steps)
        with transaction(self.conn):
            set_meta(self.conn, "logical_clock", value)
        return value

    # ------------------------------------------------------------------ indexing
    def matp_index_trajectory(
        self,
        step_id: int,
        sub_goal: str,
        tool_name: str,
        parameters: Dict[str, Any],
        status: str,
        raw_output: str,
        parent_step_id: Optional[int] = None,
        session_id: Optional[str] = None,
        scope_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> TrajectoryTuple:
        session_id = session_id or self.session_id
        scope_id = scope_id or self.scope_id
        self._ensure_session(session_id, scope_id)
        node_id = f"{session_id}:{step_id}"
        parent_node_id = f"{session_id}:{parent_step_id}" if parent_step_id is not None else None

        distilled = self.abstraction_engine.abstract_tool_output(
            step_id=step_id,
            sub_goal=sub_goal,
            tool_name=tool_name,
            parameters=parameters,
            status=status,
            raw_output=raw_output,
            parent_step_id=parent_step_id,
        )
        compact_text = distilled.to_compact_text()
        vector = self.embedder.embed(compact_text)
        signature = action_signature(tool_name, parameters, status)

        with self._lock, transaction(self.conn):
            existing = self.conn.execute(
                "SELECT * FROM trajectories WHERE node_id=?", (node_id,)
            ).fetchone()
            revision = int(existing["revision"]) + 1 if existing else 1
            tuple_json = json.dumps(distilled.to_dict())
            if existing:
                self.conn.execute(
                    """UPDATE trajectories SET
                         session_id=?, scope_id=?, step_id=?, tool_name=?, status=?, sub_goal=?,
                         tuple_json=?, compact_text=?, raw_excerpt=?, raw_tokens=?, distilled_tokens=?,
                         signature=?, revision=?, neural_summary=NULL, neural_revision=NULL,
                         validity='fresh', taint=1.0, invalidated_by=NULL, reason=NULL,
                         validity_override=NULL, created_at=?
                       WHERE node_id=?""",
                    (
                        session_id,
                        scope_id,
                        step_id,
                        tool_name,
                        status,
                        sub_goal,
                        tuple_json,
                        compact_text,
                        (raw_output or "")[:8000],
                        distilled.raw_token_count_est,
                        distilled.distilled_token_count_est,
                        signature,
                        revision,
                        time.time(),
                        node_id,
                    ),
                )
                trajectory_id = int(existing["id"])
                self.conn.execute("DELETE FROM embeddings WHERE trajectory_id=?", (trajectory_id,))
            else:
                cur = self.conn.execute(
                    """INSERT INTO trajectories(
                         node_id, session_id, scope_id, step_id, tool_name, status, sub_goal,
                         tuple_json, compact_text, raw_excerpt, raw_tokens, distilled_tokens,
                         signature, revision, validity, taint, created_at
                       ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                    (
                        node_id,
                        session_id,
                        scope_id,
                        step_id,
                        tool_name,
                        status,
                        sub_goal,
                        tuple_json,
                        compact_text,
                        (raw_output or "")[:8000],
                        distilled.raw_token_count_est,
                        distilled.distilled_token_count_est,
                        signature,
                        revision,
                        "fresh",
                        1.0,
                        time.time(),
                    ),
                )
                trajectory_id = int(cur.lastrowid)
            self.conn.execute(
                "INSERT OR REPLACE INTO embeddings(trajectory_id, dim, vector) VALUES(?,?,?)",
                (trajectory_id, len(vector), float_list_to_blob(vector)),
            )

        resolver = ResourceResolver(workspace=self.workspace, scope_id=scope_id)
        accesses = resolver.extract(
            tool_name,
            parameters,
            status,
            raw_output,
            entities=distilled.entities,
            triples=distilled.triples,
        )
        self.dag.register_node(
            node_id=node_id,
            trajectory_id=trajectory_id,
            session_id=session_id,
            scope_id=scope_id,
            accesses=accesses,
            parent_node_id=parent_node_id,
        )

        if self.enable_queue and config.NEURAL_MODE != "off":
            self.queue.enqueue(
                "refine_summary",
                {"node_id": node_id, "revision": revision},
                priority=7,
                idempotency_key=f"refine:{node_id}:{revision}",
            )
        self.refresh_cache()
        return distilled

    # ------------------------------------------------------------------ retrieval
    def matp_query_trajectory(
        self,
        query: str,
        top_k: int = 3,
        filter_status: Optional[str] = None,
        filter_tool: Optional[str] = None,
        filter_entity: Optional[str] = None,
        hybrid_alpha: float = 0.75,
        include_stale: bool = False,
        explain: bool = False,
        session_id: Optional[str] = None,
        scope_id: Optional[str] = None,
        validity_weighted: bool = True,
        time_decay: Optional[Any] = None,
        task: Optional[str] = None,
    ) -> List[Tuple[Dict[str, Any], float]]:
        now = self.current_step()

        # time_decay resolves to a steps-based half-life rate.  A plain float
        # pins the rate; "auto" / "adaptive" pulls the per-task learned value.
        lambda_steps: Optional[float] = None
        if time_decay is not None:
            if isinstance(time_decay, str):
                mode = time_decay.strip().lower()
                if mode in {"auto", "adaptive"}:
                    lambda_steps = self.decay.lambda_for(task)
                else:
                    raise ValueError(
                        f"time_decay must be a number, \"auto\" or \"adaptive\", got {time_decay!r}"
                    )
            else:
                lambda_steps = float(time_decay)

        sql = ["SELECT t.*, e.vector AS vector FROM trajectories t LEFT JOIN embeddings e ON e.trajectory_id=t.id WHERE 1=1"]
        args: List[Any] = []
        if session_id:
            sql.append("AND t.session_id=?")
            args.append(session_id)
        if scope_id:
            sql.append("AND t.scope_id=?")
            args.append(scope_id)
        if filter_status:
            sql.append("AND t.status=?")
            args.append(filter_status)
        if filter_tool:
            sql.append("AND t.tool_name=?")
            args.append(filter_tool)
        rows = self.conn.execute(" ".join(sql), tuple(args)).fetchall()
        if not rows:
            return []

        query_vec = self.embedder.embed(query or "")
        q_tokens = set(re.findall(r"[a-zA-Z0-9_\-\.]+", (query or "").lower()))
        scored: List[Tuple[Dict[str, Any], float]] = []

        for row in rows:
            tpl = json.loads(row["tuple_json"])
            if filter_entity:
                fe = filter_entity.lower()
                ent_lower = [str(e).lower() for e in tpl.get("entities", [])]
                if not any(fe in e for e in ent_lower) and fe not in row["compact_text"].lower():
                    continue

            if row["vector"]:
                vec = blob_to_float_list(row["vector"])
            else:
                vec = self.embedder.embed(row["compact_text"])
            cos_sim = self.embedder.cosine_similarity(query_vec, vec)

            target_tokens = set(re.findall(r"[a-zA-Z0-9_\-\.]+", row["compact_text"].lower()))
            for entity in tpl.get("entities", []):
                target_tokens.update(re.findall(r"[a-zA-Z0-9_\-\.]+", str(entity).lower()))
            overlap = len(q_tokens.intersection(target_tokens))
            sparse_score = (overlap / max(1, len(q_tokens))) if q_tokens else 0.0

            hybrid = (hybrid_alpha * max(0.0, cos_sim)) + ((1.0 - hybrid_alpha) * sparse_score)
            weight = self.dag.retrieval_weight(row) if validity_weighted else 1.0
            age_steps = max(0, now - int(row["step_id"] or 0))
            if lambda_steps is not None and lambda_steps > 0:
                weight *= math.exp(-lambda_steps * age_steps)
            if weight <= 0.0 and not include_stale:
                continue
            score = hybrid * weight

            entry = self._row_to_entry(row)
            if explain or lambda_steps is not None:
                entry["explain"] = {
                    "validity": row["validity_override"] or row["validity"],
                    "taint": round(float(row["taint"]), 4),
                    "weight": round(weight, 4),
                    "invalidated_by": row["invalidated_by"],
                    "reason": row["reason"],
                    "age_steps": age_steps,
                    "decay_lambda": lambda_steps,
                }
            scored.append((entry, round(score, 4)))

        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------- wisdom of history
    def record_feedback(self, task: str, useful_node_ids: List[str]) -> Dict[str, Any]:
        """Report which nodes were actually useful so the decay rate adapts.

        Ages are (current logical step - step of the node) and feed the
        half-life calibration of the per-task lambda estimate.
        """
        now = self.current_step()
        ages: List[int] = []
        unknown: List[str] = []
        for node_id in useful_node_ids or []:
            row = self.conn.execute(
                "SELECT step_id FROM trajectories WHERE node_id=?", (node_id,)
            ).fetchone()
            if row is None:
                unknown.append(node_id)
            else:
                ages.append(max(0, now - int(row["step_id"] or 0)))
        if ages:
            self.decay.observe(task, ages)
        return {
            "task": task,
            "feedback_items": len(ages),
            "unknown_item_ids": unknown,
            "lambda_value": self.decay.lambda_for(task),
        }

    def _row_to_entry(self, row) -> Dict[str, Any]:
        tpl = json.loads(row["tuple_json"])
        neural = row["neural_summary"] if row["neural_revision"] == row["revision"] else None
        return {
            "step_id": tpl.get("step_id"),
            "sub_goal": tpl.get("sub_goal"),
            "tool_name": tpl.get("tool_name"),
            "status": tpl.get("status"),
            "tuple": tpl,
            "compact_text": row["compact_text"],
            "raw_output_tokens": int(row["raw_tokens"]),
            "distilled_tokens": int(row["distilled_tokens"]),
            "indexed_at": row["created_at"],
            "node_id": row["node_id"],
            "session_id": row["session_id"],
            "scope_id": row["scope_id"],
            "revision": int(row["revision"]),
            "neural_summary": neural,
            "validity": row["validity_override"] or row["validity"],
            "taint": float(row["taint"]),
            "invalidated_by": row["invalidated_by"],
            "reason": row["reason"],
        }

    # ------------------------------------------------------------------ cache
    def refresh_cache(self) -> None:
        rows = self.conn.execute("SELECT * FROM trajectories ORDER BY id").fetchall()
        vectors = self.conn.execute("SELECT trajectory_id, vector FROM embeddings").fetchall()
        by_id = {r["trajectory_id"]: blob_to_float_list(r["vector"]) for r in vectors if r["vector"]}
        self.trajectories = [self._row_to_entry(r) for r in rows]
        self.embeddings = [by_id.get(r["id"], []) for r in rows]

    def clear(self, include_skills: bool = True, include_jobs: bool = False) -> None:
        with self._lock, transaction(self.conn):
            self.conn.execute("DELETE FROM embeddings")
            self.conn.execute("DELETE FROM accesses")
            self.conn.execute("DELETE FROM dag_edges")
            self.conn.execute("DELETE FROM resource_versions")
            self.conn.execute("DELETE FROM resources")
            self.conn.execute("DELETE FROM trajectories")
            self.conn.execute("DELETE FROM sessions")
            if include_skills:
                self.conn.execute("DELETE FROM skill_members")
                self.conn.execute("DELETE FROM skills")
            if include_jobs:
                self.conn.execute("DELETE FROM jobs")
        self.refresh_cache()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass

    # ------------------------------------------------------------------ stats
    def get_stats(self) -> Dict[str, Any]:
        row = self.conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(raw_tokens),0) AS raw, COALESCE(SUM(distilled_tokens),0) AS dist FROM trajectories"
        ).fetchone()
        raw_total = int(row["raw"])
        dist_total = int(row["dist"])
        validity_rows = self.conn.execute(
            "SELECT validity, COUNT(*) AS n FROM trajectories GROUP BY validity"
        ).fetchall()
        validity = {r["validity"]: int(r["n"]) for r in validity_rows}
        corrections = self.conn.execute(
            "SELECT COUNT(*) AS n FROM corrections WHERE enabled=1"
        ).fetchone()
        return {
            "total_entries": int(row["n"]),
            "total_raw_tokens": raw_total,
            "total_distilled_tokens": dist_total,
            "compression_ratio_pct": round((1.0 - dist_total / max(1, raw_total)) * 100.0, 2),
            "token_reduction_factor": round(raw_total / max(1, dist_total), 2),
            "validity": validity,
            "corrections": int(corrections["n"]),
            "logical_clock_step": self.current_step(),
            "queue": self.queue.stats(),
        }

    # ------------------------------------------------------------------ graph
    def get_knowledge_graph(self) -> Dict[str, Any]:
        from collections import Counter

        nodes = set()
        edges = []
        degrees = Counter()
        for row in self.conn.execute("SELECT tuple_json, validity FROM trajectories").fetchall():
            tpl = json.loads(row["tuple_json"])
            for triple in tpl.get("triples", []):
                if not isinstance(triple, (list, tuple)) or len(triple) != 3:
                    continue
                s, r, o = (str(x) for x in triple)
                nodes.add(s)
                nodes.add(o)
                degrees[s] += 1
                degrees[o] += 1
                edges.append({"subject": s, "relation": r, "object": o, "validity": row["validity"]})
        central = [entity for entity, _ in degrees.most_common(8)]
        return {
            "total_nodes": len(nodes),
            "total_edges": len(edges),
            "nodes": sorted(nodes),
            "edges": edges,
            "central_entities": central,
            "degrees": dict(degrees.most_common(20)),
        }

    def query_knowledge_graph(self, entity: str) -> List[Dict[str, Any]]:
        ent = (entity or "").lower()
        return [
            edge
            for edge in self.get_knowledge_graph()["edges"]
            if ent in edge["subject"].lower() or ent in edge["object"].lower()
        ]

    def get_session_summary(self, session_id: Optional[str] = None, use_neural: bool = True) -> str:
        if session_id:
            rows = self.conn.execute(
                "SELECT * FROM trajectories WHERE session_id=? ORDER BY step_id", (session_id,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM trajectories ORDER BY session_id, step_id"
            ).fetchall()
        if not rows:
            return "No trajectories recorded."

        lines = [f"=== MATP Session Trajectory Log ({len(rows)} steps) ==="]
        for row in rows:
            tpl = json.loads(row["tuple_json"])
            summary = tpl.get("output_summary", "")
            if use_neural and row["neural_summary"] and row["neural_revision"] == row["revision"]:
                summary = row["neural_summary"]
            flag = "" if (row["validity_override"] or row["validity"]) == "fresh" else f" !{(row['validity_override'] or row['validity']).upper()}"
            lines.append(f"Step {row['step_id']} [{row['status']}]{flag} {row['tool_name']}: {summary}")
        stats = self.get_stats()
        lines.append(
            f"Metrics: {stats['total_raw_tokens']} raw tokens distilled to "
            f"{stats['total_distilled_tokens']} tokens ({stats['compression_ratio_pct']}% saved)."
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------ DAG API
    def get_lineage(self, node_id: str) -> Dict[str, Any]:
        return self.dag.get_lineage(node_id)

    def query_dag(self, resource_id: Optional[str] = None, scope_id: Optional[str] = None) -> Dict[str, Any]:
        if resource_id:
            return self.dag.query_resource(resource_id, scope_id)
        rows = self.conn.execute(
            "SELECT * FROM resources" + (" WHERE scope_id=?" if scope_id else "") + " ORDER BY resource_id",
            ((scope_id,) if scope_id else ()),
        ).fetchall()
        return {"resources": [dict(r) for r in rows]}

    def confirm_state(self, node_id: str, note: str = "manual override") -> bool:
        result = self.dag.confirm_state(node_id, note)
        self.refresh_cache()
        return result

    def clear_override(self, node_id: str) -> bool:
        result = self.dag.clear_override(node_id)
        self.refresh_cache()
        return result

    def recompute_validity(self, scope_id: Optional[str] = None) -> Dict[str, int]:
        result = self.dag.recompute_validity(scope_id)
        self.refresh_cache()
        return result

    # ------------------------------------------------------------------ macros
    def mine_macros(self, **kwargs) -> List[Dict[str, Any]]:
        miner = MacroMiner(self)
        skills = miner.mine(**kwargs)
        if self.enable_queue and config.NEURAL_MODE != "off":
            for skill in skills:
                if skill.get("status") in {"promoted", "candidate"}:
                    self.queue.enqueue(
                        "macro_synthesize",
                        {"skill_id": skill["skill_id"]},
                        priority=6,
                        idempotency_key=f"macro:{skill['skill_id']}:v1",
                    )
        return skills

    def list_macros(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        return MacroMiner(self).list_macros(status)

    def match_macro(self, goal: str, state: Optional[Dict[str, Any]] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        return MacroMiner(self).match_macro(goal, state=state, top_k=top_k)

    def expand_macro(self, skill_id: str, bindings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return MacroMiner(self).expand_macro(skill_id, bindings=bindings)

    def record_macro_use(self, skill_id: str, success: bool) -> Dict[str, Any]:
        return MacroMiner(self).record_use(skill_id, success)

    def macro_stats(self) -> Dict[str, Any]:
        return MacroMiner(self).stats()

    # ------------------------------------------------------------------ policy
    def recommend_tools(
        self,
        sub_goal: str,
        candidate_tools: Optional[List[str]] = None,
        top_k: Optional[int] = None,
        min_samples: int = 0,
    ) -> Dict[str, Any]:
        state = self.policy.recommend(
            sub_goal, candidate_tools, top_k=top_k or config.POLICY_TOP_K
        )
        if min_samples > 0:
            state["ranked"] = [r for r in state["ranked"] if r["attempts"] >= min_samples]
            state["matrix"] = [m for m in state["matrix"] if m["attempts"] >= min_samples]
        return state

    def render_policy_prompt(self, sub_goal: str, candidate_tools: Optional[List[str]] = None) -> str:
        state = self.policy.recommend(sub_goal, candidate_tools, top_k=config.POLICY_TOP_K)
        return self.policy.render_prompt(state)

    def parameter_hints(self, tool: str, context: Optional[str] = None, top_k: int = 3) -> List[Dict[str, Any]]:
        ctx = context or f"about to call {tool}"
        return self.policy.nearest_corrections(tool, ctx, k=top_k)

    def learn_correction(self, tool: str, error_pattern: str, hint: str) -> Dict[str, Any]:
        return self.policy.add_correction(tool, error_pattern, hint)

    def list_corrections(
        self, tool: Optional[str] = None, enabled_only: bool = True
    ) -> List[Dict[str, Any]]:
        return self.policy.list_corrections(tool=tool, enabled_only=enabled_only)

    def disable_correction(self, rule_id: str) -> Dict[str, Any]:
        return {"ok": self.policy.disable_correction(rule_id)}

    def decay_stats(self) -> Dict[str, Any]:
        return self.decay.stats()

    # ------------------------------------------------------------------ worker
    def enqueue_digest(self, session_id: Optional[str] = None) -> str:
        return self.queue.enqueue(
            "session_digest",
            {"session_id": session_id or self.session_id},
            priority=8,
            idempotency_key=f"digest:{session_id or self.session_id}:{self.get_stats()['total_entries']}",
        )

    def worker_status(self) -> Dict[str, Any]:
        status = self.queue.worker_status()
        status["neural_mode"] = config.NEURAL_MODE
        status["model"] = config.OLLAMA_MODEL
        return status

    def get_trajectory(self, node_id: str) -> Optional[Dict[str, Any]]:
        row = self.conn.execute("SELECT * FROM trajectories WHERE node_id=?", (node_id,)).fetchone()
        return self._row_to_entry(row) if row else None

    def update_neural_summary(self, node_id: str, revision: int, summary: str) -> bool:
        cur = self.conn.execute(
            "UPDATE trajectories SET neural_summary=?, neural_revision=? WHERE node_id=? AND revision=?",
            (summary, revision, node_id, revision),
        )
        return cur.rowcount == 1
