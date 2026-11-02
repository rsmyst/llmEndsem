"""Memory-derived tool policy and repair-hint engine.

Unified-system port of the contribution-3 TypeScript engine (ashab).  Two
layers:

* **Tool Recommendation Matrix** — every indexed trajectory is a historical
  attempt at some sub-goal.  Recall the nearest attempts, group them per tool,
  Laplace-smooth the success rate, shrink it toward a neutral 0.5 prior with a
  Bayesian confidence term, and render prompt-injectable guidance.
* **Self-healing parameter corrections** — reusable ``errorPattern -> hint``
  rules learned from terminal/API errors, retrieved with a two-tier match
  (exact substring first, then embedding cosine) and injected when a similar
  error context reappears.
"""

from __future__ import annotations

import json
import re
import time
import uuid
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .embedder import LocalNGramEmbedder
from .storage import float_list_to_blob, blob_to_float_list, transaction

DEFAULT_TOP_K = 12
MIN_HINT_SIMILARITY = 0.40

_COLOR_GREEN = 0.75
_COLOR_AMBER = 0.45

_SUCCESS = {"success", "SUCCESS", "successes", "pass", "passed", "ok"}
_FAILURE = {"failed", "FAILURE", "error", "err", "fail", "failed_with"}


def _status_bucket(status: str) -> str:
    s = str(status or "").strip().lower()
    if s in {"success", "ok", "passed", "pass", "succeeded"}:
        return "success"
    if s in {"failed", "failure", "fail", "error", "err"}:
        return "failed"
    return "partial"


class ToolPolicyEngine:
    """Tool Recommendation Matrix + learned parameter-correction hints."""

    def __init__(self, store):
        self.store = store
        self.embedder: LocalNGramEmbedder = store.embedder

    # ------------------------------------------------------------- recall
    def recommend(
        self,
        sub_goal: str,
        candidate_tools: Optional[Iterable[str]],
        top_k: int = DEFAULT_TOP_K,
    ) -> Dict[str, Any]:
        candidates: List[str] = [str(c) for c in (candidate_tools or []) if str(c).strip()]
        hits = self.store.matp_query_trajectory(
            query=sub_goal,
            top_k=top_k,
            validity_weighted=False,
        )

        matrix: Dict[str, Dict[str, Any]] = {}
        order: List[str] = []

        def _bucket(tool: str) -> Dict[str, Any]:
            if tool not in matrix:
                matrix[tool] = {
                    "tool": tool,
                    "attempts": 0,
                    "success": 0,
                    "failed": 0,
                    "partial": 0,
                    "bestColor": "grey",
                    "exampleParameters": None,
                    "recentFailureReason": None,
                    "_lastStatus": None,
                }
                order.append(tool)
            return matrix[tool]

        for entry, _score in hits:
            tpl = entry["tuple"]
            tool = str(tpl.get("tool_name") or "")
            if not tool:
                continue
            if candidates and tool not in candidates:
                continue
            row = _bucket(tool)
            status = _status_bucket(tpl.get("status", ""))
            row["attempts"] += 1
            row[status] += 1
            row["_lastStatus"] = status
            if status == "success":
                step = int(tpl.get("step_id") or 999999)
                if row["exampleParameters"] is None or step < row.get("_example_step", 999999):
                    row["exampleParameters"] = tpl.get("parameters") or {}
                    row["_example_step"] = step
            if status == "failed" or status == "partial":
                reason = str(tpl.get("output_summary") or "")[:200] or "no failure detail recorded"
                if status == "failed" or row["recentFailureReason"] is None:
                    row["recentFailureReason"] = reason

        if not matrix:
            return {
                "subGoal": sub_goal,
                "matrix": [],
                "ranked": [],
                "autoCorrectionHints": [],
                "generatedAt": int(time.time() * 1000),
            }

        entries: List[Dict[str, Any]] = []
        for tool in order:
            m = matrix[tool]
            n = m["attempts"]
            # Laplace (add-one) smoothing: Beta(1,1) prior keeps 1/1 from
            # reading as a guaranteed 100% success rate.
            rate = (m["success"] + 1) / (n + 2)
            m["successRate"] = round(rate, 4)
            if rate >= _COLOR_GREEN:
                m["bestColor"] = "green"
            elif rate >= _COLOR_AMBER:
                m["bestColor"] = "amber"
            elif m["failed"] > 0:
                m["bestColor"] = "red"
            else:
                m["bestColor"] = "grey"
            m.pop("_lastStatus", None)
            m.pop("_example_step", None)
            entries.append(m)

        ranked = sorted(
            (
                {
                    "tool": e["tool"],
                    "successRate": e["successRate"],
                    "attempts": e["attempts"],
                    "confidence": round(self._confidence(e["successRate"], e["attempts"]), 4),
                }
                for e in entries
            ),
            key=lambda r: (-r["confidence"], -r["attempts"]),
        )

        hints = self.auto_correct(sub_goal, [r["tool"] for r in ranked])
        return {
            "subGoal": sub_goal,
            "matrix": entries,
            "ranked": ranked,
            "autoCorrectionHints": hints,
            "generatedAt": int(time.time() * 1000),
        }

    @staticmethod
    def _confidence(rate: float, attempts: int) -> float:
        # Bayesian shrinkage with prior mean 0.5 and prior strength 5.
        n = float(attempts)
        return rate * (n / (n + 5.0)) + 0.5 * (5.0 / (n + 5.0))

    # ------------------------------------------------------------- corrections
    def add_correction(
        self, tool: str, error_pattern: str, hint: str
    ) -> Dict[str, Any]:
        tool = (tool or "*").strip() or "*"
        pattern = (error_pattern or "").strip()
        hint_text = (hint or "").strip()
        if not pattern or not hint_text:
            raise ValueError("error_pattern and hint are required")
        searchable = f"{tool} | {pattern} | {hint_text}"
        vector = self.embedder.embed(searchable)
        rule_id = str(uuid.uuid4())
        with transaction(self.store.conn):
            self.store.conn.execute(
                """INSERT INTO corrections
                   (id, tool, error_pattern, hint, times_applied, enabled, searchable,
                    dim, vector, created_at)
                   VALUES(?,?,?,?,?,?,?,?,?,?)""",
                (
                    rule_id,
                    tool,
                    pattern,
                    hint_text,
                    0,
                    1,
                    searchable,
                    len(vector),
                    float_list_to_blob(vector),
                    time.time(),
                ),
            )
        return {
            "id": rule_id,
            "tool": tool,
            "errorPattern": pattern,
            "hint": hint_text,
            "timesApplied": 0,
        }

    def nearest_corrections(
        self, tool: str, error_output: str, k: int = 3
    ) -> List[Dict[str, Any]]:
        rows = self.store.conn.execute(
            "SELECT * FROM corrections WHERE enabled=1 AND (?='*' OR tool=? OR tool='*')",
            (tool or "*", tool or "*"),
        ).fetchall()
        if not rows:
            return []

        needle = (error_output or "").lower()
        results: Dict[str, Tuple[Any, float]] = {}

        # Tier 1: exact substring match pins similarity to 1.0 — a precise
        # signal beats a fuzzy one.
        for row in rows:
            pattern = str(row["error_pattern"] or "")
            if pattern.lower() and pattern.lower() in needle:
                results[str(row["id"])] = (row, 1.0)

        # Tier 2: fuzzy cosine ranking for the remaining rules.
        query_vec = self.embedder.embed(f"{tool or ''} | {needle}".strip())
        for row in rows:
            if str(row["id"]) in results:
                continue
            vec = blob_to_float_list(row["vector"]) if row["vector"] else None
            if not vec:
                continue
            sim = self.embedder.cosine_similarity(query_vec, vec)
            if sim >= MIN_HINT_SIMILARITY:
                results[str(row["id"])] = (row, round(sim, 3))

        merged = sorted(results.items(), key=lambda kv: float(kv[1][1]), reverse=True)[:k]
        out: List[Dict[str, Any]] = []
        for _rule_id, (row, sim) in merged:
            if sim < MIN_HINT_SIMILARITY and sim < 1.0:
                continue
            self.mark_fired(row["id"])
            out.append(
                {
                    "ruleId": row["id"],
                    "tool": row["tool"],
                    "errorPattern": row["error_pattern"],
                    "hint": row["hint"],
                    "similarity": round(float(sim), 3),
                }
            )
        return out

    def mark_fired(self, rule_id: str) -> None:
        try:
            with transaction(self.store.conn):
                self.store.conn.execute(
                    "UPDATE corrections SET times_applied = times_applied + 1 WHERE id=?",
                    (rule_id,),
                )
        except Exception:
            pass

    def auto_correct(
        self, sub_goal: str, ranked_tools: List[str], per_tool_k: int = 3
    ) -> List[Dict[str, Any]]:
        hints: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for tool in ranked_tools:
            for cue in self.nearest_corrections(tool, sub_goal, per_tool_k):
                key = cue["ruleId"]
                if key in seen:
                    continue
                seen.add(key)
                hints.append(cue)
        return hints

    def list_corrections(
        self, tool: Optional[str] = None, enabled_only: bool = True
    ) -> List[Dict[str, Any]]:
        sql = ["SELECT * FROM corrections WHERE 1=1"]
        args: List[Any] = []
        if enabled_only:
            sql.append("AND enabled=1")
        if tool:
            sql.append("AND (tool=? OR tool='*')")
            args.append(tool)
        rows = self.store.conn.execute(" ".join(sql), tuple(args)).fetchall()
        return [
            {
                "id": r["id"],
                "tool": r["tool"],
                "errorPattern": r["error_pattern"],
                "hint": r["hint"],
                "timesApplied": int(r["times_applied"]),
            }
            for r in rows
        ]

    def disable_correction(self, rule_id: str) -> bool:
        with transaction(self.store.conn):
            cur = self.store.conn.execute(
                "UPDATE corrections SET enabled=0 WHERE id=?", (rule_id,)
            )
        return cur.rowcount == 1

    # ------------------------------------------------------------- rendering
    def render_prompt(self, state: Dict[str, Any]) -> str:
        ranked = state.get("ranked") or []
        hints = state.get("autoCorrectionHints") or []
        if not ranked:
            return (
                "No historical tool policy available for this sub-goal yet. "
                "Proceed heuristically and report back so it can be learned."
            )
        sub_goal = state.get("subGoal", "")
        lines = [f'Memory-derived tool policy for sub-goal: "{sub_goal}"']
        for r in ranked:
            pct = round(float(r["successRate"]) * 100)
            conf = round(float(r["confidence"]) * 100)
            lines.append(
                f"- {r['tool']}: success ~{pct}% ({r['attempts']} samples, confidence {conf}%)"
            )
        if hints:
            lines.append("Parameter auto-correction hints:")
            for h in hints:
                lines.append(f'- on "{h["errorPattern"]}" -> {h["hint"]}')
        lines.append(
            "Decision rule: prefer the tool carrying the highest confidence-weighted "
            "success rate; if it previously failed with a known parameter error, apply "
            "the correction hint before calling."
        )
        return "\n".join(lines)
