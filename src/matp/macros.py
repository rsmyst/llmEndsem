"""Hierarchical macro-action distillation.

The miner slices trajectories into 10-step windows, groups near-identical
signature sequences, parameterises the varying literals into slots, and stores
the resulting composite skill.  Skills are declarative plans: MATP never
executes them, the agent expands and runs them.
"""

from __future__ import annotations

import copy
import difflib
import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

from . import config
from .signatures import action_signature, canonical_tool, fingerprint

_PLACEHOLDER = re.compile(r"\{([A-Za-z_][\w]*)\}")
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in",
    "into", "is", "it", "of", "on", "or", "that", "the", "to", "with",
}


def _content_terms(text: str) -> set:
    terms = set()
    for token in re.findall(r"[a-z0-9]+", (text or "").lower()):
        if token in _STOPWORDS or len(token) <= 1:
            continue
        terms.add(token)
        if len(token) > 3 and token.endswith("s"):
            terms.add(token[:-1])
    return terms


@dataclass
class StepView:
    trajectory_id: int
    node_id: str
    session_id: str
    scope_id: str
    step_id: int
    tool_name: str
    status: str
    parameters: Dict[str, Any]
    sub_goal: str
    compact_text: str
    tuple_json: Dict[str, Any]
    validity: str

    @property
    def signature(self) -> str:
        return action_signature(self.tool_name, self.parameters, self.status)

    @property
    def ok(self) -> bool:
        return (self.status or "").upper() in {"SUCCESS", "OK", "PASSED"}


@dataclass
class Window:
    session_id: str
    scope_id: str
    start_step: int
    steps: List[StepView]

    @property
    def signatures(self) -> List[str]:
        return [s.signature for s in self.steps]

    @property
    def key(self) -> Tuple[str, ...]:
        return tuple(self.signatures)


class MacroMiner:
    def __init__(self, store, window: Optional[int] = None, stride: Optional[int] = None):
        self.store = store
        self.conn = store.conn
        self.window = config.MACRO_WINDOW if window is None else window
        self.stride = config.MACRO_STRIDE if stride is None else stride

    # ------------------------------------------------------------------ load
    def load_sessions(self) -> Dict[str, List[StepView]]:
        rows = self.conn.execute(
            "SELECT * FROM trajectories ORDER BY session_id, step_id"
        ).fetchall()
        sessions: Dict[str, List[StepView]] = {}
        for row in rows:
            try:
                tpl = json.loads(row["tuple_json"])
            except json.JSONDecodeError:
                tpl = {}
            view = StepView(
                trajectory_id=row["id"],
                node_id=row["node_id"],
                session_id=row["session_id"],
                scope_id=row["scope_id"],
                step_id=row["step_id"],
                tool_name=row["tool_name"],
                status=row["status"],
                parameters=tpl.get("parameters", {}),
                sub_goal=tpl.get("sub_goal", ""),
                compact_text=row["compact_text"],
                tuple_json=tpl,
                validity=row["validity_override"] or row["validity"],
            )
            sessions.setdefault(view.session_id, []).append(view)
        return sessions

    def windows(self, sessions: Optional[Dict[str, List[StepView]]] = None) -> List[Window]:
        sessions = sessions or self.load_sessions()
        out: List[Window] = []
        for session_id, steps in sessions.items():
            # Macro mining is about what the agent *did*, not whether the
            # resulting state is still current, so invalidated steps are kept.
            n = len(steps)
            if n < 3:
                continue
            if n <= self.window:
                out.append(Window(session_id, steps[0].scope_id, steps[0].step_id, steps))
                continue
            for start in range(0, n - self.window + 1, max(1, self.stride)):
                out.append(
                    Window(
                        session_id,
                        steps[start].scope_id,
                        steps[start].step_id,
                        steps[start : start + self.window],
                    )
                )
            # Always include the tail window.
            if (n - self.window) % max(1, self.stride) != 0:
                out.append(Window(session_id, steps[-self.window].scope_id, steps[-self.window].step_id, steps[-self.window :]))
        return out

    # ------------------------------------------------------------------ mining
    def mine(
        self,
        window: Optional[int] = None,
        stride: Optional[int] = None,
        min_support: Optional[int] = None,
        threshold: Optional[float] = None,
        min_success: Optional[float] = None,
        hierarchical: bool = False,
    ) -> List[Dict[str, Any]]:
        if window:
            self.window = window
        if stride:
            self.stride = stride
        min_support = config.MACRO_MIN_SUPPORT if min_support is None else min_support
        threshold = config.MACRO_SIMILARITY if threshold is None else threshold
        min_success = config.MACRO_MIN_SUCCESS if min_success is None else min_success

        windows = self.windows()
        clusters = self._cluster(windows, threshold, min_support)
        skills: List[Dict[str, Any]] = []
        for cluster in clusters:
            skill = self._build_skill(cluster, min_support, min_success)
            if skill:
                skills.append(self._upsert_skill(skill))
        if hierarchical:
            skills.extend(self.mine_hierarchical(skills, min_support))
        return skills

    def _cluster(self, windows: List[Window], threshold: float, min_support: int) -> List[List[Window]]:
        groups: Dict[Tuple[str, ...], List[Window]] = {}
        for w in windows:
            groups.setdefault(w.key, []).append(w)

        clusters: List[Dict[str, Any]] = []
        for key, members in groups.items():
            best = None
            best_sim = -1.0
            for cluster in clusters:
                sim = max(
                    difflib.SequenceMatcher(None, list(key), list(rep)).ratio()
                    for rep, _ in cluster["groups"]
                )
                if sim > best_sim:
                    best_sim = sim
                    best = cluster
            if best is not None and best_sim >= threshold:
                best["groups"].append((key, members))
            else:
                clusters.append({"groups": [(key, members)]})

        result: List[List[Window]] = []
        for cluster in clusters:
            members = [w for _, ws in cluster["groups"] for w in ws]
            if len(members) >= min_support:
                result.append(members)
        return result

    def _build_skill(
        self, members: List[Window], min_support: int, min_success: float
    ) -> Optional[Dict[str, Any]]:
        if not members:
            return None
        # Medoid = the most frequently observed exact sequence.
        counts: Dict[Tuple[str, ...], List[Window]] = {}
        for w in members:
            counts.setdefault(w.key, []).append(w)
        medoid_key, medoid_windows = max(counts.items(), key=lambda kv: len(kv[1]))
        medoid = medoid_windows[0]
        same_len = [w for w in members if len(w.steps) == len(medoid.steps)]

        steps: List[Dict[str, Any]] = []
        params: List[Dict[str, Any]] = []
        used_names = set()

        for index, step in enumerate(medoid.steps):
            parameters = copy.deepcopy(step.parameters)
            for key in list(parameters.keys()):
                values = [
                    w.steps[index].parameters.get(key)
                    for w in same_len
                    if index < len(w.steps)
                ]
                present = [v for v in values if v is not None]
                if not present:
                    continue
                if all(v == present[0] for v in present) and len(present) == len(values):
                    continue  # constant across every occurrence
                slot = self._slot_name(key, index, used_names)
                used_names.add(slot)
                parameters[key] = "{" + slot + "}"
                params.append(
                    {
                        "name": slot,
                        "key": key,
                        "step_index": index,
                        "type": self._json_type(present[0]),
                        "example": present[0],
                        "required": len(present) == len(values),
                    }
                )
            steps.append(
                {
                    "tool": step.tool_name,
                    "parameters": parameters,
                    "status": step.status,
                }
            )

        successes = sum(1 for w in members if w.steps and w.steps[-1].ok)
        success_rate = successes / max(1, len(members))
        tools = [canonical_tool(s.tool_name) for s in medoid.steps]
        compact_tools = list(dict.fromkeys(tools))
        name = "macro_" + "_".join(compact_tools)[:60] + f"_{len(steps)}s"
        skill_id = "sk_" + fingerprint([*medoid.signatures, json.dumps(params, sort_keys=True)])[:12]

        preconditions = self._preconditions(medoid)
        postconditions = self._postconditions(medoid)
        description = (
            f"Reusable {len(steps)}-step routine: "
            + " -> ".join(compact_tools)
            + f" (observed {len(members)} times, success {success_rate:.0%})."
        )
        definition = {
            "skill_id": skill_id,
            "name": name,
            "description": description,
            "level": 2,
            "signature": medoid.signatures,
            "steps": steps,
            "params": params,
            "preconditions": preconditions,
            "postconditions": postconditions,
            "support": len(members),
            "success_rate": round(success_rate, 4),
            "members": [w.steps[0].node_id for w in members],
            "window": len(steps),
            "created_at": time.time(),
            "neural": {},
        }
        definition["_member_ids"] = list(
            dict.fromkeys(step.trajectory_id for w in members for step in w.steps)
        )
        definition["_promote"] = len(members) >= min_support and success_rate >= min_success
        return definition

    def _preconditions(self, medoid: Window) -> List[str]:
        if not medoid.steps:
            return []
        first = medoid.steps[0]
        rows = self.conn.execute(
            "SELECT DISTINCT resource_id FROM accesses WHERE node_id=? AND mode='read'",
            (first.node_id,),
        ).fetchall()
        return [f"resource_exists:{r['resource_id']}" for r in rows]

    def _postconditions(self, medoid: Window) -> List[str]:
        writes = []
        for step in medoid.steps:
            rows = self.conn.execute(
                "SELECT DISTINCT resource_id FROM accesses WHERE node_id=? AND mode IN ('write','delete')",
                (step.node_id,),
            ).fetchall()
            writes.extend(r["resource_id"] for r in rows)
        out = [f"resource_written:{r}" for r in dict.fromkeys(writes)]
        if medoid.steps and medoid.steps[-1].ok:
            out.append("final_status:SUCCESS")
        return out

    @staticmethod
    def _slot_name(key: str, index: int, used: set) -> str:
        base = re.sub(r"\W+", "_", key).strip("_") or "param"
        candidate = f"{base}_{index}"
        n = 1
        while candidate in used:
            candidate = f"{base}_{index}_{n}"
            n += 1
        return candidate

    @staticmethod
    def _json_type(value: Any) -> str:
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int):
            return "integer"
        if isinstance(value, float):
            return "number"
        if isinstance(value, (list, tuple)):
            return "array"
        if isinstance(value, dict):
            return "object"
        return "string"

    # ------------------------------------------------------------------ storage
    def _upsert_skill(self, definition: Dict[str, Any]) -> Dict[str, Any]:
        skill_id = definition["skill_id"]
        member_ids = definition.pop("_member_ids", [])
        promote = definition.pop("_promote", False)
        existing = self.conn.execute("SELECT * FROM skills WHERE skill_id=?", (skill_id,)).fetchone()
        now = time.time()
        if existing:
            # Preserve the human-facing name once a worker has renamed it.
            old_def = json.loads(existing["definition_json"])
            if old_def.get("neural", {}).get("name"):
                definition["neural"] = old_def["neural"]
                definition["name"] = old_def["neural"]["name"]
            status = existing["status"]
            if status == "candidate" and promote:
                status = "promoted"
            self.conn.execute(
                """UPDATE skills SET name=?, definition_json=?, support=?, status=?
                   WHERE skill_id=?""",
                (definition["name"], json.dumps(definition), definition["support"], status, skill_id),
            )
        else:
            status = "promoted" if promote else "candidate"
            self.conn.execute(
                """INSERT INTO skills(skill_id, name, version, level, definition_json,
                                      support, status, created_at)
                   VALUES(?,?,?,?,?,?,?,?)""",
                (
                    skill_id,
                    definition["name"],
                    1,
                    definition.get("level", 2),
                    json.dumps(definition),
                    definition["support"],
                    status,
                    now,
                ),
            )
        self.conn.execute("DELETE FROM skill_members WHERE skill_id=?", (skill_id,))
        for trajectory_id in member_ids:
            self.conn.execute(
                "INSERT OR IGNORE INTO skill_members(skill_id, trajectory_id) VALUES(?,?)",
                (skill_id, trajectory_id),
            )
        row = self.conn.execute("SELECT * FROM skills WHERE skill_id=?", (skill_id,)).fetchone()
        return self._decode_skill(row)

    # ------------------------------------------------------------------ hierarchy
    def mine_hierarchical(self, level2: List[Dict[str, Any]], min_support: int = 2) -> List[Dict[str, Any]]:
        """Mine chains of promoted level-2 skills into level-3 composite skills."""
        promoted = [s for s in level2 if s.get("status") == "promoted"]
        if len(promoted) < 2:
            return []
        # Map each promoted skill to the session/step where it occurs.
        occurrences: Dict[str, List[Tuple[int, str]]] = {}
        for skill in promoted:
            rows = self.conn.execute(
                """SELECT t.session_id, MIN(t.step_id) AS step
                   FROM skill_members m JOIN trajectories t ON t.id=m.trajectory_id
                   WHERE m.skill_id=? GROUP BY t.session_id""",
                (skill["skill_id"],),
            ).fetchall()
            for row in rows:
                occurrences.setdefault(row["session_id"], []).append((int(row["step"]), skill["skill_id"]))
        if not occurrences:
            return []

        chains: Dict[Tuple[str, ...], List[str]] = {}
        for session_id, items in occurrences.items():
            items.sort()
            ids = [sid for _, sid in items]
            for size in (2, 3):
                for i in range(0, max(1, len(ids) - size + 1)):
                    chain = tuple(ids[i : i + size])
                    if len(chain) == size:
                        chains.setdefault(chain, []).append(session_id)
        out = []
        for chain, sessions in chains.items():
            if len(sessions) < min_support:
                continue
            skill_id = "sk_" + fingerprint(list(chain))[:12]
            definition = {
                "skill_id": skill_id,
                "name": "macro_chain_" + "_".join(s[-4:] for s in chain),
                "description": f"Composite skill chaining {len(chain)} learned macros.",
                "level": 3,
                "signature": list(chain),
                "steps": [
                    {"tool": "macro", "parameters": {"skill_id": sid}, "status": "SUCCESS"}
                    for sid in chain
                ],
                "params": [],
                "preconditions": [],
                "postconditions": [],
                "support": len(sessions),
                "success_rate": 1.0,
                "members": [],
                "window": len(chain),
                "created_at": time.time(),
                "neural": {},
                "children": list(chain),
            }
            if not self.conn.execute("SELECT 1 FROM skills WHERE skill_id=?", (skill_id,)).fetchone():
                self.conn.execute(
                    """INSERT INTO skills(skill_id, name, version, level, definition_json,
                                          support, status, created_at)
                       VALUES(?,?,?,?,?,?,?,?)""",
                    (skill_id, definition["name"], 1, 3, json.dumps(definition), len(sessions), "candidate", time.time()),
                )
            row = self.conn.execute("SELECT * FROM skills WHERE skill_id=?", (skill_id,)).fetchone()
            out.append(self._decode_skill(row))
        return out

    # ------------------------------------------------------------------ matching
    def list_macros(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        if status:
            rows = self.conn.execute("SELECT * FROM skills WHERE status=? ORDER BY level, name", (status,)).fetchall()
        else:
            rows = self.conn.execute("SELECT * FROM skills ORDER BY level, name").fetchall()
        return [self._decode_skill(r) for r in rows]

    def match_macro(self, goal: str, state: Optional[Dict[str, Any]] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        skills = self.list_macros(status="promoted")
        if not skills:
            skills = self.list_macros(status="candidate")
        goal_vec = self.store.embedder.embed(goal or "")
        state = state or {}
        scored = []
        for skill in skills:
            text_parts = [
                skill["name"],
                skill.get("description", ""),
                " ".join(skill.get("preconditions", [])),
                " ".join(skill.get("postconditions", [])),
            ]
            for step in skill.get("steps", []):
                text_parts.append(str(step.get("tool", "")))
                for value in step.get("parameters", {}).values():
                    text_parts.append(str(value))
            for param in skill.get("params", []):
                text_parts.append(str(param.get("name", "")))
                text_parts.append(str(param.get("example", "")))
            # Member summaries give the matcher the natural-language context
            # that short tool-only macro names lack.  Lexical matching uses only
            # the curated metadata to avoid structural labels ("Summary:") noise.
            metadata_text = " ".join(part for part in text_parts if part)
            members = self.conn.execute(
                """SELECT t.compact_text FROM skill_members m
                   JOIN trajectories t ON t.id=m.trajectory_id
                   WHERE m.skill_id=? ORDER BY t.id LIMIT 3""",
                (skill.get("skill_id"),),
            ).fetchall()
            member_text = " ".join(row["compact_text"][:500] for row in members)
            text = (metadata_text + " " + member_text).strip()
            vec = self.store.embedder.embed(text)
            cosine = self.store.embedder.cosine_similarity(goal_vec, vec)
            # A small lexical prior fixes hash-collision noise from the
            # dependency-free embedder and keeps matches explainable.
            query_terms = _content_terms(goal or "")
            skill_terms = _content_terms(metadata_text)
            lexical = len(query_terms & skill_terms) / max(1, len(query_terms))
            score = 0.7 * cosine + 0.3 * lexical
            satisfied = 0
            total = 0
            for pre in skill.get("preconditions", []):
                if pre.startswith("resource_exists:"):
                    total += 1
                    resource_id = pre.split(":", 1)[1]
                    row = self.conn.execute(
                        "SELECT current_version FROM resources WHERE resource_id=?", (resource_id,)
                    ).fetchone()
                    if row and int(row["current_version"]) > 0:
                        satisfied += 1
                    elif resource_id in str(state):
                        satisfied += 1
            if total:
                # Unverified preconditions are weak negative evidence, not a
                # disqualification: a macro may legitimately create the
                # resource it later reads.
                score *= 0.95 + 0.05 * (satisfied / total)
            score += 0.1 * float(skill.get("success_rate", 0.0))
            scored.append((score, skill))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for score, skill in scored[:top_k]:
            item = dict(skill)
            item["match_score"] = round(float(score), 4)
            out.append(item)
        return out

    def expand_macro(self, skill_id: str, bindings: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        row = self.conn.execute("SELECT * FROM skills WHERE skill_id=?", (skill_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown skill: {skill_id}")
        definition = json.loads(row["definition_json"])
        bindings = bindings or {}
        required = {p["name"] for p in definition.get("params", []) if p.get("required")}
        missing = required - set(bindings)
        if missing:
            raise ValueError(f"Missing macro bindings: {sorted(missing)}")

        def substitute(value: Any) -> Any:
            if isinstance(value, str):
                def repl(match):
                    name = match.group(1)
                    if name in bindings:
                        return str(bindings[name])
                    if name in required:
                        raise ValueError(f"Missing binding '{name}'")
                    return match.group(0)
                return _PLACEHOLDER.sub(repl, value)
            if isinstance(value, dict):
                return {k: substitute(v) for k, v in value.items()}
            if isinstance(value, list):
                return [substitute(v) for v in value]
            return value

        steps = []
        for step in definition.get("steps", []):
            steps.append(
                {
                    "tool": step.get("tool"),
                    "parameters": substitute(copy.deepcopy(step.get("parameters", {}))),
                    "expected_status": step.get("status"),
                }
            )
        return {
            "skill_id": skill_id,
            "name": definition.get("name"),
            "level": definition.get("level", 2),
            "steps": steps,
            "preconditions": definition.get("preconditions", []),
            "postconditions": definition.get("postconditions", []),
        }

    def record_use(self, skill_id: str, success: bool) -> Dict[str, Any]:
        row = self.conn.execute("SELECT * FROM skills WHERE skill_id=?", (skill_id,)).fetchone()
        if row is None:
            raise KeyError(f"Unknown skill: {skill_id}")
        successes = int(row["successes"]) + (1 if success else 0)
        failures = int(row["failures"]) + (0 if success else 1)
        consecutive = 0 if success else int(row["consecutive_failures"]) + 1
        status = row["status"]
        if consecutive >= 3 and status == "promoted":
            status = "deprecated"
        self.conn.execute(
            """UPDATE skills SET successes=?, failures=?, consecutive_failures=?, status=?, last_used=?
               WHERE skill_id=?""",
            (successes, failures, consecutive, status, time.time(), skill_id),
        )
        return self._decode_skill(self.conn.execute("SELECT * FROM skills WHERE skill_id=?", (skill_id,)).fetchone())

    def stats(self) -> Dict[str, Any]:
        rows = self.conn.execute("SELECT level, status, COUNT(*) AS n FROM skills GROUP BY level, status").fetchall()
        by_level: Dict[str, int] = {}
        by_status: Dict[str, int] = {}
        for row in rows:
            by_level[str(row["level"])] = by_level.get(str(row["level"]), 0) + int(row["n"])
            by_status[row["status"]] = by_status.get(row["status"], 0) + int(row["n"])
        total_steps = self.conn.execute("SELECT COUNT(*) AS n FROM trajectories").fetchone()["n"]
        covered = self.conn.execute("SELECT COUNT(DISTINCT trajectory_id) AS n FROM skill_members").fetchone()["n"]
        return {
            "skills": sum(by_level.values()),
            "by_level": by_level,
            "by_status": by_status,
            "covered_steps": int(covered),
            "total_steps": int(total_steps),
            "coverage_pct": round(100.0 * covered / max(1, total_steps), 2),
        }

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _decode_skill(row) -> Dict[str, Any]:
        definition = json.loads(row["definition_json"]) if row else {}
        definition.setdefault("skill_id", row["skill_id"] if row else None)
        definition["status"] = row["status"] if row else None
        definition["support"] = int(row["support"]) if row else 0
        definition["successes"] = int(row["successes"]) if row else 0
        definition["failures"] = int(row["failures"]) if row else 0
        definition["level"] = int(row["level"]) if row else 0
        definition["last_used"] = row["last_used"] if row else None
        return definition
