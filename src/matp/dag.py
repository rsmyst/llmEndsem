"""Cross-session causal DAG for MATP Contribution 2.

Every indexed trajectory becomes a node.  Resource reads and writes become
edges, so a later write can invalidate earlier readers and transitively taint
everything that depended on them.  Validity is a *soft* state: invalid nodes are
kept for provenance and can be retrieved explicitly, but receive a zero or
decayed score during normal retrieval.
"""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from . import config
from .resources import ResourceAccess
from .storage import transaction

_EDGE_PROPAGATE = ("DEPENDS_ON", "INVALIDATES", "SUPERSEDES")


class CausalDAG:
    def __init__(
        self,
        conn,
        decay: Optional[float] = None,
        max_depth: Optional[int] = None,
        taint_floor: Optional[float] = None,
    ):
        self.conn = conn
        self.decay = config.DAG_DECAY if decay is None else decay
        self.max_depth = config.DAG_MAX_DEPTH if max_depth is None else max_depth
        self.taint_floor = config.DAG_TAINT_FLOOR if taint_floor is None else taint_floor

    # ------------------------------------------------------------------ nodes
    def node(self, node_id: str):
        return self.conn.execute("SELECT * FROM trajectories WHERE node_id = ?", (node_id,)).fetchone()

    @staticmethod
    def summary(row) -> Dict[str, Any]:
        if row is None:
            return {}
        return {
            "node_id": row["node_id"],
            "session_id": row["session_id"],
            "scope_id": row["scope_id"],
            "step_id": row["step_id"],
            "tool_name": row["tool_name"],
            "status": row["status"],
            "sub_goal": row["sub_goal"],
            "validity": row["validity_override"] or row["validity"],
            "raw_validity": row["validity"],
            "override": row["validity_override"],
            "taint": round(float(row["taint"]), 4),
            "invalidated_by": row["invalidated_by"],
            "reason": row["reason"],
        }

    def register_node(
        self,
        node_id: str,
        trajectory_id: int,
        session_id: str,
        scope_id: str,
        accesses: Iterable[ResourceAccess],
        parent_node_id: Optional[str] = None,
        ts: Optional[float] = None,
    ) -> None:
        ts = time.time() if ts is None else ts
        accesses = list(accesses)
        with transaction(self.conn):
            seeds = self._register_locked(
                node_id, trajectory_id, session_id, scope_id, accesses, parent_node_id, ts
            )
            if seeds:
                self._propagate_locked(seeds)

    def _register_locked(
        self,
        node_id: str,
        trajectory_id: int,
        session_id: str,
        scope_id: str,
        accesses: List[ResourceAccess],
        parent_node_id: Optional[str],
        ts: float,
    ) -> List[Tuple[str, float]]:
        seeds: List[Tuple[str, float]] = []
        # Re-indexing: drop this node's previous accesses/versions but keep its
        # incoming dependency edges (they are still meaningful to consumers).
        old_resources = [
            r["resource_id"]
            for r in self.conn.execute(
                "SELECT DISTINCT resource_id FROM accesses WHERE node_id = ?", (node_id,)
            ).fetchall()
        ]
        self.conn.execute("DELETE FROM accesses WHERE node_id = ?", (node_id,))
        self.conn.execute("DELETE FROM resource_versions WHERE node_id = ?", (node_id,))
        for resource_id in old_resources:
            self._recompute_resource_version(resource_id, scope_id)

        # Reads first so UPDATE-like tools see the pre-write version.
        for access in sorted(accesses, key=lambda a: 0 if a.mode == "read" else 1):
            self._ensure_resource(access.resource_id, scope_id, access.kind)
            current = self._current_version(access.resource_id, scope_id)

            if access.mode == "read":
                self.conn.execute(
                    """INSERT OR REPLACE INTO accesses
                       (node_id, trajectory_id, resource_id, scope_id, mode, version_at_access, digest)
                       VALUES(?,?,?,?,?,?,?)""",
                    (node_id, trajectory_id, access.resource_id, scope_id, "read", current, access.digest),
                )
                writer = self._writer_for(access.resource_id, scope_id, current)
                if writer and writer != node_id:
                    self._add_edge(writer, node_id, "DEPENDS_ON", {"resource_id": access.resource_id})
                    parent_row = self.node(writer)
                    if (
                        parent_row is not None
                        and not parent_row["validity_override"]
                        and parent_row["validity"] != "fresh"
                    ):
                        taint = max(self.taint_floor, float(parent_row["taint"] or 1.0) * self.decay)
                        self._mark(
                            node_id,
                            "suspect",
                            f"depends on invalidated node {writer}",
                            writer,
                            taint,
                        )
                        seeds.append((node_id, taint))
                continue

            mode = "delete" if access.mode == "delete" else "write"
            new_version = current + 1
            self.conn.execute(
                "UPDATE resources SET current_version = ?, kind = ? WHERE resource_id = ? AND scope_id = ?",
                (new_version, access.kind, access.resource_id, scope_id),
            )
            self.conn.execute(
                """INSERT OR REPLACE INTO resource_versions
                   (resource_id, scope_id, version, node_id, ts, digest)
                   VALUES(?,?,?,?,?,?)""",
                (access.resource_id, scope_id, new_version, node_id, ts, access.digest),
            )
            self.conn.execute(
                """INSERT OR REPLACE INTO accesses
                   (node_id, trajectory_id, resource_id, scope_id, mode, version_at_access, digest)
                   VALUES(?,?,?,?,?,?,?)""",
                (node_id, trajectory_id, access.resource_id, scope_id, mode, new_version, access.digest),
            )

            # Direct supersession of prior writers.
            prior_writers = self.conn.execute(
                """SELECT node_id, version FROM resource_versions
                   WHERE resource_id = ? AND scope_id = ? AND version < ? AND node_id != ?""",
                (access.resource_id, scope_id, new_version, node_id),
            ).fetchall()
            for writer in prior_writers:
                self._mark(
                    writer["node_id"],
                    "superseded",
                    f"{access.resource_id} v{new_version} supersedes v{writer['version']}",
                    node_id,
                    0.0,
                )
                self._add_edge(node_id, writer["node_id"], "SUPERSEDES", {"resource_id": access.resource_id})
                self._add_edge(node_id, writer["node_id"], "INVALIDATES", {"resource_id": access.resource_id})
                seeds.append((writer["node_id"], 1.0))

            # Direct staleness of prior readers (including reads of v0).
            prior_readers = self.conn.execute(
                """SELECT DISTINCT node_id FROM accesses
                   WHERE resource_id = ? AND scope_id = ? AND mode = 'read'
                     AND COALESCE(version_at_access, 0) < ? AND node_id != ?""",
                (access.resource_id, scope_id, new_version, node_id),
            ).fetchall()
            for reader in prior_readers:
                self._mark(
                    reader["node_id"],
                    "stale",
                    f"read stale version of {access.resource_id} (v{new_version} exists)",
                    node_id,
                    0.5,
                )
                self._add_edge(node_id, reader["node_id"], "INVALIDATES", {"resource_id": access.resource_id})
                seeds.append((reader["node_id"], 0.5))

        if parent_node_id:
            self._add_edge(parent_node_id, node_id, "CAUSAL_PARENT", {})
            self._add_edge(parent_node_id, node_id, "DEPENDS_ON", {"kind": "causal_parent"})
            parent_row = self.node(parent_node_id)
            if (
                parent_row is not None
                and not parent_row["validity_override"]
                and parent_row["validity"] != "fresh"
            ):
                taint = max(self.taint_floor, float(parent_row["taint"] or 1.0) * self.decay)
                self._mark(
                    node_id,
                    "suspect",
                    f"causal child of invalidated node {parent_node_id}",
                    parent_node_id,
                    taint,
                )
                seeds.append((node_id, taint))

        # A (re)indexed node starts fresh unless it was directly invalidated by
        # a write that happened *after* it in this same transaction.
        row = self.node(node_id)
        if row and row["validity"] == "fresh":
            self.conn.execute(
                "UPDATE trajectories SET validity='fresh', taint=1.0, reason=NULL, invalidated_by=NULL WHERE node_id=?",
                (node_id,),
            )
        return seeds

    # ------------------------------------------------------------------ marking
    def _mark(
        self,
        node_id: str,
        validity: str,
        reason: str,
        invalidated_by: Optional[str],
        taint: float,
    ) -> None:
        self.conn.execute(
            """UPDATE trajectories
               SET validity=?, taint=?, reason=?, invalidated_by=?, validity_override=NULL
               WHERE node_id=? AND validity != 'superseded'""",
            (validity, taint, reason, invalidated_by, node_id),
        )
        # Superseded is terminal; never downgrade it to stale/suspect.
        self.conn.execute(
            """UPDATE trajectories SET validity='superseded', taint=0.0, reason=?, invalidated_by=?,
                      validity_override=NULL
               WHERE node_id=? AND validity='superseded'""",
            (reason, invalidated_by, node_id),
        )

    def _propagate_locked(self, seeds: List[Tuple[str, float]]) -> None:
        """BFS over dependency edges, tainting downstream consumers."""
        best: Dict[str, float] = {}
        frontier: List[Tuple[str, float, int]] = []
        for node_id, taint in seeds:
            if taint < best.get(node_id, 2.0):
                best[node_id] = taint
            frontier.append((node_id, taint, 0))
        while frontier:
            node, taint, depth = frontier.pop(0)
            if depth >= self.max_depth:
                continue
            edges = self.conn.execute(
                "SELECT dst, kind FROM dag_edges WHERE src = ?", (node,)
            ).fetchall()
            for edge in edges:
                if edge["kind"] not in _EDGE_PROPAGATE:
                    continue
                dst = edge["dst"]
                if dst == node:
                    continue
                child_taint = taint * self.decay
                if child_taint < self.taint_floor:
                    continue
                if child_taint >= best.get(dst, 2.0):
                    continue
                best[dst] = child_taint
                row = self.node(dst)
                if row is None or row["validity_override"]:
                    continue
                if row["validity"] in {"superseded", "stale"}:
                    # Keep the stronger direct label but keep propagating the
                    # invalidation downstream at full strength.
                    frontier.append((dst, child_taint, depth + 1))
                    continue
                self.conn.execute(
                    """UPDATE trajectories
                       SET validity='suspect', taint=?, reason=?,
                           invalidated_by=COALESCE(invalidated_by, ?)
                       WHERE node_id=?""",
                    (
                        child_taint,
                        f"transitive dependency on invalidated node {node} (depth {depth + 1})",
                        node,
                        dst,
                    ),
                )
                frontier.append((dst, child_taint, depth + 1))

    # ------------------------------------------------------------------ validity
    def recompute_validity(self, scope_id: Optional[str] = None) -> Dict[str, int]:
        """Truth-maintenance pass over a scope.

        Directly superseded/stale nodes are recomputed from current resource
        versions, then taint is propagated transitively.  This is what lets a
        re-executed remediation step "heal" its downstream consumers.
        """
        where = "WHERE scope_id = ?" if scope_id else ""
        args = (scope_id,) if scope_id else ()
        rows = self.conn.execute(f"SELECT * FROM trajectories {where}", args).fetchall()
        if not rows:
            return {"nodes": 0, "superseded": 0, "stale": 0, "suspect": 0, "fresh": 0}

        direct: Dict[str, str] = {}
        taint: Dict[str, float] = {}
        override: Set[str] = set()

        for row in rows:
            node_id = row["node_id"]
            if row["validity_override"]:
                override.add(node_id)
                direct[node_id] = row["validity_override"]
                taint[node_id] = 1.0
                continue

            accesses = self.conn.execute(
                "SELECT * FROM accesses WHERE node_id = ?", (node_id,)
            ).fetchall()
            status = "fresh"
            for acc in accesses:
                current = self._current_version(acc["resource_id"], row["scope_id"])
                if acc["mode"] in {"write", "delete"}:
                    if current > int(acc["version_at_access"] or 0):
                        status = "superseded"
                        break
                elif acc["mode"] == "read":
                    if current > int(acc["version_at_access"] or 0):
                        status = "stale"
            direct[node_id] = status
            taint[node_id] = 0.0 if status == "superseded" else (0.5 if status == "stale" else 1.0)

        # Fixed-point propagation over the dependency graph.
        for _ in range(12):
            changed = False
            edges = self.conn.execute(
                "SELECT src, dst, kind FROM dag_edges"
            ).fetchall()
            for edge in edges:
                if edge["kind"] not in _EDGE_PROPAGATE:
                    continue
                src, dst = edge["src"], edge["dst"]
                if src not in taint or dst not in direct or dst in override:
                    continue
                if taint[src] >= 1.0:
                    continue  # fresh sources do not invalidate their consumers
                candidate = taint[src] * self.decay
                if candidate >= self.taint_floor and candidate < taint.get(dst, 2.0):
                    taint[dst] = candidate
                    if direct[dst] == "fresh":
                        direct[dst] = "suspect"
                    changed = True
            if not changed:
                break

        counts = {"nodes": len(rows), "superseded": 0, "stale": 0, "suspect": 0, "fresh": 0}
        by_id = {r["node_id"]: r for r in rows}
        with transaction(self.conn):
            for node_id, status in direct.items():
                counts[status] = counts.get(status, 0) + 1
                row = by_id[node_id]
                reason = row["reason"]
                invalidated_by = row["invalidated_by"]
                if status == "fresh":
                    reason, invalidated_by = None, None
                elif status == "superseded":
                    reason = reason or "resource has a newer version"
                elif status == "stale":
                    reason = reason or "read an outdated resource version"
                else:
                    reason = reason or "transitive dependency on invalidated node"
                self.conn.execute(
                    """UPDATE trajectories
                       SET validity=?, taint=?, reason=?, invalidated_by=?
                       WHERE node_id=?""",
                    (status, taint[node_id], reason, invalidated_by, node_id),
                )
        return counts

    def confirm_state(self, node_id: str, note: str = "manual override") -> bool:
        row = self.node(node_id)
        if row is None:
            return False
        self.conn.execute(
            """UPDATE trajectories
               SET validity='fresh', taint=1.0, validity_override='fresh',
                   reason=?, invalidated_by=NULL
               WHERE node_id=?""",
            (note, node_id),
        )
        return True

    def clear_override(self, node_id: str) -> bool:
        row = self.node(node_id)
        if row is None:
            return False
        self.conn.execute("UPDATE trajectories SET validity_override=NULL WHERE node_id=?", (node_id,))
        self.recompute_validity(row["scope_id"])
        return True

    def retrieval_weight(self, row) -> float:
        validity = row["validity_override"] or row["validity"]
        taint = float(row["taint"] or 1.0)
        if validity == "superseded":
            return 0.0
        if validity == "stale":
            weight = max(0.15, 0.5 * taint)
        elif validity == "suspect":
            weight = max(0.15, taint)
        else:
            weight = 1.0
        if config.TEMPORAL_DECAY > 0:
            age_hours = max(0.0, time.time() - float(row["created_at"] or time.time())) / 3600.0
            weight *= math.exp(-config.TEMPORAL_DECAY * age_hours)
        return weight

    # ------------------------------------------------------------------ queries
    def get_lineage(self, node_id: str, limit: int = 100) -> Dict[str, Any]:
        row = self.node(node_id)
        if row is None:
            return {}
        accesses = [
            dict(a)
            for a in self.conn.execute("SELECT * FROM accesses WHERE node_id=?", (node_id,)).fetchall()
        ]
        ancestors = self._walk(node_id, reverse=True, limit=limit)
        descendants = self._walk(node_id, reverse=False, limit=limit)
        return {
            "node": self.summary(row),
            "accesses": accesses,
            "ancestors": ancestors,
            "descendants": descendants,
            "replayable": (row["validity_override"] or row["validity"]) in {"fresh", "suspect"},
        }

    def _walk(self, start: str, reverse: bool, limit: int = 100) -> List[Dict[str, Any]]:
        col, other = ("dst", "src") if reverse else ("src", "dst")
        seen: Set[str] = set()
        out: List[Dict[str, Any]] = []
        frontier = [start]
        while frontier and len(out) < limit:
            current = frontier.pop(0)
            rows = self.conn.execute(
                f"SELECT {other} AS nxt, kind FROM dag_edges WHERE {col} = ?", (current,)
            ).fetchall()
            for r in rows:
                nxt = r["nxt"]
                if nxt in seen or nxt == start:
                    continue
                seen.add(nxt)
                node = self.node(nxt)
                if node is not None:
                    item = self.summary(node)
                    item["edge_kind"] = r["kind"]
                    out.append(item)
                    frontier.append(nxt)
        return out

    def query_resource(self, resource_id: str, scope_id: Optional[str] = None) -> Dict[str, Any]:
        if scope_id:
            resources = self.conn.execute(
                "SELECT * FROM resources WHERE resource_id=? AND scope_id=?", (resource_id, scope_id)
            ).fetchall()
        else:
            resources = self.conn.execute(
                "SELECT * FROM resources WHERE resource_id=?", (resource_id,)
            ).fetchall()
        versions = []
        dependents = []
        for res in resources:
            versions.extend(
                dict(v)
                for v in self.conn.execute(
                    """SELECT v.*, t.tool_name, t.sub_goal, t.validity
                       FROM resource_versions v LEFT JOIN trajectories t ON t.node_id=v.node_id
                       WHERE v.resource_id=? AND v.scope_id=? ORDER BY v.version""",
                    (res["resource_id"], res["scope_id"]),
                ).fetchall()
            )
            dependents.extend(
                dict(a)
                for a in self.conn.execute(
                    """SELECT a.*, t.tool_name, t.validity FROM accesses a
                       LEFT JOIN trajectories t ON t.node_id=a.node_id
                       WHERE a.resource_id=? AND a.scope_id=? ORDER BY a.id""",
                    (res["resource_id"], res["scope_id"]),
                ).fetchall()
            )
        return {"resources": [dict(r) for r in resources], "versions": versions, "dependents": dependents}

    # ------------------------------------------------------------------ helpers
    def _ensure_resource(self, resource_id: str, scope_id: str, kind: str) -> None:
        self.conn.execute(
            """INSERT INTO resources(resource_id, scope_id, kind, current_version)
               VALUES(?,?,?,0) ON CONFLICT(resource_id, scope_id) DO UPDATE SET kind=excluded.kind""",
            (resource_id, scope_id, kind),
        )

    def _current_version(self, resource_id: str, scope_id: str) -> int:
        row = self.conn.execute(
            "SELECT current_version FROM resources WHERE resource_id=? AND scope_id=?",
            (resource_id, scope_id),
        ).fetchone()
        return int(row["current_version"]) if row else 0

    def _recompute_resource_version(self, resource_id: str, scope_id: str) -> None:
        row = self.conn.execute(
            "SELECT MAX(version) AS v FROM resource_versions WHERE resource_id=? AND scope_id=?",
            (resource_id, scope_id),
        ).fetchone()
        version = int(row["v"] or 0) if row else 0
        self.conn.execute(
            "UPDATE resources SET current_version=? WHERE resource_id=? AND scope_id=?",
            (version, resource_id, scope_id),
        )

    def _writer_for(self, resource_id: str, scope_id: str, version: int) -> Optional[str]:
        if version <= 0:
            return None
        row = self.conn.execute(
            """SELECT node_id FROM resource_versions
               WHERE resource_id=? AND scope_id=? AND version=?""",
            (resource_id, scope_id, version),
        ).fetchone()
        return row["node_id"] if row else None

    def _add_edge(self, src: str, dst: str, kind: str, meta: Optional[Dict[str, Any]] = None) -> None:
        if src == dst:
            return
        import json

        self.conn.execute(
            "INSERT OR IGNORE INTO dag_edges(src, dst, kind, meta) VALUES(?,?,?,?)",
            (src, dst, kind, json.dumps(meta or {})),
        )
