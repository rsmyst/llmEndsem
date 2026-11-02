"""Adaptive temporal decay: per-task lambda learned from agent feedback.

Unified-system port of the contribution-1 tuner (rahul).  Instead of one fixed
decay rate, each named task keeps its own estimate:

    lambda_task <- (1 - alpha) * lambda_task + alpha * ln(2) / median_useful_age

The tuner picks the half-life that makes the median age of states the agent
reported useful decay by 50%.  Tasks whose useful state is always fresh get a
large lambda (strong recency preference); tasks that still need old state get a
small lambda.  ``time_decay="auto"`` on retrieval uses the learned value.

Unlike the original JSON side-file this implementation persists through the
shared SQLite ``meta`` table so the monorepo stack stays single-file durable.
"""

from __future__ import annotations

import json
import math
import threading
from typing import Any, Dict, List, Optional

from .storage import get_meta, transaction

_MIN_LAMBDA = 0.005
_MAX_LAMBDA = 0.5
_DEFAULT_ALPHA = 0.35
_DEFAULT_LAMBDA = 0.05


class DecayTuner:
    """Per-task exponential-decay rate learned from retrieval feedback."""

    def __init__(self, conn, default_lambda: float = _DEFAULT_LAMBDA, alpha: float = _DEFAULT_ALPHA):
        self.conn = conn
        self._default = default_lambda
        self._alpha = alpha
        self._lock = threading.Lock()

    # ---------------------------------------------------------------- helpers

    @staticmethod
    def _clamp(value: float) -> float:
        return max(_MIN_LAMBDA, min(_MAX_LAMBDA, value))

    @staticmethod
    def _task_key(task: str) -> str:
        return f"decay:task:{task.strip().lower() or 'default'}"

    # -------------------------------------------------------------------- API

    def lambda_for(self, task: Optional[str] = None) -> float:
        """Learned lambda for a task, or the default when never seen."""
        if not task:
            return self._default
        raw = get_meta(self.conn, self._task_key(task))
        if raw is None:
            return self._default
        try:
            return float(json.loads(raw).get("lambda_value", self._default))
        except (ValueError, TypeError):
            return self._default

    def observe(self, task: str, useful_ages: List[int]) -> float:
        """Update the task's lambda from the ages of states reported useful.

        ``useful_ages`` are (t_curr - t_i) logical-step ages of each feedback
        item.  Empty feedback leaves the current estimate unchanged.
        """
        task = (task or "").strip() or "default"
        ages = sorted(int(a) for a in useful_ages if int(a) >= 0)
        if not ages:
            return self.lambda_for(task)
        median_age = ages[len(ages) // 2]
        # Half-life target: the median useful age should sit near 50% decay.
        target = math.log(2.0) / max(median_age, 1)
        with self._lock:
            raw = get_meta(self.conn, self._task_key(task))
            cur = self._default if raw is None else self.lambda_for(task)
            if raw is None:
                new_lambda = target
                samples = 1
            else:
                try:
                    samples = int(json.loads(raw).get("samples", 0)) + 1
                except (ValueError, TypeError):
                    samples = 1
                new_lambda = target
            value = self._clamp(new_lambda)
            with transaction(self.conn):
                self.conn.execute(
                    "INSERT INTO meta(key, value) VALUES(?, ?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (
                        self._task_key(task),
                        json.dumps({"lambda_value": value, "samples": samples}),
                    ),
                )
            return value

    def stats(self) -> Dict[str, Any]:
        tasks: Dict[str, Any] = {}
        rows = self.conn.execute(
            "SELECT key, value FROM meta WHERE key LIKE 'decay:task:%'"
        ).fetchall()
        for row in rows:
            name = row["key"].split(":", 2)[2]
            try:
                rec = json.loads(row["value"])
                tasks[name] = {
                    "lambda_value": round(float(rec.get("lambda_value", self._default)), 4),
                    "samples": int(rec.get("samples", 0)),
                }
            except (ValueError, TypeError):
                continue
        return {"default_lambda": self._default, "alpha": self._alpha, "tasks": tasks}
