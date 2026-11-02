"""Environment-driven configuration for MATP Contribution 2.

All values can be overridden through environment variables so the MCP server,
the benchmark harness and the background worker can share one configuration.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("MATP_DATA_DIR", str(REPO_ROOT / "data")))


def _int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


DB_PATH = Path(os.environ.get("MATP_DB", str(DATA_DIR / "matp.db")))
LEGACY_JSON = Path(os.environ.get("MATP_LEGACY_JSON", str(DATA_DIR / "matp.memory.json")))
RESULTS_DIR = Path(os.environ.get("MATP_RESULTS_DIR", str(REPO_ROOT / "benchmarks" / "results")))

EMBED_DIM = _int("MATP_EMBED_DIM", 256)

# off | auto | on.  In "auto" mode index calls enqueue enrichment jobs but the
# worker is only started explicitly.  Enqueueing never blocks the agent loop.
NEURAL_MODE = os.environ.get("MATP_NEURAL", "auto").lower()

OLLAMA_URL = os.environ.get("MATP_OLLAMA_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("MATP_OLLAMA_MODEL", "qwen2.5:7b-instruct")
OLLAMA_TIMEOUT = _float("MATP_OLLAMA_TIMEOUT", 60.0)

DAG_DECAY = _float("MATP_DAG_DECAY", 0.8)
DAG_MAX_DEPTH = _int("MATP_DAG_MAX_DEPTH", 4)
DAG_TAINT_FLOOR = _float("MATP_DAG_TAINT_FLOOR", 0.25)
TEMPORAL_DECAY = _float("MATP_TEMPORAL_DECAY", 0.0)

# Macro mining defaults
MACRO_WINDOW = _int("MATP_MACRO_WINDOW", 10)
MACRO_STRIDE = _int("MATP_MACRO_STRIDE", 2)
MACRO_MIN_SUPPORT = _int("MATP_MACRO_MIN_SUPPORT", 3)
MACRO_SIMILARITY = _float("MATP_MACRO_SIMILARITY", 0.80)
MACRO_MIN_SUCCESS = _float("MATP_MACRO_MIN_SUCCESS", 0.80)

WORKER_LEASE_SECONDS = _int("MATP_WORKER_LEASE", 60)
WORKER_POLL_SECONDS = _float("MATP_WORKER_POLL", 1.0)

# ---------------------------------------------------------------- combined stack
# Contribution-3 tool-policy engine (ported from matp-mcp TypeScript server)
POLICY_TOP_K = _int("MATP_POLICY_TOP_K", 12)
MIN_HINT_SIMILARITY = _float("MATP_MIN_HINT_SIMILARITY", 0.22)

# Contribution-1 adaptive decay (ported from the semantic-memory buffer)
DECAY_DEFAULT_LAMBDA = _float("MATP_DEFAULT_DECAY_LAMBDA", 0.05)
DECAY_ALPHA = _float("MATP_DECAY_ALPHA", 0.35)
