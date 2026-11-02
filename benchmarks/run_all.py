"""Run all Contribution 2 benchmarks and write a single summary file.

Run:  python -m benchmarks.run_all
"""

from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks import gaia_lite, mini_swe, replay  # noqa: E402
from src.matp import benchmark  # noqa: E402

RESULTS = ROOT / "benchmarks" / "results"


def _run(name: str, fn, results: Dict[str, Any]) -> None:
    started = time.perf_counter()
    print(f"\n>>> {name}")
    try:
        results[name] = fn()
    except Exception as exc:  # keep the remaining suites running
        results[name] = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-2000:]}
        print(f"    FAILED: {type(exc).__name__}: {exc}")
    else:
        print(f"    ok ({time.perf_counter() - started:.1f}s)")


def main() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Any] = {"started_at": time.time(), "root": str(ROOT)}

    _run("compression_retrieval", lambda: benchmark.run_benchmarks(str(RESULTS)), results)
    _run("replay", lambda: replay.run(RESULTS), results)
    _run("mini_swe", lambda: mini_swe.run(RESULTS), results)
    _run("gaia_lite", lambda: gaia_lite.run(RESULTS), results)

    results["finished_at"] = time.time()
    results["duration_s"] = round(results["finished_at"] - results["started_at"], 2)
    path = RESULTS / "summary.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    failures = [name for name, value in results.items() if isinstance(value, dict) and "error" in value]
    print("\n" + "=" * 68)
    print(f"SUMMARY: {path}")
    print(f"Duration: {results['duration_s']}s | failures: {failures or 'none'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
