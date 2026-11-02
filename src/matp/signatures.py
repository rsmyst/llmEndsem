"""Canonical action signatures used for macro-action mining.

Signatures deliberately discard literal parameter values (paths, ids, SQL
text) and retain only the tool identity, the shape of its parameters and the
execution status.  This is what lets the miner recognise the same 10-step
subroutine across different sessions, repositories and entities.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Iterable, List

TOOL_CANON = {
    "sql_query": "sql",
    "db_execute": "sql",
    "postgres_exec": "sql",
    "mysql_query": "sql",
    "sqlite3": "sql",
    "python_interpreter": "python",
    "bash_exec": "shell",
    "terminal_run": "shell",
    "shell": "shell",
    "exec": "shell",
    "git_command": "git",
    "pytest": "test",
    "npm_test": "test",
    "test_runner": "test",
    "http_request": "http",
    "api_call": "http",
    "fetch_url": "http",
    "curl": "http",
    "read_file": "fs_read",
    "search_code": "fs_read",
    "list_dir": "fs_read",
    "write_file": "fs_write",
    "patch_file": "fs_write",
    "edit_file": "fs_write",
    "apply_patch": "fs_write",
    "pdf_reader": "doc",
    "doc_parser": "doc",
    "ocr_extract": "doc",
}


def canonical_tool(tool_name: str) -> str:
    return TOOL_CANON.get((tool_name or "").lower(), (tool_name or "unknown").lower())


def value_class(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    if isinstance(value, (list, tuple, set)):
        return "list"
    if isinstance(value, dict):
        return "object"
    text = str(value).strip()
    if not text:
        return "text"
    if re.match(r"^https?://", text, re.I):
        return "url"
    if re.search(r"[\\/]", text) or re.match(r"^[\w.-]+\.[A-Za-z0-9]{1,5}$", text):
        return "path"
    if re.match(r"^\d+$", text):
        return "id"
    return "text"


def status_bucket(status: str) -> str:
    status = (status or "").upper()
    if status in {"SUCCESS", "OK", "PASSED"}:
        return "OK"
    if status in {"FAILURE", "ERROR", "FAILED"}:
        return "ERR"
    return status or "UNKNOWN"


def action_signature(
    tool_name: str,
    parameters: Dict[str, Any] | None,
    status: str = "SUCCESS",
) -> str:
    """Return a stable string signature, e.g. ``sql(table:text,query:text)|OK``."""
    params = parameters or {}
    shape = ",".join(f"{k}:{value_class(params[k])}" for k in sorted(params))
    return f"{canonical_tool(tool_name)}({shape})|{status_bucket(status)}"


def sequence_signature(steps: Iterable[Dict[str, Any]]) -> List[str]:
    return [
        action_signature(s.get("tool_name", ""), s.get("parameters"), s.get("status", "SUCCESS"))
        for s in steps
    ]


def fingerprint(sequence: Iterable[str]) -> str:
    payload = json.dumps(list(sequence), separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
