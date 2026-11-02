"""Resource resolution and canonicalisation for the causal DAG.

The DAG needs a stable identity for every mutable thing an agent touches:
database tables, files, API endpoints, git branches and documents.  Resource
identifiers are scoped to a workspace snapshot so that two independent tasks on
the same repository never invalidate each other.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import urlparse

_SQL_READ = re.compile(r"^\s*(select|with|explain|pragma|show|describe)\b", re.I)
_SQL_WRITE = re.compile(r"\b(insert|update|delete|merge|create|alter|drop|truncate|replace)\b", re.I)
_SQL_DELETE = re.compile(r"\b(delete|drop|truncate)\b", re.I)
_TABLE_PAT = re.compile(r"\b(?:from|join|into|update|table)\s+([A-Za-z_][\w\.$]*)", re.I)

_READ_TOOLS = {"read_file", "search_code", "pdf_reader", "doc_parser", "ocr_extract", "list_dir"}
_WRITE_TOOLS = {"write_file", "patch_file", "edit_file", "apply_patch"}
_DOC_TOOLS = {"pdf_reader", "doc_parser", "ocr_extract"}
_READ_SHELL = re.compile(r"\b(cat|head|tail|less|more|grep|rg|ls|stat|file)\s+([^\s;&|]+)")
_WRITE_SHELL = re.compile(r"(?:^|\s)(?:>|>>)\s*([^\s;&|]+)")
_RM_SHELL = re.compile(r"\brm\s+(?:-[a-zA-Z]+\s+)*([^\s;&|]+)")
_TOUCH_SHELL = re.compile(r"\b(?:touch|mkdir|tee)\s+(?:-[a-zA-Z]+\s+)*([^\s;&|]+)")
_GIT_READ = re.compile(r"\bgit\s+(status|log|diff|show|branch|rev-parse|blame)\b", re.I)
_GIT_WRITE = re.compile(r"\bgit\s+(commit|push|merge|rebase|reset|checkout|switch|apply|cherry-pick|tag)\b", re.I)


@dataclass(frozen=True)
class ResourceAccess:
    resource_id: str
    mode: str  # read | write | delete
    kind: str = "other"
    digest: Optional[str] = None
    detail: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "resource_id": self.resource_id,
            "mode": self.mode,
            "kind": self.kind,
            "digest": self.digest,
            "detail": self.detail,
        }


def content_digest(*parts: Any) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(str(part).encode("utf-8", errors="replace"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def normalize_path(path: str, workspace: Optional[str] = None) -> str:
    """Return a stable, case-normalised, forward-slash path.

    Windows is case-insensitive and uses backslashes, so both are normalised
    before the path is used as a resource key.
    """
    path = str(path).strip().strip('"').strip("'")
    if not path:
        return path
    win_drive = re.match(r"^[A-Za-z]:", path) is not None
    if workspace and not win_drive and not os.path.isabs(path):
        path = os.path.join(workspace, path)
    if win_drive:
        # Windows-style path: normalise cross-platform without abspath, which
        # would treat a drive letter as a relative path on POSIX hosts.
        path = path.replace("\\", "/")
        drive, _, rest = path.partition("/")
        drive = drive.upper()  # drive letters are case-insensitive
        parts: list[str] = []
        for part in rest.split("/"):
            if part == "" or part == ".":
                continue
            if part == "..":
                if parts:
                    parts.pop()
                continue
            parts.append(part)
        return drive + "/" + "/".join(parts).lower()
    try:
        path = os.path.abspath(path)
    except OSError:
        pass
    path = os.path.normcase(path)
    return path.replace("\\", "/")


class ResourceResolver:
    """Extracts read/write/delete accesses from a tool invocation."""

    def __init__(self, workspace: Optional[str] = None, scope_id: str = "default"):
        self.workspace = workspace
        self.scope_id = scope_id

    # ------------------------------------------------------------------ helpers
    def _path(self, raw: Any) -> str:
        return normalize_path(str(raw), self.workspace)

    def _file(self, path: str, mode: str, digest: Optional[str] = None) -> ResourceAccess:
        return ResourceAccess(f"file:{path}", mode, "file", digest)

    def _sql(self, table: str, mode: str) -> ResourceAccess:
        table = table.strip().strip('"').strip("`").lower()
        return ResourceAccess(f"db:{table}", mode, "table")

    # ------------------------------------------------------------------ public
    def extract(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        status: str,
        raw_output: str,
        entities: Optional[Iterable[str]] = None,
        triples: Optional[Iterable[Any]] = None,
    ) -> List[ResourceAccess]:
        tool = (tool_name or "").lower()
        params = parameters or {}
        accesses: List[ResourceAccess] = []
        digest = content_digest(raw_output, params)

        if tool in {"sql_query", "db_execute", "postgres_exec", "mysql_query", "sqlite3"}:
            accesses.extend(self._sql_accesses(params, digest, raw_output))

        elif tool in _READ_TOOLS or tool in _WRITE_TOOLS:
            path = self._first_path(params)
            if path:
                mode = "write" if tool in _WRITE_TOOLS else "read"
                accesses.append(self._file(self._path(path), mode, digest if mode == "write" else None))
            elif tool in _DOC_TOOLS and params.get("file"):
                accesses.append(self._file(self._path(params["file"]), "read"))

        elif tool in {"http_request", "api_call", "fetch_url", "curl"}:
            endpoint = params.get("endpoint") or params.get("url") or ""
            if endpoint:
                method = str(params.get("method", "GET")).upper()
                mode = "read" if method in {"GET", "HEAD", "OPTIONS"} else "write"
                parsed = urlparse(endpoint)
                ident = f"{parsed.netloc}{parsed.path}" if parsed.netloc else endpoint
                if parsed.query:
                    ident = f"{ident}?{parsed.query}"
                accesses.append(ResourceAccess(f"api:{ident}", mode, "endpoint", digest if mode == "write" else None))

        elif tool in {"git_command"}:
            accesses.extend(self._git_accesses(params, digest))

        elif tool in {"bash_exec", "terminal_run", "shell", "exec"}:
            accesses.extend(self._shell_accesses(params, digest))

        elif tool in {"pytest", "npm_test", "test_runner"}:
            path = self._first_path(params)
            if path:
                accesses.append(self._file(self._path(path), "read"))

        # Document/file tools that are not in the explicit tool sets
        elif tool in _DOC_TOOLS and params.get("file"):
            accesses.append(self._file(self._path(params["file"]), "read"))

        # Fall back to explicit file parameters for unknown tools.
        if not accesses:
            path = self._first_path(params)
            if path:
                mode = "write" if status == "SUCCESS" and tool in _WRITE_TOOLS else "read"
                accesses.append(self._file(self._path(path), mode))

        # Deduplicate while keeping a deterministic order (reads before writes).
        seen = set()
        unique: List[ResourceAccess] = []
        priority = {"read": 0, "delete": 1, "write": 2}
        for acc in sorted(accesses, key=lambda a: priority.get(a.mode, 1)):
            key = (acc.resource_id, acc.mode)
            if key in seen:
                continue
            seen.add(key)
            unique.append(acc)
        return unique

    # ------------------------------------------------------------------ domain
    def _sql_accesses(self, params: Dict[str, Any], digest: str, raw_output: str = "") -> List[ResourceAccess]:
        query = str(params.get("query") or params.get("sql") or "")
        combined = f"{query} {raw_output}".strip()
        tables = []
        if params.get("table"):
            tables.append(str(params["table"]))
        tables.extend(_TABLE_PAT.findall(combined))
        if not tables:
            tables = ["unknown_table"]

        if _SQL_WRITE.search(combined):
            mode = "delete" if _SQL_DELETE.search(combined) else "write"
        elif _SQL_READ.search(combined) or not combined:
            mode = "read"
        else:
            mode = "read"

        return [self._sql(t, mode) for t in dict.fromkeys(tables)]

    def _git_accesses(self, params: Dict[str, Any], digest: str) -> List[ResourceAccess]:
        command = str(params.get("command") or params.get("cmd") or params.get("args") or "")
        repo = params.get("repo") or params.get("cwd") or self.workspace or "repo"
        repo_id = f"git:{normalize_path(repo, self.workspace)}"
        if _GIT_WRITE.search(command):
            return [ResourceAccess(repo_id, "write", "repo", digest)]
        if _GIT_READ.search(command):
            return [ResourceAccess(repo_id, "read", "repo")]
        return []

    def _shell_accesses(self, params: Dict[str, Any], digest: str) -> List[ResourceAccess]:
        command = str(params.get("command") or params.get("cmd") or params.get("script") or "")
        out: List[ResourceAccess] = []
        if not command:
            return out
        if "git " in command.lower():
            repo = params.get("cwd") or self.workspace or "repo"
            repo_id = f"git:{normalize_path(repo, self.workspace)}"
            if _GIT_WRITE.search(command):
                out.append(ResourceAccess(repo_id, "write", "repo", digest))
            elif _GIT_READ.search(command):
                out.append(ResourceAccess(repo_id, "read", "repo"))
        for m in _READ_SHELL.finditer(command):
            target = m.group(2)
            if target and not target.startswith("-"):
                out.append(self._file(self._path(target), "read"))
        for m in _WRITE_SHELL.finditer(command):
            out.append(self._file(self._path(m.group(1)), "write", digest))
        for m in _RM_SHELL.finditer(command):
            out.append(self._file(self._path(m.group(1)), "delete", digest))
        for m in _TOUCH_SHELL.finditer(command):
            out.append(self._file(self._path(m.group(1)), "write", digest))
        return out

    @staticmethod
    def _first_path(params: Dict[str, Any]) -> Optional[str]:
        for key in ("path", "file_path", "file", "filename", "target", "source", "dir"):
            value = params.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return None
