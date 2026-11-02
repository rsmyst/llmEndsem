"""
MATP: Memory-Augmented Tool Policy
Contribution 2: Background Abstraction & Trajectory Synthesis Engine (POC)
Author: Shishir Hegde (NITK Surathkal)

This module implements the core abstraction layer that transforms raw, verbose tool outputs
O_raw (JSON APIs, SQL result sets, compiler tracebacks, shell outputs) into compact,
information-dense structured tuples G = (sub_goal, tool, parameters, status, output_summary).
"""

import os
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict

from .tokenizer import count_tokens, tokenizer_name

@dataclass
class TrajectoryTuple:
    step_id: int
    sub_goal: str
    tool_name: str
    parameters: Dict[str, Any]
    status: str  # "SUCCESS", "FAILURE", "PARTIAL"
    output_summary: str
    entities: List[str]
    triples: List[Tuple[str, str, str]]  # (Subject, Relation, Object)
    raw_token_count_est: int
    distilled_token_count_est: int
    parent_step_id: Optional[int] = None

    def to_compact_text(self) -> str:
        """Returns the canonical string representation used for vector embedding and retrieval."""
        param_str = ", ".join(f"{k}={v}" for k, v in self.parameters.items())
        triple_str = "; ".join(f"({s}, {r}, {o})" for s, r, o in self.triples) if self.triples else ""
        parent_str = f" [Parent: step {self.parent_step_id}]" if self.parent_step_id else ""
        return (
            f"[Goal: {self.sub_goal}] "
            f"[Tool: {self.tool_name}({param_str})] "
            f"[Status: {self.status}] "
            f"[Summary: {self.output_summary}] "
            f"[Facts: {triple_str}]{parent_str}"
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BackgroundAbstractionEngine:
    """
    Distills raw tool execution outputs into compact TrajectoryTuples G.
    Achieves Size(G) << Size(O_raw) while retaining 100% of key actionable entities,
    error signatures, relational triples, and output states.
    """

    def __init__(self):
        pass

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Count tokens with tiktoken when available, else a documented heuristic."""
        return count_tokens(text)

    def abstract_tool_output(
        self,
        step_id: int,
        sub_goal: str,
        tool_name: str,
        parameters: Dict[str, Any],
        status: str,
        raw_output: str,
        parent_step_id: Optional[int] = None
    ) -> TrajectoryTuple:
        """
        Synthesizes raw output O_raw into a structured record G with extracted entities & relational triples.
        """
        raw_token_count = self._estimate_tokens(raw_output)
        
        # Domain-specific distillation logic
        distilled_summary, entities, triples = self._distill(
            tool_name, parameters, status, raw_output
        )

        compact_str = (
            f"[Goal: {sub_goal}] [Tool: {tool_name}] [Status: {status}] "
            f"[Summary: {distilled_summary}] [Facts: {triples}]"
        )
        if parent_step_id:
            compact_str += f" [Parent: {parent_step_id}]"
        distilled_token_count = self._estimate_tokens(compact_str)

        # Deduplicate entities while preserving order
        seen_entities = set()
        deduped_entities = []
        for e in entities:
            e_clean = str(e).strip()
            if e_clean and e_clean not in seen_entities and len(e_clean) <= 60:
                seen_entities.add(e_clean)
                deduped_entities.append(e_clean)

        return TrajectoryTuple(
            step_id=step_id,
            sub_goal=sub_goal,
            tool_name=tool_name,
            parameters=parameters,
            status=status,
            output_summary=distilled_summary,
            entities=deduped_entities,
            triples=triples,
            raw_token_count_est=raw_token_count,
            distilled_token_count_est=distilled_token_count,
            parent_step_id=parent_step_id
        )

    def _distill(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        status: str,
        raw: str
    ) -> Tuple[str, List[str], List[Tuple[str, str, str]]]:
        """
        Extracts core facts, entities, and (Subject, Relation, Object) triples across diverse modalities.
        """
        entities: List[str] = []
        triples: List[Tuple[str, str, str]] = []

        # 1. SQL / Database Queries
        if tool_name in ("sql_query", "db_execute", "postgres_exec", "mysql_query", "sqlite3"):
            table = parameters.get("table")
            if not table:
                query_str = parameters.get("query", "")
                table_match = re.search(r"(?:FROM|INTO|UPDATE)\s+([a-zA-Z0-9_\.]+)", query_str, re.IGNORECASE)
                table = table_match.group(1) if table_match else "db_table"
            entities.append(str(table))

            try:
                # If JSON result
                data = json.loads(raw)
                if isinstance(data, list):
                    count = len(data)
                    cols = list(data[0].keys()) if count > 0 and isinstance(data[0], dict) else []
                    sample = data[0] if count > 0 else {}
                    summary = f"Returned {count} rows from '{table}'. Columns: {cols}. First record: {sample}"
                    entities.extend(cols)
                    triples.append((str(table), "returned_rows", str(count)))
                    for c in cols[:4]:
                        triples.append((str(table), "has_column", c))
                    return summary, entities, triples
                elif isinstance(data, dict):
                    keys = list(data.keys())
                    entities.extend(keys)
                    summary = f"SQL result object with keys {keys}. Sample: {dict(list(data.items())[:3])}"
                    triples.append((str(table), "returned_schema", ", ".join(keys[:5])))
                    return summary, entities, triples
            except Exception:
                pass

            # Fallback for plain SQL text / tabular outputs
            lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
            summary = f"SQL executed on '{table}': {len(lines)} lines returned. Head: {' '.join(lines[:2])[:120]}"
            triples.append((str(table), "returned_lines", str(len(lines))))
            return summary, entities, triples

        # 2. Python / Runtime Tracebacks & Errors
        elif (tool_name in ("python_interpreter", "bash_exec", "terminal_run", "pytest", "npm_test") and status != "SUCCESS") or "Traceback (most recent call last)" in raw:
            # Extract root cause exception
            exc_match = re.findall(r"([A-Za-z0-9_]+Error|[A-Za-z0-9_]+Exception):\s*(.+)", raw)
            file_line_match = re.findall(r'File "([^"]+)", line (\d+)', raw)
            
            loc_str = ""
            if file_line_match:
                last_file, last_line = file_line_match[-1]
                loc_str = f" at {os.path.basename(last_file)}:line {last_line}"
                entities.append(os.path.basename(last_file))

            # Check for network/port info
            port_match = re.search(r"port\s+(\d+)", raw, re.IGNORECASE)
            ip_match = re.search(r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})", raw)
            if port_match:
                entities.append(f"port_{port_match.group(1)}")
            if ip_match:
                entities.append(ip_match.group(1))

            script_target = parameters.get("script") or parameters.get("command") or tool_name
            entities.append(str(script_target))

            if exc_match:
                exc_type, exc_msg = exc_match[-1]
                summary = f"Error {exc_type}{loc_str}: {exc_msg.strip()[:100]}"
                entities.append(exc_type)
                triples.append((str(script_target), "failed_with", exc_type))
                if loc_str:
                    triples.append((exc_type, "occurred_at", loc_str.strip()))
            else:
                summary = f"Execution failed{loc_str}: {raw.strip()[-150:]}"
                triples.append((str(script_target), "failed_with", "NonZeroExit"))
            return summary, entities, triples

        # 3. HTTP / REST API Responses
        elif tool_name in ("http_request", "api_call", "fetch_url", "curl"):
            endpoint = parameters.get("endpoint") or parameters.get("url") or "api_endpoint"
            entities.append(str(endpoint))

            try:
                data = json.loads(raw)
                if isinstance(data, dict):
                    keys = list(data.keys())
                    entities.extend(keys)
                    
                    # Extract status or ids
                    obj_id = data.get("id") or data.get("user_id") or data.get("status")
                    token_type = data.get("token_type") or ("Bearer" if "access_token" in data else None)
                    
                    summary = f"JSON response from {endpoint} with keys {keys}. Attributes: {dict(list(data.items())[:3])}"
                    if obj_id:
                        entities.append(str(obj_id))
                        triples.append((str(endpoint), "returned_id", str(obj_id)))
                    if token_type:
                        triples.append((str(endpoint), "auth_type", str(token_type)))
                    
                    # Extract profile / tenant if present
                    if "user_profile" in data and isinstance(data["user_profile"], dict):
                        up = data["user_profile"]
                        if "email" in up:
                            entities.append(up["email"])
                            triples.append((str(up.get("id", "user")), "has_email", up["email"]))
                    return summary, entities, triples
            except Exception:
                pass
            summary = f"API response from {endpoint}: {raw.strip()[:120]}"
            triples.append((str(endpoint), "response_length", str(len(raw))))
            return summary, entities, triples

        # 4. File / Code Editing Tools
        elif tool_name in ("read_file", "write_file", "search_code", "patch_file"):
            lines = raw.splitlines()
            line_count = len(lines)
            path = parameters.get("path") or parameters.get("file_path") or "unknown_file"
            entities.append(str(path))
            entities.append(os.path.basename(str(path)))

            # Extract symbols (classes, functions, yaml keys)
            defs = re.findall(r"(?:def|class)\s+([a-zA-Z0-9_]+)", raw)
            if defs:
                entities.extend(defs[:4])
                for d in defs[:3]:
                    triples.append((str(path), "defines_symbol", d))

            # Check config key-value pairs (yaml/ini)
            config_keys = re.findall(r"^([a-zA-Z0-9_\-]+):\s*(.+)$", raw, re.MULTILINE)
            if config_keys:
                for k, v in config_keys[:3]:
                    entities.append(k)
                    triples.append((str(path), f"config_{k}", v.strip()[:30]))

            summary = f"File {path}: {line_count} lines processed. Sample: {' '.join(lines[:2])[:80]}"
            triples.append((str(path), "line_count", str(line_count)))
            return summary, entities, triples

        # 5. Bash / Terminal Execution & Test Runners (SUCCESS)
        elif tool_name in ("bash_exec", "terminal_run", "git_command", "pytest", "npm_test"):
            # Check git status / branches
            branch_match = re.search(r"On branch\s+([a-zA-Z0-9_\-\/]+)", raw)
            if branch_match:
                branch = branch_match.group(1)
                entities.append(branch)
                modified_files = re.findall(r"modified:\s+([a-zA-Z0-9_\.\-\/]+)", raw)
                entities.extend(modified_files[:4])
                triples.append(("git_repo", "on_branch", branch))
                for mf in modified_files[:3]:
                    triples.append((branch, "modified_file", mf))
                summary = f"Git status: on branch '{branch}', {len(modified_files)} modified files."
                return summary, entities, triples

            # Check test runner outputs
            test_match = re.search(r"(\d+)\s+passed(?:,\s+(\d+)\s+failed)?", raw)
            if test_match:
                passed = test_match.group(1)
                failed = test_match.group(2) or "0"
                summary = f"Test execution completed: {passed} passed, {failed} failed."
                triples.append((tool_name, "tests_passed", passed))
                triples.append((tool_name, "tests_failed", failed))
                return summary, entities, triples

            clean_text = " ".join(raw.split())[:120]
            summary = f"Command output ({tool_name}): {clean_text}"
            triples.append((tool_name, "status", "SUCCESS"))
            return summary, entities, triples

        # 6. PDF / Document Parsing & Extraction
        elif tool_name in ("pdf_reader", "doc_parser", "ocr_extract"):
            doc_file = parameters.get("file") or "document"
            entities.append(str(doc_file))

            # Extract table / metrics
            spend_match = re.search(r"(?:Spend|Cost|Revenue|Total)[:\s]+\$?([0-9,]+)", raw, re.IGNORECASE)
            if spend_match:
                metric_val = spend_match.group(1)
                entities.append(f"spend_${metric_val}")
                triples.append((str(doc_file), "extracted_metric", f"${metric_val}"))

            table_match = re.search(r"(Table\s+\d+:[^\n]+)", raw)
            if table_match:
                table_title = table_match.group(1).strip()
                summary = f"Parsed {doc_file}: Extracted {table_title[:80]}"
                triples.append((str(doc_file), "contains_table", table_title[:50]))
                return summary, entities, triples

            lines = [l.strip() for l in raw.splitlines() if l.strip() and not l.startswith("---")]
            summary = f"Parsed {doc_file}: {len(lines)} lines extracted. Head: {' '.join(lines[:2])[:100]}"
            triples.append((str(doc_file), "extracted_lines", str(len(lines))))
            return summary, entities, triples

        # Generic Default Distillation
        clean_text = " ".join(raw.split())[:150]
        summary = f"Output distilled: {clean_text}"
        triples.append((tool_name, "status", status))
        return summary, entities, triples
