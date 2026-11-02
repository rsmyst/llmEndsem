"""Model Context Protocol server for MATP (unified monorepo stack).

Exposes the merged capability set over stdio JSON-RPC:
distillation-based indexing, validity-aware temporal retrieval, causal lineage,
macro skills, tool-policy ranking with self-healing parameter corrections,
adaptive per-task decay feedback and worker status.  The server is
intentionally dependency-free.
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List

from . import config
from .memory_store import MATPMemoryStore

SERVER_INFO = {"name": "matp", "version": "1.0.0"}
PROTOCOL_VERSION = "2024-11-05"

TOOLS: List[Dict[str, Any]] = [
    {
        "name": "matp_index_trajectory",
        "description": "Distill a raw tool output into a structured tuple, index it and update the causal DAG.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "step_id": {"type": "integer"},
                "sub_goal": {"type": "string"},
                "tool_name": {"type": "string"},
                "parameters": {"type": "object"},
                "status": {"type": "string"},
                "raw_output": {"type": "string"},
                "parent_step_id": {"type": "integer"},
                "session_id": {"type": "string"},
                "scope_id": {"type": "string"},
            },
            "required": ["step_id", "sub_goal", "tool_name", "parameters", "status", "raw_output"],
        },
    },
    {
        "name": "matp_query_trajectory",
        "description": "Hybrid dense+sparse retrieval with status/tool/entity filters, validity weighting and optional per-task temporal decay (time_decay: number | \"auto\").",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 3},
                "filter_status": {"type": "string"},
                "filter_tool": {"type": "string"},
                "filter_entity": {"type": "string"},
                "include_stale": {"type": "boolean", "default": False},
                "explain": {"type": "boolean", "default": False},
                "session_id": {"type": "string"},
                "scope_id": {"type": "string"},
                "time_decay": {"description": "Float lambda (per logical step) or \"auto\" for the learned per-task rate."},
                "task": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "matp_inspect_graph",
        "description": "Return the episodic Knowledge Graph (nodes, triples, central entities).",
        "inputSchema": {
            "type": "object",
            "properties": {"entity_filter": {"type": "string"}},
        },
    },
    {
        "name": "matp_get_context_summary",
        "description": "Return a compact chronological session summary for working-memory recovery.",
        "inputSchema": {
            "type": "object",
            "properties": {"session_id": {"type": "string"}, "use_neural": {"type": "boolean", "default": True}},
        },
    },
    {
        "name": "matp_get_lineage",
        "description": "Return a node's causal ancestors, descendants, resource accesses and invalidation reason.",
        "inputSchema": {
            "type": "object",
            "properties": {"node_id": {"type": "string"}},
            "required": ["node_id"],
        },
    },
    {
        "name": "matp_query_dag",
        "description": "Inspect resource versions and dependent steps for a resource (or list all resources).",
        "inputSchema": {
            "type": "object",
            "properties": {"resource_id": {"type": "string"}, "scope_id": {"type": "string"}},
        },
    },
    {
        "name": "matp_confirm_state",
        "description": "Mark a node as fresh despite automatic invalidation (manual override).",
        "inputSchema": {
            "type": "object",
            "properties": {"node_id": {"type": "string"}, "note": {"type": "string"}},
            "required": ["node_id"],
        },
    },
    {
        "name": "matp_list_macros",
        "description": "List learned composite skills, optionally filtered by status.",
        "inputSchema": {"type": "object", "properties": {"status": {"type": "string"}}},
    },
    {
        "name": "matp_match_macro",
        "description": "Rank learned macros for a goal and current state.",
        "inputSchema": {
            "type": "object",
            "properties": {"goal": {"type": "string"}, "state": {"type": "object"}, "top_k": {"type": "integer", "default": 5}},
            "required": ["goal"],
        },
    },
    {
        "name": "matp_expand_macro",
        "description": "Expand a macro into concrete tool steps using slot bindings.",
        "inputSchema": {
            "type": "object",
            "properties": {"skill_id": {"type": "string"}, "bindings": {"type": "object"}},
            "required": ["skill_id"],
        },
    },
    {
        "name": "matp_mine_macros",
        "description": "Mine recurring 10-step subroutines into reusable composite skills.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "window": {"type": "integer"},
                "stride": {"type": "integer"},
                "min_support": {"type": "integer"},
                "threshold": {"type": "number"},
                "hierarchical": {"type": "boolean", "default": False},
            },
        },
    },
    {
        "name": "matp_record_macro_use",
        "description": "Record the outcome of an expanded macro and demote repeatedly failing skills.",
        "inputSchema": {
            "type": "object",
            "properties": {"skill_id": {"type": "string"}, "success": {"type": "boolean"}},
            "required": ["skill_id", "success"],
        },
    },
    {
        "name": "matp_worker_status",
        "description": "Return queue depth, worker heartbeats and neural mode.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "matp_enqueue_digest",
        "description": "Queue an asynchronous session digest job for the neural worker.",
        "inputSchema": {"type": "object", "properties": {"session_id": {"type": "string"}}},
    },
    {
        "name": "matp_recompute_validity",
        "description": "Run a truth-maintenance pass over a scope (heals re-executed steps).",
        "inputSchema": {"type": "object", "properties": {"scope_id": {"type": "string"}}},
    },
    {
        "name": "matp_stats",
        "description": "Return token, validity, queue and macro statistics.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    # --- contribution 3: memory-derived tool policy ---------------------------
    {
        "name": "matp_recommend_tools",
        "description": "Rank candidate tools for a sub-goal from historical trajectories (Laplace-smoothed success rates, Bayesian confidence) plus parameter-correction hints.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "subGoal": {"type": "string"},
                "candidateTools": {"type": "array", "items": {"type": "string"}},
                "topK": {"type": "integer", "default": 12},
            },
            "required": ["subGoal"],
        },
    },
    {
        "name": "matp_tool_policy_prompt",
        "description": "Render a compact prompt-injectable tool policy for a sub-goal from the memory buffer.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "subGoal": {"type": "string"},
                "candidateTools": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["subGoal"],
        },
    },
    {
        "name": "matp_parameter_hints",
        "description": "Recall learned parameter-correction hints relevant to a tool call or error context.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tool": {"type": "string"},
                "context": {"type": "string"},
                "topK": {"type": "integer", "default": 3},
            },
            "required": ["tool"],
        },
    },
    {
        "name": "matp_learn_correction",
        "description": "Learn a reusable parameter-correction rule: on errorPattern, apply hint. Tool may be a specific tool name or '*' for any.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "tool": {"type": "string"},
                "errorPattern": {"type": "string"},
                "hint": {"type": "string"},
            },
            "required": ["tool", "errorPattern", "hint"],
        },
    },
    {
        "name": "matp_list_corrections",
        "description": "List learned parameter-correction rules.",
        "inputSchema": {
            "type": "object",
            "properties": {"tool": {"type": "string"}, "enabledOnly": {"type": "boolean", "default": True}},
        },
    },
    {
        "name": "matp_disable_correction",
        "description": "Disable a learned correction rule by id.",
        "inputSchema": {
            "type": "object",
            "properties": {"id": {"type": "string"}},
            "required": ["id"],
        },
    },
    # --- contribution 1: adaptive decay and logical clock ---------------------
    {
        "name": "matp_record_feedback",
        "description": "Report which memory nodes were useful for a task so the per-task temporal decay rate adapts (half-life calibration).",
        "inputSchema": {
            "type": "object",
            "properties": {"task": {"type": "string"}, "usefulNodeIds": {"type": "array", "items": {"type": "string"}}},
            "required": ["task", "usefulNodeIds"],
        },
    },
    {
        "name": "matp_get_decay_stats",
        "description": "Return learned per-task decay rates (lambda estimates and sample counts).",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "matp_advance_clock",
        "description": "Advance the global logical clock and return the current causal step.",
        "inputSchema": {
            "type": "object",
            "properties": {"steps": {"type": "integer", "default": 1}},
        },
    },
]


def _content(value: Any) -> Dict[str, Any]:
    return {"content": [{"type": "text", "text": json.dumps(value, indent=2, default=str)}]}


def _dispatch(store: MATPMemoryStore, name: str, args: Dict[str, Any]) -> Any:
    if name == "matp_index_trajectory":
        distilled = store.matp_index_trajectory(
            step_id=args.get("step_id", 0),
            sub_goal=args.get("sub_goal", ""),
            tool_name=args.get("tool_name", ""),
            parameters=args.get("parameters", {}),
            status=args.get("status", "SUCCESS"),
            raw_output=args.get("raw_output", ""),
            parent_step_id=args.get("parent_step_id"),
            session_id=args.get("session_id"),
            scope_id=args.get("scope_id"),
        )
        return distilled.to_dict()
    if name == "matp_query_trajectory":
        time_decay = args.get("time_decay")
        if time_decay is not None:
            try:
                time_decay = float(time_decay)
            except (TypeError, ValueError):
                pass  # keep string modes like "auto"
        results = store.matp_query_trajectory(
            query=args.get("query", ""),
            top_k=args.get("top_k", 3),
            filter_status=args.get("filter_status"),
            filter_tool=args.get("filter_tool"),
            filter_entity=args.get("filter_entity"),
            include_stale=args.get("include_stale", False),
            explain=args.get("explain", False),
            session_id=args.get("session_id"),
            scope_id=args.get("scope_id"),
            time_decay=time_decay,
            task=args.get("task"),
        )
        return [{"score": score, "trajectory": entry["tuple"], **{k: v for k, v in entry.items() if k != "tuple"}} for entry, score in results]
    if name == "matp_inspect_graph":
        if args.get("entity_filter"):
            return store.query_knowledge_graph(args["entity_filter"])
        return store.get_knowledge_graph()
    if name == "matp_get_context_summary":
        return store.get_session_summary(args.get("session_id"), use_neural=args.get("use_neural", True))
    if name == "matp_get_lineage":
        return store.get_lineage(args.get("node_id", ""))
    if name == "matp_query_dag":
        return store.query_dag(args.get("resource_id"), args.get("scope_id"))
    if name == "matp_confirm_state":
        return {"ok": store.confirm_state(args.get("node_id", ""), args.get("note", "manual override"))}
    if name == "matp_list_macros":
        return store.list_macros(args.get("status"))
    if name == "matp_match_macro":
        return store.match_macro(args.get("goal", ""), state=args.get("state"), top_k=args.get("top_k", 5))
    if name == "matp_expand_macro":
        return store.expand_macro(args.get("skill_id", ""), args.get("bindings", {}))
    if name == "matp_mine_macros":
        return store.mine_macros(
            window=args.get("window"),
            stride=args.get("stride"),
            min_support=args.get("min_support"),
            threshold=args.get("threshold"),
            hierarchical=args.get("hierarchical", False),
        )
    if name == "matp_record_macro_use":
        return store.record_macro_use(args.get("skill_id", ""), bool(args.get("success", False)))
    if name == "matp_worker_status":
        return store.worker_status()
    if name == "matp_enqueue_digest":
        return {"job_id": store.enqueue_digest(args.get("session_id"))}
    if name == "matp_recompute_validity":
        return store.recompute_validity(args.get("scope_id"))
    if name == "matp_stats":
        return {"memory": store.get_stats(), "macros": store.macro_stats(), "decay": store.decay_stats()}
    if name == "matp_recommend_tools":
        state = store.recommend_tools(
            sub_goal=args.get("subGoal", ""),
            candidate_tools=args.get("candidateTools"),
            top_k=args.get("topK") or 12,
            min_samples=args.get("minSamples", 0),
        )
        return {
            "ranked": state["ranked"],
            "matrix": state["matrix"],
            "autoCorrectionHints": state["autoCorrectionHints"],
            "prompt": store.policy.render_prompt(state),
        }
    if name == "matp_tool_policy_prompt":
        return store.render_policy_prompt(args.get("subGoal", ""), args.get("candidateTools"))
    if name == "matp_parameter_hints":
        return store.parameter_hints(
            tool=args.get("tool", ""),
            context=args.get("context"),
            top_k=args.get("topK", 3),
        )
    if name == "matp_learn_correction":
        return store.learn_correction(
            tool=args.get("tool", "*"),
            error_pattern=args.get("errorPattern", ""),
            hint=args.get("hint", ""),
        )
    if name == "matp_list_corrections":
        return store.list_corrections(tool=args.get("tool"), enabled_only=args.get("enabledOnly", True))
    if name == "matp_disable_correction":
        return store.disable_correction(args.get("id", ""))
    if name == "matp_record_feedback":
        return store.record_feedback(args.get("task", ""), args.get("usefulNodeIds") or [])
    if name == "matp_get_decay_stats":
        return store.decay_stats()
    if name == "matp_advance_clock":
        return {"current_step": store.advance_clock(args.get("steps", 1))}
    raise KeyError(f"Tool '{name}' not found")


def handle_rpc(store: MATPMemoryStore) -> None:
    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            req_id = request.get("id")
            method = request.get("method")
            params = request.get("params", {}) or {}

            if method == "initialize":
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": PROTOCOL_VERSION,
                        "capabilities": {"tools": {}},
                        "serverInfo": SERVER_INFO,
                    },
                }
            elif method in {"notifications/initialized", "initialized"}:
                continue
            elif method == "ping":
                response = {"jsonrpc": "2.0", "id": req_id, "result": {}}
            elif method == "tools/list":
                response = {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}
            elif method == "tools/call":
                name = params.get("name")
                args = params.get("arguments", {}) or {}
                try:
                    result = _dispatch(store, name, args)
                    response = {"jsonrpc": "2.0", "id": req_id, "result": _content(result)}
                except KeyError as exc:
                    response = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32601, "message": str(exc)},
                    }
                except Exception as exc:  # tool-level failure
                    response = {
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": {"code": -32000, "message": f"{type(exc).__name__}: {exc}"},
                    }
            else:
                response = {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32601, "message": f"Unknown method: {method}"},
                }
            sys.stdout.write(json.dumps(response, default=str) + "\n")
            sys.stdout.flush()
        except Exception as exc:
            sys.stdout.write(
                json.dumps(
                    {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": str(exc)}}
                )
                + "\n"
            )
            sys.stdout.flush()


def main() -> None:
    store = MATPMemoryStore(
        db_path=str(config.DB_PATH),
        session_id="mcp",
        scope_id="mcp",
        enable_queue=config.NEURAL_MODE != "off",
        auto_migrate=True,
    )
    try:
        handle_rpc(store)
    finally:
        store.close()


if __name__ == "__main__":
    main()
