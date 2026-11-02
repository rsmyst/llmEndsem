/**
 * Registers all MATP MCP tools on the server.
 *
 * Tools:
 *   matp_index_trajectory    — index an execution tuple into the memory buffer
 *   matp_recommend_tools     — ranked Tool Recommendation Matrix (full JSON)
 *   matp_tool_policy_prompt  — prompt-injectable policy text (inject into system prompt)
 *   matp_parameter_hints     — recall parameter auto-correction hints for a tool
 *   matp_learn_correction    — index a reusable parameter correction rule
 *   matp_list_trajectories   — browse stored trajectories (with filters)
 *   matp_delete_trajectory   — remove a trajectory by id
 *   matp_purge_old           — delete trajectories older than N days
 *   matp_list_corrections    — browse learned correction rules
 *   matp_disable_correction  — soft-delete a correction rule by id
 *   matp_stats               — buffer telemetry
 */

import { z } from 'zod';
import { MatpPolicy } from '../core/policy.js';
import type { MemoryStore } from '../memory/memory.js';
import type { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import type { SuccessStatus } from '../memory/memory.js';

const SUCCESS_STATUS: [string, ...string[]] = ['success', 'failed', 'partial'];

export function registerAll(server: McpServer, store: MemoryStore, policy: MatpPolicy): void {
  /* ------------------------------------------------------------------ */
  /* 1. Execution Trajectory Indexing — hook for the Background Synthesis Engine */
  /* ------------------------------------------------------------------ */
  server.registerTool(
    'matp_index_trajectory',
    {
      title: 'Index Execution Trajectory',
      description:
        "Memory-Augmented Tool Policy Learning: compress and index the full execution tuple (Sub-goal, Tool Selected, Parameters Used, Success Status) into the Semantic Memory Buffer. Call AFTER any tool invocation so future identical/similar sub-goals get ranked recommendations. Optional: pass the tool's output/error text as outputSummary.",
      inputSchema: z.object({
        subGoal: z.string().min(1).describe('The sub-goal the tool call was attempting (natural language).'),
        tool: z.string().min(1).describe('Name of the tool that was selected/called.'),
        parameters: z.record(z.string(), z.unknown()).default({}).describe('The parameter payload passed to the tool.'),
        status: z.enum(SUCCESS_STATUS).describe("Outcome: 'success' | 'failed' | 'partial'."),
        outputSummary: z
          .string()
          .optional()
          .describe('Short summary of the tool output, or (for failures) the error message / terminal output.'),
        meta: z.record(z.string(), z.unknown()).optional().describe('Optional free-form metadata (tenant id, task id, ...). Not vectorized.'),
      }),
    },
    async (args) => {
      const record = await store.indexTrajectory({
        subGoal: args.subGoal,
        tool: args.tool,
        parameters: args.parameters,
        status: args.status as SuccessStatus,
        outputSummary: args.outputSummary,
        meta: args.meta,
      });
      return str(record.id);
    }
  );

  /* ------------------------------------------------------------------ */
  /* 2. Dynamic Tool Candidate Ranking + Recommendation Matrix           */
  /* ------------------------------------------------------------------ */
  server.registerTool(
    'matp_recommend_tools',
    {
      title: 'Recommend Tools for Sub-goal',
      description:
        'Query the Semantic Memory Buffer for past tool calls made under similar sub-goals and produce a ranked Tool Recommendation Matrix: per-tool success rates, best working parameters, and any recent failure reasons. Do NOT call this from the model arbitrarily — the router calls it once per sub-goal before tool selection and injects renderPrompt output into the prompt.',
      inputSchema: z.object({
        subGoal: z.string().min(1).describe('Current sub-goal / task description.'),
        candidateTools: z.array(z.string()).optional().describe('Optional whitelist of tools the model is considering, to constrain ranking.'),
        topK: z.number().int().min(1).max(200).default(12).describe('Nearest-neighbour trajectories to consider.'),
      }),
    },
    async (args) => {
      const out = await policy.recommend(
        args.subGoal,
        (args.candidateTools ?? []).map((t) => ({ tool: t })),
        { topK: args.topK }
      );
      return str(JSON.stringify({ ranked: out.ranked, matrix: out.matrix, autoCorrectionHints: out.autoCorrectionHints, prompt: policy.renderPrompt(out) }, null, 2));
    }
  );

  /* ------------------------------------------------------------------ */
  /* 3. Prompt-ready policy text (inject straight into the system prompt) */
  /* ------------------------------------------------------------------ */
  server.registerTool(
    'matp_tool_policy_prompt',
    {
      title: 'Render Tool Policy Prompt',
      description:
        'Render a compact, prompt-injectable Tool Recommendation Matrix for the current sub-goal. The Predictive Context Router pre-fetches this and injects it into the agentic LLM prompt so the model picks the historically-best tool and applies parameter auto-correction hints before calling.',
      inputSchema: z.object({
        subGoal: z.string().min(1),
        candidateTools: z.array(z.string()).optional(),
        topK: z.number().int().min(1).max(200).default(12),
      }),
    },
    async (args) => {
      const out = await policy.recommend(
        args.subGoal,
        (args.candidateTools ?? []).map((t) => ({ tool: t })),
        { topK: args.topK }
      );
      return str(policy.renderPrompt(out));
    }
  );

  /* ------------------------------------------------------------------ */
  /* 3b. Parameter Auto-Correction — schema hint recall                  */
  /* ------------------------------------------------------------------ */
  server.registerTool(
    'matp_parameter_hints',
    {
      title: 'Parameter Auto-Correction Hints',
      description:
        'Learn + recall parameter-formatting/schema hints for a tool from past terminal/API error outputs stored in the buffer. Pass the current error/context and it returns schema hints that prevent syntax/runtime errors before the tool executes.',
      inputSchema: z.object({
        tool: z.string().min(1).describe('Tool about to be called.'),
        context: z.string().optional().describe('Current error output or sub-goal context to match against learned rules.'),
        topK: z.number().int().min(1).max(10).default(3),
      }),
    },
    async (args) => {
      const ctx = args.context ?? `about to call ${args.tool}`;
      const hits = await store.nearestCorrections(args.tool, ctx, args.topK);
      return str(JSON.stringify(hits.map((h) => ({ tool: h.rule.tool, errorPattern: h.rule.errorPattern, hint: h.rule.hint, similarity: h.similarity })), null, 2));
    }
  );

  /* ------------------------------------------------------------------ */
  /* 3c. Learn a correction rule from an observed error                  */
  /* ------------------------------------------------------------------ */
  server.registerTool(
    'matp_learn_correction',
    {
      title: 'Learn Parameter Correction Rule',
      description:
        "Index a reusable parameter-formatting correction rule by capturing a sub-string of the terminal/API error output plus the hint that fixes it. The router later injects this hint automatically when it sees the same error pattern.",
      inputSchema: z.object({
        tool: z.string().min(1).describe("Tool this applies to, or '*' for any tool."),
        errorPattern: z.string().min(1).describe('Sub-string observed in the error/terminal output (e.g. "expects type integer").'),
        hint: z.string().min(1).describe('The schema/formatting hint that prevents the error (e.g. "pass size as an integer, not a string").'),
      }),
    },
    async (args) => {
      const r = await store.addCorrectionRule({ tool: args.tool, errorPattern: args.errorPattern, hint: args.hint, enabled: true });
      return str(r.id);
    }
  );

  /* ------------------------------------------------------------------ */
  /* Management — list / delete / purge                                  */
  /* ------------------------------------------------------------------ */
  server.registerTool(
    'matp_list_trajectories',
    {
      title: 'List Stored Trajectories',
      description: 'Browse trajectories in the Semantic Memory Buffer. Supports filtering by tool name, status, and a result limit. Returns newest-first.',
      inputSchema: z.object({
        tool:   z.string().optional().describe('Filter to a specific tool name.'),
        status: z.enum(SUCCESS_STATUS).optional().describe("Filter by outcome: 'success' | 'failed' | 'partial'."),
        limit:  z.number().int().min(1).max(500).default(50).describe('Maximum rows to return.'),
      }),
    },
    async (args) => {
      const rows = store.listTrajectories({ tool: args.tool, status: args.status as SuccessStatus | undefined, limit: args.limit });
      return str(JSON.stringify(rows, null, 2));
    }
  );

  server.registerTool(
    'matp_delete_trajectory',
    {
      title: 'Delete Trajectory',
      description: 'Remove a single trajectory from the Semantic Memory Buffer by its id. Use matp_list_trajectories to find the id.',
      inputSchema: z.object({
        id: z.string().uuid().describe('UUID of the trajectory to delete.'),
      }),
    },
    async (args) => {
      const deleted = store.deleteTrajectory(args.id);
      return str(deleted ? `Deleted trajectory ${args.id}.` : `Trajectory ${args.id} not found.`);
    }
  );

  server.registerTool(
    'matp_purge_old',
    {
      title: 'Purge Old Trajectories',
      description: 'Delete all trajectories older than a given number of days. Returns the count removed.',
      inputSchema: z.object({
        maxAgeDays: z.number().int().min(1).describe('Maximum age in days to keep; older records are deleted.'),
      }),
    },
    async (args) => {
      const removed = store.purgeOldTrajectories(args.maxAgeDays);
      return str(`Removed ${removed} trajectory/trajectories older than ${args.maxAgeDays} day(s).`);
    }
  );

  server.registerTool(
    'matp_list_corrections',
    {
      title: 'List Correction Rules',
      description: 'Browse learned parameter correction rules, optionally filtered by tool.',
      inputSchema: z.object({
        tool:        z.string().optional().describe("Filter by tool name (or '*' for global rules)."),
        enabledOnly: z.boolean().default(true).describe('When true (default) only return enabled rules.'),
      }),
    },
    async (args) => {
      const rules = store.listCorrectionRules({ tool: args.tool, enabledOnly: args.enabledOnly });
      return str(JSON.stringify(rules, null, 2));
    }
  );

  server.registerTool(
    'matp_disable_correction',
    {
      title: 'Disable Correction Rule',
      description: 'Soft-delete (disable) a correction rule by its id so it is no longer injected. Use matp_list_corrections to find the id.',
      inputSchema: z.object({
        id: z.string().uuid().describe('UUID of the correction rule to disable.'),
      }),
    },
    async (args) => {
      const rule = store.disableCorrectionRule(args.id);
      return str(rule ? `Disabled correction rule ${args.id}.` : `Rule ${args.id} not found.`);
    }
  );

  /* ------------------------------------------------------------------ */
  /* Maintenance                                                         */
  /* ------------------------------------------------------------------ */
  server.registerTool(
    'matp_stats',
    {
      title: 'Semantic Memory Buffer Stats',
      description: 'Return counts of indexed trajectories, correction rules, status distribution and tools seen, plus the data directory.',
      inputSchema: z.object({}),
    },
    async () => str(JSON.stringify(store.stats(), null, 2))
  );
}

function str(text: string) {
  return { content: [{ type: 'text' as const, text }] };
}