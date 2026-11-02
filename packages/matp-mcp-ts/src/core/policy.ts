/**
 * MATP policy engine.
 *
 * Turns raw, similarity-nearest execution trajectories into actionable,
 * in-prompt guidance for the agentic LLM *before* it emits a tool call:
 *
 *   1. Dynamic Tool Candidate Ranking  -> rank tools by learned success rate
 *      under similar sub-goals.
 *   2. Tool Recommendation Matrix     -> compact, prompt-injectable summary
 *      ("Tool_A succeeded 90%; Tool_B failed with context overflow").
 *   3. Parameter Auto-Correction      -> schema/formatting hints derived from
 *      past terminal/API error outputs stored in the buffer.
 */

import { MemoryStore } from '../memory/memory.js';

export interface ToolCandidate {
  /** The tool the model is considering calling. */
  tool: string;
  /** Candidate limit on how many samples to consider when scoring. */
  k?: number;
}

export interface RecommendationMatrixEntry {
  tool: string;
  successRate: number | null; // null when no samples yet
  attempts: number;
  success: number;
  failure: number;
  partial: number;
  bestColor: string; // human phrase describing the top working path
  exampleParameters: Record<string, unknown> | null;
  recentFailureReason: string | null;
}

export interface RankedTool {
  tool: string;
  successRate: number | null;
  attempts: number;
  confidence: number; // shrinks with little data (Bayesian-ish smoothing)
}

export interface RecommendationOutput {
  matrix: RecommendationMatrixEntry[];
  ranked: RankedTool[];
  autoCorrectionHints: CorrectionHint[];
  generatedAt: number;
  subGoal: string;
}

export interface CorrectionHint {
  ruleId: string;
  tool: string;
  errorPattern: string;
  hint: string;
  similarity: number;
}

const DEFAULT_TOP_K = 12;
/** Below this cosine similarity a recalled hint is too unrelated to be useful; drop it. */
const MIN_HINT_SIMILARITY = 0.22;

export class MatpPolicy {
  constructor(private readonly store: MemoryStore) {}

  /** Aggregates nearest-neighbour trajectories into the recommendation matrix. */
  async recommend(
    subGoal: string,
    candidates: ToolCandidate[] | undefined,
    opts: { topK?: number; minSamples?: number } = {}
  ): Promise<RecommendationOutput> {
    const topK = opts.topK ?? DEFAULT_TOP_K;

    const nearest = await this.store.nearestTrajectories(subGoal, topK);
    const tools = candidates && candidates.length > 0 ? new Set(candidates.map((c) => c.tool)) : null;

    // Group nearest hits by tool (only for candidate tools when provided).
    const byTool = new Map<string, RecommendationMatrixEntry>();
    for (const hit of nearest) {
      const t = hit.record.tool;
      if (tools && !tools.has(t)) continue;
      let e = byTool.get(t);
      if (!e) {
        e = {
          tool: t,
          successRate: null,
          attempts: 0,
          success: 0,
          failure: 0,
          partial: 0,
          bestColor: '',
          exampleParameters: null,
          recentFailureReason: null,
        };
        byTool.set(t, e);
      }
      e.attempts += 1;
      if (hit.record.status === 'success') e.success += 1;
      else if (hit.record.status === 'failed') {
        e.failure += 1;
        const metaErr = hit.record.meta?.error;
        e.recentFailureReason = typeof metaErr === 'string' ? metaErr : (hit.record.outputSummary ?? 'unknown');
      } else e.partial += 1;
      if (hit.record.status === 'success' && !e.exampleParameters) e.exampleParameters = hit.record.parameters;
    }

    const matrix: RecommendationMatrixEntry[] = [];
    for (const e of byTool.values()) {
      // Laplacian-smoothed success rate avoids 0%/100% overconfidence on few samples.
      e.successRate = (e.success + 1) / (e.attempts + 2);
      const winner =
        e.successRate >= 0.75
          ? 'green'
          : e.successRate >= 0.45
            ? 'amber'
            : e.failure > 0
              ? 'red'
              : 'grey';
      e.bestColor = winner;
      matrix.push(e);
    }

    // Rank candidates preferring success rate, well-balanced with attempt count.
    const ranked: RankedTool[] = matrix
      .map((m) => ({
        tool: m.tool,
        successRate: m.successRate,
        attempts: m.attempts,
        confidence: confidenceScore(m),
      }))
      .sort((a, b) => b.confidence - a.confidence);

    const autoCorrectionHints = await this.autoCorrect(subGoal, ranked.map((r) => r.tool));

    return { matrix, ranked, autoCorrectionHints, generatedAt: Date.now(), subGoal };
  }

  /** Fetches parameter-formatting hints for a specific tool from the error-correction buffer. */
  async autoCorrect(subGoal: string, tools: string[]): Promise<CorrectionHint[]> {
    const hints: CorrectionHint[] = [];
    const seen = new Set<string>();
    for (const tool of tools) {
      const hits = await this.store.nearestCorrections(tool, subGoal, 3);
      for (const h of hits) {
        if (h.similarity < MIN_HINT_SIMILARITY) continue; // too unrelated to be actionable
        if (seen.has(h.rule.id)) continue; // a '*' rule may match several tools; inject once
        seen.add(h.rule.id);
        hints.push({ ruleId: h.rule.id, tool: h.rule.tool, errorPattern: h.rule.errorPattern, hint: h.rule.hint, similarity: round(h.similarity, 3) });
        this.store.markCorrectionFired(h.rule.id);
      }
    }
    return hints;
  }

  /** Renders the matrix as compact, prompt-ready text (what you inject into the system prompt). */
  renderPrompt(s: RecommendationOutput): string {
    const lines: string[] = [];
    if (s.ranked.length === 0) {
      return 'No historical tool policy available for this sub-goal yet. Proceed heuristically and report back so it can be learned.';
    }
    lines.push(`Memory-derived tool policy for sub-goal: "${s.subGoal}"`);
    for (const r of s.ranked) {
      const rate = r.successRate === null ? 'n/a' : `${Math.round(r.successRate * 100)}%`;
      lines.push(`- ${r.tool}: success ~${rate} (${r.attempts} samples, confidence ${Math.round(r.confidence * 100)}%)`);
    }
    if (s.autoCorrectionHints.length > 0) {
      lines.push('Parameter auto-correction hints:');
      for (const h of s.autoCorrectionHints) {
        lines.push(`  - on "${h.errorPattern}" -> ${h.hint}`);
      }
    }
    lines.push('Rule: with equal intent, prefer the tool carrying the highest confidence-weighted success rate; if it previously failed with a known parameter error, apply the correction hint before calling.');
    return lines.join('\n');
  }
}

/** Bayesian-ish confidence: max data confidence grown by how many samples exist. */
function confidenceScore(m: RecommendationMatrixEntry): number {
  if (m.attempts === 0) return 0;
  const rate = m.successRate ?? 0;
  const n = m.attempts;
  // weight = n / (n + prior), classic beta posterior mean confidence.
  return rate * (n / (n + 5)) + 0.5 * (5 / (n + 5));
}

function round(x: number, d: number): number {
  const f = 10 ** d;
  return Math.round(x * f) / f;
}