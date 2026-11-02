/**
 * End-to-end smoke test of the MATP policy pipeline (no MCP host needed).
 * Run: npm run smoke
 */

import { rm } from 'node:fs/promises';
import { buildEmbedder } from './core/embed.js';
import { MatpPolicy } from './core/policy.js';
import { MemoryStore } from './memory/memory.js';

async function main(): Promise<void> {
  const embedder = buildEmbedder();
  const dir = '/tmp/matp-smoke-' + Date.now();
  const store = new MemoryStore(embedder, dir);

  // Seed historical trajectories.
  const seed = [
    { subGoal: 'parse a 200 page PDF and extract all tables', tool: 'pdf_reader', status: 'failed', outputSummary: 'Error: context length exceeded' },
    { subGoal: 'extract tables from a long multi-page PDF report', tool: 'pdf_reader', status: 'success' },
    { subGoal: 'extract tables from a long multi-page PDF report', tool: 'pdf_reader', status: 'success' },
    { subGoal: 'extract tables from a long multi-page PDF report', tool: 'pdf_chunker', status: 'partial' },
    { subGoal: 'search financial documents by keyword', tool: 'semantic_search', status: 'success' },
    { subGoal: 'search financial documents by keyword', tool: 'semantic_search', status: 'success' },
    { subGoal: 'search legal filings', tool: 'grep', status: 'failed', outputSummary: "'--regexp' expected an argument" },
  ] as const;

  for (const s of seed) {
    await store.indexTrajectory({ subGoal: s.subGoal, tool: s.tool, parameters: {}, status: s.status, outputSummary: 'outputSummary' in s ? s.outputSummary : undefined });
  }

  await store.addCorrectionRule({
    tool: 'pdf_reader',
    errorPattern: 'context length exceeded',
    hint: 'Set max_tokens explicitly or stream pages; do not pass the whole 200-page text at once.',
    enabled: true,
  });

  const policy = new MatpPolicy(store);

  // 1) Recommend for a similar NEW sub-goal.
  const r = await policy.recommend('parse this big PDF and pull the tables out', ['pdf_reader', 'pdf_chunker', 'grep'].map((t) => ({ tool: t })));
  console.log('=== ranked ===');
  console.table(r.ranked);
  console.log('=== auto-correction hints ===');
  console.log(r.autoCorrectionHints);

  // 2) Render the prompt.
  console.log('\n=== injected prompt ===\n' + policy.renderPrompt(r));

  // 3) Parameter hints channel.
  const hints = await store.nearestCorrections('pdf_reader', 'context length exceeded', 3);
  console.log('\n=== parameter hints ===');
  console.log(hints.map((h) => h.rule.hint));

  // 4) Stats.
  console.log('\n=== stats ===');
  console.log(store.stats());

  // 5) Persistence check: flush usage counters, reload from disk.
  store.flush();
  const store2 = new MemoryStore(embedder, dir);
  console.log('reloaded trajectories:', store2.stats().trajectories);
  const rules = await store2.nearestCorrections('pdf_reader', 'context length exceeded', 1);
  console.log('reloaded rule timesApplied:', rules[0].rule.timesApplied);

  // 6) Threshold check: an unrelated sub-goal must NOT surface the pdf hint.
  const unrelated = await store2.nearestCorrections('pdf_reader', 'hello world foo bar', 1);
  console.log('unrelated-hint similarity (expect < 0.22):', unrelated[0]?.similarity ?? null);

  await rm(dir, { recursive: true, force: true });
  console.log('\nSMOKE OK');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});