/**
 * Memory-Augmented Tool Policy Learning (MATP) MCP server.
 *
 * Serves over stdio. Hooks into the Predictive Context Router and the
 * Background Synthesis Engine by pre-fetching "Tool Execution Recipes" —
 * ranked historical tool policy + parameter auto-correction — instead of, or
 * alongside, plain factual data.
 */

import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { buildEmbedder } from './core/embed.js';
import { MatpPolicy } from './core/policy.js';
import { MemoryStore } from './memory/memory.js';
import { registerAll } from './tools/indexTools.js';

const SERVER_NAME = 'matp-mcp';
const SERVER_VERSION = '0.1.0';

async function main(): Promise<void> {
  const embedder = buildEmbedder();
  console.error(`[matp] embedder: ${embedder.label}`);

  const store = new MemoryStore(embedder);
  const policy = new MatpPolicy(store);

  const server = new McpServer({ name: SERVER_NAME, version: SERVER_VERSION });

  registerAll(server, store, policy);

  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error(`[matp] ${SERVER_NAME} v${SERVER_VERSION} ready over stdio.`);

  // ── Graceful shutdown ────────────────────────────────────────────────────
  // Flush pending usage-counter writes before the process exits so that
  // timesApplied increments are not lost between indexing calls.
  async function shutdown(signal: string): Promise<void> {
    console.error(`[matp] received ${signal}, shutting down…`);
    try {
      store.flush();               // persist any dirty correction counters
      await server.close();        // tear down MCP transport cleanly
    } catch (err) {
      console.error(`[matp] shutdown error: ${(err as Error).message}`);
    }
    process.exit(0);
  }

  process.on('SIGINT',  () => { void shutdown('SIGINT'); });
  process.on('SIGTERM', () => { void shutdown('SIGTERM'); });
}

main().catch((err) => {
  console.error(`[matp] fatal: ${(err as Error).message}`);
  process.exit(1);
});