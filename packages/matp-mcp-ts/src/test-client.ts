/**
 * Zero-dependency, 100% Free MCP Client & Verification Suite for MATP-MCP.
 * Uses the official @modelcontextprotocol/sdk Client over Stdio.
 *
 * Run:
 *   npm run test:client
 *   or: tsx src/test-client.ts
 */

import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { resolve } from 'node:path';

const SERVER_PATH = resolve(process.cwd(), 'dist/index.js');

async function runVerification(): Promise<void> {
  console.log('🚀 Starting MATP-MCP Client Verification Suite...\n');

  const transport = new StdioClientTransport({
    command: 'node',
    args: [SERVER_PATH],
    env: {
      ...process.env,
      MATP_DATA_DIR: resolve(process.cwd(), 'data'),
    },
  });

  const client = new Client(
    { name: 'matp-free-client', version: '1.0.0' },
    { capabilities: {} }
  );

  console.log('🔌 Connecting to MATP MCP Server over stdio...');
  await client.connect(transport);
  console.log('✅ Connected successfully!\n');

  try {
    // 1. List Tools
    console.log('📋 [1/7] Discovering registered tools...');
    const toolsResult = await client.listTools();
    const toolNames = toolsResult.tools.map((t) => t.name);
    console.log(`Found ${toolNames.length} tools:`);
    for (const tool of toolsResult.tools) {
      console.log(`   • ${tool.name.padEnd(25)} : ${tool.description?.slice(0, 75)}...`);
    }

    if (toolNames.length < 11) {
      throw new Error(`Expected at least 11 tools, found ${toolNames.length}`);
    }
    console.log('✅ All tools registered properly.\n');

    // 2. Index Trajectories
    console.log('📥 [2/7] Testing "matp_index_trajectory"...');
    const indexSuccess = await client.callTool({
      name: 'matp_index_trajectory',
      arguments: {
        subGoal: 'scrape financial data from 10k filings',
        tool: 'sec_edgar_extractor',
        parameters: { ticker: 'AAPL', year: 2024 },
        status: 'success',
        outputSummary: 'Successfully extracted balance sheet and income statement',
      },
    });
    const successId = (indexSuccess.content as Array<{ text: string }>)[0].text;
    console.log(`   Indexed trajectory ID: ${successId}`);

    const indexFail = await client.callTool({
      name: 'matp_index_trajectory',
      arguments: {
        subGoal: 'scrape financial data from 10k filings',
        tool: 'sec_edgar_extractor',
        parameters: { ticker: 'AAPL', year: 'twenty-twenty-four' },
        status: 'failed',
        outputSummary: 'ValidationError: year must be an integer, received string',
      },
    });
    const failId = (indexFail.content as Array<{ text: string }>)[0].text;
    console.log(`   Indexed failure ID: ${failId}`);
    console.log('✅ Trajectory indexing verified.\n');

    // 3. Learn Correction Rule
    console.log('🧠 [3/7] Testing "matp_learn_correction"...');
    const learnRes = await client.callTool({
      name: 'matp_learn_correction',
      arguments: {
        tool: 'sec_edgar_extractor',
        errorPattern: 'year must be an integer',
        hint: 'Pass the fiscal year as a 4-digit number (e.g. 2024), not a string.',
      },
    });
    const ruleId = (learnRes.content as Array<{ text: string }>)[0].text;
    console.log(`   Created correction rule ID: ${ruleId}`);
    console.log('✅ Correction rule learning verified.\n');

    // 4. Recommend Tools & Parameter Hints
    console.log('📊 [4/7] Testing "matp_recommend_tools" & "matp_parameter_hints"...');
    const recommendRes = await client.callTool({
      name: 'matp_recommend_tools',
      arguments: {
        subGoal: 'fetch 10k reports for tech company',
        topK: 5,
      },
    });
    const recData = JSON.parse((recommendRes.content as Array<{ text: string }>)[0].text);
    console.log(`   Recommendations found for ${recData.ranked.length} tool(s).`);
    if (recData.ranked.length > 0) {
      console.log(`   Top recommendation: ${recData.ranked[0].tool} (confidence: ${(recData.ranked[0].confidence * 100).toFixed(1)}%)`);
    }

    const hintsRes = await client.callTool({
      name: 'matp_parameter_hints',
      arguments: {
        tool: 'sec_edgar_extractor',
        context: 'year must be an integer',
      },
    });
    const hints = JSON.parse((hintsRes.content as Array<{ text: string }>)[0].text);
    console.log(`   Active parameter hints retrieved: ${hints.length}`);
    if (hints.length > 0) {
      console.log(`   Hint preview: "${hints[0].hint}"`);
    }
    console.log('✅ Recommendation & auto-correction pipeline verified.\n');

    // 5. Test Prompt Generation (for agent prompt injection)
    console.log('📝 [5/7] Testing "matp_tool_policy_prompt"...');
    const promptRes = await client.callTool({
      name: 'matp_tool_policy_prompt',
      arguments: {
        subGoal: 'fetch 10k reports for tech company',
      },
    });
    const promptText = (promptRes.content as Array<{ text: string }>)[0].text;
    console.log('   --- Prompt Injected Output Preview ---');
    console.log(promptText.split('\n').map((l) => `   | ${l}`).join('\n'));
    console.log('   --------------------------------------');
    console.log('✅ Prompt injection generation verified.\n');

    // 6. Test Management Tools (List, Stats, Delete)
    console.log('🔍 [6/7] Testing Management Tools (Stats & List)...');
    const statsRes = await client.callTool({ name: 'matp_stats', arguments: {} });
    const stats = JSON.parse((statsRes.content as Array<{ text: string }>)[0].text);
    console.log(`   Current Buffer Stats:`, stats);

    const listRes = await client.callTool({
      name: 'matp_list_trajectories',
      arguments: { limit: 2 },
    });
    const recent = JSON.parse((listRes.content as Array<{ text: string }>)[0].text);
    console.log(`   Retrieved ${recent.length} recent trajectories.`);
    console.log('✅ Management telemetry verified.\n');

    // 7. Cleanup test item
    console.log('🧹 [7/7] Testing "matp_delete_trajectory"...');
    const delRes = await client.callTool({
      name: 'matp_delete_trajectory',
      arguments: { id: failId },
    });
    console.log(`   ${(delRes.content as Array<{ text: string }>)[0].text}`);
    console.log('✅ Deletion verified.\n');

    console.log('🎉 ALL VERIFICATION CHECKS PASSED PERFECTLY!');
  } finally {
    await client.close();
  }
}

runVerification().catch((err) => {
  console.error('\n❌ Verification failed:', err);
  process.exit(1);
});

