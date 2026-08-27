import { Hono } from "hono";
import { serve } from "@hono/node-server";
import { cors } from "hono/cors";
import { streamText } from "hono/streaming";
import {
  getSystemPrompt,
  LLM,
  type LLMResult,
  type RecalledMemory,
} from "./llm";
import { randomUUID } from "crypto";
import { ChatRole, type ChatMessage } from "@repo/shared";
import { ContextWindow, MemoryStore } from "./memory";

const app = new Hono();
app.use("*", cors());

const llm = new LLM();
const memoryStore = await MemoryStore.create();

app.post("/chat", async (c) => {
  return streamText(c, async (stream) => {
    const window = new ContextWindow();
    const { prompt } = await c.req.json<{ prompt: string }>();
    const turnId = randomUUID();

    memoryStore.append(turnId, {
      role: ChatRole.user,
      content: prompt,
    });
    const inferenceMessages = [
      getSystemPrompt(),
      ...memoryStore.getMessages(1),
    ];
    window.addMany(inferenceMessages);

    const inference = await llm.getResponse(window.build());
    const output = await resolveInference(inference, inferenceMessages);
    memoryStore.append(turnId, {
      role: ChatRole.assistant,
      content: output,
    });

    void memoryStore.rememberTurn(turnId).catch((err) => {
      console.error("Failed to save turn", err);
    });

    const tokens = output?.split(" ") ?? [];

    for (const token of tokens) {
      await stream.write(`${token} `);
    }
  });
});

app.get("/context", async (c) => {
  return c.json(memoryStore.getAll());
});

app.post("/memory/search", async (c) => {
  const { prompt, limit } = await c.req.json<{
    prompt: string;
    limit?: number;
  }>();

  return c.json(await memoryStore.debugVectorDb.search(prompt, limit));
});

serve({
  fetch: app.fetch,
  port: 3000,
});

console.log("Server running on http://localhost:3000");

async function resolveInference(result: LLMResult, messages: ChatMessage[]) {
  if (result.type === "message") {
    console.log("[llm] answered without tool call");
    return result.content;
  }

  console.log(
    `[llm] requested tool "${result.name}" with query "${result.arguments.query}"`,
  );

  const memories = await memoryStore.recall(
    result.arguments.query,
    result.arguments.limit,
  );
  console.log(
    `[memory] recalled ${memories.length} turn(s) for query "${result.arguments.query}"`,
  );
  const toolResult = formatToolResult(memories);

  const followup = await llm.getResponse([
    ...messages,
    {
      role: ChatRole.assistant,
      content: result.rawContent ?? "",
    },
    {
      role: ChatRole.user,
      content: `<tool_response name="${result.name}">\n${toolResult}\n</tool_response>`,
    },
  ]);

  if (followup.type === "message") return followup.content;

  return "I tried to retrieve additional memories, but I could not produce a final answer from them.";
}

function formatToolResult(memories: RecalledMemory[]) {
  if (memories.length === 0) {
    return "No matching memories were found.";
  }

  return JSON.stringify(memories);
}
