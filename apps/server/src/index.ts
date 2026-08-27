import { Hono } from "hono";
import { serve } from "@hono/node-server";
import { cors } from "hono/cors";
import { streamText } from "hono/streaming";
import { MemoryStore } from "./memory/memoryStore";
import {
  getSystemPrompt,
  LLM,
  type LLMResult,
  type RecalledMemory,
} from "./llm";
import { rememberTurn, VectorDb } from "./memory/vector";
import { randomUUID } from "crypto";
import { ChatRole, type ChatMessage } from "@repo/shared";
import { ContextWindow } from "./memory/contextWindow";

const app = new Hono();
app.use("*", cors());

const memoryStore = new MemoryStore();

const llm = new LLM();
const vectorDb = new VectorDb();
await vectorDb.init();

app.post("/chat", async (c) => {
  return streamText(c, async (stream) => {
    const window = new ContextWindow();
    const { prompt } = await c.req.json<{ prompt: string }>();
    const turnId = randomUUID();

    const promptId = memoryStore.append(turnId, {
      role: ChatRole.user,
      content: prompt,
    });
    const inferenceMessages = [getSystemPrompt(), ...memoryStore.getMessages(1)];
    window.addMany(inferenceMessages);

    const inference = await llm.getResponse(window.build());
    logInferenceChoice(inference);
    const output = await resolveInference(inference, inferenceMessages);
    const responseId = memoryStore.append(turnId, {
      role: ChatRole.assistant,
      content: output,
    });

    void rememberTurn(vectorDb, turnId, [
      { id: promptId, role: ChatRole.user, content: prompt },
      { id: responseId, role: ChatRole.assistant, content: output },
    ]).catch((err) => {
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

  return c.json(await vectorDb.search(prompt, limit));
});

serve({
  fetch: app.fetch,
  port: 3000,
});

console.log("Server running on http://localhost:3000");

async function resolveInference(result: LLMResult, messages: ChatMessage[]) {
  if (result.type === "message") return result.content;

  const memories = await recall(result.arguments.query, result.arguments.limit);
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

async function recall(query: string, limit: number) {
  const similar = await vectorDb.search(query, limit);
  return memoryStore.getMemories(similar.map((s) => s.id));
}

function formatToolResult(memories: RecalledMemory[]) {
  if (memories.length === 0) {
    return "No matching memories were found.";
  }

  return JSON.stringify(memories);
}

function logInferenceChoice(result: LLMResult) {
  if (result.type === "message") {
    console.log("[llm] answered without tool call");
    return;
  }

  console.log(
    `[llm] requested tool "${result.name}" with query "${result.arguments.query}"`,
  );
}
