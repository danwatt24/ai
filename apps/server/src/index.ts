import { Hono } from "hono";
import { serve } from "@hono/node-server";
import { cors } from "hono/cors";
import { streamText } from "hono/streaming";
import { LLM } from "./llm";
import { randomUUID } from "crypto";
import { ChatRole } from "@repo/shared";
import { MemoryStore } from "./memory";

const app = new Hono();
app.use("*", cors());

const memoryStore = await MemoryStore.create();
const llm = new LLM(memoryStore);

app.post("/chat", async (c) => {
  return streamText(c, async (stream) => {
    const { prompt } = await c.req.json<{ prompt: string }>();
    const turnId = randomUUID();

    const userPrompt = {
      role: ChatRole.user,
      content: prompt,
    };
    memoryStore.append(turnId, userPrompt);

    const inference = await llm.getResponse(userPrompt);
    memoryStore.append(turnId, {
      role: ChatRole.assistant,
      content: inference,
    });

    void memoryStore.rememberTurn(turnId).catch((err) => {
      console.error("Failed to save turn", err);
    });

    const tokens = inference?.split(" ") ?? [];

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
