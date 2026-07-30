import { Hono } from "hono";
import { serve } from "@hono/node-server";
import { cors } from "hono/cors";
import { streamText } from "hono/streaming";
import { ContextWindow } from "./memory/contextWindow";
import { LLM } from "./llm";
import { recallSimilar, rememberTurn, VectorDb } from "./memory/vector";
import { randomUUID } from "crypto";
import { ChatRole } from "@repo/shared";

const app = new Hono();
app.use("*", cors());

const ctxWindow = new ContextWindow();

const llm = new LLM();
const vectorDb = new VectorDb();
await vectorDb.init();

app.post("/chat", async (c) => {
  return streamText(c, async (stream) => {
    const { prompt } = await c.req.json<{ prompt: string }>();
    const turnId = randomUUID();

    const promptId = ctxWindow.append(turnId, {
      role: ChatRole.user,
      content: prompt,
    });
    const inference = await llm.getResponse(ctxWindow.getMessages());
    const responseId = ctxWindow.append(turnId, {
      role: ChatRole.assistant,
      content: inference,
    });

    void rememberTurn(vectorDb, turnId, [
      { id: promptId, role: ChatRole.user, content: prompt },
      { id: responseId, role: ChatRole.assistant, content: inference },
    ]).catch((err) => {
      console.error("Failed to save turn", err);
    });

    const tokens = inference?.split(" ") ?? [];

    for (const token of tokens) {
      await stream.write(`${token} `);
    }
  });
});

app.get("/context", async (c) => {
  return c.json(ctxWindow.getAll());
});

app.post("/memory/search", async (c) => {
  const { prompt, limit } = await c.req.json<{
    prompt: string;
    limit?: number;
  }>();

  return c.json(await recallSimilar(vectorDb, prompt, limit));
});

serve({
  fetch: app.fetch,
  port: 3000,
});

console.log("Server running on http://localhost:3000");
