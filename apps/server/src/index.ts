import { Hono } from "hono";
import { serve } from "@hono/node-server";
import { cors } from "hono/cors";
import { streamText } from "hono/streaming";
import { ContextWindow } from "./memory/contextWindow";
import { LLM } from "./llm";
import { rememberTurn, VectorDb } from "./memory/vector";
import { randomUUID } from "crypto";
import { ChatRole } from "@repo/shared";

const app = new Hono();
app.use("*", cors());

const ctxWindow = await ContextWindow.create("ctx.json");
const llm = new LLM();
const vectorDb = new VectorDb();

app.post("/chat", async (c) => {
  return streamText(c, async (stream) => {
    const { prompt } = await c.req.json<{ prompt: string }>();

    await ctxWindow.append({ role: ChatRole.user, content: prompt });
    const inference = await llm.getResponse(ctxWindow.getAll());
    await ctxWindow.append({ role: ChatRole.assistant, content: inference });

    void rememberTurn(vectorDb, randomUUID(), prompt, inference).catch(
      (err) => {
        console.error("Failed to save turn", err);
      },
    );

    const tokens = inference?.split(" ") ?? [];

    for (const token of tokens) {
      await stream.write(`${token} `);
    }
  });
});

app.get("/context", async (c) => {
  return c.json(ctxWindow.getAll());
});

serve({
  fetch: app.fetch,
  port: 3000,
});

console.log("Server running on http://localhost:3000");
