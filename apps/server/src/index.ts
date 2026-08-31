import { Hono } from "hono";
import { serve } from "@hono/node-server";
import { cors } from "hono/cors";
import { streamText } from "hono/streaming";
import { Session } from "./session";

const app = new Hono();
app.use("*", cors());

const session = await Session.create();

app.post("/chat", async (c) => {
  return streamText(c, async (stream) => {
    const { prompt } = await c.req.json<{ prompt: string }>();
    const inference = await session.send(prompt);

    const tokens = inference?.split(" ") ?? [];
    for (const token of tokens) {
      await stream.write(`${token} `);
    }
  });
});

app.get("/context", async (c) => {
  return c.json(session.debugContext());
});

app.post("/memory/search", async (c) => {
  const { prompt, limit } = await c.req.json<{
    prompt: string;
    limit?: number;
  }>();

  return c.json(await session.debugSearch(prompt, limit));
});

serve({
  fetch: app.fetch,
  port: 3000,
});

console.log("Server running on http://localhost:3000");
