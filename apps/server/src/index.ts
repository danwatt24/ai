import { Hono } from "hono";
import { serve } from "@hono/node-server";
import { cors } from "hono/cors";
import { streamText } from "hono/streaming";
import { sendPromptWithContext } from "./llm";

const app = new Hono();
app.use("*", cors());

app.post("/chat", async (c) => {
  return streamText(c, async (stream) => {
    const { prompt } = await c.req.json<{ prompt: string }>();
    const resp = await sendPromptWithContext(prompt);
    const tokens = resp?.split(" ") ?? [];

    for (const token of tokens) {
      await stream.write(`${token} `);
    }
  });
});

serve({
  fetch: app.fetch,
  port: 3000,
});

console.log("Server running on http://localhost:3000");
