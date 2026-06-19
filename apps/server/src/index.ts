import { Hono } from "hono";
import { serve } from "@hono/node-server";
import { cors } from "hono/cors";
import { streamText } from "hono/streaming";

const app = new Hono();
app.use("*", cors());

app.post("/chat", async (c) => {
  return streamText(c, async (stream) => {
    const tokens = ["hello", " ", "world", "!"];

    for (const token of tokens) {
      await stream.write(token);
      await new Promise((r) => setTimeout(r, 500));
    }
  });
});

serve({
  fetch: app.fetch,
  port: 3000,
});

console.log("Server running on http://localhost:3000");
