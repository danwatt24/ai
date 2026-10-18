import { appendFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";

const chunks: Buffer[] = [];

for await (const chunk of process.stdin) {
  chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk));
}

const input = Buffer.concat(chunks).toString("utf8");
const event = JSON.parse(input) as unknown;
const logPath = fileURLToPath(new URL("../debug.logs", import.meta.url));
const logEntry = JSON.stringify({
  timestamp: new Date().toISOString(),
  event,
});

await appendFile(logPath, `${logEntry}\n`, "utf8");
