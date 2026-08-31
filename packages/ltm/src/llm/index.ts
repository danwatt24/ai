import { ChatRole, type ChatMessage } from "@repo/shared";
import { parseToolCall, recallTooling } from "./toolParser";
import { model } from "./model";
import type { MemoryStore, RecalledMemory } from "../memory";
import type { ChatCompletionMessage } from "openai/resources";
import { ContextWindow } from "./contextWindow";

export class LLM {
  private _memStore: MemoryStore;
  constructor(memStore: MemoryStore) {
    this._memStore = memStore;
  }

  async getResponse(prompt: ChatMessage): Promise<string> {
    const window = new ContextWindow().addMany([
      model.getSystemPrompt(),
      prompt,
    ]);
    const output = await model.getResponse(window.build(), recallTooling);
    return await this.resolveInference(output, window);
  }

  private async resolveInference(
    output: ChatCompletionMessage,
    window: ContextWindow,
  ) {
    for (let i = 0; i < 3; i++) {
      const toolCall = parseToolCall(
        output.tool_calls?.filter((f) => f.type === "function"),
        output.content,
      );

      if (!toolCall) {
        console.log("[llm] answered without tool call");
        return output.content ?? "";
      }

      console.log(
        `[llm] requested tool "${toolCall.name}" with query "${toolCall.arguments.query}"`,
      );

      const memories = await this._memStore.recall(
        toolCall.arguments.query,
        toolCall.arguments.limit,
      );

      console.log(
        `[memory] recalled ${memories.length} turn(s) for query "${toolCall.arguments.query}"`,
      );

      window
        .add({
          role: ChatRole.assistant,
          content: toolCall.rawContent ?? "",
        })
        .add({
          role: ChatRole.user,
          content: `<tool_response name="${toolCall.name}">\n${formatToolResult(memories)}\n</tool_response>`,
        });
      output = await model.getResponse(window.build(), recallTooling);
    }
    return "Too many tool call requests";
  }
}

function formatToolResult(memories: RecalledMemory[]) {
  if (memories.length === 0) {
    return "No matching memories were found.";
  }

  return JSON.stringify(memories);
}
