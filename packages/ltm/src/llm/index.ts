import { ChatRole, type ChatMessage } from "@repo/shared";
import { parseToolCall, schemas, tools } from "../tools";
import { model } from "./model";
import type { MemoryStore, RecalledMemory } from "../memory";
import type { ChatCompletionMessage } from "openai/resources";
import { ContextWindow } from "./contextWindow";
import { emit } from "../events";

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
    const output = await model.getResponse(window.build(), schemas);
    return await this.resolveInference(output, window);
  }

  private async resolveInference(
    output: ChatCompletionMessage,
    window: ContextWindow,
  ) {
    for (let i = 0; i < 3; i++) {
      const parsed = parseToolCall(
        output.tool_calls?.filter((f) => f.type === "function"),
        output.content,
      );

      if (!parsed) {
        emit({ type: "llm.answer", log: "[llm] answered without tool call" });
        return output.content ?? "";
      }

      const { toolCall, progress } = parsed;

      if (progress) {
        emit({
          type: "llm.progress",
          content: progress,
          log: progress,
        });
      }

      const tool = tools[toolCall.name];
      if (!tool) {
        emit({
          type: "llm.tool_requested",
          log: `[llm] requested invalid tool "${toolCall.name}"`,
        });
        return `The model requested an unsupported tool: ${toolCall.name}`;
      }

      emit({
        type: "llm.tool_requested",
        tool: toolCall,
        log: `[llm] requested tool ${toolCall.name}`,
      });

      const memories = await tool.execute(toolCall, this._memStore);
      emit({
        type: "memory.recalled",
        count: memories.length,
        log: `[memory] recalled ${memories.length} turn(s)`,
      });

      window
        .add({
          role: ChatRole.assistant,
          content: toolCall.rawContent ?? "",
        })
        .add({
          role: ChatRole.user,
          content: `<tool_response name="${toolCall.name}">\n${formatToolResult(memories)}\n</tool_response>`,
        });
      output = await model.getResponse(window.build(), schemas);
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
