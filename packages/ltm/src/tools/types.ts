import type { ChatCompletionTool } from "openai/resources";
import type { MemoryStore, RecalledMemory } from "../memory";

export interface Tool<TName extends string, TArgs> {
  type: "tool_call";
  name: TName;
  arguments: TArgs;
  rawContent?: string;
}

type AnyToolCall = Tool<string, unknown>;
export type LtmTool = {
  name: string;
  schema: ChatCompletionTool;
  normalize(args: unknown, rawContent: string): AnyToolCall | null;
  execute(
    toolCall: AnyToolCall,
    memoryStore: MemoryStore,
  ): Promise<RecalledMemory[]>;
};
