import type { ChatCompletionTool } from "openai/resources";
import type { LtmTool, Tool } from "./types";

const TOOL_NAME = "recency_recall" as const;

const schema: ChatCompletionTool = {
  type: "function",
  function: {
    name: TOOL_NAME,
    description: "Search remembered conversation history by recency",
    parameters: {
      type: "object",
      properties: {
        role: {
          type: "string",
          description: "The role to return to filter by",
        },
        limit: {
          type: "number",
          description: "Maximum number of memories to retrieve",
        },
      },
      required: ["limit"],
    },
  },
};

type RecencyRecallTool = Tool<
  typeof TOOL_NAME,
  { role?: "user" | "assistant"; limit: number }
>;

export function normalize(
  args: unknown,
  rawContent: string,
): RecencyRecallTool | null {
  if (!args || typeof args !== "object") return null;
  const maybe = args as Partial<RecencyRecallTool["arguments"]>;

  return {
    type: "tool_call",
    name: TOOL_NAME,
    rawContent,
    arguments: {
      role:
        maybe.role === "user" || maybe.role === "assistant"
          ? maybe.role
          : undefined,
      limit: typeof maybe.limit === "number" ? maybe.limit : 5,
    },
  };
}

export const recencyRecall: LtmTool = {
  name: TOOL_NAME,
  schema,
  normalize,
  execute: async ({ arguments: { role, limit } }: RecencyRecallTool, store) =>
    store.recencyRecall(store.activeTurnId, role, limit),
};
