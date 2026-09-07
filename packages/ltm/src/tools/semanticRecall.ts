import type { ChatCompletionTool } from "openai/resources";
import type { LtmTool, Tool } from "./types";

const TOOL_NAME = "semantic_recall" as const;

const schema: ChatCompletionTool = {
  type: "function",
  function: {
    name: TOOL_NAME,
    description: "Search remembered conversation history by relevancy",
    parameters: {
      type: "object",
      properties: {
        mode: {
          type: "string",
          enum: ["semantic"],
        },
        query: {
          type: "string",
          description: "Search query for remembered conversation history",
        },
        limit: {
          type: "number",
          description: "Maximum number of memories to retrieve",
        },
      },
      required: ["mode", "query"],
      additionalProperties: false,
    },
  },
};

type SemanticRecallTool = Tool<
  typeof TOOL_NAME,
  { mode: "semantic"; query: string; limit: number }
>;

export function normalize(
  args: unknown,
  rawContent: string,
): SemanticRecallTool | null {
  if (!args || typeof args !== "object") return null;
  const maybe = args as Partial<SemanticRecallTool["arguments"]>;

  if (typeof maybe.query !== "string") return null;

  return {
    type: "tool_call",
    name: TOOL_NAME,
    rawContent,
    arguments: {
      mode: "semantic",
      query: maybe.query,
      limit: typeof maybe.limit === "number" ? maybe.limit : 5,
    },
  };
}

export const semanticRecall: LtmTool = {
  name: TOOL_NAME,
  schema,
  normalize,
  execute: async (
    { arguments: { query, limit } }: SemanticRecallTool,
    store,
  ) => {
    return store.semanticRecall(query, limit);
  },
};
