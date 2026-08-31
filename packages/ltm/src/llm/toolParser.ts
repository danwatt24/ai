import type {
  ChatCompletionMessageFunctionToolCall,
  ChatCompletionTool,
} from "openai/resources";

export const recallTooling: ChatCompletionTool = {
  type: "function",
  function: {
    name: "recall",
    description: "Search remembered conversation history",
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
          description: "Maximum number of memories to retrieve.",
        },
      },
      required: ["mode", "query"],
      additionalProperties: false,
    },
  },
};

interface Tool<TName extends string, TArgs> {
  type: "tool_call";
  name: TName;
  arguments: TArgs;
  rawContent?: string;
}

export type ToolRecall = Tool<
  "recall",
  {
    mode: "semantic";
    query: string;
    limit: number;
  }
>;

export function parseToolCall(
  toolCalls: ChatCompletionMessageFunctionToolCall[] = [],
  content: string | null,
) {
  if (!content) content = "";
  const native = toolCalls?.find((call) => call.function.name === "recall");

  if (native) {
    return normalizeRecallToolCall(
      {
        name: native.function.name,
        arguments: parseArguments(native.function.arguments),
      },
      content,
    );
  }

  const trimmed = content.trim();

  const jsonText =
    trimmed.match(/<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/)?.[1] ??
    trimmed.match(/<tools>\s*([\s\S]*?)\s*<\/tools>/)?.[1] ??
    trimmed;

  try {
    return normalizeRecallToolCall(JSON.parse(jsonText), content);
  } catch {
    return null;
  }
}

function parseArguments(value: unknown) {
  if (!value) return null;
  if (typeof value === "object" && value !== null) return value;
  try {
    if (typeof value === "string") return JSON.parse(value);
  } catch {
    return null;
  }
  return null;
}

function normalizeRecallToolCall(
  value: unknown,
  rawContent: string,
): ToolRecall | null {
  if (!value || typeof value !== "object") return null;

  const maybe = value as Partial<ToolRecall>;

  if (maybe.name !== "recall") return null;
  if (!maybe.arguments || typeof maybe.arguments.query !== "string")
    return null;

  return {
    type: "tool_call",
    name: "recall",
    rawContent,
    arguments: {
      mode: "semantic",
      query: maybe.arguments.query,
      limit:
        typeof maybe.arguments.limit === "number" ? maybe.arguments.limit : 5,
    },
  };
}
