import type { ChatCompletionMessageFunctionToolCall } from "openai/resources";
import { tools, type ToolCall } from ".";

export function parseToolCall(
  toolCalls: ChatCompletionMessageFunctionToolCall[] = [],
  content: string | null,
) {
  if (!content) content = "";
  const native = toolCalls.find((call) => call.function.name in tools);

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
  args: unknown,
  rawContent: string,
): ToolCall | null {
  if (!args || typeof args !== "object") return null;

  const maybe = args as Partial<ToolCall>;
  if (!maybe.name) return null;

  const tool = tools[maybe.name];
  if (!tool) return null;
  return tool.normalize(maybe.arguments, rawContent);
}
