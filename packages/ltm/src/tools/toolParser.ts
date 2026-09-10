import type { ChatCompletionMessageFunctionToolCall } from "openai/resources";
import { tools, type ToolCall } from ".";

export function parseToolCall(
  toolCalls: ChatCompletionMessageFunctionToolCall[] = [],
  content: string | null,
) {
  if (!content) content = "";
  const native = toolCalls.find((call) => call.function.name in tools);

  let toolCall: ToolCall | null;
  let progress: string = "";
  if (native) {
    toolCall = normalizeRecallToolCall(
      {
        name: native.function.name,
        arguments: parseArguments(native.function.arguments),
      },
      content,
    );
    progress = content;
  } else {
    const trimmed = content.trim();
    const blocks = trimmed.split(/\r?\n\s*\r?\n/);

    const jsonText =
      trimmed.match(/<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/)?.[1] ??
      trimmed.match(/<tools>\s*([\s\S]*?)\s*<\/tools>/)?.[1] ??
      blocks.at(-1) ??
      trimmed;

    try {
      toolCall = normalizeRecallToolCall(JSON.parse(jsonText), jsonText);
      progress = blocks.slice(0, -1).join("\n\n").trim();
    } catch {
      return null;
    }
  }
  return toolCall ? { toolCall, progress } : null;
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
