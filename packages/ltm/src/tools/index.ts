import { semanticRecall } from "./semanticRecall";
import type { LtmTool } from "./types";

export { parseToolCall } from "./toolParser";

export const tools: Record<string, LtmTool> = {
  [semanticRecall.name]: semanticRecall,
};

export type ToolCall = NonNullable<
  ReturnType<(typeof tools)[keyof typeof tools]["normalize"]>
>;

export const schemas = Object.values(tools).map((t) => t.schema);
