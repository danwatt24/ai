import type { ToolCall } from "./tools";

export type LtmEvent = { log?: string } & (
  | { type: "llm.answer" }
  | { type: "llm.tool_requested"; tool?: ToolCall }
  | { type: "memory.recalled"; count: number }
  | { type: "llm.progress"; content: string }
);

type Listener = (event: LtmEvent) => void;

const listeners = new Set<Listener>();

export function emit(event: LtmEvent) {
  for (const listener of listeners) {
    listener(event);
  }
}

export function subscribe(listener: Listener) {
  listeners.add(listener);
  return () => listeners.delete(listener);
}
