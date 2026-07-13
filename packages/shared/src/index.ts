export const ChatRole = {
  system: "system",
  user: "user",
  assistant: "assistant",
} as const;

export type ChatRole = (typeof ChatRole)[keyof typeof ChatRole];

export interface ChatMessage {
  role: ChatRole;
  content: string;
}
