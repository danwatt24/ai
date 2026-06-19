import OpenAI from "openai";
import type { ChatMessage } from "@repo/shared";
import type { ChatCompletionMessageParam } from "openai/resources";

const openai = new OpenAI({
  baseURL: "http://localhost:8080/v1",
  apiKey: "blah",
});

interface ChatPayload {
  systemPrompt: string;
  historicalContext: string;
  recentDialogue: string;
}

const model = "ignored-on-local";

function getMessages(payload: ChatPayload): ChatCompletionMessageParam[] {
  return [
    {
      role: "system",
      content: payload.systemPrompt,
    },
    {
      role: "user",
      content: `<historical_context>\n${payload.historicalContext}\n</historical_context>\n\n<recent_dialogue>\n${payload.recentDialogue}\n</recent_dialogue>`,
    },
  ];
}

export async function sendPromptWithHistory(payload: ChatPayload) {
  const messages = getMessages(payload);
  try {
    const response = await openai.chat.completions.create({
      model,
      messages,
      temperature: 0.3,
    });
    return response.choices[0].message.content;
  } catch (err) {
    console.error("error connecting to llama-server:", err);
  }
}

const context: ChatMessage[] = [];

export async function sendPromptWithContext(prompt: string) {
  context.push({
    role: "user",
    content: prompt,
  });
  try {
    const output = await openai.chat.completions.create({
      model,
      messages: context,
      temperature: 0.3,
    });
    const response = output.choices[0].message.content;
    context.push({
      role: "assistant",
      content: response || "",
    });
    return response;
  } catch (err) {
    console.error("error connecting to llama-server:", err);
  }
}
