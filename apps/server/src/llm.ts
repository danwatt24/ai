import OpenAI from "openai";
import type { ChatMessage } from "@repo/shared";

interface ChatPayload {
  systemPrompt: string;
  historicalContext: string;
  recentDialogue: string;
}

interface Props {
  baseURL: string;
  apiKey: string;
}

const model = "ignored-on-local";

const defaultProps: Props = {
  baseURL: "http://localhost:8080/v1",
  apiKey: "blah",
};

export class LLM {
  private readonly _openai: OpenAI;

  constructor({ baseURL, apiKey } = defaultProps) {
    this._openai = new OpenAI({ baseURL, apiKey });
  }

  async getResponse(messages: ChatMessage[]) {
    const output = await this._openai.chat.completions.create({
      model,
      messages: messages,
      temperature: 0.3,
    });
    return output.choices[0].message.content || "";
  }
}
