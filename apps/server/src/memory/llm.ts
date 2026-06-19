import OpenAI from "openai";
import type { ChatMessage } from "@repo/shared";
import type { ChatCompletionMessageParam } from "openai/resources";

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

  // private getMessages(payload: ChatPayload): ChatCompletionMessageParam[] {
  //   return [
  //     {
  //       role: "system",
  //       content: payload.systemPrompt,
  //     },
  //     {
  //       role: "user",
  //       content: `<historical_context>\n${payload.historicalContext}\n</historical_context>\n\n<recent_dialogue>\n${payload.recentDialogue}\n</recent_dialogue>`,
  //     },
  //   ];
  // }

  // async sendPromptWithHistory(payload: ChatPayload) {
  //   const messages = this.getMessages(payload);
  //   try {
  //     const response = await this._openai.chat.completions.create({
  //       model,
  //       messages,
  //       temperature: 0.3,
  //     });
  //     return response.choices[0].message.content;
  //   } catch (err) {
  //     console.error("error connecting to llama-server:", err);
  //   }
  // }

  async sendContext(messages: ChatMessage[]) {
    const output = await this._openai.chat.completions.create({
      model,
      messages: messages,
      temperature: 0.3,
    });
    return output.choices[0].message.content || "";
  }
}
