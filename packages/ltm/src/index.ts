import { ChatRole } from "@repo/shared";
import { randomUUID } from "crypto";
import { LLM } from "./llm";
import { MemoryStore } from "./memory";

export class Session {
  private _memStore: MemoryStore;
  private _llm: LLM;

  private constructor(memStore: MemoryStore, llm: LLM) {
    this._memStore = memStore;
    this._llm = llm;
  }

  async send(prompt: string) {
    const turnId = randomUUID();

    const userPrompt = {
      role: ChatRole.user,
      content: prompt,
    };
    this._memStore.append(turnId, userPrompt);

    const inference = await this._llm.getResponse(userPrompt);
    this._memStore.append(turnId, {
      role: ChatRole.assistant,
      content: inference,
    });

    void this._memStore.rememberTurn(turnId).catch((err) => {
      console.error("Failed to save turn", err);
    });

    return inference;
  }

  debugContext() {
    return this._memStore.getAll();
  }

  debugSearch(prompt: string, limit?: number) {
    return this._memStore.debugVectorDb.search(prompt, limit);
  }

  static async create() {
    const memoryStore = await MemoryStore.create();
    const llm = new LLM(memoryStore);
    return new Session(memoryStore, llm);
  }
}
