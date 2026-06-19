import type { ContextWindow } from "./dal/contextWindow";
import type { LLM } from "./llm";

export class LTM {
  private readonly _llm: LLM;
  private readonly _ctx: ContextWindow;

  constructor(llm: LLM, ctx: ContextWindow) {
    this._llm = llm;
    this._ctx = ctx;
  }

  async getInference(prompt: string) {
    await this._ctx.append({
      role: "user",
      content: prompt,
    });
    const output = await this._llm.sendContext(this._ctx.getAll());
    await this._ctx.append({
      role: "assistant",
      content: output,
    });

    return output;
  }

  getContext() {
    return this._ctx.getAll();
  }
}
