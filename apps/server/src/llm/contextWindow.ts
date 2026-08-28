import { type ChatMessage } from "@repo/shared";

export class ContextWindow {
  private readonly _messages: ChatMessage[] = [];

  add(message: ChatMessage) {
    this.addMany([message]);
    return this;
  }

  addMany(messages: ChatMessage[]) {
    this._messages.push(...messages);
    return this;
  }

  build() {
    if (!this._messages.length)
      throw new Error("cannot build window without messages");

    return structuredClone(this._messages);
  }
}
