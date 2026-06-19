import type { ChatMessage } from "@repo/shared";
import fs from "fs/promises";

export class ContextWindow {
  private readonly _filePath: string;
  private readonly _window: ChatMessage[];

  private constructor(filePath: string, window: ChatMessage[]) {
    this._filePath = filePath;
    this._window = window;
  }

  async append(msg: ChatMessage) {
    this._window.push(msg);

    return writeWindow(this._filePath, this._window);
  }

  getAll() {
    return structuredClone(this._window);
  }

  async [Symbol.dispose]() {
    return writeWindow(this._filePath, this._window);
  }

  static async create(filePath = "ctx.json") {
    let data = JSON.stringify([]);
    try {
      data = await fs.readFile(filePath, "utf-8");
    } catch {}
    return new ContextWindow(filePath, JSON.parse(data));
  }
}

function writeWindow(path: string, window: ChatMessage[]) {
  const data = JSON.stringify(window);
  return fs.writeFile(path, data);
}
