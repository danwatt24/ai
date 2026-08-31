import type { ChatMessage } from "@repo/shared";
import { RelationalDb } from "./rdb";
import { VectorDb } from "./vector";

export type RecalledMemory = {
  turnId: string;
  messages: ChatMessage[];
};

export class MemoryStore {
  private constructor(
    private readonly _rdb: RelationalDb,
    private readonly _vectorDb: VectorDb,
  ) {}

  get debugVectorDb() {
    return this._vectorDb;
  }

  append(turnId: string, msg: ChatMessage) {
    return this._rdb.insertMessage(turnId, msg);
  }

  getAll() {
    return this._rdb.getMessages();
  }

  getMessages(limit?: number): ChatMessage[] {
    let msgs = this.getAll();
    if (limit) msgs = msgs.slice(-limit);
    return msgs.map(({ role, content }) => ({ role, content }));
  }

  getMemories(ids: string[]): RecalledMemory[] {
    const rows = this._rdb.getTurnsByMessageIds(ids);
    const memories = new Map<string, ChatMessage[]>();

    rows.forEach((row) => {
      if (!memories.has(row.turnId)) memories.set(row.turnId, []);
      memories.get(row.turnId)?.push({ role: row.role, content: row.content });
    });

    return [...memories.entries()].map(([turnId, messages]) => ({
      turnId,
      messages,
    }));
  }

  async rememberTurn(turnId: string) {
    const messages = this._rdb.getTurn(turnId);

    await Promise.all(
      messages.map((msg) =>
        this._vectorDb.remember({
          id: msg.id,
          role: msg.role,
          content: msg.content,
          turnId: msg.turnId,
          createdAt: msg.createdAt,
        }),
      ),
    );
  }

  async recall(query: string, limit: number = 5) {
    const similar = await this._vectorDb.search(query, limit);
    return this.getMemories(similar.map((s) => s.id));
  }

  static async create() {
    const rdb = new RelationalDb();
    const vectorDb = new VectorDb();
    await vectorDb.init();

    return new MemoryStore(rdb, vectorDb);
  }
}
