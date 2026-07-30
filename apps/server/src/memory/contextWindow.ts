import Database, { type Database as DbType } from "better-sqlite3";
import { randomUUID } from "crypto";
import type { ChatMessage, ChatRole } from "@repo/shared";

const schema = `create table if not exists messages (
  sequence integer primary key autoincrement,
  id text unique not null,
  turn_id text not null,
  role text not null,
  content text not null,
  created_at text not null
);`;

type MessageRow = {
  id: string;
  turn_id: string;
  role: ChatRole;
  content: string;
  created_at: string;
};

export class ContextWindow {
  private _db: DbType;

  constructor() {
    this._db = new Database(":memory:");
    this._db.prepare(schema).run();
  }

  append(turnId: string, msg: ChatMessage) {
    const row = {
      id: randomUUID(),
      turnId,
      createdAt: new Date().toISOString(),
      ...msg,
    };

    this._db
      .prepare<
        [string, string, ChatRole, string, string]
      >("insert into messages (id, turn_id, role, content, created_at) values (?, ?, ?, ?, ?)")
      .run(row.id, row.turnId, row.role, row.content, row.createdAt);

    return row.id;
  }

  getAll() {
    const rows = this._db
      .prepare<
        [],
        MessageRow
      >("select id, turn_id, role, content, created_at from messages order by sequence asc")
      .all();

    return rows.map((row) => ({
      id: row.id,
      turnId: row.turn_id,
      role: row.role,
      content: row.content,
      createdAt: row.created_at,
    }));
  }

  getMessages(): ChatMessage[] {
    return this.getAll().map(({ role, content }) => ({ role, content }));
  }
}
