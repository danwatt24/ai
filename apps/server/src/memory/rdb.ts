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

export type StoredMessage = {
  id: string;
  turnId: string;
  role: ChatRole;
  content: string;
  createdAt: string;
};

export class RelationalDb {
  private _db: DbType;

  constructor() {
    this._db = new Database(":memory:");
    this._db.prepare(schema).run();
  }

  insertMessage(turnId: string, msg: ChatMessage) {
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

  getMessages(): StoredMessage[] {
    const rows = this._db
      .prepare<
        [],
        MessageRow
      >("select id, turn_id, role, content, created_at from messages order by sequence asc")
      .all();

    return rows.map(toStoredMessage);
  }

  getTurn(turnId: string): StoredMessage[] {
    const rows = this._db
      .prepare<[string], MessageRow>(
        `select id, turn_id, role, content, created_at
        from messages
        where turn_id = ?
        order by sequence asc`,
      )
      .all(turnId);

    return rows.map(toStoredMessage);
  }

  getTurnsByMessageIds(ids: string[]): StoredMessage[] {
    if (ids.length === 0) return [];

    const rows = this._db
      .prepare<[string], MessageRow>(
        `select id, turn_id, role, content, created_at
        from messages
        where turn_id in (
          select distinct turn_id
          from messages
          where id in (
            select value from json_each(?)
          )
        )
        order by sequence asc`,
      )
      .all(JSON.stringify(ids));

    return rows.map(toStoredMessage);
  }
}

function toStoredMessage(row: MessageRow): StoredMessage {
  return {
    id: row.id,
    turnId: row.turn_id,
    role: row.role,
    content: row.content,
    createdAt: row.created_at,
  };
}
