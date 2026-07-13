import { QdrantClient } from "@qdrant/js-client-rest";
import { ChatRole } from "@repo/shared";
import { randomUUID } from "crypto";
import { embedder } from "./embedder";

type MemoryRecord = {
  id: string;
  role: ChatRole;
  content: string;
  turnId: string;
  createdAt: string;
};

export class VectorDb {
  private readonly _db: QdrantClient;
  private readonly _collections: Set<string>;

  constructor() {
    this._db = new QdrantClient({ url: "http://127.0.0.1:6333" });
    this._collections = new Set();
  }

  private async ensureCollection(name: string) {
    if (this._collections.has(name)) return;
    try {
      await this._db.getCollection(name);
    } catch {
      await this._db.createCollection(name, {
        vectors: { size: 384, distance: "Cosine" },
      });
      this._collections.add(name);
    }
  }

  async upsert(record: MemoryRecord, vector: number[]) {
    await this.ensureCollection("general");

    const resp = await this._db.upsert("general", {
      points: [{ id: record.id, vector, payload: record }],
    });
    if (resp.status !== "completed") {
      // probably should not throw but retry instead
      throw new Error("failed to insert into vector db");
    }
  }

  async get(collectionName: string, ...vectorIds: string[]) {
    const results = await this._db.retrieve(collectionName, {
      ids: vectorIds,
      with_payload: true,
    });
    return results.map((r) => ({
      id: r.id as string,
      ...(r.payload as Omit<MemoryRecord, "id">),
    }));
  }

  async purge() {
    const result = await this._db.getCollections();
    await Promise.allSettled(
      result.collections.map((col) => this._db.deleteCollection(col.name)),
    );
  }
}

export async function rememberTurn(
  vectorDb: VectorDb,
  turnId: string,
  prompt: string,
  inference: string,
) {
  const createdAt = new Date().toISOString();

  const turn = [
    { role: ChatRole.user, content: prompt },
    { role: ChatRole.assistant, content: inference },
  ];

  await Promise.all(
    turn.map(async (item) => {
      const embedding = await embedder.embed(item.content);

      await vectorDb.upsert(
        {
          id: randomUUID(),
          role: item.role,
          content: item.content,
          turnId,
          createdAt,
        },
        embedding,
      );
    }),
  );
}
