import { QdrantClient } from "@qdrant/js-client-rest";
import { pipeline } from "@huggingface/transformers";
import { ChatRole } from "@repo/shared";

type Interaction = {
  id: string;
  role: ChatRole;
  content: string;
};

type MemoryRecord = Interaction & {
  turnId: string;
  createdAt: string;
};

type MemorySearchResult = MemoryRecord & {
  score: number;
};

const extractor = await pipeline(
  "feature-extraction",
  "../../../../models/bge-small-en-v1.5",
);

const embed = async (text: string) => {
  const output = await extractor(text, { pooling: "mean", normalize: true });
  return Array.from(output.data) as number[];
};

export class VectorDb {
  private readonly _db: QdrantClient;
  private readonly _collection = "general";

  constructor() {
    this._db = new QdrantClient({ url: "http://127.0.0.1:6333" });
  }

  async init() {
    await this.createCollectionIfNeeded(this._collection);
  }

  private async createCollectionIfNeeded(name: string) {
    try {
      await this._db.getCollection(name);
    } catch {
      await this._db.createCollection(name, {
        vectors: { size: 384, distance: "Cosine" },
      });
    }
  }

  async upsert(record: MemoryRecord, vector: number[]) {
    const resp = await this._db.upsert(this._collection, {
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

  async search(
    prompt: string,
    limit = 5,
    threshold = 0.55,
  ): Promise<MemorySearchResult[]> {
    const vector = await embed(prompt);

    const results = await this._db.search(this._collection, {
      vector,
      limit,
      with_payload: true,
      score_threshold: threshold,
    });

    return results.map((r) => ({
      ...(r.payload as MemoryRecord),
      score: r.score,
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
  interactions: Interaction[],
) {
  const createdAt = new Date().toISOString();

  await Promise.all(
    interactions.map(async (item) => {
      const embedding = await embed(item.content);

      await vectorDb.upsert(
        {
          id: item.id,
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
