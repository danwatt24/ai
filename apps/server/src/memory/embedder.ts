import { pipeline } from "@huggingface/transformers";

const extractor = await pipeline(
  "feature-extraction",
  "../../../../models/bge-small-en-v1.5",
);

export const embedder = {
  embed: async (text: string) => {
    const output = await extractor(text, { pooling: "mean", normalize: true });
    return Array.from(output.data) as number[];
  },
};
