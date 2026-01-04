export const config = {
  vocabSize: 20,
  seqLen: 5,
  dModel: 8,
  numheads: 2, // dModel must be evenly divisible by numHeads
  numBlocks: 1,
} as const;
