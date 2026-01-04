export const config = {
  vocabSize: 20,
  seqLen: 4,
  dModel: 8,
  numheads: 2, // dModel must be evenly divisible by numHeads
} as const;
