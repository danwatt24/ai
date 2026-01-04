import { Tensor } from "../Tensor";

function softmax(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

function marginConfidence(logits: number[]): number {
  const probs = softmax(logits);
  let first = 0,
    second = 0;

  for (const p of probs) {
    if (p > first) {
      second = first;
      first = p;
    } else if (p > second) {
      second = p;
    }
  }

  return first - second;
}

export function outputConfidence(logits: Tensor): number {
  const [seqLen, vocabSize] = logits.shape;
  let sum = 0;

  for (let i = 0; i < seqLen; i++) {
    const row = logits.data.slice(i * vocabSize, (i + 1) * vocabSize);
    sum += marginConfidence(Array.from(row));
  }

  return sum / seqLen;
}
