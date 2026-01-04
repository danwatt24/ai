import { Tensor } from "../Tensor";

function softmaxRow(logits: number[]): number[] {
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

function entropyFromProbs(probs: number[]): number {
  let h = 0;
  for (const p of probs) {
    if (p > 0) h -= p * Math.log(p);
  }
  return h;
}

export function entropyConfidence(logits: Tensor): number {
  const [seqLen, vocabSize] = logits.shape;
  let total = 0;

  for (let i = 0; i < seqLen; i++) {
    const row = Array.from(
      logits.data.slice(i * vocabSize, (i + 1) * vocabSize)
    );
    const probs = softmaxRow(row);
    const h = entropyFromProbs(probs);
    const conf = 1 - h / Math.log(vocabSize);

    total += conf;
  }

  return total / seqLen;
}
