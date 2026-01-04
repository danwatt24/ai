import { Tensor } from "./Tensor";

export function crossEntropyWithGrad(
  logits: Tensor,
  targets: number[]
): number {
  const [seqLen, vocabSize] = logits.shape;

  logits.grad = new Float32Array(logits.data.length);

  let loss = 0;

  for (let i = 0; i < seqLen; i++) {
    // softmax (numerically stable)
    let max = -Infinity;
    for (let j = 0; j < vocabSize; j++) {
      max = Math.max(max, logits.data[i * vocabSize + j]);
    }

    let sum = 0;
    for (let j = 0; j < vocabSize; j++) {
      sum += Math.exp(logits.data[i * vocabSize + j] - max);
    }

    for (let j = 0; j < vocabSize; j++) {
      const p = Math.exp(logits.data[i * vocabSize + j] - max) / sum;
      const grad = p - (j === targets[i] ? 1 : 0);

      logits.grad[i * vocabSize + j] = grad;
    }

    loss -= Math.log(
      Math.exp(logits.data[i * vocabSize + targets[i]] - max) / sum
    );
  }

  return loss / seqLen;
}
