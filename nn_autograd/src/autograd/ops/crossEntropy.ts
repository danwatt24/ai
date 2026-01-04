import { AutogradEngine } from "../Engine";
import { Op } from "../Op";
import { Tensor } from "../../Tensor";

class crossEntropyOp implements Op {
  inputs: [Tensor];
  output: Tensor;
  targets: number[];
  name: string;

  constructor(logits: Tensor, targets: number[]) {
    this.name = "CE";
    this.inputs = [logits];
    this.targets = targets;

    const [seqLen, vocabSize] = logits.shape;
    this.output = new Tensor([1]);

    let loss = 0;

    for (let i = 0; i < seqLen; i++) {
      let max = -Infinity;
      for (let j = 0; j < vocabSize; j++) {
        max = Math.max(max, logits.data[i * vocabSize + j]);
      }

      let sum = 0;
      for (let j = 0; j < vocabSize; j++) {
        sum += Math.exp(logits.data[i * vocabSize + j] - max);
      }

      const idx = targets[i];
      loss -= Math.log(Math.exp(logits.data[i * vocabSize + idx] - max) / sum);
    }

    this.output.data[0] = loss / seqLen;
  }
  backward(): void {
    const logits = this.inputs[0];
    const [seqLen, vocabSize] = logits.shape;

    if (!logits.grad) {
      logits.grad = new Float32Array(logits.data.length);
    }

    for (let i = 0; i < seqLen; i++) {
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
        const grad = p - (j === this.targets[i] ? 1 : 0);
        logits.grad[i * vocabSize + j] += grad / seqLen;
      }
    }
  }
}

export function crossEntropy(logits: Tensor, targets: number[]): Tensor {
  const op = new crossEntropyOp(logits, targets);
  AutogradEngine.record(op);
  return op.output;
}
