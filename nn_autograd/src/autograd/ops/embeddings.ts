import { AutogradEngine } from "../Engine";
import { Op } from "../Op";
import { Tensor } from "../../Tensor";

class EmbeddingOp implements Op {
  inputs: [Tensor, Tensor]; // tokenEmb, posEmb
  output: Tensor;
  name: string;

  private tokens: number[];
  private seqLen: number;
  private dModel: number;

  constructor(tokens: number[], tokenEmb: Tensor, posEmb: Tensor) {
    this.name = "Embedding";
    this.inputs = [tokenEmb, posEmb];
    this.tokens = tokens;

    this.seqLen = tokens.length;
    this.dModel = tokenEmb.shape[1];

    this.output = new Tensor([this.seqLen, this.dModel]);

    // forward
    for (let i = 0; i < this.seqLen; i++) {
      const tok = tokens[i];
      for (let j = 0; j < this.dModel; j++) {
        this.output.data[i * this.dModel + j] =
          tokenEmb.data[tok * this.dModel + j] +
          posEmb.data[i * this.dModel + j];
      }
    }
  }

  backward(): void {
    const [tokenEmb, posEmb] = this.inputs;
    const gradOut = this.output.grad!;

    if (!tokenEmb.grad) {
      tokenEmb.grad = new Float32Array(tokenEmb.data.length);
    }
    if (!posEmb.grad) {
      posEmb.grad = new Float32Array(posEmb.data.length);
    }

    // scatter-add gradients
    for (let i = 0; i < this.seqLen; i++) {
      const tok = this.tokens[i];
      for (let j = 0; j < this.dModel; j++) {
        const g = gradOut[i * this.dModel + j];
        tokenEmb.grad[tok * this.dModel + j] += g;
        posEmb.grad[i * this.dModel + j] += g;
      }
    }
  }
}

export function embed(
  tokens: number[],
  tokenEmb: Tensor,
  posEmb: Tensor
): Tensor {
  const op = new EmbeddingOp(tokens, tokenEmb, posEmb);
  AutogradEngine.record(op);
  return op.output;
}
