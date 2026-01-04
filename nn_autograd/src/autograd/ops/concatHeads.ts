import { config } from "../../nn/transformer/config";
import { AutogradEngine } from "../Engine";
import { Op } from "../Op";
import { Tensor } from "../../Tensor";

const headDim = config.dModel / config.numheads;

class ConcatHeadsOp implements Op {
  name: string;
  inputs: Tensor[]; // heads;
  output: Tensor;

  private headDim: number;
  private numHeads: number;
  private seqLen: number;

  constructor(heads: Tensor[], headDim: number) {
    this.name = "concatHeads";
    this.inputs = heads;
    this.numHeads = heads.length;
    this.headDim = headDim;
    this.seqLen = heads[0].shape[0];

    this.output = new Tensor([this.seqLen, this.numHeads * headDim]);

    // forward
    for (let h = 0; h < this.numHeads; h++) {
      const head = heads[h];
      const offset = h * headDim;

      for (let i = 0; i < this.seqLen; i++) {
        for (let j = 0; j < headDim; j++) {
          this.output.data[i * this.numHeads * headDim + offset + j] =
            head.data[i * headDim + j];
        }
      }
    }
  }
  backward(): void {
    const gradOut = this.output.grad!;
    const stride = this.numHeads * this.headDim;

    for (let h = 0; h < this.numHeads; h++) {
      const head = this.inputs[h];
      if (!head.grad) {
        head.grad = new Float32Array(head.data.length);
      }

      const offset = h * this.headDim;

      for (let i = 0; i < this.seqLen; i++) {
        for (let j = 0; j < this.headDim; j++) {
          head.grad[i * this.headDim + j] += gradOut[i * stride + offset + j];
        }
      }
    }
  }
}

export function concatHeads(heads: Tensor[]): Tensor {
  const op = new ConcatHeadsOp(heads, headDim);
  AutogradEngine.record(op);
  return op.output;
}
