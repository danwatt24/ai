import { AutogradEngine } from "../Engine";
import { Op } from "../Op";
import { Tensor } from "../../Tensor";

class TransposeOp implements Op {
  inputs: [Tensor];
  output: Tensor;

  private m: number;
  private n: number;

  constructor(x: Tensor) {
    this.inputs = [x];
    const [m, n] = x.shape;
    this.m = m;
    this.n = n;

    this.output = new Tensor([n, m]);

    // forward
    for (let i = 0; i < m; i++) {
      for (let j = 0; j < n; j++) {
        this.output.data[j * m + i] = x.data[i * n + j];
      }
    }
  }
  backward(): void {
    const [x] = this.inputs;
    const gradOut = this.output.grad!;
    if (!x.grad) x.grad = new Float32Array(x.data.length);

    // backward = transpose again
    for (let i = 0; i < this.m; i++) {
      for (let j = 0; j < this.n; j++) {
        x.grad[i * this.n + j] += gradOut[j * this.m + i];
      }
    }
  }
}

export function transpose(x: Tensor): Tensor {
  const op = new TransposeOp(x);
  AutogradEngine.record(op);
  return op.output;
}
