import { AutogradEngine } from "../Engine";
import { Op } from "../Op";
import { Tensor } from "../../Tensor";

class AddBiasOp implements Op {
  inputs: [Tensor, Tensor];
  output: Tensor;

  private dim: number;

  constructor(x: Tensor, b: Tensor) {
    this.inputs = [x, b];
    this.output = new Tensor(x.shape);

    this.dim = b.shape[0];
    for (let i = 0; i < x.data.length; i++) {
      this.output.data[i] = x.data[i] + b.data[i % this.dim];
    }
  }

  backward(): void {
    const [x, b] = this.inputs;
    const outGrad = this.output.grad!;

    if (!x.grad) {
      x.grad = new Float32Array(x.data.length);
    }
    if (!b.grad) {
      b.grad = new Float32Array(b.data.length);
    }

    for (let i = 0; i < outGrad.length; i++) {
      const g = outGrad[i];
      x.grad[i] += g;
      b.grad[i % this.dim] += g;
    }
  }
}

export function addBias(x: Tensor, b: Tensor): Tensor {
  const op = new AddBiasOp(x, b);
  AutogradEngine.record(op);
  return op.output;
}
