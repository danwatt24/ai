import { AutogradEngine } from "../Engine";
import { Op } from "../Op";
import { Tensor } from "../../Tensor";

class AddOp implements Op {
  inputs: [Tensor, Tensor];
  output: Tensor;

  constructor(a: Tensor, b: Tensor) {
    this.inputs = [a, b];
    this.output = new Tensor(a.shape);

    const [seq, dModel] = a.shape;

    for (let i = 0; i < seq; i++) {
      for (let j = 0; j < dModel; j++) {
        const idx = i * dModel + j;
        this.output.data[idx] = a.data[idx] + b.data[j];
      }
    }
  }

  backward(): void {
    const [a, b] = this.inputs;
    const grad = this.output.grad!;

    if (!a.grad) a.grad = new Float32Array(a.data.length);
    const [seq, dModel] = a.shape;
    for (let i = 0; i < seq; i++) {
      for (let j = 0; j < dModel; j++) {
        const idx = i * dModel + j;
        a.grad[idx] += grad[idx];
      }
    }

    if (!b.grad) b.grad = new Float32Array(b.data.length);
    for (let j = 0; j < dModel; j++) {
      let sum = 0;
      for (let i = 0; i < seq; i++) {
        const idx = i * dModel + j;
        sum += grad[idx];
      }
      b.grad[j] += sum;
    }
  }
}

export function add(a: Tensor, b: Tensor): Tensor {
  const op = new AddOp(a, b);
  AutogradEngine.record(op);
  return op.output;
}
