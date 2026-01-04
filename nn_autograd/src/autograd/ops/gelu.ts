import { AutogradEngine } from "../Engine";
import { Op } from "../Op";
import { Tensor } from "../../Tensor";

const SQRT_2_OVER_PI = Math.sqrt(2 / Math.PI);
const COEFF = 0.044715;

class GeluOp implements Op {
  inputs: [Tensor];
  output: Tensor;
  name: string;

  constructor(x: Tensor) {
    this.name = "gelu";
    this.inputs = [x];
    this.output = new Tensor(x.shape);

    for (let i = 0; i < x.data.length; i++) {
      const v = x.data[i];
      const u = SQRT_2_OVER_PI * (v + COEFF * v * v * v);
      this.output.data[i] = 0.5 * v * (1 + Math.tanh(u));
    }
  }

  backward(): void {
    const [x] = this.inputs;
    const outGrad = this.output.grad!;
    if (!x.grad) {
      x.grad = new Float32Array(x.data.length);
    }

    for (let i = 0; i < x.data.length; i++) {
      const v = x.data[i];
      const u = SQRT_2_OVER_PI * (v + COEFF * v * v * v);
      const tanhU = Math.tanh(u);

      const du_dx = SQRT_2_OVER_PI * (1 + 3 * COEFF * v * v);
      const dy_dx = 0.5 * (1 + tanhU) + 0.5 * v * (1 - tanhU * tanhU) * du_dx;

      x.grad[i] += outGrad[i] * dy_dx;
    }
  }
}

export function gelu(x: Tensor): Tensor {
  const op = new GeluOp(x);
  AutogradEngine.record(op);
  return op.output;
}
