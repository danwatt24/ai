import { AutogradEngine } from "../Engine";
import { Op } from "../Op";
import { Tensor } from "../../Tensor";

class MatMulOp implements Op {
  inputs: [Tensor, Tensor];
  output: Tensor;
  name: string;
  parent: string;

  private m: number;
  private n: number;
  private p: number;

  constructor(parent: string, a: Tensor, b: Tensor) {
    this.parent = parent;
    this.name = "matmul";
    // a: [m, n]
    // b: [n, p]
    this.inputs = [a, b];

    const [m, n] = a.shape;
    const [, p] = b.shape;

    this.m = m;
    this.n = n;
    this.p = p;

    this.output = new Tensor([m, p]);

    for (let i = 0; i < m; i++) {
      for (let j = 0; j < p; j++) {
        let sum = 0;
        for (let k = 0; k < n; k++) {
          sum += a.data[i * n + k] * b.data[k * p + j];
        }
        this.output.data[i * p + j] = sum;
      }
    }
  }

  backward(): void {
    const [a, b] = this.inputs;
    const gradOut = this.output.grad!;
    // console.log("-----------", gradOut);
    const { m, n, p } = this;

    // init grads
    if (!a.grad) a.grad = new Float32Array(a.data.length);
    if (!b.grad) b.grad = new Float32Array(b.data.length);

    // dL/dA = grad . B^T
    for (let i = 0; i < m; i++) {
      for (let k = 0; k < n; k++) {
        let sum = 0;
        for (let j = 0; j < p; j++) {
          sum += gradOut[i * p + j] * b.data[k * p + j];
        }
        a.grad[i * n + k] += sum;
      }
    }

    // dL/dB = A^T . grad
    for (let k = 0; k < n; k++) {
      for (let j = 0; j < p; j++) {
        let sum = 0;
        for (let i = 0; i < m; i++) {
          sum += a.data[i * n + k] * gradOut[i * p + j];
        }
        b.grad[k * p + j] += sum;
      }
    }
  }
}

export function matmul(parent: string, a: Tensor, b: Tensor): Tensor {
  const op = new MatMulOp(parent, a, b);
  AutogradEngine.record(op);
  return op.output;
}
