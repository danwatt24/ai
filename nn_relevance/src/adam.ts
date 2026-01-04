import { Tensor } from "./Tensor";

export class Adam {
  lr: number;
  beta1 = 0.9;
  beta2 = 0.999;
  eps = 1e-8;
  t = 0;

  m = new Map<Tensor, Float32Array>();
  v = new Map<Tensor, Float32Array>();

  constructor(params: Tensor[], lr = 1e-3) {
    this.lr = lr;

    for (const p of params) {
      this.m.set(p, new Float32Array(p.data.length));
      this.v.set(p, new Float32Array(p.data.length));
    }
  }

  step() {
    this.t++;

    for (const [p, m] of this.m.entries()) {
      if (!p.grad) continue;

      const v = this.v.get(p)!;

      for (let i = 0; i < p.data.length; i++) {
        const g = p.grad[i];

        m[i] = this.beta1 * m[i] + (1 - this.beta1) * g;
        v[i] = this.beta2 * v[i] + (1 - this.beta2) * g * g;

        const mHat = m[i] / (1 - Math.pow(this.beta1, this.t));
        const vHat = v[i] / (1 - Math.pow(this.beta2, this.t));

        const amt = (this.lr * mHat) / (Math.sqrt(vHat) + this.eps);
        p.data[i] -= amt * p.relevance;
      }
    }
  }

  zeroGrad() {
    for (const p of this.m.keys()) {
      p.zeroGrad();
    }
  }
}
