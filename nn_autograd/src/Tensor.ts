export class Tensor {
  data: Float32Array;
  grad: Float32Array | null = null;
  shape: number[];

  constructor(shape: number[], data?: Float32Array) {
    this.shape = shape;
    this.data = data ?? new Float32Array(shape.reduce((a, b) => a * b));
  }

  zeroGrad() {
    if (this.grad) this.grad.fill(0);
  }

  static randn(shape: number[]) {
    const t = new Tensor(shape);
    for (let i = 0; i < t.data.length; i++) {
      t.data[i] = Math.random() * 0.02;
    }

    return t;
  }
}
