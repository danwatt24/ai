import { AutogradEngine } from "../Engine";
import { Op } from "../Op";
import { Tensor } from "../../Tensor";

const EPS = 1e-5;

class LayerNormOp implements Op {
  inputs: [Tensor, Tensor, Tensor]; // x, gamma, beta
  output: Tensor;

  private seqLen: number;
  private dModel: number;
  private means: Float32Array;
  private vars: Float32Array;
  private xHat: Float32Array;

  constructor(x: Tensor, gamma: Tensor, beta: Tensor) {
    this.inputs = [x, gamma, beta];

    const [seqLen, dModel] = x.shape;
    this.seqLen = seqLen;
    this.dModel = dModel;

    this.output = new Tensor([seqLen, dModel]);
    this.means = new Float32Array(seqLen);
    this.vars = new Float32Array(seqLen);
    this.xHat = new Float32Array(x.data.length);

    for (let i = 0; i < seqLen; i++) {
      // compute mean
      let mean = 0;
      for (let j = 0; j < dModel; j++) {
        mean += x.data[i * dModel + j];
      }
      mean /= dModel;
      this.means[i] = mean;

      // variance
      let variance = 0;
      for (let j = 0; j < dModel; j++) {
        const diff = x.data[i * dModel + j] - mean;
        variance += diff * diff;
      }
      variance /= dModel;
      this.vars[i] = variance;

      const denom = Math.sqrt(variance + EPS);

      // normalize + affine
      for (let j = 0; j < dModel; j++) {
        const idx = i * dModel + j;
        const norm = (x.data[idx] - mean) / denom;
        this.xHat[idx] = norm;
        this.output.data[idx] = gamma.data[j] * norm + beta.data[j];
      }
    }
  }

  backward(): void {
    const [x, gamma, beta] = this.inputs;
    const outGrad = this.output.grad!;

    if (!x.grad) x.grad = new Float32Array(x.data.length);
    if (!gamma.grad) gamma.grad = new Float32Array(gamma.data.length);
    if (!beta.grad) beta.grad = new Float32Array(beta.data.length);

    for (let i = 0; i < this.seqLen; i++) {
      const denom = Math.sqrt(this.vars[i] + EPS);

      // step 1: dxHat
      let meanDxHat = 0;
      let meanDxHatXHat = 0;

      for (let j = 0; j < this.dModel; j++) {
        const idx = i * this.dModel + j;
        const dxhat = outGrad[idx] * gamma.data[j];
        meanDxHat += dxhat;
        meanDxHatXHat += dxhat * this.xHat[idx];

        // accumulate gamma/beta grads
        gamma.grad[j] += outGrad[idx] * this.xHat[idx];
        beta.grad[j] += outGrad[idx];
      }

      meanDxHat /= this.dModel;
      meanDxHatXHat /= this.dModel;

      // step 2: dx
      for (let j = 0; j < this.dModel; j++) {
        const idx = i * this.dModel + j;
        const dxhat = outGrad[idx] * gamma.data[j];

        x.grad[idx] +=
          (dxhat - meanDxHat - this.xHat[idx] * meanDxHatXHat) / denom;
      }
    }
  }
}

export function layerNorm(x: Tensor, gamma: Tensor, beta: Tensor): Tensor {
  const op = new LayerNormOp(x, gamma, beta);
  AutogradEngine.record(op);
  return op.output;
}
