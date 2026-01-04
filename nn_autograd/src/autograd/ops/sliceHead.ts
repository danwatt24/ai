import { AutogradEngine } from "../Engine";
import { Op } from "../Op";
import { Tensor } from "../../Tensor";

class SliceHeadOp implements Op {
  inputs: [Tensor];
  output: Tensor;
  name: string;

  private head: number;
  private headDim: number;
  private seqLen: number;
  private dModel: number;
  private offset: number;

  constructor(x: Tensor, head: number, headDim: number) {
    this.name = "sliceHead";
    this.inputs = [x];
    this.head = head;
    this.headDim = headDim;

    const [seqLen, dModel] = x.shape;
    this.seqLen = seqLen;
    this.dModel = dModel;
    this.offset = head * headDim;

    this.output = new Tensor([seqLen, headDim]);

    // forward (gather)
    for (let i = 0; i < seqLen; i++) {
      for (let j = 0; j < headDim; j++) {
        this.output.data[i * headDim + j] =
          x.data[i * dModel + this.offset + j];
      }
    }
  }
  backward(): void {
    const [x] = this.inputs;
    const gradOut = this.output.grad!;
    if (!x.grad) x.grad = new Float32Array(x.data.length);

    // backward scatter
    for (let i = 0; i < this.seqLen; i++) {
      for (let j = 0; j < this.headDim; j++) {
        x.grad[i * this.dModel + this.offset + j] +=
          gradOut[i * this.headDim + j];
      }
    }
  }
}

export function sliceHead(x: Tensor, head: number, headDim: number): Tensor {
  const op = new SliceHeadOp(x, head, headDim);
  AutogradEngine.record(op);
  return op.output;
}
