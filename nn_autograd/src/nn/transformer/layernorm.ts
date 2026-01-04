import { layerNorm } from "../../autograd/ops/layerNorm";
import { Tensor } from "../../Tensor";

export class LayerNorm {
  gamma: Tensor;
  beta: Tensor;

  constructor(dModel: number) {
    this.gamma = Tensor.randn([dModel]);
    this.gamma.data.fill(1);

    this.beta = Tensor.randn([dModel]);
    this.beta.data.fill(0);
  }

  forward(x: Tensor): Tensor {
    return layerNorm(x, this.gamma, this.beta);
  }
}
