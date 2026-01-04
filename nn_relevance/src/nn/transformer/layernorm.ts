import { layerNorm } from "../../autograd/ops/layerNorm";
import { RelevanceEngine } from "../../relevance/RelevanceEngine";
import { Tensor } from "../../Tensor";

export class LayerNorm {
  gamma: Tensor;
  beta: Tensor;

  readonly key: string;

  constructor(key: string, dModel: number) {
    this.key = key;

    this.gamma = Tensor.randn([dModel]);
    this.gamma.data.fill(1);

    this.beta = Tensor.randn([dModel]);
    this.beta.data.fill(0);
  }

  forward(x: Tensor): Tensor {
    RelevanceEngine.enter(this.key);
    return layerNorm(x, this.gamma, this.beta);
  }
}
