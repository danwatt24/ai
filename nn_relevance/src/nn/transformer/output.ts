import { matmul } from "../../autograd/ops/matmul";
import { RelevanceEngine } from "../../relevance/RelevanceEngine";
import { Tensor } from "../../Tensor";
import { config } from "../../config";

export class OutputHead {
  W = Tensor.randn([config.dModel, config.vocabSize]);

  readonly key: string;

  constructor(key: string) {
    this.key = key;
  }

  forward(x: Tensor): Tensor {
    RelevanceEngine.enter(this.key);
    return matmul(x, this.W);
  }
}
