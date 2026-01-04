import { matmul } from "../../autograd/ops/matmul";
import { Tensor } from "../../Tensor";
import { config } from "./config";

export class OutputHead {
  W = Tensor.randn([config.dModel, config.vocabSize]);

  forward(x: Tensor): Tensor {
    return matmul("out 0", x, this.W);
  }
}
