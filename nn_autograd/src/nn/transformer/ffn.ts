import { addBias } from "../../autograd/ops/addBias";
import { gelu } from "../../autograd/ops/gelu";
import { matmul } from "../../autograd/ops/matmul";
import { Tensor } from "../../Tensor";
import { config } from "./config";

const dFF = config.dModel * 4;

export class FeedForward {
  W1 = Tensor.randn([config.dModel, dFF]);
  b1 = Tensor.randn([dFF]);
  W2 = Tensor.randn([dFF, config.dModel]);
  b2 = Tensor.randn([config.dModel]);

  forward(x: Tensor): Tensor {
    // x: [seqLen, dModel]
    const h = matmul("FFN 0", x, this.W1); // matmul op
    const hBias = addBias(h, this.b1); // addBias op
    const hAct = gelu(hBias); // gelu op

    const out = matmul("FFN 1", hAct, this.W2); // matmul op
    const outBias = addBias(out, this.b2); // addBias op
    return outBias;
  }
}
