import { concatHeads } from "../../autograd/ops/concatHeads";
import { matmul } from "../../autograd/ops/matmul";
import { sliceHead } from "../../autograd/ops/sliceHead";
import { transpose } from "../../autograd/ops/transpose";
import { Tensor } from "../../Tensor";
import { applyCausalMask, softmax } from "../../utils";
import { config } from "./config";

const headDim = config.dModel / config.numheads;

export class MultiHeadAttention {
  Wq = Tensor.randn([config.dModel, config.dModel]);
  Wk = Tensor.randn([config.dModel, config.dModel]);
  Wv = Tensor.randn([config.dModel, config.dModel]);
  Wo = Tensor.randn([config.dModel, config.dModel]);

  forward(x: Tensor): Tensor {
    const Q = matmul("attn 0", x, this.Wq);
    const K = matmul("attn 1", x, this.Wk);
    const V = matmul("attn 2", x, this.Wv);

    const heads: Tensor[] = [];

    for (let h = 0; h < config.numheads; h++) {
      const qh = sliceHead(Q, h, headDim);
      const kh = sliceHead(K, h, headDim);
      const vh = sliceHead(V, h, headDim);

      const scores = matmul("attn 3", qh, transpose(kh));
      applyCausalMask(scores);
      softmax(scores);

      const out = matmul("attn 4", scores, vh);
      heads.push(out);
    }

    const concatenated = concatHeads(heads);
    return matmul("attn 5", concatenated, this.Wo);
  }
}
