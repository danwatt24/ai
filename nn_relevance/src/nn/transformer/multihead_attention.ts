import { concatHeads } from "../../autograd/ops/concatHeads";
import { matmul } from "../../autograd/ops/matmul";
import { sliceHead } from "../../autograd/ops/sliceHead";
import { transpose } from "../../autograd/ops/transpose";
import { RelevanceEngine } from "../../relevance/RelevanceEngine";
import { Tensor } from "../../Tensor";
import { applyCausalMask, softmax } from "../../utils";
import { config } from "../../config";

const headDim = config.dModel / config.numheads;

export class MultiHeadAttention {
  Wq = Tensor.randn([config.dModel, config.dModel]);
  Wk = Tensor.randn([config.dModel, config.dModel]);
  Wv = Tensor.randn([config.dModel, config.dModel]);
  Wo = Tensor.randn([config.dModel, config.dModel]);

  readonly key: string;

  constructor(key: string) {
    this.key = key;
  }

  forward(x: Tensor): Tensor {
    RelevanceEngine.enter(this.key);
    const Q = matmul(x, this.Wq);
    const K = matmul(x, this.Wk);
    const V = matmul(x, this.Wv);

    const heads: Tensor[] = [];

    for (let h = 0; h < config.numheads; h++) {
      const qh = sliceHead(Q, h, headDim);
      const kh = sliceHead(K, h, headDim);
      const vh = sliceHead(V, h, headDim);

      const scores = matmul(qh, transpose(kh));
      applyCausalMask(scores);
      softmax(scores);

      const out = matmul(scores, vh);
      heads.push(out);
    }

    const concatenated = concatHeads(heads);
    return matmul(concatenated, this.Wo);
  }
}
