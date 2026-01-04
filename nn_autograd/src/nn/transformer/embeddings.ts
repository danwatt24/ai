import { embed } from "../../autograd/ops/embeddings";
import { Tensor } from "../../Tensor";

export class Embeddings {
  token: Tensor;
  position: Tensor;

  constructor(vocabSize: number, maxSeqLen: number, dModel: number) {
    this.token = Tensor.randn([vocabSize, dModel]);
    this.position = Tensor.randn([maxSeqLen, dModel]);
  }

  forward(tokens: number[]): Tensor {
    return embed(tokens, this.token, this.position);
  }
}
