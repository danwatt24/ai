import { embed } from "../../autograd/ops/embeddings";
import { RelevanceEngine } from "../../relevance/RelevanceEngine";
import { Tensor } from "../../Tensor";

export class Embeddings {
  token: Tensor;
  position: Tensor;

  readonly key: string;

  constructor(
    key: string,
    vocabSize: number,
    maxSeqLen: number,
    dModel: number
  ) {
    this.key = key;
    this.token = Tensor.randn([vocabSize, dModel]);
    this.position = Tensor.randn([maxSeqLen, dModel]);
  }

  forward(tokens: number[]): Tensor {
    RelevanceEngine.enter(this.key);
    return embed(tokens, this.token, this.position);
  }
}
