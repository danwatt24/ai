import { Tensor } from "../../Tensor";
import { Embeddings } from "./embeddings";
import { OutputHead } from "./output";
import { TransformerBlock } from "./transformerBlock";

export class MiniTransformer {
  embed: Embeddings;
  blocks: TransformerBlock[];
  out: OutputHead;

  constructor(config: Record<string, number>) {
    const { vocabSize, seqLen, dModel } = config;
    this.embed = new Embeddings(`embed`, vocabSize, seqLen, dModel);

    this.blocks = [];
    for (let i = 0; i < config.numBlocks; i++) {
      this.blocks.push(
        new TransformerBlock(`blocks[${this.blocks.length}]`, dModel)
      );
    }

    this.out = new OutputHead(`out`);
  }

  forward(tokens: number[]): Tensor {
    let x = this.embed.forward(tokens);

    for (const block of this.blocks) {
      x = block.forward(x);
    }

    return this.out.forward(x);
  }

  get parameters(): Record<string, Tensor[]> {
    const params: Record<string, Tensor[]> = {
      [this.embed.key]: [this.embed.token, this.embed.position],
      [this.out.key]: [this.out.W],
    };
    for (const block of this.blocks) {
      Object.assign(params, block.parameters);
    }

    return params;
  }
}
