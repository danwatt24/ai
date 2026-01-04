import { add } from "../../autograd/ops/add";
import { Tensor } from "../../Tensor";
import { config } from "./config";
import { Embeddings } from "./embeddings";
import { FeedForward } from "./ffn";
import { LayerNorm } from "./layernorm";
import { MultiHeadAttention } from "./multihead_attention";
import { OutputHead } from "./output";

export class MiniTransformer {
  embed = new Embeddings(config.vocabSize, config.seqLen, config.dModel);

  ln1 = new LayerNorm(config.dModel);
  attn = new MultiHeadAttention();

  ln2 = new LayerNorm(config.dModel);
  ffn = new FeedForward();

  out = new OutputHead();

  forward(tokens: number[]): Tensor {
    let x = this.embed.forward(tokens);

    // Pre-LN attention
    const attnOut = this.attn.forward(this.ln1.forward(x));
    x = add(x, attnOut);

    // Pre-LN FFN
    const ffnOut = this.ffn.forward(this.ln2.forward(x));
    x = add(x, ffnOut);

    return this.out.forward(x);
  }
}
