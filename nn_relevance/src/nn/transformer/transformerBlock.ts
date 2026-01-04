import { add } from "../../autograd/ops/add";
import { Tensor } from "../../Tensor";
import { FeedForward } from "./ffn";
import { LayerNorm } from "./layernorm";
import { MultiHeadAttention } from "./multihead_attention";

export class TransformerBlock {
  ln1: LayerNorm;
  attn: MultiHeadAttention;

  ln2: LayerNorm;
  ffn: FeedForward;

  constructor(path: string, dModel: number) {
    this.ln1 = new LayerNorm(`${path}.ln1`, dModel);
    this.attn = new MultiHeadAttention(`${path}.attn`);

    this.ln2 = new LayerNorm(`${path}.ln2`, dModel);
    this.ffn = new FeedForward(`${path}.ffn`);
  }

  forward(x: Tensor): Tensor {
    // Pre-LN attention
    const attnOut = this.attn.forward(this.ln1.forward(x));
    x = add(x, attnOut);

    // Pre-LN FFN
    const ffnOut = this.ffn.forward(this.ln2.forward(x));
    x = add(x, ffnOut);

    return x;
  }

  get parameters(): Record<string, Tensor[]> {
    return {
      [this.attn.key]: [this.attn.Wq, this.attn.Wk, this.attn.Wv, this.attn.Wo],
      [this.ffn.key]: [this.ffn.W1, this.ffn.b1, this.ffn.W2, this.ffn.b2],
      [this.ln1.key]: [this.ln1.gamma, this.ln1.beta],
      [this.ln2.key]: [this.ln2.gamma, this.ln2.beta],
    };
  }
}
