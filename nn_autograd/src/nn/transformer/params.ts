import { MiniTransformer } from ".";
import { Tensor } from "../../Tensor";

export function getParams(model: MiniTransformer): Tensor[] {
  return [
    model.embed.token,
    model.embed.position,

    model.attn.Wq,
    model.attn.Wk,
    model.attn.Wv,
    model.attn.Wo,

    model.ffn.W1,
    model.ffn.b1,
    model.ffn.W2,
    model.ffn.b2,

    model.ln1.gamma,
    model.ln1.beta,
    model.ln2.gamma,
    model.ln2.beta,

    model.out.W,
  ];
}
