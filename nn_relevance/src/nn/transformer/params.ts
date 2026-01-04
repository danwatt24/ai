// import { MiniTransformer } from ".";
// import { SubsystemId } from "../../RelevanceEngine";
// import { Tensor } from "../../Tensor";

// export function getParams(
//   model: MiniTransformer
// ): Record<SubsystemId, Tensor[]> {
//   return {
//     [SubsystemId.embed]: [model.embed.token, model.embed.position],
//     [SubsystemId.attn]: [
//       model.attn.Wq,
//       model.attn.Wk,
//       model.attn.Wv,
//       model.attn.Wo,
//     ],
//     [SubsystemId.ffn]: [model.ffn.W1, model.ffn.b1, model.ffn.W2, model.ffn.b2],
//     [SubsystemId.ln1]: [model.ln1.gamma, model.ln1.beta],
//     [SubsystemId.ln2]: [model.ln2.gamma, model.ln2.beta],
//     [SubsystemId.output]: [model.out.W],
//   };
// }
