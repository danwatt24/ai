import { Tensor } from "./Tensor";

export enum SubsystemId {
  embed = "embed",
  attn = "attn",
  ffn = "ffn",
  ln1 = "ln1",
  ln2 = "ln2",
  output = "output",
}

interface SubsystemState {
  gradEMA: number;
  relevance: number;
}

export class RelevanceEngine {
  private static order: SubsystemId[] = [];
  private static state = new Map<SubsystemId, SubsystemState>();

  static enter(id: SubsystemId) {
    if (!this.order.includes(id)) {
      this.order.push(id);
    }

    if (!this.state.has(id)) {
      this.state.set(id, { gradEMA: 0, relevance: 1 });
    }
  }

  static clear() {
    this.order = [];
  }

  static update(layers: Record<SubsystemId, Tensor[]>) {
    let cap = 1.0;

    for (const id of this.order) {
      const s = this.state.get(id)!;
      const { gradNorm, weightNorm } = computeNorms(layers[id]);

      const r = gradNorm / (weightNorm + 1e-8);
      s.gradEMA = 0.95 * s.gradEMA + 0.05 * r;

      const localReadiness = sigmoid(10 * (0.9 - s.gradEMA));
      s.relevance = Math.min(localReadiness, cap);

      cap = s.relevance;
    }

    for (const [sub, tensors] of Object.entries(layers)) {
      const s = this.state.get(sub as SubsystemId)!;
      tensors.forEach((t) => (t.relevance = s.relevance));
    }
  }
}

function computeNorms(tensors: Tensor[]) {
  let gradSq = 0;
  let weightSq = 0;

  for (const t of tensors) {
    if (t.grad) {
      for (let i = 0; i < t.grad.length; i++) {
        const g = t.grad[i];
        gradSq += g * g;
      }
    }
    for (let i = 0; i < t.data.length; i++) {
      const w = t.data[i];
      weightSq += w * w;
    }
  }

  return {
    gradNorm: Math.sqrt(gradSq),
    weightNorm: Math.sqrt(weightSq),
  };
}

function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}
