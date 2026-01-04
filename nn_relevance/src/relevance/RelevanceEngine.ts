import { Tensor } from "../Tensor";
import { outputConfidence } from "./marginConfidence";

export type SubsystemId = string;

export interface SubsystemState {
  gradEMA: number;
  relevance: number;
}

const MIN_RELEVANCE = 0.02;
const NEUTRAL = 0.5;
const tau = 0.4; // or even 0.02
const k = 5; // soften the transition

let count = 0;

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

  static update(layers: Record<SubsystemId, Tensor[]>, logits: Tensor) {
    let cap = 1.0;

    for (const id of this.order) {
      const s = this.state.get(id)!;
      if (id === "out") {
        const conf = outputConfidence(logits);
        s.gradEMA = 0.9 * s.gradEMA + 0.1 * conf;

        s.relevance = Math.min(s.gradEMA, cap);
        cap = s.relevance;
      } else {
        const { gradNorm, weightNorm } = computeNorms(layers[id]);
        const r = gradNorm / (weightNorm + 1e-8);
        // console.log("-------------", Math.abs(r - s.gradEMA).toFixed(4));
        if (Math.abs(r - s.gradEMA) < 0.002) {
          s.gradEMA = 0.99 * s.gradEMA + 0.01 * NEUTRAL;
        } else if (r > s.gradEMA) {
          s.gradEMA = 0.95 * s.gradEMA + 0.05 * r;
        } else {
          s.gradEMA = 0.8 * s.gradEMA + 0.2 * r;
        }

        const localReadiness = sigmoid(k * (tau - s.gradEMA));

        s.relevance = Math.max(MIN_RELEVANCE, Math.min(localReadiness, cap));
        // s.relevance = Math.min(localReadiness, cap);
        cap = s.relevance;
      }
    }

    for (const [sub, tensors] of Object.entries(layers)) {
      const s = this.state.get(sub as SubsystemId)!;
      tensors.forEach((t) => (t.relevance = s.relevance));
    }
    if (count++ >= 10) {
      const relevancy = this.order
        .map((id) => `${id}:${this.state.get(id)!.relevance.toFixed(4)}`)
        .join(" | ");

      console.log(relevancy);
      count = 0;
    }

    return this.state;
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
