import { Tensor } from "./Tensor";

export function transpose(t: Tensor): Tensor {
  const [m, n] = t.shape;
  const out = new Tensor([n, m]);

  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      out.data[j * m + i] = t.data[i * n + j];
    }
  }

  return out;
}

export function applyCausalMask(scores: Tensor) {
  const [n] = scores.shape;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      scores.data[i * n + j] = -1e9;
    }
  }
}

export function softmax(t: Tensor) {
  const [n, m] = t.shape;

  for (let i = 0; i < n; i++) {
    let max = -Infinity;
    for (let j = 0; j < m; j++) {
      max = Math.max(max, t.data[i * m + j]);
    }

    let sum = 0;
    for (let j = 0; j < m; j++) {
      t.data[i * m + j] = Math.exp(t.data[i * m + j] - max);
      sum += t.data[i * m + j];
    }

    for (let j = 0; j < m; j++) {
      t.data[i * m + j] /= sum;
    }
  }
}
