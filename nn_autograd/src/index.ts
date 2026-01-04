import { MiniTransformer } from "./nn/transformer";
import { Adam } from "./adam";
import { getParams } from "./nn/transformer/params";
import { AutogradEngine } from "./autograd/Engine";
import { crossEntropy } from "./autograd/ops/crossEntropy";

const model = new MiniTransformer();
const params = getParams(model);
const optimizer = new Adam(params, 1e-2);

// tiny deterministic dataset
const data = [
  [1, 2, 3, 4, 5],
  [1, 2, 3, 4, 5],
  [1, 2, 3, 4, 5],
];

for (let step = 0; step < 200; step++) {
  let totalLoss = 0;

  for (const seq of data) {
    const inputs = seq.slice(0, -1); // [1,2,3,4]
    const targets = seq.slice(1); // [2,3,4,5]

    AutogradEngine.clear();

    const logits = model.forward(inputs);
    const loss = crossEntropy(logits, targets);
    totalLoss += loss.data[0];

    AutogradEngine.backward(loss);

    // update
    optimizer.step();
    optimizer.zeroGrad();
  }

  if (step % 10 === 0) {
    console.log(`step ${step} | loss = ${totalLoss.toFixed(4)}`);
  }
}
