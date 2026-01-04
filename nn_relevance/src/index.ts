import { MiniTransformer } from "./nn/transformer";
import { Adam } from "./adam";
import { AutogradEngine } from "./autograd/Engine";
import { crossEntropy } from "./autograd/ops/crossEntropy";
import { RelevanceEngine, SubsystemState } from "./relevance/RelevanceEngine";
import { config } from "./config";
import readline from "readline";

function waitForKey() {
  return new Promise((resolve) => {
    readline.emitKeypressEvents(process.stdin);
    process.stdin.setRawMode(true);

    process.stdin.once("keypress", (str, key) => {
      process.stdin.setRawMode(false);

      if (key.ctrl && key.name === "c") {
        process.exit();
      }

      resolve(undefined);
    });
  });
}

const model = new MiniTransformer(config);
const params = model.parameters;
const optimizer = new Adam(Object.values(params).flat(), 1e-2);

// tiny deterministic dataset
const badData = [
  [1, 2, 3, 4, 5],
  [2, 3, 4, 5, 1],
  [3, 4, 5, 1, 2],
];

const goodData = [
  [1, 2, 3, 4, 5],
  [1, 2, 3, 4, 5],
  [1, 2, 3, 4, 5],
];

let relevance: Map<string, SubsystemState> | undefined = undefined;

while (true) {
  let data = badData;
  if (relevance) {
    const values = relevance?.values().toArray();
    const sum = values.reduce((a, b) => a + b.relevance, 0);
    const avg = sum / values.length;
    if (avg <= 0.02) {
      data = goodData;
      console.log("---------switched to good data");
    } else if (avg > 0.85) {
      data = badData;
      console.log("---------switched to bad data");
    } else {
      console.log("-------------- NO DATA CHANGE");
    }
    await waitForKey();
  }

  for (let step = 0; step < 201; step++) {
    let totalLoss = 0;

    for (const seq of data) {
      const inputs = seq.slice(0, -1); // [1,2,3,4]
      const targets = seq.slice(1); // [2,3,4,5]

      AutogradEngine.clear();
      RelevanceEngine.clear();

      const logits = model.forward(inputs);
      const loss = crossEntropy(logits, targets);
      totalLoss += loss.data[0];

      AutogradEngine.backward(loss);
      relevance = RelevanceEngine.update(params, logits);

      // update
      optimizer.step();
      optimizer.zeroGrad();
    }

    if (step % 10 === 0) {
      console.log(`step ${step} | loss = ${totalLoss.toFixed(4)}`);
    }
  }
}
