import { Op } from "./Op";
import { Tensor } from "../Tensor";

export class AutogradEngine {
  private static tape: Op[] = [];

  static record(op: Op) {
    this.tape.push(op);
  }

  static backward(loss: Tensor) {
    // seed gradient
    loss.grad = new Float32Array(loss.data.length).fill(1);

    // reverse execution
    for (let i = this.tape.length - 1; i >= 0; i--) {
      this.tape[i].backward();
    }
  }

  static clear() {
    this.tape.length = 0;
  }
}
