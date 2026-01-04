import { Tensor } from "../Tensor";

export interface Op {
  inputs: Tensor[];
  output: Tensor;
  backward(): void;
}
