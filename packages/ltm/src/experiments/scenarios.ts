type Scenario = {
  name: string;
  prompts: string[];
  expectation: string;
};

type Mode = "classic" | "recall";

export const basicSemanticRecall: Scenario = {
  name: "direct topic recall",
  prompts: [
    "What kind of model are you?",
    "What did I ask earlier about your model?",
  ],
  expectation: "Second response should recall the first question.",
};
