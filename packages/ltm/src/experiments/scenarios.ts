type Scenario = {
  name: string;
  prompts: {
    input: string;
    toolCalls?: string[];
  }[];
  expectation: string;
};

type Mode = "classic" | "recall";

export const basicSemanticRecall: Scenario = {
  name: "direct topic recall",
  prompts: [
    { input: "What kind of model are you?" },
    {
      input: "What did I ask earlier about your model?",
      toolCalls: ["semantic_recall"],
    },
  ],
  expectation: "Second response should recall the first question.",
};

export const vagueChronologicalRecall: Scenario = {
  name: "vague chronological recall",
  prompts: [
    { input: "What kind of model are you?" },
    { input: "What was my last question?", toolCalls: ["recency_recall"] },
  ],
  expectation:
    "The final response should identify the SQLite question, but semantic recall may fail without chronological tooling.",
};
