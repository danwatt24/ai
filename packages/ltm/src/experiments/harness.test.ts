import { QdrantClient } from "@qdrant/js-client-rest";
import { Session, subscribe } from "..";
import type { LtmEvent } from "../events";

type Step = { input: string; toolsCalled?: string[] };

const qdrant = new QdrantClient({ url: "http://127.0.0.1:6333" });
const ltmEvents: LtmEvent[] = [];
const timeout = 120_000;

describe("memory recall experiments", () => {
  let unsub: () => void;
  beforeAll(() => {
    unsub = subscribe((e) => {
      ltmEvents.push(e);
      // if (e.log) console.log(e.log);
    });
  });

  afterAll(() => {
    unsub();
  });

  let _session: Session;
  beforeEach(async () => {
    await qdrant.deleteCollection("general").catch(() => {});
    _session = await Session.create();
  });

  test(
    "no tool needed",
    async () => {
      const steps: Step[] = [
        { input: "What kind of model are you?" },
        { input: "How many grams are in a pound?" },
      ];

      for (const step of steps) await runStep(_session, step);
    },
    timeout,
  );

  test(
    "direct topic recall",
    async () => {
      const steps: Step[] = [
        { input: "What kind of model are you?" },
        {
          input: "What did I ask earlier about your model?",
          toolsCalled: ["semantic_recall"],
        },
      ];
      for (const step of steps) await runStep(_session, step);
    },
    timeout,
  );

  test(
    "vague chronological recall",
    async () => {
      const steps: Step[] = [
        { input: "What kind of model are you?" },
        {
          input: "What was my last question?",
          toolsCalled: ["recency_recall"],
        },
      ];
      for (const step of steps) await runStep(_session, step);
    },
    timeout,
  );

  test(
    "ambiguous reference recall",
    async () => {
      const steps: Step[] = [
        { input: "What subaru would you recommend for bad weather?" },
        {
          input: "Why did you choose that one?",
          toolsCalled: ["recency_recall"],
        },
      ];
      for (const step of steps) await runStep(_session, step);
    },
    timeout,
  );
});

async function runStep(session: Session, step: Step) {
  ltmEvents.length = 0;
  const response = await session.send(step.input);
  console.log(
    JSON.stringify(
      {
        input: step.input,
        events: ltmEvents.map(({ log: raw, ...others }) => others),
        response,
      },
      null,
      2,
    ),
  );

  await session.waitForIdle();
  const toolsCalled = ltmEvents
    .filter((e) => e.type === "llm.tool_requested")
    .map((e) => e.tool?.name);
  expect(toolsCalled).toEqual(step.toolsCalled ?? []);
}
