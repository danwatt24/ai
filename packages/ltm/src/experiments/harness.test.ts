import { QdrantClient } from "@qdrant/js-client-rest";
import { Session, subscribe } from "..";
import type { LtmEvent } from "../events";
import { basicSemanticRecall, vagueChronologicalRecall } from "./scenarios";
import { describe, expect, test, afterAll, beforeAll } from "vitest";

const scenarios = [basicSemanticRecall, vagueChronologicalRecall];

const qdrant = new QdrantClient({ url: "http://127.0.0.1:6333" });

describe("memory recall experiments", () => {
  const events: LtmEvent[] = [];

  let unsub: () => void;
  beforeAll(() => {
    unsub = subscribe((e) => {
      events.push(e);
      // if (e.log) console.log(e.log);
    });
  });

  afterAll(() => {
    unsub();
  });

  for (const scenario of scenarios) {
    let response = "";
    test(
      scenario.name,
      async () => {
        await qdrant.deleteCollection("general").catch(() => {});
        const session = await Session.create();

        for (const step of scenario.prompts) {
          events.length = 0;

          response = await session.send(step.input);
          console.log(
            JSON.stringify(
              {
                input: step.input,
                events: events.map(({ log: raw, ...others }) => others),
                response,
              },
              null,
              2,
            ),
          );

          await session.waitForIdle();
          const toolsCalled = events
            .filter((e) => e.type === "llm.tool_requested")
            .map((e) => e.tool?.name);

          expect(toolsCalled).toEqual(step.toolCalls ?? []);
        }
      },
      120_000,
    );
  }
});
