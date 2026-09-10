import { Session, subscribe } from "..";
import type { LtmEvent } from "../events";
import { basicSemanticRecall, vagueChronologicalRecall } from "./scenarios";

const events: LtmEvent[] = [];
const unsub = subscribe((e) => {
  events.push(e);
  // if (e.log) console.log(e.log);
});

// const prompts = basicSemanticRecall.prompts;
const prompts = vagueChronologicalRecall.prompts;

const session = await Session.create();
for (const prompt of prompts) {
  events.length = 0;
  const response = await session.send(prompt);
  console.log(
    JSON.stringify(
      {
        prompt,
        response,
        events: events.map(({ log: raw, ...others }) => others),
      },
      null,
      2,
    ),
  );
}

unsub();
