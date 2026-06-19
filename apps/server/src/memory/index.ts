import { LTM } from "./ltm";
import { ContextWindow } from "./dal/contextWindow";
import { LLM } from "./llm";

const ctxWindow = await ContextWindow.create("ctx.json");
const llm = new LLM();
export const Memory = new LTM(llm, ctxWindow);
