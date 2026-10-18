# Memory Recall Field Notes

This file is a repo-local memory aid for the LTM project. It is not release notes
or a task board. Keep it focused on what has been proven, what remains fragile,
and which experiments should come next.

## Current Checkpoint

- The current implementation uses a tiny active context: system prompt, current
  prompt, tool calls, and tool results. This remains a useful recall test bed,
  but is no longer assumed to be the final context policy.
- SQLite is the source of truth for remembered message chronology.
- Qdrant is a semantic lookup projection over remembered messages.
- The model can choose between `semantic_recall` and `recency_recall`.
- `semantic_recall` works for simple content-specific continuity.
- `recency_recall` naively works for "last question" style prompts.
- The live experiment harness now runs through Vitest.
- Scenario steps can assert expected tool-call sequences.
- `Session.waitForIdle()` lets experiments wait for background memory writes.

## Current Direction

- Preserve every raw interaction. Retention is exhaustive even though recall is
  intentionally selective.
- Grow an active conversation context until a topic, time, manual, or token
  boundary starts a new segment.
- Run implicit recall before inference so the main model is not solely
  responsible for recognizing that its visible context is incomplete.
- Keep explicit recall tools available when implicit recall is insufficient or
  the user asks for deeper historical reasoning.
- Use a small model with task-specific instruction sets for jobs such as recall
  query expansion and eventual topic-boundary classification.
- Treat recalled memories as optional background context rather than current
  instructions or guaranteed conversation chronology.
- Record which existing memories were retrieved and injected for each turn by
  ID, without duplicating their content.
- Recall should provide a useful sample, not reconstruct all relevant history.
  Query expansion may intentionally seek direct, contradictory, and lateral or
  orthogonal memories so retrieval does not become an echo chamber.
- The graph remains a future relationship layer for provenance, evolving ideas,
  corrections, topic structures, and cross-domain associations. SQLite remains
  the source of the original interaction text.

## Proven So Far

- The model can answer directly when no prior memory is needed.
- The model can request semantic recall when the prompt asks about a prior topic.
- The model can request recency recall when the prompt asks about recent order.
- The active-turn boundary prevents recency recall from returning the current
  user prompt as the "previous" user message.
- The parser can handle model output shaped as progress text followed by a final
  JSON tool request block.
- Background memory persistence can remain fire-and-forget in normal use while
  experiments explicitly wait for it.

## Naive Or Fragile

- Recency recall is simple chronological lookup, not true temporal reasoning.
- "Question" detection is not modeled; `role: "user"` is standing in for now.
- Semantic recall quality depends heavily on the model-generated query.
- The parser is tolerant but still protocol-based; malformed mixed prose/tool
  output can still fail.
- The experiment harness depends on a live local LLM and Qdrant stack.
- SQLite is currently in-memory, so the canonical memory store resets per run.
- Qdrant and SQLite can still drift if a process dies between writes.
- Tool-call assertions check selected tools, not final-answer quality.

## Next Experiments

Use the Vitest harness to compare behavior as the context and recall policies
evolve:

- No recall needed: simple factual prompt should not call tools.
- Semantic recall: prior content-specific question should call `semantic_recall`.
- Recency recall: "last question" should call `recency_recall`.
- Assistant recall: "what did you recommend earlier?" should retrieve assistant
  output.
- Ambiguous reference: "what do you mean by that?" should expose whether recency,
  semantic recall, or clarification is needed.
- Aside and return: topic A, brief topic B aside, then "back to what we were
  saying" should test thread/frame recovery.
- Failed recall: asking for nonexistent prior memory should not hallucinate.
- Correction memory: later correction should override or qualify an earlier
  assistant mistake.
- Preference memory: user preference should be recalled only when relevant.
- Multi-step recall: model may need more than one tool call to answer well.
- Growing context: ordinary follow-ups should not require recall while their
  active segment still contains the needed history.
- Implicit recall: a prompt whose missing context is not obvious to the main
  model should receive useful memories before inference.
- Diverse recall: direct, contradictory, and lateral retrieval should broaden
  the available context without forcing irrelevant connections into the answer.
- Topic boundary: an unrelated prompt should be able to begin a fresh active
  segment while the prior segment remains fully recallable.

## Deferred Ideas

- Topic/thread/frame recall for "back to that" style prompts.
- Model elevation: route simple prompts to cheaper models and harder prompts to
  stronger ones.
- Graph storage for provenance, associations, and derived topic structures.
- Mutable topic summaries that may become searchable memories.
- Reconciliation checks between SQLite and Qdrant.

## Guiding Principle

Store the complete interaction history, but do not equate complete storage with
complete recall. Maintain a useful active conversation segment, retrieve a
small and deliberately varied set of older memories, and keep every retrieval
and derived relationship traceable to its source interactions.
