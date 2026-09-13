# Memory Recall Field Notes

This file is a repo-local memory aid for the LTM project. It is not release notes
or a task board. Keep it focused on what has been proven, what remains fragile,
and which experiments should come next.

## Current Checkpoint

- The active model context is intentionally tiny: system prompt, current prompt,
  tool calls, and tool results.
- SQLite is the source of truth for remembered message chronology.
- Qdrant is a semantic lookup projection over remembered messages.
- The model can choose between `semantic_recall` and `recency_recall`.
- `semantic_recall` works for simple content-specific continuity.
- `recency_recall` naively works for "last question" style prompts.
- The live experiment harness now runs through Vitest.
- Scenario steps can assert expected tool-call sequences.
- `Session.waitForIdle()` lets experiments wait for background memory writes.

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

Expand the Vitest harness with small behavior scenarios before adding new
architecture:

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

## Deferred Ideas

- Preloaded recall: a cheap router or deterministic layer retrieves likely
  context before the main model runs, while tools remain available.
- Topic boundaries as metadata, not active context resets.
- Topic/thread/frame recall for "back to that" style prompts.
- Model elevation: route simple prompts to cheaper models and harder prompts to
  stronger ones.
- Graph storage for provenance, associations, and derived topic structures.
- Mutable topic summaries that may become searchable memories.
- Reconciliation checks between SQLite and Qdrant.

## Guiding Principle

Do not maintain large context by default. Reconstruct the smallest useful context
for the current prompt, and keep the reconstruction inspectable.
