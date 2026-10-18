# Codex LTM Field Notes

This file is a repo-local memory aid for the Codex LTM integration. It records
the current direction, important boundaries, and unresolved questions. It is
not release notes or a rigid roadmap.

## Purpose

- Add persistent, searchable memory to an existing agent instead of building
  and maintaining another agent, model loop, or chat UI first.
- Use Codex hooks and session records to backfill existing conversations and
  automatically capture new interactions.
- Keep the memory layer usable across hosted and local models. Qwen was a useful
  first local model, not an architectural commitment.
- Preserve every raw interaction while keeping recall selective, traceable, and
  optional for the model.
- Incubate the integration in this monorepo while keeping its responsibility
  recognizable enough that it could later become a standalone repository,
  package, or Codex plugin.

## Current Direction

- Codex supplies the agent loop, UI, tools, approvals, and conversation
  lifecycle. This package should integrate with those facilities rather than
  recreate them.
- User-level configuration under `~/.codex` should invoke versioned code from
  this package so memory can operate across repositories.
- Existing Codex JSONL sessions provide both historical interaction data and
  useful telemetry such as timestamps, thread IDs, turn IDs, response IDs,
  token usage, cache usage, and reasoning-token counts.
- The current long-running Codex conversation is a valuable initial dataset. It
  naturally contains compression, corrections, terminology drift, model
  changes, topic switching, and forgotten decisions.
- The same memory archive should remain available when the underlying model is
  changed. Differences in how models reconstruct the same history are part of
  the experiment rather than data loss.

## Recall Model

- Implicit recall should run before inference because the main model may believe
  it has enough context when relevant history is actually missing.
- Explicit recall should remain available for intentional, deeper, or corrective
  searches.
- Recall is not expected to find every relevant interaction. Imperfect and
  selective recall allows conversations to develop rather than reconstructing
  all history on every turn.
- Retrieval may eventually seek direct, contradictory, and lateral or
  orthogonal memories. Empty retrieval lanes are valid when no such history
  exists.
- Record which memory IDs were supplied to each turn. Supplying a memory does
  not prove the model used it, so provenance should distinguish retrieval from
  influence.

## Topic Boundaries

- A topic boundary is a semantic event; a context reset is a resource-management
  decision. They should not be treated as the same operation automatically.
- A soft boundary can tell the model that the current prompt starts a new topic
  and that it should not force relationships to earlier discussion unless the
  connection is explicit or strongly relevant.
- A hard reset or truncation should be driven separately by token pressure,
  latency, cache state, explicit user intent, or accumulated irrelevant context.
- A local lightweight model may eventually classify transitions as `continue`,
  `branch`, `return`, `unrelated`, or `uncertain`, with confidence and topic IDs.
- Topic detection should begin as annotation and telemetry. It should not be
  trusted to destroy or evict context until its behavior has been observed.
- Codex hooks can inject context and record a detected transition, but they do
  not currently rewrite or truncate the existing conversation window.

## Package Boundary

- This package owns Codex-specific event ingestion, backfilling, hook behavior,
  and translation between Codex lifecycle data and long-term memory concepts.
- The older `@repo/ltm` package is useful reference material, but its agent loop,
  model client, context window, and UI-oriented orchestration are not assumed to
  belong here.
- Isolation is conceptual rather than hermetic. This monorepo is intentionally a
  sandbox, so shared packages, root-managed dependencies, and adapted snippets
  are acceptable when they genuinely help.
- Avoid hidden coupling that an eventual extraction would not reveal through a
  normal build: assumptions about the current working directory, hardcoded
  checkout paths, writes into another package's storage, required startup order,
  or imports with unrelated side effects.
- Missing npm dependencies and shared imports can be rediscovered mechanically
  during extraction. Prematurely duplicating useful code solely for hypothetical
  independence is not a goal.

## Hook Placement

- Active hooks are expected to be user-level because the memory system should
  follow the user across repositories.
- The implementation belongs in source control here; `~/.codex` should contain
  only the configuration or minimal bridge needed to invoke it.
- Do not enable equivalent user-level and repo-level hooks simultaneously.
  Codex combines matching hook sources, which could capture each event twice.
- Hook processes run with the active session's working directory, so this
  package must resolve its own code and assets without assuming the session is
  running from this repository.

## Data Considerations

- Preserve raw source interactions even if normalized records and derived
  indexes are also created.
- Treat Codex JSONL as an external ingestion format rather than a stable internal
  domain model. Parsing and backfill logic should tolerate format evolution.
- Stable source identifiers should make backfills idempotent and prevent live
  hooks from duplicating records already imported from session history.
- SQLite remains a strong candidate for canonical local history and chronology.
  Semantic and graph stores remain projections for retrieval and relationships,
  not replacements for original content.
- Token and cache records can be correlated with turn and model metadata to
  measure the actual cost and latency effects of context growth, recall,
  compaction, model switches, and topic boundaries.

## Near-Term Questions

- Which Codex events are the minimum needed to capture complete prompt/response
  turns without duplicates?
- How should existing JSONL sessions be discovered, versioned, and backfilled?
- What schema preserves raw events while exposing normalized threads, turns,
  messages, model identity, and usage records?
- Where should canonical memory data live so it is user-scoped but independent
  of this checkout?
- How should implicit recall be injected through `UserPromptSubmit` without
  overwhelming the model or degrading the existing context?
- What telemetry is needed to compare soft topic boundaries, hard resets,
  compaction, and unmodified Codex behavior?
- Which local model is reliable enough for narrow background tasks such as query
  expansion and topic-boundary classification?

## Guiding Principle

Use the existing agent as the live environment and let every interaction become
evidence. Preserve history exhaustively, retrieve it selectively, measure the
effects, and formalize boundaries only after the experiments reveal where those
boundaries actually belong.
