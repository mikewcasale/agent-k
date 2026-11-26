# Event and SSE contract

Mission events are persisted before they are exposed to the browser. Worker-local sequences are
contiguous and idempotent. The store assigns a mission-global stream sequence used as the SSE
`id`. Reconnecting clients send `Last-Event-ID`; replay begins strictly after that cursor.

Each browser frame remains compatible with the existing frontend:

```text
id: 42
data: {"type":"phase-start","timestamp":"...","data":{"phase":"research"}}
```

No custom SSE `event:` field is used, so standard `EventSource.onmessage` continues to work.

## Stable event names

- State: `state-snapshot`, `state-delta`
- Phase: `phase-start`, `phase-complete`, `phase-error`
- Task: `task-start`, `task-progress`, `task-complete`, `task-error`
- Tool: `tool-start`, `tool-thinking`, `tool-result`, `tool-error`
- Evolution: `generation-start`, `generation-complete`, `fitness-update`,
  `submission-result`, `convergence-detected`
- Memory: `memory-store`, `memory-retrieve`, `checkpoint-created`
- Recovery: `error-occurred`, `recovery-attempt`, `recovery-complete`
- Terminal: `mission-complete`

Generation events retain `generation` and add `iteration` and `totalIterations`. Relevant model and
evaluation events may add `modelFamily` (`lightgbm`, `kumo`, `vision`, or `text`) and `modelSize`
(`small`, `medium`, or `large`). `mission-complete` retains `evolutionGenerations` and adds
`evolutionIterations` and `evolutionFailureSummary`.

The terminal event is part of the job commit invariant. A successful result cannot be committed
unless `finalEventSequence` points to its contiguous successful `mission-complete` event.
