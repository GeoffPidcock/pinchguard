# Telemetry Rules & Join Contract 📟

## What this skill is for

Pinchguard studies whether an agent's *behaviour* drifts as it reads peer
content, and whether that drift is visible in the model's *activations*. To do
that we must be able to line up, for each turn:

- **what the agent did** — the tool call (this telemetry log), and
- **what the model computed** — the shim's `traces.jsonl` row + the matching
  `activations/<step_id>.npz`.

This skill produces the first half. The shim produces the second half.

## The join contract

| Artifact | Written by | Key fields |
|----------|-----------|------------|
| `openclaw_telemetry.jsonl` | this skill / the harness | `turn_index`, `step_id`, `turn_id`, `timestamp`, `model_ref` |
| `traces.jsonl` | the shim | `step_id`, `turn_id`, `turn`, `timestamp`, `model_id` |
| `activations/<step_id>.npz` | the shim | filename stem == `step_id` |

**Preferred join:** `telemetry.step_id == traces.step_id`. The shim returns
`pinchguard.step_id` in every chat-completion response. Whenever the runtime
surfaces that value to you, pass it via `--step-id`; the join is then exact and
one telemetry row maps to exactly one activation file.

**Fallback joins**, when `step_id` isn't visible to the agent:

1. `telemetry.turn_id == traces.turn_id` (if a shared `turn_id` is threaded
   through the chat-completion request body), then
2. time-ordering: sort both logs by `timestamp` and align by `(model_ref,
   order)`. A turn may span several shim calls (tool-use loops); expect a
   one-telemetry-row → many-trace-rows fan-out in that case.

## Invariants the run validator checks (if telemetry is present)

1. The file is newline-delimited JSON; every line parses.
2. Every row has an integer `turn_index` and a `schema_version`.
3. `turn_index` values are unique and strictly increasing.
4. Any non-null `step_id` must match a `step_id` present in `traces.jsonl`
   (no telemetry row may reference an activation capture that never happened).

A run with **no** `openclaw_telemetry.jsonl` is still valid (e.g. a pure shim
loopback) — telemetry is an optional sidecar, validated only when present.

## Hard rules

- One row per action turn. Append-only. Never rewrite.
- Never log secrets. `tool_args` carries ids and short text previews only.
- Telemetry is best-effort: never block or abort the session on a log failure.
