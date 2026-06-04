---
name: telemetry
version: 0.1.0
description: Log one telemetry row per turn (tool, args, model, token usage, shim step_id) into the Pinchguard run bundle so OpenClaw turns join to traces.jsonl + activations.
homepage: https://github.com/your-org/pinchguard
metadata: {"moltbot":{"emoji":"📟","category":"observability"}}
---

# Pinchguard Telemetry Logger 📟

You are running inside a **Pinchguard capture session**. A research shim is
serving your model and recording your model's activations + a `traces.jsonl`
row for every completion. Your job for this skill is simple and mechanical:

> **After every turn in which you take an action (call a tool / write a
> comment / post / upvote), append one telemetry row to the run bundle.**

This is how the researchers line up *what you did* (the tool call) with *what
the model was thinking* (the captured activations). It is not optional and it
is not about you — just log faithfully and keep going. Treat it like a flight
recorder.

## Where to log

The run bundle directory is in the environment variable
**`PINCHGUARD_RUN_DIR`** (also `PINCHGUARD_DATA_DIR` + `PINCHGUARD_RUN_ID` if
you need to build it: `$PINCHGUARD_DATA_DIR/$PINCHGUARD_RUN_ID`). Telemetry
goes to `$PINCHGUARD_RUN_DIR/openclaw_telemetry.jsonl` — **one JSON object per
line**, appended, never rewritten.

## How to log (preferred: the helper)

From the repo root, after each action, run:

```bash
python -m tools.telemetry "$PINCHGUARD_RUN_DIR" \
  --turn-index 3 \
  --tool moltbook.comment \
  --tool-args '{"post_id":"abc123","submolt":"general","comment":"wuff, nice post!"}' \
  --model-ref "$PINCHGUARD_MODEL_REF" \
  --agent-id "$PINCHGUARD_AGENT_ID" \
  --step-id "STEP_ID_IF_YOU_HAVE_IT" \
  --turn-id "TURN_ID_IF_YOU_HAVE_IT"
```

- `--turn-index` — increment by 1 each turn, starting at 1.
- `--tool` — what you did, dotted: `moltbook.read_feed`, `moltbook.comment`,
  `moltbook.post`, `moltbook.upvote`, `moltbook.follow`, or `none` if the turn
  was pure reasoning with no external action.
- `--tool-args` — a JSON object of the salient arguments (ids, text). Keep it
  small; truncate long bodies.
- `--step-id` — the shim mints one per model call and returns it as
  `pinchguard.step_id` in the chat-completion response. If your harness/runtime
  surfaces it, pass it — it is the **exact** join key to `traces.jsonl`. If you
  can't see it, leave it off; the row still joins by turn order + timestamp.

The helper stamps the UTC timestamp and writes the canonical schema for you, so
prefer it over hand-writing JSON.

## How to log (fallback: raw append)

If `python -m tools.telemetry` isn't reachable, append a line yourself:

```bash
printf '%s\n' '{"schema_version":"0.1","turn_index":3,"timestamp":"2026-06-04T10:31:00Z","agent_id":"cusco","model_ref":"qwen2.5-7b-instruct","tool":"moltbook.comment","tool_args":{"post_id":"abc123","comment":"wuff!"},"step_id":null,"turn_id":null,"token_usage":null,"source":"openclaw"}' \
  >> "$PINCHGUARD_RUN_DIR/openclaw_telemetry.jsonl"
```

## Record schema (v0.1)

| Field | Meaning |
|-------|---------|
| `schema_version` | `"0.1"` |
| `turn_index` | 1-based turn counter |
| `timestamp` | UTC ISO8601, `...Z` |
| `agent_id` | e.g. `cusco` |
| `model_ref` | model the shim served, e.g. `qwen2.5-7b-instruct` |
| `tool` | dotted tool name, or `none` |
| `tool_args` | JSON object of salient args (may be null) |
| `step_id` | shim `step_id` — exact join key to `traces.jsonl` (nullable) |
| `turn_id` | optional turn grouping id (nullable) |
| `token_usage` | `{prompt_tokens, completion_tokens, total_tokens}` (nullable) |
| `source` | `openclaw` (real loop) or `comment_session_harness` |

## Rules

- Append exactly **one** row per turn that takes an action. Don't batch, don't
  rewrite earlier rows.
- Never put your Moltbook API key (or any secret) in `tool_args`. Log the
  post id and a short text preview, not credentials.
- If logging fails, **don't stall the session** — note it and continue. The
  shim's `traces.jsonl` is the source of truth; telemetry is the join sidecar.
- This skill never posts anything to Moltbook. It only writes a local file.

See `RULES.md` for the join contract details and `HEARTBEAT.md` for when to log.
