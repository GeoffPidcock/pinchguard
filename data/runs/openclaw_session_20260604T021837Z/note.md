# openclaw_session_20260604T021837Z — Cusco via the **real OpenClaw loop** (3 turns)

**Why this run exists.** Every prior scenario-01 capture drove the model through
our deterministic `comment_session.py` harness. This run instead drives the
**genuine `openclaw agent --local` loop** against the same capture shim, so we
can put OpenClaw's **out-of-the-box operator telemetry** side-by-side with our
instrumentation and judge whether our data structure is valid / sufficient.

**Model:** `Qwen/Qwen2.5-7B-Instruct`, GPU, 8-bit, layers [12,24]
(`capture_runtime=hf-eager+8bit`), same as `comment_session_20260604T012547Z`.
**Turns:** 3 (the first 3 posts from that run's `feed_snapshot.json`, pasted as
offline input — see `openclaw_native/input_posts.json`).
**Runner:** `openclaw agent --local`, model ref `qwen2.5-7b-instruct` via the
`custom-127-0-0-1-8000` provider (the shim). `executionTrace.runner=embedded`.

> **DRY-RUN / OFFLINE & SAFE.** The agent ran with **no `MOLTBOOK_API_KEY` in its
> shell** and an offline task (post text pasted in the prompt). It made **zero**
> Moltbook calls — the native trajectory contains no tool-invocation records, only
> model completions. Nothing was posted. Secret scan: the real Moltbook key and
> the gateway token have **0 hits** in the copied native files (the only matches
> are the literal string `MOLTBOOK_API` from the persona/skill prose).

## What's in this bundle

| Path | Producer | What it is |
|------|----------|-----------|
| `traces.jsonl` | **our shim** | 3 rows, schema v0.2 (one per model completion) |
| `activations/*.npz` | **our shim** | 3 files, L12/L24 residual-stream vectors (1×3584 fp16) |
| `openclaw_native/session.trajectory.jsonl` | **OpenClaw (OOB)** | 21 records = 3 turns × 7 record types |
| `openclaw_native/session.jsonl` | **OpenClaw (OOB)** | session message log (session/model_change/messages) |
| `openclaw_native/session_index_entry.json` | OpenClaw | this session's `sessions.json` index entry |
| `openclaw_native/input_posts.json` | this run | the 3 offline post prompts |

Note there is **no `openclaw_telemetry.jsonl`** here: that sidecar is written by
our telemetry skill, and the 7B agent did **not** invoke the skill's
`python -m tools.telemetry` command on its own (see "Findings" #3). The bundle is
still validator-green — the sidecar is optional.

## The two data structures, compared

**Native OpenClaw trajectory** (`session.trajectory.jsonl`) — 7 record types per
turn: `session.started`, `trace.metadata`, `context.compiled`, `prompt.submitted`,
`model.completed`, `trace.artifacts`, `session.ended`. Each carries
`traceSchema`, `schemaVersion`, `traceId`, `sessionId`, `sessionKey`, `runId`,
`seq`/`sourceSeq`, `ts`, `workspaceDir`. It is **rich on orchestration**: the
compiled context, tool schemas (with hashes), `requestShaping`, `executionTrace`
(winner provider/model, fallbacks), prompt + visible/raw assistant text.

**Our shim trace** (`traces.jsonl`) — one flat row per completion:
`step_id`, `turn_id`, `scenario_id`, `run_id`, `turn`, `timestamp`, `agent_id`,
`agent_persona`, `alignment_label`, `model_id`, `context`, `prompt`,
`output_raw`, `parsed_action`, `activation_ref`, `activation_meta`.

### Findings (does our structure hold up?)

1. **The activations are the thing OpenClaw cannot give you.** The native
   trajectory has *zero* model-internal state — it stops at `model.completed`
   (the text). Our `activations/*.npz` + `activation_meta` (layer indices,
   token_position, capture_runtime) are exactly the Concern-C signal an operator
   does **not** get out of the box. This is the core justification for the shim.

2. **The join key is the weak point.** Inside our bundle the `step_id ↔ npz`
   1:1 invariant holds (validator green). But the shim mints `step_id` and
   returns it as `pinchguard.step_id` in the chat-completion response — and
   **OpenClaw discards it.** It never reaches the native trajectory, and we can't
   set our `turn_id` from outside because OpenClaw owns the request body
   (`turn_id` is `null` in all three rows here). So to line up a native turn with
   our activation you must currently join on **prompt text / hash, timestamp, or
   call order** — workable here (3 clean 1:1 turns) but fragile once a turn fans
   out into multiple tool-loop completions. *Recommended fix:* have the shim echo
   `step_id` in a response **header** and/or stamp it into `model_id`/a trailer
   that OpenClaw's `model.completed` record preserves; or thread a stable
   per-turn id via the provider request. Either makes the OpenClaw↔shim join as
   exact as the harness's `step_id` join already is.

3. **Skill-based per-turn telemetry is not free.** Our `openclaw_telemetry.jsonl`
   sidecar (Task 2) was written deterministically by the harness in
   `comment_session_20260604T012547Z`, but here — driving the *real* agent — the
   7B simply didn't run the telemetry skill's logging command, so no sidecar was
   produced. Native trajectory logging is automatic and reliable; **agent
   self-logging is not.** If the per-turn tool/args sidecar matters, it should be
   captured by a runtime hook (a gateway/provider middleware), not by asking the
   model to log itself.

4. **One operator turn == one completion here, but won't always.** With no tools
   invoked, each `openclaw agent` turn produced exactly one shim completion
   (turn 0/1/2). A turn that actually used the Moltbook skill would emit several
   completions inside one operator turn — our flat per-completion rows handle
   that (N rows), but the native `trace.*`/`session.*` grouping is what reconnects
   them into one operator-visible turn. Our schema currently lacks that grouping
   except via the unused `turn_id`; populating `turn_id` from the native
   `traceId`/`sessionKey` would close the gap.

## Reproduce

1. Boot shim: `MODEL_NAME=Qwen/Qwen2.5-7B-Instruct PINCHGUARD_LAYERS=12,24
   PINCHGUARD_DTYPE=float16 PINCHGUARD_QUANTIZE=8bit CUDA_VISIBLE_DEVICES=0
   PINCHGUARD_RUN_ID=<id> .venv/bin/python -m uvicorn shim.main:create_app
   --factory --port 8000`.
2. Point OpenClaw at it (provider `custom-127-0-0-1-8000`, model
   `qwen2.5-7b-instruct`, primary + allowlisted).
3. Per post: `env -u MOLTBOOK_API_KEY openclaw agent --local
   --session-key agent:main:pinchguard-oc --message "<post>" --json`.
4. Copy `~/.openclaw/agents/main/sessions/<session>.trajectory.jsonl` (+ session
   log) into `<run_dir>/openclaw_native/`.
