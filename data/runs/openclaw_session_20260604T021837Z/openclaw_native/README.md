# OpenClaw native telemetry (out-of-the-box)

These files are copied verbatim from
`~/.openclaw/agents/main/sessions/ecb8df20-4123-41a3-8100-1509022b08fa.*` — i.e.
what an OpenClaw operator gets **by default**, no Pinchguard instrumentation.

- `session.trajectory.jsonl` — the rich per-turn trajectory. 3 turns ×
  7 record types: `session.started`, `trace.metadata`, `context.compiled`,
  `prompt.submitted`, `model.completed`, `trace.artifacts`, `session.ended`.
- `session.jsonl` — high-level session message log.
- `session.trajectory-path.json` — OpenClaw's pointer to the trajectory file.
- `session_index_entry.json` — this session's entry from `sessions.json`.
- `input_posts.json` — the 3 offline post prompts fed to the agent (this run).

**No secrets:** scanned before copying — the real `MOLTBOOK_API_KEY` value and
the gateway auth token have 0 hits; only the literal string `MOLTBOOK_API`
(persona/skill prose) appears.

See `../note.md` for the side-by-side comparison with our shim `traces.jsonl` +
`activations/` and what it says about the validity of our data structure.
