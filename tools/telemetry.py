"""OpenClaw per-turn telemetry log for a Pinchguard run bundle.

One JSON line per agent *turn* is appended to
``<run_dir>/openclaw_telemetry.jsonl``, sitting next to the shim's
``traces.jsonl`` + ``activations/*.npz``. The log lets us join an OpenClaw
agentic turn (which tool it called, with what args, token usage) to the
activation capture the shim recorded for the model call(s) inside that turn.

**Join contract.** The primary join key is ``step_id`` — the shim mints one
per ``/v1/chat/completions`` call and returns it in the response
``pinchguard.step_id`` (and writes it to every traces.jsonl row). A caller that
sees the shim response (e.g. the ``comment_session.py`` harness) should pass it
here so the join to traces/activations is exact. Callers that *don't* see the
shim response (the real OpenClaw loop, which only sees the OpenAI-spec body) may
leave ``step_id`` null and join on ``turn_id`` and/or ``(model_ref, timestamp)``
ordering instead.

Single source of truth for the record schema: both the Python harness
(``import``) and the OpenClaw telemetry skill (``python -m tools.telemetry``
via the shell) write through this module so the two never drift.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "0.1"
TELEMETRY_FILENAME = "openclaw_telemetry.jsonl"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def telemetry_record(
    *,
    turn_index: int,
    tool: str | None = None,
    tool_args: Any = None,
    step_id: str | None = None,
    turn_id: str | None = None,
    agent_id: str | None = None,
    model_ref: str | None = None,
    token_usage: dict[str, Any] | None = None,
    source: str = "openclaw",
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Build one telemetry row. Unknown fields are explicit nulls so a consumer
    can tell "not captured" from "field absent" (mirrors traces.jsonl)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "turn_index": turn_index,
        "timestamp": timestamp or _utc_now_iso(),
        "agent_id": agent_id,
        "model_ref": model_ref,
        "tool": tool,
        "tool_args": tool_args,
        "step_id": step_id,
        "turn_id": turn_id,
        "token_usage": token_usage,
        "source": source,
    }


def append_record(run_dir: Path | str, record: dict[str, Any]) -> Path:
    """Append one telemetry row to ``<run_dir>/openclaw_telemetry.jsonl``."""
    path = Path(run_dir) / TELEMETRY_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def _parse_json_arg(raw: str | None) -> Any:
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Tolerate a bare string (e.g. a comment body) passed without quoting.
        return raw


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Append one OpenClaw telemetry row to a run bundle."
    )
    parser.add_argument("run_dir", type=Path, help="Run directory (holds traces.jsonl)")
    parser.add_argument("--turn-index", type=int, required=True)
    parser.add_argument("--tool", default=None, help="Tool name, e.g. moltbook.comment")
    parser.add_argument("--tool-args", default=None, help="Tool args as a JSON string")
    parser.add_argument("--step-id", default=None, help="Shim step_id (exact join key)")
    parser.add_argument("--turn-id", default=None)
    parser.add_argument("--agent-id", default=None)
    parser.add_argument("--model-ref", default=None)
    parser.add_argument("--token-usage", default=None, help="Token usage as a JSON string")
    parser.add_argument("--source", default="openclaw")
    args = parser.parse_args(argv)

    record = telemetry_record(
        turn_index=args.turn_index,
        tool=args.tool,
        tool_args=_parse_json_arg(args.tool_args),
        step_id=args.step_id,
        turn_id=args.turn_id,
        agent_id=args.agent_id,
        model_ref=args.model_ref,
        token_usage=_parse_json_arg(args.token_usage),
        source=args.source,
    )
    path = append_record(args.run_dir, record)
    print(f"telemetry += turn {args.turn_index} -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
