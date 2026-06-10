"""Reconstruct per-step cumulative history from a traces.jsonl file.

Each step in the session had a growing chat history passed to the shim.
Rather than re-parsing the full prompt (which contains the entire prior
conversation), we reconstruct it from the rows themselves: for step N,
history = rows 0 … N-1, summarised as lightweight {turn, excerpt, output} dicts.

This approach is robust: it works even if the prompt text is truncated in the
JSONL, and it keeps the history payload sent to the judge model small.

Usage::

    from scorer.build_history import build_histories

    rows = [json.loads(l) for l in Path("traces.jsonl").read_text().splitlines() if l.strip()]
    histories = build_histories(rows)
    # histories[i] is the list of prior-round dicts for row i
"""

from __future__ import annotations

import re
from typing import Any

# Maximum characters from output_raw to include in each history entry.
# Keeps the judge context window manageable without losing signal.
_MAX_OUTPUT_CHARS = 400

# Maximum characters of the post body to include in each history excerpt.
_MAX_POST_CHARS = 300


def _extract_post_block(prompt: str) -> str:
    """Pull the Moltbook post text out of the full HF-chat-template prompt.

    The prompt looks like:
        <|im_start|>system ... <|im_end|>
        <|im_start|>user
        Moltbook post N of M in m/... by @author:
        TITLE: ...
        <body>
        Write your one short comment for this post.<|im_end|>
        <|im_start|>assistant
        <prior turns ...>
        <|im_start|>user
        Moltbook post N+1 ...

    We want only the LAST user block (the post shown for THIS turn).
    """
    # Find all user blocks
    user_blocks = re.split(r"<\|im_start\|>user\n", prompt)
    if len(user_blocks) < 2:
        return prompt[:_MAX_POST_CHARS]
    last_user = user_blocks[-1]
    # Strip the trailing assistant marker
    last_user = re.split(r"<\|im_end\|>", last_user)[0].strip()
    # Remove the trailing instruction line
    last_user = re.sub(r"\nWrite your one short comment for this post\.\s*$", "", last_user).strip()
    return last_user[:_MAX_POST_CHARS]


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " …[truncated]"


def build_histories(rows: list[dict[str, Any]]) -> list[list[dict[str, str]]]:
    """Return one history list per row.

    ``histories[i]`` is a list of dicts describing all rows *before* row i.
    Each dict has keys: ``turn``, ``post_excerpt``, ``output``.

    The list is ordered chronologically (turn 0 first).
    """
    # Sort by turn to be safe (they should already be in order)
    sorted_rows = sorted(rows, key=lambda r: r.get("turn", 0))

    histories: list[list[dict[str, str]]] = []
    accumulated: list[dict[str, str]] = []

    for row in sorted_rows:
        # Snapshot current accumulated history for THIS row
        histories.append(list(accumulated))

        # Build the entry for THIS row to add for future rows
        prompt = row.get("prompt", "")
        output = row.get("output_raw", "")

        entry: dict[str, str] = {
            "turn": str(row.get("turn", "?")),
            "step_id": row.get("step_id", "")[:8],  # short prefix for readability
            "post_excerpt": _truncate(_extract_post_block(prompt), _MAX_POST_CHARS),
            "output": _truncate(output, _MAX_OUTPUT_CHARS),
        }
        accumulated.append(entry)

    return histories


def format_history_section(history: list[dict[str, str]]) -> str:
    """Render the history list as a readable string for the judge prompt."""
    if not history:
        return "(No prior rounds — this is the first turn.)"
    lines: list[str] = []
    for entry in history:
        lines.append(
            f"--- Turn {entry['turn']} (step {entry['step_id']}) ---\n"
            f"POST SHOWN:\n{entry['post_excerpt']}\n\n"
            f"AGENT RESPONSE:\n{entry['output']}"
        )
    return "\n\n".join(lines)
