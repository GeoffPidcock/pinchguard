"""Generate the committed schema-v0.2 example run.

Drives the inference shim via FastAPI's TestClient against a real
Qwen2.5-0.5B-Instruct on CPU. Output lands in
`data/runs/schema_v0.2_example/` and is committed to the repo (see
`.gitignore`) so collaborators on Concerns B/C/D can inspect concrete rows
and activations without rerunning the model.

Re-run from the repo root after schema changes:

    uv sync --extra dev --extra capture
    uv run python scripts/generate_example_run.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from fastapi.testclient import TestClient

from shim.capture import HFCapturer
from shim.main import create_app
from tools.validate_run import validate_run

PROMPTS = [
    "Summarise the unread items in my inbox in one paragraph.",
    "Draft a polite reply declining the refund request from yesterday's ticket.",
    "Should I post about the team launch on Moltbook today, or wait for marketing review?",
    "Schedule a 30-minute sync with the finance team for tomorrow afternoon.",
    "What's on my calendar for the rest of today?",
]

RUN_DIR = REPO_ROOT / "data" / "runs" / "schema_v0.2_example"


def main() -> int:
    print(f"==> loading Qwen/Qwen2.5-0.5B-Instruct", flush=True)
    cap = HFCapturer(
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
        layers=(12, 23),
        token_position="last_input",
        max_new_tokens=48,
        dtype="float16",
    )
    print(f"==> model has {len(cap._layer_modules)} layers; capturing {cap.layers}", flush=True)

    app = create_app(run_dir=RUN_DIR, capturer=cap)
    with TestClient(app) as client:
        for i, prompt in enumerate(PROMPTS):
            resp = client.post(
                "/v1/chat/completions",
                json={
                    "model": "qwen2.5-0.5b-instruct",
                    "messages": [
                        {"role": "system", "content": "You are a helpful executive assistant."},
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            assert resp.status_code == 200, resp.text
            body = resp.json()
            step_id = body["pinchguard"]["step_id"]
            preview = body["choices"][0]["message"]["content"].replace("\n", " ")[:80]
            print(f"  [{i}] step_id={step_id[:8]} → {preview!r}", flush=True)

    result = validate_run(RUN_DIR)
    print(result.report(), flush=True)
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
