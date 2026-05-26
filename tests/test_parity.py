"""Phase 1 parity tests — drive the shim with N requests, assert N rows and
N npz files, every row's activation_ref resolves, validator clean.

Uses FastAPI's TestClient + `MockCapturer` so CI runs without GPU or weights.
The real-weights smoke (50 curl-driven completions on Qwen2.5-0.5B) lives in
`scripts/smoke_shim.sh` and is run manually on Codespace.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from shim.capture import MockCapturer  # noqa: E402
from shim.main import create_app  # noqa: E402
from tools.validate_run import validate_run  # noqa: E402


def _client(run_dir: Path) -> TestClient:
    app = create_app(run_dir=run_dir, capturer=MockCapturer())
    return TestClient(app)


def _read_rows(run_dir: Path) -> list[dict]:
    text = (run_dir / "traces.jsonl").read_text().splitlines()
    return [json.loads(line) for line in text if line.strip()]


def test_n_requests_yield_n_rows_n_npz_validator_clean(tmp_path: Path) -> None:
    n = 5
    run = tmp_path / "run"
    client = _client(run)

    for i in range(n):
        resp = client.post(
            "/v1/chat/completions",
            json={
                "model": "mock-test",
                "messages": [{"role": "user", "content": f"hello {i}"}],
            },
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["choices"][0]["message"]["role"] == "assistant"
        assert body["pinchguard"]["step_id"]
        assert body["pinchguard"]["activation_ref"].startswith("activations/")

    rows = _read_rows(run)
    npzs = sorted((run / "activations").glob("*.npz"))
    assert len(rows) == n
    assert len(npzs) == n

    # Every row's activation_ref must resolve to an existing npz with matching stem.
    for row in rows:
        ref_path = run / row["activation_ref"]
        assert ref_path.exists(), f"missing {ref_path}"
        assert ref_path.stem == row["step_id"]

    result = validate_run(run)
    assert result.ok, result.report()
    assert result.rows_checked == n


def test_row_carries_required_phase1_fields(tmp_path: Path) -> None:
    run = tmp_path / "run"
    client = _client(run)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "mock-test", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200

    row = _read_rows(run)[0]
    # Phase 1 hard requirements from TODO.md.
    for field in ("schema_version", "step_id", "activation_ref", "activation_meta"):
        assert row.get(field) is not None, f"missing {field}"
    assert row["schema_version"] == "0.2"

    meta = row["activation_meta"]
    for key in ("token_position", "layers", "dtype", "capture_runtime"):
        assert key in meta, f"activation_meta missing {key}"
    assert meta["token_position"] == "last_input"
    assert meta["dtype"] == "float16"
    assert isinstance(meta["layers"], list) and meta["layers"]

    # Bookkeeping fields.
    assert row["turn"] == 0
    # model_id reflects the capturer's actual model, not the client's request.model.
    assert row["model_id"] == "mock-qwen2.5-0.5b"
    assert row["run_id"] == run.name
    assert row["prompt"]
    assert row["output_raw"]

    # OpenAI-spec response still echoes the client-supplied model.
    assert resp.json()["model"] == "mock-test"


def test_turn_id_null_when_not_supplied(tmp_path: Path) -> None:
    run = tmp_path / "run"
    client = _client(run)
    resp = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "no turn id"}]},
    )
    assert resp.status_code == 200
    assert _read_rows(run)[0]["turn_id"] is None


def test_turn_id_passthrough_when_supplied(tmp_path: Path) -> None:
    run = tmp_path / "run"
    client = _client(run)
    turn_id = "sidecar-turn-abc123"
    resp = client.post(
        "/v1/chat/completions",
        json={
            "messages": [{"role": "user", "content": "with turn id"}],
            "turn_id": turn_id,
        },
    )
    assert resp.status_code == 200
    assert _read_rows(run)[0]["turn_id"] == turn_id


def test_turn_counter_monotonic(tmp_path: Path) -> None:
    run = tmp_path / "run"
    client = _client(run)
    for i in range(3):
        resp = client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": f"m{i}"}]},
        )
        assert resp.status_code == 200
    assert [r["turn"] for r in _read_rows(run)] == [0, 1, 2]
