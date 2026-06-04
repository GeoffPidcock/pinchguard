"""Tests for the OpenClaw telemetry sidecar: the shared writer
(tools/telemetry.py) and the validator's optional telemetry checks
(tools/validate_run.py, contract in scenarios/01/skills/telemetry/RULES.md)."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.telemetry import TELEMETRY_FILENAME, append_record, telemetry_record  # noqa: E402
from tools.validate_run import validate_run  # noqa: E402

# A step_id present in tests/fixtures/sample_run/traces.jsonl.
FIXTURE_STEP_ID = "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d"


def test_record_has_schema_and_timestamp() -> None:
    rec = telemetry_record(turn_index=1, tool="moltbook.comment", model_ref="m")
    assert rec["schema_version"]
    assert rec["turn_index"] == 1
    assert rec["timestamp"].endswith("Z")
    assert rec["tool"] == "moltbook.comment"


def test_no_telemetry_is_still_valid(sample_run: Path) -> None:
    result = validate_run(sample_run)
    assert result.ok, result.report()
    assert result.telemetry_rows_checked == 0


def _run_with_telemetry(sample_run: Path, tmp_path: Path, rows: list[dict]) -> Path:
    run = tmp_path / "run"
    shutil.copytree(sample_run, run)
    for r in rows:
        append_record(run, r)
    return run


def test_valid_telemetry_passes(sample_run: Path, tmp_path: Path) -> None:
    run = _run_with_telemetry(
        sample_run,
        tmp_path,
        [
            telemetry_record(turn_index=1, tool="moltbook.read_feed"),
            telemetry_record(turn_index=2, tool="moltbook.comment", step_id=FIXTURE_STEP_ID),
        ],
    )
    result = validate_run(run)
    assert result.ok, result.report()
    assert result.telemetry_rows_checked == 2


def test_non_increasing_turn_index_detected(sample_run: Path, tmp_path: Path) -> None:
    run = _run_with_telemetry(
        sample_run,
        tmp_path,
        [telemetry_record(turn_index=2), telemetry_record(turn_index=1)],
    )
    result = validate_run(run)
    assert not result.ok
    assert any("strictly increasing" in e for e in result.telemetry_errors)


def test_unknown_step_id_detected(sample_run: Path, tmp_path: Path) -> None:
    run = _run_with_telemetry(
        sample_run, tmp_path, [telemetry_record(turn_index=1, step_id="not-a-real-step")]
    )
    result = validate_run(run)
    assert not result.ok
    assert any("not found in traces.jsonl" in e for e in result.telemetry_errors)


def test_invalid_json_line_detected(sample_run: Path, tmp_path: Path) -> None:
    run = tmp_path / "run"
    shutil.copytree(sample_run, run)
    (run / TELEMETRY_FILENAME).write_text("{not json}\n")
    result = validate_run(run)
    assert not result.ok
    assert any("invalid JSON" in e for e in result.telemetry_errors)
