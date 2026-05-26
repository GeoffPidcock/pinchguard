"""Phase 0 schema invariant tests — exercise tools/validate_run.py against the
hand-rolled sample run from conftest.

Covers AGENTS.md §4 invariant #1 (step_id ↔ npz parity) on the happy path plus
the three failure modes the validator must detect.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tools.validate_run import validate_run  # noqa: E402


def test_sample_run_passes(sample_run: Path) -> None:
    result = validate_run(sample_run)
    assert result.ok, result.report()
    assert result.rows_checked == 2
    assert result.missing_npz == []
    assert result.orphan_npz == []
    assert result.bad_rows == []


def test_missing_npz_detected(sample_run: Path, tmp_path: Path) -> None:
    import shutil

    run = tmp_path / "run"
    shutil.copytree(sample_run, run)
    next(run.glob("activations/*.npz")).unlink()

    result = validate_run(run)
    assert not result.ok
    assert len(result.missing_npz) == 1


def test_orphan_npz_detected(sample_run: Path, tmp_path: Path) -> None:
    import shutil

    run = tmp_path / "run"
    shutil.copytree(sample_run, run)
    orphan = run / "activations" / "deadbeef-dead-dead-dead-deadbeefdead.npz"
    np.savez_compressed(orphan, L12=np.zeros((1, 4), dtype=np.float16))

    result = validate_run(run)
    assert not result.ok
    assert len(result.orphan_npz) == 1


def test_step_id_mismatch_detected(sample_run: Path, tmp_path: Path) -> None:
    import json
    import shutil

    run = tmp_path / "run"
    shutil.copytree(sample_run, run)
    traces = run / "traces.jsonl"
    rows = [json.loads(l) for l in traces.read_text().splitlines() if l.strip()]
    rows[0]["activation_ref"] = "activations/not-matching-step-id.npz"
    traces.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    result = validate_run(run)
    assert not result.ok
    assert any("does not match step_id" in r for r in result.bad_rows)
