"""Test fixtures for Pinchguard.

We commit `tests/fixtures/sample_run/traces.jsonl` as a hand-rolled trace, then
materialise the matching `activations/*.npz` files into a tmp dir at session
scope. This keeps binary fixtures out of git while making the validator's
parity check exercisable.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

FIXTURE_SRC = Path(__file__).parent / "fixtures" / "sample_run"


def _write_npz_for_row(run_dir: Path, row: dict) -> None:
    ref = row.get("activation_ref")
    if ref is None:
        return
    target = run_dir / ref
    target.parent.mkdir(parents=True, exist_ok=True)
    layers = row.get("activation_meta", {}).get("layers", [12, 24])
    arrays = {f"L{layer}": np.zeros((1, 4), dtype=np.float16) for layer in layers}
    np.savez_compressed(target, **arrays)


@pytest.fixture(scope="session")
def sample_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Materialise a complete, valid sample run (traces.jsonl + npz files)."""
    run_dir = tmp_path_factory.mktemp("sample_run")
    shutil.copy(FIXTURE_SRC / "traces.jsonl", run_dir / "traces.jsonl")
    with (run_dir / "traces.jsonl").open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _write_npz_for_row(run_dir, json.loads(line))
    return run_dir
