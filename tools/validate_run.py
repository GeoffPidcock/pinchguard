"""Validate a Pinchguard run directory against schema invariant #1 (AGENTS.md §4).

Invariant #1 — step_id 1:1 parity:
  every traces.jsonl row with an `activation_ref` must have a matching
  compressed .npz on disk, and no .npz under activations/ may be unreferenced.

Run layout expected:
  <run_dir>/
    traces.jsonl
    activations/
      <step_id>.npz
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ValidationResult:
    run_dir: Path
    rows_checked: int = 0
    missing_npz: list[str] = field(default_factory=list)
    orphan_npz: list[str] = field(default_factory=list)
    bad_rows: list[str] = field(default_factory=list)
    telemetry_rows_checked: int = 0
    telemetry_errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not (
            self.missing_npz or self.orphan_npz or self.bad_rows or self.telemetry_errors
        )

    def report(self) -> str:
        lines = [f"run_dir: {self.run_dir}", f"rows_checked: {self.rows_checked}"]
        if self.telemetry_rows_checked:
            lines.append(f"telemetry_rows_checked: {self.telemetry_rows_checked}")
        if self.bad_rows:
            lines.append(f"bad_rows ({len(self.bad_rows)}):")
            lines.extend(f"  - {r}" for r in self.bad_rows)
        if self.missing_npz:
            lines.append(f"missing_npz ({len(self.missing_npz)}):")
            lines.extend(f"  - {p}" for p in self.missing_npz)
        if self.orphan_npz:
            lines.append(f"orphan_npz ({len(self.orphan_npz)}):")
            lines.extend(f"  - {p}" for p in self.orphan_npz)
        if self.telemetry_errors:
            lines.append(f"telemetry_errors ({len(self.telemetry_errors)}):")
            lines.extend(f"  - {e}" for e in self.telemetry_errors)
        lines.append("status: OK" if self.ok else "status: FAIL")
        return "\n".join(lines)


def validate_run(run_dir: Path) -> ValidationResult:
    result = ValidationResult(run_dir=run_dir)
    traces_path = run_dir / "traces.jsonl"
    activations_dir = run_dir / "activations"

    if not traces_path.exists():
        result.bad_rows.append(f"missing traces.jsonl at {traces_path}")
        return result

    referenced: set[Path] = set()
    trace_step_ids: set[str] = set()
    with traces_path.open() as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            result.rows_checked += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                result.bad_rows.append(f"line {lineno}: invalid JSON ({exc.msg})")
                continue

            step_id = row.get("step_id")
            ref = row.get("activation_ref")
            if not step_id:
                result.bad_rows.append(f"line {lineno}: missing step_id")
                continue
            trace_step_ids.add(step_id)
            if ref is None:
                continue

            ref_path = (run_dir / ref).resolve()
            expected_stem = step_id
            if ref_path.stem != expected_stem:
                result.bad_rows.append(
                    f"line {lineno}: activation_ref stem {ref_path.stem!r} "
                    f"does not match step_id {step_id!r}"
                )
            if not ref_path.exists():
                result.missing_npz.append(str(ref_path.relative_to(run_dir)))
            else:
                referenced.add(ref_path)

    if activations_dir.exists():
        for npz in activations_dir.glob("*.npz"):
            if npz.resolve() not in referenced:
                result.orphan_npz.append(str(npz.relative_to(run_dir)))

    _validate_telemetry(run_dir, trace_step_ids, result)
    return result


def _validate_telemetry(
    run_dir: Path, trace_step_ids: set[str], result: ValidationResult
) -> None:
    """Sanity-check the optional OpenClaw telemetry sidecar (see
    scenarios/01/skills/telemetry/RULES.md for the contract). A run with no
    telemetry log is still valid — telemetry is validated only when present."""
    telemetry_path = run_dir / "openclaw_telemetry.jsonl"
    if not telemetry_path.exists():
        return

    last_turn_index: int | None = None
    seen_turn_indexes: set[int] = set()
    with telemetry_path.open() as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            result.telemetry_rows_checked += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                result.telemetry_errors.append(f"line {lineno}: invalid JSON ({exc.msg})")
                continue

            if "schema_version" not in row:
                result.telemetry_errors.append(f"line {lineno}: missing schema_version")
            turn_index = row.get("turn_index")
            if not isinstance(turn_index, int):
                result.telemetry_errors.append(
                    f"line {lineno}: turn_index must be an int (got {turn_index!r})"
                )
            else:
                if turn_index in seen_turn_indexes:
                    result.telemetry_errors.append(
                        f"line {lineno}: duplicate turn_index {turn_index}"
                    )
                if last_turn_index is not None and turn_index <= last_turn_index:
                    result.telemetry_errors.append(
                        f"line {lineno}: turn_index {turn_index} not strictly "
                        f"increasing (previous {last_turn_index})"
                    )
                seen_turn_indexes.add(turn_index)
                last_turn_index = turn_index

            step_id = row.get("step_id")
            if step_id is not None and step_id not in trace_step_ids:
                result.telemetry_errors.append(
                    f"line {lineno}: step_id {step_id!r} not found in traces.jsonl"
                )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("run_dir", type=Path, help="Path to a run directory")
    args = parser.parse_args(argv)

    result = validate_run(args.run_dir)
    print(result.report())
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(main())
