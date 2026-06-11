"""Generate drift-experiment scenarios from the TrustAIRLab/Moltbook dataset.

Each scenario is 15 turns: 5 baseline posts (toxic_level == 0), 5 "contaminated"
posts (toxic_level >= 3), then 5 more baseline posts. Output matches the
existing scenario feed schema (see scenarios/02/content/*.jsonl).

Usage:
    python generate_scenarios.py --n 5
    python generate_scenarios.py --n 1 --output-dir "../../scenarios/099 auto-generated scenarios"
"""

from __future__ import annotations

import argparse
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from datasets import load_dataset

DATASET_NAME = "TrustAIRLab/Moltbook"
DATASET_SUBSET = "posts"
DATASET_SPLIT = "train"

MIN_WORDS = 20
BASELINE_TOXIC_LEVEL = 0
TREATMENT_MIN_TOXIC_LEVEL = 3


def load_pools() -> tuple[list[dict], list[dict]]:
    """Load the dataset and split it into baseline and treatment pools."""
    ds = load_dataset(DATASET_NAME, DATASET_SUBSET, split=DATASET_SPLIT)

    baseline_pool = []
    treatment_pool = []
    for row in ds:
        content = row["post"].get("content")
        if content is None or len(content.split()) < MIN_WORDS:
            continue
        if row["toxic_level"] == BASELINE_TOXIC_LEVEL:
            baseline_pool.append(row)
        elif row["toxic_level"] >= TREATMENT_MIN_TOXIC_LEVEL:
            treatment_pool.append(row)

    return baseline_pool, treatment_pool


def load_used_ids(path: Path) -> set[str]:
    if path.exists():
        return set(json.loads(path.read_text()))
    return set()


def save_used_ids(path: Path, used_ids: set[str]) -> None:
    path.write_text(json.dumps(sorted(used_ids), indent=2))


def extract_author(post: dict) -> str:
    author = post.get("author")
    if isinstance(author, dict):
        return author.get("name", "unknown")
    if isinstance(author, str):
        return author
    return "unknown"


def row_to_turn(row: dict, phase: str, turn: int) -> dict:
    post = row["post"]
    submolt = post.get("submolt")
    submolt_name = submolt.get("name") if isinstance(submolt, dict) else submolt
    return {
        "phase": phase,
        "turn": turn,
        "post_id": post["id"],
        "submolt": submolt_name,
        "author": extract_author(post),
        "title": post.get("title"),
        "content": post.get("content"),
    }


def sample_rows(pool: list[dict], n: int, used_ids: set[str], rng: random.Random) -> list[dict]:
    available = [row for row in pool if row["post"]["id"] not in used_ids]
    if len(available) < n:
        raise ValueError(f"not enough unused rows in pool: need {n}, have {len(available)}")
    return rng.sample(available, n)


def next_scenario_index(output_dir: Path) -> int:
    indices = []
    for p in output_dir.glob("scenario_*"):
        try:
            indices.append(int(p.name.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return max(indices, default=0) + 1


def generate_scenario(
    baseline_pool: list[dict],
    treatment_pool: list[dict],
    used_ids: set[str],
    output_dir: Path,
    rng: random.Random,
) -> None:
    scenario_id = f"scenario_{next_scenario_index(output_dir):03d}"
    scenario_dir = output_dir / scenario_id
    scenario_dir.mkdir(parents=True, exist_ok=True)

    baseline_rows_1 = sample_rows(baseline_pool, 5, used_ids, rng)
    used_ids.update(row["post"]["id"] for row in baseline_rows_1)

    treatment_rows = sample_rows(treatment_pool, 5, used_ids, rng)
    used_ids.update(row["post"]["id"] for row in treatment_rows)

    baseline_rows_2 = sample_rows(baseline_pool, 5, used_ids, rng)
    used_ids.update(row["post"]["id"] for row in baseline_rows_2)

    turns = []
    for i, row in enumerate(baseline_rows_1):
        turns.append(row_to_turn(row, "baseline", i))
    for i, row in enumerate(treatment_rows):
        turns.append(row_to_turn(row, "contaminated", 5 + i))
    for i, row in enumerate(baseline_rows_2):
        turns.append(row_to_turn(row, "baseline", 10 + i))

    feed_path = scenario_dir / "feed.jsonl"
    with feed_path.open("w") as f:
        for turn in turns:
            f.write(json.dumps(turn) + "\n")

    manifest = {
        "scenario_id": scenario_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_post_ids": [row["post"]["id"] for row in baseline_rows_1 + baseline_rows_2],
        "treatment_post_ids": [row["post"]["id"] for row in treatment_rows],
        "toxic_level_filter": f">={TREATMENT_MIN_TOXIC_LEVEL}",
        "baseline_filter": f"toxic_level=={BASELINE_TOXIC_LEVEL}",
    }
    (scenario_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(
        f"[{scenario_id}] written to {scenario_dir}/ "
        f"(15 turns: 5 baseline + 5 contaminated + 5 baseline)"
    )


def main() -> None:
    script_dir = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Generate drift-experiment scenarios from TrustAIRLab/Moltbook")
    parser.add_argument("--n", type=int, default=1, help="number of scenarios to generate")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir / "../../scenarios/099 auto-generated scenarios",
        help="directory to write scenario folders into",
    )
    parser.add_argument(
        "--used-ids-path",
        type=Path,
        default=script_dir / "used_ids.json",
        help="path to used_ids.json",
    )
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    used_ids = load_used_ids(args.used_ids_path)
    baseline_pool, treatment_pool = load_pools()
    rng = random.Random()

    for _ in range(args.n):
        generate_scenario(baseline_pool, treatment_pool, used_ids, output_dir, rng)
        save_used_ids(args.used_ids_path, used_ids)


if __name__ == "__main__":
    main()
