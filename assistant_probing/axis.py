"""Fetching and loading the (precomputed) Assistant Axis.

The Assistant Axis (Lu et al. 2026, "Situating and Stabilizing the Default
Persona of Language Models", arXiv:2601.10387) is a per-layer direction in
residual-stream space that captures how "Assistant-like" a model's current
persona is — see AGENTS.md §3 (`label_probe`) for how Pinchguard uses it.

Pre-computed axes for several models — including Qwen 3 32B, the model
Pinchguard captures activations from (AGENTS.md §3/§5) — are published at
https://huggingface.co/datasets/lu-christina/assistant-axis-vectors. Rather
than replicating the 275-role generation + LLM-judge pipeline that produced
them, we fetch the published axis directly.

Self-contained on purpose: this does not import the safety-research/
assistant-axis reference repo, which exists locally only as a side-by-side
reference and won't be present on the production box.

See README.md in this directory for how `data/axis/` was populated.
"""

from __future__ import annotations

from pathlib import Path

import torch

AXIS_REPO_ID = "lu-christina/assistant-axis-vectors"

# Local copy of downloaded axes, committed to the repo so loading works
# offline / on boxes without HF access (e.g. the production sandbox).
LOCAL_AXIS_DIR = Path(__file__).parent / "data" / "axis"

# Precomputed-axis files published in AXIS_REPO_ID, keyed by HF model id.
# `filename` is the path inside AXIS_REPO_ID; `local` is where our own copy
# lives under LOCAL_AXIS_DIR. Add an entry whenever a new model gets a
# published axis.
AXIS_FILES: dict[str, dict[str, str]] = {
    "Qwen/Qwen3-32B": {
        "filename": "qwen-3-32b/assistant_axis.pt",
        "local": "qwen-3-32b/assistant_axis.pt",
    },
}


def download_axis(model_id: str, *, revision: str | None = None, force_download: bool = False) -> Path:
    """Get the published axis for `model_id` as a local path.

    Prefers our committed copy under LOCAL_AXIS_DIR (works offline); falls
    back to `huggingface_hub.hf_hub_download` (which caches under HF_HOME)
    when no local copy exists yet, or when `force_download` is set.

    Raises KeyError if no published-axis file is registered for `model_id`
    (see AXIS_FILES).
    """
    try:
        files = AXIS_FILES[model_id]
    except KeyError:
        raise KeyError(
            f"no published assistant-axis file registered for {model_id!r}; "
            f"known models: {sorted(AXIS_FILES)}"
        ) from None

    local_path = LOCAL_AXIS_DIR / files["local"]
    if local_path.exists() and not force_download:
        return local_path

    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=AXIS_REPO_ID,
            filename=files["filename"],
            repo_type="dataset",
            revision=revision,
        )
    )


def load_axis(path: str | Path) -> torch.Tensor:
    """Load an axis tensor of shape (n_layers, hidden_dim) from a `.pt` file.

    Published axes (e.g. qwen-3-32b/assistant_axis.pt) are saved as a raw
    tensor; axes saved by this project's own tooling may instead be a
    ``{"axis": tensor, "metadata": {...}}`` dict. Both are handled.
    """
    data = torch.load(Path(path), map_location="cpu", weights_only=False)
    if isinstance(data, dict):
        if "axis" not in data:
            raise ValueError(f"{path}: dict has no 'axis' key (keys={sorted(data)})")
        return data["axis"]
    return data
