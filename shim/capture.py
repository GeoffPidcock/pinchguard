"""Activation capture backends for the Pinchguard inference shim.

Two backends, sharing a common `Capturer` protocol:

* `MockCapturer` — numpy-only, deterministic per (step_id, prompt). Used by
  parity tests and as the `auto`-fallback when real weights aren't available.
* `HFCapturer` — HuggingFace transformers + forward hooks on the residual
  stream at each configured layer. CPU-friendly for Qwen2.5-0.5B; the same
  module is the eventual home of the nnterp wrapping (PLAN.md Phase 4).

Schema fields produced (`activation_meta`, per AGENTS.md §3 v0.2):
  - token_position: which token's hidden state we kept (default `last_input`).
  - layers: integer indices captured.
  - dtype: storage dtype string.
  - capture_runtime: identifies the backend so future replays disambiguate.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

DEFAULT_LAYERS: tuple[int, ...] = (12, 24)
DEFAULT_TOKEN_POSITION: str = "last_input"
DEFAULT_DTYPE: str = "float16"


@dataclass
class CaptureResult:
    completion_text: str
    activations: dict[str, np.ndarray]
    activation_meta: dict[str, Any]
    prompt_text: str


class Capturer(Protocol):
    model_id: str

    def capture(self, messages: list[dict[str, Any]], step_id: str) -> CaptureResult: ...


def _flatten_messages(messages: list[dict[str, Any]]) -> str:
    return "\n".join(f"{m.get('role', '?')}: {m.get('content', '')}" for m in messages)


class MockCapturer:
    """Deterministic, numpy-only capture for tests and CPU-poor dev loops."""

    def __init__(
        self,
        *,
        layers: tuple[int, ...] = DEFAULT_LAYERS,
        token_position: str = DEFAULT_TOKEN_POSITION,
        hidden_dim: int = 8,
        model_id: str = "mock-qwen2.5-0.5b",
    ) -> None:
        self.layers = tuple(layers)
        self.token_position = token_position
        self.hidden_dim = hidden_dim
        self.model_id = model_id

    def capture(self, messages: list[dict[str, Any]], step_id: str) -> CaptureResult:
        prompt_text = _flatten_messages(messages)
        seed_bytes = hashlib.sha256(f"{step_id}|{prompt_text}".encode()).digest()[:8]
        rng = np.random.default_rng(int.from_bytes(seed_bytes, "big"))
        activations = {
            f"L{layer}": rng.standard_normal(size=(1, self.hidden_dim)).astype(np.float16)
            for layer in self.layers
        }
        meta: dict[str, Any] = {
            "token_position": self.token_position,
            "layers": list(self.layers),
            "dtype": DEFAULT_DTYPE,
            "capture_runtime": "mock",
        }
        completion = f"[mock {step_id[:8]}] ack: {prompt_text[-32:]}"
        return CaptureResult(
            completion_text=completion,
            activations=activations,
            activation_meta=meta,
            prompt_text=prompt_text,
        )


class HFCapturer:
    """HuggingFace transformers + forward-hook activation capture.

    Forward-hooks the residual stream output of each requested decoder layer,
    runs a prompt-only forward to pull `token_position` activations, then runs
    a separate greedy `generate` for the completion text. Two passes keeps
    activation semantics clean (no generation context bleeding back into the
    captured token's hidden state).

    nnterp swap-in (PLAN.md Phase 4): replace `_resolve_layers` + the hook
    machinery with `StandardizedTransformer.trace`; bump `capture_runtime` to
    `nnterp+hf-eager`.
    """

    def __init__(
        self,
        *,
        model_name: str,
        layers: tuple[int, ...] = DEFAULT_LAYERS,
        token_position: str = DEFAULT_TOKEN_POSITION,
        max_new_tokens: int = 64,
        dtype: str | None = None,
    ) -> None:
        import torch  # type: ignore[import-not-found]
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore[import-not-found]

        if token_position != "last_input":
            raise ValueError(
                f"token_position={token_position!r} not yet supported "
                "(only 'last_input' implemented)"
            )

        self._torch = torch
        self.model_id = model_name
        self.layers = tuple(layers)
        self.token_position = token_position
        self.max_new_tokens = max_new_tokens

        load_dtype = getattr(torch, dtype) if dtype else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # `low_cpu_mem_usage=True` skips the float32 init copy when we target
        # float16 — important on Codespaces, where total RAM is ~8 GiB and
        # the float32 spike OOMs even a 0.5B model.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, dtype=load_dtype, low_cpu_mem_usage=True
        )
        self.model.eval()

        self._layer_modules = self._resolve_layers()
        max_layer = len(self._layer_modules) - 1
        for layer in self.layers:
            if layer < 0 or layer > max_layer:
                raise ValueError(
                    f"layer {layer} out of range; model has {len(self._layer_modules)} layers"
                )

    def _resolve_layers(self) -> Any:
        m = self.model
        if hasattr(m, "model") and hasattr(m.model, "layers"):
            return m.model.layers
        if hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            return m.transformer.h
        raise RuntimeError(
            f"Could not locate decoder layers on {type(m).__name__}; "
            "extend HFCapturer._resolve_layers."
        )

    def capture(self, messages: list[dict[str, Any]], step_id: str) -> CaptureResult:
        torch = self._torch
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt")

        captured: dict[int, Any] = {}
        handles = []

        def _make_hook(idx: int):
            def _hook(_module, _inp, out):
                tensor = out[0] if isinstance(out, tuple) else out
                captured[idx] = tensor.detach()
            return _hook

        for layer_idx in self.layers:
            handles.append(
                self._layer_modules[layer_idx].register_forward_hook(_make_hook(layer_idx))
            )

        try:
            with torch.no_grad():
                self.model(**inputs)
        finally:
            for h in handles:
                h.remove()

        activations: dict[str, np.ndarray] = {}
        for layer_idx, tensor in captured.items():
            vec = tensor[:, -1, :]  # last_input position
            activations[f"L{layer_idx}"] = vec.to(torch.float16).cpu().numpy()

        with torch.no_grad():
            out_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        completion = self.tokenizer.decode(
            out_ids[0, inputs.input_ids.shape[1] :], skip_special_tokens=True
        )

        meta: dict[str, Any] = {
            "token_position": self.token_position,
            "layers": list(self.layers),
            "dtype": DEFAULT_DTYPE,
            "capture_runtime": "hf-eager",
        }
        return CaptureResult(
            completion_text=completion,
            activations=activations,
            activation_meta=meta,
            prompt_text=prompt_text,
        )
