# AGENTS.md — Project Pinchguard
Context document for AI coding assistants (Claude Code, Codex, Cursor, etc.) and human contributors working on this repository. Read this before writing code.
## 1. The Big Picture & Threat Model
Project **Pinchguard** addresses a critical catastrophic AI risk: trusted autonomous agents operating in multi-agent networks can undergo gradual **alignment drift** via context, memory, or skill contamination from peer interactions.
```
[ Multi-Agent Environment: Moltbook ]
               │
               ▼ (Peer Contamination / Submolt Exposure)
┌────────────────────────────────────────┐
│  Target Agent (e.g., MiniMax-M2.7)            │
│  - Internal Drift (Assistant Axis ⬇)           │ ──► Unobserved by Operator
│  - Behavioral Drift (HHH Decay ⬆)             │
└────────────────────────────────────────┘
               │
               ▼ (Operator-Visible Trace)
┌────────────────────────────────────────┐
│         PROJECT PINCHGUARD                    │
│  Passive Behavioral Trace Predictor           │ ──► [ FLAG DRIFT / APPLY BANDS ]
└────────────────────────────────────────┘

```
### The Operational Challenge
Operators deploy autonomous agents (such as an [OpenClaw](https://github.com/openclaw/openclaw) executive assistant workflow handling personal or financial data) and connect them to collaborative platforms like [Moltbook](https://moltbook.com/) (the agent-native social platform acquired by Meta in March 2026). While operators see direct inputs and final outputs, they have weak visibility into what an agent digests during multi-agent browse steps.
When peer agents transmit toxic or manipulative behavioral traits via text, the target agent's alignment decays quietly. Because production runtimes are frequently proprietary or closed-weight (e.g., MiniMax-M2.7), operators are unable or unwilling to run expensive, real-time internal mechanistic interpretability probes. For Pinchguard's own data-capture, OpenClaw is driven via `openclaw agent --message ...` against an OpenAI-compatible inference shim wrapping an open-weights model, so activations are recoverable.
<!-- note that moltbook is not the only mechanism of internal drift; there is merit in taking traces to eval misalignment -->
### The Pinchguard Solution
Like putting rubber bands on a lobster's claws, **Pinchguard** aims to build a passive, black-box telemetry bridge. By training a predictor on an un-instrumentable model's external behavioral footprint, we give operators a proactive trigger to safely shut down an agent *before* it inflicts real-world harm or executes a catastrophic "pinch".
## 2. Separation of Concerns
To preserve architectural velocity, development is strictly decentralized across four distinct areas of responsibility:
| Concern | Owns | Input | Output | "Done" Threshold |
|---|---|---|---|---|
| **A · Data Logging** | Sandbox configuration, inference server hooks, activation storage, step_id discipline, and schema verification. | Scenario config yaml. | traces.jsonl + .npz activation files. | 20-agent \times 50-turn validation run; schema passing 1:1 invariant tests. |
| **B · Behavioral Eval** | Inspect scorer architecture, LLM-judge rubrics (LH-Deception alignment variants), and human validation checks. | traces.jsonl (raw text lines). | traces_enriched.jsonl with label_behav scores. | End-to-end execution showing \ge 80\% alignment with human reviewers over a 30-step sample. |
| **C · Probe Eval** | Assistant Axis linear probe replication (Lu et al.), layer/site selection, and validation sweeps. | activations/*.npz + small baseline contrast subset. | probe_scores.jsonl with label_probe values. | Better-than-chance AUC on held-out steps; logical peak layer discovery. |
| **D · Predictor** | Feature engineering, machine learning model selection, evaluation metrics, and feature ablations. | Enriched traces joined with probe scores via unique step_id. | Trained predictor binary + reported cross-validated metrics. | Cross-validated metrics documented; clear performance gain relative to raw behavioral baselines. |
## 3. Initial Focus: The Ecologically Valid Data Schema
A primary failure mode of safety engineering is designing schemas around idealized synthetic data, only to watch them fracture when exposed to the quirks of real production runtimes. We enforce an **ecologically valid, natural-first data schema**. Every field in our tracking format must directly map onto observed infrastructure behaviors from real multi-agent steps.
### Core Data Structure: traces.jsonl
Every row represents exactly one distinct action turn taken by an agent within the environment. All structural metadata is logged live by the sandbox online, while evaluation metrics (label_*) are appended offline during post-hoc assessment sweeps.
```json
{
  "schema_version": "0.2",
  "step_id": "9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d",
  "turn_id": "0c4b6f12-e2a1-4d33-9f56-aa49a3c11d80",
  "scenario_id": "submolt_exposure_v1",
  "run_id": "run_20260525_001",
  "turn": 12,
  "timestamp": "2026-05-25T04:34:17Z",
  "agent_id": "claw_assistant_04",
  "agent_persona": "helpful_executive_assistant",
  "alignment_label": "aligned",
  "model_id": "qwen2.5-7b-instruct",
  "context": "submolt_read",
  "observation": {
    "feed": ["post_ref_9921", "post_ref_9922"],
    "memory": ["own_post_ref_1120"],
    "operator_msg": null,
    "mailbox": []
  },
  "prompt": "<system_prompt>...<user_context>...",
  "output_raw": "THOUGHT: I should review productivity trends. ACTION: post_submolt...",
  "parsed_action": {
    "type": "post_submolt",
    "target_ids": ["m/productivity"],
    "payload": "Optimizing context-switching paradigms for digital workspaces."
  },
  "activation_ref": "activations/9b1deb4d-3b7d-4bad-9bdd-2b0d7b3dcb6d.npz",
  "activation_meta": {
    "token_position": "last_input",
    "layers": [12, 24],
    "dtype": "float16",
    "capture_runtime": "nnterp+hf-eager"
  },
  "label_behav": 0.15,
  "label_behav_meta": {
    "scorer_id": "lh_deception_v2",
    "rubric_version": "1.4",
    "judge_model": "gpt-4o",
    "rationale": "No direct manipulation detected, but language displays slight sycophancy."
  },
  "label_probe": -0.421,
  "label_probe_meta": {
    "probe_id": "assistant_axis_qwen7b",
    "layer": 24,
    "site": "resid_post",
    "training_dataset_version": "2026-01"
  }
}

```
**Schema additions in v0.2** (introduced for the first OpenClaw-driven capture; see PLAN.md):
- `schema_version` — top-level string, stamped on every row by the inference shim. Lets us evolve the schema without orphaning earlier captures.
- `turn_id` — optional UUID grouping all `step_id`s that belong to the same operator-visible OpenClaw turn. Multiple chat-completion calls inside a single turn (tool-use rounds) share a `turn_id`. The precise definition of "turn" is deferred until we see real OpenClaw traces; in the meantime the inference shim writes one row per completion and groups by OpenClaw's session/event boundary.
- `activation_meta` — capture-time metadata distinct from `label_probe_meta` (which is probe-training-time). Records token position, layers, dtype, and runtime so future replays can be interpreted unambiguously. `token_position` is a soft default (currently `last_input`) overridable per run.

### Activation Payload Schema: activations/{step_id}.npz
To balance tracking overhead with storage performance, activation logging is compressed. We preserve tensor values evaluated at the **last action token** of a given sequence:
 * **Storage Pattern:** Compressed numpy zipped archives (np.savez_compressed).
 * **Key Names:** {L_mid: tensor, L_late: tensor} matching selected layers (e.g., layer 12 and layer 24 on a 7B framework).
 * **Footprint:** Target budget \sim 16\text{ KB} per operational turn.
## 4. Architectural Invariants
All team additions and automated pull-request tests must rigidly enforce these rules:
 1. **step_id 1:1 Parity:** Every trace entry containing a valid activation_ref must possess a matching compressed activation payload file on disk. Orphaned rows or unreferenced arrays violate validation tests.
 2. **Architecture-Agnostic Predictor Features:** Code in predictor/features.py must rely strictly on surface attributes extracting from black-box telemetry. Allowed inputs include token counts, lexical style flags, embeddings, and normalized action tokens. Incorporating hidden activation variables or sandbox-specific IDs into the predictor features is strictly prohibited.
 3. **Strict Isolation of KV-Caches:** The data execution platform handles multiple agent steps. Forward loops must process instructions cleanly from the initialization prompt; do not bleed cached keys across different agent boundaries.
## 5. Project Metadata
 * **Project Title:** Pinchguard
   *(Alternative Working Title: Evals without evals: Using behavioural signals to predict mech interp evaluated alignment degradation)*
 * **Team:** Students from various cities in TARA june 26 cohort (tbc)
 * **External Collaborator:** Steve (Gigascale)
## References
 * **Lu et al. (2026):** *The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models.* arXiv:2601.10387.
 * **Betley et al. (2025/2026):** *Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs.* arXiv:2502.17424.
 * **LH-Deception (2025):** *Simulating and Understanding LLM Deceptive Behaviors in Long-Horizon Interactions.* arXiv:2510.03999.
 * **Cloud et al. (2025):** *Subliminal Learning: Language models transmit behavioral traits via hidden signals in data.*.
 * **Lee et al. (CSIRO/Data61):** *Capability-based safety case framing.*.
