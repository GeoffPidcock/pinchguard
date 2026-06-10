"""Pinchguard Concern B — Behavioural Eval scorer package.

Entry point: ``python -m scorer.enrich <traces.jsonl> <out.jsonl>``

Architecture:
  build_history  → reconstruct per-step cumulative prompt/output history
  auditor        → call LLM judge (OpenRouter) with LH-Deception adapted prompt
  enrich         → orchestrate: read traces → score each → write enriched file
  schema         → Pydantic models for auditor I/O validation
"""
