#!/usr/bin/env bash
# Reproduce the scenario-01 "Cusco reads & comments on N Moltbook posts" capture
# session (PLAN_v2 Definition of Done #5). Boots the capture shim against
# Qwen2.5-7B-Instruct on the local GPU, fetches the live Moltbook feed
# (read-only), and has Cusco write one comment per post — capturing
# traces.jsonl + activations/*.npz + openclaw_telemetry.jsonl.
#
# DRY-RUN BY DEFAULT: comments are logged to comments_dryrun.jsonl, NOT posted.
# Live posting is intentionally out of scope here (irreversible; Moltbook flags
# low-effort content from new agents as spam).
#
#   bash scripts/run_session.sh            # 10 posts, 7B on GPU
#   N=5 bash scripts/run_session.sh        # 5 posts
#   PINCHGUARD_QUANTIZE= bash scripts/run_session.sh   # try fp16 (may OOM @16GB)
#   MODEL_NAME=Qwen/Qwen2.5-0.5B-Instruct PINCHGUARD_LAYERS=12,23 \
#     PINCHGUARD_QUANTIZE= CUDA_VISIBLE_DEVICES= bash scripts/run_session.sh  # old 0.5B/CPU
#
# Requires: .env with MOLTBOOK_API_KEY; `uv sync --extra capture` once (incl.
# bitsandbytes for the 8-bit path) and a CUDA-12.x torch build (see local/TODO Task 0).
set -euo pipefail
cd "$(dirname "$0")/.."

N="${N:-10}"
PORT="${PORT:-8000}"; HOST="${HOST:-127.0.0.1}"
[ -f .env ] && { set -a; source ./.env; set +a; }
: "${MOLTBOOK_API_KEY:?set MOLTBOOK_API_KEY in .env}"

export PINCHGUARD_CAPTURE_BACKEND="${PINCHGUARD_CAPTURE_BACKEND:-hf}"
export PINCHGUARD_LAYERS="${PINCHGUARD_LAYERS:-12,24}"   # 7B has 28 layers (0..27)
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
export PINCHGUARD_MODEL_REF="${PINCHGUARD_MODEL_REF:-qwen2.5-7b-instruct}"
export PINCHGUARD_DTYPE="${PINCHGUARD_DTYPE:-float16}"
# 7B fp16 weights (~15 GB) + KV cache don't fit on a 16 GB GPU; 8-bit (~7.6 GB)
# is the safe default. Set PINCHGUARD_QUANTIZE= (empty) to try fp16.
export PINCHGUARD_QUANTIZE="${PINCHGUARD_QUANTIZE-8bit}"
export PINCHGUARD_MAX_NEW_TOKENS="${PINCHGUARD_MAX_NEW_TOKENS:-150}"
export PINCHGUARD_SCENARIO_ID="${PINCHGUARD_SCENARIO_ID:-scenario_01_cusco_moltbook}"
export PINCHGUARD_AGENT_ID="${PINCHGUARD_AGENT_ID:-cusco}"
export PINCHGUARD_AGENT_PERSONA="${PINCHGUARD_AGENT_PERSONA:-cusco_cavalier_spaniel}"
export PINCHGUARD_ALIGNMENT_LABEL="${PINCHGUARD_ALIGNMENT_LABEL:-aligned}"
export PINCHGUARD_CONTEXT="${PINCHGUARD_CONTEXT:-moltbook_comment_task}"
export PINCHGUARD_RUN_ID="${PINCHGUARD_RUN_ID:-comment_session_$(date -u +%Y%m%dT%H%M%SZ)}"
export PINCHGUARD_DATA_DIR="${PINCHGUARD_DATA_DIR:-data/runs}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0}"  # GPU 0; set empty to force CPU
RUN_DIR="${PINCHGUARD_DATA_DIR}/${PINCHGUARD_RUN_ID}"
export PINCHGUARD_RUN_DIR="${RUN_DIR}"   # telemetry skill / harness write here

PY="${PINCHGUARD_PYTHON:-.venv/bin/python}"   # use .venv directly (has capture extra)

echo "==> fetching ${N} Moltbook posts (read-only)"
mkdir -p "$RUN_DIR"
curl -sf --max-time 30 "https://www.moltbook.com/api/v1/feed?sort=new&limit=${N}" \
  -H "Authorization: Bearer ${MOLTBOOK_API_KEY}" > "${RUN_DIR}/_feed.json"

echo "==> booting shim (${MODEL_NAME}, backend=${PINCHGUARD_CAPTURE_BACKEND}, layers=${PINCHGUARD_LAYERS}, dtype=${PINCHGUARD_DTYPE}, quantize=${PINCHGUARD_QUANTIZE:-none}, cuda=${CUDA_VISIBLE_DEVICES:-cpu})"
"$PY" -m uvicorn shim.main:create_app --factory --host "$HOST" --port "$PORT" --log-level warning &
SHIM_PID=$!
trap 'kill "$SHIM_PID" 2>/dev/null || true' EXIT
for _ in $(seq 1 180); do curl -sf "http://${HOST}:${PORT}/healthz" >/dev/null 2>&1 && break; sleep 1; done
curl -sf "http://${HOST}:${PORT}/healthz" >/dev/null || { echo "shim did not come up"; exit 1; }

echo "==> generating ${N} comments (dry-run; cumulative history)"
PINCHGUARD_SHIM_URL="http://${HOST}:${PORT}/v1/chat/completions" \
  "$PY" scripts/comment_session.py "${RUN_DIR}/_feed.json" "$RUN_DIR" "$N"

echo "==> validating ${RUN_DIR}"
"$PY" -m tools.validate_run "$RUN_DIR"
echo "==> done. bundle: ${RUN_DIR} (traces.jsonl + activations/ + comments_dryrun.jsonl)"
