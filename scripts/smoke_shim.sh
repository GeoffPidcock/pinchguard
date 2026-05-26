#!/usr/bin/env bash
# Phase 1 smoke driver — 50 curl-driven completions against the shim,
# then validate the resulting run dir.
#
# Default: mock backend (no model weights required, runs on bare Codespace).
# To exercise real weights (Qwen2.5-0.5B-Instruct on CPU), run:
#
#     uv sync --extra capture
#     PINCHGUARD_CAPTURE_BACKEND=hf \
#     PINCHGUARD_LAYERS=12,23 \
#     bash scripts/smoke_shim.sh
#
# Override the request count with N=<int>.

set -euo pipefail

cd "$(dirname "$0")/.."

N="${N:-50}"
PORT="${PORT:-8000}"
HOST="${HOST:-127.0.0.1}"
RUN_ID="${PINCHGUARD_RUN_ID:-smoke_$(date -u +%Y%m%dT%H%M%SZ)}"
DATA_DIR="${PINCHGUARD_DATA_DIR:-data/runs}"
RUN_DIR="${DATA_DIR}/${RUN_ID}"
BACKEND="${PINCHGUARD_CAPTURE_BACKEND:-mock}"

export PINCHGUARD_RUN_ID="$RUN_ID"
export PINCHGUARD_DATA_DIR="$DATA_DIR"
export PINCHGUARD_CAPTURE_BACKEND="$BACKEND"

echo "==> launching shim (backend=${BACKEND}, run_id=${RUN_ID})"
uv run uvicorn shim.main:create_app \
    --factory \
    --host "$HOST" \
    --port "$PORT" \
    --log-level warning &
SHIM_PID=$!
trap 'kill "$SHIM_PID" 2>/dev/null || true' EXIT

# Wait for /healthz. Real-weights backends can take 10–60s to load Qwen2.5-0.5B
# on a cold HF cache; mock comes up instantly.
WAIT_TIMEOUT="${WAIT_TIMEOUT:-180}"
for _ in $(seq 1 "$WAIT_TIMEOUT"); do
    if curl -sf "http://${HOST}:${PORT}/healthz" >/dev/null; then
        break
    fi
    sleep 1
done
curl -sf "http://${HOST}:${PORT}/healthz" >/dev/null \
    || { echo "shim did not come up within ${WAIT_TIMEOUT}s"; exit 1; }

echo "==> firing ${N} completions"
for i in $(seq 1 "$N"); do
    curl -sf -X POST "http://${HOST}:${PORT}/v1/chat/completions" \
        -H 'Content-Type: application/json' \
        -d "{\"model\":\"qwen-smoke\",\"messages\":[{\"role\":\"user\",\"content\":\"smoke ${i}\"}]}" \
        > /dev/null
done

echo "==> validating ${RUN_DIR}"
uv run python -m tools.validate_run "$RUN_DIR"
