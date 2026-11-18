#!/usr/bin/env bash
# End-to-end training pipeline (data prep, SFT, RL, export)
set -euo pipefail
export UV_LINK_MODE=copy

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [ -f "${REPO_ROOT}/.venv/bin/python" ]; then
    PY="${REPO_ROOT}/.venv/bin/python"
elif [ -f "${REPO_ROOT}/.venv/Scripts/python.exe" ]; then
    PY="${REPO_ROOT}/.venv/Scripts/python.exe"
elif command -v uv >/dev/null 2>&1; then
    PY="uv run --no-sync python"
else
    PY="python3"
fi

echo "[1/5] Inspecting repo & Stockfish..."
"${PY}" "${SCRIPT_DIR}/inspect_repo.py" --download-stockfish

TRAIN_PT="${SCRIPT_DIR}/data/sft_datamix/train_datamix.pt"
VAL_PT="${SCRIPT_DIR}/data/sft_datamix/val_positions.pt"

if [ -f "${TRAIN_PT}" ] && [ -f "${VAL_PT}" ]; then
    echo "[2/5] Existing dataset found, skipping generation."
else
    echo "[2/5] Preparing SFT dataset..."
    "${PY}" "${SCRIPT_DIR}/prepare_data.py"
fi

echo "[3/5] Running SFT..."
"${PY}" "${SCRIPT_DIR}/train_sft.py"

echo "[4/5] Running RL (GRPO + Q-distillation)..."
"${PY}" "${SCRIPT_DIR}/train_rl_grpo.py"

echo "[5/5] Exporting & verifying submission..."
"${PY}" "${SCRIPT_DIR}/export_submission.py" --run_engine_eval
