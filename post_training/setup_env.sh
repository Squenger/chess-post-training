#!/usr/bin/env bash
# Install dependencies, CUDA PyTorch (if GPU available), and Stockfish
set -euo pipefail
export UV_LINK_MODE=copy

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
mkdir -p "${SCRIPT_DIR}/bin"
cd "${REPO_ROOT}"

if head -c 24 "${REPO_ROOT}/base_model/model.safetensors" 2>/dev/null | grep -q "version https://git-lfs"; then
    if command -v git-lfs >/dev/null 2>&1; then
        git lfs pull
    elif command -v uvx >/dev/null 2>&1; then
        uvx hf download nathanael-fijalkow/chess-post-training --repo-type space --local-dir "${REPO_ROOT}"
    fi
fi

PACKAGES=(
    "chess==1.11.2"
    "safetensors>=0.4"
    "numpy>=1.24"
    "huggingface_hub>=0.24"
    "cryptography>=42"
    "datasets>=2.18"
    "stockfish>=3.28"
    "tqdm>=4.66"
    "wandb>=0.16"
    "accelerate>=0.28"
)

if command -v uv >/dev/null 2>&1; then
    uv sync || true
    if command -v nvidia-smi >/dev/null 2>&1; then
        uv pip install --link-mode=copy --reinstall-package torch --reinstall-package sympy --reinstall-package mpmath torch sympy mpmath --index-url https://download.pytorch.org/whl/cu124 || uv pip install "torch>=2.2"
    fi
    uv pip install --link-mode=copy "${PACKAGES[@]}"
    if [ -f "${REPO_ROOT}/.venv/bin/python" ]; then
        PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
    elif [ -f "${REPO_ROOT}/.venv/Scripts/python.exe" ]; then
        PYTHON_BIN="${REPO_ROOT}/.venv/Scripts/python.exe"
    else
        PYTHON_BIN="uv run --no-sync python"
    fi
else
    [ ! -d "${REPO_ROOT}/.venv" ] && python3 -m venv "${REPO_ROOT}/.venv"
    if [ -f "${REPO_ROOT}/.venv/bin/pip" ]; then
        PIP_CMD="${REPO_ROOT}/.venv/bin/pip"
        PYTHON_BIN="${REPO_ROOT}/.venv/bin/python"
    else
        PIP_CMD="${REPO_ROOT}/.venv/Scripts/pip.exe"
        PYTHON_BIN="${REPO_ROOT}/.venv/Scripts/python.exe"
    fi
    "${PIP_CMD}" install --upgrade pip
    if command -v nvidia-smi >/dev/null 2>&1; then
        "${PIP_CMD}" install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124 || "${PIP_CMD}" install "torch>=2.2"
    fi
    "${PIP_CMD}" install "${PACKAGES[@]}"
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/inspect_repo.py" --download-stockfish
