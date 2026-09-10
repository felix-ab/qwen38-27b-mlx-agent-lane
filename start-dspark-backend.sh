#!/bin/bash
# Text lane launcher: mlx-dspark 0.18.0 (lossless speculative decoding) + the incoai DFlash2 drafter serving a 6-bit
# OrcaRouter Qwen3.8-27B checkpoint on 127.0.0.1:8044. Sanitized template of the production launcher (2026-09-09).
# Env overrides: DSPARK_VENV DSPARK_MODEL DSPARK_DRAFTER DSPARK_SLOTS DSPARK_RUNGS DSPARK_PREFIX_CACHE_DIR DSPARK_CPU_SPLIT BACKEND_PORT
# Usage: start-dspark-backend.sh [dflash|dspark|lookup|baseline] [auto|<int>|derived]
set -euo pipefail
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export PYTHONNOUSERSITE=1
unset PYTHONPATH PYTHONHOME
VENV="${DSPARK_VENV:-$HOME/venvs/mlx-dspark}"
MODEL_PATH="${DSPARK_MODEL:-$HOME/models/Qwen3.8-27B-OrcaRouter-Uncensored-MLX/6-bit}"   # orcarouter/Qwen3.8-27B-Uncensored-MLX, 6-bit/
DRAFTER_PATH="${DSPARK_DRAFTER:-$HOME/models/Qwen3.8-27B-DFlash2}"                        # incoai/Qwen3.8-27B-DFlash2
MODE="${1:-dflash}"                # dflash | dspark | lookup | baseline
MAX_DRAFT="${2:-auto}"             # auto = adapt per round (production); integer = pinned width (use for A/Bs); derived = server default
case "$MODE" in dflash|dspark|lookup|baseline) ;; *) echo "bad mode $MODE" >&2; exit 64 ;; esac
[ -d "$MODEL_PATH" ] || { echo "model missing: $MODEL_PATH" >&2; exit 66; }
[ -d "$DRAFTER_PATH" ] || { echo "drafter missing: $DRAFTER_PATH" >&2; exit 66; }
DRAFT_ARGS=(); [ "$MAX_DRAFT" != "derived" ] && DRAFT_ARGS=(--max-draft "$MAX_DRAFT")
# CPU co-prefill: DSPARK_CPU_SPLIT=0 disables; a fraction forces; unset = calibrated. Needs the wide_gemm.py patch
# (Accelerate fp32 BLAS) — the stock bf16 BNNS path segfaulted mid-prefill on macOS 26 (mlx-dspark issue #31).
CPU_SPLIT_ARGS=(); [ -n "${DSPARK_CPU_SPLIT:-}" ] && CPU_SPLIT_ARGS=(--cpu-split "$DSPARK_CPU_SPLIT")
exec "$VENV/bin/mlx-dspark" serve \
  --model "$MODEL_PATH" --drafter "$DRAFTER_PATH" --mode "$MODE" ${DRAFT_ARGS[@]+"${DRAFT_ARGS[@]}"} \
  --host 127.0.0.1 --port "${BACKEND_PORT:-8044}" \
  --trust-remote-code --context-window 65536 --max-tokens-cap 4096 \
  --prefix-cache-slots "${DSPARK_SLOTS:-1}" --prefix-cache-rungs "${DSPARK_RUNGS:-4096}" \
  --prefix-cache-dir "${DSPARK_PREFIX_CACHE_DIR:-$HOME/.cache/dspark-prefix/qwen38-orca-6bit}" \
  --prefix-cache-max-ram-mb 0 \
  ${CPU_SPLIT_ARGS[@]+"${CPU_SPLIT_ARGS[@]}"} \
  --reasoning-effort medium --default-temperature 1.0 --default-top-p 0.95 --default-top-k 20
