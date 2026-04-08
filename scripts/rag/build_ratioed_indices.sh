#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=project
#SBATCH --job-name=rag_indices_136m
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=720G
#SBATCH --time=8:00:00
#SBATCH --output=slurm/build_index/ratioed_rag_136m_%A.out
#SBATCH --error=slurm/build_index/ratioed_rag_136m_%A.err

# ============================================================
# Build multiple ratioed FAISS indices from merged OR sharded source outputs.
#
# Default output layout:
#   <repo>/indices/136M_tok
#   <repo>/indices/272M_tok
#   ...
#
# Override by exporting env vars before sbatch:
#   SRC_DIR, OUT_ROOT, TOKEN_TARGETS_MILLIONS, SEED, INPUT_LAYOUT
# ============================================================

set -e

# --- CONFIGURATION (override via env vars) ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SRC_DIR="${SRC_DIR:-${ROOT_DIR}/rag_out}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/indices}"
TOKEN_TARGETS_MILLIONS="${TOKEN_TARGETS_MILLIONS:-136, 272, 680, 1360, 2040, 2720}"
SEED="${SEED:-42}"
INPUT_LAYOUT="${INPUT_LAYOUT:-auto}"
TOKEN_COUNT_WORKERS="${TOKEN_COUNT_WORKERS:-${SLURM_CPUS_PER_TASK:-1}}"
FAISS_DEVICE="${FAISS_DEVICE:-gpu}"
FAISS_NUM_GPUS="${FAISS_NUM_GPUS:-1}"
PARALLEL_INDEX_JOBS="${PARALLEL_INDEX_JOBS:-1}"

# Optional store cleanup before subsampling/indexing (backwards-compatible: off by default)
APPLY_STORE_FILTER="${APPLY_STORE_FILTER:-1}"
DEDUPE_EXACT_TEXT="${DEDUPE_EXACT_TEXT:-1}"
MIN_CHUNK_TOKENS="${MIN_CHUNK_TOKENS:-20}"
MAX_CHUNK_TOKENS="${MAX_CHUNK_TOKENS:-}"
MIN_CHUNK_CHARS="${MIN_CHUNK_CHARS:-80}"
MIN_ALPHA_RATIO="${MIN_ALPHA_RATIO:-0.6}"

# Keep index hyperparameters aligned with current build/merge scripts.
INDEX_TYPE="ivfpq"
NLIST=32768
PQ_M=64 # 128
PQ_NBITS=8
NPROBE=64
ENCODER="Qwen/Qwen3-Embedding-8B"

export PATH=$PATH:~/.local/bin

# Activate virtual environment
if command -v module >/dev/null 2>&1; then
  module load "${PYTHON_MODULE:-python/3.11.9}" || true
fi
if [[ -n "${VENV_PATH:-}" && -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${VENV_PATH}/bin/activate"
fi

export LD_PRELOAD=/usr/lib64/libstdc++.so.6
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH

# Navigate to project directory
cd "${ROOT_DIR}"

mkdir -p "$OUT_ROOT"
mkdir -p build_index

echo "=== Building ratioed RAG indices ==="
echo "Source index output dir:   $SRC_DIR"
echo "Output root dir:         $OUT_ROOT"
echo "Token targets (M):       ${TOKEN_TARGETS_MILLIONS:-<none>}"
echo "Seed:                    $SEED"
echo "Input layout:            $INPUT_LAYOUT"
echo "Token count workers:     $TOKEN_COUNT_WORKERS"
echo "Index type:              $INDEX_TYPE"
echo "FAISS device:            $FAISS_DEVICE"
echo "FAISS num GPUs:          $FAISS_NUM_GPUS"
echo "Parallel index jobs:     $PARALLEL_INDEX_JOBS"
echo "Apply store filter:      $APPLY_STORE_FILTER"
echo "Dedupe exact text:       $DEDUPE_EXACT_TEXT"
echo "Min chunk tokens:        $MIN_CHUNK_TOKENS"
echo "Max chunk tokens:        ${MAX_CHUNK_TOKENS:-<none>}"
echo "Min chunk chars:         $MIN_CHUNK_CHARS"
echo "Min alpha ratio:         $MIN_ALPHA_RATIO"
echo "Node:                    $(hostname)"
echo "GPU:                     $CUDA_VISIBLE_DEVICES"

FILTER_ARGS=()
if [ "$APPLY_STORE_FILTER" = "1" ]; then
  FILTER_ARGS+=(--enable_store_filter)
  if [ "$DEDUPE_EXACT_TEXT" = "1" ]; then
    FILTER_ARGS+=(--dedupe_exact_text)
  fi
  FILTER_ARGS+=(--min_chunk_tokens "$MIN_CHUNK_TOKENS")
  FILTER_ARGS+=(--min_chunk_chars "$MIN_CHUNK_CHARS")
  FILTER_ARGS+=(--min_alpha_ratio "$MIN_ALPHA_RATIO")
  if [ -n "$MAX_CHUNK_TOKENS" ]; then
    FILTER_ARGS+=(--max_chunk_tokens "$MAX_CHUNK_TOKENS")
  fi
fi

python build_ratioed_indices.py \
    --src_dir "$SRC_DIR" \
    --out_root "$OUT_ROOT" \
    --token_targets_millions "$TOKEN_TARGETS_MILLIONS" \
    --seed "$SEED" \
    --input_layout "$INPUT_LAYOUT" \
    --token_count_workers "$TOKEN_COUNT_WORKERS" \
    --index_type "$INDEX_TYPE" \
    --nlist "$NLIST" \
    --pq_m "$PQ_M" \
    --pq_nbits "$PQ_NBITS" \
    --nprobe "$NPROBE" \
    --encoder "$ENCODER" \
    --faiss_device "$FAISS_DEVICE" \
    --faiss_num_gpus "$FAISS_NUM_GPUS" \
    --parallel_index_jobs "$PARALLEL_INDEX_JOBS" \
    --gpu_use_float16 \
    "${FILTER_ARGS[@]}"

echo "=== Ratioed index build complete ==="
