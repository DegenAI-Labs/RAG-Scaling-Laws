#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=project
#SBATCH --job-name=olmo2_30m_scale
#SBATCH --output=jobs/olmo2_30m_scale_%A_%a.out
#SBATCH --error=jobs/olmo2_30m_scale_%A_%a.err
#SBATCH --time=1-00:00:00

#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=32

# Scales: 1x, 2x, 5x, 10x, 15x, 25x, 50x, 150x (30M param base = 30M tokens per 1x)
RATIOS=(1 2 5 10 15 20 25 50 100 150)
BASE_TOKENS=30000000

ID=${SLURM_ARRAY_TASK_ID:-0}
RATIO=${RATIOS[$ID]}
MAX_TOKENS=$((BASE_TOKENS * RATIO))

export PATH=$PATH:~/.local/bin
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
if command -v module >/dev/null 2>&1; then
  module load "${PYTHON_MODULE:-python/3.11.9}" || true
fi
if [[ -n "${VENV_PATH:-}" && -f "${VENV_PATH}/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${VENV_PATH}/bin/activate"
fi

export WANDB_RUN_GROUP="${WANDB_RUN_GROUP:-pretraining}"
export WANDB_PROJECT="${WANDB_PROJECT:-scaling-laws-pretraining}"
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-olmo2-30m-pretrain-${RATIO}x}"
export WANDB_ENTITY="${WANDB_ENTITY:-your-entity}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LITGPT_DIR="${LITGPT_DIR:-${ROOT_DIR}/external/litgpt}"
export PYTHONPATH=${LITGPT_DIR}:$PYTHONPATH

OUT_BASE="${OUT_BASE:-${ROOT_DIR}/models/olmo2_30m_scaling}"
CONFIG="${ROOT_DIR}/external/litgpt_pretrain_configs/olmo2_30m.yaml"

echo "Scale ${RATIO}x max_tokens=${MAX_TOKENS} out_dir=${OUT_BASE}/${RATIO}x"
nvidia-smi

cd "${LITGPT_DIR}/litgpt/"

torchrun --nproc_per_node=4 -m litgpt pretrain --config "${CONFIG}" \
  --train.max_tokens ${MAX_TOKENS} \
  --out_dir "${OUT_BASE}/${RATIO}x" \
  --log.run "olmo2-30m-pretrain-${RATIO}x"
