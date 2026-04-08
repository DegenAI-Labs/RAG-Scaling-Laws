#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=project
#SBATCH --job-name=olmo2_1b_100B_wsds
#SBATCH --output=jobs/olmo2_1b_100B_wsds_%j.out
#SBATCH --error=jobs/olmo2_1b_100B_wsds_%j.err
#SBATCH --time=2-00:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=4
#SBATCH --cpus-per-task=32

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
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-olmo2-1b-100B-wsds}"
export WANDB_ENTITY="${WANDB_ENTITY:-your-entity}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LITGPT_DIR="${LITGPT_DIR:-${ROOT_DIR}/external/litgpt}"
export PYTHONPATH=${LITGPT_DIR}:$PYTHONPATH

echo "Starting OLMo-2 1B 100B-token Pretraining (WSD-S) on $(hostname)"
nvidia-smi

CONFIG="${ROOT_DIR}/external/litgpt_pretrain_configs/olmo2_1b.yaml"

cd "${LITGPT_DIR}/litgpt/"

echo "Running config: ${CONFIG}"
torchrun --nproc_per_node=4 -m litgpt pretrain --config "${CONFIG}"
