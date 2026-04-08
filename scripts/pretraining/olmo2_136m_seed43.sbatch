#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=project
#SBATCH --job-name=olmo2_136m_100B_wsds_seed43
#SBATCH --output=jobs/olmo2_136m_100B_wsds_seed43_%j.out
#SBATCH --error=jobs/olmo2_136m_100B_wsds_seed43_%j.err
#SBATCH --time=1-00:00:00

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
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-olmo2-136m-100B-wsds-seed43}"
export WANDB_ENTITY="${WANDB_ENTITY:-your-entity}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LITGPT_DIR="${LITGPT_DIR:-${ROOT_DIR}/external/litgpt}"
export PYTHONPATH=${LITGPT_DIR}:$PYTHONPATH

echo "Starting OLMo-2 136M 100B-token Pretraining (WSD-S) with seed 43 on $(hostname)"
nvidia-smi

CONFIG="${ROOT_DIR}/external/litgpt_pretrain_configs/olmo2_136m_seed43.yaml"

cd "${LITGPT_DIR}/litgpt/"

echo "Running config: ${CONFIG}"
torchrun --nproc_per_node=4 -m litgpt pretrain --config "${CONFIG}"
