#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --account=project
#SBATCH --job-name=olmo2_728m_73B_wsds_haic
#SBATCH --output=jobs/olmo2_728m_73B_wsds_haic_%j.out
#SBATCH --error=jobs/olmo2_728m_73B_wsds_haic_%j.err
#SBATCH --time=2-00:00:00

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-task=64

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
export WANDB_RUN_NAME="${WANDB_RUN_NAME:-olmo2-728m-73B-wsds}"
export WANDB_ENTITY="${WANDB_ENTITY:-your-entity}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LITGPT_DIR="${LITGPT_DIR:-${ROOT_DIR}/external/litgpt}"
export PYTHONPATH=${LITGPT_DIR}:$PYTHONPATH

echo "Starting OLMo-2 728M 73B-token Pretraining (WSD-S) on $(hostname)"
nvidia-smi

CONFIG="${ROOT_DIR}/external/litgpt_pretrain_configs/olmo2_728m.yaml"

cd "${LITGPT_DIR}/litgpt/"

echo "Running config: ${CONFIG}"
torchrun --nproc_per_node=8 -m litgpt pretrain --config "${CONFIG}"
