#!/bin/bash
# Convert LitGPT checkpoint to HuggingFace format
# Usage: ./convert_litgpt_to_hf.sh <model_size> <percentage>
# Example: ./convert_litgpt_to_hf.sh 30m 70pct

set -e

MODEL_SIZE=${1:-30m}
PCT=${2:-70pct}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LITGPT_DIR="${LITGPT_DIR:-${SCRIPT_DIR}/../external/litgpt}"
MODELS_BASE="${MODELS_BASE:-${SCRIPT_DIR}/../models}"

INPUT_DIR="${MODELS_BASE}/olmo2_${MODEL_SIZE}_${PCT}/final"
OUTPUT_DIR="${MODELS_BASE}/olmo2_${MODEL_SIZE}_${PCT}_hf"

echo "Converting OLMo-2 ${MODEL_SIZE} ${PCT} checkpoint to HuggingFace format"
echo "Input:  ${INPUT_DIR}"
echo "Output: ${OUTPUT_DIR}"

cd "${LITGPT_DIR}"
export PYTHONPATH="${LITGPT_DIR}:${PYTHONPATH}"

python3 -m litgpt.scripts.convert_lit_checkpoint "${INPUT_DIR}" "${OUTPUT_DIR}"
# HF expects pytorch_model.bin (or model.safetensors), not model.pth
python3 "${SCRIPT_DIR}/pth_to_pytorch_model_bin.py" "${OUTPUT_DIR}"
# LitGPT conversion does not copy tokenizer; HF needs tokenizer files in the dir
python3 "${SCRIPT_DIR}/save_olmo2_tokenizer.py" "${OUTPUT_DIR}"
echo "✓ Conversion complete! Output saved to ${OUTPUT_DIR}"

# Optional upload step can be added by your deployment workflow.