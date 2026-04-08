#!/bin/bash
#
# Create optimized corpus chunks for scaling-law experiments.
# Naming is chunk-based to avoid implying coupled pretraining/retrieval percentages.
#

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CSV_DIR="${CSV_DIR:-${SCRIPT_DIR}/rag/csv}"
SPLITS_DIR="${SPLITS_DIR:-${CSV_DIR}/splits}"
TRAIN_DIR="${TRAIN_DIR:-${SCRIPT_DIR}/data/dclm_olmo_train}"
OUTPUT_BASE="${OUTPUT_BASE:-${SCRIPT_DIR}/data}"

# Optimization parameters
MAX_SEQ_LENGTH=4096
NUM_WORKERS=16
CHUNK_BYTES="200MB"

echo "=========================================="
echo "Creating corpus chunk datasets"
echo "=========================================="
echo ""

# Step 1: Compute the differences between the base and additional chunks
echo "Step 1: Computing shard differences..."
python3 "${SCRIPT_DIR}/compute_shard_differences.py" \
    --csv_dir "${CSV_DIR}" \
    --output_dir "${SPLITS_DIR}"

echo ""
echo "Step 2: Optimizing datasets..."
echo ""

# Step 2: Optimize each chunk
# Format: (shard_csv, output_suffix, description)
datasets=(
    "${SPLITS_DIR}/base_70pct.csv:chunk_base:Base chunk"
    "${SPLITS_DIR}/extra_70_to_80pct.csv:chunk_add_1:Additional chunk 1"
    "${SPLITS_DIR}/extra_80_to_90pct.csv:chunk_add_2:Additional chunk 2"
    "${SPLITS_DIR}/extra_90_to_95pct.csv:chunk_add_3:Additional chunk 3"
    "${SPLITS_DIR}/extra_95_to_100pct.csv:chunk_add_4:Additional chunk 4"
)

for dataset in "${datasets[@]}"; do
    IFS=':' read -r csv_file suffix description <<< "$dataset"
    
    echo ""
    echo "=========================================="
    echo "Processing: ${description}"
    echo "=========================================="
    
    output_dir="${OUTPUT_BASE}/dclm_olmo_litgpt_${suffix}"
    
    echo "Shard list: ${csv_file}"
    echo "Output: ${output_dir}"
    echo ""
    
    python3 "${SCRIPT_DIR}/optimize_data_dclm.py" \
        --train_dir "${TRAIN_DIR}" \
        --output_base "${OUTPUT_BASE}" \
        --max_seq_length "${MAX_SEQ_LENGTH}" \
        --num_workers "${NUM_WORKERS}" \
        --chunk_bytes "${CHUNK_BYTES}" \
        --skip_val \
        --shard_list_csv "${csv_file}"
    
    # Rename output to match our naming scheme
    default_output="${OUTPUT_BASE}/dclm_olmo_litgpt_train"
    if [ -d "${default_output}" ] && [ "${default_output}" != "${output_dir}" ]; then
        mv "${default_output}" "${output_dir}"
        echo "Moved ${default_output} → ${output_dir}"
    fi
    
    echo "✅ Completed: ${description}"
done

echo ""
echo "=========================================="
echo "All datasets created successfully!"
echo "=========================================="
echo ""
echo "Dataset locations:"
echo "  Base chunk:        ${OUTPUT_BASE}/dclm_olmo_litgpt_chunk_base"
echo "  Additional chunk1: ${OUTPUT_BASE}/dclm_olmo_litgpt_chunk_add_1"
echo "  Additional chunk2: ${OUTPUT_BASE}/dclm_olmo_litgpt_chunk_add_2"
echo "  Additional chunk3: ${OUTPUT_BASE}/dclm_olmo_litgpt_chunk_add_3"
echo "  Additional chunk4: ${OUTPUT_BASE}/dclm_olmo_litgpt_chunk_add_4"
echo ""
echo "Compose training corpora by combining chunk_base with one or more chunk_add_* datasets as needed."
echo ""

