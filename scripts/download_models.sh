#!/bin/bash
# Download model checkpoints from GCS into external/litgpt/<model_name>/
#
# Usage: bash scripts/download_models.sh [model_name ...]
#   With no args, prints usage and exits.
#   With model names (e.g. olmo2_30m_wsd_1x), downloads each from GCS into external/litgpt/<name>/.
#
# Env (optional):
#   GCS_BUCKET          default: dclm_olmo
#   GCS_MODELS_PREFIX   default: models/pct   (so base path is gs://dclm_olmo/models/pct/)
#   SKIP_DOWNLOAD       not used here; run_all_evals.sh sets it to skip calling this script
set -e

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LITGPT_DIR="${ROOT_DIR}/external/litgpt"

GCS_BUCKET="${GCS_BUCKET:-dclm_olmo}"
GCS_MODELS_PREFIX="${GCS_MODELS_PREFIX:-models/olmo2_1b_wsd}"
GCS_BASE="gs://${GCS_BUCKET}/${GCS_MODELS_PREFIX}"

if [[ $# -eq 0 ]]; then
    echo "usage: bash scripts/download_models.sh <model_name> [model_name ...]"
    echo "example: bash scripts/download_models.sh olmo2_30m_wsd_1x olmo2_136m_wsd_1x"
    echo ""
    echo "Downloads from ${GCS_BASE}/<model_name>/ into ${LITGPT_DIR}/<model_name>/"
    echo "Override: GCS_BUCKET=my-bucket GCS_MODELS_PREFIX=path bash scripts/download_models.sh ..."
    exit 1
fi

mkdir -p "$LITGPT_DIR"

for name in "$@"; do
    dest="${LITGPT_DIR}/${name}"
    if [[ -d "$dest" ]] && [[ -f "${dest}/model.pth" || -f "${dest}/lit_model.pth" || -f "${dest}/model.safetensors" ]]; then
        echo "[SKIP] ${name} already present at ${dest}"
        continue
    fi

    src="${GCS_BASE}/${name}"
    echo "[DOWNLOAD] ${name} from ${src}/ -> ${dest}/"
    mkdir -p "$dest"
    if ! gsutil -m cp -r "${src}/*" "${dest}/" 2>/dev/null; then
        echo "[WARN] gsutil failed for ${name}. Check that ${src}/ exists and you have access."
        echo "       Or put checkpoint manually in ${dest}/ and run with SKIP_DOWNLOAD=1."
        exit 1
    fi
done

echo "Done."
