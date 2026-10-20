#!/usr/bin/env bash
# End-to-end smoke run for original_smoke_beauty on iota (single A800-80GB).
# All paths are explicit; nothing reads the repo's bundled checkpoints/mappings.
#
# Usage:
#   bash scripts/run_smoke_iota.sh <REPO_DIR> <RAW_DATA_DIR> <ST_MODEL_DIR> <RUNS_BASE>
#     REPO_DIR     TIGER checkout on iota (repro/original-smoke branch)
#     RAW_DATA_DIR dir with reviews_Beauty_5.json.gz and meta_Beauty.json.gz
#     ST_MODEL_DIR local snapshot dir of sentence-transformers/sentence-t5-base
#     RUNS_BASE    base dir for run outputs (e.g. ~/tiger_runs)
set -euo pipefail

REPO_DIR="${1:?REPO_DIR required}"
RAW_DIR="${2:?RAW_DATA_DIR required}"
MODEL_DIR="${3:?ST_MODEL_DIR required}"
RUNS_BASE="${4:?RUNS_BASE required}"

STAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_DIR="${RUNS_BASE}/original_smoke_beauty_${STAMP}"
LOGS="${RUN_DIR}/06_logs"
mkdir -p "${RUN_DIR}"/{01_encode,02_rqvae,03_codes,04_rec,05_reload,06_logs} "${LOGS}"

PY="${PYTHON_BIN:-python}"
cd "${REPO_DIR}"

echo "=== original_smoke_beauty run ${STAMP} ===" | tee "${LOGS}/run_main.log"
{
  echo "repo_dir: ${REPO_DIR}"
  echo "git_head: $(git rev-parse HEAD)"
  echo "git_branch: $(git branch --show-current)"
  echo "run_dir: ${RUN_DIR}"
  echo "pid: $$"
  echo "started_utc: ${STAMP}"
  echo "cmdline: $0 $*"
} > "${RUN_DIR}/run_info.txt"

nvidia-smi > "${LOGS}/nvidia_smi_start.txt" 2>&1 || true

echo "[stage 1/7] preprocess" | tee -a "${LOGS}/run_main.log"
${PY} data/preprocess.py \
  --raw_dir "${RAW_DIR}" \
  --out_dir "${RUN_DIR}/data" \
  2>&1 | tee "${LOGS}/01_preprocess.log"

echo "[stage 2/7] text encoding" | tee -a "${LOGS}/run_main.log"
${PY} data/encode_text.py \
  --meta "${RAW_DIR}/meta_Beauty.json.gz" \
  --item_mapping "${RUN_DIR}/data/item_mapping.npy" \
  --model_dir "${MODEL_DIR}" \
  --out_file "${RUN_DIR}/01_encode/item_emb.parquet" \
  --device cuda:0 \
  --batch_size 256 \
  --revision "${ST_MODEL_REVISION:-unknown}" \
  2>&1 | tee "${LOGS}/02_encode.log"

echo "[stage 3/7] RQ-VAE training (100 epochs, seed 2024, all other defaults)" | tee -a "${LOGS}/run_main.log"
${PY} rqvae/main.py \
  --epochs 100 \
  --data_path "${RUN_DIR}/01_encode/item_emb.parquet" \
  --ckpt_dir "${RUN_DIR}/02_rqvae" \
  --device cuda:0 \
  2>&1 | tee "${LOGS}/03_rqvae.log"

RQ_CKPT_DIR="$(ls -d "${RUN_DIR}"/02_rqvae/*/ | head -1)"
echo "rq-vae ckpt dir: ${RQ_CKPT_DIR}" | tee -a "${LOGS}/run_main.log"

echo "[stage 4/7] semantic ID generation" | tee -a "${LOGS}/run_main.log"
${PY} rqvae/generate_code.py \
  --ckpt_path "${RQ_CKPT_DIR}/best_collision_model.pth" \
  --data_path "${RUN_DIR}/01_encode/item_emb.parquet" \
  --output_file "${RUN_DIR}/03_codes/codes.npy" \
  --device cuda:0 \
  2>&1 | tee "${LOGS}/04_generate_code.log"

echo "[stage 5/7] recommendation model training (2 epochs, seed 2025)" | tee -a "${LOGS}/run_main.log"
${PY} model/main.py \
  --num_epochs 2 \
  --device cuda:0 \
  --dataset_path "${RUN_DIR}/data" \
  --code_path "${RUN_DIR}/03_codes/codes.npy" \
  --log_path "${LOGS}/05_rec_train.log" \
  --save_path "${RUN_DIR}/04_rec/tiger_best.pth" \
  2>&1 | tee "${LOGS}/05_rec_train_stdout.log"

echo "[stage 6/7] checkpoint selection + full test evaluation" | tee -a "${LOGS}/run_main.log"
BEST_NDCG="$(${PY} -c "import json;print(json.load(open('${RUN_DIR}/04_rec/rec_train_summary.json'))['best_ndcg20_batch_mean'])")"
if "${PY}" -c "import sys; sys.exit(0 if float('${BEST_NDCG}') > 0 else 1)"; then
  SELECTED="${RUN_DIR}/04_rec/tiger_best.pth"
  echo "selected best checkpoint (valid NDCG@20 batch-mean = ${BEST_NDCG})" | tee -a "${LOGS}/run_main.log"
else
  SELECTED="${RUN_DIR}/04_rec/tiger_best_last_smoke.pth"
  echo "no positive validation NDCG@20 (${BEST_NDCG}); using last_smoke for the chain check" | tee -a "${LOGS}/run_main.log"
fi
echo "${SELECTED}" > "${RUN_DIR}/05_reload/selected_ckpt.txt"

${PY} model/evaluate.py \
  --save_path "${SELECTED}" \
  --code_path "${RUN_DIR}/03_codes/codes.npy" \
  --dataset_path "${RUN_DIR}/data" \
  --device cuda:0 \
  --split test \
  --out_dir "${RUN_DIR}/05_reload" \
  2>&1 | tee "${LOGS}/06_test_eval.log"

echo "[stage 7/7] reload verification (new process)" | tee -a "${LOGS}/run_main.log"
${PY} model/evaluate.py \
  --save_path "${SELECTED}" \
  --code_path "${RUN_DIR}/03_codes/codes.npy" \
  --dataset_path "${RUN_DIR}/data" \
  --device cuda:0 \
  --reload_check \
  --out_dir "${RUN_DIR}/05_reload/reload" \
  2>&1 | tee "${LOGS}/07_reload_check.log"

sha256sum "${RUN_DIR}/data/"*.parquet \
          "${RUN_DIR}/01_encode/item_emb.parquet" \
          "${RUN_DIR}/03_codes/codes.npy" \
          "${SELECTED}" \
  > "${RUN_DIR}/artifact_sha256.txt"

nvidia-smi > "${LOGS}/nvidia_smi_end.txt" 2>&1 || true
echo "=== run complete: ${RUN_DIR} ===" | tee -a "${LOGS}/run_main.log"
