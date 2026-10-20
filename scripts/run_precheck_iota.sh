#!/usr/bin/env bash
# End-to-end precheck run for paper_align_precheck_beauty on iota (A800-80GB).
# Paper-aligned candidate on branch repro/paper-aligned-beauty. Short training
# budgets only (RQ-VAE 100 epochs, rec 1000 steps); this validates gradients,
# data flow and run stability and measures timing for the formal-run budget.
# NOT the formal experiment: long training starts fresh from another run dir.
#
# Usage:
#   bash scripts/run_precheck_iota.sh <REPO_DIR> <RAW_DATA_DIR> <ST_MODEL_DIR> <RUNS_BASE>
#     REPO_DIR     TIGER checkout on iota (repro/paper-aligned-beauty branch)
#     RAW_DATA_DIR dir with reviews_Beauty_5.json.gz and meta_Beauty.json.gz
#     ST_MODEL_DIR local snapshot dir of sentence-transformers/sentence-t5-base
#     RUNS_BASE    base dir for run outputs (e.g. ~/tiger_runs)
set -euo pipefail

REPO_DIR="${1:?REPO_DIR required}"
RAW_DIR="${2:?RAW_DATA_DIR required}"
MODEL_DIR="${3:?ST_MODEL_DIR required}"
RUNS_BASE="${4:?RUNS_BASE required}"

RQ_EPOCHS="${RQ_EPOCHS:-100}"
REC_STEPS="${REC_STEPS:-1000}"

STAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_DIR="${RUNS_BASE}/paper_align_precheck_beauty_${STAMP}"
LOGS="${RUN_DIR}/06_logs"
mkdir -p "${RUN_DIR}"/{01_encode,02_rqvae,03_codes,04_rec,05_reload,06_logs} "${LOGS}"

PY="${PYTHON_BIN:-python}"
cd "${REPO_DIR}"

echo "=== paper_align_precheck_beauty run ${STAMP} ===" | tee "${LOGS}/run_main.log"
{
  echo "repo_dir: ${REPO_DIR}"
  echo "git_head: $(git rev-parse HEAD)"
  echo "git_branch: $(git branch --show-current)"
  echo "run_dir: ${RUN_DIR}"
  echo "pid: $$"
  echo "started_utc: ${STAMP}"
  echo "cmdline: $0 $*"
  echo "rq_epochs: ${RQ_EPOCHS}"
  echo "rec_steps: ${REC_STEPS}"
  echo "python: $(${PY} -c 'import sys; print(sys.version)')"
  echo "torch: $(${PY} -c 'import torch; print(torch.__version__, "cuda_available", torch.cuda.is_available())')"
} > "${RUN_DIR}/run_info.txt"

nvidia-smi > "${LOGS}/nvidia_smi_start.txt" 2>&1 || true

echo "[stage 1/7] preprocess" | tee -a "${LOGS}/run_main.log"
${PY} data/preprocess.py \
  --raw_dir "${RAW_DIR}" \
  --out_dir "${RUN_DIR}/data" \
  2>&1 | tee "${LOGS}/01_preprocess.log"

echo "[stage 2/7] text encoding (paper features: title/price/brand/categories)" | tee -a "${LOGS}/run_main.log"
${PY} data/encode_text.py \
  --meta "${RAW_DIR}/meta_Beauty.json.gz" \
  --item_mapping "${RUN_DIR}/data/item_mapping.npy" \
  --model_dir "${MODEL_DIR}" \
  --out_file "${RUN_DIR}/01_encode/item_emb.parquet" \
  --device cuda:0 \
  --batch_size 256 \
  --revision "${ST_MODEL_REVISION:-unknown}" \
  2>&1 | tee "${LOGS}/02_encode.log"

echo "[stage 3/7] RQ-VAE training (${RQ_EPOCHS} epochs, seed 2024, paper-aligned defaults)" | tee -a "${LOGS}/run_main.log"
${PY} rqvae/main.py \
  --epochs "${RQ_EPOCHS}" \
  --data_path "${RUN_DIR}/01_encode/item_emb.parquet" \
  --ckpt_dir "${RUN_DIR}/02_rqvae" \
  --device cuda:0 \
  2>&1 | tee "${LOGS}/03_rqvae.log"

RQ_CKPT_DIR="$(ls -d "${RUN_DIR}"/02_rqvae/*/ | head -1)"
echo "rq-vae ckpt dir: ${RQ_CKPT_DIR}" | tee -a "${LOGS}/run_main.log"

echo "[stage 4/7] semantic ID generation (nearest-neighbor + 4th disambiguation digit)" | tee -a "${LOGS}/run_main.log"
${PY} rqvae/generate_code.py \
  --ckpt_path "${RQ_CKPT_DIR}/final_model.pth" \
  --data_path "${RUN_DIR}/01_encode/item_emb.parquet" \
  --output_file "${RUN_DIR}/03_codes/codes.npy" \
  --device cuda:0 \
  2>&1 | tee "${LOGS}/04_generate_code.log"

echo "[stage 5/7] rec model training (${REC_STEPS} steps, seed 2025, lr 0.01 inverse-sqrt after 10k)" | tee -a "${LOGS}/run_main.log"
${PY} model/main.py \
  --max_steps "${REC_STEPS}" \
  --device cuda:0 \
  --dataset_path "${RUN_DIR}/data" \
  --code_path "${RUN_DIR}/03_codes/codes.npy" \
  --user_mapping_path "${RUN_DIR}/data/user_mapping.npy" \
  --log_path "${LOGS}/05_rec_train.log" \
  --save_path "${RUN_DIR}/04_rec/tiger_final.pth" \
  2>&1 | tee "${LOGS}/05_rec_train_stdout.log"

echo "[stage 6/7] full valid + test evaluation" | tee -a "${LOGS}/run_main.log"
${PY} model/evaluate.py \
  --save_path "${RUN_DIR}/04_rec/tiger_final.pth" \
  --code_path "${RUN_DIR}/03_codes/codes.npy" \
  --user_map_path "${RUN_DIR}/04_rec/user_bucket_map.json" \
  --dataset_path "${RUN_DIR}/data" \
  --device cuda:0 \
  --split both \
  --out_dir "${RUN_DIR}/05_reload" \
  2>&1 | tee "${LOGS}/06_eval.log"

echo "[stage 7/7] reload verification (new process)" | tee -a "${LOGS}/run_main.log"
${PY} model/evaluate.py \
  --save_path "${RUN_DIR}/04_rec/tiger_final.pth" \
  --code_path "${RUN_DIR}/03_codes/codes.npy" \
  --user_map_path "${RUN_DIR}/04_rec/user_bucket_map.json" \
  --dataset_path "${RUN_DIR}/data" \
  --device cuda:0 \
  --reload_check \
  --out_dir "${RUN_DIR}/05_reload/reload" \
  2>&1 | tee "${LOGS}/07_reload_check.log"

sha256sum "${RUN_DIR}/data/"*.parquet "${RUN_DIR}/data/user_mapping.npy" \
          "${RUN_DIR}/01_encode/item_emb.parquet" \
          "${RUN_DIR}/03_codes/codes.npy" \
          "${RUN_DIR}/04_rec/user_bucket_map.json" \
          "${RUN_DIR}/04_rec/tiger_final.pth" \
  > "${RUN_DIR}/artifact_sha256.txt"

nvidia-smi > "${LOGS}/nvidia_smi_end.txt" 2>&1 || true
echo "=== run complete: ${RUN_DIR} ===" | tee -a "${LOGS}/run_main.log"
