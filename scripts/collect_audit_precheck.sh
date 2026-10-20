#!/usr/bin/env bash
# Assemble the <50MB audit pack for paper_align_precheck_beauty.
# Usage: bash scripts/collect_audit_precheck.sh <REPO_DIR> <RUN_DIR> <BASE_SHA> <OUT_TARBALL>
# Large artifacts (checkpoints, embeddings, codes, raw data) are NOT included;
# only their hashes (already in RUN_DIR/artifact_sha256.txt).
set -euo pipefail

REPO_DIR="${1:?}"
RUN_DIR="${2:?}"
BASE_SHA="${3:?}"
OUT_TARBALL="${4:?}"

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
AUDIT="${WORK}/audit_paper_align_precheck_beauty"
mkdir -p "${AUDIT}"/{logs,diffs,env,configs,artifacts}

cd "${REPO_DIR}"
git rev-parse HEAD > "${AUDIT}/git_head.txt"
git branch --show-current > "${AUDIT}/git_branch.txt"
git log --oneline "${BASE_SHA}..HEAD" > "${AUDIT}/commits.txt" 2>/dev/null || true
git diff "${BASE_SHA}..HEAD" > "${AUDIT}/diffs/changes_vs_frozen_smoke.diff"
git diff --stat "${BASE_SHA}..HEAD" > "${AUDIT}/diffs/changes_vs_frozen_smoke.stat"

cp requirements.txt "${AUDIT}/env/requirements_original.txt" 2>/dev/null || true
"${PYTHON_BIN:-python}" -m pip freeze > "${AUDIT}/env/pip_freeze_precheck_env.txt"

for f in run_info.txt artifact_sha256.txt; do
  [ -f "${RUN_DIR}/${f}" ] && cp "${RUN_DIR}/${f}" "${AUDIT}/artifacts/"
done
# run-level configs and reports (json/small text only, incl. eval_report.json,
# train_stats.json, rec_train_summary.json, encode_report.json, dataset_stats.json)
find "${RUN_DIR}" -maxdepth 4 -name "*.json" -size -2M \
  | while read -r j; do cp "${j}" "${AUDIT}/configs/$(echo "${j#${RUN_DIR}/}" | tr '/' '_')"; done

# logs: truncate each to its last 4000 lines to stay small
find "${RUN_DIR}/06_logs" -type f 2>/dev/null | while read -r l; do
  tail -n 4000 "${l}" > "${AUDIT}/logs/$(basename "${l}").tail"
done

[ -f "${RUN_DIR}/data/dataset_stats.json" ] && cp "${RUN_DIR}/data/dataset_stats.json" "${AUDIT}/artifacts/"

# per-sample boolean hit matrices (valid + test) so metrics can be recomputed
# independently, plus the first-10 review parquets
find "${RUN_DIR}/05_reload" -name "*_per_sample.npz" \
  | while read -r p; do cp "${p}" "${AUDIT}/artifacts/"; done
find "${RUN_DIR}/05_reload" -name "*head10.parquet" -size -2M \
  | while read -r p; do cp "${p}" "${AUDIT}/artifacts/"; done
# user bucket map binding artifact (small json; hash also in ckpt)
[ -f "${RUN_DIR}/04_rec/user_bucket_map.json" ] && cp "${RUN_DIR}/04_rec/user_bucket_map.json" "${AUDIT}/artifacts/"

tar -czf "${OUT_TARBALL}" -C "${WORK}" audit_paper_align_precheck_beauty
ls -lh "${OUT_TARBALL}"
echo "audit pack: ${OUT_TARBALL}"
