#!/usr/bin/env bash
# Assemble the <50MB audit pack for original_smoke_beauty.
# Usage: bash scripts/collect_audit.sh <REPO_DIR> <RUN_DIR> <BASE_SHA> <OUT_TARBALL>
# Large artifacts (checkpoints, embeddings, raw data) are NOT included; only
# their hashes (already in RUN_DIR/artifact_sha256.txt).
set -euo pipefail

REPO_DIR="${1:?}"
RUN_DIR="${2:?}"
BASE_SHA="${3:?}"
OUT_TARBALL="${4:?}"

WORK="$(mktemp -d)"
trap 'rm -rf "${WORK}"' EXIT
AUDIT="${WORK}/audit_original_smoke_beauty"
mkdir -p "${AUDIT}"/{logs,diffs,env,configs,artifacts}

cd "${REPO_DIR}"
git rev-parse HEAD > "${AUDIT}/git_head.txt"
git branch --show-current > "${AUDIT}/git_branch.txt"
git log --oneline "${BASE_SHA}..HEAD" > "${AUDIT}/commits.txt" 2>/dev/null || true
git diff "${BASE_SHA}..HEAD" > "${AUDIT}/diffs/changes_vs_base.diff"
git diff --stat "${BASE_SHA}..HEAD" > "${AUDIT}/diffs/changes_vs_base.stat"

cp envs/requirements-smoke.txt "${AUDIT}/env/" 2>/dev/null || true
cp requirements.txt "${AUDIT}/env/requirements_original.txt"
[ -f "${RUN_DIR}/report.md" ] && cp "${RUN_DIR}/report.md" "${AUDIT}/report.md"
find "${RUN_DIR}/env_freeze" -type f 2>/dev/null | while read -r f; do cp "${f}" "${AUDIT}/env/"; done

# run-level configs and reports (json/small text only)
for f in run_info.txt artifact_sha256.txt; do
  [ -f "${RUN_DIR}/${f}" ] && cp "${RUN_DIR}/${f}" "${AUDIT}/artifacts/"
done
find "${RUN_DIR}" -maxdepth 3 -name "*.json" -size -2M \
  | while read -r j; do cp "${j}" "${AUDIT}/configs/$(echo "${j#${RUN_DIR}/}" | tr '/' '_')"; done

# logs: truncate each to its last 4000 lines to stay small
find "${RUN_DIR}/06_logs" -type f 2>/dev/null | while read -r l; do
  tail -n 4000 "${l}" > "${AUDIT}/logs/$(basename "${l}").tail"
done
[ -f "${RUN_DIR}/data/dataset_stats.json" ] && cp "${RUN_DIR}/data/dataset_stats.json" "${AUDIT}/artifacts/"

# a couple of per-sample review files for spot checks
find "${RUN_DIR}/05_reload" -name "*head10.parquet" -size -2M \
  | while read -r p; do cp "${p}" "${AUDIT}/artifacts/"; done
# full per-sample boolean hit matrix so metrics can be recomputed independently
[ -f "${RUN_DIR}/05_reload/test_per_sample.npz" ] && cp "${RUN_DIR}/05_reload/test_per_sample.npz" "${AUDIT}/artifacts/"

tar -czf "${OUT_TARBALL}" -C "${WORK}" audit_original_smoke_beauty
ls -lh "${OUT_TARBALL}"
echo "audit pack: ${OUT_TARBALL}"
