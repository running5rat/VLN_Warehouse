#!/usr/bin/env bash
# Separate, pinned Julia environment. Does not change Isaac or Python packages.
set -euo pipefail
ORACLE_REPO="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
ORACLE_ENV="${TICVLA_ALTRO_ENV:-/data/ticvla/envs/warehouse-oracle}"
ORACLE_TOOLS="${TICVLA_ORACLE_TOOLS:-/data/ticvla/tools}"
ORACLE_JULIA="${TICVLA_JULIA:-${ORACLE_TOOLS}/julia-1.10.10/bin/julia}"
ORACLE_LOCK="${ORACLE_REPO}/configs/dataset_preparation/altro"
if [[ ! -x "${ORACLE_JULIA}" ]]; then
  mkdir -p -- "${ORACLE_TOOLS}"
  "${TICVLA_ANNOTATION_PYTHON:-/data/ticvla/envs/warehouse-io/bin/python}" - "${ORACLE_LOCK}/julia-download-lock.json" "${ORACLE_TOOLS}" <<'PY'
import hashlib, json, pathlib, sys, urllib.request
lock = json.loads(pathlib.Path(sys.argv[1]).read_text())
out = pathlib.Path(sys.argv[2]) / 'julia-1.10.10.tar.gz'
if not out.exists():
    urllib.request.urlretrieve(lock['url'], out)
if hashlib.file_digest(out.open('rb'), 'sha256').hexdigest() != lock['sha256']:
    raise SystemExit('Julia archive checksum mismatch; refusing extraction')
PY
  tar -xzf "${ORACLE_TOOLS}/julia-1.10.10.tar.gz" -C "${ORACLE_TOOLS}"
fi
mkdir -p -- "${ORACLE_ENV}"
for ORACLE_FILE in Project.toml Manifest.toml; do
  if [[ -f "${ORACLE_ENV}/${ORACLE_FILE}" ]] && ! cmp -s "${ORACLE_LOCK}/${ORACLE_FILE}" "${ORACLE_ENV}/${ORACLE_FILE}"; then
    echo "Existing Oracle environment differs: ${ORACLE_ENV}/${ORACLE_FILE}; use a new TICVLA_ALTRO_ENV." >&2
    exit 1
  fi
  cp -- "${ORACLE_LOCK}/${ORACLE_FILE}" "${ORACLE_ENV}/${ORACLE_FILE}"
done
export JULIA_DEPOT_PATH="${ORACLE_ENV}/depot"
export JULIA_NUM_THREADS=2
exec "${ORACLE_JULIA}" --project="${ORACLE_ENV}" -e 'using Pkg; Pkg.instantiate(); using Altro; println("Pinned ALTRO environment ready")'
