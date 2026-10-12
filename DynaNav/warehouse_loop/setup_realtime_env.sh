#!/usr/bin/env bash
# Extend the installed environments without upgrading Isaac or either old ML environment.
set -euo pipefail
RT_SETUP_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RT_REPO="$(cd -- "$RT_SETUP_DIR/../.." && pwd)"
RT_ENV="${TICVLA_REALTIME_ENV:-/data/ticvla/envs/warehouse-realtime}"
if [[ ! -x "$RT_ENV/bin/python" ]]; then
  /root/miniconda3/envs/tic-vla/bin/python -m venv "$RT_ENV"
fi
"$RT_ENV/bin/python" - "$RT_ENV" <<'PY'
import sys
from pathlib import Path
assert sys.version_info[:2] == (3, 11), "Expected Python 3.11 for inherited binary packages"
paths = [Path(p) / "lib/python3.11/site-packages" for p in (
    "/data/ticvla/envs/landmark-perception", "/data/ticvla/envs/warehouse-io",
    "/root/miniconda3/envs/tic-vla")]
assert all(p.is_dir() for p in paths), "Required baseline ML environments are missing"
(Path(sys.argv[1]) / "lib/python3.11/site-packages/inherited.pth").write_text(
    "\n".join(map(str, paths)) + "\n")
PY
"$RT_ENV/bin/python" -m pip install -r "$RT_REPO/requirements-warehouse-realtime.txt"
"$RT_ENV/bin/python" -m pip freeze > "$RT_ENV/environment-lock.txt"
echo "Realtime Python: $RT_ENV/bin/python"
