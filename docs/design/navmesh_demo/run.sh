#!/usr/bin/env bash
set -euo pipefail
DEMO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${DEMO_DIR}/../../.." && pwd)"
NAV_PYTHON="${NAVMESH_PYTHON:-/root/miniconda3/bin/python}"
NAV_SOURCE="${NAVMESH_RECAST_SOURCE:-/data/ticvla/tools/recast-v1.6.0}"
NAV_OUTPUT="/data/ticvla/outputs/navmesh/warehouse-$(date +%Y%m%d-%H%M%S)"
NAV_PUBLISH="${REPO_ROOT}/demo/navmesh/latest"
NAV_SKIP=0
while (($#)); do
  case "$1" in
    --output) NAV_OUTPUT="$2"; shift 2;;
    --publish) NAV_PUBLISH="$2"; shift 2;;
    --skip-export) NAV_SKIP=1; shift;;
    *) echo "Unknown option: $1" >&2; exit 2;;
  esac
done
mkdir -p -- "$NAV_SOURCE" "$NAV_OUTPUT" "$NAV_PUBLISH"
NAV_OUTPUT="$(realpath -- "$NAV_OUTPUT")"
NAV_PUBLISH="$(realpath -- "$NAV_PUBLISH")"
"$NAV_PYTHON" - "$DEMO_DIR/source-lock.json" "$NAV_SOURCE/source.tar.gz" <<'PY'
import hashlib,json,sys
from pathlib import Path
lock=json.loads(Path(sys.argv[1]).read_text());archive=Path(sys.argv[2])
if not archive.exists():
    import requests
    session=requests.Session();session.trust_env=False
    response=session.get(lock['source'],timeout=(15,60));response.raise_for_status()
    archive.write_bytes(response.content)
assert hashlib.sha256(archive.read_bytes()).hexdigest()==lock['sha256'],'Recast archive hash mismatch'
PY
tar -xzf "$NAV_SOURCE/source.tar.gz" -C "$NAV_SOURCE" --strip-components=1
cmake -S "$DEMO_DIR" -B "$NAV_SOURCE/warehouse-build" -DRECAST_SOURCE="$NAV_SOURCE" -DCMAKE_BUILD_TYPE=Release > "$NAV_OUTPUT/cmake.log" 2>&1
cmake --build "$NAV_SOURCE/warehouse-build" -j4 >> "$NAV_OUTPUT/cmake.log" 2>&1
if (( ! NAV_SKIP )); then
  if [[ -f /.dockerenv ]]; then
    # The existing Isaac bridge runs on the host, avoiding container glibc mismatch.
    chown "${NAVMESH_HOST_UID:-1000}:${NAVMESH_HOST_GID:-1000}" "$NAV_OUTPUT"
    NAV_HOST_SCRIPT="${TICVLA_HOST_REPO:-/home/xinglinlu/autodriving_ws/docker/VLN/TIC-VLA}/docs/design/navmesh_demo/export_scene.py"
    timeout 180 nsenter -t 1 -m -- /usr/sbin/runuser -u "${TICVLA_HOST_USER:-xinglinlu}" -- \
      /usr/bin/env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
      -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONHOME -u PYTHONEXE -u PYTHONPATH -u LD_LIBRARY_PATH -u LD_PRELOAD \
      /data/isaacsim/python.sh "$NAV_HOST_SCRIPT" --output "$NAV_OUTPUT" > "$NAV_OUTPUT/export.log" 2>&1
  else
    timeout 180 env -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u PYTHONHOME -u PYTHONEXE -u PYTHONPATH -u LD_LIBRARY_PATH -u LD_PRELOAD \
      /data/isaacsim/python.sh "$DEMO_DIR/export_scene.py" --output "$NAV_OUTPUT" > "$NAV_OUTPUT/export.log" 2>&1
  fi
fi
test -f "$NAV_OUTPUT/scene.bin"
"$NAV_PYTHON" "$DEMO_DIR/prepare_queries.py" --output "$NAV_OUTPUT"
NAV_BIN="$NAV_SOURCE/warehouse-build/build_navmesh"
"$NAV_BIN" "$NAV_OUTPUT/fixture/scene.bin" "$NAV_OUTPUT/fixture" 0.35 "$NAV_OUTPUT/fixture/queries.txt"
"$NAV_BIN" "$NAV_OUTPUT/scene.bin" "$NAV_OUTPUT/radius-035" 0.35 "$NAV_OUTPUT/queries.txt"
"$NAV_BIN" "$NAV_OUTPUT/scene.bin" "$NAV_OUTPUT/radius-000" 0 "$NAV_OUTPUT/queries.txt"
"$NAV_PYTHON" "$DEMO_DIR/render_report.py" --output "$NAV_OUTPUT" --publish "$NAV_PUBLISH" > "$NAV_OUTPUT/render.log"
"$NAV_PYTHON" - "$NAV_OUTPUT" "$NAV_PUBLISH" "$DEMO_DIR" "$NAV_SOURCE" <<'PY'
import json,hashlib,platform,subprocess,sys
from pathlib import Path
import matplotlib,numpy
root,pub,code,source=map(Path,sys.argv[1:])
lock={'python':sys.version,'numpy':numpy.__version__,'matplotlib':matplotlib.__version__,
      'compiler':subprocess.check_output(['g++','--version'],text=True).splitlines()[0],
      'cmake':subprocess.check_output(['cmake','--version'],text=True).splitlines()[0],
      'recast':json.loads((code/'source-lock.json').read_text()),
      'code_sha256':{f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in code.iterdir() if f.suffix in ('.py','.cpp','.sh')},
      'binary_sha256':hashlib.sha256((source/'warehouse-build/build_navmesh').read_bytes()).hexdigest()}
for dest in [root,pub]:(dest/'environment-lock.json').write_text(json.dumps(lock,indent=2)+'\n')
report=json.loads((root/'report.json').read_text())
print(json.dumps(report['checks'],indent=2));print('Report:',pub/'index.html')
if not all(report['checks'].values()):raise SystemExit('One or more checks failed; see report')
PY
