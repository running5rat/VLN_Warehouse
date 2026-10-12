#!/usr/bin/env bash
set -euo pipefail

OUTPUT_ROOT="${TICVLA_OUTPUT_DIR:-${HOME}/ticvla_outputs}"
LOGS_ROOT="${OUTPUT_ROOT}/dynanav/logs"
FPS="${TICVLA_VIDEO_FPS:-10}"

usage() {
  echo "Usage: $0 [RUN_ID|RUN_DIRECTORY]" >&2
  echo "If omitted, the newest run under ${LOGS_ROOT} is used." >&2
}

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "Error: ffmpeg is required to generate MP4 files." >&2
  exit 1
fi

if [[ $# -gt 1 ]]; then
  usage
  exit 2
fi

if [[ $# -eq 1 ]]; then
  if [[ -d "$1" ]]; then
    RUN_DIR="$(cd "$1" && pwd)"
  else
    RUN_DIR="${LOGS_ROOT}/$1"
  fi
else
  RUN_DIR="$(find "${LOGS_ROOT}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-)"
fi

if [[ -z "${RUN_DIR:-}" || ! -d "${RUN_DIR}" ]]; then
  echo "Error: benchmark run directory does not exist: ${RUN_DIR:-<none>}" >&2
  exit 1
fi

mapfile -d '' FRAME_DIRS < <(
  find "${RUN_DIR}" -type f -name 'tp_frame_*.jpg' -printf '%h\0' \
    | sort -zu
)

if [[ ${#FRAME_DIRS[@]} -eq 0 ]]; then
  echo "Error: no tp_frame_*.jpg sequences found under ${RUN_DIR}" >&2
  exit 1
fi

generated=0
for frame_dir in "${FRAME_DIRS[@]}"; do
  episode_dir="$(dirname "${frame_dir}")"
  episode_name="$(basename "${episode_dir}")"
  output_file="${episode_dir}/${episode_name}.mp4"
  frame_count="$(find "${frame_dir}" -maxdepth 1 -type f -name 'tp_frame_*.jpg' | wc -l)"

  echo "Encoding ${episode_name}: ${frame_count} frames at ${FPS} FPS"
  ffmpeg -hide_banner -loglevel error -y \
    -framerate "${FPS}" \
    -pattern_type glob \
    -i "${frame_dir}/tp_frame_*.jpg" \
    -c:v libx264 \
    -preset medium \
    -crf 20 \
    -pix_fmt yuv420p \
    -movflags +faststart \
    "${output_file}"
  echo "Saved: ${output_file}"
  generated=$((generated + 1))
done

echo "Generated ${generated} episode video(s) from ${RUN_DIR}."
