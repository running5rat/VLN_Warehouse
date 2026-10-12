"""Wall-clock throughput and source-time sampling are deliberately separate."""

import json
from pathlib import Path
import numpy as np


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def complete_lines(stream):
    """Do not consume a partially committed JSONL tail."""
    while True:
        offset = stream.tell()
        line = stream.readline()
        if not line or not line.endswith("\n"):
            stream.seek(offset)
            return
        yield json.loads(line)


def rate_report(rows, target, start, end):
    times = np.array([r["wall"] for r in rows], dtype=float)
    service = [r["service_s"] for r in rows if "service_s" in r]
    window = max(end - start, 1e-9)
    intervals = np.diff(times)
    steady = (
        float((len(times) - 1) / (times[-1] - times[0]))
        if len(times) > 1 and times[-1] > times[0]
        else 0.0
    )
    return {
        "count": len(rows),
        "target_hz": target,
        "wall_hz_including_startup_and_drain": len(rows) / window,
        "completion_wall_hz": steady,
        "interval_p95_s": float(np.percentile(intervals, 95))
        if len(intervals)
        else None,
        "service_p50_s": float(np.median(service)) if service else None,
        "service_p95_s": float(np.percentile(service, 95)) if service else None,
        "meets_target_95pct": bool(len(rows) >= 10 and steady >= target * 0.95),
    }
