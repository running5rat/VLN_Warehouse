"""Summarize saved Phase-1 experiments without altering model responses."""

import argparse
import html
import json
import os
from pathlib import Path
import numpy as np
from DynaNav.warehouse_loop.live_metrics import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--offline", type=Path, required=True)
    p.add_argument("--online", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=True)
    cases = json.loads((a.offline / "report.json").read_text())["cases"]
    timing = {}
    for field in (
        "image_encoding",
        "native_generation",
        "prefill_gpu",
        "decode_gpu",
        "reference_and_compressor",
    ):
        values = [
            r["stages"][field]
            for r in cases
            if r.get("stages") and r["stages"].get(field) is not None
        ]
        timing[field] = (
            dict(
                p50_s=float(np.percentile(values, 50)),
                p95_s=float(np.percentile(values, 95)),
            )
            if values
            else None
        )
    recursive = []
    backward = []
    for r in cases:
        if r["round"]:
            prior = next(
                x
                for x in cases
                if x["case"] == r["case"] and x["round"] == r["round"] - 1
            )
            recursive.append(
                dict(
                    case=r["case"],
                    round=r["round"],
                    actual_previous_route_reused=r["route_input"]
                    == prior.get("route_output"),
                )
            )
        if r["case"] == "reverse":
            folder = a.offline / f"reverse-{r['round']}"
            response = json.loads((folder / "response.json").read_text())
            parsed = response.get("parsed") or {}
            goal = (parsed.get("route") or {}).get("current_subgoal", {})
            points = parsed.get("anchors", [])
            row = dict(
                round=r["round"],
                fixture_remaining_m=5 - 2 * r["round"],
                task_understood=goal.get("action") == "relative_motion"
                and goal.get("direction") == "backward",
                waypoint_backward=bool(points and points[-1][0] < 0),
                execution="execution_not_evaluated",
            )
            if (folder / "tensors.npz").exists():
                arrays = np.load(folder / "tensors.npz")
                line = arrays["reference_xy"][arrays["reference_mask"]]
                if len(line):
                    row.update(
                        raw_decoder_endpoint_m=line[-1].tolist(),
                        decoder_endpoint_backward=bool(line[-1, 0] < 0),
                        decoder_endpoint_norm_m=float(np.linalg.norm(line[-1])),
                        endpoint_to_expected_backward_m=float(
                            np.linalg.norm(line[-1] - [-row["fixture_remaining_m"], 0])
                        ),
                    )
            backward.append(row)
    online = (
        json.loads((a.online / "report.json").read_text())["cases"]
        if (a.online / "report.json").exists()
        else []
    )
    summary = dict(
        offline=dict(
            requests=len(cases),
            format_valid=sum(r["format_valid"] for r in cases),
            route_committed=sum(r["route_committed"] for r in cases),
            reference_consistent=sum(
                r.get("reference_validation", {}).get("consistency_valid", False)
                for r in cases
            ),
            flow_finite=sum(r.get("flow_finite", False) for r in cases),
            recursive_checks=recursive,
            timing=timing,
            backward_diagnostics=backward,
        ),
        online=online,
        scope="Engineering interfaces and limited-budget trials. No trained navigation quality, success-rate or realtime acceptance claim.",
    )
    atomic_json(a.output / "summary.json", summary)
    sections = []
    for r in cases:
        folder = a.offline / f"{r['case']}-{r['round']}"
        relative = os.path.relpath(folder, a.output)
        sections.append(
            f'<h3>{r["case"]} / {r["round"]}</h3><pre>{html.escape(json.dumps(r,ensure_ascii=False,indent=2))}</pre>'
            f'<a href="{relative}/response.json">Response</a><img width="500" src="{relative}/map.png">'
        )
    for row in online:
        folder = Path(row["report"]).parent
        relative = os.path.relpath(folder, a.output)
        label = html.escape(" / ".join(row[k] for k in ("scene", "task", "backend")))
        sections.append(
            f'<h3>Isaac: {label}</h3><a href="{relative}/index.html">Viewer</a> · '
            f'<a href="{relative}/report.json">Report</a><br>'
            f'<img loading="lazy" width="480" src="{relative}/navigation/rgb.png">'
            f'<img loading="lazy" width="480" src="{relative}/navigation/map.png">'
        )
    (a.output / "index.html").write_text(
        '<meta charset="utf-8"><h1>Phase 1 validation</h1><p>Only actual model outputs; failed cases retained.</p><pre>'
        + html.escape(json.dumps(summary, ensure_ascii=False, indent=2))
        + "</pre>"
        + "".join(sections)
    )


if __name__ == "__main__":
    main()
