"""Create an evidence-only navigation comparison, without navigation quality claims."""

import argparse
from collections import Counter
import html
import json
import os
from pathlib import Path

import numpy as np

from ..mapping.semantic_frame import write_json


def lines(path):
    return (
        [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
        if path.exists()
        else []
    )


def summarize(root):
    result = json.loads((root / "report.json").read_text())
    metrics = lines(root / "navigation/metrics.jsonl")
    executed = lines(root / "navigation/executed-commands.jsonl")
    tracks = [x for x in metrics if x["module"] == "person_track"]
    visible = False
    lost = resumed = 0
    for row in tracks:
        if visible and not row["visible"]:
            lost += 1
        if not visible and row["visible"] and lost:
            resumed += 1
        visible = row["visible"]
    selections = [x for x in metrics if x["module"] == "candidate_selection"]
    rejected = Counter(
        score["reason"]
        for row in selections
        for score in row["scores"]
        if score["reason"]
    )
    source_rejected = Counter()
    for path in (root / "navigation/flow").glob("*/source-scores.json"):
        source_rejected.update(
            x["reason"] for x in json.loads(path.read_text())["scores"] if x["reason"]
        )
    manual = [x for x in metrics if x["module"] == "manual_binding_replay"]
    recovery = [x for x in manual if x.get("replayed", 0) > 0 and x["visible"]]
    vlm_modes = {}
    for worker in metrics:
        if (
            worker["module"] != "worker"
            or worker.get("kind") != "vlm"
            or worker.get("error")
        ):
            continue
        request = (
            root / "navigation/vlm" / f"{worker['request_id']:06d}" / "request.json"
        )
        if request.exists():
            mode = json.loads(request.read_text())["mode"]
            vlm_modes.setdefault(mode, []).append(worker["service_s"])
    vlm_timing = {
        mode: dict(
            count=len(values),
            p50=float(np.percentile(values, 50)),
            p95=float(np.percentile(values, 95)),
        )
        for mode, values in vlm_modes.items()
    }
    summary = dict(
        path=str(root),
        task=result["task_mode"],
        backend=result["execution_backend"],
        geometry_frames=result["geometry_frames"],
        geometry_complete=result.get("geometry_complete", False),
        distance_m=result.get("capture", {}).get("distance_m"),
        executed_translation_commands=result.get("executed_translation_commands", 0),
        observed_movement=result.get("observed_movement", False),
        valid_text=result["valid_text"],
        valid_prefill=result["valid_prefill"],
        flow_batches=result["flow_batches"],
        model_interface_pass=result["model_interface_pass"],
        model_motion_pass=result["model_motion_pass"],
        completed_hz=result.get("completed_hz"),
        geometry_age_p95_s=result.get("geometry_age_p95_s"),
        queue_replacements=result.get("queue_replacements"),
        snapshot_rejections=Counter(x["reason"] for x in result["rejected_snapshots"]),
        command_reasons=Counter(x["reason"] for x in executed),
        failures=result.get("failures", []),
        tracking_loss_transitions=lost,
        tracking_resume_transitions=resumed,
        manual_reconfirmations=manual,
        text_after_reconfirmation=any(
            x["module"] == "snapshot_committed"
            and x["mode"] == "TEXT"
            and x["monotonic"] > event["monotonic"]
            for event in recovery
            for x in metrics
        ),
        prefill_after_reconfirmation=any(
            x["module"] == "snapshot_committed"
            and x["mode"] == "PREFILL_FOLLOW"
            and x["monotonic"] > event["monotonic"]
            for event in recovery
            for x in metrics
        ),
        binding_ids=sorted({x["binding_id"] for x in tracks if x.get("binding_id")}),
        vlm_mode_service_seconds=vlm_timing,
        service_seconds={k: v for k, v in result.items() if k.endswith("_service_s")},
        selection_events=len(selections),
        candidate_rejection_reasons=rejected,
        source_frame_rejection_reasons=source_rejected,
        quality_evaluated=False,
    )
    return summary, executed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", type=Path, required=True)
    parser.add_argument("--offline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, cards = [], []
    for run in args.runs:
        run = run.resolve()
        row, executed = summarize(run)
        rows.append(row)
        figure, axis = plt.subplots(figsize=(5, 4))
        if executed:
            xy = np.asarray([x["pose"] for x in executed])[:, :2, 3]
            axis.plot(xy[:, 0], xy[:, 1], label="Actual odometry")
            axis.scatter(*xy[0], marker="o", label="Start")
            axis.scatter(*xy[-1], marker="x", label="End")
        axis.set(xlabel="odom x (m)", ylabel="odom y (m)", title=run.name)
        axis.axis("equal")
        axis.grid(True)
        axis.legend()
        figure.tight_layout()
        figure.savefig(root / (run.name + "-trajectory.png"), dpi=130)
        plt.close(figure)
        link = html.escape(os.path.relpath(run, root))
        cells = "".join(
            f"<td>{html.escape(str(row[k]))}</td>"
            for k in (
                "task",
                "backend",
                "geometry_frames",
                "valid_text",
                "valid_prefill",
                "flow_batches",
                "observed_movement",
            )
        )
        cards.append(
            f'<tr>{cells}<td><a href="{link}/index.html">viewer</a> · <a href="{link}/report.json">report</a></td></tr>'
        )
    offline = json.loads((args.offline / "report.json").read_text())
    write_json(
        root / "report.json",
        dict(
            online=rows,
            offline=offline,
            acceptance_scope="Interfaces and motion feedback only; no navigation quality or deployment real-time acceptance.",
            heads_quality="Fixed-seed initialization; untrained reference, compressor and nine-mode Flow.",
        ),
    )
    images = "".join(
        f'<figure><img width="470" src="{html.escape(Path(r["path"]).name)}-trajectory.png"><figcaption>{html.escape(Path(r["path"]).name)} — odometry drift alone does not count as controlled movement.</figcaption></figure>'
        for r in rows
    )
    (root / "index.html").write_text(
        """<!doctype html><meta charset="utf-8"><title>v0.6 navigation validation</title>
<style>body{font:16px system-ui;max-width:1450px;margin:30px auto;padding:0 20px}td,th{padding:9px;border-bottom:1px solid #ccc}figure{display:inline-block;margin:10px}figcaption{max-width:470px;font-size:13px}img{max-width:100%}</style>
<h1>v0.6 navigation validation</h1><p>Real model interfaces and engineering motion are reported separately. All new heads are untrained. Geometry-baseline movement is not model-driven movement. No navigation quality or deployment real-time claim.</p>
<p><a href="report.json">Full measured report</a></p><table><tr><th>Task</th><th>Control source</th><th>Geometry frames</th><th>TEXT commits</th><th>PREFILL commits</th><th>Flow batches</th><th>Controlled translation</th><th>Artifacts</th></tr>"""
        + "".join(cards)
        + "</table>"
        + images
    )


if __name__ == "__main__":
    main()
