"""Run two Isaac circuits only after the held-out backend selection gate passes."""

import argparse
import json
from pathlib import Path
import shutil
import subprocess
from ..mapping import MapStore, SemanticEvent
from ..mapping.semantic_frame import consume_categories, write_json


def run(root, render_hz=30):
    comparison = json.loads((root / "comparison.json").read_text())
    if not all(v["complete"] for v in comparison["backends"].values()):
        raise ValueError("Held-out evaluation is incomplete")
    backend = comparison["winner"]
    prefix = "live-" if render_hz == 30 else f"live-render{render_hz}-"
    report_path = root / (
        "live-validation.json"
        if render_hz == 30
        else f"live-validation-render{render_hz}.json"
    )
    report = dict(backend=backend, runs=[], status="not_eligible", render_hz=render_hz)
    if backend is not None:
        if (
            backend not in ("yoloe-ppocr", "florence2")
            or not comparison["backends"][backend]["gate_pass"]
        ):
            raise ValueError("Invalid backend selection")
        report["status"] = "completed"
        for variant in ("original", "combined"):
            output = root / (prefix + variant)
            if not (output / "report.json").exists():
                with (root / (prefix + variant + ".log")).open("w") as log:
                    subprocess.run(
                        [
                            "bash",
                            "DynaNav/warehouse_loop/run_live.sh",
                            "--semantic-backend",
                            "llm",
                            "--llm-mode",
                            "objects-only",
                            "--base-model-only",
                            "--perception-backend",
                            backend,
                            "--landmark-variant",
                            variant,
                            "--headless",
                            "--render-hz",
                            str(render_hz),
                            "--output",
                            str(output),
                        ],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            result = json.loads((output / "report.json").read_text())
            simulation = json.loads(
                (output / "capture/performance-report.json").read_text()
            )
            motion_commits = sum(
                row.get("accepted_instances", 0) > 0
                and simulation["run_wall_start"]
                <= row["wall"]
                <= simulation["run_wall_end"]
                for row in map(
                    json.loads,
                    (output / "mapping/metrics.jsonl").read_text().splitlines(),
                )
                if row["module"] == "semantic"
            )
            store = MapStore(output / "mapping/map")
            try:
                before = store.summary()
                audit = output / "restart-replay"
                audit.mkdir(exist_ok=True)
                for response in sorted(
                    (output / "mapping/requests").glob("*/llm/response.json")
                ):
                    saved = json.loads(response.read_text())
                    destination = audit / response.parent.parent.name
                    if not destination.exists():
                        shutil.copytree(
                            response.parent.parent / "segmentation", destination
                        )
                    consume_categories(
                        store,
                        [SemanticEvent(**e) for e in saved["semantic_events"]],
                        None,
                        destination,
                    )
                if before != store.summary():
                    raise AssertionError("Live restart replay added evidence")
                landmarks = store.observed_landmarks()
                if any(lm["status"] == "verified" for lm in landmarks):
                    raise AssertionError("Candidate landmarks became static anchors")
            finally:
                store.close()
            report["runs"].append(
                dict(
                    variant=variant,
                    report=str(output / "report.json"),
                    loop_complete=result["loop_complete"],
                    all_geometry_consumed=result["capture_frames"]
                    == result["geometry_frames"],
                    semantic_updates_during_motion=result["modules"]["semantic"][
                        "count"
                    ],
                    accepted_semantic_commits_during_motion=motion_commits,
                    restart_replay_unchanged=True,
                    pipeline_pass=result["pipeline_pass"],
                    actual_motion_hz={
                        k: result["modules"][k]["wall_hz_during_motion"]
                        for k in ("geometry", "semantic")
                    },
                    all_requested_rates_met=result["all_requested_rates_met"],
                )
            )
        report["functional_pass"] = all(
            r["loop_complete"]
            and r["all_geometry_consumed"]
            and r["pipeline_pass"]
            and r["semantic_updates_during_motion"] > 0
            and r["accepted_semantic_commits_during_motion"] > 0
            for r in report["runs"]
        )
    write_json(report_path, report)
    acceptance_path = root / "acceptance.json"
    if acceptance_path.exists():
        acceptance = json.loads(acceptance_path.read_text())
        acceptance.update(
            live_rollout_status=report["status"],
            live_functional_pass=report.get("functional_pass"),
            live_validation_report=str(report_path),
            live_5hz_geometry_1hz_semantics_verified=bool(report["runs"])
            and all(r["all_requested_rates_met"] for r in report["runs"]),
        )
        write_json(acceptance_path, acceptance)
    print("LIVE_VALIDATION", json.dumps(report), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--render-hz", type=int, default=30)
    args = p.parse_args()
    if args.render_hz < 5 or args.render_hz > 60 or 60 % args.render_hz:
        p.error("--render-hz must divide 60 and be >=5")
    run(args.root.resolve(), args.render_hz)
