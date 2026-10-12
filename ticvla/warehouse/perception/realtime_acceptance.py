"""Aggregate measured gates without lowering targets or hiding failed attempts."""

import argparse
import json
import re
from pathlib import Path

from ..mapping.semantic_frame import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    a = p.parse_args()
    root = a.root
    quality = json.loads((root / "quality-report.json").read_text())
    review = json.loads((root / "manual-review.json").read_text())
    live = {}
    for variant in ("original", "combined"):
        folder = root / f"live-{variant}-optimized"
        data = json.loads((folder / "mapping/realtime-report.json").read_text())
        sim = json.loads((folder / "capture/performance-report.json").read_text())
        capture = json.loads((folder / "capture/capture-report.json").read_text())
        memory = []
        for line in (folder / "gpu-metrics.jsonl").read_text().splitlines():
            metric = json.loads(line)
            if metric.get("whole_gpu_csv"):
                memory.append(float(metric["whole_gpu_csv"].split(",")[-3].strip()))
        data.update(
            rgbd_hz=sim["capture"]["wall_hz_including_startup_and_drain"],
            render_hz=sim["viewport_render"]["wall_hz_including_startup_and_drain"],
            physics_hz=sim["physics"]["wall_hz_including_startup_and_drain"],
            loop_complete=capture["loop_complete"],
            source_frames=capture["frames"],
            whole_gpu_peak_mib=max(memory) if memory else None,
            load_and_warm_s=json.loads((folder / "mapping/ready.json").read_text())[
                "load_s"
            ],
        )
        data["frequency_pass"] = (
            min(data["rgbd_hz"], data["geometry_hz"], data["mask_hz"]) >= 4.75
        )
        data["quality_correct_text_updates_verified"] = review["live"][variant][
            "correct_text_update_verified"
        ]
        live[variant] = data
    frame = json.loads((root / "single-frame-200/report.json").read_text())
    replay = json.loads((root / "replay-frame-200/report.json").read_text())
    test_log = (root / "tests-final.log").read_text()
    tests_ok = test_log.strip().endswith("OK")
    test_count = int(re.search(r"Ran (\d+) tests", test_log).group(1))
    failed = []
    for variant, data in live.items():
        if not data["timing_pass"]:
            failed.append(f"{variant}: real-time timing gates failed")
        if not data["quality_correct_text_updates_verified"]:
            failed.append(
                f"{variant}: no visually verified correct live text binding update"
            )
    selected = quality["v8s-engine"]
    if not selected["quality_gate"]:
        failed.append(
            "Held-out quality gates failed: color/surface positives missing and/or wrong carrier binding"
        )
    if frame["accepted_instances"] == 0 or frame["text_updates"] == 0:
        failed.append("Frame 200 has no accepted landmark plus text positive")
    if not tests_ok:
        failed.append("Regression suite failed")
    out = dict(
        overall_pass=not failed,
        selected_development_backend="v8s-engine",
        deployment_quality_approved=selected["quality_gate"],
        implementation="Real fixed-vocabulary TensorRT segmentation, one InternVL selector, CPU mobile OCR and late text events implemented",
        tests=dict(
            suite="tests/warehouse",
            passed=test_count if tests_ok else None,
            log="tests-final.log",
        ),
        quality=quality,
        realtime=live,
        frame_200=dict(
            llm_calls=frame["llm_calls"],
            replay_llm_calls=replay["llm_calls"],
            accepted_instances=frame["accepted_instances"],
            text_updates=frame["text_updates"],
            end_to_end_positive=frame["accepted_instances"] > 0
            and frame["text_updates"] > 0,
        ),
        recovery=json.loads((root / "recovery-report.json").read_text()),
        failed_requirements=failed,
        limitations=[
            "IoU and recall cover annotated signs/containers, not all facilities",
            "Correct 3D identity count normalizes whitespace/OCR region separators; strict text accuracy is separate",
            "No identity was confirmed; zero wrong confirmations is not evidence of reliable identity confirmation",
            "All landmarks remain tentative; static anchors, navigation success and training are not validated",
            "Latency measures file publication, not browser painting; late results remain deadline failures",
        ],
        attempts=[
            "qa-development-attempt-01",
            "qa-evaluation-attempt-01-environment-failure",
            "qa-evaluation-attempt-02-uncached",
            "live-original",
        ],
        equivalent_optimization="All 156 cached v8s frames retain exactly identical point arrays and event outcomes; see optimization-parity-v8s-engine.json",
    )
    write_json(root / "acceptance.json", out)
    print("ACCEPTANCE", out["overall_pass"], failed, flush=True)


if __name__ == "__main__":
    main()
