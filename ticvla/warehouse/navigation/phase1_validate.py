"""Sequential Isaac validation matrix; never edits model answers or injects subgoals."""

import argparse
import html
import json
from pathlib import Path
import subprocess
from DynaNav.warehouse_loop.live_metrics import atomic_json

INSTRUCTIONS = {
    "landmark": "前往72/4R标牌旁并停下。",
    "route": "沿通道走到尽头，左转，然后前往72/4R标牌旁停下。",
    "explore": "探索仓库的未知区域。",
    "follow": "跟随前方唯一的行人。",
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seconds", type=int, default=30)
    p.add_argument("--port", type=int, default=8772)
    a = p.parse_args()
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    rows = []
    cases = [
        (scene, task, backend, instruction)
        for scene in ("warehouse", "junction-fixture")
        for task, instruction in INSTRUCTIONS.items()
        for backend in ("geometry-baseline", "model")
    ]
    cases.append(("warehouse", "reverse", "reference-only", "倒退5 m。"))
    for scene, task, backend, instruction in cases:
        name = f"{scene}-{task}-{backend}"
        folder = root / name
        cmd = [
            "bash",
            "DynaNav/warehouse_loop/run_navigation.sh",
            "--instruction",
            instruction,
            "--execution-backend",
            backend,
            "--scene-layout",
            scene,
            "--output",
            str(folder),
            "--headless",
            "--max-steps",
            str((a.seconds + 6) * 60),
            "--budget-seconds",
            str(a.seconds),
            "--port",
            str(a.port),
        ]
        if task == "follow":
            cmd += ["--scene-person"]
        # Registered observed map only for original landmark cases; never hidden scene labels.
        known = Path(
            "/data/ticvla/outputs/warehouse-loop/instances-original-20261012-154216/mapping/map"
        )
        if task == "landmark" and scene == "warehouse" and known.exists():
            cmd += [
                "--map",
                str(known),
                "--map-anchor-image",
                "instances-original-20261012-154216:00000000",
            ]
        print("START", name, flush=True)
        with (root / (name + ".log")).open("w") as log:
            try:
                code = subprocess.run(
                    cmd, stdout=log, stderr=subprocess.STDOUT, timeout=600
                ).returncode
            except subprocess.TimeoutExpired:
                code = 124
        report_path = folder / "report.json"
        report = json.loads(report_path.read_text()) if report_path.exists() else {}
        row = dict(
            scene=scene,
            task=task,
            backend=backend,
            command=cmd,
            exit_code=code,
            report=str(report_path),
            geometry_complete=report.get("geometry_complete", False),
            interface_pass=report.get("model_interface_pass", False),
            valid_text=report.get("valid_text", 0),
            observed_movement=report.get("observed_movement", False),
            executed_translation_commands=report.get(
                "executed_translation_commands", 0
            ),
            executed_rotation_commands=report.get("executed_rotation_commands", 0),
            reference_consistent=report.get("reference_consistent", 0),
            task_completion=report.get("task_completion", "runtime_failed"),
            semantic_TEXT_branch_pass=report.get("semantic_TEXT_branch_pass", False),
        )
        rows.append(row)
        atomic_json(
            root / "report.json",
            dict(
                cases=rows,
                training=False,
                quality_acceptance=False,
                scope="limited-budget real Isaac task trials; failure never replaced by a manual plan",
            ),
        )
        print("END", json.dumps(row, ensure_ascii=False), flush=True)
        (root / "index.html").write_text(
            '<meta charset="utf-8"><h1>Phase 1 Isaac task trials</h1><p>Task success, interface success and motion are distinct.</p>'
            + "".join(
                f'<h2>{html.escape(x["scene"]+" / "+x["task"]+" / "+x["backend"])}</h2><pre>{html.escape(json.dumps(x,ensure_ascii=False,indent=2))}</pre><a href="{x["scene"]}-{x["task"]}-{x["backend"]}/index.html">Viewer</a><img width="600" src="{x["scene"]}-{x["task"]}-{x["backend"]}/navigation/map.png">'
                for x in rows
            )
        )


if __name__ == "__main__":
    main()
