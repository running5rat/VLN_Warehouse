"""Combine actual stage results; never infer success from process survival."""

import argparse
import json
import re
from pathlib import Path

from ticvla.warehouse.schema import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--mapping", required=True)
    parser.add_argument("--io", required=True)
    parser.add_argument("--unit-log", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = {
        "capture": Path(args.capture) / "capture-report.json",
        "mapping": Path(args.mapping) / "report.json",
        "io": Path(args.io) / "report.json",
        "unit_tests": Path(args.unit_log),
    }
    capture, mapping, io = (
        json.loads(paths[k].read_text()) for k in ("capture", "mapping", "io")
    )
    unit_log = paths["unit_tests"].read_text()
    match = re.search(r"Ran (\d+) tests", unit_log)
    unit_pass = (
        match is not None and int(match[1]) > 0 and unit_log.rstrip().endswith("OK")
    )
    stages = {
        "V0": bool(capture["capture_pass"] and unit_pass),
        "V1": bool(mapping["V1_projection_pass"]),
        "V2": bool(mapping["V2_pass"]),
        "V3": bool(mapping["V3_pass"]),
        "V4": bool(io["V4_pass"]),
        "V5": bool(io["V5_dispatch_pass"]),
    }
    repo = Path(__file__).resolve().parents[2]
    files = sorted((repo / "ticvla/warehouse").rglob("*.py")) + sorted(
        (repo / "tests/warehouse").glob("*.py")
    )
    files += [
        repo / "configs/warehouse_mapping_prompts.json",
        repo / "requirements-warehouse-io.txt",
    ]
    result = {
        "purpose": "semantic-map engineering correctness and real VLM protocol; no training or model-quality claim",
        "stages": stages,
        "passed": all(stages.values()),
        "unit_tests": int(match[1]) if match else 0,
        "reports": {
            k: {"path": str(p.resolve()), "sha256": file_sha256(p)}
            for k, p in paths.items()
        },
        "implementation_files_sha256": {
            str(p.relative_to(repo)): file_sha256(p) for p in files
        },
        "isaac_version": capture["isaac_version"],
        "instances": mapping["instances"],
        "registration_errors": mapping["registration_errors"],
        "analytic_geometry": mapping["analytic_geometry"],
        "llm_modes": io["modes"],
        "limitations": [
            "fixed synthetic warehouse fixture; no generalization benchmark",
            "new odom registration tested by coordinate replay, not physical relocation",
            "manual verified ROI/staticness for map acceptance; LLM semantic correctness not tested",
            "rendered MapView baseline, not learned MapEncoder",
            "reference/context heads untrained; no policy/controller deployment",
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {"passed": result["passed"], "stages": stages, "report": str(output)},
            indent=2,
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
