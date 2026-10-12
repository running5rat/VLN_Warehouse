"""Score manual-category diagnostics separately from the frozen end-to-end run."""

import json
from pathlib import Path
import numpy as np
from .evaluate import reading_key
from ..mapping.semantic_frame import write_json


def summarize(root):
    truth = {
        r["key"]: r["targets"]
        for r in json.loads((root / "evaluation/annotations.json").read_text())["rows"]
    }
    results = {}
    for backend in ("yoloe-ppocr", "florence2"):
        cases = []
        for path in sorted((root / ("diagnostic-" + backend)).glob("*/report.json")):
            report = json.loads(path.read_text())
            if not report["diagnostic"]:
                raise ValueError("Expected explicitly marked manual diagnostic")
            targets = truth[path.parent.name]
            instances = report["processed"]["instances"]
            masks = []
            for instance in instances:
                with np.load(
                    path.parent / "segmentation" / instance["mask_file"],
                    allow_pickle=False,
                ) as array:
                    masks.append(array["mask"])
            pairs = []
            for ti, target in enumerate(targets):
                with np.load(
                    root / "evaluation" / target["mask_file"], allow_pickle=False
                ) as array:
                    mask = array["mask"]
                for pi, pred in enumerate(masks):
                    pairs.append(
                        (
                            float((mask & pred).sum() / max(1, (mask | pred).sum())),
                            ti,
                            pi,
                        )
                    )
            matched, used = {}, set()
            for iou, ti, pi in sorted(pairs, reverse=True):
                if iou >= 0.5 and ti not in matched and pi not in used:
                    matched[ti] = (pi, iou)
                    used.add(pi)
            rows = []
            for i, target in enumerate(targets):
                instance = instances[matched[i][0]] if i in matched else None
                reading = instance.get("identity_text") if instance else None
                rows.append(
                    dict(
                        target=target["uid"],
                        expected=target["text"],
                        readable=target["readable"],
                        unique=target["identity_expected_unique"],
                        detected=instance is not None,
                        iou=matched[i][1] if i in matched else 0,
                        reading=reading,
                        correct_3d=bool(
                            instance
                            and target["readable"] is True
                            and target["identity_expected_unique"]
                            and instance["result"]["status"] == "accepted"
                            and reading_key(reading or "")
                            == reading_key(target["text"])
                        ),
                    )
                )
            cases.append(
                dict(
                    key=path.parent.name,
                    manual_prompts=report["manual_category_prompts"],
                    detected_targets=len(matched),
                    targets=len(targets),
                    readable_unique_targets=sum(
                        t["readable"] is True and t["identity_expected_unique"]
                        for t in targets
                    ),
                    correct_3d=sum(t["correct_3d"] for t in rows),
                    rows=rows,
                )
            )
        results[backend] = cases
    write_json(
        root / "diagnostic-summary.json",
        dict(
            scope="Manual category prompts, not VLM-selected; excluded from end-to-end scores and backend selection.",
            backends=results,
        ),
    )
    print(
        {
            b: [(r["key"], r["detected_targets"], r["correct_3d"]) for r in rows]
            for b, rows in results.items()
        }
    )


if __name__ == "__main__":
    import sys

    summarize(Path(sys.argv[1]))
