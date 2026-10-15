"""Readable sidecars for real navigation requests; never rewrites model answers."""

import json
from pathlib import Path
import numpy as np


def write_audit(folder, observation, result):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "prompt.txt").write_text(result.get("prompt", ""), encoding="utf-8")
    (folder / "answer.txt").write_text(result.get("text", ""), encoding="utf-8")
    manifest = dict(
        scope="Model input audit; image placeholders are replaced by visual embeddings.",
        original_instruction=observation["task"]["instruction"],
        route_input=observation.get("route"),
        route_revision=observation.get("route_revision"),
        images=observation.get("rgb_history", []),
        map_image=observation.get("map_image") if observation.get("map_valid") else None,
        input_audit=result.get("input_audit"),
        generated_tokens=result.get("generated_tokens"),
        inference_mode=result.get("inference_mode"),
        interface_valid=result.get("valid", False),
        error=result.get("error"),
        stage_seconds=result.get("stage_seconds", {}),
        timing_scope=result.get("timing_scope", "Legacy timing; no wall-clock decode measurement."),
        quality_validated=False,
    )
    (folder / "audit.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if "reference_xy" in result:
        def array(x):
            return x.detach().cpu().numpy() if hasattr(x, "detach") else np.asarray(x)

        xy, mask = array(result["reference_xy"]), array(result["reference_mask"])
        # Export ALL raw points, including the masked-out suffix. Mask is not a safety score.
        np.savetxt(
            folder / "reference.csv",
            np.column_stack((np.arange(len(xy)), xy, mask.astype(int))),
            delimiter=",", header="index,x_m,y_m,decoder_mask", comments="",
            fmt=["%d", "%.9g", "%.9g", "%d"],
        )
