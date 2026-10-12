"""Post-run evaluation only. Catalog/world truth never enters live perception."""

import json
from pathlib import Path
from collections import Counter
import numpy as np
from ticvla.warehouse.geometry import transform_points
from ticvla.warehouse.mapping import MapStore
from DynaNav.warehouse_loop.live_metrics import atomic_json


def evaluate(root):
    root = Path(root)
    capture = root / "capture"
    origin = np.array(
        json.loads((capture / "evaluation-origin.json").read_text())["T_world_odom"]
    )
    catalog = json.loads((capture / "shelf-catalog.json").read_text())
    centers = np.array([spec["center_world"] for spec in catalog])
    store = MapStore(root / "mapping/map")
    try:
        landmarks = [
            lm for lm in store.records("landmarks") if lm["status"] == "verified"
        ]
        rows = []
        for lm in landmarks:
            points = transform_points(store.points(lm), origin)
            position = transform_points(np.array(lm["position"])[None], origin)[0]
            nearest = catalog[
                int(np.argmin(np.linalg.norm(centers - position, axis=1)))
            ]
            spec = next((s for s in catalog if s["label"] == lm["name"]), None)
            surface = error = None
            if spec:
                center, half = (
                    np.array(spec["center_world"]),
                    np.array(spec["size_m"]) / 2,
                )
                lo, hi = center - half, center + half
                outside = np.linalg.norm(
                    np.maximum(np.maximum(lo - points, points - hi), 0), axis=1
                )
                inside = np.min(np.minimum(points - lo, hi - points), axis=1)
                surface = float(
                    np.percentile(np.where(outside > 0, outside, inside), 95)
                )
                error = float(np.linalg.norm(position - center))
            rows.append(
                dict(
                    label=lm["name"],
                    nearest_catalog_label=nearest["label"],
                    centroid_error_m=error,
                    surface_p95_m=surface,
                    independent_observations=len(set(lm["observation_ids"])),
                )
            )
        observations = store.records("observations")
        times = np.array([o["t_obs"] for o in observations])
        trace = [
            json.loads(line)
            for line in (capture / "trajectory.jsonl").read_text().splitlines()
        ]
        trace_t = np.array([r["t"] for r in trace])
        trace_p = np.array([r["position"] for r in trace])
        actual = np.array(
            [(origin @ np.array(o["T_odom_body"]))[:2, 3] for o in observations]
        )
        expected = np.column_stack(
            [np.interp(times, trace_t, trace_p[:, i]) for i in (0, 1)]
        )
        errors = np.linalg.norm(actual - expected, axis=1)
        # Final parked frame can lie beyond the logged driving interval. Report
        # all frames too, but use interpolation only inside the trace interval.
        covered = (times >= trace_t.min()) & (times <= trace_t.max())
        max_error = float(errors[covered].max()) if covered.any() else None
        counts = Counter(row["label"] for row in rows)
        gates = dict(
            has_verified_landmarks=bool(rows),
            nearest_identity=bool(rows)
            and all(r["label"] == r["nearest_catalog_label"] for r in rows),
            centroid_position=bool(rows)
            and all(
                r["centroid_error_m"] is not None and r["centroid_error_m"] <= 0.15
                for r in rows
            ),
            surface_projection=bool(rows)
            and all(
                r["surface_p95_m"] is not None and r["surface_p95_m"] <= 0.05
                for r in rows
            ),
            repeated_evidence=bool(rows)
            and all(r["independent_observations"] >= 2 for r in rows),
            unique_identities=all(c == 1 for c in counts.values()),
            synchronized_source_pose=max_error is not None and max_error <= 0.05,
            source_times_monotonic=bool(np.all(np.diff(times) > 0)),
        )
        report = dict(
            pass_all=all(gates.values()),
            gates=gates,
            landmarks=rows,
            max_source_pose_error_m=max_error,
            max_source_pose_error_including_endpoints_m=float(errors.max()),
            interpolated_frames=int(covered.sum()),
            scope="Post-run geometric/catalog checks only. No live Oracle masks: mask IoU not measured in this run.",
        )
        expected_labels = {f"{i:02d}/{row}R" for row in (4, 5) for i in range(1, 5)}
        observed_labels = {row["label"] for row in rows}
        report["route_coverage"] = {
            "expected_labels": sorted(expected_labels),
            "verified_labels": sorted(observed_labels),
            "missing_labels": sorted(expected_labels - observed_labels),
            "complete": expected_labels <= observed_labels,
            "note": "Coverage is reported separately from correctness of observed landmarks.",
        }
        atomic_json(root / "map-validation.json", report)
        return report
    finally:
        store.close()
