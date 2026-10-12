"""Validate saved predictions, restart recovery and geometry without loading models."""

import argparse
import json
from pathlib import Path
import shutil
import time
import numpy as np

from ..mapping import MapStore, SourceObservation, SemanticEvent
from ..mapping.semantic_frame import consume_categories, write_json


def validate(root, backend):
    manifest = json.loads((root / "manifest.json").read_text())
    destination = root / ("replay-validation-" + backend)
    destination.mkdir(exist_ok=False)
    stores, timings = {}, []
    try:
        for row in manifest["rows"]:
            if row["variant"] == "development":
                continue
            original = root / backend / row["key"]
            expected = json.loads((original / "report.json").read_text())
            variant = row["variant"]
            if variant not in stores:
                stores[variant] = MapStore(destination / "maps" / variant)
            store = stores[variant]
            start = time.perf_counter()
            source = store.add_observation(SourceObservation(**row["source"]))
            geometry_s = time.perf_counter() - start
            target = destination / row["key"] / "segmentation"
            shutil.copytree(original / "segmentation", target)
            qa = expected["qa"]
            events = [SemanticEvent(**e) for e in qa["events"]]
            measured = {}
            processed = consume_categories(
                store, events, None, target, timings=measured
            )

            def statuses(p):
                return [
                    (i["category"], i["result"]["status"], i["diagnostics"])
                    for i in p["instances"]
                ]

            if statuses(processed) != statuses(expected["processed"]):
                raise AssertionError(
                    "Model-free replay changed admission/point filtering"
                )
            before = store.summary()
            store.close()
            store = stores[variant] = MapStore(destination / "maps" / variant)
            store.add_observation(SourceObservation(**row["source"]))
            consume_categories(store, events, None, target)
            if before != store.summary():
                raise AssertionError("Restart/replay changed evidence or map version")
            if store.view(source["image_id"])["landmarks"]:
                raise AssertionError("Tentative identity leaked into static MapView")
            timings.append(dict(key=row["key"], geometry_s=geometry_s, **measured))
        latency = {}
        for field in ("geometry_s", "projection_and_persistence_s"):
            values = [r[field] for r in timings]
            latency[field] = dict(
                p50_s=float(np.median(values)), p95_s=float(np.percentile(values, 95))
            )
        report = dict(
            passed=True,
            frames=len(timings),
            latency=latency,
            rows=timings,
            scope="Independent CPU replay; source hashing, projection, association and disk writes included. Not online throughput.",
            restart_after_every_frame=True,
            duplicate_evidence_added=False,
            tentative_static_layer_leak=False,
        )
        write_json(destination / "report.json", report)
        print("REPLAY_VALIDATION", backend, len(timings), latency, flush=True)
    finally:
        for store in stores.values():
            store.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--backend", choices=("yoloe-ppocr", "florence2"), required=True)
    a = p.parse_args()
    validate(a.root, a.backend)
