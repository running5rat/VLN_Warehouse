"""Verify CPU/storage optimizations against immutable, previously scored masks."""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .realtime import consume_masks, consume_text
from ..mapping import MapStore, SourceObservation
from ..mapping.semantic_frame import write_json


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--model", default="v8s-engine")
    a = p.parse_args()
    stores, stats = {}, []
    manifest = json.loads((a.root / "manifest.json").read_text())
    try:
        for r in manifest["rows"]:
            f = a.root / a.model / r["key"] / "frame.json"
            d = json.loads(f.read_text())
            store = stores.get(r["variant"])
            if store is None:
                store = stores[r["variant"]] = MapStore(
                    a.root / ("optimization-parity-" + a.model) / r["variant"],
                    array_compression_level=0,
                )
            src = d["source"]
            t = time.time()
            obs = store.add_observation(
                SourceObservation(
                    **{k: src[k] for k in SourceObservation.__dataclass_fields__}
                )
            )
            geo = time.time() - t
            batch = dict(
                source_image_id=src["image_id"],
                rgb_hash=src["rgb_hash"],
                provenance=d["provenance"],
                instances=[],
            )
            for i, inst in enumerate(d["instances"]):
                batch["instances"].append(
                    {**inst, "mask": np.load(f.parent / f"mask-{i:03}.npz")["mask"]}
                )
            t = time.time()
            rows = consume_masks(store, obs, batch, d["selection"])
            projection = time.time() - t
            for new, old in zip(rows, d["rows"]):
                assert new["result"]["status"] == old["result"]["status"], r["key"]
                if new["geometry_file"]:
                    np.testing.assert_array_equal(
                        np.load(new["geometry_file"])["points"],
                        np.load(old["geometry_file"])["points"],
                    )
            binding = consume_text(store, obs, batch, rows, d["ocr"])
            assert [x["status"] for x in binding["updates"]] == [
                x["status"] for x in d["binding"]["updates"]
            ]
            stats.append(
                dict(
                    key=r["key"],
                    geometry_s=geo,
                    projection_s=projection,
                    geometry_exact=True,
                    statuses_equal=True,
                )
            )
        write_json(a.root / f"optimization-parity-{a.model}.json", stats)
        print(
            "PARITY",
            a.model,
            len(stats),
            "all points and statuses identical",
            flush=True,
        )
    finally:
        for store in stores.values():
            store.close()


if __name__ == "__main__":
    main()
