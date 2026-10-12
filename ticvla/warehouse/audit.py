"""Report split/label coverage and fail on invalid records before training."""

import argparse
import json
from collections import Counter
from pathlib import Path

from .schema import load_manifest, validate_record


def audit_manifest(path):
    path = Path(path)
    manifest = load_manifest(path)
    counts = {split: Counter() for split in ("train", "val", "test")}
    layouts = {split: set() for split in counts}
    episodes = {split: set() for split in counts}
    for entry in manifest["records"]:
        record = validate_record(json.loads((path.parent / entry["path"]).read_text()))
        obs, sup = record["observation"], record["supervision"]
        count = counts[entry["split"]]
        count["records"] += 1
        count["map_available"] += bool(obs.get("map_valid"))
        count["reference_labeled"] += bool(sup["reference_known"])
        count["text_labeled"] += bool(sup["text_valid"])
        count["action_labeled"] += bool(any(sup["action_mask"]))
        count["action_valid_timesteps"] += sum(sup["action_mask"])
        count["missing_rgb"] += sum(
            not Path(f["path"]).is_file() for f in obs["rgb_history"]
        )
        layouts[entry["split"]].add(entry["layout_family"])
        episodes[entry["split"]].add(entry["episode_id"])
    return {
        "generalization": manifest.get("generalization", "unspecified"),
        "splits": {
            s: {**dict(c), "layouts": sorted(layouts[s]), "episodes": len(episodes[s])}
            for s, c in counts.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()
    report = audit_manifest(args.manifest)
    text = json.dumps(report, indent=2)
    if args.output:
        Path(args.output).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
