"""Explicit real-weight integration smoke, using synthetic labels (not evaluation evidence).

Run from the repo: python tests/warehouse/run_local_smoke.py --model ... --vlm-checkpoint ...
"""

import argparse
import json
import tempfile
from pathlib import Path

import torch
from fixtures import make_fixture

from ticvla.warehouse.cache import build_cache
from ticvla.warehouse.evaluate import evaluate_action, evaluate_reference
from ticvla.warehouse.train import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--vlm-checkpoint", required=True)
    parser.add_argument("--output-parent", default="/tmp")
    args = parser.parse_args()
    root = Path(
        tempfile.mkdtemp(prefix="warehouse-integration-", dir=args.output_parent)
    )
    cfg = make_fixture(root)
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.model_path = args.model
    cfg.vlm_checkpoint = args.vlm_checkpoint
    cfg.cache_dir = str(root / "real-cache")
    cfg.max_new_tokens = 24
    cfg.lora_rank = 4
    cfg.batch_size = 1
    cfg.max_steps = 1
    print("OUTPUT", root, flush=True)
    reference = train(cfg, "reference", validation_limit=1)
    cfg.reference_checkpoint = str(reference)
    build_cache(cfg, reference)
    action = train(cfg, "flow", validation_limit=1)
    reference_report = evaluate_reference(
        cfg, reference, root / "reference-eval", limit=1
    )
    action_report = evaluate_action(cfg, action, root / "action-eval", limit=1)
    report = {
        "purpose": "synthetic-label real-weight integration smoke only",
        "reference_checkpoint": str(reference),
        "action_checkpoint": str(action),
        "reference": reference_report,
        "action": action_report,
    }
    (root / "smoke_report.json").write_text(json.dumps(report, indent=2))
    print("REPORT", root / "smoke_report.json", flush=True)


if __name__ == "__main__":
    main()
