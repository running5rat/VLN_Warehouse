"""Cache actual generated reference + RAW K/V, never trainable compressed tokens."""

import argparse
import json
import time
from pathlib import Path

import torch

from . import ARCHITECTURE_VERSION
from .config import load_config
from .data import RecordDataset
from .schema import file_sha256, fingerprint, load_manifest
from .train import atomic_save, load_reference


def build_cache(cfg, checkpoint_path, map_mode="correct", split="all", limit=None):
    model, _ = load_reference(checkpoint_path, cfg.device)
    for key in (
        "num_reference_points",
        "coordinate_scale",
        "image_size",
        "horizon",
        "action_dt",
    ):
        if getattr(model.cfg, key) != getattr(cfg, key):
            raise ValueError(f"Reference checkpoint/cache config mismatch: {key}")
    model.cfg.max_new_tokens = cfg.max_new_tokens
    model.eval().requires_grad_(False)
    output = Path(cfg.cache_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_hash = file_sha256(checkpoint_path)
    manifest = load_manifest(cfg.manifest)
    index = {
        "architecture": ARCHITECTURE_VERSION,
        "manifest_hash": fingerprint(manifest),
        "checkpoint_hash": checkpoint_hash,
        "visual_dim": model.visual_dim,
        "kv_dim": model.kv_dim,
        "map_mode": map_mode,
        "config": cfg.to_dict(),
        "records": {},
    }
    index_path = output / "index.json"
    if index_path.exists():
        old = json.loads(index_path.read_text())
        if any(
            old[k] != index[k]
            for k in ("checkpoint_hash", "manifest_hash", "map_mode", "config")
        ):
            raise ValueError(
                "Use a new cache directory for changed checkpoint, manifest or settings"
            )
        index = old
    for subset in ("train", "val", "test") if split == "all" else [split]:
        data = RecordDataset(cfg.manifest, subset)
        # A reproducible same-split map permutation; never read privileged maps.
        for i in range(len(data)):
            if limit is not None and i >= limit:
                break
            record = data[i]
            obs = record["observation"]
            sid = record["provenance"]["sample_id"]
            if sid in index["records"]:
                continue
            donor = data[(i + max(1, len(data) // 2)) % len(data)]["observation"]
            if map_mode == "shuffled" and len(data) < 2:
                raise ValueError("Map shuffle requires at least two records per split")
            if cfg.device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            result = model.generate_record(obs, map_mode, donor.get("map_image"))
            visual, visual_mask = model.current_visual(obs)
            if cfg.device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            cache = {
                "checkpoint_hash": checkpoint_hash,
                "generation_id": i,
                "raw_kv": result["raw_kv"][0].cpu().to(torch.float16),
                "kv_mask": result["kv_mask"][0].cpu(),
                "reference_xy": result["reference_xy"][0].cpu(),
                "reference_mask": result["reference_mask"][0].cpu(),
                "confidence": float(result["confidence_logits"].sigmoid()),
                "text": result["text"],
                "parse_valid": result["parsed"] is not None,
                "visual": visual[0].cpu().half(),
                "visual_mask": visual_mask[0].cpu(),
                "elapsed_s": elapsed,
            }
            atomic_save(cache, output / (sid + ".pt"))
            index["records"][sid] = {
                "path": sid + ".pt",
                "observation_hash": fingerprint(obs),
                "map_grid": donor.get("map_grid")
                if map_mode == "shuffled"
                else obs.get("map_grid"),
            }
            tmp = index_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(index, indent=2))
            tmp.replace(index_path)
            print(
                json.dumps(
                    {
                        "sample_id": sid,
                        "parse_valid": cache["parse_valid"],
                        "elapsed_s": elapsed,
                    }
                ),
                flush=True,
            )
    return index_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--map-mode", choices=["correct", "removed", "shuffled"], default="correct"
    )
    parser.add_argument(
        "--split", choices=["all", "train", "val", "test"], default="all"
    )
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(
        build_cache(
            load_config(args.config),
            args.checkpoint,
            args.map_mode,
            args.split,
            args.limit,
        )
    )


if __name__ == "__main__":
    main()
