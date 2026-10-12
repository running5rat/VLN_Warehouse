"""Two-stage warehouse training. Reference adapters and action heads are separate."""

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from . import ARCHITECTURE_VERSION
from .config import WarehouseConfig, load_config
from .data import (
    CachedActionDataset,
    RecordDataset,
    action_collate,
    reference_targets,
    to_device,
)
from .models import FlowActionExpert, QueryActionExpert
from .schema import file_sha256, fingerprint, load_manifest


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def atomic_save(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, tmp)
    os.replace(tmp, path)


def action_model(cfg, visual_dim, kv_dim):
    model_type = FlowActionExpert if cfg.action_head == "flow" else QueryActionExpert
    return model_type(visual_dim, kv_dim, cfg).to(cfg.device)


def load_reference(path, device="cuda"):
    from .vlm import InternVLReference

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("architecture") != ARCHITECTURE_VERSION
        or checkpoint.get("stage") != "reference"
    ):
        raise ValueError("Expected warehouse reference checkpoint")
    cfg = WarehouseConfig(**{**checkpoint["config"], "device": device})
    model = InternVLReference(cfg)
    model.load_adapter(checkpoint["state_dict"])
    if fingerprint(model.tokenizer.get_vocab()) != checkpoint["tokenizer_hash"]:
        raise ValueError("Tokenizer vocabulary differs from reference checkpoint")
    return model, checkpoint


def load_action(path, device="cuda"):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        checkpoint.get("architecture") != ARCHITECTURE_VERSION
        or checkpoint.get("stage") != "flow"
    ):
        raise ValueError("Expected warehouse action checkpoint")
    cfg = WarehouseConfig(**{**checkpoint["config"], "device": device})
    model = action_model(cfg, checkpoint["visual_dim"], checkpoint["kv_dim"])
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model, checkpoint


def train(cfg, stage, resume=None, validation_limit=None):
    if validation_limit is not None and validation_limit < 1:
        raise ValueError("validation_limit must be positive")
    seed_all(cfg.seed)
    output = Path(cfg.output_dir) / stage
    output.mkdir(parents=True, exist_ok=True)
    if (output / "last.pt").exists() and resume is None:
        raise FileExistsError(
            "Use --resume or a new output directory; refusing to overwrite a training run"
        )
    manifest = load_manifest(cfg.manifest)
    manifest_hash = fingerprint(manifest)
    (output / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2))
    if stage == "reference":
        from .vlm import InternVLReference

        model = InternVLReference(cfg)
        train_data, val_data = (
            RecordDataset(cfg.manifest, "train"),
            RecordDataset(cfg.manifest, "val"),
        )

        def labeled(dataset):
            return Subset(
                dataset,
                [
                    i
                    for i in range(len(dataset))
                    if dataset[i]["supervision"]["reference_known"]
                    or dataset[i]["supervision"]["text_valid"]
                ],
            )

        train_data, val_data = labeled(train_data), labeled(val_data)
        collate = list
        lr = cfg.reference_learning_rate
    else:
        train_data = CachedActionDataset(cfg, "train", training=True)
        val_data = CachedActionDataset(cfg, "val")
        cache_index = train_data.cache_index
        if (
            cfg.reference_checkpoint
            and file_sha256(cfg.reference_checkpoint) != cache_index["checkpoint_hash"]
        ):
            raise ValueError(
                "Configured reference checkpoint differs from cached semantics"
            )
        model = action_model(cfg, cache_index["visual_dim"], cache_index["kv_dim"])
        collate, lr = action_collate, cfg.learning_rate
    if not len(train_data) or not len(val_data):
        raise ValueError(
            "Training and validation need independent nonempty labeled splits"
        )
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=cfg.weight_decay,
    )
    loader = DataLoader(
        train_data,
        batch_size=cfg.batch_size,
        shuffle=True,
        collate_fn=collate,
        num_workers=0,
    )
    start_epoch, step, best = 0, 0, float("inf")
    if resume:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        if (
            checkpoint["architecture"] != ARCHITECTURE_VERSION
            or checkpoint["stage"] != stage
        ):
            raise ValueError("Wrong resume checkpoint stage/version")
        if (
            checkpoint["manifest_hash"] != manifest_hash
            or checkpoint["config"] != cfg.to_dict()
        ):
            raise ValueError("Resume requires the same manifest and configuration")
        if stage == "reference":
            model.load_adapter(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint["state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch, step, best = (
            checkpoint["epoch"] + 1,
            checkpoint["step"],
            checkpoint["best"],
        )
        random.setstate(checkpoint["rng"]["python"])
        np.random.set_state(checkpoint["rng"]["numpy"])
        torch.set_rng_state(checkpoint["rng"]["torch"])
        if cfg.device.startswith("cuda"):
            torch.cuda.set_rng_state_all(checkpoint["rng"]["cuda"])
    start_time = time.perf_counter()
    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running, count = 0.0, 0
        for batch_index, batch in enumerate(loader):
            # Last partial accumulation group uses its actual number of batches.
            group_size = min(
                cfg.accumulation_steps,
                len(loader)
                - (batch_index // cfg.accumulation_steps) * cfg.accumulation_steps,
            )
            if stage == "reference":
                batch_loss = 0.0
                for record in batch:
                    prediction, text_loss = model.forward_record(record)
                    parts = model.decoder.loss(
                        prediction, reference_targets(record, cfg.device)
                    )
                    parts["text"] = text_loss
                    loss = sum(
                        cfg.reference_loss_weights[k] * value
                        for k, value in parts.items()
                    )
                    (loss / (len(batch) * group_size)).backward()
                    batch_loss += float(loss.detach()) / len(batch)
            else:
                loss = model.loss(to_device(batch, cfg.device))
                (loss / group_size).backward()
                batch_loss = float(loss.detach())
            if not np.isfinite(batch_loss):
                raise FloatingPointError("Nonfinite training loss")
            running += batch_loss
            count += 1
            if (
                batch_index + 1
            ) % cfg.accumulation_steps == 0 or batch_index + 1 == len(loader):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    cfg.grad_clip,
                    error_if_nonfinite=True,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                if cfg.max_steps and step >= cfg.max_steps:
                    break
        model.eval()
        losses, parsed, total = [], 0, 0
        with torch.no_grad():
            for i in range(len(val_data)):
                if validation_limit is not None and i >= validation_limit:
                    break
                record = val_data[i]
                if stage == "reference":
                    prediction = model.generate_record(record["observation"])
                    parts = model.decoder.loss(
                        prediction, reference_targets(record, cfg.device)
                    )
                    losses.append(
                        float(
                            sum(
                                cfg.reference_loss_weights[k] * v
                                for k, v in parts.items()
                            )
                        )
                    )
                    parsed += prediction["parsed"] is not None
                else:
                    gen = torch.Generator(device=cfg.device).manual_seed(cfg.seed + i)
                    losses.append(
                        float(
                            model.loss(
                                to_device(action_collate([record]), cfg.device),
                                generator=gen,
                            )
                        )
                    )
                total += 1
        val_loss = float(np.mean(losses))
        is_best = val_loss < best
        best = min(best, val_loss)
        checkpoint = {
            "architecture": ARCHITECTURE_VERSION,
            "stage": stage,
            "config": cfg.to_dict(),
            "manifest_hash": manifest_hash,
            "manifest": manifest,
            "epoch": epoch,
            "step": step,
            "best": best,
            "optimizer": optimizer.state_dict(),
            "metrics": {"train_loss": running / count, "val_loss": val_loss},
            "state_dict": model.adapter_state()
            if stage == "reference"
            else {k: v.cpu() for k, v in model.state_dict().items()},
        }
        checkpoint["rng"] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()
            if cfg.device.startswith("cuda")
            else [],
        }
        if stage == "reference":
            checkpoint["tokenizer_hash"] = fingerprint(model.tokenizer.get_vocab())
            checkpoint["metrics"]["generated_parse_rate"] = parsed / max(total, 1)
            model.tokenizer.save_pretrained(output / "tokenizer")
        else:
            checkpoint.update(
                {
                    "visual_dim": cache_index["visual_dim"],
                    "kv_dim": cache_index["kv_dim"],
                    "reference_checkpoint_hash": cache_index["checkpoint_hash"],
                    "cache_map_mode": cache_index["map_mode"],
                }
            )
        atomic_save(checkpoint, output / "last.pt")
        if is_best:
            atomic_save(checkpoint, output / "best.pt")
        report = {
            "epoch": epoch,
            "step": step,
            **checkpoint["metrics"],
            "validation_samples": total,
            "elapsed_s": time.perf_counter() - start_time,
        }
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(report) + "\n")
        print(json.dumps(report), flush=True)
        if cfg.max_steps and step >= cfg.max_steps:
            break
    return output / "last.pt"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pipeline", choices=["warehouse_mvp"], default="warehouse_mvp"
    )
    parser.add_argument("--stage", choices=["reference", "flow"], required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--validation-limit", type=int)
    parser.add_argument(
        "--action-head",
        choices=["flow", "query"],
        help="Matched-condition regression ablation",
    )
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    if args.action_head:
        cfg.action_head = args.action_head
        if args.action_head == "query":
            cfg.output_dir = str(Path(cfg.output_dir) / "query")
    print(train(cfg, args.stage, args.resume, args.validation_limit))


if __name__ == "__main__":
    main()
