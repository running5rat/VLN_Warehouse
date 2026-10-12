"""Real-weight, no-training LLM protocol verification on a saved deployment map."""

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .config import WarehouseConfig
from .mapping import MapStore
from .mapping.cli import export_view
from .schema import file_sha256, fingerprint
from .structured_io import (
    StructuredVLM,
    make_observation,
    packet_and_events,
    save_result,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True)
    parser.add_argument("--image-id", required=True)
    parser.add_argument(
        "--task",
        required=True,
        help="JSON containing task_id, instruction, hard_constraints",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--vlm-checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--segmenter-weights",
        help="Consume real LLM events using YOLO-E after releasing VLM",
    )
    args = parser.parse_args()
    root = Path(args.output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(42)
    cfg = WarehouseConfig(
        model_path=args.model,
        vlm_checkpoint=args.vlm_checkpoint,
        device=args.device,
        max_new_tokens=args.max_new_tokens,
    )
    store = MapStore(args.store)
    try:
        view = export_view(store, args.image_id, root / "input-map")
        task = json.loads(Path(args.task).read_text())
        obs = make_observation(store, args.image_id, view, task)
        (root / "observation.json").write_text(json.dumps(obs, indent=2))
        model = StructuredVLM(cfg)
        # Exact untrained heads are saved to make tensor outputs reproducible.
        torch.save(
            {
                "reference": model.decoder.state_dict(),
                "compressor": model.compressor.state_dict(),
                "config": cfg.to_dict(),
                "quality_validated": False,
            },
            root / "interface-heads.pt",
        )
        checkpoint_hash = file_sha256(args.vlm_checkpoint)
        base_files = {
            p.name: file_sha256(p)
            for p in sorted(Path(args.model).iterdir())
            if p.is_file() and p.suffix in (".json", ".py", ".safetensors")
        }
        rows, all_events = [], []
        for generation, mode in enumerate(("TEXT", "PREFILL_FOLLOW")):
            request_id = fingerprint({"observation": obs, "mode": mode})[:24]
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()
            try:
                result = model.infer(obs, mode)
            except (ValueError, RuntimeError) as exc:
                result = {
                    "valid": False,
                    "error": str(exc),
                    "inference_mode": mode,
                    "text": "",
                    "parsed": None,
                    "generated_tokens": 0,
                }
            if args.device.startswith("cuda"):
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            request = {
                "request_id": request_id,
                "mode": mode,
                "observation_hash": fingerprint(obs),
                "checkpoint_sha256": checkpoint_hash,
                "map_version": view["map_version"],
                "task_id": task["task_id"],
                "elapsed_local_s": elapsed,
            }
            packet, events, state = (None, [], None)
            if result["valid"]:
                packet, events, state = packet_and_events(
                    result, obs, request_id, generation, elapsed
                )
                packet.validate(
                    t_use=packet.t_ready,
                    task_id=task["task_id"],
                    odom_epoch=obs["odom_epoch"],
                    route_revision=state.route_revision,
                    max_age=elapsed + 1,
                )
                if mode == "PREFILL_FOLLOW":
                    assert (
                        not events
                        and not result["text"]
                        and state.route == obs["route"]
                    )
                    assert state.route_revision == obs["route_revision"]
                all_events.extend(events)
            save_result(root / mode.lower(), request, result, packet, events, state)
            rows.append(
                {
                    "mode": mode,
                    "valid": result["valid"],
                    "error": result["error"],
                    "generated_tokens": result["generated_tokens"],
                    "elapsed_s": elapsed,
                    "semantic_events": len(events),
                }
            )
            print("VLM_IO", json.dumps(rows[-1]), flush=True)
        del model
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
        processed = []
        if args.segmenter_weights:
            from .mapping.segmentation import YOLOESegmenter, process_event

            backend = YOLOESegmenter(args.segmenter_weights, args.device)
            processed = [process_event(store, e, backend) for e in all_events]
        (root / "events.json").write_text(
            json.dumps([asdict(e) for e in all_events], indent=2)
        )
        report = {
            "purpose": "real model protocol only; no training or semantic-quality evaluation",
            "modes": rows,
            "V4_pass": all(r["valid"] for r in rows),
            "events_processed": processed,
            "V5_dispatch_pass": bool(args.segmenter_weights)
            and all(r["valid"] for r in rows)
            and len(processed) == len(all_events),
            "config": cfg.to_dict(),
            "weights_sha256": checkpoint_hash,
            "base_model_files_sha256": base_files,
            "head_quality_validated": False,
        }
        (root / "report.json").write_text(json.dumps(report, indent=2))
        if not report["V4_pass"]:
            raise SystemExit("V4 failed; see per-mode raw output and error")
    finally:
        store.close()


if __name__ == "__main__":
    main()
