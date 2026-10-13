"""Real VLM + nine-mode Flow, deterministic source frame; no simulator needed."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time
import numpy as np
import torch
from ..config import WarehouseConfig
from ..mapping import MapStore, SourceObservation
from ..mapping.cli import export_view
from ..schema import file_sha256
from .vlm import NavigationVLM
from ..structured_io import make_observation
from ..vlm import image_tensor
from .flow import FlowMatchingActionExpert
from .workers import NAVIGATION_SYSTEM
from .core import NavigationState, check_candidates
from .runtime import flow_batch, draw_view
from DynaNav.warehouse_loop.live_metrics import atomic_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode", type=Path, required=True)
    p.add_argument("--frame", type=int, default=200)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="/data/ticvla/models/InternVL3-1B")
    args = p.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cfg = WarehouseConfig(model_path=args.model, max_new_tokens=2048)
    meta = json.loads(args.episode.read_text())
    frames = [
        json.loads(x)
        for x in (args.episode.parent / meta["frames_file"]).read_text().splitlines()
    ]
    store = MapStore(root / "map", array_compression_level=0)
    for i in range(max(0, args.frame - 15), args.frame + 1):
        f = frames[i]
        obs = SourceObservation(
            f"{meta['episode_id']}:{i:08d}",
            str(args.episode.parent / f["rgb"]),
            str(args.episode.parent / f["depth"]),
            f["K"],
            f["T_body_camera"],
            f["T_odom_body"],
            f["t_obs"],
            f["odom_epoch"],
            f["odom_epoch"],
            "replay",
        )
        source = store.add_observation(obs)
    view = export_view(store, source["image_id"], root / "input-map")
    grid_data = np.load(view["arrays"])
    model = NavigationVLM(cfg)
    model.system_prompt = NAVIGATION_SYSTEM
    flow = FlowMatchingActionExpert(model.visual_dim, cfg).cuda().eval()
    torch.save(
        dict(
            reference=model.decoder.state_dict(),
            compressor=model.compressor.state_dict(),
            flow=flow.state_dict(),
            config=cfg.to_dict(),
            quality_validated=False,
        ),
        root / "heads.pt",
    )
    report = []
    for generation, (task, mode) in enumerate(
        [
            ("landmark", "TEXT"),
            ("explore", "TEXT"),
            ("follow", "TEXT"),
            ("follow", "PREFILL_FOLLOW"),
        ],
        1,
    ):
        folder = root / (task + "-" + mode.lower())
        folder.mkdir()
        nav = NavigationState("offline-" + task, task, "visible sign")
        observation = make_observation(
            store, source["image_id"], view, asdict(nav.state.contract)
        )
        observation.update(
            task_mode=task,
            landmark_schema="legacy",
            previous_reference=[],
            follow_target=None,
        )
        if task == "follow":
            # Protocol-only synthetic binding, explicitly NOT a real person tracking acceptance.
            observation.update(
                target_visible=True,
                target_binding_id="interface-fixture-not-person-truth",
            )
        observation["source_monotonic"] = time.monotonic()
        source["source_monotonic"] = observation["source_monotonic"]
        atomic_json(folder / "input.json", observation)
        started = time.monotonic()
        result = model.infer(observation, mode)
        torch.cuda.synchronize()
        ready = time.monotonic()
        serial = {k: v for k, v in result.items() if not isinstance(v, torch.Tensor)}
        atomic_json(folder / "response.json", serial)
        row = dict(
            task=task,
            mode=mode,
            valid=result["valid"],
            service_s=ready - started,
            error=result.get("error"),
            stage_seconds=result.get("stage_seconds"),
        )
        if result["valid"]:
            # Offline run verifies packet construction, not a 10 s online acceptance deadline.
            packet = nav.commit(
                result,
                observation,
                generation,
                started,
                ready,
                ready,
                ready - observation["source_monotonic"] + 1,
            )
            batch = flow_batch(
                cfg, grid_data["grid"], grid_data["age"], source, [0, 0], packet, None
            )
            batch = {k: torch.as_tensor(v, device="cuda") for k, v in batch.items()}
            with torch.inference_mode():
                visual = model.backbone.extract_feature(
                    image_tensor(source["rgb"], 448)[None].cuda().to(torch.bfloat16)
                ).float()
                batch["visual"] = visual
                batch["visual_mask"] = torch.ones(
                    visual.shape[:2], device="cuda", dtype=torch.bool
                )
                torch.cuda.synchronize()
                t = time.monotonic()
                candidates = (
                    flow.sample(batch, torch.Generator(device="cuda").manual_seed(42))[
                        "trajectories"
                    ][0]
                    .cpu()
                    .numpy()
                )
                torch.cuda.synchronize()
                row["flow_s"] = time.monotonic() - t
            ref = result["reference_xy"].numpy()[result["reference_mask"].numpy()]
            selected, scores = check_candidates(candidates, grid_data["grid"], ref)
            np.savez(
                folder / "tensors.npz",
                z_sem=result["z_sem"].numpy(),
                reference=result["reference_xy"].numpy(),
                reference_mask=result["reference_mask"].numpy(),
                candidates=candidates,
            )
            atomic_json(
                folder / "selection.json",
                dict(
                    scores=scores,
                    selected=selected is not None,
                    scorer="rules-baseline",
                    quality_validated=False,
                ),
            )
            draw_view(
                folder,
                grid_data["grid"],
                source,
                [],
                ref,
                candidates,
                selected,
                dict(mode=mode, task=task),
            )
            row.update(
                shape=list(candidates.shape),
                finite=bool(np.isfinite(candidates).all()),
                safe_candidates=sum(x["reason"] is None for x in scores),
                text_generated=bool(result["text"]),
            )
        report.append(row)
        print(json.dumps(row), flush=True)
    atomic_json(
        root / "report.json",
        dict(
            cases=report,
            interface_pass=all(x["valid"] and x.get("finite") for x in report),
            weights_sha256=file_sha256(root / "heads.pt"),
            model_files={
                p.name: file_sha256(p) for p in Path(args.model).glob("*.safetensors")
            },
            source_image_id=source["image_id"],
            rgb_sha256=source["rgb_hash"],
            depth_sha256=source["depth_hash"],
            quality_validated=False,
            follow_scope="PREFILL protocol fixture only; no person tracking or motion claim",
        ),
    )
    (root / "index.html").write_text(
        '<meta charset="utf-8"><h1>Real VLM / Flow interface replay</h1><p>Untrained heads; no navigation quality claim.</p>'
        + "".join(
            f'<h2>{x["task"]} {x["mode"]}</h2><img width="600" src="{x["task"]}-{x["mode"].lower()}/map.png"><a href="{x["task"]}-{x["mode"].lower()}/response.json">actual response</a>'
            for x in report
        )
    )
    store.close()


if __name__ == "__main__":
    main()
