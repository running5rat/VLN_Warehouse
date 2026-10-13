"""Bounded independent model services; simulator modules are never imported."""

import time
import traceback
from pathlib import Path
import numpy as np
from PIL import Image
from ..config import WarehouseConfig
from ..mapping.semantic_frame import write_json
from ..schema import file_sha256

NAVIGATION_SYSTEM = """Navigate a wheeled robot using observed RGB and MapView only.
Body coordinates are metres, x forward and y left. Preserve the original task and constraints.
Output exactly these tags in this order, without explanations:
<landmark>JSON array</landmark><local>location or unknown</local><Route>remaining instruction</Route><waypoint>five (x,y) pairs separated by semicolons, or []</waypoint><REFLINE>
Landmark records have only image_id and category. Select at most one currently visible physical
object useful to the task; use its supplied image_id and a short English category, or [].
The category must name an object, not the warehouse, room, scene, aisle, or a motion.
Keep local and Route concise (prefer under eight words each). Do not invent free space.
For following preserve the bound person and separation; loss means stop and reacquire that person.
Stop immediately after <REFLINE>. Do not claim completion without execution evidence.
"""


def cpu_value(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, dict):
        return {k: cpu_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [cpu_value(x) for x in value]
    return value


def worker(kind, jobs, replies, root, options):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    try:
        import torch

        torch.set_num_threads(2)
        torch.manual_seed(42)
        start = time.monotonic()
        cfg = WarehouseConfig(model_path=options["model"], max_new_tokens=2048)
        if kind == "vlm":
            from .vlm import NavigationVLM

            model = NavigationVLM(cfg)
            model.system_prompt = NAVIGATION_SYSTEM
            torch.save(
                dict(
                    reference=model.decoder.state_dict(),
                    compressor=model.compressor.state_dict(),
                    config=cfg.to_dict(),
                    quality_validated=False,
                ),
                root / "heads.pt",
            )
            warm = root / "warmup.png"
            Image.new("RGB", (448, 448)).save(warm)
            obs = dict(
                rgb_history=[
                    dict(
                        path=str(warm),
                        image_id="warmup",
                        t_obs=0,
                        T_odom_body=np.eye(4).tolist(),
                    )
                ],
                t_obs=0,
                T_odom_body=np.eye(4).tolist(),
                map_valid=False,
                task=dict(instruction="Wait.", hard_constraints=[]),
                route="Wait.",
                robot_state=[0, 0],
                robot_state_mask=[False, False],
            )
            model.infer(obs, "PREFILL_FOLLOW")
            model.infer(obs, "TEXT")
            provenance = dict(
                model=options["model"],
                base_files={
                    p.name: file_sha256(p)
                    for p in Path(options["model"]).glob("*.safetensors")
                },
                heads_sha256=file_sha256(root / "heads.pt"),
                quality_validated=False,
            )
        elif kind == "flow":
            from ..vlm import InternVLReference, image_tensor
            from .flow import FlowMatchingActionExpert

            vision = InternVLReference(cfg).eval().requires_grad_(False)
            model = FlowMatchingActionExpert(vision.visual_dim, cfg).cuda().eval()
            torch.save(
                dict(
                    flow=model.state_dict(),
                    config=cfg.to_dict(),
                    quality_validated=False,
                ),
                root / "heads.pt",
            )
            pixels = torch.zeros(1, 3, 448, 448, device="cuda", dtype=torch.bfloat16)
            with torch.inference_mode():
                features = vision.backbone.extract_feature(pixels).float()
                from types import SimpleNamespace
                from .runtime import flow_batch

                source = dict(T_odom_body=np.eye(4), odom_epoch="warmup", t_obs=0.0)
                packet = SimpleNamespace(
                    refline_xy=np.zeros((32, 2)),
                    refline_mask=np.ones(32, bool),
                    T_odom_body_obs=np.eye(4),
                    odom_epoch="warmup",
                    refline_conf=0.0,
                    z_sem=np.zeros((16, 512)),
                    t_obs=time.monotonic(),
                    inference_mode="TEXT",
                )
                batch = flow_batch(
                    cfg,
                    np.zeros((160, 160)),
                    np.zeros((160, 160)),
                    source,
                    [0.0, 0.0],
                    packet,
                    None,
                )
                batch = {k: torch.as_tensor(v, device="cuda") for k, v in batch.items()}
                batch.update(
                    visual=features,
                    visual_mask=torch.ones(
                        features.shape[:2], device="cuda", dtype=torch.bool
                    ),
                )
                model.sample(batch, torch.Generator(device="cuda").manual_seed(42))
                torch.cuda.synchronize()
            provenance = dict(
                heads_sha256=file_sha256(root / "heads.pt"),
                quality_validated=False,
                modes=9,
                solver_steps=4,
            )
        else:
            from ..mapping.segmentation import YOLOESegmenter

            model = YOLOESegmenter(options["weights"], cache_classes=True, half=True)
            if kind == "person":
                model._set_text_classes(["person"])
            else:
                from ..perception.realtime import MobileReader

                reader = MobileReader()
            warm = root / "warmup.png"
            Image.new("RGB", (960, 720)).save(warm)
            model.segment_instances(
                dict(image_id="warmup", rgb=str(warm), rgb_hash=file_sha256(warm)),
                ["person"] if kind == "person" else ["sign"],
            )
            provenance = model.provenance
        replies.put(
            dict(
                kind=kind,
                ready=True,
                load_warmup_s=time.monotonic() - start,
                provenance=provenance,
            )
        )
        while True:
            job = jobs.get()
            if job is None:
                break
            start = time.monotonic()
            folder = root / f"{job['id']:06d}"
            folder.mkdir()
            clean = {
                k: v
                for k, v in job.items()
                if k not in ("batch", "reference_check_grid")
            }
            write_json(folder / "request.json", clean)
            try:
                if kind == "vlm":
                    result = cpu_value(model.infer(job["observation"], job["mode"]))
                elif kind == "flow":
                    with torch.inference_mode():
                        pixels = (
                            image_tensor(job["rgb"], 448)[None]
                            .cuda()
                            .to(torch.bfloat16)
                        )
                        visual = vision.backbone.extract_feature(pixels).float()
                        batch = {
                            k: torch.as_tensor(v, device="cuda")
                            for k, v in job["batch"].items()
                        }
                        batch["visual"] = visual
                        batch["visual_mask"] = torch.ones(
                            visual.shape[:2], dtype=torch.bool, device="cuda"
                        )
                        result = cpu_value(
                            model.sample(
                                batch,
                                torch.Generator(device="cuda").manual_seed(
                                    42 + job["generation"]
                                ),
                            )
                        )
                    if (
                        result["trajectories"].shape != (1, 9, 1, cfg.horizon, 2)
                        or not np.isfinite(result["trajectories"]).all()
                    ):
                        raise ValueError(
                            "Invalid/nonfinite nine-mode trajectory interface"
                        )
                else:
                    result = model.segment_instances(job["source"], job["categories"])
                    if kind == "semantic":
                        result = {
                            "segmentation": result,
                            "ocr": reader.predict(job["source"]),
                        }
                torch.cuda.synchronize()
                end = time.monotonic()
                torch.save(result, folder / "result.pt")
                # Tensor artifact plus human-readable exact VLM response.
                if kind == "vlm":
                    serial = {
                        k: v for k, v in result.items() if not isinstance(v, np.ndarray)
                    }
                    write_json(folder / "response.json", serial)
                    if result.get("valid"):
                        np.savez(
                            folder / "snapshot.npz",
                            **{
                                k: result[k]
                                for k in ("z_sem", "reference_xy", "reference_mask")
                            },
                        )
                replies.put(
                    dict(
                        kind=kind,
                        id=job["id"],
                        data=result,
                        started=start,
                        ready_at=end,
                        service_s=end - start,
                        artifact=str(folder),
                    )
                )
            except Exception as exc:
                write_json(
                    folder / "error.json",
                    dict(error=str(exc), traceback=traceback.format_exc()),
                )
                replies.put(
                    dict(
                        kind=kind,
                        id=job["id"],
                        error=str(exc),
                        traceback=traceback.format_exc(),
                        started=start,
                        ready_at=time.monotonic(),
                    )
                )
    except BaseException:
        replies.put(dict(kind=kind, fatal=traceback.format_exc()))
