"""Six real-model recursive TEXT tests. Execution facts are explicitly offline fixtures."""

import argparse
import copy
from dataclasses import asdict
import html
import json
from pathlib import Path
import time
import numpy as np
import torch
from ..config import WarehouseConfig
from ..mapping import MapStore, SourceObservation
from ..mapping.cli import export_view
from ..schema import file_sha256
from ..structured_io import make_observation
from ..vlm import image_tensor
from .vlm import NavigationVLM
from .core import NavigationState
from .flow import FlowMatchingActionExpert
from .runtime import flow_batch, draw_view
from .audit import write_audit
from DynaNav.warehouse_loop.live_metrics import atomic_json

CASES = {
    "landmark": (
        "前往72/4R标牌旁并停下。",
        [
            "尚未开始移动。",
            "目标尚未在地图中唯一匹配，车辆保持停止。",
            "地图中仍无唯一的72/4R，未到达目标。",
        ],
    ),
    "route_order": (
        "沿通道走到尽头，左转，然后前往72/4R旁停下。",
        [
            "尚未开始移动。",
            "已在已观测的通道尽头停下，未转弯。",
            "在通道尽头已完成左转90度，尚未到达标牌。",
        ],
    ),
    "intersection_count": (
        "在第二个路口左转，然后到72/4R旁停下。",
        [
            "尚未通过路口。",
            "已通过一个经进入和离开确认的路口，尚未转弯。",
            "已经到达第二个路口，并完成左转，尚未到达标牌。",
        ],
    ),
    "explore": (
        "探索仓库的未知区域。",
        [
            "尚未开始移动。",
            "已到达第一个可达前沿，现在有另一个可达前沿。",
            "已到达第二个前沿，探索时间预算已耗尽，不代表全部区域探索成功。",
        ],
    ),
    "follow": (
        "跟随前方唯一的行人。",
        [
            "检测到前方唯一的行人，已绑定。",
            "已向绑定行人前进0.5米，行人仍可见。",
            "绑定行人已消失，车辆已经停止，不能切换到另一个人。",
        ],
    ),
    "reverse": (
        "倒退5 m。",
        [
            "从任务起点位移为0米。",
            "诊断夹具：相对锁定子目标起点已倒退2米。",
            "诊断夹具：相对同一锁定起点已倒退4米。",
        ],
    ),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode", type=Path, required=True)
    p.add_argument("--frame", type=int, default=200)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="/data/ticvla/models/InternVL3-1B")
    p.add_argument("--cases", nargs="+", choices=list(CASES), default=list(CASES))
    p.add_argument("--rounds", type=int, choices=(1, 2, 3), default=3)
    p.add_argument("--prompt-style", choices=("original", "compact"), default="original",
                   help="Offline diagnostic only; does not change the deployed prompt.")
    a = p.parse_args()
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    cfg = WarehouseConfig(model_path=a.model, max_new_tokens=2048)
    meta = json.loads(a.episode.read_text())
    frames = [
        json.loads(x)
        for x in (a.episode.parent / meta["frames_file"]).read_text().splitlines()
    ]
    store = MapStore(root / "map", array_compression_level=0)
    for i in range(max(0, a.frame - 15), a.frame + 1):
        f = frames[i]
        source = store.add_observation(
            SourceObservation(
                f"{meta['episode_id']}:{i:08d}",
                str(a.episode.parent / f["rgb"]),
                str(a.episode.parent / f["depth"]),
                f["K"],
                f["T_body_camera"],
                f["T_odom_body"],
                f["t_obs"],
                f["odom_epoch"],
                f["odom_epoch"],
                "offline",
            )
        )
    view = export_view(store, source["image_id"], root / "input-map")
    grid = np.load(view["arrays"])
    model = NavigationVLM(cfg)
    if a.prompt_style == "compact":
        from .route_protocol import COMPACT_NAVIGATION_SYSTEM
        model.navigation_system_override = COMPACT_NAVIGATION_SYSTEM
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
    rows = []
    for name, (instruction, fixtures) in CASES.items():
        if name not in a.cases:
            continue
        nav = NavigationState("offline-" + name, instruction=instruction)
        for step, fixture in enumerate(fixtures[:a.rounds]):
            folder = root / f"{name}-{step}"
            folder.mkdir()
            obs = make_observation(
                store, source["image_id"], view, asdict(nav.state.contract)
            )
            obs.update(
                route=copy.deepcopy(nav.state.route),
                route_revision=nav.state.route_revision,
                route_schema="structured-v1",
                execution_state=dict(
                    scope="OFFLINE INJECTED EXECUTION FACTS; images unchanged; NOT real motion or perception evidence",
                    facts=fixture,
                    previous_commit_valid=bool(nav.route_history),
                ),
                previous_reference=[],
                landmark_schema="legacy",
            )
            obs["source_monotonic"] = time.monotonic()
            source["source_monotonic"] = obs["source_monotonic"]
            obs["previous_local"] = nav.local_input(obs)
            atomic_json(folder / "input.json", obs)
            start = time.monotonic()
            r = model.infer(obs)
            ready = time.monotonic()
            atomic_json(
                folder / "response.json",
                {k: v for k, v in r.items() if not isinstance(v, torch.Tensor)},
            )
            write_audit(folder, obs, r)
            row = dict(
                case=name,
                round=step,
                format_valid=r["valid"],
                service_s=ready - start,
                stages=r.get("stage_seconds"),
                error=r.get("error"),
                fixture=fixture,
                route_input=obs["route"],
                llm_calls=1,
                prompt_style=a.prompt_style,
                route_committed=False,
            )
            if r["valid"]:
                try:
                    packet = nav.commit(
                        r, obs, step + 1, start, ready, ready, ready - start + 1
                    )
                    row.update(
                        route_committed=True,
                        route_output=r["parsed"]["route"],
                        reference_validation=packet.reference_validation,
                    )
                    batch = flow_batch(
                        cfg, grid["grid"], grid["age"], source, [0, 0], packet, None
                    )
                    batch = {
                        k: torch.as_tensor(v, device="cuda") for k, v in batch.items()
                    }
                    with torch.inference_mode():
                        visual = model.backbone.extract_feature(
                            image_tensor(source["rgb"], 448)[None]
                            .cuda()
                            .to(torch.bfloat16)
                        ).float()
                        batch.update(
                            visual=visual,
                            visual_mask=torch.ones(
                                visual.shape[:2], device="cuda", dtype=torch.bool
                            ),
                        )
                        torch.cuda.synchronize()
                        flow_start = time.monotonic()
                        candidates = (
                            flow.sample(
                                batch, torch.Generator(device="cuda").manual_seed(42)
                            )["trajectories"][0]
                            .cpu()
                            .numpy()
                        )
                        torch.cuda.synchronize()
                    row.update(
                        flow_s=time.monotonic() - flow_start,
                        flow_shape=list(candidates.shape),
                        flow_finite=bool(np.isfinite(candidates).all()),
                    )
                    np.savez(
                        folder / "tensors.npz",
                        z_sem=r["z_sem"].numpy(),
                        reference_xy=r["reference_xy"].numpy(),
                        reference_mask=r["reference_mask"].numpy(),
                        confidence=r["confidence"],
                        candidates=candidates,
                    )
                    draw_view(
                        folder,
                        grid["grid"],
                        source,
                        [],
                        r["reference_xy"].numpy()[r["reference_mask"].numpy()],
                        candidates,
                        None,
                        row,
                    )
                except ValueError as exc:
                    row["commit_error"] = str(exc)
            atomic_json(folder / "state.json", nav.checkpoint())
            atomic_json(folder / "report.json", row)
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            atomic_json(
                root / "report.json",
                dict(
                    cases=rows,
                    scope="offline fixture; no physical motion",
                    training=False,
                    thresholds=dict(anchor_m=0.30, endpoint_m=0.50),
                    quality_validated=False,
                ),
            )
    atomic_json(
        root / "provenance.json",
        dict(
            heads_sha256=file_sha256(root / "heads.pt"),
            base_files={
                p.name: file_sha256(p) for p in Path(a.model).glob("*.safetensors")
            },
            rgb_hash=source["rgb_hash"],
            depth_hash=source["depth_hash"],
        ),
    )
    (root / "index.html").write_text(
        '<meta charset="utf-8"><h1>Natural instruction / recursive Route</h1><p>Real model; injected execution fixtures; untrained heads.</p>'
        + "".join(
            f'<h2>{r["case"]} round {r["round"]}</h2><pre>{html.escape(json.dumps(r,ensure_ascii=False,indent=2))}</pre><a href="{r["case"]}-{r["round"]}/response.json">Full actual response</a><img width="600" src="{r["case"]}-{r["round"]}/map.png">'
            for r in rows
        )
    )
    store.close()


if __name__ == "__main__":
    main()
