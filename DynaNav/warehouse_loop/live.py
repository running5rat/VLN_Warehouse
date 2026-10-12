"""One command: host Isaac 5.0 GUI + incremental mapping + optional web viewer."""

import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time

from DynaNav.warehouse_loop.live_metrics import atomic_json, rate_report

HTML = """<!doctype html><meta charset="utf-8"><title>Live warehouse semantic map</title>
<style>body{background:#19232e;color:#edf5ff;font:15px sans-serif;margin:18px}.panes{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}img{width:100%;object-fit:contain;background:white}pre{white-space:pre-wrap}table{border-collapse:collapse;width:100%;margin:12px 0}td,th{border:1px solid #516171;padding:6px;text-align:left}h3{margin:8px 0}@media(max-width:1000px){.panes{grid-template-columns:1fr}}</style>
<h2>实时语义建图 / Live semantic map</h2><pre id="status">Waiting for RGB-D...</pre>
<div class="panes"><div><h3>持续地图 / Map</h3><img id="map"></div><div><h3>全部检出 / Perception masks + boxes</h3><img id="detections"></div><div><h3>最终入图 / Accepted masks + boxes</h3><img id="semantic"></div></div>
<p>图像来自最近处理的源帧，可能落后于车辆。左图橙色地标为 tentative；右图只显示通过地图写入检查的实例。固定词表模式下检测全部核心类别，LLM 只决定登记子集。文字图使用独立标注的源帧。</p>
<pre id="source">Waiting for semantic inference...</pre><details><summary>LLM 原始回答</summary><pre id="answer"></pre></details>
<h3>VLM 筛选 → 感知检出 → 地图写入</h3><table><thead><tr><th>类别</th><th>结果</th><th>实例数</th><th>接受数</th><th>原因</th></tr></thead><tbody id="categories"></tbody></table>
<h3>逐实例来源、框与最终状态</h3><table><thead><tr><th>ID</th><th>来源</th><th>类别 / 编号</th><th>置信度</th><th>框 [x1,y1,x2,y2]</th><th>最终状态</th><th>地标 ID / 原因</th></tr></thead><tbody id="instances"></tbody></table>
<details><summary>文字源帧与区域</summary><img id="ocr" style="max-width:960px"></details><h3>文字区域与绑定结果</h3><table><thead><tr><th>文字原文</th><th>识别分数</th><th>所属实例</th><th>处理状态</th></tr></thead><tbody id="texts"></tbody></table>
<h3>感知任务失败</h3><table><thead><tr><th>类别</th><th>区域</th><th>原因</th></tr></thead><tbody id="failures"></tbody></table>
<script>
let mapVersion=0, semanticVersion=0, latest=null;
function rows(id,values){let body=document.getElementById(id);body.replaceChildren();for(let valuesRow of values){let tr=document.createElement('tr');for(let value of valuesRow){let td=document.createElement('td');td.textContent=String(value??'');tr.appendChild(td);}body.appendChild(tr);}}
async function update(){try{let s=await(await fetch('mapping/status.json?t='+Date.now())).json();
document.getElementById('status').textContent=`Geometry ${s.geometry_frames} (${s.geometry_hz.toFixed(2)} Hz) | ${s.selection_mode==='instances'?'Candidates':'Semantic'} ${s.semantic_frames} (${s.semantic_hz.toFixed(2)} Hz) | Admitted frames ${s.admitted_frames??'N/A'} | Verified ${s.verified} | Tentative ${s.tentative} | Pending ${s.semantic_pending}`;
if(s.updated_wall!==mapVersion){mapVersion=s.updated_wall;document.getElementById('map').src='mapping/map.png?v='+mapVersion;}
try{let response=await fetch('mapping/inspection.json?t='+Date.now());if(response.ok)latest=await response.json();}catch(e){}
if(latest){let v=latest;document.getElementById('source').textContent=`Source ${v.image_id} | source lag ${Math.max(0,s.source_time-v.t_obs).toFixed(2)}s | ${v.llm_valid?'VLM parsed':'VLM '+((v.category_selection||{}).status||v.llm_error)} | publish latency ${v.source_publish_latency_s==null?'N/A':v.source_publish_latency_s.toFixed(3)+'s'} | selection source ${(v.category_selection||{}).source_image_id||'N/A'} | pending vocabulary ${((v.category_selection||{}).proposed_categories||[]).join(', ')} | text source ${v.text_source_image_id||v.image_id}`;
if(v.updated_wall!==semanticVersion){semanticVersion=v.updated_wall;for(let name of ['detections','semantic'])document.getElementById(name).src='mapping/'+name+'.png?v='+semanticVersion;
document.getElementById('answer').textContent=v.raw_answer;
if(v.text_source_image_id)document.getElementById('ocr').src='mapping/ocr.png?v='+semanticVersion;
rows('texts',(v.texts||[]).map(t=>[t.text,t.confidence==null?'N/A':t.confidence.toFixed(3),t.owner_instance==null?'unassigned':(v.category_selection?.selection_mode==='instances'?'d':'L')+t.owner_instance,t.status]));
rows('failures',(v.perception_failures||[]).map(f=>[f.category,JSON.stringify(f.bbox||null),f.reason]));
rows('categories',v.categories.map(c=>[c.request.category,c.status,c.instances,c.accepted,c.reason]).concat(v.filtered_categories.map(c=>[c.category,'filtered',0,0,c.reason])));
rows('instances',v.instances.map(r=>[r.instance_id,r.origin,r.category+(r.name?' / '+r.name:''),r.confidence==null?'N/A':r.confidence.toFixed(3),r.bbox.map(x=>Math.round(x)).join(', '),r.status+(r.landmark_status?' / '+r.landmark_status:'')+(r.identity_status?' / '+r.identity_status:''),(r.landmark_uid||'')+(r.reason?' | '+r.reason:'')+((r.reobservation_reasons||[]).length?' | needs_reobservation: '+r.reobservation_reasons.join(', '):'')]));}}
}catch(e){}setTimeout(update,200);}update();
</script>"""


class QuietHandler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass


def aggregate(root):
    capture = json.loads((root / "capture/capture-report.json").read_text())
    sim = json.loads((root / "capture/performance-report.json").read_text())
    mapping = json.loads((root / "mapping/report.json").read_text())
    if mapping.get("perception_backend") == "yoloe-realtime":
        report = dict(
            mapping,
            capture_frames=capture["frames"],
            loop_complete=capture["loop_complete"],
            distance_m=capture["distance_m"],
            all_requested_rates_met=mapping["realtime"]["timing_pass"],
            pipeline_pass=capture["frames"] == mapping["geometry"]["count"],
        )
        atomic_json(root / "report.json", report)
        return report
    start, end = sim["run_wall_start"], sim["run_wall_end"]
    rows = [
        json.loads(x) for x in (root / "mapping/metrics.jsonl").read_text().splitlines()
    ]
    modules = {
        k: sim[k] for k in ("physics", "viewport_render", "capture", "map_panel")
    }
    if "record_write" in sim:
        modules["record_write"] = sim["record_write"]
    for key, target in (
        ("geometry", 5),
        ("semantic", mapping.get("semantic_sample_hz", 1)),
        ("map_publish", 5),
    ):
        active = [r for r in rows if r["module"] == key and start <= r["wall"] <= end]
        metrics = rate_report(active, target, start, end)
        metrics["wall_hz_during_motion"] = len(active) / (end - start)
        metrics["meets_target_95pct"] = bool(
            len(active) >= 10 and len(active) / (end - start) >= target * 0.95
        )
        modules[key] = metrics
    report = dict(
        loop_complete=capture["loop_complete"],
        distance_m=capture["distance_m"],
        capture_frames=capture["frames"],
        geometry_frames=mapping["geometry"]["count"],
        semantic_frames=mapping["semantic"]["count"],
        real_time_factor=sim["real_time_factor"],
        wall_seconds=sim["wall_seconds"],
        simulation_seconds=sim["simulation_elapsed_seconds"],
        capture_source_hz=sim["capture_source_hz"],
        modules=modules,
        semantic_backlog_max=mapping["semantic_backlog_max"],
        max_semantic_source_lag_s=mapping["max_semantic_source_lag_s"],
        geometry_latency_p95_s=mapping["capture_to_geometry_p95_s"],
        yolo_calls=mapping["yolo_calls"],
        perception_backend=mapping.get("perception_backend"),
        perception_calls=mapping.get("perception_calls", {}),
        store=mapping["store"],
        pipeline_pass=capture["frames"] == mapping["geometry"]["count"]
        and (
            mapping["yolo_calls"] > 0
            or mapping.get("perception_calls", {}).get("segmentation", 0) > 0
        ),
        all_requested_rates_met=all(
            modules[k]["meets_target_95pct"]
            for k in ("viewport_render", "capture", "geometry", "semantic")
        ),
        all_measured_module_rates_met=all(
            m["meets_target_95pct"] for m in modules.values()
        ),
        rate_tolerance_fraction=0.05,
        notes=[
            "Target rates require >=95% actual wall-clock throughput; sampling in simulation time alone is insufficient.",
            "viewport_render measures scheduled viewport update completions (capture shares its slot), not GPU present frames.",
            "No Oracle or simulator object identities enter mapping; odometry is simulator pose.",
            "Semantic throughput counts source frames processed including no-object frames; YOLO calls are reported separately.",
        ],
    )
    report["semantic_backend"] = mapping.get("semantic_backend", "ocr")
    if report["semantic_backend"] == "llm":
        for key in (
            "semantic_frames_dropped",
            "semantic_sample_hz",
            "llm_valid_frames",
            "accepted_instances",
            "source_projection_consistent",
            "weight_mode",
        ):
            report[key] = mapping[key]
        report["pipeline_pass"] = bool(
            report["pipeline_pass"]
            and mapping["llm_valid_frames"] > 0
            and mapping["accepted_instances"] > 0
            and mapping["source_projection_consistent"]
        )
        report["notes"].append(
            "LLM sample rate is a requested upper bound; actual generation throughput and skipped sources are reported. Semantic quality and global map accuracy are not evaluated."
        )
        for key in (
            "llm_mode",
            "llm_calls",
            "structured_inference_executed",
            "marker_accepted_instances",
            "marker_yolo_calls",
            "identity_confirmed",
            "needs_reobservation",
        ):
            report[key] = mapping.get(key)
    atomic_json(root / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--weights", default="/data/ticvla/models/yoloe-v8l-seg.pt")
    p.add_argument("--max-steps", type=int, default=18000)
    p.add_argument("--headless", action="store_true")
    p.add_argument(
        "--render-hz",
        type=int,
        default=30,
        help="Viewport rendering rate; RGB-D/geometry sampling remains 5 Hz",
    )
    from .landmark_variants import VARIANTS

    p.add_argument("--landmark-variant", choices=VARIANTS)
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--bind", default="127.0.0.1")
    p.add_argument(
        "--keep-viewer",
        action="store_true",
        help="Keep web viewer open after capture/map complete",
    )
    from .live_llm import add_arguments, forwarded_arguments

    add_arguments(p)
    p.add_argument(
        "--semantic-hz",
        type=float,
        default=1.0,
        help="Source sampling rate, not a guarantee of LLM completion rate",
    )
    args = p.parse_args()
    from ticvla.warehouse.object_io import validate_mode_args

    validate_mode_args(p, args)
    if args.perception_backend == "yoloe-realtime":
        profile = json.loads(Path(args.realtime_profile).read_text())
        args.weights = profile["weights"]
        args.render_hz = 5
    if args.semantic_hz <= 0:
        p.error("--semantic-hz must be positive")
    if args.render_hz < 5 or args.render_hz > 60 or 60 % args.render_hz:
        p.error("--render-hz must divide 60 and be at least the 5 Hz capture rate")
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    # The host desktop user creates capture/; root ML processes retain write access.
    if Path("/.dockerenv").exists() and os.getuid() == 0:
        os.chown(
            root,
            int(os.environ.get("TICVLA_HOST_UID", "1000")),
            int(os.environ.get("TICVLA_HOST_GID", "1000")),
        )
    (root / "index.html").write_text(HTML)
    server = ThreadingHTTPServer(
        (args.bind, args.port), functools.partial(QuietHandler, directory=str(root))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(
        f"LIVE_VIEW http://{args.bind}:{server.server_port} (also embedded in Isaac GUI)",
        flush=True,
    )
    repo = Path(__file__).resolve().parents[2]
    children = []
    logs = []
    resource_stop = threading.Event()
    if args.perception_backend == "yoloe-realtime":

        def sample_gpu():
            with (root / "gpu-metrics.jsonl").open("w", buffering=1) as stream:
                while not resource_stop.is_set():
                    try:
                        value = subprocess.check_output(
                            [
                                "nvidia-smi",
                                "--query-gpu=timestamp,memory.used,memory.total,utilization.gpu",
                                "--format=csv,noheader,nounits",
                            ],
                            text=True,
                            timeout=3,
                        ).strip()
                        stream.write(
                            json.dumps(
                                dict(
                                    wall=time.time(),
                                    whole_gpu_csv=value,
                                    scope="whole GPU, includes Isaac and unrelated processes",
                                )
                            )
                            + "\n"
                        )
                    except (OSError, subprocess.SubprocessError) as exc:
                        stream.write(
                            json.dumps(dict(wall=time.time(), error=str(exc))) + "\n"
                        )
                    resource_stop.wait(1)

        threading.Thread(target=sample_gpu, daemon=True).start()

    def launch(command, logname):
        log = (root / logname).open("w")
        logs.append(log)
        process = subprocess.Popen(
            command,
            cwd=repo,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        children.append(process)
        return process

    try:
        command = [
            sys.executable,
            "-m",
            "DynaNav.warehouse_loop.live_mapping",
            "--capture",
            str(root / "capture"),
            "--output",
            str(root / "mapping"),
            "--weights",
            args.weights,
            "--semantic-hz",
            str(args.semantic_hz),
        ]
        command.extend(forwarded_arguments(args))
        if Path("/.dockerenv").exists():
            command.append("--host-tesseract")
        mapper = launch(command, "mapping.log")
        deadline = time.monotonic() + (
            600 if args.perception_backend == "yoloe-realtime" else 180
        )
        while not (root / "mapping/ready.json").exists():
            if mapper.poll() is not None:
                raise RuntimeError("Mapping initialization failed; see mapping.log")
            if time.monotonic() > deadline:
                raise TimeoutError("Mapping initialization exceeded 180 seconds")
            time.sleep(0.2)
        command = [
            "bash",
            str(repo / "DynaNav/warehouse_loop/run_host.sh"),
            "--output",
            str(root / "capture"),
            "--capture-hz",
            "5",
            "--render-hz",
            str(args.render_hz),
            "--realtime",
            "--live-map",
            str(root / "mapping"),
            "--max-steps",
            str(args.max_steps),
        ]
        if not args.headless:
            command.append("--gui")
        if args.landmark_variant:
            command.extend(["--landmark-variant", args.landmark_variant])
        isaac = launch(command, "isaac.log")
        while isaac.poll() is None:
            if mapper.poll() is not None:
                raise RuntimeError("Mapping process failed; see mapping.log")
            time.sleep(0.5)
        capture_ok = (root / "capture/capture-report.json").exists() and not (
            root / "capture/capture-error.txt"
        ).exists()
        atomic_json(
            root / "capture-process.json",
            dict(
                exit_code=isaac.returncode if capture_ok else 1,
                finished_wall=time.time(),
            ),
        )
        if isaac.returncode or not capture_ok:
            raise RuntimeError(f"Isaac exited {isaac.returncode}; see isaac.log")
        print("Capture finished; draining queued semantic observations...", flush=True)
        deadline = time.monotonic() + 600
        while mapper.poll() is None:
            if time.monotonic() > deadline:
                raise TimeoutError(
                    "Semantic drain exceeded 600 seconds; incomplete run"
                )
            time.sleep(0.2)
        if mapper.returncode:
            raise RuntimeError("Mapping failed; see mapping.log")
        report = aggregate(root)
        if args.semantic_backend == "ocr":
            from DynaNav.warehouse_loop.live_evaluate import evaluate

            validation = evaluate(root)
            report["map_validation_pass"] = validation["pass_all"]
            report["route_coverage"] = validation["route_coverage"]
            report["pipeline_pass"] = report["pipeline_pass"] and validation["pass_all"]
        else:
            report["map_validation_scope"] = (
                "Source projection self-consistency only; no instance truth or absolute accuracy evaluation"
            )
            report["model_accuracy_evaluated"] = False
        atomic_json(root / "report.json", report)
        print(
            "REPORT",
            str(root / "report.json"),
            json.dumps(
                {
                    key: value
                    for key, value in report.items()
                    if key not in ("store", "notes")
                }
            ),
            flush=True,
        )
        if args.keep_viewer:
            print("Viewer retained; Ctrl+C to close", flush=True)
            while True:
                time.sleep(1)
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt) and (root / "report.json").exists():
            print("Viewer closed; completed reports retained.", flush=True)
            return
        atomic_json(
            root / "failure.json",
            {
                "error": str(exc),
                "type": type(exc).__name__,
                "child_exit_codes": [child.poll() for child in children],
                "all_requested_rates_met": False,
            },
        )
        raise
    finally:
        resource_stop.set()
        for child in reversed(children):
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
        for child in reversed(children):
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        for log in logs:
            log.close()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
