"""One-command closed-loop navigation, separate from fixed-loop mapping."""

import argparse
import functools
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from .live_metrics import atomic_json

HTML = """<!doctype html><meta charset="utf-8"><title>Warehouse Navigation</title>
<style>body{font:16px sans-serif;background:#18222c;color:white;margin:20px}img{max-width:48%;vertical-align:top}pre{white-space:pre-wrap}button{padding:10px}</style>
<h2>v0.6 navigation — model / geometry baseline</h2><pre id="status"></pre>
<img id="rgb"><img id="map"><p>跟随：点击“冻结图像”后在原图拖框，提交首帧目标绑定。绑定只针对所示源帧。</p>
<button id="freeze">冻结图像</button><button id="bind">提交框选</button><button id="cancel">停止任务</button><pre id="roi"></pre><pre id="vlm"></pre>
<script>
let source=null,locked=false,start=null,box=null;let rgb=document.querySelector('#rgb');
document.querySelector('#freeze').onclick=()=>{locked=!locked;};
rgb.onpointerdown=e=>{if(!locked)return;let r=rgb.getBoundingClientRect();start=[(e.clientX-r.left)*rgb.naturalWidth/r.width,(e.clientY-r.top)*rgb.naturalHeight/r.height];e.preventDefault();};
rgb.onpointerup=e=>{if(!start)return;let r=rgb.getBoundingClientRect();let end=[(e.clientX-r.left)*rgb.naturalWidth/r.width,(e.clientY-r.top)*rgb.naturalHeight/r.height];box=[Math.min(start[0],end[0]),Math.min(start[1],end[1]),Math.max(start[0],end[0]),Math.max(start[1],end[1])];document.querySelector('#roi').textContent=JSON.stringify({source,box});start=null;};
document.querySelector('#bind').onclick=async()=>{if(!source||!box)return;let r=await fetch('/target-roi',{method:'POST',body:JSON.stringify({...source,bbox:box})});document.querySelector('#roi').textContent=await r.text();locked=false;};
document.querySelector('#cancel').onclick=()=>fetch('/cancel',{method:'POST',body:'{}'});
async function update(){try{let s=await(await fetch('navigation/status.json?t='+Date.now())).json();document.querySelector('#status').textContent=JSON.stringify(s,null,2);document.querySelector('#map').src='navigation/map.png?t='+s.updated_wall;
if(!locked){let src=await(await fetch('navigation/source.json?t='+Date.now())).json();rgb.onload=()=>{source=src;};rgb.src='capture/raw/rgb/'+src.image_id.split(':').pop()+'.png';}
let v=await(await fetch('navigation/last-vlm.json?t='+Date.now())).json();document.querySelector('#vlm').textContent=v.text||JSON.stringify(v);
}catch(e){}setTimeout(update,500)}update();</script>"""


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        root = Path(self.directory)
        try:
            data = json.loads(
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
            )
            if self.path == "/target-roi":
                if (
                    not isinstance(data.get("image_id"), str)
                    or len(data.get("bbox", [])) != 4
                ):
                    raise ValueError("image_id and pixel bbox required")
                import numpy as np

                b = np.asarray(data["bbox"], float)
                if not np.isfinite(b).all() or not (
                    0 <= b[0] < b[2] <= 960 and 0 <= b[1] < b[3] <= 720
                ):
                    raise ValueError("Invalid source ROI")
                atomic_json(root / "navigation/target-roi.json", data)
            elif self.path == "/cancel":
                atomic_json(root / "cancel.json", data)
            else:
                raise ValueError("Unknown endpoint")
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Accepted; source binding will be checked by worker.")
        except (ValueError, KeyError) as e:
            self.send_error(400, str(e))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--task-mode", choices=("landmark", "explore", "follow"), required=True
    )
    p.add_argument(
        "--execution-backend", choices=("model", "geometry-baseline"), default="model"
    )
    p.add_argument("--target-landmark")
    p.add_argument("--map")
    p.add_argument(
        "--map-anchor-image",
        help="Explicit old-map observation known to overlap the new initial view; geometric verification required",
    )
    p.add_argument("--target-roi", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--headless", action="store_true")
    p.add_argument("--keep-viewer", action="store_true")
    p.add_argument("--port", type=int, default=8766)
    p.add_argument("--bind", default="127.0.0.1")
    p.add_argument("--max-steps", type=int, default=3600)
    p.add_argument("--model", default="/data/ticvla/models/InternVL3-1B")
    p.add_argument("--weights", default="/data/ticvla/models/yoloe-v8s-seg.pt")
    p.add_argument("--landmark-variant", default="original")
    a = p.parse_args()
    if a.task_mode == "landmark" and not a.target_landmark:
        p.error("landmark requires --target-landmark UID or text")
    if a.task_mode != "follow" and a.target_roi:
        p.error("--target-roi only applies to follow")
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    if os.getuid() == 0:
        os.chown(root, 1000, 1000)
    (root / "navigation").mkdir()
    os.chmod(root / "navigation", 0o777)
    if a.target_roi:
        atomic_json(
            root / "navigation/target-roi.json", json.loads(a.target_roi.read_text())
        )
    options = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    options["output"] = str(root)
    atomic_json(root / "config.json", options)
    (root / "index.html").write_text(HTML)
    server = ThreadingHTTPServer(
        (a.bind, a.port), functools.partial(Handler, directory=str(root))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"NAVIGATION_VIEW http://{a.bind}:{server.server_port}", flush=True)
    repo = Path(__file__).resolve().parents[2]
    children = []
    logs = []
    import importlib.metadata

    versions = {}
    for name in (
        "torch",
        "transformers",
        "ultralytics",
        "lm-format-enforcer",
        "numpy",
        "paddleocr",
        "paddlepaddle",
        "tensorrt",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    atomic_json(
        root / "environment.json",
        dict(python=sys.version, executable=sys.executable, packages=versions),
    )
    resource_stop = threading.Event()

    def resources():
        with (root / "gpu-metrics.jsonl").open("w", buffering=1) as f:
            while not resource_stop.is_set():
                try:
                    value = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=memory.used,utilization.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                        timeout=3,
                    )
                    f.write(
                        json.dumps(
                            dict(
                                wall=time.time(),
                                whole_gpu_memory_utilization=value.strip(),
                            )
                        )
                        + "\n"
                    )
                except (OSError, subprocess.SubprocessError):
                    pass
                resource_stop.wait(1)

    threading.Thread(target=resources, daemon=True).start()

    def launch(cmd, name):
        log = (root / name).open("w")
        logs.append(log)
        proc = subprocess.Popen(
            cmd, cwd=repo, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        children.append(proc)
        return proc

    try:
        runtime = launch(
            [
                sys.executable,
                "-m",
                "ticvla.warehouse.navigation.runtime",
                "--config",
                str(root / "config.json"),
            ],
            "navigation.log",
        )
        deadline = time.monotonic() + 600
        while not (root / "navigation/ready.json").exists():
            if runtime.poll() is not None:
                raise RuntimeError(
                    "Navigation initialization failed; see navigation.log"
                )
            if time.monotonic() > deadline:
                raise TimeoutError("Navigation initialization timed out")
            time.sleep(0.2)
        cmd = [
            "bash",
            str(repo / "DynaNav/warehouse_loop/run_host.sh"),
            "--output",
            str(root / "capture"),
            "--capture-hz",
            "5",
            "--render-hz",
            "5",
            "--realtime",
            "--live-map",
            str(root / "navigation"),
            "--navigation-root",
            str(root / "navigation"),
            "--navigation-task",
            a.task_mode,
            "--landmark-variant",
            a.landmark_variant,
            "--max-steps",
            str(a.max_steps),
        ]
        if not a.headless:
            cmd.append("--gui")
        isaac = launch(cmd, "isaac.log")
        while isaac.poll() is None:
            if runtime.poll() is not None:
                raise RuntimeError("Navigation process failed; see navigation.log")
            time.sleep(0.5)
        atomic_json(
            root / "capture-process.json",
            dict(exit_code=isaac.returncode, finished_wall=time.time()),
        )
        runtime.wait(timeout=90)
        report = (
            json.loads((root / "navigation/report.json").read_text())
            if (root / "navigation/report.json").exists()
            else {}
        )
        if (root / "capture/capture-report.json").exists():
            report["capture"] = json.loads(
                (root / "capture/capture-report.json").read_text()
            )
            report["geometry_complete"] = (
                report.get("geometry_frames") == report["capture"]["frames"]
            )
            report["observed_movement"] = report["capture"]["distance_m"] > 0.05
        executed_path = root / "navigation/executed-commands.jsonl"
        executed = (
            [json.loads(x) for x in executed_path.read_text().splitlines()]
            if executed_path.exists()
            else []
        )
        report["executed_translation_commands"] = sum(
            abs(x["v"]) > 0.001 for x in executed
        )
        report["executed_rotation_commands"] = sum(
            abs(x["w"]) > 0.001 for x in executed
        )
        report["observed_movement"] = bool(
            report.get("observed_movement") and report["executed_translation_commands"]
        )
        report["model_motion_pass"] = (
            a.execution_backend == "model" and report["observed_movement"]
        )
        report.update(isaac_exit=isaac.returncode, runtime_exit=runtime.returncode)
        atomic_json(root / "report.json", report)
        print(
            json.dumps(
                {k: v for k, v in report.items() if k not in ("store", "task_state")},
                indent=2,
            ),
            flush=True,
        )
        if isaac.returncode != 0 or runtime.returncode != 0:
            raise RuntimeError(
                f"Navigation run failed: Isaac={isaac.returncode}, runtime={runtime.returncode}; report and logs preserved in {root}"
            )
        if a.keep_viewer:
            print("Viewer remains open; Ctrl+C to close.", flush=True)
            while True:
                time.sleep(1)
    finally:
        resource_stop.set()
        for proc in children:
            if proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
        for proc in children:
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
        for log in logs:
            log.close()
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
