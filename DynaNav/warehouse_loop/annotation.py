"""Human-driven capture and source-bound annotation. No navigation model is run."""

import argparse
import functools
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import signal
import subprocess
import threading
import time

import numpy as np
from PIL import Image

from ticvla.warehouse.config import WarehouseConfig
from ticvla.warehouse.geometry import LocalMapBuilder, render_map_image
from ticvla.warehouse.navigation.geometry import render_local
from ticvla.warehouse.dataset_tools.core import (
    approve, extract, export_project, finalize_project, read_json, read_lines, sample_context, write_json,
    prepare_projection_cameras,
)


class AnnotationSession:
    def __init__(self, root, mode):
        self.root = Path(root)
        self.mode = mode
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.source = None
        self.geometry_frames = 0
        self.error = None
        self.state = "replay" if mode == "replay" else "idle"
        self.process = None

    def event(self, filename, data):
        data = {**data, "monotonic": time.monotonic(), "wall": time.time()}
        if self.source:
            data.update(image_id=self.source["image_id"], t_obs=self.source["t_obs"])
        with (self.root / filename).open("a") as f:
            f.write(json.dumps(data, ensure_ascii=False, allow_nan=False) + "\n")
        return data

    def command(self, data):
        if self.mode != "capture":
            raise ValueError("Replay cannot drive the robot")
        v, w = float(data.get("v", 0)), float(data.get("w", 0))
        if not np.isfinite([v, w]).all() or v < 0 or v > .3 or abs(w) > .6:
            raise ValueError("Expected 0 <= v <= .3 and |w| <= .6; no reverse")
        if self.state != "recording" or self.source is None:
            v = w = 0.
        now = time.monotonic()
        cmd = dict(v=v, w=w, created_monotonic=now, valid_until=now+.3,
                   backend="manual", reason="human_teleoperation",
                   safety_policy="human_exploration",
                   source_image_id=(self.source or {}).get("image_id"),
                   odom_epoch=(self.source or {}).get("odom_epoch"))
        write_json(self.root / "navigation/command.json", cmd)
        self.event("manual-commands.jsonl", cmd)
        return cmd

    def status(self):
        with self.lock:
            return dict(mode=self.mode, state=self.state, source=self.source,
                        geometry_frames=self.geometry_frames, error=self.error,
                        isaac_exit=self.process.poll() if self.process else None,
                        project_ready=(self.root / "annotation/project.json").exists())

    def post(self, endpoint, data):
        with self.lock:
            if endpoint == "/command":
                return self.command(data)
            if endpoint == "/record":
                if self.mode != "capture" or self.state in ("finished", "replay"):
                    raise ValueError("Capture is no longer active")
                if self.process and self.process.poll() is not None:
                    raise ValueError("Isaac has already exited; start a new capture")
                state = data["state"]
                if state not in ("recording", "paused", "finished"):
                    raise ValueError("Unknown recording state")
                if state == "recording" and not self.source:
                    raise ValueError("Wait for synchronized RGB-D before recording")
                if state == "recording" and time.monotonic()-self.source.get("source_monotonic", time.monotonic()) > .6:
                    raise ValueError("Wait for a fresh source frame after renderer warm-up")
                self.state = state
                self.event("recording-events.jsonl", dict(state=state))
                if state != "recording": self.command({})
                if state == "finished":
                    write_json(self.root / "navigation/task-finished.json", {"source": "human"})
                return self.status()
            if endpoint == "/event":
                allowed = {"start", "intersection", "turn_started", "turn_completed",
                           "target_visible", "stopped", "arrival_confirmed", "note"}
                if self.state != "recording" or data.get("kind") not in allowed:
                    raise ValueError("Events require a recording and a supported kind")
                if not self.source or data.get("image_id") != self.source["image_id"]:
                    raise ValueError("Displayed source changed; retry on the current frame")
                if any(e.get("image_id") == self.source["image_id"] and e["kind"] == data["kind"]
                       for e in read_lines(self.root / "annotation-events.jsonl")):
                    raise ValueError("Duplicate event on the same source frame")
                return self.event("annotation-events.jsonl", dict(kind=data["kind"], note=str(data.get("note", ""))[:600], source="human"))
            project_root = self.root / "annotation"
            if endpoint == "/approve":
                return approve(project_root, int(data["index"]), data)
            if endpoint == "/draft":
                i = int(data["index"])
                sample = sample_context(project_root, i)
                if data.get("source_binding") != sample["provenance"]["source_binding"]:
                    raise ValueError("Draft source mismatch")
                write_json(project_root / sample["folder"] / "draft.json", data)
                return {"saved": True}
            if endpoint == "/export":
                return export_project(project_root, self.root / "exports" / ("revision-" + str(time.time_ns())))
            if endpoint == "/finalize":
                if data.get("human_reviewed") is not True:
                    raise ValueError("Explicit human task/scene review required")
                return finalize_project(project_root, data.get("annotator", ""))
            raise ValueError("Unknown endpoint")

    def geometry(self):
        # Match the existing navigation geometry consumer's bounded live workload.
        cfg = WarehouseConfig(depth_stride=32, map_max_age=2.0, map_resolution=0.05)
        builder = LocalMapBuilder(cfg)
        raw = self.root / "capture/raw"
        try:
            while not self.stop.is_set():
                if not (raw / "episode.json").exists():
                    self.stop.wait(.1)
                    continue
                meta = read_json(raw / "episode.json")
                frames = read_lines(raw / "frames.jsonl")
                for i in range(self.geometry_frames, len(frames)):
                    f = frames[i]
                    builder.add(depth=np.load(raw / f["depth"], allow_pickle=False), K=f["K"],
                                T_body_camera=f["T_body_camera"], T_odom_body=f["T_odom_body"],
                                t_obs=f["t_obs"], odom_epoch=f["odom_epoch"])
                    started = time.monotonic()
                    grid, age = render_local(builder, f["T_odom_body"], f["t_obs"], f["odom_epoch"])
                    nav = self.root / "navigation"
                    with (nav / "safety.tmp.npz").open("wb") as stream:
                        np.savez(stream, grid=grid, pose=f["T_odom_body"], resolution=cfg.map_resolution,
                                 odom_epoch=f["odom_epoch"], source_monotonic=f["source_monotonic"])
                    (nav / "safety.tmp.npz").replace(nav / "safety.npz")
                    render_map_image(grid, cfg.map_resolution, 640).save(nav / "map.tmp.png", compress_level=0)
                    (nav / "map.tmp.png").replace(nav / "map.png")
                    source = dict(image_id=f"{meta['episode_id']}:{i:08d}", t_obs=f["t_obs"],
                                  odom_epoch=f["odom_epoch"], T_odom_body=f["T_odom_body"],
                                  image=f"capture/raw/{f['rgb']}",
                                  source_monotonic=f["source_monotonic"])
                    with self.lock:
                        self.source = source
                        self.geometry_frames = i + 1
                    write_json(nav / "source.json", source)
                    with (self.root / "geometry-performance.jsonl").open("a") as metric:
                        metric.write(json.dumps(dict(image_id=source["image_id"],
                            service_s=time.monotonic()-started,
                            source_age_s=time.monotonic()-f["source_monotonic"]))+"\n")
                if self.process and self.process.poll() is not None and self.geometry_frames == len(read_lines(raw / "frames.jsonl")):
                    # Wait until the async writer has closed; all complete lines consumed.
                    break
                self.stop.wait(.03)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            with self.lock:
                self.state = "error"
                self.command({})


class Handler(SimpleHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def json_response(self, value, code=200):
        payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        from urllib.parse import urlparse, parse_qs
        url = urlparse(self.path)
        try:
            if url.path == "/status":
                return self.json_response(self.server.session.status())
            if url.path == "/sample":
                index = int(parse_qs(url.query)["index"][0])
                with self.server.session.lock:
                    root = self.server.session.root / "annotation"
                    s = sample_context(root, index)
                    draft = root / s["folder"] / "draft.json"
                    s["draft"] = read_json(draft) if draft.exists() else None
                return self.json_response(s)
            return super().do_GET()
        except (ValueError, KeyError, IndexError, OSError) as exc:
            return self.json_response({"error": str(exc)}, 400)

    def do_POST(self):
        try:
            if not self.headers.get("Content-Type", "").startswith("application/json"):
                raise ValueError("application/json required")
            n = int(self.headers.get("Content-Length", 0))
            if not 0 < n <= 1024 * 1024:
                raise ValueError("Invalid request size")
            data = json.loads(self.rfile.read(n))
            self.json_response(self.server.session.post(self.path, data))
        except (ValueError, KeyError, IndexError, TypeError, OSError) as exc:
            self.json_response({"error": str(exc)}, 400)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("capture", "replay"), required=True)
    p.add_argument("--instruction")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--episode", type=Path, help="Existing raw episode to extract for replay")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--port", type=int, default=8767)
    p.add_argument("--bind", default="127.0.0.1")
    p.add_argument("--max-steps", type=int, default=216000)
    p.add_argument("--exit-after-capture", action="store_true", help="For acquisition diagnostics; no annotation approval")
    a = p.parse_args()
    if a.mode == "capture" and not (a.instruction or "").strip():
        p.error("capture requires --instruction")
    root = a.output.resolve()
    if a.mode == "capture":
        root.mkdir(parents=True, exist_ok=False)
        (root / "navigation").mkdir()
        if os.getuid() == 0:
            os.chown(root, 1000, 1000)
        os.chmod(root / "navigation", 0o777)
        write_json(root / "navigation/task.json", dict(task_id=root.name, instruction=a.instruction,
                                                       task_type="INSTRUCTION", hard_constraints=[]))
        write_json(root / "session.json", dict(mode="manual", static=True, scene="original",
                                               task_validated=False, training=False))
    else:
        root.mkdir(parents=True, exist_ok=True)
        if not (root / "annotation/project.json").exists():
            if not a.episode: p.error("replay requires an existing project or --episode")
            extract(a.episode, root / "annotation")
        prepare_projection_cameras(root / "annotation")
    ui = Path(__file__).with_name("annotation.html")
    (root / "index.html").write_text(ui.read_text())
    (root / "reference_projection.js").write_text(ui.with_name("reference_projection.js").read_text())
    session = AnnotationSession(root, a.mode)
    server = ThreadingHTTPServer((a.bind, a.port), functools.partial(Handler, directory=str(root)))
    server.session = session
    threading.Thread(target=server.serve_forever, daemon=True).start()
    print(f"ANNOTATION_VIEW http://{a.bind}:{server.server_port}", flush=True)
    log = thread = None
    try:
        if a.mode == "capture":
            repo = Path(__file__).resolve().parents[2]
            cmd = ["bash", str(repo / "DynaNav/warehouse_loop/run_host.sh"),
                   "--output", str(root / "capture"), "--capture-hz", "5", "--render-hz", "5",
                   "--realtime", "--live-map", str(root / "navigation"),
                   "--navigation-root", str(root / "navigation"), "--navigation-task", "instruction",
                   "--max-steps", str(a.max_steps)]
            if not a.headless: cmd.append("--gui")
            log = (root / "isaac.log").open("w")
            session.process = subprocess.Popen(cmd, cwd=repo, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            thread = threading.Thread(target=session.geometry, daemon=True)
            thread.start()
            while session.process.poll() is None:
                if session.error: raise RuntimeError(session.error)
                time.sleep(.2)
            thread.join(timeout=60)
            if thread.is_alive(): raise RuntimeError("Geometry consumer did not finish")
            if session.process.returncode: raise RuntimeError("Isaac failed; inspect isaac.log")
            if session.error: raise RuntimeError(session.error)
            report = dict(isaac_exit=session.process.returncode, geometry_frames=session.geometry_frames,
                          capture_frames=len(read_lines(root / "capture/raw/frames.jsonl")),
                          human_approved=False, task_validated=False)
            report["geometry_complete"] = report["geometry_frames"] == report["capture_frames"]
            write_json(root / "capture-status.json", report)
            if read_lines(root / "recording-events.jsonl"):
                extract(root / "capture/raw/episode.json", root / "annotation")
                session.state = "replay"
            else:
                session.state = "no_human_recording"
            if a.exit_after_capture: return
        while True: time.sleep(.5)
    except KeyboardInterrupt:
        pass
    finally:
        session.stop.set()
        if session.process and session.process.poll() is None:
            with session.lock:
                session.state = "finished"
                session.command({})
                write_json(root / "navigation/task-finished.json", {"reason": "server_exit"})
            try: session.process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(session.process.pid, signal.SIGTERM)
        server.shutdown()
        if log: log.close()


if __name__ == "__main__":
    main()
