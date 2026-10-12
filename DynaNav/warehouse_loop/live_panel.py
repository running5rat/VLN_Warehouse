"""Isaac-only display adapter; all USD/UI calls stay on the simulation thread."""

import json
from pathlib import Path
import time


class LiveMapPanel:
    def __init__(self, root):
        import omni.ui as ui

        self.root = Path(root)
        self.last_stamp = None
        self.last_check = 0.0
        self.provider = ui.ByteImageProvider()
        self.detect_provider = ui.ByteImageProvider()
        self.accept_provider = ui.ByteImageProvider()
        self.semantic_stamp = None
        self.window = ui.Window("Live Semantic Map", width=570, height=680)
        viewport = ui.Workspace.get_window("Viewport")
        window = ui.Workspace.get_window("Live Semantic Map")
        if viewport and window:
            window.dock_in(viewport, ui.DockPosition.RIGHT, 0.40)
        with self.window.frame:
            with ui.VStack():
                self.label = ui.Label(
                    "Waiting for RGB-D / map worker...", height=55, word_wrap=True
                )
                ui.ImageWithProvider(
                    self.provider, fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT
                )
                ui.Label("YOLO detections (left) | Accepted in map (right)", height=22)
                with ui.HStack(height=210):
                    ui.ImageWithProvider(
                        self.detect_provider,
                        fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT,
                    )
                    ui.ImageWithProvider(
                        self.accept_provider,
                        fill_policy=ui.IwpFillPolicy.IWP_PRESERVE_ASPECT_FIT,
                    )
                self.semantic_label = ui.Label(
                    "Waiting for masks / boxes...", height=105, word_wrap=True
                )

    def update(self):
        from PIL import Image

        now = time.monotonic()
        if now - self.last_check < 0.10:
            return False
        self.last_check = now
        path = self.root / "map.png"
        try:
            stamp = path.stat().st_mtime_ns
            if stamp == self.last_stamp:
                return False
            with Image.open(path) as source:
                image = source.convert("RGBA")
                image.thumbnail((400, 423))
                self.provider.set_bytes_data(list(image.tobytes()), list(image.size))
            status = json.loads((self.root / "status.json").read_text())
            self.label.text = (
                f"Geometry {status['geometry_frames']} ({status['geometry_hz']:.2f} Hz) | "
                f"{'Candidates' if status.get('selection_mode') == 'instances' else 'Semantic'} {status['semantic_frames']} ({status['semantic_hz']:.2f} Hz)\n"
                f"Verified {status['verified']} | Tentative {status['tentative']} | "
                f"Semantic backlog {status['semantic_pending']} | source {status['source_time']:.1f}s"
            )
            self.last_stamp = stamp
            self.update_semantics(status)
            return True
        except (FileNotFoundError, json.JSONDecodeError):
            return False

    def update_semantics(self, status):
        from PIL import Image

        try:
            report = json.loads((self.root / "inspection.json").read_text())
            if self.semantic_stamp != report["updated_wall"]:
                for name, provider in (
                    ("detections.png", self.detect_provider),
                    ("semantic.png", self.accept_provider),
                ):
                    with Image.open(self.root / name) as source:
                        image = source.convert("RGBA")
                        image.thumbnail((360, 270))
                        provider.set_bytes_data(list(image.tobytes()), list(image.size))
                self.semantic_stamp = report["updated_wall"]
            labels = [
                r["name"] or r["category"]
                for r in report["instances"]
                if r["status"] == "accepted"
            ]
            asked = report.get("category_selection", {}).get(
                "selected_categories",
                [r["request"]["category"] for r in report["categories"]],
            )
            if report.get("category_selection", {}).get("selection_mode") == "instances":
                asked = [x["instance_id"] for x in report["category_selection"].get("selected_instances", [])]
            self.semantic_label.text = (
                f"Source {report['image_id']} | lag {max(0, status['source_time'] - report['t_obs']):.1f}s\n"
                f"LLM: {', '.join(asked) if report['llm_valid'] else str(report.get('category_selection', {}).get('status', report['llm_error']))}\nIn map: {', '.join(labels) or 'none'}\n"
                "Tentative is not a static anchor. Full per-instance results: web viewer."
            )
        except (FileNotFoundError, json.JSONDecodeError):
            pass
