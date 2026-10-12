"""Persistent isolated worker. JSON/NPZ files are the only inference boundary."""

import json
import os
from pathlib import Path
import subprocess
import time
import uuid
import numpy as np
from ..schema import file_sha256


def add_arguments(parser):
    parser.add_argument("--selection-mode", choices=("instances", "categories"), default="instances")
    parser.add_argument(
        "--realtime-profile",
        default="/data/ticvla/models/warehouse-realtime/selected.json",
    )
    parser.add_argument(
        "--perception-backend", choices=("yoloe-ppocr", "florence2", "yoloe-realtime")
    )
    parser.add_argument(
        "--perception-python",
        default="/data/ticvla/envs/landmark-perception/bin/python",
    )
    parser.add_argument(
        "--florence-model", default="/data/ticvla/models/Florence-2-large"
    )


class PerceptionClient:
    def __init__(
        self, name, output, weights, device="cuda:0", python=None, florence_model=None
    ):
        if name == "yoloe-realtime":
            raise ValueError(
                "yoloe-realtime requires the dedicated asynchronous or single-frame entry"
            )
        self.root = Path(output).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.root = self.root / ("worker-" + uuid.uuid4().hex)
        self.root.mkdir()
        self.log = (self.root / "worker.log").open("w")
        args = [
            python or "/data/ticvla/envs/landmark-perception/bin/python",
            "-m",
            "ticvla.warehouse.perception.worker",
            "--backend",
            name,
            "--root",
            str(self.root),
            "--weights",
            str(weights),
            "--device",
            device,
            "--florence-model",
            florence_model or "/data/ticvla/models/Florence-2-large",
        ]
        env = dict(
            os.environ,
            PYTHONPATH=str(Path(__file__).resolve().parents[3]),
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
            PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK="True",
        )
        self.process = subprocess.Popen(
            args, stdout=self.log, stderr=subprocess.STDOUT, env=env
        )
        ready = self._wait(self.root / "ready.json", 900)
        self.provenance = ready["provenance"]

    def _wait(self, path, timeout):
        start = time.monotonic()
        while not path.exists():
            if self.process.poll() is not None:
                raise RuntimeError(
                    "Perception worker exited; see " + str(self.root / "worker.log")
                )
            if time.monotonic() - start > timeout:
                self.close()
                raise TimeoutError("Perception timeout: " + str(path))
            time.sleep(0.05)
        value = json.loads(path.read_text())
        if value.get("error"):
            raise RuntimeError(value["error"])
        return value

    def perceive(self, observation, categories):
        if file_sha256(observation["rgb"]) != observation["rgb_hash"]:
            raise ValueError("Source RGB changed")
        job = self.root / uuid.uuid4().hex
        job.mkdir()
        # Whitelist: no Oracle labels, old map identities or evaluation truth.
        source = {k: observation[k] for k in ("image_id", "rgb", "rgb_hash")}
        temp = job / "request.tmp"
        temp.write_text(json.dumps(dict(observation=source, categories=categories)))
        temp.replace(job / "request.json")
        result = self._wait(job / "result.json", 600)
        if result["source"] != source or result["categories"] != categories:
            raise ValueError("Perception source/request mismatch")
        for instance in result["instances"]:
            # Index is relative to this request's prompt list, never an identity
            # or a text read from the object. The name/index mapping is bijective.
            instance["category_index"] = categories.index(instance["category"])
            path = job / instance["mask_file"]
            if file_sha256(path) != instance["mask_sha256"]:
                raise ValueError("Perception mask hash mismatch")
            with np.load(path, allow_pickle=False) as data:
                instance["mask"] = data["mask"]
        result["provenance"]["artifact_directory"] = str(job)
        return result

    segment_instances = perceive

    def close(self):
        if getattr(self, "process", None) is not None and self.process.poll() is None:
            (self.root / "stop").touch()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=10)
        if getattr(self, "log", None):
            self.log.close()

    def __del__(self):
        self.close()
