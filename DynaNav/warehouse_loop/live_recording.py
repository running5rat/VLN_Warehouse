"""Bounded background disk writer; Isaac arrays are copied before handing off."""

import json
import queue
import threading
import time

import numpy as np
from ticvla.warehouse.collect import EpisodeRecorder


class AsyncEpisodeRecorder:
    def __init__(self, output, metadata):
        self.recorder = EpisodeRecorder(output, metadata)
        self.queue = queue.Queue(maxsize=4)
        self.count = 0
        self.last_t = -float("inf")
        self.error = None
        self.closed = False
        self.metrics = []
        try:
            self.metric_stream = (
                self.recorder.output.parent / "writer-performance.jsonl"
            ).open("w", buffering=1)
        except BaseException:
            self.recorder.close()
            raise
        self.thread = threading.Thread(
            target=self._write, name="rgbd-disk-writer", daemon=True
        )
        self.thread.start()

    def _write(self):
        with self.metric_stream as stream:
            while True:
                record = self.queue.get()
                try:
                    if record is None:
                        return
                    if self.error is not None:
                        continue
                    started = time.perf_counter()
                    self.recorder.append(**record)
                    metric = dict(
                        wall=time.time(),
                        service_s=time.perf_counter() - started,
                        t_obs=record["t_obs"],
                    )
                    self.metrics.append(metric)
                    stream.write(json.dumps(metric) + "\n")
                except BaseException as exc:
                    self.error = exc
                finally:
                    self.queue.task_done()

    def append(self, **record):
        if self.closed:
            raise RuntimeError("Recorder closed")
        if self.error is not None:
            raise RuntimeError("RGB-D disk writer failed") from self.error
        stamp = record["t_obs"]
        if not np.isfinite(stamp) or stamp <= self.last_t:
            raise ValueError("Nonmonotonic capture timestamp")
        if record.get("privileged") is not None:
            raise ValueError("Live recorder does not accept privileged buffers")
        # Replicator may reuse returned CPU buffers on its next render.
        record = {
            key: value.copy() if isinstance(value, np.ndarray) else value
            for key, value in record.items()
        }
        self.queue.put(record)
        self.count += 1
        self.last_t = stamp

    def close(self):
        if self.closed:
            return
        self.closed = True
        self.queue.put(None)
        self.thread.join()
        self.recorder.close()
        if self.error is not None:
            raise RuntimeError("RGB-D disk writer failed") from self.error
