"""Single-robot submaps with transactional metadata and immutable geometry.

Coordinates remain in the source odom/submap. Registration changes transforms,
never original observations. Oracle stores cannot be opened as deployment stores.
"""

import hashlib
import itertools
import json
import math
import sqlite3
import uuid
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_erosion
from scipy.spatial import cKDTree

from ..config import WarehouseConfig
from ..geometry import LocalMapBuilder, project_depth, transform_points, xy_to_grid
from ..schema import file_sha256, fingerprint


@dataclass
class MappingConfig:
    voxel_size: float = 0.05
    min_depth_points: int = 20
    min_confidence: float = 0.25
    association_distance: float = 0.20
    association_overlap: float = 0.35
    registration_residual: float = 0.15
    registration_overlap: float = 0.55
    min_landmark_spread: float = 0.20
    # Explicit deployment policy for physically installed shelf placards only.
    # OCR alone never makes arbitrary objects static. Default remains disabled.
    fixed_shelf_markers: bool = False

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Mapping configuration must be finite")
        if (
            min(
                self.voxel_size,
                self.min_depth_points,
                self.association_distance,
                self.registration_residual,
                self.min_landmark_spread,
            )
            <= 0
        ):
            raise ValueError("Mapping lengths/counts must be positive")
        for value in (
            self.min_confidence,
            self.association_overlap,
            self.registration_overlap,
        ):
            if not 0 <= value <= 1:
                raise ValueError("Mapping probabilities must be in [0,1]")


@dataclass
class SourceObservation:
    image_id: str
    rgb: str
    depth: str
    K: list
    T_body_camera: list
    T_odom_body: list
    t_obs: float
    odom_epoch: str
    submap_id: str
    task_id: str


@dataclass
class SemanticEvent:
    message_id: str
    image_id: str
    category: str
    source: str = "manual"
    roi: list | None = None
    name: str | None = None
    static_confirmed: bool = False
    kind: str = "object"
    request_id: str | None = None
    navigation_role: str | None = None
    reason: str | None = None
    text_evidence: list = field(default_factory=list)


def rigid(value):
    value = np.asarray(value, dtype=float)
    if (
        value.shape != (4, 4)
        or not np.isfinite(value).all()
        or not np.allclose(value[3], [0, 0, 0, 1])
        or not np.allclose(value[:3, :3].T @ value[:3, :3], np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(value[:3, :3]), 1, atol=1e-5)
    ):
        raise ValueError("Expected a finite proper rigid transform")
    return value


def voxelize(points, resolution):
    points = np.asarray(points, dtype=float).reshape(-1, 3)
    if not np.isfinite(points).all():
        raise ValueError("Nonfinite point cloud")
    if not len(points):
        return points
    cells = np.floor(points / resolution).astype(np.int64)
    low, high = cells.min(0), cells.max(0)
    shape = [int(b) - int(a) + 1 for a, b in zip(low, high)]
    if math.prod(shape) <= np.iinfo(np.int64).max:
        # Bijective scalar cell key: identical first-point selection, cheaper than
        # NumPy's lexicographic structured-array sort for every dense mask.
        relative = cells - low
        keys = (relative[:, 0] * shape[1] + relative[:, 1]) * shape[2] + relative[:, 2]
        _, ids = np.unique(keys, return_index=True)
    else:
        _, ids = np.unique(cells, axis=0, return_index=True)
    return points[np.sort(ids)]


def overlap(a, b, distance):
    if not len(a) or not len(b):
        return 0.0
    if np.any(a.min(0) - b.max(0) > distance) or np.any(b.min(0) - a.max(0) > distance):
        return 0.0
    # Both directions prevent a tiny unrelated patch being accepted as a match.
    return float(
        min(
            (cKDTree(b).query(a)[0] <= distance).mean(),
            (cKDTree(a).query(b)[0] <= distance).mean(),
        )
    )


def semantic_points(mask, depth, observation, config, *, diagnostics=None):
    """Reject mask boundaries and depth-discontinuous background before voxelizing.

    Small boundary errors otherwise receive disproportionate weight after voxel
    downsampling. Filtering uses only source depth, never instance truth.
    """
    if np.shape(mask) != np.shape(depth):
        raise ValueError("Mask/source dimensions differ")
    valid = binary_erosion(mask, iterations=1) & np.isfinite(depth) & (depth > 0)
    if diagnostics is not None:
        diagnostics.update(
            mask_pixels=int(np.count_nonzero(mask)),
            valid_interior_depth_points=int(valid.sum()),
            filtered_depth_points=0,
            voxel_points=0,
        )
    if valid.sum() < config.min_depth_points:
        raise ValueError("Insufficient interior mask depth")
    values = depth[valid]
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    valid &= np.abs(depth - median) <= max(0.25, 4.5 * float(mad))
    if diagnostics is not None:
        diagnostics["filtered_depth_points"] = int(valid.sum())
        diagnostics["depth_retention"] = float(
            valid.sum() / max(diagnostics["valid_interior_depth_points"], 1)
        )
    if valid.sum() < config.min_depth_points:
        raise ValueError("Insufficient depth after discontinuity filtering")
    points = project_depth(
        np.where(valid, depth, 0), observation["K"], observation["T_body_camera"]
    )
    points = voxelize(
        transform_points(points, observation["T_odom_body"]), config.voxel_size
    )
    if diagnostics is not None:
        diagnostics["voxel_points"] = len(points)
    return points


class MapStore:
    def __init__(self, root, config=None, *, oracle=False, array_compression_level=6):
        if (
            not isinstance(array_compression_level, int)
            or not 0 <= array_compression_level <= 9
        ):
            raise ValueError("NPZ compression level must be an integer in [0,9]")
        # Operational I/O choice only; array format and map semantics are unchanged.
        self.array_compression_level = array_compression_level
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "arrays").mkdir(exist_ok=True)
        self.db = sqlite3.connect(self.root / "map.sqlite")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS observations(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS submaps(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS landmarks(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS constraints(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS text_events(id TEXT PRIMARY KEY, payload TEXT NOT NULL);
        """)
        saved = dict(self.db.execute("SELECT key,value FROM meta"))
        self.oracle = oracle
        if saved:
            if saved.get("schema") != "1" or saved["oracle"] != str(bool(oracle)):
                self.close()
                raise ValueError("Map schema or Oracle/deployment mode mismatch")
            self.config = MappingConfig(**json.loads(saved["config"]))
            if config and asdict(config) != asdict(self.config):
                self.close()
                raise ValueError("Map configuration is immutable; use another store")
        else:
            self.config = config or MappingConfig()
            with self.db:
                self.db.executemany(
                    "INSERT INTO meta VALUES (?,?)",
                    [
                        ("schema", "1"),
                        ("oracle", str(bool(oracle))),
                        ("version", "0"),
                        ("config", json.dumps(asdict(self.config))),
                    ],
                )

    def close(self):
        self.db.close()

    @property
    def version(self):
        return int(
            self.db.execute("SELECT value FROM meta WHERE key='version'").fetchone()[0]
        )

    def _bump(self):
        self.db.execute(
            "UPDATE meta SET value=CAST(value AS INTEGER)+1 WHERE key='version'"
        )

    def _put(self, table, key, value):
        assert table in {
            "observations",
            "submaps",
            "events",
            "text_events",
            "landmarks",
            "constraints",
        }
        self.db.execute(
            f"INSERT OR REPLACE INTO {table} VALUES (?,?)",
            (key, json.dumps(value, allow_nan=False)),
        )

    def get(self, table, key):
        assert table in {
            "observations",
            "submaps",
            "events",
            "text_events",
            "landmarks",
            "constraints",
        }
        row = self.db.execute(
            f"SELECT payload FROM {table} WHERE id=?", (key,)
        ).fetchone()
        return json.loads(row[0]) if row else None

    def records(self, table):
        assert table in {
            "observations",
            "submaps",
            "events",
            "text_events",
            "landmarks",
            "constraints",
        }
        return [
            json.loads(row[0])
            for row in self.db.execute(f"SELECT payload FROM {table} ORDER BY id")
        ]

    def _array(self, **arrays):
        # Unique immutable file first, then SQL commit; a crash can only leave an
        # unreferenced file, never a committed reference to a partially written file.
        name = f"arrays/{uuid.uuid4().hex}.npz"
        tmp = self.root / (name + ".tmp")
        with tmp.open("xb") as stream:
            if self.array_compression_level == 6:
                np.savez_compressed(stream, **arrays)
            else:
                with zipfile.ZipFile(
                    stream,
                    "w",
                    compression=zipfile.ZIP_DEFLATED,
                    compresslevel=self.array_compression_level,
                ) as archive:
                    for key, array in arrays.items():
                        with archive.open(
                            key + ".npy", "w", force_zip64=True
                        ) as member:
                            np.lib.format.write_array(
                                member, np.asarray(array), allow_pickle=False
                            )
        tmp.replace(self.root / name)
        return name

    def points(self, record):
        with np.load(self.root / record["geometry"], allow_pickle=False) as arrays:
            return arrays["points"].copy()

    def add_observation(self, observation):
        obs = asdict(observation)
        if not all(obs[k] for k in ("image_id", "odom_epoch", "submap_id", "task_id")):
            raise ValueError("Source identifiers must be nonempty")
        if not math.isfinite(obs["t_obs"]):
            raise ValueError("Nonfinite source time")
        for key in ("T_body_camera", "T_odom_body"):
            rigid(obs[key])
        from PIL import Image

        rgb = Path(obs["rgb"]).resolve()
        depth_path = Path(obs["depth"]).resolve()
        depth = np.load(depth_path, allow_pickle=False)
        with Image.open(rgb) as image:
            if depth.ndim != 2 or image.size != (depth.shape[1], depth.shape[0]):
                raise ValueError("RGB/depth dimensions differ")
        obs.update(
            rgb=str(rgb),
            depth=str(depth_path),
            rgb_hash=file_sha256(rgb),
            depth_hash=file_sha256(depth_path),
        )
        prior = self.get("observations", obs["image_id"])
        if prior:
            if any(prior[k] != v for k, v in obs.items()):
                raise ValueError("Observation ID reused with different content")
            return prior
        submap = self.get("submaps", obs["submap_id"])
        if submap and submap["odom_epoch"] != obs["odom_epoch"]:
            raise ValueError("New odom epoch requires a new submap")
        same = [
            o
            for o in self.records("observations")
            if o["submap_id"] == obs["submap_id"]
        ]
        if same and obs["t_obs"] <= max(o["t_obs"] for o in same):
            raise ValueError("Nonmonotonic submap source time")
        body_points = project_depth(depth, obs["K"], obs["T_body_camera"], stride=4)
        points = voxelize(
            transform_points(body_points, obs["T_odom_body"]), self.config.voxel_size
        )
        obs["geometry"] = self._array(points=points, depth=depth)
        with self.db:
            if not submap:
                first = not self.records("submaps")
                self._put(
                    "submaps",
                    obs["submap_id"],
                    {
                        "submap_id": obs["submap_id"],
                        "odom_epoch": obs["odom_epoch"],
                        "T_map_submap": np.eye(4).tolist() if first else None,
                        "status": "root" if first else "unregistered",
                        "constraint_id": None,
                    },
                )
            self._put("observations", obs["image_id"], obs)
            self._bump()
        return obs

    def source_arrays(self, image_id):
        obs = self.get("observations", image_id)
        if obs is None:
            raise ValueError("Unknown source image")
        if (
            file_sha256(obs["rgb"]) != obs["rgb_hash"]
            or file_sha256(obs["depth"]) != obs["depth_hash"]
        ):
            raise ValueError("Source content changed")
        with np.load(self.root / obs["geometry"], allow_pickle=False) as arrays:
            return obs, arrays["depth"].copy()

    def observed_landmarks(self, *, category=None, text=None):
        """Searchable observation layer, explicitly separate from static MapView.

        Positions remain in each record's source submap coordinates. Callers
        must use validated submap transforms before comparing different epochs.
        """
        result = []
        for record in self.records("landmarks"):
            if record["status"] == "revoked":
                continue
            if category and record["category"].casefold() != category.casefold():
                continue
            if (
                text
                and text.casefold()
                not in " ".join(record.get("identity_candidates", [])).casefold()
            ):
                continue
            result.append(
                dict(
                    record,
                    identity_status=record.get("identity_status", "unknown"),
                    text_evidence=record.get("text_evidence", []),
                )
            )
        return result

    def landmark_view(self):
        """Derived identity/selection view; database and static MapView unchanged."""
        from .postprocess import LandmarkViewCache
        if not hasattr(self, "_landmark_view_cache"):
            self._landmark_view_cache = LandmarkViewCache()
        return self._landmark_view_cache.evaluate(
            self.observed_landmarks(), lambda image_id: self.get("observations", image_id))

    def prepare_projection(self, image_id, mask):
        """Transient source-bound geometry; does not admit an event or landmark."""
        obs, depth = self.source_arrays(image_id)
        mask = np.asarray(mask, dtype=bool)
        diagnostics = {}
        result = dict(owner=self, source=fingerprint(obs), config=asdict(self.config),
                      mask_hash=hashlib.sha256(mask.tobytes()).hexdigest(),
                      diagnostics=diagnostics, points=np.empty((0, 3)), error=None)
        try:
            result["points"] = semantic_points(mask, depth, obs, self.config, diagnostics=diagnostics)
            result["points"].setflags(write=False)
        except ValueError as exc:
            result["error"] = str(exc)
        return result

    def consume(self, event, mask=None, confidence=0.0, *, segmentation=None, prepared_projection=None):
        payload = asdict(event)
        prior = self.get("events", event.message_id)
        if prior:
            if asdict(SemanticEvent(**prior["event"])) != payload:
                raise ValueError("Event ID reused with different content")
            return {**prior, "duplicate": True}
        result = {
            "event": payload,
            "status": "rejected",
            "reason": None,
            "landmark_uid": None,
            "segmentation": segmentation,
        }
        points = None
        obs = self.get("observations", event.image_id)
        try:
            if (
                event.source not in ("manual", "llm", "oracle", "perception")
                or (event.source == "oracle") != self.oracle
            ):
                raise ValueError("Oracle/deployment event mismatch")
            if not event.message_id or not event.category.strip():
                raise ValueError("Missing event ID/category")
            if event.kind != "object":
                raise ValueError(
                    "Topology/relation requires geometry verification, not a mask"
                )
            obs, depth = self.source_arrays(event.image_id)
            if mask is None:
                raise ValueError("No detection")
            mask = np.asarray(mask, dtype=bool)
            if mask.shape != depth.shape:
                raise ValueError("Mask/source dimensions differ")
            unavailable = (
                confidence is None
                and (segmentation or {}).get("backend") == "florence2"
            )
            if not unavailable and (
                confidence is None
                or not math.isfinite(confidence)
                or not self.config.min_confidence <= confidence <= 1.0
            ):
                raise ValueError("Low or nonfinite segmentation confidence")
            if event.roi is not None:
                roi = np.asarray(event.roi, dtype=float)
                if roi.shape != (4,) or not np.isfinite(roi).all():
                    raise ValueError("Invalid ROI")
                x1, y1, x2, y2 = roi
                if not (
                    0 <= x1 < x2 <= depth.shape[1] and 0 <= y1 < y2 <= depth.shape[0]
                ):
                    raise ValueError("ROI outside source image")
                area = np.zeros_like(mask)
                area[int(y1) : int(math.ceil(y2)), int(x1) : int(math.ceil(x2))] = True
                if (mask & area).sum() / max(mask.sum(), 1) < 0.5:
                    raise ValueError("Mask does not match prompt ROI")
            diagnostics = {}
            result["projection_diagnostics"] = diagnostics
            if prepared_projection is None:
                points = semantic_points(mask, depth, obs, self.config, diagnostics=diagnostics)
            else:
                p = prepared_projection
                if (p["owner"] is not self or p["source"] != fingerprint(obs)
                    or p["config"] != asdict(self.config)
                    or p["mask_hash"] != hashlib.sha256(mask.tobytes()).hexdigest()):
                    raise ValueError("Prepared projection source/mask mismatch")
                diagnostics.update(p["diagnostics"])
                if p["error"]:
                    raise ValueError(p["error"])
                points = p["points"]
            if len(points) < 3:
                raise ValueError("Degenerate semantic geometry")
            # A text prediction of staticness is not sufficient evidence.
            static = bool(
                event.static_confirmed and event.source in ("manual", "oracle")
            )
            if (
                self.config.fixed_shelf_markers
                and event.source == "perception"
                and event.category == "shelf_marker"
                and event.static_confirmed
            ):
                static = True
            dynamic = event.category.lower() in {
                "person",
                "human",
                "pedestrian",
                "forklift",
                "robot",
                "car",
                "cart",
                "hand truck",
                "platform trolley",
                "platform cart",
                "shopping cart",
                "trolley",
                "push cart",
                "truck",
                "vehicle",
            }
            if dynamic:
                static = False
            matches = []
            for lm in self.records("landmarks"):
                if (
                    lm["submap_id"] != obs["submap_id"]
                    or lm["category"] != event.category
                    or lm["status"] == "revoked"
                ):
                    continue
                score = overlap(
                    points, self.points(lm), self.config.association_distance
                )
                if event.name and lm["name"] and event.name != lm["name"]:
                    if score >= self.config.association_overlap:
                        raise ValueError(
                            "Conflicting landmark identity for observed geometry"
                        )
                    continue
                if score >= self.config.association_overlap:
                    matches.append((score, lm))
            matches.sort(key=lambda x: x[0], reverse=True)
            if len(matches) > 1 and matches[0][0] - matches[1][0] < 0.15:
                raise ValueError("Ambiguous landmark association")
            lm = matches[0][1] if matches else None
            if lm and event.image_id in lm["observation_ids"]:
                result.update(
                    status="accepted",
                    reason="Source evidence already recorded",
                    landmark_uid=lm["landmark_uid"],
                )
                with self.db:
                    self._put("events", event.message_id, result)
                return result
            uid = lm["landmark_uid"] if lm else uuid.uuid4().hex
            if lm:
                points = voxelize(
                    np.concatenate((self.points(lm), points)), self.config.voxel_size
                )
                static = (static or lm["static_confirmed"]) and not dynamic
            position = points.mean(0)
            record = {
                "landmark_uid": uid,
                "submap_id": obs["submap_id"],
                "category": event.category,
                "name": event.name or (lm["name"] if lm else None),
                "observation_ids": (lm["observation_ids"] if lm else [])
                + [event.image_id],
                "geometry": self._array(points=points),
                "position": position.tolist(),
                "covariance": (np.cov(points.T) / len(points)).tolist(),
                "orientation": None,
                "orientation_kind": "unobservable",
                "symmetry": "unknown",
                "static_confirmed": static,
                "status": "verified" if static else "tentative",
                "first_seen": min(lm["first_seen"], obs["t_obs"])
                if lm
                else obs["t_obs"],
                "last_seen": max(lm["last_seen"], obs["t_obs"]) if lm else obs["t_obs"],
                "confidence": None if confidence is None else float(confidence),
                "position_kind": "observed_surface_centroid",
                "potentially_dynamic": dynamic,
            }
            result.update(status="accepted", landmark_uid=uid)
            identity_records = []
            if (segmentation or {}).get("backend") in (
                "yoloe-ppocr",
                "florence2",
                "yoloe-realtime",
            ):
                from .identity import update_identity, mark_duplicate_texts

                update_identity(record, lm, obs, event.text_evidence)
                identity_records = mark_duplicate_texts(
                    [
                        r
                        for r in self.records("landmarks")
                        if r["landmark_uid"] != uid
                        and r.get("identity_status")
                        and r["status"] != "revoked"
                    ]
                    + [record]
                )
            with self.db:
                for identity_record in identity_records:
                    self._put(
                        "landmarks", identity_record["landmark_uid"], identity_record
                    )
                self._put("landmarks", uid, record)
                self._put("events", event.message_id, result)
                self._bump()
            return result
        except (ValueError, OSError) as exc:
            result["reason"] = str(exc)
        with self.db:
            self._put("events", event.message_id, result)
        return result

    def attach_text_evidence(self, event):
        """Append late same-source text without reprojecting or counting a new view."""
        from .identity import update_identity, mark_duplicate_texts
        from ..perception.common import normalize_text, polygon_mask

        payload = json.loads(json.dumps(event, allow_nan=False))
        key = payload["message_id"]
        previous = self.db.execute(
            "SELECT payload FROM text_events WHERE id=?", (key,)
        ).fetchone()
        if previous:
            prior = json.loads(previous[0])
            if prior["event"] != payload:
                raise ValueError("Text event ID reused with different content")
            return {**prior, "duplicate": True}
        result = dict(event=payload, status="rejected", reason=None, landmark_uid=None)
        try:
            parent = self.get("events", payload["parent_event_id"])
            if not parent or parent["status"] != "accepted":
                raise ValueError("Text parent instance is absent or rejected")
            if parent["event"]["image_id"] != payload["image_id"]:
                raise ValueError("Text source differs from parent instance")
            obs, depth = self.source_arrays(payload["image_id"])
            if any(payload[k] != obs[k] for k in ("rgb_hash", "depth_hash")):
                raise ValueError("Text source hash mismatch")
            lm = self.get("landmarks", parent["landmark_uid"])
            if lm is None or lm["status"] == "revoked":
                raise ValueError("Text parent landmark unavailable")
            texts = json.loads(json.dumps(payload["texts"]))
            for text in texts:
                polygon_mask(text["polygon"], depth.shape)
                score = text.get("confidence")
                if text.get("status") != "bound" or not normalize_text(text["text"]):
                    raise ValueError("Only unambiguous bound text may update identity")
                if score is not None and (
                    not math.isfinite(score) or score < 0.8 or score > 1
                ):
                    raise ValueError("Text confidence is invalid or insufficient")
                text["normalized"] = normalize_text(text["text"])
            record = dict(lm)
            update_identity(record, lm, obs, texts, evidence_id=key)
            changed = mark_duplicate_texts(
                [
                    r
                    for r in self.records("landmarks")
                    if r["landmark_uid"] != lm["landmark_uid"]
                    and r.get("identity_status")
                    and r["status"] != "revoked"
                ]
                + [record]
            )
            result.update(status="accepted", landmark_uid=lm["landmark_uid"])
            with self.db:
                for other in changed:
                    self._put("landmarks", other["landmark_uid"], other)
                self._put("landmarks", lm["landmark_uid"], record)
                self.db.execute(
                    "INSERT INTO text_events VALUES (?,?)",
                    (key, json.dumps(result, allow_nan=False)),
                )
                self._bump()
            return result
        except (ValueError, KeyError, OSError) as exc:
            result["reason"] = str(exc)
        with self.db:
            self.db.execute(
                "INSERT INTO text_events VALUES (?,?)",
                (key, json.dumps(result, allow_nan=False)),
            )
        return result

    def register(self, source, target):
        """Conservative RANSAC over category/name candidates; no GT correspondences.

        Centroids only propose transforms. Symmetric point-cloud overlap validates
        them; insufficient common surfaces deliberately cause rejection.
        """
        src, dst = self.get("submaps", source), self.get("submaps", target)
        if not src or not dst or source == target or dst["T_map_submap"] is None:
            raise ValueError(
                "Registration requires distinct source and registered target"
            )
        if src["status"] != "unregistered":
            raise ValueError("Revoke existing registration before replacing it")
        a = [
            x
            for x in self.records("landmarks")
            if x["submap_id"] == source and x["status"] == "verified"
        ]
        b = [
            x
            for x in self.records("landmarks")
            if x["submap_id"] == target and x["status"] == "verified"
        ]
        candidates = [
            (i, j)
            for i, x in enumerate(a)
            for j, y in enumerate(b)
            if x["category"] == y["category"]
            and not (x["name"] and y["name"] and x["name"] != y["name"])
        ]
        if len(candidates) >= 3 and math.comb(len(candidates), 3) > 3000:
            return {
                "status": "rejected",
                "reason": "Candidate ambiguity exceeds verification budget",
                "hypotheses": 0,
            }
        hypotheses = []
        for triple in itertools.combinations(candidates, 3):
            ia, ib = zip(*triple)
            if len(set(ia)) != 3 or len(set(ib)) != 3:
                continue
            pa = np.array([a[i]["position"][:2] for i in ia])
            pb = np.array([b[i]["position"][:2] for i in ib])
            if (
                min(
                    np.linalg.svd(pa - pa.mean(0), compute_uv=False)[-1],
                    np.linalg.svd(pb - pb.mean(0), compute_uv=False)[-1],
                )
                < self.config.min_landmark_spread
            ):
                continue
            u, _, vt = np.linalg.svd((pa - pa.mean(0)).T @ (pb - pb.mean(0)))
            rot = vt.T @ np.diag([1, np.linalg.det(vt.T @ u.T)]) @ u.T
            trans = pb.mean(0) - rot @ pa.mean(0)
            residual = float(np.linalg.norm(pa @ rot.T + trans - pb, axis=1).max())
            if residual > self.config.registration_residual:
                continue
            tf = np.eye(4)
            tf[:2, :2], tf[:2, 3] = rot, trans
            scores = [
                overlap(
                    transform_points(self.points(a[i]), tf),
                    self.points(b[j]),
                    self.config.association_distance,
                )
                for i, j in triple
            ]
            if min(scores) < self.config.registration_overlap:
                continue
            if any(np.allclose(tf, h[1], atol=0.03) for h in hypotheses):
                continue
            hypotheses.append((float(np.mean(scores)), tf, residual, triple))
        hypotheses.sort(key=lambda h: h[0], reverse=True)
        if not hypotheses or (
            len(hypotheses) > 1 and hypotheses[0][0] - hypotheses[1][0] < 0.15
        ):
            return {
                "status": "rejected",
                "reason": "Insufficient or ambiguous geometric evidence",
                "hypotheses": len(hypotheses),
            }
        score, tf, residual, pairs = hypotheses[0]
        cid = uuid.uuid4().hex
        constraint = {
            "constraint_id": cid,
            "source": source,
            "target": target,
            "T_target_source": tf.tolist(),
            "overlap": score,
            "residual": residual,
            "pairs": [[a[i]["landmark_uid"], b[j]["landmark_uid"]] for i, j in pairs],
            "active": True,
        }
        src.update(
            T_map_submap=(rigid(dst["T_map_submap"]) @ tf).tolist(),
            status="registered",
            constraint_id=cid,
        )
        with self.db:
            self._put("constraints", cid, constraint)
            self._put("submaps", source, src)
            self._bump()
        return {"status": "accepted", **constraint}

    def revoke(self, constraint_id):
        constraint = self.get("constraints", constraint_id)
        if not constraint:
            raise ValueError("Unknown constraint")
        # Invalidate descendants: their transforms depended on this edge.
        pending = [constraint["source"]]
        with self.db:
            while pending:
                sid = pending.pop()
                for edge in self.records("constraints"):
                    if edge["active"] and (
                        edge["source"] == sid or edge["target"] == sid
                    ):
                        edge["active"] = False
                        self._put("constraints", edge["constraint_id"], edge)
                        if edge["target"] == sid:
                            pending.append(edge["source"])
                sm = self.get("submaps", sid)
                sm.update(T_map_submap=None, status="unregistered", constraint_id=None)
                self._put("submaps", sid, sm)
            self._bump()

    def revoke_landmark(self, landmark_uid, reason):
        landmark = self.get("landmarks", landmark_uid)
        if landmark is None or not reason.strip():
            raise ValueError("Known landmark and revocation reason required")
        for edge in self.records("constraints"):
            if edge["active"] and any(landmark_uid in pair for pair in edge["pairs"]):
                self.revoke(edge["constraint_id"])
        landmark.update(status="revoked", revocation_reason=reason)
        with self.db:
            self._put("landmarks", landmark_uid, landmark)
            self._bump()

    def view(self, image_id, cfg=None):
        """Current local occupancy plus registered, verified static landmarks.

        Old generic geometry is never revived as current free space. Current
        observed free cells invalidate overlapping historical occupied pixels in
        this view; live occupied pixels always win. Raw evidence remains stored.
        """
        cfg = cfg or WarehouseConfig()
        obs, _ = self.source_arrays(image_id)
        sm = self.get("submaps", obs["submap_id"])
        builder = LocalMapBuilder(cfg)
        history = sorted(
            (
                o
                for o in self.records("observations")
                if o["submap_id"] == obs["submap_id"]
                and 0 <= obs["t_obs"] - o["t_obs"] <= cfg.map_max_age
            ),
            key=lambda o: o["t_obs"],
        )
        for frame in history:
            _, depth = self.source_arrays(frame["image_id"])
            builder.add(
                depth=depth,
                K=frame["K"],
                T_body_camera=frame["T_body_camera"],
                T_odom_body=frame["T_odom_body"],
                t_obs=frame["t_obs"],
                odom_epoch=frame["odom_epoch"],
            )
        grid, age = builder.render(obs["T_odom_body"], obs["t_obs"], obs["odom_epoch"])
        landmarks = []
        aliases = {}

        def canonical(uid):
            while uid in aliases:
                uid = aliases[uid]
            return uid

        for edge in self.records("constraints"):
            if edge["active"]:
                for source_uid, target_uid in edge["pairs"]:
                    source_uid, target_uid = (
                        canonical(source_uid),
                        canonical(target_uid),
                    )
                    if source_uid != target_uid:
                        aliases[source_uid] = target_uid
        visible = {}
        for lm in self.records("landmarks"):
            if lm["status"] != "verified":
                continue
            lsm = self.get("submaps", lm["submap_id"])
            if lm["submap_id"] == obs["submap_id"]:
                # Restrict replay to its causal prefix.
                sources = [self.get("observations", x) for x in lm["observation_ids"]]
                if any(x["t_obs"] > obs["t_obs"] for x in sources):
                    continue
                tf = np.linalg.inv(rigid(obs["T_odom_body"]))
            elif sm["T_map_submap"] is not None and lsm["T_map_submap"] is not None:
                tf = np.linalg.inv(
                    rigid(sm["T_map_submap"]) @ rigid(obs["T_odom_body"])
                ) @ rigid(lsm["T_map_submap"])
            else:
                continue
            points = transform_points(self.points(lm), tf)
            points = points[
                (points[:, 2] >= cfg.obstacle_min_height)
                & (points[:, 2] <= cfg.obstacle_max_height)
            ]
            rc = xy_to_grid(points[:, :2], len(grid), cfg.map_resolution)
            rc = rc[((rc >= 0) & (rc < len(grid))).all(-1)]
            if len(rc):
                r, c = rc.T
                unknown = grid[r, c] == -1
                grid[r[unknown], c[unknown]] = 1
            uid = canonical(lm["landmark_uid"])
            if uid in visible:
                visible[uid]["aliases"].append(lm["landmark_uid"])
                visible[uid]["observation_ids"] = sorted(
                    set(visible[uid]["observation_ids"] + lm["observation_ids"])
                )
                continue
            record = {
                "landmark_uid": uid,
                "aliases": [lm["landmark_uid"]],
                "category": lm["category"],
                "name": lm["name"],
                "position": transform_points(np.array([lm["position"]]), tf)[
                    0
                ].tolist(),
                "orientation_kind": lm["orientation_kind"],
                "observation_ids": lm["observation_ids"],
            }
            visible[uid] = record
            landmarks.append(record)
        return {
            "grid": grid,
            "age": age,
            "landmarks": landmarks,
            "map_version": str(self.version),
            "submap_id": sm["submap_id"],
            "registered": sm["T_map_submap"] is not None,
            "oracle": self.oracle,
            "map_interface": "rendered_mapview_baseline",
        }

    def summary(self):
        return {
            "schema_version": 1,
            "map_version": self.version,
            "oracle": self.oracle,
            "config": asdict(self.config),
            **{
                t: self.records(t)
                for t in (
                    "submaps",
                    "landmarks",
                    "events",
                    "text_events",
                    "constraints",
                )
            },
        }
