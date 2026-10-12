"""Serializable MVP configuration, independent of legacy training settings."""

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class WarehouseConfig:
    pipeline: str = "warehouse_mvp"
    model_path: str = "${TICVLA_BASE_MODEL_PATH}"
    vlm_checkpoint: str | None = None
    reference_checkpoint: str | None = None
    manifest: str = "data/warehouse/manifest.json"
    cache_dir: str = "data/warehouse/cache"
    output_dir: str = "outputs/warehouse"
    device: str = "cuda"
    seed: int = 42
    image_size: int = 448
    history_offsets: list[float] = field(default_factory=lambda: [3.0, 2.0, 1.0, 0.0])
    map_size_m: float = 16.0
    map_resolution: float = 0.1
    map_max_age: float = 5.0
    depth_stride: int = 8
    obstacle_min_height: float = 0.15
    obstacle_max_height: float = 2.0
    footprint_radius: float = 0.35
    num_control_points: int = 8
    num_reference_points: int = 32
    spline_degree: int = 3
    reference_spacing: float = 0.25
    reference_hidden_dim: int = 256
    context_queries: int = 16
    hidden_dim: int = 512
    num_layers: int = 3
    num_heads: int = 8
    horizon: int = 30
    action_dt: float = 0.1
    coordinate_scale: float = 10.0
    flow_steps: int = 4
    action_head: str = "flow"
    lora_rank: int = 16
    lora_alpha: float = 32.0
    max_new_tokens: int = 384
    batch_size: int = 2
    epochs: int = 10
    learning_rate: float = 1.0e-4
    reference_learning_rate: float = 3.0e-5
    weight_decay: float = 1.0e-3
    grad_clip: float = 1.0
    accumulation_steps: int = 1
    max_steps: int | None = None
    reference_loss_weights: dict = field(
        default_factory=lambda: {
            "text": 1.0,
            "points": 1.0,
            "length": 1.0,
            "validity": 1.0,
            "anchors": 0.1,
        }
    )
    delay_seconds: list[float] = field(
        default_factory=lambda: [0.0, 0.5, 1.0, 2.0, 4.0]
    )
    max_snapshot_age: float = 3.0
    reference_noise_std: float = 0.1
    condition_dropout: float = 0.1
    max_speed: float = 1.5
    max_acceleration: float = 2.0

    def __post_init__(self):
        if self.pipeline != "warehouse_mvp":
            raise ValueError("Expected pipeline=warehouse_mvp")
        if (
            self.num_heads < 1
            or self.hidden_dim % self.num_heads
            or not 1 <= self.spline_degree < self.num_control_points
        ):
            raise ValueError("Invalid attention dimensions or spline degree")
        if self.action_head not in ("flow", "query"):
            raise ValueError("action_head must be flow or query")
        for name in (
            "map_size_m",
            "map_resolution",
            "action_dt",
            "coordinate_scale",
            "horizon",
            "flow_steps",
            "num_reference_points",
            "reference_spacing",
            "batch_size",
            "accumulation_steps",
            "context_queries",
            "lora_rank",
            "map_max_age",
            "num_layers",
            "epochs",
            "depth_stride",
            "max_new_tokens",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_steps is not None and self.max_steps < 1:
            raise ValueError("max_steps must be positive when provided")
        if not 0 <= self.condition_dropout <= 1 or self.reference_noise_std < 0:
            raise ValueError("Invalid condition augmentation")
        if self.footprint_radius < 0 or self.max_snapshot_age < 0:
            raise ValueError("Footprint radius and maximum age must be nonnegative")
        if any(x < 0 for x in self.delay_seconds + self.history_offsets):
            raise ValueError("History offsets and delays must be nonnegative")

    def to_dict(self):
        return asdict(self)


def load_config(path):
    import yaml

    text = Path(path).read_text()
    raw = json.loads(text) if str(path).endswith(".json") else yaml.safe_load(text)

    def expand(value):
        if isinstance(value, str):
            return os.path.expanduser(os.path.expandvars(value))
        if isinstance(value, dict):
            return {k: expand(v) for k, v in value.items()}
        if isinstance(value, list):
            return [expand(v) for v in value]
        return value

    return WarehouseConfig(**expand(raw))
