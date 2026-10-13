"""Vectorized ray rasterization of retained source 3D geometry (no grid warping)."""

import numpy as np
from ..geometry import transform_points, xy_to_grid, grid_to_xy


def render_local(builder, pose, stamp, epoch):
    cfg = builder.config
    size = round(cfg.map_size_m / cfg.map_resolution)
    grid = np.full((size, size), -1, np.int8)
    seen = np.full((size, size), -np.inf)
    inverse = np.linalg.inv(pose)
    evidence = False
    for frame in builder.frames:
        if frame["epoch"] != epoch or not 0 <= stamp - frame["t"] <= cfg.map_max_age:
            continue
        evidence = True
        points = transform_points(frame["points"], inverse)
        origin = transform_points(frame["origin"][None], inverse)[0]
        points = points[
            (points[:, 2] >= -0.15)
            & (points[:, 2] <= cfg.obstacle_max_height)
            & (np.linalg.norm(points[:, :2] - origin[:2], axis=1) <= cfg.map_size_m * 2)
        ]
        if not len(points):
            continue
        start = xy_to_grid(origin[:2], size, cfg.map_resolution)
        ends = xy_to_grid(points[:, :2], size, cfg.map_resolution)
        inside = ((ends >= 0) & (ends < size)).all(1)
        obstacle = (points[:, 2] >= cfg.obstacle_min_height) & inside
        counts = np.abs(ends - start).max(1) + 1
        step = np.arange(int(counts.max()))[None, :]
        fraction = step / np.maximum(counts[:, None] - 1, 1)
        cells = np.rint(
            start[None, None] + fraction[..., None] * (ends - start)[:, None]
        ).astype(np.int32)
        valid = (step < counts[:, None]) & ((cells >= 0) & (cells < size)).all(2)
        valid &= ~(obstacle[:, None] & (step == counts[:, None] - 1))
        rc = cells[valid]
        grid[rc[:, 0], rc[:, 1]] = 0
        seen[rc[:, 0], rc[:, 1]] = frame["t"]
        hits = ends[obstacle]
        grid[hits[:, 0], hits[:, 1]] = 1
        seen[hits[:, 0], hits[:, 1]] = frame["t"]
    if evidence:
        xy = grid_to_xy(
            np.indices(grid.shape).transpose(1, 2, 0), size, cfg.map_resolution
        )
        own = (
            np.linalg.norm(xy, axis=-1)
            <= cfg.footprint_radius + cfg.map_resolution / np.sqrt(2)
        ) & (grid != 1)
        grid[own] = 0
        seen[own] = stamp
    return grid, np.where(
        np.isfinite(seen), np.maximum(stamp - seen, 0), cfg.map_max_age
    ).astype("float32")
