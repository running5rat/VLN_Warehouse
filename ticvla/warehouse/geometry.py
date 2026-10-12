"""Metric geometry: optical RGB-D -> 3-D alignment -> local occupancy.

Body frame is FLU, transforms map column vectors from the named source to target.
Grid row increases backwards; column increases to the robot's right.
"""

import heapq
import itertools
import math

import numpy as np

UNKNOWN, FREE, OCCUPIED = -1, 0, 1


def transform_points(points, transform):
    points = np.asarray(points, dtype=np.float64)
    return points @ np.asarray(transform)[:3, :3].T + np.asarray(transform)[:3, 3]


def planar_pose(x=0.0, y=0.0, yaw=0.0):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array(
        [[c, -s, 0.0, x], [s, c, 0.0, y], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
    )


def project_depth(depth, intrinsics, T_body_camera, stride=1):
    """Depth is optical-axis Z in metres, NOT distance along the viewing ray."""
    depth = np.asarray(depth, dtype=np.float64)
    K = np.asarray(intrinsics, dtype=np.float64)
    if depth.ndim != 2 or K.shape != (3, 3) or min(K[0, 0], K[1, 1]) <= 0:
        raise ValueError("Invalid depth or camera intrinsics")
    v, u = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    z = depth[::stride, ::stride]
    valid = np.isfinite(z) & (z > 0)
    z, u, v = z[valid], u[valid], v[valid]
    points = np.stack(((u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z), -1)
    return transform_points(points, T_body_camera)


def xy_to_grid(xy, size, resolution):
    xy = np.asarray(xy)
    return np.floor(size // 2 - xy / resolution + 0.5).astype(np.int64)


def grid_to_xy(rc, size, resolution):
    return (size // 2 - np.asarray(rc)) * resolution


def line_cells(start, end, size):
    count = int(np.max(np.abs(np.asarray(end) - start))) + 1
    cells = np.rint(np.linspace(start, end, max(count, 1))).astype(int)
    valid = ((cells >= 0) & (cells < size)).all(-1)
    return cells[valid]


class LocalMapBuilder:
    def __init__(self, config):
        self.config = config
        self.frames = []
        self.last_timestamp = -float("inf")

    def add(self, *, depth, K, T_body_camera, T_odom_body, t_obs, odom_epoch):
        if t_obs <= self.last_timestamp:
            raise ValueError("RGB-D timestamps must strictly increase")
        if self.frames and self.frames[-1]["epoch"] != odom_epoch:
            self.frames.clear()
        self.last_timestamp = t_obs
        points = project_depth(depth, K, T_body_camera, self.config.depth_stride)
        self.frames.append(
            {
                "points": transform_points(points, T_odom_body),
                "origin": transform_points(
                    np.asarray(T_body_camera)[None, :3, 3], T_odom_body
                )[0],
                "t": t_obs,
                "epoch": odom_epoch,
            }
        )
        self.frames = [
            f for f in self.frames if t_obs - f["t"] <= self.config.map_max_age
        ]

    def render(self, T_odom_body, t_obs, odom_epoch):
        cfg = self.config
        size = round(cfg.map_size_m / cfg.map_resolution)
        grid = np.full((size, size), UNKNOWN, np.int8)
        observed_at = np.full((size, size), -np.inf, np.float64)
        inv = np.linalg.inv(T_odom_body)
        has_evidence = False
        for frame in self.frames:
            if frame["t"] > t_obs or frame["epoch"] != odom_epoch:
                continue
            if t_obs - frame["t"] > cfg.map_max_age:
                continue
            has_evidence = True
            points = transform_points(frame["points"], inv)
            origin = transform_points(frame["origin"][None], inv)[0]
            start = xy_to_grid(origin[:2], size, cfg.map_resolution)
            free_cells, occupied_cells = [], []
            for p in points:
                if not -0.15 <= p[2] <= cfg.obstacle_max_height:
                    continue
                if np.linalg.norm(p[:2] - origin[:2]) > cfg.map_size_m * 2:
                    continue
                end = xy_to_grid(p[:2], size, cfg.map_resolution)
                cells = line_cells(start, end, size)
                is_obstacle = p[2] >= cfg.obstacle_min_height
                endpoint_in_map = ((end >= 0) & (end < size)).all()
                if is_obstacle and endpoint_in_map:
                    occupied_cells.append(end)
                    cells = cells[:-1]
                free_cells.extend(cells)
            # Same-frame hits win over free rays; later evidence supersedes older hits.
            for cells, value in ((free_cells, FREE), (occupied_cells, OCCUPIED)):
                if len(cells):
                    r, c = np.asarray(cells).T
                    grid[r, c] = value
                    observed_at[r, c] = frame["t"]
        if has_evidence:
            # The currently occupied robot footprint is proprioceptive free-space
            # evidence. Camera blind space OUTSIDE it remains unknown. Never erase a hit.
            rc = np.indices(grid.shape).transpose(1, 2, 0)
            xy = grid_to_xy(rc, size, cfg.map_resolution)
            own_footprint = np.linalg.norm(
                xy, axis=-1
            ) <= cfg.footprint_radius + cfg.map_resolution / math.sqrt(2)
            own_footprint &= grid != OCCUPIED
            grid[own_footprint] = FREE
            observed_at[own_footprint] = t_obs
        age = np.where(
            np.isfinite(observed_at),
            np.maximum(t_obs - observed_at, 0),
            cfg.map_max_age,
        )
        return grid, age.astype(np.float32)


def render_map_image(grid, resolution, image_size=448):
    from PIL import Image, ImageDraw

    colors = np.array([[120, 120, 120], [245, 245, 245], [30, 30, 30]], dtype=np.uint8)
    canvas = Image.fromarray(colors[np.asarray(grid) + 1]).resize(
        (image_size, image_size - 40), Image.Resampling.NEAREST
    )
    result = Image.new("RGB", (image_size, image_size), "white")
    result.paste(canvas, (0, 0))
    draw = ImageDraw.Draw(result)
    x, y = image_size // 2, (image_size - 40) // 2
    draw.line((x, y + 8, x, y - 18), fill="red", width=3)
    draw.polygon([(x, y - 22), (x - 5, y - 12), (x + 5, y - 12)], fill="red")
    draw.text(
        (4, image_size - 37), "white=free black=occupied gray=unknown", fill="black"
    )
    draw.text(
        (4, image_size - 20),
        f"{len(grid) * resolution:g}m x {len(grid) * resolution:g}m; up=+x left=+y; red=robot",
        fill="black",
    )
    return result


def traversable(grid, resolution, footprint_radius):
    from scipy.ndimage import binary_dilation

    radius = math.ceil(footprint_radius / resolution)
    r, c = np.mgrid[-radius : radius + 1, -radius : radius + 1]
    disk = r * r + c * c <= radius * radius
    return ~binary_dilation(grid != FREE, structure=disk, border_value=1)


def collision_free(points, grid, resolution, footprint_radius):
    if len(points) == 0:
        return True
    allowed = traversable(grid, resolution, footprint_radius)
    cells = xy_to_grid(points, len(grid), resolution)
    if not ((cells >= 0) & (cells < len(grid))).all():
        return False
    for a, b in itertools.pairwise(cells):
        rc = line_cells(a, b, len(grid))
        if not allowed[rc[:, 0], rc[:, 1]].all():
            return False
    return bool(allowed[cells[:, 0], cells[:, 1]].all())


def astar_reference(grid, goal_xy, resolution, footprint_radius):
    allowed = traversable(grid, resolution, footprint_radius)
    size = len(grid)
    start = tuple(xy_to_grid([0.0, 0.0], size, resolution))
    goal = tuple(xy_to_grid(goal_xy, size, resolution))
    for cell in (start, goal):
        if not all(0 <= x < size for x in cell) or not allowed[cell]:
            return np.empty((0, 2), np.float32)
    queue, costs, parent = [(0.0, start)], {start: 0.0}, {}
    directions = [(r, c) for r in (-1, 0, 1) for c in (-1, 0, 1) if r or c]
    while queue:
        _, here = heapq.heappop(queue)
        if here == goal:
            path = [here]
            while path[-1] != start:
                path.append(parent[path[-1]])
            xy = grid_to_xy(path[::-1], size, resolution).astype(np.float32)
            xy[0] = 0.0
            # Visibility shortcuts; smooth with a cubic only if footprint-safe.
            simplified, i = [xy[0]], 0
            while i < len(xy) - 1:
                j = len(xy) - 1
                while j > i + 1 and not collision_free(
                    xy[[i, j]], grid, resolution, footprint_radius
                ):
                    j -= 1
                simplified.append(xy[j])
                i = j
            simplified = np.asarray(simplified)
            if len(simplified) >= 3:
                from scipy.interpolate import CubicSpline

                distances = np.r_[
                    0.0, np.cumsum(np.linalg.norm(np.diff(simplified, axis=0), axis=-1))
                ]
                smooth = CubicSpline(distances, simplified, axis=0, bc_type="natural")(
                    np.linspace(
                        0, distances[-1], max(2, int(distances[-1] / (resolution / 2)))
                    )
                )
                if collision_free(smooth, grid, resolution, footprint_radius):
                    return smooth.astype(np.float32)
            return simplified.astype(np.float32)
        for dr, dc in directions:
            nxt = here[0] + dr, here[1] + dc
            if not all(0 <= x < size for x in nxt) or not allowed[nxt]:
                continue
            if (
                dr
                and dc
                and (
                    not allowed[here[0] + dr, here[1]]
                    or not allowed[here[0], here[1] + dc]
                )
            ):
                continue
            cost = costs[here] + math.hypot(dr, dc)
            if cost < costs.get(nxt, float("inf")):
                costs[nxt] = cost
                parent[nxt] = here
                heapq.heappush(queue, (cost + math.dist(nxt, goal), nxt))
    return np.empty((0, 2), np.float32)


def sample_reference(points, count, spacing):
    output, mask = np.zeros((count, 2), np.float32), np.zeros(count, bool)
    if len(points) < 2:
        return output, mask
    arc = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=-1))]
    keep = np.r_[True, np.diff(arc) > 1.0e-6]
    arc, points = arc[keep], np.asarray(points)[keep]
    if len(points) < 2:
        return output, mask
    n = min(count, max(2, math.ceil(arc[-1] / spacing) + 1))
    sample_arc = np.linspace(0.0, min(arc[-1], spacing * (count - 1)), n)
    output[:n] = np.stack(
        [np.interp(sample_arc, arc, points[:, i]) for i in range(2)], -1
    )
    mask[:n] = True
    return output, mask


def align_reference(
    points, mask, T_source, T_current, source_epoch, current_epoch, trim=True
):
    if source_epoch != current_epoch:
        raise ValueError("Cannot align different odom epochs")
    points, mask = np.asarray(points), np.asarray(mask, bool)
    if not np.array_equal(mask, np.arange(len(mask)) < mask.sum()):
        raise ValueError("Reference mask must be a prefix")
    transformed = transform_points(
        np.c_[points[mask], np.zeros(mask.sum())], np.linalg.inv(T_current) @ T_source
    )[:, :2]
    if trim and len(transformed) > 1:
        a, b = transformed[:-1], transformed[1:]
        ab = b - a
        fraction = np.clip(
            -(a * ab).sum(-1) / np.maximum((ab * ab).sum(-1), 1.0e-10), 0, 1
        )
        projections = a + fraction[:, None] * ab
        i = int(np.argmin(np.linalg.norm(projections, axis=-1)))
        # Project onto the nearest segment, retain only its unexecuted suffix.
        transformed = np.concatenate([projections[i : i + 1], transformed[i + 1 :]])
        if (
            len(transformed) > 1
            and np.linalg.norm(transformed[1] - transformed[0]) < 1.0e-5
        ):
            transformed = transformed[1:]
    output = np.zeros_like(points)
    output[: len(transformed)] = transformed
    return output, np.arange(len(mask)) < len(transformed)


def delay_features(T_source, T_current, age):
    relative = np.linalg.inv(T_source) @ T_current
    yaw = math.atan2(relative[1, 0], relative[0, 0])
    return np.array(
        [age, relative[0, 3], relative[1, 3], math.sin(yaw), math.cos(yaw)], np.float32
    )
