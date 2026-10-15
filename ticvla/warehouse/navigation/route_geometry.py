"""Conservative local corridor observations, using only the observed occupancy grid."""

import math
import numpy as np
from ..geometry import (
    astar_reference,
    traversable,
    grid_to_xy,
    xy_to_grid,
    transform_points,
)


def corridor_observation(grid, resolution, footprint):
    safe = traversable(grid, resolution, footprint)

    # Look for continuous free rays from candidate corridor centres, not hidden topology.
    def extent(origin, direction, cap):
        distance = np.arange(0, cap + resolution / 2, resolution)
        points = np.asarray(origin) + distance[:, None] * np.asarray(direction)
        cells = xy_to_grid(points, len(grid), resolution)
        within = ((cells >= 0) & (cells < len(grid))).all(1)
        valid = np.zeros(len(cells), bool)
        valid[within] = safe[cells[within, 0], cells[within, 1]]
        first = np.flatnonzero(~valid)
        n = int(first[0]) if len(first) else len(distance)
        return float(distance[max(0, n - 1)]), points[min(n, len(points) - 1)]

    ahead, boundary = extent([0, 0], [1, 0], 5.0)
    junctions = []
    for x in np.arange(0.25, max(0.26, ahead), 0.25):
        left, _ = extent([x, 0], [0, 1], 3.0)
        right, _ = extent([x, 0], [0, -1], 3.0)
        # Require a visible side branch and forward/backward continuity.
        if max(left, right) > 2.0 and ahead - x > 0.4:
            junctions.append(dict(position=[float(x), 0.0], left=left, right=right))
    left, _ = extent([0, 0], [0, 1], 3.0)
    right, _ = extent([0, 0], [0, -1], 3.0)
    rc = xy_to_grid(np.asarray([boundary]), len(grid), resolution)[0]
    # Check occupancy behind footprint boundary; unknown is never a corridor end.
    i, j = rc
    r = max(1, int(math.ceil(footprint / resolution)) + 1)
    occupied_boundary = bool(
        np.any(
            grid[
                max(0, i - r) : min(len(grid), i + r + 1),
                max(0, j - r) : min(len(grid), j + r + 1),
            ]
            == 1
        )
    )
    return dict(
        forward_free_m=ahead,
        junctions=junctions,
        inside_intersection=max(left, right) > 2.0 and ahead > 0.4,
        corridor_end_observed=occupied_boundary and ahead < 1.5,
        source="observed occupancy ray connectivity; conservative local candidates",
    )


def instruction_path(goal, facts, grid, pose, resolution, footprint):
    action = goal.get("action")
    if action == "relative_motion":
        if goal["direction"] == "backward":
            return None, "execution_not_evaluated", None
        remaining = max(
            0.0,
            facts["subgoal"]["distance_m"] - facts.get("subgoal_displacement", [0])[0],
        )
        if remaining <= 0.05:
            return None, "relative_distance_reached", (0.0, 0.0)
        target = np.asarray([[facts["subgoal"]["distance_m"], 0.0, 0.0]])
        tf = np.linalg.inv(np.asarray(pose)) @ np.asarray(facts["subgoal_start_pose"])
        target = transform_points(target, tf)[0, :2]
        endpoint = target * min(1.0, 1.0 / max(np.linalg.norm(target), 1e-9))
    elif action == "turn":
        sign = 1 if goal["direction"] == "left" else -1
        remaining = math.pi / 2 - sign * facts["subgoal_turn_rad"]
        if remaining <= 0.1:
            return None, "turn_observed_complete", (0.0, 0.0)
        return None, "turning_from_locked_pose", (0.0, sign * min(0.3, remaining))
    elif action in ("advance_to_intersection", "advance_to_corridor_end"):
        local = corridor_observation(grid, resolution, footprint)
        if (
            action == "advance_to_intersection"
            and facts.get("subgoal_intersections", 0) >= facts["subgoal"]["ordinal"]
        ):
            return None, "intersection_count_reached", (0.0, 0.0)
        if action == "advance_to_corridor_end" and local["corridor_end_observed"]:
            return None, "corridor_end_reached", (0.0, 0.0)
        endpoint = [min(1.0, local["forward_free_m"]), 0.0]
        if endpoint[0] < 0.1:
            return None, "corridor_unobserved_or_blocked", None
    elif action in ("wait", "stop"):
        return None, "waiting" if action == "wait" else "stopping", (0.0, 0.0)
    else:
        return None, "awaiting_executable_subgoal", None
    path = astar_reference(grid, endpoint, resolution, footprint)
    return (
        path,
        "observed_subgoal_path" if len(path) > 1 else "subgoal_path_blocked",
        None,
    )
