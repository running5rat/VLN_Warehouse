"""Explicit overlapping observation anchor + geometric verification; never identity TF by default."""

import numpy as np
from scipy.spatial import cKDTree
from ..geometry import planar_pose, transform_points


def verify_anchor(old_points, new_points):
    old = np.asarray(old_points)
    new = np.asarray(new_points)
    old = old[(old[:, 2] > 0.15) & (old[:, 2] < 3)]
    new = new[(new[:, 2] > 0.15) & (new[:, 2] < 3)]
    if min(len(old), len(new)) < 200:
        raise ValueError("Insufficient anchor geometry")
    old = old[:: max(1, len(old) // 3000)]
    new = new[:: max(1, len(new) // 3000)]
    tree = cKDTree(new)
    solutions = []
    for yaw in (0.0, -0.3, 0.3):
        for x, y in ((0, 0), (-0.5, 0), (0.5, 0), (0, -0.5), (0, 0.5)):
            tf = planar_pose(x, y, yaw)
            for _ in range(20):
                transformed = transform_points(old, tf)
                distance, ids = tree.query(transformed)
                valid = distance < 0.5
                if valid.sum() < 100:
                    break
                a, b = transformed[valid, :2], new[ids[valid], :2]
                ca, cb = a.mean(0), b.mean(0)
                u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
                r = vt.T @ u.T
                if np.linalg.det(r) < 0:
                    vt[-1] *= -1
                    r = vt.T @ u.T
                update = np.eye(4)
                update[:2, :2] = r
                update[:2, 3] = cb - r @ ca
                tf = update @ tf
                if np.linalg.norm(update - np.eye(4)) < 1e-5:
                    break
            distance, _ = tree.query(transform_points(old, tf))
            ratio = float((distance < 0.12).mean())
            residual = float(np.median(distance))
            solutions.append((ratio, residual, tf))
    solutions.sort(key=lambda s: (-s[0], s[1]))
    ratio, error, tf = solutions[0]
    if ratio < 0.8 or error > 0.06 or np.linalg.norm(np.ptp(old[:, :2], axis=0)) < 3:
        raise ValueError(
            f"Anchor registration rejected: ratio={ratio:.3f}, residual={error:.3f}m"
        )
    for r, e, t in solutions[1:]:
        separated = (
            np.linalg.norm(t[:2, 3] - tf[:2, 3]) > 0.3
            or abs(np.arctan2(t[1, 0], t[0, 0]) - np.arctan2(tf[1, 0], tf[0, 0])) > 0.15
        )
        if separated and r >= ratio - 0.05 and e < error + 0.03:
            raise ValueError("Ambiguous anchor registration")
    return dict(
        T_current_old=tf.tolist(),
        inlier_ratio=ratio,
        median_residual_m=error,
        source="explicit_overlap_anchor_plus_geometry",
        global_relocalization_validated=False,
    )
