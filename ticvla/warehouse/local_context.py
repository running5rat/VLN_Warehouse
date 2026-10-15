"""Source-bound, fallible Local memory; relative transforms never update localization."""
import math
import numpy as np

LOCAL_CONTEXT_INSTRUCTION = (
    "previous_local is a fallible previous semantic estimate, not a verified localization. "
    "Use it only when valid, together with current/history RGB and observed map. "
    "current_in_previous_body is the CURRENT body pose in the PREVIOUS Local source body frame: "
    "x forward, y left in metres, yaw counterclockwise in radians. It is net motion, not travelled path. "
    "Infer the current local relation cautiously; prefix historical-only inference with 'Inferred:'. "
    "New contradictory observations take precedence; insufficient evidence means unknown. "
    "Repetition does not constitute a fresh visual confirmation; never turn Local into a localization TF."
)


def local_memory(text, obs, *, origin, revision):
    """Bind a response to its request source, never the pose at response arrival."""
    if not text or text.strip().casefold() == "unknown":
        return None
    image_id = obs.get("image_id")
    if not image_id:
        image_id = next((h["image_id"] for h in obs.get("rgb_history", [])
                         if h.get("t_obs") == obs.get("t_obs")), None)
    if not image_id or obs.get("t_obs") is None:
        return None
    return dict(text=text, image_id=image_id, t_obs=obs["t_obs"],
                T_odom_body=obs["T_odom_body"], odom_epoch=obs["odom_epoch"],
                origin=origin, revision=revision, evidence_status="unverified_semantic_estimate")


def local_context(memory, obs, max_age_s=30.0):
    """Compact planar projection of inv(T_odom_previous) @ T_odom_current."""
    empty = dict(valid=False, reason="no_previous_local", max_age_s=max_age_s)
    if memory is None:
        return empty
    out = dict(empty, text=memory["text"], image_id=memory["image_id"],
               t_obs=memory["t_obs"], origin=memory["origin"], revision=memory["revision"],
               evidence_status="unverified_semantic_estimate")
    age = float(obs["t_obs"]) - float(memory["t_obs"])
    out["age_s"] = age if math.isfinite(age) else None
    if memory["odom_epoch"] != obs["odom_epoch"]:
        return dict(out, reason="odom_epoch_changed")
    if not math.isfinite(age) or age <= 0:
        return dict(out, reason="non_past_source")
    if age > max_age_s:
        return dict(out, reason="expired")
    transforms = [np.asarray(x, dtype=float) for x in (memory["T_odom_body"], obs["T_odom_body"])]
    for T in transforms:
        if (T.shape != (4, 4) or not np.isfinite(T).all()
            or not np.allclose(T[3], [0, 0, 0, 1])
            or not np.allclose(T[:3, :3].T @ T[:3, :3], np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(T[:3, :3]), 1, atol=1e-4)):
            return dict(out, reason="invalid_pose")
    relative = np.linalg.inv(transforms[0]) @ transforms[1]
    return dict(out, valid=True, reason=None, current_in_previous_body=dict(
        x_m=float(relative[0, 3]), y_m=float(relative[1, 3]),
        yaw_rad=math.atan2(relative[1, 0], relative[0, 0])))
