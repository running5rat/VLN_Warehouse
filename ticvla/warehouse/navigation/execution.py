"""Source-bound execution facts, independent of commanded motion and model claims."""

import copy
import json
import math
import numpy as np


def subgoal_key(goal):
    # Wording and decreasing remaining intersection count cannot reset the origin.
    keys = ("action", "direction", "target_text", "category")
    return json.dumps({k: goal[k] for k in keys if k in goal}, sort_keys=True)


class ExecutionFacts:
    def __init__(self, record=None):
        self.data = (
            copy.deepcopy(record)
            if record
            else dict(
                events=[],
                path_m=0.0,
                turn_rad=0.0,
                subgoal_path_m=0.0,
                subgoal_turn_rad=0.0,
                intersections=[],
                subgoal_key=None,
                subgoal=None,
                subgoal_start_pose=None,
                last_pose=None,
                last_time=None,
                odom_epoch=None,
                terminal=None,
            )
        )

    def event(self, kind, **data):
        row = dict(
            kind=kind,
            source_time=self.data.get("last_time"),
            odom_epoch=self.data.get("odom_epoch"),
            **data,
        )
        if row not in self.data["events"]:
            self.data["events"].append(row)
        return row

    def observe(self, pose, stamp, epoch, velocity=(0, 0)):
        d = self.data
        pose = np.asarray(pose, float)
        if d["odom_epoch"] is not None and d["odom_epoch"] != epoch:
            self.event("odom_epoch_changed", previous=d["odom_epoch"], current=epoch)
            d["terminal"] = "odom_epoch_changed"
        if (
            d["last_time"] is not None
            and epoch == d["odom_epoch"]
            and stamp <= d["last_time"]
        ):
            return
        if d["last_pose"] is not None and epoch == d["odom_epoch"]:
            tf = np.linalg.inv(d["last_pose"]) @ pose
            distance = float(np.linalg.norm(tf[:2, 3]))
            turn = float(np.arctan2(tf[1, 0], tf[0, 0]))
            d["path_m"] += distance
            d["turn_rad"] += turn
            d["subgoal_path_m"] += distance
            d["subgoal_turn_rad"] += turn
        if "task_start_pose" not in d:
            d.update(task_start_pose=pose.tolist(), task_start_time=stamp)
        d.update(
            last_pose=pose.tolist(),
            last_time=stamp,
            odom_epoch=epoch,
            velocity=list(velocity),
        )
        if d["subgoal_start_pose"] is not None:
            tf = np.linalg.inv(d["subgoal_start_pose"]) @ pose
            d["subgoal_displacement"] = tf[:3, 3].tolist()

    def switch(self, goal, pose, revision):
        d = self.data
        key = subgoal_key(goal)
        if key != d["subgoal_key"]:
            self.event(
                "subgoal_switch",
                previous=d["subgoal"],
                current=goal,
                basis="validated VLM Route",
                route_revision=revision,
            )
            d.update(
                subgoal_key=key,
                subgoal=copy.deepcopy(goal),
                subgoal_start_pose=copy.deepcopy(pose),
                subgoal_path_m=0.0,
                subgoal_turn_rad=0.0,
                subgoal_displacement=[0.0, 0.0, 0.0],
                subgoal_intersection_start=len(d["intersections"]),
                subgoal_revision=revision,
                subgoal_started_at=d.get("last_time", 0),
            )
        # Same motion subgoal keeps its original requested total, even if a model changes it.

    def intersection(self, position, inside, image_id):
        """Confirm only on exit after entering, spatially deduplicated across frames."""
        d = self.data
        if inside and not d.get("intersection_pending"):
            d["intersection_pending"] = dict(position=list(position), image_id=image_id)
        if not inside and d.get("intersection_pending"):
            pending = d.pop("intersection_pending")
            if (
                np.linalg.norm(
                    np.asarray(position)[:2] - np.asarray(pending["position"])[:2]
                )
                < 0.4
            ):
                d["intersection_pending"] = pending
                return
            if not any(
                np.linalg.norm(
                    np.asarray(pending["position"])[:2] - np.asarray(p["position"])[:2]
                )
                < 1.0
                for p in d["intersections"]
            ):
                d["intersections"].append(pending)
                self.event("intersection_passed", **pending, exit_image_id=image_id)

    def completion_evidence(self):
        d = self.data
        g = d.get("subgoal") or {}
        action = g.get("action")
        if d.get("terminal"):
            return False
        if action == "relative_motion":
            if g["direction"] == "backward":
                return False  # diagnostic never claims execution
            return d.get("subgoal_displacement", [0])[0] >= g["distance_m"] - 0.05
        if action == "turn":
            sign = 1 if g["direction"] == "left" else -1
            return sign * d["subgoal_turn_rad"] >= math.pi / 2 - 0.1
        accepted = {
            "approach_landmark": "landmark_neighborhood_reached",
            "advance_to_corridor_end": "corridor_end_reached",
            "advance_to_intersection": "intersection_passed",
            "stop": "observed_stopped",
            "wait": "observed_stopped",
        }
        kind = accepted.get(action)
        if action == "advance_to_intersection":
            return (
                len(d["intersections"]) - d.get("subgoal_intersection_start", 0)
                >= g["ordinal"]
            )
        return bool(
            kind
            and any(
                e["kind"] == kind
                and e.get("source_time", -1)
                >= d.get("subgoal_started_at", float("inf"))
                for e in d["events"]
            )
        )

    def record(self):
        result = copy.deepcopy(self.data)
        result["confirmed_intersections"] = len(result["intersections"])
        result["subgoal_intersections"] = len(result["intersections"]) - result.get(
            "subgoal_intersection_start", 0
        )
        # Bound prompt history, retain full history in persisted state/event stream.
        return result
