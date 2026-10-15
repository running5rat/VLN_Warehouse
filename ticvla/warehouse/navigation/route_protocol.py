"""Natural language Route protocol. Syntax checks do not establish task success."""

import json
import math
import numpy as np

ACTIONS = (
    "approach_landmark",
    "advance_to_intersection",
    "advance_to_corridor_end",
    "turn",
    "relative_motion",
    "explore",
    "follow_person",
    "wait",
    "stop",
)
STATUSES = ("active", "waiting", "needs_clarification", "completed")
ROUTE_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["remaining_instruction", "current_subgoal", "status"],
    "properties": {
        "remaining_instruction": {"type": "string", "minLength": 1, "maxLength": 600},
        "current_subgoal": {
            "type": "object",
            "additionalProperties": False,
            "required": ["action"],
            "properties": {
                "action": {"type": "string", "enum": list(ACTIONS)},
                "target_text": {"type": "string", "maxLength": 120},
                "category": {"type": "string", "maxLength": 80},
                "description": {"type": "string", "maxLength": 160},
                "ordinal": {"type": "integer", "minimum": 1, "maximum": 100},
                "direction": {
                    "type": "string",
                    "enum": ["left", "right", "forward", "backward"],
                },
                "distance_m": {"type": "number", "exclusiveMinimum": 0, "maximum": 100},
                "origin": {"type": "string", "enum": ["subgoal_start"]},
            },
        },
        "status": {"type": "string", "enum": list(STATUSES)},
    },
}

# Discriminated union: constrain syntax for each action, never choose an action for the model.
_goal_properties = ROUTE_SCHEMA["properties"]["current_subgoal"]["properties"]
_fields = {
    "approach_landmark": ["target_text", "category", "description"],
    "advance_to_intersection": ["ordinal"],
    "turn": ["direction"],
    "relative_motion": ["direction", "distance_m", "origin"],
    "follow_person": ["description"],
}
_variants = []
for _action in ACTIONS:
    _props = {"action": {"type": "string", "enum": [_action]}}
    _props.update({k: _goal_properties[k] for k in _fields.get(_action, [])})
    if _action in ("turn", "relative_motion"):
        _props["direction"] = {
            "type": "string",
            "enum": ["left", "right"] if _action == "turn" else ["forward", "backward"],
        }
    _required = ["action"] + (
        [] if _action == "approach_landmark" else _fields.get(_action, [])
    )
    _variants.append(
        dict(
            type="object",
            properties=_props,
            required=_required,
            additionalProperties=False,
        )
    )
ROUTE_SCHEMA["properties"]["current_subgoal"] = {"anyOf": _variants}

NAVIGATION_SYSTEM = """You are the robot's navigation VLM. Understand the original natural-language instruction yourself.
Use the previous remaining_route and measured execution_state to update the remaining instruction and current subgoal.
Never overwrite the original goal, reset a distance on every image, or claim success from an issued command.
There is no human-provided task decomposition. If ambiguous or unsupported, return needs_clarification with wait.
Reply with exactly these tags in this order, no commentary:
<landmark>JSON array</landmark><local>short relation or unknown</local><Route>JSON object</Route><waypoint>five coordinate pairs or []</waypoint><REFLINE>
landmark: at most three visible physical object classes, each with image_id from supplied RGB and category (short English class).
A sign is an object; an intersection/corridor end is a relation, not a segmentation object. Empty array is allowed.
local: short observed or cautiously inferred spatial relation (English, at most 80 characters), never a localization transform.
Route: remaining_instruction (remaining task in concise English), current_subgoal (object), status.
status: active, waiting, needs_clarification, completed. Completion needs measured evidence for the original task.
current_subgoal.action and parameters:
approach_landmark: target_text and/or category/description;
advance_to_intersection: ordinal (positive count of intersections still to pass);
advance_to_corridor_end: no parameters;
turn: direction left or right;
relative_motion: direction forward or backward, distance_m (total distance from locked subgoal_start), origin subgoal_start;
explore: budgets come from execution configuration;
follow_person: description (e.g. the only person ahead);
wait or stop: no parameters.
Do not output coordinates or task types instead of an action. Do not add fields outside this protocol.
Only reduce ordinal or remaining text when the supplied execution facts support progress.
Keep the same subgoal's origin and total distance across updates; use measured displacement to describe the remainder.
waypoint: exactly FIVE ordered spatial anchors on the current local route, in the REQUEST SOURCE robot frame.
Units metres, x forward and y left. They describe the SAME route as current_subgoal and the continuous REFLINE.
They are not timed actions or wheel speeds. Syntax (x,y);(x,y);(x,y);(x,y);(x,y). Use [] if you cannot give a route.
For backward relative_motion diagnostics, express a route extending behind the source robot (negative x),
with the requested distance minus measured progress from the locked origin. These diagnostics are never executed;
unobserved rear space is not a reason to omit this diagnostic. Do not invent a successful execution.
End immediately after <REFLINE>. No second question will repair your answer.
"""


# Offline ablation only. The production default remains NAVIGATION_SYSTEM.
COMPACT_NAVIGATION_SYSTEM = """Update a robot's remaining task using instruction, remaining_route and execution_state.
Keep the original final goal. Only measured facts establish progress or completion; issued commands do not.
Return exactly <landmark>...</landmark><local>...</local><Route>...</Route><waypoint>...</waypoint><REFLINE>.
landmark: [] or up to three visible objects with supplied image_id and short English category. Intersections are not objects.
local: observed relation in English, at most 80 characters, or unknown.
Route: JSON with remaining_instruction (remaining task, not protocol text), current_subgoal, status.
current_subgoal is an object with action and its parameters:
approach_landmark: target_text and/or category/description; advance_to_intersection: ordinal;
advance_to_corridor_end: no parameters; turn: direction left/right;
relative_motion: direction forward/backward, distance_m, origin="subgoal_start";
explore: no parameters; follow_person: description; wait/stop: no parameters.
status: active/waiting/needs_clarification/completed. For unclear requests use wait and needs_clarification.
Keep a motion subgoal's locked origin and total distance. Describe measured progress in remaining_instruction.
waypoint: five ordered (x,y) anchors separated by semicolons, or [] if unavailable. Metres, source robot frame,
x forward, y left; same route as current_subgoal and REFLINE, not timed actions. Backward diagnostics extend
in negative x by the remaining distance; rear visibility does not prohibit diagnostics. Never execute reverse here.
Stop after <REFLINE>.
"""


def parse_route(value, require_structured=False):
    if isinstance(value, str):
        if not value.strip():
            raise ValueError("Empty Route")
        if value.lstrip().startswith("{"):
            try:
                value = json.loads(value)
            except json.JSONDecodeError as exc:
                raise ValueError("Invalid Route JSON") from exc
        elif not require_structured:
            return value
        else:
            raise ValueError("Natural instruction requires structured Route")
    if not isinstance(value, dict) or set(value) != set(ROUTE_SCHEMA["required"]):
        raise ValueError("Invalid Route fields")
    text, goal, status = (
        value["remaining_instruction"],
        value["current_subgoal"],
        value["status"],
    )
    if (
        not isinstance(text, str)
        or not text.strip()
        or len(text) > 600
        or status not in STATUSES
    ):
        raise ValueError("Invalid remaining instruction or Route status")
    if not isinstance(goal, dict) or goal.get("action") not in ACTIONS:
        raise ValueError("Invalid subgoal action")
    a = goal["action"]
    allowed = {
        "approach_landmark": {"target_text", "category", "description"},
        "advance_to_intersection": {"ordinal"},
        "turn": {"direction"},
        "relative_motion": {"direction", "distance_m", "origin"},
        "follow_person": {"description"},
    }.get(a, set())
    if set(goal) - allowed - {"action"}:
        raise ValueError("Unexpected subgoal parameters")
    for field in ("target_text", "category", "description"):
        if field in goal and (
            not isinstance(goal[field], str)
            or not goal[field].strip()
            or len(goal[field]) > 160
        ):
            raise ValueError("Invalid target description")
    if a == "approach_landmark" and not any(goal.get(k) for k in allowed):
        raise ValueError("Landmark requires a target reference")
    if a == "advance_to_intersection" and (
        type(goal.get("ordinal")) is not int or not 1 <= goal["ordinal"] <= 100
    ):
        raise ValueError("Invalid intersection ordinal")
    if a == "turn" and goal.get("direction") not in ("left", "right"):
        raise ValueError("Invalid turn direction")
    if a == "follow_person" and not goal.get("description"):
        raise ValueError("Follow requires target description")
    if a == "relative_motion":
        d = goal.get("distance_m")
        if (
            goal.get("direction") not in ("forward", "backward")
            or goal.get("origin") != "subgoal_start"
            or type(d) not in (float, int)
            or not math.isfinite(d)
            or not 0 < d <= 100
        ):
            raise ValueError("Invalid relative motion parameters")
    return value


def reference_diagnostics(
    parsed, xy, mask, mode="TEXT", anchor_limit=0.30, endpoint_limit=0.50
):
    line = np.asarray(xy)[np.asarray(mask, bool)]
    result = dict(
        anchor_threshold_m=anchor_limit,
        endpoint_threshold_m=endpoint_limit,
        waypoint_applicable=mode == "TEXT",
        reference_available=len(line) >= 2,
        consistency_valid=True,
        reasons=[],
    )
    if mode != "TEXT":
        return result
    anchors = np.asarray(
        parsed.get("anchors", parsed.get("waypoint", [])), dtype=float
    ).reshape(-1, 2)
    result.update(waypoint_count=len(anchors), decoder_valid_points=len(line))
    if len(anchors) == 0:
        result["reasons"] = ["empty_waypoint_nonempty_decoder"] if len(line) else []
    elif len(line) < 2:
        result["reasons"] = ["decoder_route_unavailable"]
    else:
        start, delta = line[:-1], np.diff(line, axis=0)
        length = np.linalg.norm(delta, axis=1)
        t = np.clip(
            ((anchors[:, None] - start) * delta).sum(-1) / np.maximum(length**2, 1e-12),
            0,
            1,
        )
        distances = np.linalg.norm(
            anchors[:, None] - (start + t[:, :, None] * delta), axis=-1
        )
        idx = distances.argmin(axis=1)
        nearest = distances[np.arange(len(anchors)), idx]
        order = (
            np.r_[0, np.cumsum(length)][idx]
            + t[np.arange(len(anchors)), idx] * length[idx]
        )
        endpoint = float(np.linalg.norm(anchors[-1] - line[-1]))
        result.update(
            anchor_distances_m=nearest.tolist(),
            arclength_m=order.tolist(),
            endpoint_difference_m=endpoint,
        )
        if np.any(nearest > anchor_limit):
            result["reasons"].append("anchors_off_decoder")
        if np.any(np.diff(order) < -1e-5):
            result["reasons"].append("anchor_order")
        if endpoint > endpoint_limit:
            result["reasons"].append("endpoint_difference")
    route = parsed.get("route")
    if isinstance(route, dict):
        g = route["current_subgoal"]
        result["original_instruction_semantics"] = "not_automatically_verified"
        if (
            g["action"] in ("wait", "stop")
            and len(anchors)
            and np.any(np.linalg.norm(anchors, axis=1) > 0.05)
        ):
            result["reasons"].append("motion_waypoint_for_wait_or_stop")
        if g["action"] == "turn" and len(anchors):
            sign = 1 if g["direction"] == "left" else -1
            if anchors[-1, 1] * sign <= 0:
                result["reasons"].append("waypoint_turn_direction_mismatch")
        if g["action"] == "relative_motion":
            sign = -1 if g["direction"] == "backward" else 1
            result.update(
                requested_distance_m=g["distance_m"],
                diagnostic_direction=g["direction"],
                decoder_endpoint=line[-1].tolist() if len(line) else None,
                waypoint_endpoint=anchors[-1].tolist() if len(anchors) else None,
            )
            if len(anchors) and (
                anchors[-1, 0] * sign <= 0 or np.any(anchors[:, 0] * sign < -0.05)
            ):
                result["reasons"].append("waypoint_direction_mismatch")
            if len(line) and line[-1, 0] * sign <= 0:
                result["reasons"].append("decoder_direction_mismatch")
    result["consistency_valid"] = not result["reasons"]
    return result
