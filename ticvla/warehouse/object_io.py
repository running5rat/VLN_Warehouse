"""One visual QA call -> source-bound category events; no navigation inference."""

import re
import time
from pathlib import Path

from .mapping import SemanticEvent
from .mapping.semantic_frame import write_json

OBJECT_QUESTION = (
    "<image>\nList the names of individual physical objects visible in this image. "
    "Do not describe the overall scene or suggest movements. Give a short list of object names."
)
OBJECT_SYSTEM = (
    "You are a helpful visual assistant. Answer the question about the image."
)
NAVIGATION_QUESTION = (
    OBJECT_QUESTION
    + " Do not include floors, walls, ceilings, or text read from signs."
)


def object_question(observation):
    if observation.get("core_vocabulary"):
        from .perception.realtime import selection_question
        return selection_question(observation["navigation_task"])
    if observation.get("navigation_task"):
        return (
            "<image>\nTask: "
            + observation["navigation_task"]
            + "\nList visible physical object categories useful as location references, "
            "task targets, or obstacles. Include visible signs and distinctive facilities. "
            "Do not read text or numbers. Exclude plain floors, walls and ceilings. "
            "Short English names only, one per line, each category once. No explanation."
        )
    return (
        NAVIGATION_QUESTION
        if observation.get("object_policy", "all") == "navigation"
        else OBJECT_QUESTION
    )


def category_policy(categories, policy):
    """Exclude background classes, not their depth geometry; retain raw answers."""
    kept, filtered = [], []
    for category in categories:
        words = set(category.replace("-", " ").split())
        background = bool(
            words
            & {
                "floor",
                "floors",
                "flooring",
                "ground",
                "wall",
                "walls",
                "ceiling",
                "ceilings",
            }
        ) or category in {
            "warehouse",
            "warehouse interior",
            "aisle",
            "aisles",
            "warehouse aisle",
        }
        # Markings, doors, signs and lights are discrete cues, not the substrate.
        cue = bool(
            words
            & {
                "marking",
                "markings",
                "line",
                "lines",
                "sign",
                "signs",
                "light",
                "lights",
                "lamp",
                "lamps",
                "door",
                "doors",
                "panel",
                "panels",
            }
        )
        # "floor with yellow lines" still names the floor, unlike "floor markings".
        # Do not let a trailing useful cue exempt a generic background surface.
        if "with" in words:
            cue = False
        if policy == "navigation" and background and not cue:
            filtered.append(
                dict(
                    category=category,
                    reason="background surface/scene; retained in depth geometry only",
                )
            )
        else:
            kept.append(category)
    return kept, filtered


def add_mode_argument(parser):
    parser.add_argument(
        "--llm-mode", choices=("objects-only", "structured"), default="objects-only"
    )
    parser.add_argument(
        "--object-policy", choices=("navigation", "all"), default="navigation"
    )


def validate_mode_args(parser, args):
    if args.llm_mode == "objects-only" and args.object_evidence:
        parser.error("--object-evidence requires --llm-mode structured")
    if getattr(args, "perception_backend", None):
        if getattr(args, "shelf_signs", False):
            parser.error("--perception-backend conflicts with --shelf-signs")
        if args.llm_mode != "objects-only" or not args.base_model_only:
            parser.error(
                "--perception-backend requires --llm-mode objects-only --base-model-only"
            )
        if getattr(args, "semantic_backend", "llm") != "llm":
            parser.error("--perception-backend requires --semantic-backend llm")


def object_observation(source, policy="navigation", task=None):
    """Only this RGB is consumed by describe_objects; pose/time are provenance."""
    observation = dict(
        object_policy=policy,
        rgb_history=[
            dict(
                image_id=source["image_id"],
                path=source["rgb"],
                t_obs=source["t_obs"],
                T_odom_body=source["T_odom_body"],
            )
        ],
        t_obs=source["t_obs"],
        odom_epoch=source["odom_epoch"],
        T_odom_body=source["T_odom_body"],
    )
    if task is not None:
        observation["navigation_task"] = task
    return observation


def parse_object_list(text):
    """Accept line lists, not arbitrary prose or navigation/XML responses.

    This is syntax validation, not proof that a noun names a real visible object.
    Any malformed line rejects the answer rather than silently dropping content.
    """
    text = text.strip()
    if text == "[]":
        return []
    if not text:
        raise ValueError("Empty object answer")
    categories = []
    # A comma-separated noun list is an equivalent syntax, not a second model
    # pass or a category repair. A final full stop is punctuation only.
    if "," in text and not any(c in text for c in "<>[]{}:"):
        text = text.rstrip(".").replace(",", "\n")
    for line in text.splitlines():
        if not line.strip():
            continue
        value = re.sub(r"^\s*(?:[-*•]\s+|\d+[.)]\s+)", "", line).strip()
        value = " ".join(value.split()).casefold()
        if len(value) > 80 or not re.fullmatch(
            r"[a-z][a-z0-9'-]*(?: [a-z0-9][a-z0-9'-]*){0,7}", value
        ):
            raise ValueError("Invalid object-list line: " + line)
        if value not in categories:
            categories.append(value)
    return categories


def run_object_qa(model, observation, request_id, output):
    """Exactly one model call; return no route, snapshot, refline or z_sem."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frames = observation["rgb_history"]
    if len(frames) != 1:
        raise ValueError("objects-only requires exactly one source RGB")
    frame = frames[0]
    question = object_question(observation)
    (output / "prompt.txt").write_text(OBJECT_SYSTEM + "\n" + question)
    write_json(output / "input-images.json", [frame["path"]])
    start = time.perf_counter()
    evidence, events, parsed, error = None, [], None, None
    try:
        evidence = model.describe_objects(observation)
        write_json(output / "object-answer.json", evidence)
        if evidence["image_id"] != frame["image_id"]:
            raise ValueError("Object answer source image mismatch")
        if evidence["token_count"] >= evidence["max_new_tokens"]:
            raise ValueError("Object answer may be truncated")
        raw_categories = parse_object_list(evidence["answer"])
        categories, filtered = category_policy(
            raw_categories, observation.get("object_policy", "all")
        )
        parsed = dict(
            categories=categories,
            raw_categories=raw_categories,
            filtered_categories=filtered,
        )
        events = [
            SemanticEvent(
                f"{request_id}:{index}",
                frame["image_id"],
                category,
                source="llm",
                request_id=request_id,
            )
            for index, category in enumerate(categories)
        ]
    except (ValueError, RuntimeError) as exc:
        error = str(exc)
    result = dict(
        valid=error is None,
        error=error,
        parsed=parsed,
        text=evidence["answer"] if evidence else "",
        inference_mode="OBJECTS_ONLY",
        llm_calls=1,
        structured_inference_executed=False,
        generated_tokens=evidence["token_count"] if evidence else 0,
        format_constraint="unconstrained visual QA + deterministic line-list parser",
        prompt=question,
        system=OBJECT_SYSTEM,
        answer_status="failed" if error else ("objects" if events else "empty"),
    )
    return result, events, dict(object_qa_s=time.perf_counter() - start)
