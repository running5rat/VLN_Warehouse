"""Small, explicitly gated Responses API annotation runs; never auto-approve."""

import base64
import json
import os
import mimetypes
from pathlib import Path
import time
import urllib.request
import urllib.error
from urllib.parse import urlparse

from ..schema import file_sha256
from .core import read_json, write_json, sample_context, tagged_text, verify_observation_files


from ..compact_input import RULES, VERSION, model_context, image_label, render_context

TEACHER_INSTRUCTION = (
    "Annotate the current navigation observation. Return English JSON with exactly "
    "landmark (category array), local (string, <=90 characters), route (string, <=600 characters or DONE). "
    "No trajectories, coordinates, explanations or Markdown. "
) + RULES


def teacher_input(sample):
    """Full causal audit context; model_context() is the only model-facing projection."""
    from ..navigation.route_protocol import COMPACT_ROUTE_SCHEMA, remaining_instruction, compact_route

    obs = sample["observation"]
    # Construct an allowlisted input, not a serialization of the annotation record.
    context = {k:obs[k] for k in ("task", "route", "route_revision", "robot_state", "execution")}
    try:
        context["route"] = compact_route(obs["route"])
    except ValueError:
        context["route"] = remaining_instruction(obs["route"])
    context["robot_state_valid"] = obs.get("robot_state_mask", [False, False])
    context["source_time"] = obs["t_obs"]
    context["current_image_id"] = obs["image_id"]
    context["landmark_protocol"] = "current-categories-v1"
    context["source_rgb"] = [{"image_id": h["image_id"], "t_obs": h["t_obs"],
                               **{k: h[k] for k in ("T_odom_body", "T_odom_camera_flu") if k in h}}
                              for h in obs["rgb_history"]]
    context["previous_local"] = obs.get("previous_local", {"valid": False, "reason": "not_supplied"})
    context["source_pose"] = obs.get("T_odom_body", obs.get("T_odom_camera_flu"))
    context["pose_reference"] = obs.get("pose_reference", "robot_body_flu")
    view = obs.get("mapview", {})
    context["map_interface"] = ("MapView-rendered; prefix observations only; metres"
        if view.get("image") else "unavailable; no depth/map supplied; do not invent map evidence")
    context["route_schema"] = COMPACT_ROUTE_SCHEMA
    context["route_protocol"] = "compact-v2"
    images = [h["path"] for h in obs["rgb_history"]]
    if view.get("image"):
        images.append(view["image"])
    from ..semantic_input import visual_context, candidate_image
    context.update(visual_context(obs))
    overlay = candidate_image(obs)
    if overlay:
        images.append(overlay)
    return context, images


def run_teacher(project, config, output):
    from ..structured_io import parse_structured

    project, output = Path(project), Path(output)
    cfg = read_json(config)
    output.mkdir(parents=True, exist_ok=False)
    report = dict(status="not_started", actual_api_calls=0, samples=[], human_approved=False)
    write_json(output / "report.json", report)
    info = read_json(project / "project.json")
    if not info.get("human_episode_approved"):
        report.update(status="blocked", reason="First episode requires explicit human finalization before teacher calls")
        write_json(output / "report.json", report)
        return report
    model = cfg.get("model")
    token = os.environ.get(cfg.get("api_key_env", "OPENAI_API_KEY"))
    base = cfg.get("base_url", "https://api.openai.com/v1").rstrip("/")
    if not model or not token:
        report.update(status="blocked", reason="Explicit model ID and API key environment variable required")
        write_json(output / "report.json", report)
        return report
    if urlparse(base).scheme != "https" or urlparse(base).username or urlparse(base).password:
        raise ValueError("Teacher endpoint must be HTTPS without credentials in its URL")
    indices = cfg.get("indices", list(range(min(6, len(info["samples"])))))
    if len(indices) > 6 or len(indices) != len(set(indices)):
        raise ValueError("At most six distinct source frames per project; no automatic batch expansion")
    report.update(status="running", model=model, base_url=base, config_sha256=file_sha256(config))
    for i in indices:
        s = sample_context(project, int(i))
        verify_observation_files(s)
        if not s["annotation"] or s["annotation"]["parent_revision"] != s["parent_revision"]:
            raise ValueError("Missing or stale approved reference")
        audit, images = teacher_input(s)
        context = model_context(s["observation"])
        request = dict(model=model, instructions=TEACHER_INSTRUCTION, store=False,
                       max_output_tokens=int(cfg.get("max_output_tokens", 2048)),
                       input=[dict(role="user", content=[dict(type="input_text", text=render_context(context))])])
        payload = request["input"][0]["content"]
        for i, path in enumerate(images):
            label = image_label(s["observation"], i) if i < len(s["observation"]["rgb_history"]) else "Observed map / current detector overlay"
            payload.append(dict(type="input_text", text=label))
            mime = mimetypes.guess_type(path)[0] or "image/png"
            payload.append(dict(type="input_image", image_url=f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()))
        run = output / f"{i:08d}"
        # Full causal input + image references/hashes reproduce the byte payload without duplicating base64.
        write_json(run / "input.json", dict(model=model, instructions=TEACHER_INSTRUCTION, context=context,
                                            audit_context=audit, input_format=VERSION, user_text=render_context(context),
                                            images=[dict(path=p, sha256=file_sha256(p)) for p in images],
                                            source_binding=s["provenance"]["source_binding"],
                                            max_output_tokens=request["max_output_tokens"], store=False))
        started = time.monotonic()
        row = dict(index=i, source_image_id=s["observation"]["image_id"], status="request_started")
        report["actual_api_calls"] += 1
        write_json(output / "report.json", report)
        try:
            req = urllib.request.Request(base + "/responses", data=json.dumps(request).encode(),
                                         headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=float(cfg.get("timeout_seconds", 180))) as response:
                raw = json.load(response)
            write_json(run / "response.json", raw)
            row["usage"] = raw.get("usage")
            row["cost"] = None  # No guessed price or unverified model rate.
            if raw.get("status") != "completed":
                raise ValueError("Incomplete/refused/failed model response")
            text = "".join(c["text"] for item in raw.get("output", []) if item.get("type") == "message"
                           for c in item.get("content", []) if c.get("type") == "output_text")
            (run / "raw.txt").write_text(text)
            answer = json.loads(text)
            if not isinstance(answer, dict) or set(answer) != {"landmark", "local", "route"}:
                raise ValueError("Teacher field mismatch")
            full = tagged_text(answer["landmark"], answer["local"], answer["route"], s["annotation"]["waypoint"])
            parsed = parse_structured(full, [h["image_id"] for h in s["observation"]["rgb_history"]],
                structured_route=isinstance(answer["route"], dict),
                compact_route=isinstance(answer["route"], str), current_image_id=s["observation"]["image_id"])
            write_json(run / "proposal.json", dict(text=full, parsed=parsed, approved=False,
                        field_sources=dict(landmark="teacher", local="teacher", route="teacher", waypoint="approved_reference"),
                        semantic_review="required", reference_revision=s["annotation"]["revision_id"]))
            row.update(status="format_valid_awaiting_human_review",
                       exact_route_match=parsed["route"] == s["annotation"]["parsed"]["route"])
        except urllib.error.HTTPError as exc:
            # Response body is useful for diagnostics; request Authorization is never saved.
            (run / "http-error.txt").write_bytes(exc.read())
            row.update(status="failed", reason=f"HTTP {exc.code}")
        except (ValueError, KeyError, OSError) as exc:
            row.update(status="failed", reason=str(exc))
        row["elapsed_seconds"] = time.monotonic() - started
        report["samples"].append(row)
        write_json(output / "report.json", report)
    report["status"] = "completed_awaiting_human_review"
    report["semantic_acceptance_passed"] = False
    write_json(output / "report.json", report)
    return report
