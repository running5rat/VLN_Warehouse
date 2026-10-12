"""Identity evidence is independent of static-anchor admission."""

import itertools
import math
import numpy as np


def update_identity(record, previous, observation, texts, *, evidence_id=None):
    evidence = list((previous or {}).get("text_evidence", []))
    # Multiple text regions remain a literal spatially ordered signature, never
    # a regex-repaired serial number. Raw regions and readings remain available.
    bound = [t for t in texts if t.get("status") == "bound"]
    bound.sort(
        key=lambda t: (
            round(min(p[1] for p in t["polygon"]) / 12),
            min(p[0] for p in t["polygon"]),
        )
    )
    signature = " | ".join(dict.fromkeys(t["normalized"] for t in bound))
    if evidence_id is not None:
        # A late text result fills the empty geometry-stage placeholder. Distinct
        # conflicting reads are retained, but never count as distinct source views.
        evidence = [
            e
            for e in evidence
            if not (e["image_id"] == observation["image_id"] and not e.get("signature"))
        ]
    if (
        evidence_id is not None
        and not any(e.get("evidence_id") == evidence_id for e in evidence)
    ) or (
        evidence_id is None
        and not any(e["image_id"] == observation["image_id"] for e in evidence)
    ):
        evidence.append(
            dict(
                image_id=observation["image_id"],
                t_obs=observation["t_obs"],
                camera_pose=(
                    np.array(observation["T_odom_body"])
                    @ np.array(observation["T_body_camera"])
                ).tolist(),
                signature=signature or None,
                texts=bound,
                **({"evidence_id": evidence_id} if evidence_id is not None else {}),
            )
        )
    signatures = {e["signature"] for e in evidence if e["signature"]}
    supports = [e for e in evidence if e["signature"]]
    distinct_view = False
    for a, b in itertools.combinations(supports, 2):
        pa, pb = np.array(a["camera_pose"]), np.array(b["camera_pose"])
        angle = math.acos(np.clip((np.trace(pa[:3, :3].T @ pb[:3, :3]) - 1) / 2, -1, 1))
        if np.linalg.norm(pa[:3, 3] - pb[:3, 3]) >= 0.15 or angle >= math.radians(10):
            distinct_view = True
            break
    conflict = len(signatures) > 1
    confirmed = (
        len(signatures) == 1
        and len({e["image_id"] for e in supports}) >= 3
        and distinct_view
    )
    state = (
        "conflict"
        if conflict
        else (
            "identity_confirmed"
            if confirmed
            else ("candidate" if signatures else "unknown")
        )
    )
    reasons = []
    if conflict:
        reasons.append("conflicting_text_readings")
    elif not signatures:
        reasons.append("no_readable_unambiguous_text")
    elif not confirmed:
        reasons.append("insufficient_independent_views")
    record.update(
        text_evidence=evidence,
        identity_status=state,
        identity_text=next(iter(signatures)) if len(signatures) == 1 else None,
        identity_candidates=sorted(signatures),
        needs_reobservation=bool(reasons),
        reobservation_reasons=reasons,
        duplicate_text=False,
    )
    # name is legacy display metadata, not geometry matching input for new events.
    record["name"] = record["identity_text"] if confirmed else None
    return record


def mark_duplicate_texts(records):
    groups = {}
    for record in records:
        for text in record.get("identity_candidates", []):
            groups.setdefault(text, []).append(record)
    changed = {}
    for group in groups.values():
        if len(group) < 2:
            continue
        for record in group:
            record.update(duplicate_text=True, needs_reobservation=True)
            if record["identity_status"] == "identity_confirmed":
                record["identity_status"] = "duplicate_text"
            if "text_seen_on_other_instance" not in record["reobservation_reasons"]:
                record["reobservation_reasons"].append("text_seen_on_other_instance")
            changed[record["landmark_uid"]] = record
    return list(changed.values())
