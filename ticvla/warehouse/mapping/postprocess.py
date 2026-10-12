"""Reversible landmark views. Never modifies evidence, geometry or static status."""
from collections import Counter
from dataclasses import asdict, dataclass
import math
import unicodedata

import numpy as np


@dataclass(frozen=True)
class ViewPolicy:
    version: str = "landmark-view-v2-substring"
    min_text_confidence: float = 0.8
    repeated_categories: tuple = ("storage rack", "floor", "wall")
    # Deployment vocabulary, not an OCR template or a claim of global uniqueness.
    generic_texts: tuple = ("CAUTION", "FRAGILE")


def normalized(text):
    return " ".join(unicodedata.normalize("NFKC", text).split())


def reading(evidence, policy):
    """Join adjacent same-line OCR regions only; preserve punctuation and case."""
    regions = [t for t in evidence.get("texts", []) if t.get("status") == "bound"]
    if not regions:
        return None
    bounds = [np.asarray(t["polygon"], float) for t in regions]
    regions = sorted(zip(regions, bounds), key=lambda x: (x[1][:, 1].min(), x[1][:, 0].min()))
    heights = [max(1., p[:, 1].max()-p[:, 1].min()) for _, p in regions]
    cy = [p[:, 1].mean() for _, p in regions]
    inline = max(cy)-min(cy) <= .5*max(heights)
    if inline:
        regions.sort(key=lambda x: x[1][:, 0].min())
    texts = [normalized(t.get("text", "")) for t, _ in regions]
    adjacent = inline and all(
        0 <= b[:, 0].min()-a[:, 0].max() <= 2*max(heights)
        for (_, a), (_, b) in zip(regions, regions[1:]))
    value = ("" if adjacent else " | ").join(texts)
    confidence = [t.get("confidence") for t, _ in regions]
    eligible = bool(value) and all(c is not None and math.isfinite(c) and c >= policy.min_text_confidence for c in confidence)
    return dict(image_id=evidence["image_id"], text=value, eligible=eligible,
                polygon=np.concatenate([p for _, p in regions]).tolist(),
                camera_pose=evidence.get("camera_pose"))


def independent_views(rows):
    for i, a in enumerate(rows):
        for b in rows[i+1:]:
            if a["image_id"] == b["image_id"] or a["camera_pose"] is None or b["camera_pose"] is None:
                continue
            pa, pb = np.asarray(a["camera_pose"]), np.asarray(b["camera_pose"])
            angle = math.acos(np.clip((np.trace(pa[:3, :3].T @ pb[:3, :3])-1)/2, -1, 1))
            if np.linalg.norm(pa[:3, 3]-pb[:3, 3]) >= .15 or angle >= math.radians(10):
                return True
    return False


def resolve_identity(record, image_size, policy):
    rows = [r for e in record.get("text_evidence", []) if (r := reading(e, policy))]
    # Replays and multiple events on a source frame cannot increase support.
    rows = list({(r["image_id"], r["text"]): r for r in rows}.values())
    eligible = [r for r in rows if r["eligible"]]
    candidates = sorted({r["text"] for r in eligible})
    # User-selected policy: discard strict substrings; publish only a unique
    # remaining parent. Never choose by arrival order, majority or length alone.
    # image_size remains an unused compatibility argument; no clipping is required.
    maximal = [c for c in candidates if not any(c != other and c in other for other in candidates)]
    suppressed = [c for c in candidates if c not in maximal]
    chosen = maximal[0] if len(maximal) == 1 else None
    complete = [r for r in eligible if r["text"] == chosen]
    partial = [r for r in eligible if chosen and r["text"] != chosen]
    confirmed = chosen is not None and len({r["image_id"] for r in complete}) >= 3 and independent_views(complete)
    state = "identity_confirmed" if confirmed else "candidate" if chosen else "conflict" if eligible else "unknown"
    return dict(text=chosen, status=state, candidates=candidates,
                registration_status="accepted" if chosen else "rejected" if eligible else "no_eligible_text",
                maximal_candidates=maximal, ignored_substrings=suppressed,
                full_support_frames=sorted({r["image_id"] for r in complete}),
                compatible_partial_frames=sorted({r["image_id"] for r in partial}),
                readings=[dict(image_id=r["image_id"], text=r["text"],
                    relation="low_confidence" if not r["eligible"] else
                    "full_candidate" if r["text"] == chosen else
                    "ignored_substring" if r["text"] in suppressed else "rejected_noncontaining_reading") for r in rows],
                reason="substring_parent_selected" if partial else "noncontaining_readings_rejected" if state == "conflict" else "insufficient_full_views" if state == "candidate" else "no_eligible_text" if state == "unknown" else "consistent_full_views")


def build_view(records, image_size=lambda image_id: None, policy=None):
    policy = policy or ViewPolicy()
    views = []
    for record in records:
        if record.get("status") == "revoked":
            continue
        identity = resolve_identity(record, image_size, policy)
        # Copies only: legacy status/text and all evidence remain available.
        views.append(dict(record, raw_identity_text=record.get("identity_text"),
                          raw_identity_status=record.get("identity_status", "unknown"),
                          resolved_identity=identity))
    frequency = Counter(v["resolved_identity"]["text"] for v in views if v["resolved_identity"]["text"])
    for v in views:
        identity = v["resolved_identity"]
        text = identity["text"]
        generic = bool(text and text.upper() in policy.generic_texts)
        duplicate = bool(text and frequency[text] > 1)
        if duplicate and identity["status"] == "identity_confirmed":
            identity["status"] = "duplicate_text"
        if generic:
            decision, reason = "hide", "generic_text_attribute"
        elif v["category"].casefold() in policy.repeated_categories:
            if identity["candidates"]:
                decision, reason = "review", "text_on_repeated_structure_requires_carrier_check"
            else:
                decision, reason = "hide", "repeated_structure_without_distinguishing_evidence"
        elif v["category"].casefold() in ("sign", "label"):
            decision, reason = "keep", "marker_identity_candidate"
        else:
            decision, reason = "review", "facility_distinctiveness_unverified"
        v.update(identity_text=text, identity_status=identity["status"],
                 raw_duplicate_text=v.get("duplicate_text", False), duplicate_text=duplicate,
                 name=text if identity["status"] == "identity_confirmed" else None,
                 raw_reobservation_reasons=v.get("reobservation_reasons", []),
                 needs_reobservation=identity["status"] != "identity_confirmed" or decision == "review" or duplicate,
                 reobservation_reasons=([identity["reason"]] if identity["status"] != "identity_confirmed" else [])
                     + ([reason] if decision == "review" else [])
                     + (["text_seen_on_other_instance"] if duplicate else []),
                 landmark_view=dict(policy_version=policy.version, decision=decision,
                     reason=reason, duplicate_resolved_text=duplicate,
                     visible=decision != "hide", globally_unique_verified=False))
    return dict(policy=asdict(policy), landmarks=views,
                counts=dict(raw=len(views), decisions=dict(Counter(v["landmark_view"]["decision"] for v in views)),
                            identity_states=dict(Counter(v["identity_status"] for v in views))))


class LandmarkViewCache:
    """Rebuild only when landmark evidence changes, not on each geometry frame."""
    def __init__(self):
        self.key = None
        self.value = None

    def evaluate(self, records, observation):
        import json
        key = json.dumps(records, sort_keys=True, ensure_ascii=False)
        if key != self.key:
            self.value = build_view(records)
            self.key = key
        return self.value
