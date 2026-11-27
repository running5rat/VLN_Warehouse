"""Compact model output; source identities remain explicit in internal records."""

import re

CATEGORY_PATTERN = r"[A-Za-z][A-Za-z0-9 -]{0,39}"
CURRENT_LANDMARK_INSTRUCTION = (
    'landmark is a JSON array of at most three distinct short English category strings, '
    'for example ["sign","fire extinguisher"], or []. These are format examples, not answers. '
    'Only register objects visible in the CURRENT source RGB. Historical RGB may inform '
    'Local and Route, but never supplies landmark candidates. Do not output image_id, '
    'category keys, reasons or roles; the program binds categories to the current source. '
    'When current_candidates is supplied, select only its allowed_categories with a '
    'source-bound segmentation mask AND supporting current RGB evidence. Copy the exact '
    'candidate category name; do not invent or rename classes. Empty candidates require []. '
    'Reject detector false positives. A visible object missed by segmentation must not be '
    'registered in landmark in this mode. Without current_candidates, use the current-RGB rule.'
)
LEGACY_LANDMARK_INSTRUCTION = (
    'Legacy landmark format for this record: output at most three objects with image_id '
    'and category, using supplied RGB source IDs. Preserve historical source references.'
)


def current_source_id(observation):
    """Use explicit source/time metadata, never the ordering of historical frames."""
    frames = [f for f in observation["rgb_history"] if f.get("path")
              and f.get("t_obs") == observation["t_obs"]]
    source = observation.get("image_id")
    if source:
        frames = [f for f in frames if f["image_id"] == source]
    ids = {f["image_id"] for f in frames}
    if len(ids) != 1:
        raise ValueError("Missing or ambiguous current source RGB")
    return ids.pop()


def bind_landmarks(items, image_ids, current_image_id=None):
    """Resolve compact categories without guessing which of several images is current.

    Legacy object records are validated by parse_structured and retain their source.
    Mixed schemas are rejected. Repeated compact categories count once as evidence.
    """
    if not isinstance(items, list) or len(items) > 3:
        raise ValueError("Landmark must be an array of at most three entries")
    if not items:
        return []
    if all(isinstance(item, dict) for item in items):
        return items
    if not all(isinstance(item, str) for item in items):
        raise ValueError("Mixed or invalid landmark schemas")
    if current_image_id is None and len(image_ids) == 1:
        current_image_id = image_ids[0]
    if not current_image_id or current_image_id not in image_ids:
        raise ValueError("Compact landmarks require an available explicit current source RGB")
    result, seen = [], set()
    for item in items:
        category = " ".join(item.split()).lower()
        if len(item) > 40 or not re.fullmatch(CATEGORY_PATTERN, category):
            raise ValueError("Invalid compact landmark category")
        if category not in seen:
            result.append(dict(image_id=current_image_id, category=category))
            seen.add(category)
    return result
