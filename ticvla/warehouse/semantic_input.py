"""Shared causal visual contract for annotation, training and navigation.

The raw RGB is never replaced. Candidate overlays are fallible current detections;
the semantic map contains only previously committed, source-bound evidence.
"""
import numpy as np
from PIL import ImageDraw
from .schema import file_sha256
from .geometry import render_map_image, transform_points
from .local_context import LANDMARK_LOCAL_INSTRUCTION

INSTRUCTION = (
    "The semantic map is past evidence in the CURRENT source body frame (metres, x forward, y left), "
    "not a map of ground truth. Orange labels are tentative objects; OCR text may be wrong. "
    "Gray space is unknown. Current detector boxes/IDs supplement, never replace, the raw RGB; "
    "detector labels are fallible. When current_candidates is supplied, Landmark selects only "
    "allowed_categories supported by same-frame masks AND current RGB, not IDs or remembered-only "
    "objects. Empty candidates require []. A selected category admits all its detected instances "
    "subject to depth/association checks; this is not per-instance selection. "
    "Use prior map evidence for Local/Route, but do not claim arrival or fresh visibility from memory. "
) + LANDMARK_LOCAL_INSTRUCTION


def map_records(landmarks, source, limit=16):
    inv = np.linalg.inv(np.asarray(source['T_odom_body']))
    records = []
    for lm in landmarks:
        if (lm['submap_id'] != source['submap_id'] or lm['status'] == 'revoked'
                or not lm.get('landmark_view', {}).get('visible', True)):
            continue
        registered_history = bool(lm.get('original_submap_id')
                                  and lm['original_submap_id'] != source['submap_id'])
        if not registered_history and lm['last_seen'] >= source['t_obs']:
            raise ValueError('Semantic input contains current/future evidence')
        records.append(dict(uid=lm['landmark_uid'], category=lm['category'],
            text=lm.get('identity_text'), identity_status=lm.get('identity_status', 'unknown'),
            status=lm['status'], position=transform_points(np.asarray([lm['position']]), inv)[0].tolist(),
            position_kind=lm.get('position_kind', 'visible_surface_center'),
            source_image_ids=lm['observation_ids'], last_seen=lm['last_seen'],
            **(dict(evidence_time_domain='registered_prior_episode', original_submap_id=lm['original_submap_id'])
               if registered_history else {}),
            needs_reobservation=lm.get('needs_reobservation', True)))
    records.sort(key=lambda r: (r['category'] not in ('sign', 'label'),
                               np.linalg.norm(r['position'][:2]), r['uid']))
    return records[:limit]


def render_semantic_map(grid, resolution, records, image_size=640):
    canvas = render_map_image(grid, resolution, image_size)
    draw = ImageDraw.Draw(canvas)
    span = len(grid) * resolution
    for i, r in enumerate(records):
        x, y = r['position'][:2]
        u, v = image_size*(.5-y/span), (image_size-40)*(.5-x/span)
        if 0 <= u < image_size and 0 <= v < image_size-40:
            draw.ellipse((u-4, v-4, u+4, v+4), fill='orange', outline='black')
            label = f"L{i+1} {r['category']} {r.get('text') or ''} [{r['status']}]"
            draw.text((max(0,min(u+5,image_size-220)), max(0,v-12)), label, fill='blue')
    return canvas


def visual_context(obs):
    """Single allowlist used by both teacher and deployment prompt builders."""
    view = obs.get('mapview', {})
    records = obs.get('map_landmarks', view.get('landmarks', []))
    contract = obs.get('semantic_input')
    result = dict(observed_map_landmarks=records[:16]) if records else {}
    if contract:
        for key in ('source_until', 'semantic_source_until'):
            if contract.get(key) is not None and contract[key] >= obs['t_obs']:
                raise ValueError('Semantic map cutoff must precede current source')
        for record in records:
            old_epoch = record.get('evidence_time_domain') == 'registered_prior_episode'
            if old_epoch and not record.get('original_submap_id'):
                raise ValueError('Registered map evidence lacks original submap')
            if (not old_epoch and record['last_seen'] >= obs['t_obs']
                    or obs['image_id'] in record['source_image_ids']):
                raise ValueError('Noncausal landmark evidence')
        result.update(semantic_input=contract, semantic_input_instruction=INSTRUCTION)
    candidate = obs.get('perception_candidates')
    if contract or candidate:
        result['semantic_prompt_version'] = 'candidate-landmark-local-v2'
    if candidate:
        current = [f for f in obs['rgb_history'] if f.get('image_id') == obs['image_id']
                   and f['t_obs'] == obs['t_obs']]
        if len(current) != 1:
            raise ValueError('Missing/ambiguous candidate current RGB')
        current = current[0]
        if (candidate['source_image_id'] != obs['image_id']
                or candidate['rgb_hash'] != file_sha256(current['path'])
                or candidate['overlay_hash'] != file_sha256(candidate['overlay'])):
            raise ValueError('Candidate source/image binding mismatch')
        geometry = candidate.get('source_geometry', {})
        if (geometry.get('t_obs') is not None and geometry['t_obs'] != obs['t_obs']
                or geometry.get('T_odom_body') is not None
                and not np.allclose(geometry['T_odom_body'], obs['T_odom_body'])):
            raise ValueError('Candidate source pose/time mismatch')
        result['current_candidates'] = dict(batch_id=candidate['batch_id'],
            source_image_id=candidate['source_image_id'],
            allowed_categories=list(dict.fromkeys(c['category'] for c in candidate['candidates'])),
            possibly_truncated=candidate['possibly_truncated'],
            instances=[{k:c[k] for k in ('instance_id','category','bbox_normalized','confidence')}
                       for c in candidate['candidates']])
        result.setdefault('semantic_input_instruction', INSTRUCTION)
    return result


def candidate_image(obs):
    visual_context(obs)  # Validate before exposing the file to either encoder.
    c = obs.get('perception_candidates')
    return c['overlay'] if c else None
