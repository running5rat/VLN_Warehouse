"""Shared English Local limits for teacher validation and deployed decoding."""
LOCAL_MAX_CHARACTERS = 90
LOCAL_PATTERN = rf"[A-Za-z0-9][A-Za-z0-9 ,.';:()_/-]{{0,{LOCAL_MAX_CHARACTERS - 1}}}"
COMPLETION_EVENTS = frozenset({'target_visible', 'stopped', 'arrival_confirmed'})

def completion_evidence(observation):
    events = {e['kind'] for e in observation.get('execution', {}).get('events', [])}
    return 'confirmed' if COMPLETION_EVENTS <= events else 'unverified'
