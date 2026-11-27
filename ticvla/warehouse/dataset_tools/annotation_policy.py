"""Isolate invalid teacher answers without hiding gaps in recursive evidence."""

CONTINUE = 'skip_invalid_keep_last_valid_v1'
STOP = 'stop_on_invalid_v1'


def failure_policy(cfg):
    policy = cfg.get('annotation_failure_policy', CONTINUE)
    if policy not in (CONTINUE, STOP):
        raise ValueError('Unknown annotation_failure_policy')
    return policy


def must_stop(cfg, exc, stage):
    # Transport/budget, source integrity and partially committed map failures
    # cannot be treated as harmless bad text or automatically retried.
    if stage != 'response_validation' or isinstance(exc, OSError):
        return True
    if getattr(exc, 'request_sent', True) is False:
        return True
    return failure_policy(cfg) == STOP


def gap_metadata(previous, rejected):
    return dict(
        policy=CONTINUE,
        context_source_image_id=(previous['observation']['image_id'] if previous else None),
        skipped_source_image_ids=list(rejected),
        original_instruction_if_no_valid_response=previous is None,
        human_context_fallback=False,
    )
