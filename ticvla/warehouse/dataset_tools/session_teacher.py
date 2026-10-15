"""Sequential, auditable conversation-assisted proposals, without an API call.

This is an import/review interface, not an inference endpoint. The assistant reads
one prepared input and explicitly submits its answer. Human review is never implied.
"""
import copy
import json
from pathlib import Path

from ..local_context import local_context, local_memory
from ..schema import file_sha256, fingerprint
from .core import read_json, write_json
from .semantic_annotation import (validate_sample, assemble_answer, agreement,
                                  render_report)
from .teacher import teacher_input, TEACHER_INSTRUCTION


def _samples(bundle):
    bundle = Path(bundle).resolve()
    manifest = read_json(bundle / 'manifest.json')
    if manifest.get('protocol') not in ('semantic-teacher-pilot-v1','semantic-episode-review-v1') or manifest.get('context_mode') != 'recursive':
        raise ValueError('A recursive semantic pilot bundle is required')
    records = manifest['records']
    if not 1 <= len(records) <= (128 if manifest['protocol']=='semantic-episode-review-v1' else 6):
        raise ValueError('Source count exceeds the declared session scope')
    samples = []
    for record in records:
        path = (bundle / record['path']).resolve()
        if not path.is_relative_to(bundle) or file_sha256(path) != record['sha256']:
            raise ValueError('Bundle source mismatch')
        sample = validate_sample(read_json(path))
        if sample['observation']['image_id'] != record['image_id']:
            raise ValueError('Manifest image ID mismatch')
        samples.append(sample)
    observations = [s['observation'] for s in samples]
    if (len({o['task']['task_id'] for o in observations}) != 1 or
        len({o['task']['instruction'] for o in observations}) != 1 or
        any(b['t_obs'] <= a['t_obs'] for a, b in zip(observations, observations[1:]))):
        raise ValueError('Sources must be one chronological task')
    return samples


def start_session(bundle, output):
    samples = _samples(bundle)
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    report = dict(protocol='conversation-semantic-pilot-v1', status='awaiting_answer',
        source='codex_conversation_assisted', context_mode='recursive',
        bundle=str(Path(bundle).resolve()), bundle_sha256=file_sha256(Path(bundle)/'manifest.json'),
        expected_samples=len(samples), samples=[], actual_api_calls=0,
        real_teacher_interface_pass=False, api_interface_verified=False,
        model_id=None, usage=None, inference_latency_seconds=None,
        context_isolation=False, independent_blind_evaluation=False,
        context_limitations='Assistant conversation already discussed manual labels and later scene content; not a blind comparison.',
        human_approved=False, semantic_accuracy=None, training=False,
        waypoint_and_reference_teacher_generated=False)
    write_json(output/'report.json', report)
    return prepare_next(output)


def _load(output):
    output = Path(output); report = read_json(output/'report.json')
    if report.get('protocol') != 'conversation-semantic-pilot-v1':
        raise ValueError('Not a conversation session')
    if file_sha256(Path(report['bundle'])/'manifest.json') != report['bundle_sha256']:
        raise ValueError('Bundle manifest changed')
    return output, report, _samples(report['bundle'])


def _causal_sample(samples, index, output, report):
    sample = copy.deepcopy(samples[index]); obs = sample['observation']
    obs['route'] = obs['task']['instruction']; obs['route_revision'] = index
    obs['previous_local'] = dict(valid=False, reason='no_previous_session_response')
    obs['execution']['route_context_source'] = 'original_instruction'
    if index:
        row = report['samples'][index-1]; folder = output/row['folder']
        if row['status'] != 'format_valid_unreviewed':
            raise ValueError('Previous response failed; no repair or human fallback')
        if file_sha256(folder/'proposal.json') != row['proposal_sha256']:
            raise ValueError('Previous proposal changed')
        previous = read_json(folder/'proposal.json')
        obs['route'] = previous['parsed']['route']
        obs['execution']['route_context_source'] = 'previous_unreviewed_session_proposal'
        if 'T_odom_body' in obs:
            memory = local_memory(previous['parsed']['local'], samples[index-1]['observation'],
                origin='unreviewed_session_proposal', revision=row['raw_sha256'])
            obs['previous_local'] = local_context(memory, obs)
        else:
            obs['previous_local'] = dict(valid=False, reason='external_robot_base_pose_not_audited')
    # Evaluation text and future geometry remain outside the presentation payload.
    return sample


def _presentation(sample):
    context, paths = teacher_input(sample)
    images = [dict(path=p, sha256=file_sha256(p), label=(
        'RGB '+context['source_rgb'][i]['image_id'] if i < len(context['source_rgb']) else 'Causal observed MapView'))
        for i, p in enumerate(paths)]
    result = dict(instructions=TEACHER_INSTRUCTION, context=context, images=images,
        source_binding=sample['provenance']['source_binding'],
        source='codex_conversation_assisted', request_ready=True,
        generation_constraints='Instructions only; post-generation deployment grammar validation; no constrained decoding claim')
    result['input_sha256'] = fingerprint(result)
    return result


def prepare_next(output):
    output, report, samples = _load(output)
    if report['status'] != 'awaiting_answer':
        return dict(status=report['status'], report=str(output/'report.json'))
    index = len(report['samples'])
    sample = _causal_sample(samples, index, output, report)
    inp = _presentation(sample)
    folder = output/f'{index:08d}'; folder.mkdir(exist_ok=True)
    if (folder/'input.json').exists():
        if read_json(folder/'input.json') != inp:
            raise ValueError('Pending input changed')
    else:
        write_json(folder/'input.json', inp)
        write_json(folder/'geometric-supervision.json', sample['geometric_supervision'])
    return dict(status='awaiting_answer', index=index, input=str(folder/'input.json'),
                input_sha256=inp['input_sha256'], source_binding=inp['source_binding'])


def submit_answer(output, response_file):
    output, report, samples = _load(output)
    if report['status'] != 'awaiting_answer':
        raise ValueError('Session is complete or stopped')
    index = len(report['samples']); folder = output/f'{index:08d}'
    sample = _causal_sample(samples, index, output, report)
    inp = read_json(folder/'input.json')
    if inp != _presentation(sample):
        raise ValueError('Prepared source/input changed')
    if (folder/'raw.txt').exists():
        raise ValueError('Response already archived; refusing overwrite')
    raw = Path(response_file).read_text()
    (folder/'raw.txt').write_text(raw)
    row = dict(index=index, folder=folder.name, source_image_id=sample['observation']['image_id'],
        raw_sha256=file_sha256(folder/'raw.txt'), input_sha256=inp['input_sha256'])
    try:
        envelope = json.loads(raw)
        if set(envelope) != {'input_sha256', 'source_binding', 'answer'}:
            raise ValueError('Response requires input_sha256, source_binding and answer only')
        if envelope['input_sha256'] != inp['input_sha256'] or envelope['source_binding'] != inp['source_binding']:
            raise ValueError('Response/input binding mismatch')
        proposal = assemble_answer(envelope['answer'], sample)
        checks = agreement(proposal['parsed'], sample)
        proposal.update(source='codex_conversation_assisted', raw_sha256=row['raw_sha256'],
            input_sha256=inp['input_sha256'], rule_checks_passed=not checks['rule_flags'])
        for key in ('landmark', 'local', 'route'):
            proposal['field_sources'][key] = 'codex_conversation_assisted'
        write_json(folder/'proposal.json', proposal)
        row.update(status='rule_rejected_unreviewed' if checks['rule_flags'] else 'format_valid_unreviewed',
            rule_flags=checks['rule_flags'], proposal_sha256=file_sha256(folder/'proposal.json'))
    except (ValueError, KeyError, TypeError) as exc:
        row.update(status='failed', reason=str(exc))
    report['samples'].append(row)
    if row['status'] != 'format_valid_unreviewed':
        report['status'] = 'stopped_failed_response'
    elif len(report['samples']) == len(samples):
        report['status'] = 'completed_awaiting_human_review'
    report['format_valid_count'] = sum(r['status'] in ('format_valid_unreviewed','rule_rejected_unreviewed') for r in report['samples'])
    report['rule_pass_count'] = sum(r['status'] == 'format_valid_unreviewed' for r in report['samples'])
    report['all_samples_completed'] = report['rule_pass_count'] == len(samples)
    report['format_success_fraction_all_selected'] = report['format_valid_count']/len(samples)
    # Reveal comparison only after the sequence ends, so later inputs never contain gold answers.
    if report['status'] != 'awaiting_answer':
        comparisons = []
        for r in report['samples']:
            p = output/r['folder']/'proposal.json'
            if p.exists():
                comparison = agreement(read_json(p)['parsed'], samples[r['index']])
                write_json(p.parent/'comparison.json', comparison)
                r['comparison'] = comparison
                if comparison.get('human_reference_available'):
                    comparisons.append(comparison)
        report['human_comparison_count'] = len(comparisons)
        report['agreement_counts'] = {k:sum(bool(c.get(k)) for c in comparisons) for k in
            ('local_exact','route_action_exact','route_subgoal_exact','route_status_exact','remaining_instruction_exact')}
        report['agreement_interpretation'] = 'Non-blind agreement, not semantic accuracy; failed/unrun samples remain in denominator.'
    write_json(output/'report.json', report)
    if report['status'] != 'awaiting_answer':
        render_report(output, report)
    return dict(status=report['status'], last_result=row['status'], processed=len(report['samples']),
                expected=len(samples))
