"""Run a bounded compact-protocol pilot; reuse identical paid results, never retry them."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
from pathlib import Path
import shutil
from threading import Event

from .core import read_json, write_json
from .credentials import load_credentials
from .semantic_annotation import build_request, run_pilot, write_bundle
from ..schema import file_sha256


def run(pack, configs, output, reuse_qwen, credentials):
    pack, configs, output, reuse_qwen = map(Path, (pack, configs, output, reuse_qwen))
    manifest = read_json(pack/'comparison.json')
    if file_sha256(manifest['stage_B_report']) != manifest['stage_B_report_sha256']:
        raise ValueError('Stage B report changed')
    output.mkdir(parents=True, exist_ok=False)
    load_credentials(credentials)
    write_json(output/'source-pack.json', manifest)
    write_json(output/'provider-status.json', {
        m: dict(status='unavailable', reason='credit_balance_exhausted_user_confirmed_2026-10-26',
                new_api_calls=0) for m in ('gpt-6-luna', 'gpt-6.1-sol', 'gpt-6-astra')})
    for path in configs.glob('*.json'):
        cfg = read_json(path)
        cfg['price_checked_date'] = '2026-10-26'
        cfg['request_revision'] = 'current-categories-compact-route-v3'
        write_json(output/'configs'/path.name, cfg)
    jobs = {m: [] for m in ('glm-5.3-flashx', 'qwen3.8-max')}
    for b in manifest['bundles']:
        if not b['context_ready']: continue
        folder = Path(b['path'])
        if file_sha256(folder/'manifest.json') != b['manifest_sha256']:
            raise ValueError('Comparison bundle changed')
        bm = read_json(folder/'manifest.json')
        samples = []
        for r in bm['records']:
            if file_sha256(folder/r['path']) != r['sha256']:
                raise ValueError('Source sample changed')
            samples.append(read_json(folder/r['path']))
        for model in jobs:
            selected = samples
            if model == 'qwen3.8-max' and b['name'] == 'manual-0':
                cfg = read_json(output/'configs'/f'{model}.json')
                _, summary = build_request(samples[0], cfg)
                prior = read_json(reuse_qwen/'00000000/input.json')
                if summary['request_sha256'] != prior['request_sha256']:
                    raise ValueError('Previously paid Qwen request does not match; inspect instead of resubmitting')
                dest = output/model/'manual-0-reused'
                shutil.copytree(reuse_qwen, dest)
                rr = read_json(dest/'report.json')
                rr.update(actual_api_calls=0, reused_api_calls=1, reused_from=str(reuse_qwen),
                          reused_response_sha256=file_sha256(dest/'00000000/response.json'))
                for row in rr['samples']: row['reused'] = True
                write_json(dest/'report.json', rr)
                selected = samples[1:]
            bundle = output/'bundles'/model/b['name']
            write_bundle(bundle, selected, source=b['name'], context_mode=bm['context_mode'])
            jobs[model].append((bundle, output/'configs'/f'{model}.json', output/model/b['name']))
    # Two independent episode sequences per provider; each sequence remains causal.
    def lane(model, tasks, stop):
        for bundle, cfg, dest in tasks:
            if stop.is_set():
                bm = read_json(bundle/'manifest.json')
                write_json(dest/'report.json', dict(model=model, actual_api_calls=0, status='not_run_provider_error',
                    samples=[dict(source_image_id=r['image_id'], folder=f'{i:08d}',
                                  status='not_run_provider_error') for i,r in enumerate(bm['records'])]))
                continue
            report = run_pilot(bundle, cfg, dest)
            print(model, bundle.name, report['actual_api_calls'], report['format_valid_count'], flush=True)
            if any(r['status'] in ('not_run_provider_error','not_sent_preflight_rejected') or
                   str(r.get('reason','')).startswith('HTTP ') for r in report['samples']):
                stop.set()
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = []
        for model, tasks in jobs.items():
            stop = Event()
            futures.extend(pool.submit(lane, model, tasks[i::2], stop) for i in range(2))
        for f in futures: f.result()
    write_json(output/'run-complete.json', dict(completed=True, human_review_complete=False,
        source_pack_sha256=file_sha256(pack/'comparison.json'),
        code_sha256=file_sha256(__file__), pending_gold_context_samples_per_model=32))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('pack','configs','output','reuse-qwen','credentials'): p.add_argument('--'+key, required=True)
    a = p.parse_args()
    run(a.pack, a.configs, a.output, a.reuse_qwen, a.credentials)
