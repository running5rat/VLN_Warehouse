"""Bounded, real schema-v2 reference training; never an independent evaluation."""
import argparse
import gc
import hashlib
import json
import importlib.metadata
import time
from pathlib import Path

import numpy as np
import torch

from .. import ARCHITECTURE_VERSION
from ..config import WarehouseConfig
from ..data import RecordDataset, reference_targets
from ..schema import file_sha256, fingerprint
from ..train import atomic_save, load_reference, seed_all
from .core import export_project, read_json, write_json


def parameter_hash(model, trainable):
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if p.requires_grad == trainable:
            h.update(name.encode())
            h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def file_inventory(project, model_path, records, legacy=None):
    files = set(Path(project).rglob('*.json'))
    files.update(p for p in Path(model_path).rglob('*') if p.is_file())
    if legacy:
        files.add(Path(legacy))
    for r in records:
        o = r['observation']
        files.update(Path(h['path']) for h in o['rgb_history'] if h.get('path'))
        for key in ('map_image', 'map_grid'):
            if o.get(key): files.add(Path(o[key]))
        for key in ('source_episode',):
            if r['provenance'].get(key): files.add(Path(r['provenance'][key]))
    return {str(p): file_sha256(p) for p in sorted(files)}


def loss_parts(model, record):
    prediction, text = model.forward_record(record)
    parts = model.decoder.loss(prediction, reference_targets(record, model.cfg.device))
    parts['text'] = text
    return prediction, parts


def arrays(pred):
    return {k: v.detach().cpu().numpy() for k, v in pred.items()
            if isinstance(v, torch.Tensor) and k not in ('raw_kv', 'kv_mask')}


def plot_reference(path, gt, teacher, generated):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 6))
    for label, xy, mask in (
        ('approved GT', gt['reference_xy'], gt['reference_mask']),
        ('teacher-forced decoder', teacher['reference_xy'][0], teacher['reference_mask'][0]),
        ('generated-text decoder', generated['reference_xy'][0], generated['reference_mask'][0]),
    ):
        xy, mask = np.asarray(xy), np.asarray(mask, bool)
        ax.plot(xy[mask, 1], xy[mask, 0], '.-', label=label)
    ax.set(xlabel='body y / left (m)', ylabel='body x / forward (m)', aspect='equal')
    ax.invert_xaxis(); ax.grid(); ax.legend(fontsize=8)
    fig.savefig(path, dpi=120); plt.close(fig)


@torch.no_grad()
def evaluate(model, records, output, step, generate=True):
    model.eval()
    folder = output / f'check-{step:04d}'
    folder.mkdir()
    rows = []
    for i, r in enumerate(records):
        t = time.perf_counter()
        pred, parts = loss_parts(model, r)
        row = dict(sample_id=r['provenance']['sample_id'], losses={k: float(v) for k,v in parts.items()})
        teacher = arrays(pred)
        np.savez_compressed(folder / f'{i:02d}-teacher-forced.npz', **teacher)
        prompt, images = model.build_prompt(r['observation'])
        write_json(folder / f'{i:02d}-input.json', dict(prompt=prompt, images=images,
            image_sha256=[file_sha256(p) for p in images], observation=r['observation']))
        if generate:
            generated = model.generate_record(r['observation'])
            parse_error = None
            try:
                from ..structured_io import parse_structured
                parsed = parse_structured(generated['text'],
                    [h['image_id'] for h in r['observation']['rgb_history']],
                    structured_route=r['observation'].get('route_protocol')=='structured-v1',
                    compact_route=r['observation'].get('route_protocol')=='compact-v2',
                    current_image_id=r['observation']['image_id'])
            except ValueError as e:
                parsed = None; parse_error = str(e)
            ga = arrays(generated)
            np.savez_compressed(folder / f'{i:02d}-generated.npz', **ga)
            write_json(folder / f'{i:02d}-response.json', dict(text=generated['text'], parsed=parsed,
                parse_error=parse_error, query_complete=generated['query_complete'],
                diagnostic_query_appended=not generated['query_complete'],
                expected_text=r['supervision'].get('text'), quality_scope='training_examples_only'))
            plot_reference(folder / f'{i:02d}.png', r['supervision'], teacher, ga)
            row['format_valid'] = parsed is not None and generated['query_complete']
        row['elapsed_s'] = time.perf_counter()-t
        rows.append(row)
        print(json.dumps(dict(check=step, **row)), flush=True)
    means = {k: float(np.mean([r['losses'][k] for r in rows])) for k in rows[0]['losses']}
    result = dict(step=step, mean_losses=means, samples=rows, evaluation_scope='training_examples_only')
    write_json(folder / 'report.json', result)
    return result


def run(args):
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    versions = {name: importlib.metadata.version(name) for name in ('torch','transformers','numpy','Pillow')}
    write_json(output/'environment.json', versions)
    source_files = [Path(__file__), Path(__file__).parents[1]/'vlm.py',
                    Path(__file__).parents[1]/'models.py', Path(__file__).parents[1]/'data.py']
    write_json(output/'implementation-hashes.json', {str(p):file_sha256(p) for p in source_files})
    torch.set_num_threads(4)
    seed_all(42)
    export_project(args.project, output / 'export')
    manifest = read_json(output / 'export/manifest.json')
    manifest.update(role='diagnostic', optimizer_updates_permitted=True,
                    max_optimizer_steps=args.steps, evaluation_scope='training_examples_only',
                    authorization='explicit user request for bounded manual-demo training')
    # Separate run manifest; regression export remains unchanged.
    for e in manifest['records']: e['path'] = 'export/' + e['path']
    write_json(output / 'manifest.json', manifest)
    dataset = RecordDataset(output / 'manifest.json', 'diagnostic')
    records = [dataset[i] for i in range(len(dataset))]
    assert len(records) == 10 and all(not any(r['supervision']['action_mask']) for r in records)
    if not manifest['optimizer_updates_permitted'] or args.steps > manifest['max_optimizer_steps']:
        raise ValueError('Optimizer updates not permitted')
    cfg = WarehouseConfig(**{**read_json(Path(args.project)/'project.json')['config'],
        'model_path': args.model, 'vlm_checkpoint': args.legacy, 'manifest': str(output/'manifest.json'),
        'output_dir': str(output), 'batch_size': 1, 'accumulation_steps': 4,
        'seed': 42, 'max_steps': args.steps, 'max_new_tokens': 2048,
        'language_attention': 'sdpa', 'gradient_checkpointing': True})
    write_json(output/'config.json', cfg.to_dict())
    hashes = file_inventory(args.project, args.model, records, args.legacy)
    write_json(output/'source-hashes-before.json', hashes)
    from ..vlm import InternVLReference
    model = InternVLReference(cfg)
    frozen_before = parameter_hash(model, False)
    initial = {n:p.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    groups = [dict(params=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('decoder.')], lr=3e-5),
              dict(params=list(model.decoder.parameters()), lr=1e-4)]
    optimizer = torch.optim.AdamW(groups, weight_decay=cfg.weight_decay)
    checks = [evaluate(model, records, output, 0, generate=not args.legacy)]
    history, gradients = [], []
    rng = np.random.default_rng(42); order = []
    start = time.perf_counter()
    for step in range(1, args.steps+1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        losses = []
        for _ in range(4):
            if not order: order = list(rng.permutation(len(records)))
            _, parts = loss_parts(model, records[order.pop()])
            loss = sum(cfg.reference_loss_weights[k]*v for k,v in parts.items())
            if not torch.isfinite(loss): raise FloatingPointError('Nonfinite training loss')
            (loss/4).backward()
            losses.append({k:float(v.detach()) for k,v in parts.items()})
        norms = {}
        for label, select in [('lora', lambda n:not n.startswith('decoder.')), ('decoder', lambda n:n.startswith('decoder.'))]:
            ps = [p for n,p in model.named_parameters() if p.requires_grad and select(n)]
            norms[label] = float(torch.sqrt(sum((p.grad.float()**2).sum() for p in ps if p.grad is not None)))
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise AssertionError('Frozen parameter received gradient')
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        row = dict(step=step, losses={k:float(np.mean([l[k] for l in losses])) for k in losses[0]},
                   gradient_norms=norms, elapsed_s=time.perf_counter()-start,
                   peak_gpu_bytes=torch.cuda.max_memory_allocated())
        history.append(row); gradients.append(norms)
        with (output/'metrics.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
        print(json.dumps(row), flush=True)
        if step in (20, args.steps):
            checks.append(evaluate(model, records, output, step, generate=not args.legacy))
    model.eval()
    with torch.no_grad(): expected = arrays(model.forward_record(records[0])[0])
    deltas = {label:sum(float((p.detach().cpu()-initial[n]).abs().sum())
        for n,p in model.named_parameters() if p.requires_grad and (n.startswith('decoder.') == (label=='decoder')))
        for label in ('lora','decoder')}
    frozen_unchanged = frozen_before == parameter_hash(model, False)
    checkpoint = dict(architecture=ARCHITECTURE_VERSION, stage='reference', config=cfg.to_dict(),
        state_dict=model.adapter_state(), tokenizer_hash=fingerprint(model.tokenizer.get_vocab()),
        manifest_hash=fingerprint(manifest), manifest=manifest, step=args.steps,
        diagnostic_only=True, evaluation_scope='training_examples_only', optimizer=optimizer.state_dict())
    atomic_save(checkpoint, output/'last.pt')
    del checkpoint, optimizer, model, initial; gc.collect(); torch.cuda.empty_cache()
    reloaded, _ = load_reference(output/'last.pt'); reloaded.eval()
    with torch.no_grad(): actual = arrays(reloaded.forward_record(records[0])[0])
    reload_equal = all(np.array_equal(expected[k], actual[k]) for k in expected)
    after = {p:file_sha256(p) for p in hashes}
    write_json(output/'source-hashes-after.json', after)
    decrease = {k:checks[-1]['mean_losses'][k] < checks[0]['mean_losses'][k] for k in ('text','points')}
    passed = ((bool(args.legacy) or all(decrease.values())) and frozen_unchanged and hashes==after and reload_equal
              and all(v>0 for v in deltas.values()) and all(np.isfinite(list(g.values())).all() for g in gradients))
    report = dict(passed=passed, optimizer_steps=args.steps, sample_count=len(records), checks=checks,
        loss_decreased=decrease, frozen_unchanged=frozen_unchanged, source_files_unchanged=hashes==after,
        strict_reload_equal=reload_equal, parameter_change_l1=deltas,
        evaluation_scope='training_examples_only', navigation_generalization_evaluated=False,
        model_initialization='TIC-VLA' if args.legacy else 'InternVL3-1B',
        elapsed_training_and_checks_s=time.perf_counter()-start)
    write_json(output/'report.json', report)
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for key in ('text','points','length','validity'):
        ax.plot([r['step'] for r in history], [r['losses'][key] for r in history],label=key)
    ax.set(xlabel='optimizer step',ylabel='training loss'); ax.legend(); fig.savefig(output/'loss.png'); plt.close(fig)
    (output/'index.html').write_text('<meta charset="utf-8"><h1>Manual training diagnostic</h1><p>Same-episode fit, not generalization.</p><a href="report.json">Report</a><br><img src="loss.png">'+''.join(
        f'<h2>Step {c["step"]}</h2>'+''.join(f'<a href="check-{c["step"]:04d}/{i:02d}-response.json"><img width="350" src="check-{c["step"]:04d}/{i:02d}.png"></a>' for i in range(len(records))) for c in checks))
    print(json.dumps({k:v for k,v in report.items() if k!='checks'}),flush=True)
    return passed


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',required=True); p.add_argument('--output',required=True)
    p.add_argument('--model',default='/data/ticvla/models/InternVL3-1B')
    p.add_argument('--steps',type=int,default=100); p.add_argument('--legacy')
    a=p.parse_args()
    if a.steps not in (1,100): p.error('Use 100 steps for base, 1 step for legacy compatibility')
    if bool(a.legacy) != (a.steps==1): p.error('Legacy compatibility uses exactly one step')
    try:
        passed = run(a)
    except Exception as error:
        # Keep failed attempts reviewable; never replace them with a success report.
        if not isinstance(error, FileExistsError) and Path(a.output).is_dir() and not (Path(a.output)/'report.json').exists():
            write_json(Path(a.output)/'failure.json', dict(passed=False,
                error_type=type(error).__name__, error=str(error), stage='A'))
        raise
    raise SystemExit(0 if passed else 1)


if __name__=='__main__': main()
