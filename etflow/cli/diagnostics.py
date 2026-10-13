"""Fixed-checkpoint reference-state diagnostics; no trajectory or training updates.

Each molecule is an atomic NPZ commit. The validation-derived design is frozen
before test inference. All bootstrap units are molecules, never individual atoms.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch_geometric.data import Batch

from etflow.commons.pfm_artifacts import atomic_file, digest, file_digest, write_json, source_digest, run_lock
from etflow.data.dataset import EuclideanDataset
from etflow.models.pfm_model import PosteriorFlow
from etflow.models import pfm_probability as prob


def pair_indices(count, maximum, repeats, seed):
    rng = np.random.default_rng(seed)
    selected = np.sort(rng.choice(count, min(count, maximum), replace=False))
    return np.repeat(selected, repeats)


def atom_statistics(residual_v, marginal_v, remaining):
    """Trace compares with 3D squared error, not with one-coordinate error."""
    error_v = np.square(residual_v).sum(-1)
    trace_v = 3 * marginal_v
    if not np.isfinite(trace_v).all() or np.any(trace_v <= 0):
        raise ValueError("Invalid predicted projected variance")
    return dict(error_v=error_v, trace_v=trace_v,
                error_x=remaining**2 * error_v, trace_x=remaining**2 * trace_v,
                ratio=error_v / trace_v)


def weighted_quantiles(values, weights, quantiles):
    order = np.argsort(values)
    x, w = values[order], weights[order]
    p = (np.cumsum(w) - 0.5 * w) / w.sum()
    return np.interp(quantiles, p, x)


def freeze_design(files, config):
    """Equal-molecule quantile bins and isotropic scale fit using validation only."""
    loaded = [dict(np.load(p, allow_pickle=False)) for p in files]
    edges, scales = [], []
    for ti, t in enumerate(config['times']):
        u = [x['trace_x'][ti].reshape(-1) for x in loaded]
        w = [np.full(len(a), 1 / len(a)) for a in u]
        cuts = weighted_quantiles(np.concatenate(u), np.concatenate(w),
                                  np.linspace(0, 1, config['bins'] + 1)[1:-1])
        edges.append(cuts.tolist())
        # g(t)P has marginal trace 3*g(t)*(1-1/N). Fit subspace MSE.
        scales.append(float(np.mean([x['error_v'][ti].mean() /
                                     (3 * (1 - 1 / int(x['num_atoms']))) for x in loaded])))
    return dict(times=config['times'], inner_bin_edges=edges,
                time_only_preprojection_variance=scales,
                validation_files={p.name: file_digest(p) for p in files},
                weighting=config['weighting'])


def summarize(files, design, config):
    records = [dict(np.load(p, allow_pickle=False)) for p in files]
    rng = np.random.default_rng(config['seed'] + 900)
    boot = rng.integers(len(records), size=(config['bootstrap_repeats'], len(records)))

    def interval(values):
        a = np.asarray(values, dtype=float)
        means = a[boot].mean(axis=1)
        return dict(mean=float(a.mean()), low=float(np.quantile(means, .025)),
                    high=float(np.quantile(means, .975)))

    result = []
    for ti, t in enumerate(config['times']):
        stats = {k: interval([x[k][ti].mean() for x in records])
                 for k in ['trace_x', 'error_x', 'ratio', 'nll', 'bound_fraction']}
        baseline_u, baseline_ratio, baseline_nll, means = [], [], [], []
        sums = np.zeros((len(records), config['bins'], 4))
        for mi, x in enumerate(records):
            n = int(x['num_atoms'])
            g = design['time_only_preprojection_variance'][ti]
            base_v = 3 * g * (1 - 1/n)
            u, e = x['trace_x'][ti].ravel(), x['error_x'][ti].ravel()
            idx = np.searchsorted(design['inner_bin_edges'][ti], u, side='right')
            for bi in range(config['bins']):
                mask = idx == bi
                # Weighted contributions retain equal molecule mass globally.
                sums[mi, bi] = [u[mask].sum()/len(u), e[mask].sum()/len(e),
                                mask.mean(), int(mask.sum())]
            baseline_u.append((1-t)**2 * base_v)
            baseline_ratio.append(x['error_v'][ti].mean()/base_v)
            baseline_nll.append(.5 * (x['error_v'][ti].mean() / (3*g*(1-1/n)) + np.log(g)))
            means.append(x['residual_v'][ti].mean(axis=(0, 1)))
        stats['baseline_trace_x'] = interval(baseline_u)
        stats['baseline_ratio'] = interval(baseline_ratio)
        stats['baseline_nll'] = interval(baseline_nll)
        stats['residual_mean'] = [interval(np.asarray(means)[:, k]) for k in range(3)]
        bins = []
        for bi in range(config['bins']):
            a = sums[:, bi]
            mass = a[:, 2].sum()
            if mass == 0:
                bins.append(dict(bin=bi, observations=0))
                continue
            bs = a[boot].sum(axis=1)
            valid = bs[:, 2] > 0
            estimate = bs[valid, 1]/bs[valid, 2]
            bins.append(dict(bin=bi, predicted=float(a[:, 0].sum()/mass),
                             observed=float(a[:, 1].sum()/mass),
                             low=float(np.quantile(estimate, .025)),
                             high=float(np.quantile(estimate, .975)),
                             observations=int(a[:, 3].sum()),
                             molecules=int((a[:, 2]>0).sum())))
        result.append(dict(t=t, **stats, bins=bins))
    return dict(molecules=len(records), timepoints=result,
                interval_unit='molecule; fixed checkpoint, excludes training-seed uncertainty',
                interpretation='Grouped first/second-moment diagnostics, not final-sample risk or full conditional calibration')


def collect(dataset, indices, model, output, config, device, identity_id):
    output.mkdir(parents=True, exist_ok=True)
    paths, excluded = [], []
    started = time.perf_counter()
    for rank, index in enumerate(indices):
        source = dataset.data_files[index]
        raw = torch.load(source, map_location='cpu', weights_only=False)
        if '.' in raw.smiles:
            excluded.append(dict(index=int(index), file=source.name, reason='disconnected_input'))
            continue
        path = output / f'{index:06d}.npz'
        if path.exists():
            with np.load(path, allow_pickle=False) as old:
                if str(old['identity_id']) != identity_id or str(old['data_sha256']) != file_digest(source):
                    raise ValueError('Resume identity mismatch')
            paths.append(path)
            continue
        split_offset = 0 if dataset.split == 'val' else 1000000
        seed = config['seed'] + split_offset + int(index)*100
        conf = pair_indices(len(raw.pos), config['max_conformers'], config['noise_repeats'], seed)
        data = dataset[index]
        n, pairs = len(data.pos), len(conf)
        batch = Batch.from_data_list([data]*pairs).to(device)
        x1 = prob.center(raw.pos[conf].reshape(-1, 3).to(device).float(), batch.batch)
        gen = torch.Generator(device=device).manual_seed(seed+1)
        x0 = prob.center(torch.randn(x1.shape, device=device, generator=gen) * model.source_scale, batch.batch)
        values = {}
        for t in config['times']:
            xt = (1-t)*x0 + t*x1
            with torch.no_grad():
                v, lv = model.predict_posterior(batch.atomic_numbers,
                    torch.full((pairs, 1), t, device=device), xt, batch.edge_index,
                    batch.batch, batch.node_attr, batch.edge_attr)
                residual = (x1-x0-v)
                marginal = prob.marginal_variance(lv.exp(), batch.batch)
                a = atom_statistics(residual.cpu().numpy().reshape(pairs, n, 3),
                    marginal.cpu().numpy().reshape(pairs, n), 1-t)
                a['nll'] = prob.projected_nll(residual, lv, batch.batch).cpu().numpy()
                lo, hi = model.logvar_min, model.logvar_max
                a['bound_fraction'] = (((lv-lo)<.01*(hi-lo)) | ((hi-lv)<.01*(hi-lo))).float().cpu().numpy().reshape(pairs,n)
                a['residual_v'] = residual.cpu().numpy().reshape(pairs,n,3)
                a['marginal_v'] = marginal.cpu().numpy().reshape(pairs,n)
            for k, value in a.items():
                if not np.isfinite(value).all():
                    raise ValueError(f'Nonfinite {k} at molecule {index}, t={t}')
                values.setdefault(k, []).append(value)
        payload = {k: np.stack(v) for k,v in values.items()}
        payload.update(identity_id=identity_id, data_sha256=file_digest(source),
                       molecule_index=int(index), num_atoms=n, conformer_indices=conf,
                       pair_seed=seed+1, times=np.array(config['times']), atomic_numbers=raw.atomic_numbers.numpy())
        with atomic_file(path) as f:
            np.savez_compressed(f, **payload)
        paths.append(path)
        if rank % 10 == 0:
            print(json.dumps(dict(split=dataset.split, completed=rank+1, total=len(indices),
                                  elapsed_seconds=time.perf_counter()-started)), flush=True)
    write_json(output/'manifest.json', dict(identity_id=identity_id, molecules=len(paths),
               excluded=excluded, files={p.name:file_digest(p) for p in paths},
               elapsed_seconds=time.perf_counter()-started))
    return paths


def run(args):
    config = json.loads(args.config.read_text())
    sha = file_digest(args.checkpoint)
    source_sha = source_digest()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    model_cfg = ckpt['pfm_config']['model_args']
    model = PosteriorFlow(**model_cfg).to(args.device).eval()
    model.load_state_dict(ckpt['state_dict'])
    datasets = {s: EuclideanDataset(args.data_root, split=s, partition='qm9') for s in ['val','test']}
    val_count = min(len(datasets['val']), config['validation_molecules'])
    val_indices = np.sort(np.random.default_rng(config['seed']).choice(len(datasets['val']), val_count, replace=False))
    test_indices = np.arange(min(len(datasets['test']), args.max_molecules or len(datasets['test'])))
    identities = {s:[dict(index=int(i),name=datasets[s].data_files[i].name,
                         sha256=file_digest(datasets[s].data_files[i]))
                    for i in indices] for s,indices in [('val',val_indices),('test',test_indices)]}
    identity = dict(config=config, checkpoint_sha256=sha, source_sha=source_sha,
                    subset=args.max_molecules, selected_data=identities, torch=torch.__version__,
                    device=torch.cuda.get_device_name() if args.device=='cuda' else args.device)
    args.output.mkdir(parents=True, exist_ok=True)
    identity_path = args.output/'identity.json'
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError('Output identity differs')
    write_json(identity_path, identity)
    identity_id = digest(identity)
    val = collect(datasets['val'], val_indices, model, args.output/'val', config, args.device, identity_id)
    design = freeze_design(val, config)
    frozen = args.output/'frozen-design.json'
    if frozen.exists() and json.loads(frozen.read_text()) != design:
        raise ValueError('Validation design changed')
    write_json(frozen, design)
    test = collect(datasets['test'], test_indices, model, args.output/'test', config, args.device, identity_id)
    write_json(args.output/'summary.json', summarize(test, design, config))
    write_json(args.output/'acceptance.json', dict(passed=True, identity_id=identity_id,
        checkpoint_sha256=sha, source_sha=source_sha, subset=args.max_molecules,
        validation_molecules=len(val), test_molecules=len(test),
        design_sha256=file_digest(frozen), summary_sha256=file_digest(args.output/'summary.json')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cpu', choices=['cuda','cpu'])
    parser.add_argument('--max-molecules', type=int)
    args = parser.parse_args()
    with run_lock(args.output):
        run(args)


if __name__ == "__main__":
    main()
