"""Resumable PFM conformers with verified native ETFlow exports and CPU metrics."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch_geometric.data import Batch, Data
from etflow.data.dataset import EuclideanDataset
from etflow.models.pfm_model import PosteriorFlow
from etflow.commons.covmat import CovMatEvaluator, print_covmat_results
from etflow.commons.pfm_artifacts import (
    EvaluationRun,
    atomic_file,
    digest,
    file_digest,
    run_lock,
    source_digest,
    write_json,
)


def evaluate_metrics(root, workers):
    run = EvaluationRun(root)
    packed, generation = run.load_complete()
    if workers < 1:
        raise ValueError("workers must be positive")
    evaluator = CovMatEvaluator(num_workers=workers, use_force_field=False)
    metric_identity = dict(
        run_id=run.run_id,
        packed_sha256=generation["packed_sha256"],
        source_sha256=source_digest(),
        use_force_field=False,
        filter_disconnected=evaluator.filter_disconnected,
        thresholds=evaluator.thresholds.tolist(),
        ratio=2,
        versions={p: version(p) for p in ["rdkit", "numpy"]},
    )
    metric_id = digest(metric_identity)
    folder = root / "metrics" / metric_id
    values, exclusions = [], []
    for i, data in enumerate(packed):
        path = folder / f"{i:06d}.json"
        if evaluator.filter_disconnected and "." in data.smiles:
            # This is the native input-eligibility rule, not a generated-quality filter.
            exclusion = dict(index=i, smiles=data.smiles, reason="upstream_disconnected_filter")
            record = dict(metric_id=metric_id, status="excluded", **exclusion)
            if path.exists() and json.loads(path.read_text()) != record:
                raise ValueError("Metric exclusion shard mismatch")
            if not path.exists():
                write_json(path, record)
            exclusions.append(exclusion)
            continue
        if path.exists():
            record = json.loads(path.read_text())
            if (
                record["metric_id"] != metric_id
                or record["index"] != i
                or record.get("status") != "evaluated"
            ):
                raise ValueError("Metric shard identity mismatch")
        else:
            # Keep the native symmetry-aware RMSD and COV/MAT definitions.
            # Commit one complete molecule so a CPU timeout also preserves work.
            results, rmsd = evaluator([data])
            if len(results["CoverageR"]) != 1:
                raise ValueError("Evaluator skipped a molecule")
            if any(not np.isfinite(v["rmsd"]).all() for v in rmsd.values()):
                raise ValueError("Nonfinite RMSD: inspect failures before reporting metrics")
            record = dict(
                metric_id=metric_id,
                index=i,
                status="evaluated",
                values={
                    key: np.asarray(results[key][0]).tolist()
                    for key in ["CoverageR", "CoverageP", "MatchingR", "MatchingP"]
                },
            )
            write_json(path, record)
        if set(record["values"]) != {"CoverageR", "CoverageP", "MatchingR", "MatchingP"}:
            raise ValueError("Invalid metric shard keys")
        for key, value in record["values"].items():
            expected = (len(evaluator.thresholds),) if key.startswith("Coverage") else ()
            if np.shape(value) != expected or not np.isfinite(value).all():
                raise ValueError("Invalid metric shard")
        values.append(record["values"])
    if not values:
        raise ValueError("No eligible molecules after the native disconnected-input filter")
    results = {
        key: np.asarray([v[key] for v in values])
        for key in ["CoverageR", "CoverageP", "MatchingR", "MatchingP"]
    }
    results["thresholds"] = evaluator.thresholds
    df, matching = print_covmat_results(results)
    coverage = folder / "coverage.csv"
    with atomic_file(coverage, "w") as f:
        df.to_csv(f, index=False)
    row = int(np.argmin(abs(results["thresholds"] - 0.5)))
    metrics = {
        **{k: float(v) for k, v in matching.items()},
        "COV-R_mean": float(results["CoverageR"][:, row].mean()),
        "COV-P_mean": float(results["CoverageP"][:, row].mean()),
        "molecules": len(values),
        "input_molecules": len(packed),
        "excluded_molecules": len(exclusions),
        "exclusions": exclusions,
        "threshold_angstrom": 0.5,
        "run_id": run.run_id,
        "packed_sha256": generation["packed_sha256"],
        "metric_id": metric_id,
        "metric_identity": metric_identity,
        "coverage_file": str(coverage.relative_to(root)),
        "coverage_sha256": file_digest(coverage),
        "use_force_field": False,
    }
    # The summary points only to a fully finished, versioned coverage table.
    write_json(root / "metrics.json", metrics)
    print(json.dumps(metrics))


def generate(args):
    scale = getattr(args, "diffusion_scale", None)
    if scale is not None and (not np.isfinite(scale) or scale < 0):
        raise ValueError("Diffusion scale must be finite and nonnegative")
    checkpoint_sha = file_digest(args.checkpoint)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint_sha != file_digest(args.checkpoint):
        raise ValueError("Checkpoint changed while loading")
    cfg = ckpt["pfm_config"]
    sampling = cfg["sampling"].copy()
    if scale is not None:
        sampling["diffusion_scale"] = float(scale)
    if getattr(args, "diffusion_schedule", None) is not None:
        sampling["diffusion_schedule"] = args.diffusion_schedule
    if args.steps is not None:
        sampling["n_timesteps"] = args.steps
    if sampling["batch_size"] < 1 or sampling["n_timesteps"] < 1:
        raise ValueError("Batch size and steps must be positive")
    dataset = EuclideanDataset(
        data_dir=Path(args.data_dir or cfg["data_dir"]), partition="qm9", split="test"
    )
    if args.max_molecules is not None and args.max_molecules < 1:
        raise ValueError("max-molecules must be positive")
    n = min(len(dataset), args.max_molecules or len(dataset))
    device = torch.device(args.device)
    identity = dict(
        schema=1,
        protocol="pfm-qm9-2K-v1",
        checkpoint_sha256=checkpoint_sha,
        mode=args.mode,
        sampling=sampling,
        seed=cfg["seed"],
        precision="fp32",
        execution=dict(
            device=device.type,
            cuda_runtime=torch.version.cuda,
            gpu=torch.cuda.get_device_name() if device.type == "cuda" else None,
            matmul_precision="highest",
        ),
        scope="full" if n == len(dataset) else "pilot",
        data_files=[{"file": p.name, "sha256": file_digest(p)} for p in dataset.data_files[:n]],
        source_sha256=source_digest(),
        versions={
            p: version(p) for p in ["torch", "torch-geometric", "torch-cluster", "rdkit", "numpy"]
        },
    )
    run = EvaluationRun(args.output_dir, identity)
    if (args.output_dir / "generation.json").exists():
        _, report = run.load_complete()
        print(f"Verified complete generation: {report['molecules']} molecules; no resampling")
        return
    model = PosteriorFlow(**cfg["model_args"])
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    torch.set_float32_matmul_precision("highest")
    chunk = sampling["batch_size"]
    sampler_args = {k: v for k, v in sampling.items() if k != "batch_size"}
    total = 0
    for i in range(n):
        saved = run.read_molecule(i)
        if saved is not None:
            total += len(saved["data"].pos_gen)
            print(f"{args.mode} reuse molecule {i + 1}/{n}", flush=True)
            continue
        data = dataset[i]
        ref = torch.load(dataset.data_files[i], weights_only=False).pos.numpy()
        if file_digest(dataset.data_files[i]) != identity["data_files"][i]["sha256"]:
            raise ValueError("Dataset changed while generating")
        generated, diagnostic, seconds = [], None, 0.0
        count = 2 * len(ref)
        for start in range(0, count, chunk):
            size = min(chunk, count - start)
            b = Batch.from_data_list([data] * size).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            began = time.perf_counter()
            sampled = model.sample(
                b.atomic_numbers,
                b.edge_index,
                b.batch,
                b.node_attr,
                b.edge_attr,
                sampler_type=args.mode,
                seed=cfg["seed"] + i * 100000 + start,
                return_trajectory=i < 8 and start == 0,
                **sampler_args,
            )
            result, frames = sampled if i < 8 and start == 0 else (sampled, [])
            if device.type == "cuda":
                torch.cuda.synchronize()
            seconds += time.perf_counter() - began
            generated.append(result.reshape(size, -1, 3).cpu().numpy())
            if i < 8 and start == 0:
                diagnostic = dict(smiles=data.smiles, num_atoms=len(data.pos), frames=frames)
        packed = Data(
            smiles=data.smiles, pos_ref=ref, rdmol=data.mol, pos_gen=np.concatenate(generated)
        )
        run.commit_molecule(i, packed, diagnostic, seconds)
        total += count
        print(f"{args.mode} committed molecule {i + 1}/{n}, conformers {total}", flush=True)
    if source_digest() != identity["source_sha256"]:
        raise ValueError("Source changed during generation")
    run.complete(args.checkpoint)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--mode", choices=["ode", "sde"], default="ode")
    parser.add_argument("--max-molecules", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument(
        "--diffusion-scale", type=float, help="Override sampler lambda; weights stay unchanged"
    )
    parser.add_argument("--diffusion-schedule", choices=["constant", "linear"])
    parser.add_argument("--metrics-only", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if not args.metrics_only and not args.checkpoint:
        parser.error("--checkpoint is required for generation")
    with run_lock(args.output_dir):
        if args.metrics_only:
            evaluate_metrics(args.output_dir, args.workers)
        else:
            generate(args)


if __name__ == "__main__":
    main()
