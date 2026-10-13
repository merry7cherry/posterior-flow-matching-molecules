import argparse

import json

from pathlib import Path

import sys

import numpy as np

import pytest

import torch

from rdkit import Chem

from rdkit.Chem import AllChem

from torch_geometric.data import Data

from etflow.commons.pfm_artifacts import (
    EvaluationRun,
    atomic_file,
    run_lock,
)

from etflow.models.pfm_model import PosteriorFlow

from etflow.cli import evaluate as eval_pfm

@pytest.fixture
def small_experiment(tmp_path):
    torch.set_num_threads(1)
    torch.manual_seed(9)
    data_dir = tmp_path / "data"
    folder = data_dir / "qm9/test"
    folder.mkdir(parents=True)
    for idx, smiles in enumerate(["CC", "CCO"]):
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        for i, atom in enumerate(mol.GetAtoms()):
            atom.SetAtomMapNum(i + 1)
        smiles = Chem.MolToSmiles(mol, canonical=False, allHsExplicit=True)
        pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)
        torch.save(
            Data(
                smiles=smiles,
                pos=torch.stack([pos, pos + 0.01]),
                atomic_numbers=torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()]),
            ),
            folder / f"{idx}.pt",
        )
    model_args = dict(
        hidden_channels=32,
        num_layers=2,
        num_heads=4,
        node_attr_dim=10,
        edge_attr_dim=1,
        so3_equivariant=True,
        clip_during_norm=True,
    )
    model = PosteriorFlow(**model_args)
    cfg = dict(
        model_args=model_args,
        seed=42,
        data_dir=str(data_dir),
        sampling=dict(
            batch_size=2, n_timesteps=2, terminal_time=0.99, diffusion_scale=0.1, probes=2
        ),
    )
    checkpoint = tmp_path / "model.ckpt"
    torch.save(dict(pfm_config=cfg, state_dict=model.state_dict()), checkpoint)
    return argparse.Namespace(
        checkpoint=checkpoint,
        data_dir=data_dir,
        output_dir=tmp_path / "run",
        mode="ode",
        steps=None,
        max_molecules=None,
        device="cpu",
    )

def generate(args):
    with run_lock(args.output_dir):
        eval_pfm.generate(args)

def test_resume_matches_uninterrupted_and_rejects_changed_identity(small_experiment, monkeypatch):
    args = small_experiment
    original = PosteriorFlow.sample
    calls = []

    def interrupt(self, *a, **kw):
        calls.append(kw["seed"])
        if len(calls) == 3:
            raise RuntimeError("injected interruption")
        return original(self, *a, **kw)

    monkeypatch.setattr(PosteriorFlow, "sample", interrupt)
    with pytest.raises(RuntimeError, match="injected"):
        generate(args)
    run = EvaluationRun(args.output_dir)
    assert run.read_molecule(0) is not None and run.read_molecule(1) is None
    assert not (args.output_dir / "generation.json").exists()
    with pytest.raises(ValueError, match="incomplete"):
        eval_pfm.evaluate_metrics(args.output_dir, 1)
    with pytest.raises(ValueError, match="incomplete"):
        EvaluationRun(args.output_dir).load_complete()
    # Only the unfinished molecule is regenerated, with its original seeds.
    calls.clear()
    generate(args)
    assert calls == [100042, 100044]
    resumed, report = run.load_complete()
    calls.clear()
    generate(args)
    assert calls == []
    assert report["molecules"] == 2 and report["conformers"] == 8
    monkeypatch.setattr(PosteriorFlow, "sample", original)
    args.output_dir = args.output_dir.parent / "continuous"
    generate(args)
    continuous, _ = EvaluationRun(args.output_dir).load_complete()
    for a, b in zip(resumed, continuous):
        np.testing.assert_array_equal(a.pos_gen, b.pos_gen)
    args.mode = "sde"
    with pytest.raises(ValueError, match="identity differs"):
        generate(args)
    args.mode, args.steps = "ode", 3
    with pytest.raises(ValueError, match="identity differs"):
        generate(args)
    args.steps = None
    changed = args.checkpoint.with_name("changed.ckpt")
    changed.write_bytes(args.checkpoint.read_bytes() + b"changed checkpoint")
    old_checkpoint, args.checkpoint = args.checkpoint, changed
    with pytest.raises(ValueError, match="identity differs"):
        generate(args)
    args.checkpoint = old_checkpoint
    p = args.data_dir / "qm9/test/0.pt"
    p.write_bytes(p.read_bytes() + b"changed data")
    with pytest.raises(ValueError, match="identity differs"):
        generate(args)

def test_publication_interruption_reuses_shards(small_experiment, monkeypatch):
    args = small_experiment
    import etflow.commons.pfm_artifacts as artifacts

    original_write = artifacts.write_json

    def fail_marker(path, value):
        if Path(path).name == "generation.json":
            raise RuntimeError("publication interrupted")
        return original_write(path, value)

    monkeypatch.setattr(artifacts, "write_json", fail_marker)
    with pytest.raises(RuntimeError, match="publication"):
        generate(args)
    assert (args.output_dir / "generated_files.pkl").exists()
    with pytest.raises(ValueError, match="incomplete"):
        EvaluationRun(args.output_dir).load_complete()
    monkeypatch.setattr(artifacts, "write_json", original_write)
    monkeypatch.setattr(
        PosteriorFlow, "sample", lambda *a, **kw: pytest.fail("resampled committed molecule")
    )
    generate(args)
    EvaluationRun(args.output_dir).load_complete()
    # Consumers reject both a damaged native export and a changed shard.
    packed = args.output_dir / "generated_files.pkl"
    previous = packed.read_bytes()
    packed.write_bytes(previous + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        eval_pfm.evaluate_metrics(args.output_dir, 1)
    packed.write_bytes(previous)
    shard = args.output_dir / "molecules/000000.pt"
    shard.write_bytes(shard.read_bytes() + b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        EvaluationRun(args.output_dir).load_complete()

def test_atomic_writes_lock_and_legacy_directory(tmp_path):
    target = tmp_path / "value"
    target.write_bytes(b"old")
    with pytest.raises(RuntimeError):
        with atomic_file(target) as f:
            f.write(b"partial")
            raise RuntimeError("interrupted write")
    assert target.read_bytes() == b"old"
    with run_lock(tmp_path):
        with pytest.raises(RuntimeError, match="in use"):
            with run_lock(tmp_path):
                pass
        with pytest.raises(ValueError, match="not empty"):
            EvaluationRun(tmp_path, dict(data_files=[]))
    assert target.read_bytes() == b"old"
    with pytest.raises(ValueError, match="No managed"):
        EvaluationRun(tmp_path)

def test_cpu_metrics_resume_preserves_native_aggregate(small_experiment, monkeypatch):
    args = small_experiment
    generate(args)
    original = eval_pfm.CovMatEvaluator.__call__
    calls = []

    def interrupted(self, packed, *a, **kw):
        calls.append(len(packed))
        if len(calls) == 2:
            raise RuntimeError("metrics interrupted")
        return original(self, packed, *a, **kw)

    monkeypatch.setattr(eval_pfm.CovMatEvaluator, "__call__", interrupted)
    with pytest.raises(RuntimeError, match="metrics interrupted"):
        eval_pfm.evaluate_metrics(args.output_dir, 2)
    assert not (args.output_dir / "metrics.json").exists()
    assert len(list((args.output_dir / "metrics").glob("*/000000.json"))) == 1
    calls.clear()
    eval_pfm.evaluate_metrics(args.output_dir, 2)
    assert calls == [1]
    report = json.loads((args.output_dir / "metrics.json").read_text())
    assert (args.output_dir / report["coverage_file"]).exists()
    packed, _ = EvaluationRun(args.output_dir).load_complete()
    monkeypatch.setattr(eval_pfm.CovMatEvaluator, "__call__", original)
    native, _ = eval_pfm.CovMatEvaluator(num_workers=1, use_force_field=False)(packed)
    _, matching = eval_pfm.print_covmat_results(native)
    for key, value in matching.items():
        assert report[key] == value
    row = int(np.argmin(abs(native["thresholds"] - 0.5)))
    assert report["COV-R_mean"] == native["CoverageR"][:, row].mean()
    assert report["COV-P_mean"] == native["CoverageP"][:, row].mean()

@pytest.mark.parametrize("all_disconnected", [False, True])
def test_native_disconnected_exclusions_are_explicit_and_resumable(
    small_experiment, monkeypatch, all_disconnected
):
    args = small_experiment
    folder = args.data_dir / "qm9/test"
    # Real disconnected molecular inputs exercise the native eligibility rule.
    for idx in [0, 1] if all_disconnected else [1]:
        mol = Chem.AddHs(Chem.MolFromSmiles("C.N"))
        AllChem.EmbedMolecule(mol, randomSeed=42)
        for i, atom in enumerate(mol.GetAtoms()):
            atom.SetAtomMapNum(i + 1)
        pos = torch.tensor(mol.GetConformer().GetPositions(), dtype=torch.float32)
        torch.save(
            Data(
                smiles=Chem.MolToSmiles(mol, canonical=False, allHsExplicit=True),
                pos=torch.stack([pos, pos + 0.01]),
                atomic_numbers=torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()]),
            ),
            folder / f"{idx}.pt",
        )
    generate(args)
    packed, generation = EvaluationRun(args.output_dir).load_complete()
    assert generation["molecules"] == 2
    if all_disconnected:
        for _ in range(2):
            with pytest.raises(ValueError, match="No eligible molecules"):
                eval_pfm.evaluate_metrics(args.output_dir, 1)
        assert not (args.output_dir / "metrics.json").exists()
        records = list((args.output_dir / "metrics").glob("*/*.json"))
        assert len(records) == 2
        assert all(json.loads(p.read_text())["status"] == "excluded" for p in records)
        return
    native, _ = eval_pfm.CovMatEvaluator(num_workers=1, use_force_field=False)(packed)
    _, matching = eval_pfm.print_covmat_results(native)
    eval_pfm.evaluate_metrics(args.output_dir, 1)
    summary = args.output_dir / "metrics.json"
    report = json.loads(summary.read_text())
    assert report["input_molecules"] == 2
    assert report["molecules"] == 1
    assert report["excluded_molecules"] == 1
    assert report["exclusions"] == [
        dict(index=1, smiles=packed[1].smiles, reason="upstream_disconnected_filter")
    ]
    assert report["metric_identity"]["filter_disconnected"] is True
    for key, value in matching.items():
        assert report[key] == value
    row = int(np.argmin(abs(native["thresholds"] - 0.5)))
    assert report["COV-R_mean"] == native["CoverageR"][:, row].mean()
    assert report["COV-P_mean"] == native["CoverageP"][:, row].mean()

    def unexpected(*a, **kw):
        raise AssertionError("Completed metrics must not be recomputed")

    monkeypatch.setattr(eval_pfm.CovMatEvaluator, "__call__", unexpected)
    eval_pfm.evaluate_metrics(args.output_dir, 1)
    assert json.loads(summary.read_text()) == report
    _, unchanged = EvaluationRun(args.output_dir).load_complete()
    assert unchanged == generation
    exclusion = args.output_dir / "metrics" / report["metric_id"] / "000001.json"
    record = json.loads(exclusion.read_text())
    record["reason"] = "wrong"
    exclusion.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="exclusion shard mismatch"):
        eval_pfm.evaluate_metrics(args.output_dir, 1)

def test_lambda_override_identity_zero_limit_and_invalid_inputs(small_experiment):
    args = small_experiment
    parent = args.output_dir.parent
    # Explicit lambda is numerically identical to the old default.
    args.mode = "sde"
    generate(args)
    default, _ = EvaluationRun(args.output_dir).load_complete()
    args.output_dir = parent / "explicit"
    args.diffusion_scale = 0.1
    generate(args)
    explicit, _ = EvaluationRun(args.output_dir).load_complete()
    assert all(np.array_equal(a.pos_gen, b.pos_gen) for a, b in zip(default, explicit))
    args.diffusion_scale = 0.2
    with pytest.raises(ValueError, match="identity"):
        generate(args)
    args.output_dir = parent / "ode-zero"
    args.mode = "ode"
    args.diffusion_scale = 0.0
    generate(args)
    ode, _ = EvaluationRun(args.output_dir).load_complete()
    args.output_dir = parent / "sde-zero"
    args.mode = "sde"
    generate(args)
    zero, _ = EvaluationRun(args.output_dir).load_complete()
    assert all(np.array_equal(a.pos_gen, b.pos_gen) for a, b in zip(ode, zero))
    for invalid in [-0.1, float("nan"), float("inf")]:
        args.output_dir = parent / "invalid"
        args.diffusion_scale = invalid
        with pytest.raises(ValueError, match="finite and nonnegative"):
            eval_pfm.generate(args)
        assert not args.output_dir.exists()