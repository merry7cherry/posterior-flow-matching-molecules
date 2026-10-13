"""Prepare only QM9, preserving the published SORTED-pickle split convention."""

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import numpy as np
import torch
from tqdm import tqdm
from rdkit import Chem
from etflow.data.prepare import process_mol, process_test_mol
from etflow.commons.featurization import MoleculeFeaturizer


def identity(data):
    mol = MoleculeFeaturizer().get_mol(data.smiles)
    numbers = torch.tensor([a.GetAtomicNum() for a in mol.GetAtoms()])
    if not torch.equal(numbers, data.atomic_numbers):
        raise ValueError("Atom order differs between coordinates and featurized molecule")
    # Atom-mapped SMILES must retain original coordinate indices.
    mapping = [a.GetAtomMapNum() for a in mol.GetAtoms()]
    if len(set(mapping)) == len(mapping) and mapping != sorted(mapping):
        raise ValueError("Non-monotonic atom mapping")
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)


def finalize(args, files, splits):
    """Preserve held-out sets; quarantine training graphs duplicated across splits."""
    manifests = {
        split: json.loads((args.output_dir / f"{split}.json").read_text())
        for split in ["train", "val", "test"]
    }
    identities = {s: {x["smiles"] for x in m} for s, m in manifests.items()}
    if identities["val"] & identities["test"]:
        raise ValueError("Validation/test molecule leakage; held-out protocol needs review")
    blocked = identities["val"] | identities["test"]
    excluded_path = args.output_dir / "excluded-train.json"
    excluded = json.loads(excluded_path.read_text()) if excluded_path.exists() else []
    prior = {x["file"] for x in excluded}
    for entry in manifests["train"]:
        if entry["smiles"] in blocked and entry["file"] not in prior:
            excluded.append({**entry, "reason": "same molecular graph as validation or test"})
    excluded_path.write_text(json.dumps(excluded, indent=2))
    quarantine = args.output_dir / "excluded-train"
    quarantine.mkdir(exist_ok=True)
    for entry in excluded:
        src = args.output_dir / "qm9" / "train" / entry["file"]
        dst = quarantine / entry["file"]
        if src.exists():
            src.replace(dst)
        elif not dst.exists():
            raise ValueError(f"Missing excluded source {src}")
    excluded_files = {x["file"] for x in excluded}
    manifests["train"] = [x for x in manifests["train"] if x["file"] not in excluded_files]
    (args.output_dir / "train.json").write_text(json.dumps(manifests["train"]))
    report = {
        "raw_files": len(files),
        "split_sha256": hashlib.sha256((args.metadata_dir / "split.npy").read_bytes()).hexdigest(),
        "ordering": "sorted *.pickle (Torsional Diffusion convention)",
        "counts": {s: len(m) for s, m in manifests.items()},
        "excluded_train_records": len(excluded),
        "held_out_policy": "Preserve original val/test; exclude matching training molecular graphs",
        "identity_overlap": 0,
        "skipped_raw": {},
    }
    for split, indices in zip(["train", "val"], splits[:2]):
        present = {x["file"] for x in manifests[split]}
        if split == "train":
            present |= excluded_files
        report["skipped_raw"][split] = [
            files[int(i)].name for i in indices if files[int(i)].stem + ".pt" not in present
        ]
    identities = {s: {x["smiles"] for x in m} for s, m in manifests.items()}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        assert not identities[a] & identities[b]
    for split, manifest in manifests.items():
        actual = {p.name for p in (args.output_dir / "qm9" / split).glob("*.pt")}
        if actual != {x["file"] for x in manifest}:
            raise ValueError(f"Files differ from the {split} manifest")
    (args.output_dir / "preparation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "skipped_raw"}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--metadata-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Audit existing manifests without reprocessing conformers",
    )
    args = parser.parse_args()
    files = sorted(args.raw_dir.glob("*.pickle"))
    if not files:
        raise ValueError("No raw QM9 pickle files")
    splits = np.load(args.metadata_dir / "split.npy", allow_pickle=True)
    if len(splits) != 3:
        raise ValueError("Expected train/validation/test split indices")
    sets = [set(map(int, s)) for s in splits]
    if any(sets[i] & sets[j] for i in range(3) for j in range(i)):
        raise ValueError("Overlapping raw split indices")
    if args.audit_only:
        finalize(args, files, splits)
        return
    for split, indices in zip(["train", "val"], splits[:2]):
        folder = args.output_dir / "qm9" / split
        folder.mkdir(parents=True, exist_ok=True)
        manifest = []
        for idx in tqdm(sorted(indices), desc=split):
            path = files[int(idx)]
            out = folder / (path.stem + ".pt")
            data = (
                torch.load(out, weights_only=False)
                if out.exists()
                else process_mol(path, "qm9", split)
            )
            if data is None:
                continue
            key = identity(data)
            torch.save(data, out)
            manifest.append(
                {
                    "file": out.name,
                    "smiles": key,
                    "conformers": len(data.pos),
                    "atoms": len(data.atomic_numbers),
                }
            )
        (args.output_dir / f"{split}.json").write_text(json.dumps(manifest))
    manifest = []
    with (args.metadata_dir / "test_mols.pkl").open("rb") as f:
        test = pickle.load(f)
    folder = args.output_dir / "qm9" / "test"
    folder.mkdir(parents=True, exist_ok=True)
    for key, mols in tqdm(test.items(), desc="test"):
        data = process_test_mol(mols, "qm9")
        if data is None:
            raise ValueError(f"Failed to process test molecule {key}")
        canonical = identity(data)
        out = folder / (hashlib.md5(key.encode()).hexdigest() + ".pt")
        torch.save(data, out)
        manifest.append(
            {
                "file": out.name,
                "smiles": canonical,
                "conformers": len(data.pos),
                "atoms": len(data.atomic_numbers),
            }
        )
    (args.output_dir / "test.json").write_text(json.dumps(manifest))
    finalize(args, files, splits)


if __name__ == "__main__":
    main()
