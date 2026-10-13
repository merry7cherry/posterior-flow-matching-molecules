"""Keep held-out molecules fixed while removing duplicate training identities."""

import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest


def test_training_leakage_quarantine_is_repeatable(tmp_path, monkeypatch):
    from etflow.cli.prepare import finalize

    root = tmp_path / "processed"
    root.mkdir()
    metadata = tmp_path / "metadata"
    metadata.mkdir()
    splits = np.array([np.array([0, 1]), np.array([2]), np.array([3])], dtype=object)
    np.save(metadata / "split.npy", splits)
    files = [Path(f"{i}.pickle") for i in range(4)]
    entries = {
        "train": [("0.pt", "A"), ("1.pt", "B")],
        "val": [("2.pt", "A")],
        "test": [("3.pt", "C")],
    }
    for split, rows in entries.items():
        folder = root / "qm9" / split
        folder.mkdir(parents=True)
        manifest = [
            {"file": name, "smiles": smiles, "atoms": 3, "conformers": 2} for name, smiles in rows
        ]
        (root / f"{split}.json").write_text(json.dumps(manifest))
        for name, _ in rows:
            (folder / name).write_bytes(b"unchanged conformer fixture")
    held_out = [(root / f"{s}.json").read_bytes() for s in ["val", "test"]]
    args = SimpleNamespace(output_dir=root, metadata_dir=metadata)
    for _ in range(2):
        finalize(args, files, splits)
        report = json.loads((root / "preparation.json").read_text())
        assert report["counts"] == {"train": 1, "val": 1, "test": 1}
        assert report["excluded_train_records"] == 1
        assert report["skipped_raw"] == {"train": [], "val": []}
        assert (root / "excluded-train/0.pt").read_bytes() == b"unchanged conformer fixture"
        assert [(root / f"{s}.json").read_bytes() for s in ["val", "test"]] == held_out
    (root / "test.json").write_text(json.dumps([{"file": "3.pt", "smiles": "A"}]))
    with pytest.raises(ValueError, match="Validation/test"):
        finalize(args, files, splits)
