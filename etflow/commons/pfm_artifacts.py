"""Identity, atomic molecule commits and complete-result reads for PFM evaluation.

One directory is one immutable generation identity. Molecule shards are the
recovery authority; the native packed pickle is a verified export for ETFlow.
"""

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile

import numpy as np
import torch


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def source_digest():
    # Hash installed Python sources; independent of checkout location and .git.
    root = Path(__file__).resolve().parents[1]
    files = sorted(root.rglob("*.py"))
    return digest([(str(p.relative_to(root)), file_digest(p)) for p in files])


@contextmanager
def atomic_file(path, mode="wb"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".tmp-", dir=path.parent)
    try:
        with os.fdopen(fd, mode) as f:
            yield f
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def write_json(path, value):
    with atomic_file(path, "w") as f:
        json.dump(value, f, indent=2, allow_nan=False)


@contextmanager
def run_lock(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    # Never unlink a lock file: another process could still hold its inode.
    with (root / ".lock").open("a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Evaluation directory is in use: {root}") from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


class EvaluationRun:
    """Call under run_lock for both writers and consumers."""

    def __init__(self, root, identity=None):
        self.root = Path(root)
        manifest = self.root / "identity.json"
        if manifest.exists():
            self.identity = json.loads(manifest.read_text())
            if identity is not None and identity != self.identity:
                raise ValueError("Generation identity differs; choose a new output directory")
        else:
            if identity is None:
                raise ValueError("No managed generation identity; regenerate in a new directory")
            existing = [
                p
                for p in self.root.iterdir()
                if p.name != ".lock" and not p.name.startswith(".tmp-")
            ]
            if existing:
                raise ValueError("Unmanaged output directory is not empty; choose a new directory")
            self.identity = identity
            write_json(manifest, identity)
        self.run_id = digest(self.identity)
        self.n = len(self.identity["data_files"])

    def shard_path(self, index):
        if not 0 <= index < self.n:
            raise ValueError("Molecule index outside generation identity")
        return self.root / "molecules" / f"{index:06d}.pt"

    def read_molecule(self, index):
        path = self.shard_path(index)
        if not path.exists():
            return None
        record = torch.load(path, map_location="cpu", weights_only=False)
        self.validate(record, index)
        return record

    def validate(self, record, index):
        if record["run_id"] != self.run_id or record["index"] != index:
            raise ValueError("Molecule shard identity mismatch")
        data = record["data"]
        ref, gen = np.asarray(data.pos_ref), np.asarray(data.pos_gen)
        if (
            ref.ndim != 3
            or ref.shape[0] < 1
            or ref.shape[2] != 3
            or gen.shape != (2 * len(ref), *ref.shape[1:])
            or not np.isfinite(ref).all()
            or not np.isfinite(gen).all()
        ):
            raise ValueError("Invalid conformer shapes, counts or coordinates")
        elapsed = record["sampling_seconds"]
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Invalid sampling time")
        diagnostic = record["diagnostic"]
        if index < 8:
            if (
                diagnostic is None
                or diagnostic["smiles"] != data.smiles
                or diagnostic["num_atoms"] != ref.shape[1]
            ):
                raise ValueError("Missing or mismatched molecular diagnostic")
            frames = diagnostic["frames"]
            if not frames:
                raise ValueError("Empty diagnostic")
            for frame in frames:
                mean, var = frame["endpoint_mean"], frame["endpoint_variance"]
                if (
                    mean.ndim != 2
                    or mean.shape[1] != 3
                    or len(mean) < ref.shape[1]
                    or var.shape != (len(mean),)
                    or not torch.isfinite(mean).all()
                    or not torch.isfinite(var).all()
                    or (var < 0).any()
                    or not 0 <= frame["t"] < 1
                ):
                    raise ValueError("Invalid endpoint diagnostic")
            if not np.array_equal(frames[-1]["endpoint_mean"][: ref.shape[1]].numpy(), gen[0]):
                raise ValueError("Diagnostic endpoint does not match the generated conformer")

    def commit_molecule(self, index, data, diagnostic, seconds):
        record = dict(
            run_id=self.run_id,
            index=index,
            data=data,
            diagnostic=diagnostic,
            sampling_seconds=seconds,
        )
        self.validate(record, index)
        if self.shard_path(index).exists():
            raise ValueError("Refusing to overwrite a committed molecule")
        with atomic_file(self.shard_path(index)) as f:
            torch.save(record, f)

    def complete(self, checkpoint):
        records = [self.read_molecule(i) for i in range(self.n)]
        if not records or any(r is None for r in records):
            raise ValueError("Generation is incomplete")
        packed = [r["data"] for r in records]
        path = self.root / "generated_files.pkl"
        with atomic_file(path) as f:
            pickle.dump(packed, f)
        total = sum(len(d.pos_gen) for d in packed)
        seconds = sum(r["sampling_seconds"] for r in records)
        report = dict(
            run_id=self.run_id,
            checkpoint=str(checkpoint),
            checkpoint_sha256=self.identity["checkpoint_sha256"],
            mode=self.identity["mode"],
            sampling=self.identity["sampling"],
            molecules=self.n,
            conformers=total,
            sampling_seconds=seconds,
            seconds_per_conformer=seconds / total,
            precision="fp32",
            scope=self.identity["scope"],
            packed_sha256=file_digest(path),
            shard_sha256=[file_digest(self.shard_path(i)) for i in range(self.n)],
        )
        # Completion marker is published last. Incomplete exports are never consumed.
        write_json(self.root / "generation.json", report)
        return report

    def load_complete(self):
        path = self.root / "generation.json"
        if not path.exists():
            raise ValueError("Generation is incomplete; resume generation before consuming it")
        report = json.loads(path.read_text())
        packed_path = self.root / "generated_files.pkl"
        if (
            report["run_id"] != self.run_id
            or report["molecules"] != self.n
            or report["packed_sha256"] != file_digest(packed_path)
            or report["shard_sha256"] != [file_digest(self.shard_path(i)) for i in range(self.n)]
        ):
            raise ValueError("Completed generation artifact identity or checksum mismatch")
        with packed_path.open("rb") as f:
            packed = pickle.load(f)
        if len(packed) != self.n or sum(len(d.pos_gen) for d in packed) != report["conformers"]:
            raise ValueError("Completed generation count mismatch")
        return packed, report
