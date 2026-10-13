"""Conservative whole-conformer reflection using input stereo, never reference RMSD."""

import argparse
from collections import Counter
import copy
from importlib.metadata import version
from pathlib import Path
import time

import numpy as np
from rdkit import Chem
import torch
from etflow.commons.pfm_artifacts import EvaluationRun, file_digest, run_lock, write_json

TETRA = {Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW}
POLICY = dict(name="input-stereo-global-reflection-v1", axis=0,
              normalized_volume_min=1e-3, unresolved="keep", reference_selection=False)


class StereoCheck:
    def __init__(self, template):
        self.expected = {a.GetIdx(): a.GetChiralTag() for a in template.GetAtoms()
                         if a.GetChiralTag() in TETRA}
        self.neighbors = {i: [a.GetIdx() for a in template.GetAtomWithIdx(i).GetNeighbors()]
                          for i in self.expected}
        self.mol = Chem.Mol(template)
        self.mol.RemoveAllConformers()
        conf = Chem.Conformer(template.GetNumAtoms())
        conf.Set3D(True)
        self.mol.AddConformer(conf, assignId=True)

    def classify(self, positions):
        if not self.expected:
            return "no_specified_centers"
        pos = np.asarray(positions, dtype=np.float64)
        if pos.shape != (self.mol.GetNumAtoms(), 3) or not np.isfinite(pos).all():
            raise ValueError("Invalid stereo coordinates")
        for i, neighbors in self.neighbors.items():
            if len(neighbors) != 4:
                return "unsupported_center"
            points = pos[neighbors]
            scale = np.sqrt(np.mean(np.sum((points - pos[i]) ** 2, axis=1)))
            volume = abs(np.linalg.det(points[:3] - points[3]))
            if scale < 1e-12 or volume / scale**3 < POLICY["normalized_volume_min"]:
                return "near_planar"
        self.mol.GetConformer().SetPositions(pos)
        Chem.AssignAtomChiralTagsFromStructure(self.mol, confId=0, replaceExistingTags=True)
        observed = [self.mol.GetAtomWithIdx(i).GetChiralTag() for i in self.expected]
        if any(tag not in TETRA for tag in observed):
            return "undetermined"
        matches = [tag == expected for tag, expected in zip(observed, self.expected.values())]
        return "all_correct" if all(matches) else ("all_opposite" if not any(matches) else "mixed")


def correct(data):
    checker = StereoCheck(data.rdmol)
    result = copy.deepcopy(data)
    result.pos_gen = np.array(data.pos_gen, copy=True)
    stats, flipped = Counter(), []
    for j, pos in enumerate(data.pos_gen):
        before = checker.classify(pos)
        stats[before] += 1
        if before == "all_opposite":
            candidate = np.array(pos, copy=True)
            candidate[:, POLICY["axis"]] *= -1
            if checker.classify(candidate) != "all_correct":
                raise ValueError("Global reflection failed stereo revalidation")
            result.pos_gen[j] = candidate
            flipped.append(j)
    # Audit reference consistency separately; it never controls a correction.
    conflicts = []
    for j, pos in enumerate(data.pos_ref):
        status = checker.classify(pos)
        if status not in {"all_correct", "no_specified_centers"}:
            conflicts.append(dict(reference_index=j, status=status))
    result.chirality_audit = dict(counts=dict(stats), flipped=flipped,
                                 specified_centers=len(checker.expected),
                                 reference_conflicts=conflicts)
    return result


def transformed_diagnostic(diagnostic, data):
    if diagnostic is None:
        return None
    result = copy.deepcopy(diagnostic)
    n = data.pos_gen.shape[1]
    for frame in result["frames"]:
        mean = frame["endpoint_mean"].reshape(-1, n, 3)
        for j in data.chirality_audit["flipped"]:
            if j < len(mean):
                mean[j, :, POLICY["axis"]] *= -1
    result["postprocessing"] = (
        "Original diagnostic means reflected using the final stereo decision. "
        "Variance values remain original PFM diagnostics, not postprocessed posterior calibration.")
    return result


def process(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("Use independent source and output directories")
    with run_lock(source), run_lock(output):
        original = EvaluationRun(source)
        _, complete = original.load_complete()
        identity = copy.deepcopy(original.identity)
        identity["postprocessing"] = dict(
            policy=POLICY, source_run_id=original.run_id,
            source_packed_sha256=complete["packed_sha256"],
            source_shard_sha256=complete["shard_sha256"],
            implementation_sha256=file_digest(__file__), rdkit=version("rdkit"))
        target = EvaluationRun(output, identity)
        if (output / "generation.json").exists():
            target.load_complete()
            if not (output / "chirality.json").exists():
                raise ValueError("Missing correction audit")
            return
        all_counts, eligible_counts, conflicts = Counter(), Counter(), []
        changed, eligible_changed, seconds = 0, 0, 0.
        for i in range(original.n):
            saved = target.read_molecule(i)
            if saved is None:
                parent = original.read_molecule(i)
                began = time.perf_counter()
                corrected = correct(parent["data"])
                corrected.chirality_audit["processing_seconds"] = time.perf_counter() - began
                diagnostic = transformed_diagnostic(parent["diagnostic"], corrected)
                target.commit_molecule(i, corrected, diagnostic, parent["sampling_seconds"])
                saved = target.read_molecule(i)
            audit = saved["data"].chirality_audit
            all_counts.update(audit["counts"])
            changed += len(audit["flipped"])
            if "." not in saved["data"].smiles:
                eligible_counts.update(audit["counts"])
                eligible_changed += len(audit["flipped"])
            seconds += audit["processing_seconds"]
            if audit["reference_conflicts"]:
                conflicts.append(dict(molecule_index=i, references=audit["reference_conflicts"]))
        write_json(output / "chirality.json", dict(
            policy=POLICY, source_run_id=original.run_id, output_run_id=target.run_id,
            counts=dict(all_counts), eligible_counts=dict(eligible_counts),
            corrected_conformers=changed, eligible_corrected_conformers=eligible_changed,
            processing_seconds=seconds, reference_conflicts=conflicts,
            reference_coordinates_unchanged=True, conformers_dropped=0))
        target.complete(complete["checkpoint"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    process(args.source, args.output)


if __name__ == "__main__":
    main()
