import copy
import numpy as np
import pytest
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data
from etflow.cli.chirality import correct, StereoCheck, process

def stereo_data(smiles="C[C@H](O)[C@@H](F)Cl"):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    assert AllChem.EmbedMolecule(mol, randomSeed=42) == 0
    pos = mol.GetConformer().GetPositions()
    return Data(smiles=smiles, rdmol=mol, pos_ref=np.array([pos]),
                pos_gen=np.array([pos, pos * [-1, 1, 1]]))

def test_reflection_preserves_geometry_and_uses_input_not_reference():
    original = stereo_data()
    expected = original.pos_gen.copy()
    output = correct(original)
    assert output.chirality_audit["flipped"] == [1]
    assert output.chirality_audit["counts"] == {"all_correct": 1, "all_opposite": 1}
    np.testing.assert_array_equal(original.pos_gen, expected)
    np.testing.assert_array_equal(output.pos_ref, original.pos_ref)
    np.testing.assert_allclose(output.pos_gen[0], output.pos_gen[1])
    for a, b in zip(original.pos_gen, output.pos_gen):
        np.testing.assert_allclose(np.linalg.norm(a[:, None] - a[None, :], axis=-1),
                                   np.linalg.norm(b[:, None] - b[None, :], axis=-1))
    changed_reference = copy.deepcopy(original)
    changed_reference.pos_ref *= [-1, 1, 1]
    np.testing.assert_array_equal(correct(changed_reference).pos_gen, output.pos_gen)
    assert correct(changed_reference).chirality_audit["reference_conflicts"]

def test_mixed_planar_and_unspecified_are_kept():
    data = stereo_data()
    centers = list(StereoCheck(data.rdmol).expected)
    mol = Chem.Mol(data.rdmol)
    mol.GetAtomWithIdx(centers[0]).InvertChirality()
    data.rdmol = mol
    assert StereoCheck(mol).classify(data.pos_gen[0]) == "mixed"
    np.testing.assert_array_equal(correct(data).pos_gen, data.pos_gen)
    flat = stereo_data()
    flat.pos_gen[:, :, 2] = 0
    assert correct(flat).chirality_audit["counts"] == {"near_planar": 2}
    unspecified = stereo_data("CC(O)C(F)Cl")
    assert correct(unspecified).chirality_audit["counts"] == {"no_specified_centers": 2}

def test_postprocessing_resumes_and_binds_parent(tmp_path):
    from etflow.commons.pfm_artifacts import EvaluationRun, run_lock
    source, target = tmp_path / "raw", tmp_path / "corrected"
    data = stereo_data()
    ident = dict(data_files=[{"file": "0.pt", "sha256": "fixture"}],
                 checkpoint_sha256="fixture", mode="ode", sampling={}, scope="pilot")
    diag = dict(smiles=data.smiles, num_atoms=len(data.pos_gen[0]), frames=[dict(
        t=.99, endpoint_mean=torch.tensor(data.pos_gen.reshape(-1, 3)),
        endpoint_variance=torch.ones(data.pos_gen.size // 3))])
    with run_lock(source):
        run = EvaluationRun(source, ident)
        run.commit_molecule(0, data, diag, 1.)
        parent = run.complete("fixture.ckpt")
    process(source, target)
    process(source, target)
    values, _ = EvaluationRun(target).load_complete()
    assert StereoCheck(values[0].rdmol).classify(values[0].pos_gen[1]) == "all_correct"
    assert EvaluationRun(source).load_complete()[1] == parent
    with pytest.raises(ValueError, match="independent"):
        process(source, source)
