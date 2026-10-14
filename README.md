# Moved to the PFM monorepo

The maintained **Molecules** implementation is now in
[`posterior-flow-matching/molecules/`](https://github.com/merry7cherry/posterior-flow-matching/tree/main/molecules).

PFM now uses one repository with four parallel directories: `images/`,
`pointcloud/`, `molecules/`, and `pusht/`. Follow the task README in the main
repository for installation, training, ODE/SDE sampling, evaluation and diagnostics.

This repository is archived for its release history. Its code and instructions
below describe the previous standalone release.

[All PFM code](https://github.com/merry7cherry/posterior-flow-matching#code) ·
[Project page](https://merry7cherry.github.io/posterior-flow-matching/)

---

# Posterior Flow Matching — Molecules

Standard Posterior Flow Matching (PFM) for **GEOM-QM9 conformer generation**, built on the native ET-Flow SO(3)-equivariant molecular network and featurization. Learn a velocity mean and atomwise variance, then use one model for ODE/SDE generation and endpoint posterior readouts.

[Project page](https://merry7cherry.github.io/posterior-flow-matching/) · [Paper PDF](https://merry7cherry.github.io/posterior-flow-matching/assets/paper.pdf) · [All tasks / Images](https://github.com/merry7cherry/posterior-flow-matching) · [Point clouds](https://github.com/merry7cherry/posterior-flow-matching-pointcloud) · [PushT](https://github.com/merry7cherry/posterior-flow-matching-pusht)

This release contains the PFM mainline: data preparation, training and same-run epoch-boundary recovery, ODE/SDE sampling, native COV/MAT evaluation, conservative chirality correction, and held-out posterior diagnostics. No trained weights or datasets are included.

## Install

Use Python 3.11 and a new environment. Linux CPU installation:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install torch-cluster==1.6.3 -f https://data.pyg.org/whl/torch-2.5.1+cpu.html
python -m pip install '.[test,visualize]'
```

Apple Silicon CPU uses the normal PyTorch wheel and compiles the matching PyG extension; Xcode Command Line Tools are required:

```bash
python -m pip install torch==2.5.1 numpy==1.26.4 setuptools wheel
CXXFLAGS=-Wno-invalid-specialization MAX_JOBS=2 python -m pip install torch-cluster==1.6.3 --no-build-isolation
python -m pip install '.[test,visualize]'
```

The Clang flag handles a PyTorch 2.5 header diagnostic on recent macOS SDKs. CPU installation and runtime checks are recorded in [VALIDATION.md](VALIDATION.md). CUDA, mixed precision, distributed training, speed, and benchmark reproduction were not validated for this release.

## Quick CPU example

Create RDKit-generated **fixture data**, not GEOM benchmark data, and run the native network with a reduced-size configuration:

```bash
pfm-molecules-example --output-dir data/example
pfm-molecules-train --config configs/cpu-smoke.yaml --data-dir data/example --run-dir runs/example --cpu
pfm-molecules-evaluate --checkpoint runs/example/checkpoints/final.ckpt --data-dir data/example --output-dir runs/example-ode --mode ode --device cpu
pfm-molecules-evaluate --output-dir runs/example-ode --metrics-only --workers 1
pfm-molecules-visualize --run-dir runs/example-ode --output runs/example-ode/conformer.png
```

The example checks plumbing and finite outputs; its metrics are not scientific results. All commands work from an installed wheel without Git; pass configuration and data paths explicitly when working outside this checkout.

## GEOM-QM9 data

Obtain GEOM raw QM9 conformer pickles from the [GEOM dataset](https://dataverse.harvard.edu/api/access/datafile/4327252). Obtain the QM9 `split.npy` and `test_mols.pkl` from the [Torsional Diffusion data instructions](https://github.com/gcorso/torsional-diffusion#training-model). Keep the **complete raw set**: split indices refer to the sorted raw `*.pickle` filenames. Dataset files have their own terms; they are not distributed by this repository. Only load trusted pickle/checkpoint files.

```bash
pfm-molecules-prepare --raw-dir /path/to/rdkit_folder/qm9 --metadata-dir /path/to/QM9 --output-dir data/processed
```

This preserves the supplied validation/test sets, checks atom ordering, and quarantines training graphs that duplicate held-out molecules. Training and validation retain at most 30 conformers ranked by the supplied Boltzmann weights. The output includes manifests and `preparation.json`; data live at `data/processed/qm9/{train,val,test}/*.pt`.

## Train and resume

```bash
pfm-molecules-train --config configs/qm9.yaml --data-dir data/processed --run-dir runs/qm9 --cpu
# After an interruption, resume the same configuration and planned training length:
pfm-molecules-train --config configs/qm9.yaml --data-dir data/processed --run-dir runs/qm9 --resume runs/qm9/checkpoints/last.ckpt --cpu
```

`configs/qm9.yaml` retains the native 20-layer, 160-channel network. Configure batch size, epochs, seed and optimization before starting. The public CPU configuration is not a claim to recreate the historical paper training run. `configs/cpu-smoke.yaml` reduces width/depth only for installation checks. Checkpoints retain model, optimizer, scheduler, global step, configuration and RNG states. Resume is supported at saved epoch boundaries; a partial epoch is rerun from the last completed checkpoint. Changing the configured schedule is rejected on resume. `--stop-after-epoch N` exercises an interruption without changing the planned schedule.

The training objective is the Gaussian NLL on the centered `3(N-1)`-dimensional coordinate subspace. Mean, covariance, covariance divergence and SDE noise all use the same centering projection. [ARCHITECTURE.md](ARCHITECTURE.md) describes the implementation.

## Sample, correct chirality and evaluate

```bash
pfm-molecules-evaluate --checkpoint runs/qm9/checkpoints/final.ckpt --data-dir data/processed --output-dir runs/qm9-ode --mode ode --device cpu
pfm-molecules-evaluate --checkpoint runs/qm9/checkpoints/final.ckpt --data-dir data/processed --output-dir runs/qm9-sde --mode sde --device cpu
pfm-molecules-chirality --source runs/qm9-sde --output runs/qm9-sde-corrected
pfm-molecules-evaluate --output-dir runs/qm9-sde-corrected --metrics-only --workers 1
```

Generation uses twice the reference conformer count per molecule, 50 integration steps by default, and the endpoint mean at terminal time 0.99. The main configuration uses the manuscript's linear schedule `lambda(t)=0.3*(1-t)` in both drift and noise. The API retains `constant` mode for compatibility; explicitly select `--diffusion-schedule constant` if needed. Both modes include the projected covariance divergence. `--max-molecules`, `--steps` and `--diffusion-scale` permit smaller runs or explicit sampler changes; a changed configuration requires a new output directory.

Generation commits one molecule at a time and resumes completed shards after verifying identities. Native heavy-atom, symmetry-aware RMSD produces COV-R/COV-P (higher is better) and MAT-R/MAT-P in angstroms (lower is better), with coverage summarized at 0.5 Å. Disconnected inputs are explicitly excluded from metrics. No force-field relaxation is applied. Correction reflects a whole conformer only when all specified tetrahedral centers disagree with input stereo; mixed/unresolved cases are kept, no conformers are dropped and reference RMSD never selects a correction. Raw outputs remain separate.

## Posterior diagnostics

```bash
pfm-molecules-diagnostics --checkpoint runs/qm9/checkpoints/final.ckpt --config configs/diagnostics.json --data-root data/processed --output runs/qm9-diagnostics --device cpu
pfm-molecules-visualize --run-dir runs/qm9-sde --output runs/qm9-sde/conformer.png
```

Diagnostics freeze variance bins and a time-only covariance baseline on validation molecules, then measure grouped residual moments and Gaussian NLL on test reference interpolations. Molecules are weighted equally. They do not establish full conditional calibration or final-sample risk. Visualization colors are endpoint covariance trace; after chirality correction, variance values remain raw-model readouts, not a recalibrated posterior.

## Tests and provenance

```bash
python -m pytest -q
python -m build
```

See [VALIDATION.md](VALIDATION.md) for checks actually executed and their limits. [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) records ET-Flow and TorchMD-Net attribution. The source snapshot keeps the PFM mainline only; previous task variants, non-QM9 datasets, special continuation schedules and cluster/campaign tooling are excluded. The original research history is not part of this release.

## Citation and license

Please cite **Posterior Flow Matching** (2026 preprint; [CITATION.cff](CITATION.cff)) and the upstream ET-Flow work when using its architecture. See [ET-Flow](https://github.com/shenoynikhil/ETFlow) for its paper and citation. Code is MIT with the retained upstream notices in [LICENSE](LICENSE) and [LICENSES](LICENSES).
