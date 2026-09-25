# test42 — single-phase SiO2 crystal ablation of test40

**A deliberately narrowed diagnostic, not a new model.** Same architecture and
today's changes as `../test40` (batched (frame, sigma) samples per step,
sigma/cell conditioning re-injected after every message-passing block,
width=128, layers=5, cutoff=6) — with the phase dimension removed entirely
and trained on **only** the crystal reference: 62 NPT thermal snapshots of
one fixed β-cristobalite lattice, same topology throughout.

## Why this exists

test40's "middle" sigma (σ ≈ √(σ_min·σ_max), roughly a third of the nearest-
bead spacing) validation loss plateaus well above the zero-predictor
baseline, and did not clearly improve after adding batching, per-layer
conditioning or more capacity — on a dataset with 11 independent glass
structures and 62 crystal frames, not just one of each.

Crystal, sharing one fixed periodic lattice across all 62 frames, should be
the *easiest* possible case for this: local relative geometry alone should
in principle identify which regular lattice site a bead belongs to, unlike
glass's disordered, structure-specific networks. There is also no
shared-model phase-conditioning or crystal/glass data-imbalance to confound
the reading. If "middle" plateaus here too, that points away from data
diversity, phase-sharing or the specific hyperparameters tried so far, and
toward a more fundamental limit: recovering a bead's identity from local,
cutoff-limited, permutation-equivariant geometry alone, at that noise scale.

test42 is not meant to be extended into its own generator/force-field
project; once it has answered this question, treat it as disposable.

## Coarse-graining, architecture, training design

Identical to test40 (see its README for full detail): SiO4-tetrahedron CG
beads, the same `VectorBlock`/`Score` scalar-vector message passing (test40's
`PhaseScore` minus the phase embedding, which would carry no information
here), the same analytically exact periodic wrapped-Gaussian score target,
the same VE-SDE Euler-Maruyama generation.

## Data

`examples/crystal/`: 62 frames, extracted the same way as test40's crystal
data (`md/traj_0.lammpstrj`/`traj_1.lammpstrj`, β-cristobalite NPT MD,
`--index 1000::300 --split-gap 2`). `md/silica_beta_cristobalite.in` +
init data are bundled so a fresh `STAGE=lammps`/`full` run can regenerate or
extend it; see `licenses/ScoreMD-MIT.txt` for that source's provenance.

## Running

```bash
git clone git@github.com:haru2225/test42.git   # or wherever this is hosted
cd test42
singularity build test42.sif Singularity.def

# Bare qsub (STAGE=train, the default) trains on the bundled examples/crystal.
qsub -P <課題番号> run_test42.pbs

qsub -P <課題番号> -v RESUME=1,UPDATES=30000 run_test42.pbs
qsub -P <課題番号> -v STAGE=generate run_test42.pbs
```

## Local checks

```bash
python -m venv .venv
. .venv/bin/activate
pip install torch==2.6.0
pip install -r requirements.txt
python -m pytest -q
```
