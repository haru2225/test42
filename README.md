# test42 — single-phase SiO2 crystal ablation of test40

The original single-phase ablation now uses a periodic Bloch-wave EGNN,
adapted from `mila-iqia/diffusion_for_multi_scale_molecular_dynamics`.
`egnn_vendor.py` contains E_GCL/EGNN; `test42.py:Score` supplies the periodic
embedding and score projection. Provenance and MIT notice are in the vendor
module and `licenses/diffusion_for_multi_scale_molecular_dynamics-MIT.txt`.

## Architecture and scope

Fractional coordinates are mapped to **interleaved** cos/sin pairs, processed
by EGNN, and projected through matching 2x2 Gamma blocks. Graph edges carry
minimum-image Cartesian distances. The projected fractional covector is
divided by cell lengths and trained against `-sigma * score` in Cartesian
coordinates, matching the existing sampler and wrapped-Gaussian target.
One SiO4 bead species is used; node input is log(sigma).

`--bloch-shells 1` uses the nonzero integer grid [-1,1]^3 modulo inversion:
13 vectors / 26 embedding dimensions. This differs from the source's first
complete length shell (3 vectors / 6 dimensions). The grid extent is saved
in checkpoints and can be set via `BLOCH_SHELLS` in the PBS script.

This construction **preserves invariance to uniform translation**: translation
rotates cos/sin pairs, and the final projection cancels that rotation. It
therefore does not identify an absolute lattice origin or automatically
resolve denoising ambiguity. It also does not enforce arbitrary continuous
rotations at a fixed orthorhombic cell. A lower middle-noise validation loss
or improved generation quality must be established by training experiments.

Prepared v1 (including the bundled crystal data) and v2 datasets remain usable. Checkpoints now use
`test42-crystal-bloch-egnn-v3`; old architecture/partial-port checkpoints
must be retrained in a fresh output directory. Training and generation
resume are supported for new checkpoints.

SiO4 mass-weighted mapping, periodic wrapped-Gaussian targets and VE-SDE
Euler-Maruyama sampling are retained. This generates structures, not an
energy model or equilibrium MD trajectory.

## Data

`examples/crystal/`: 62 frames, extracted the same way as test40's crystal
data (`md/traj_0.lammpstrj`/`traj_1.lammpstrj`, β-cristobalite NPT MD,
`--index 1000::300 --split-gap 2`). `md/silica_beta_cristobalite.in` +
init data are bundled so a fresh `STAGE=lammps`/`full` run can regenerate or
extend it; see `licenses/ScoreMD-MIT.txt` for that source's provenance.

## スパコンでの実行（PBS / GPU 1台）

既存の環境では `cd test42 && git pull --ff-only` で更新します。
新規の場合は以下で取得してください。

```bash
git clone git@github.com:haru2225/test42.git
cd test42
```

サイト指定の Singularity または Apptainer モジュールをロードしてください。
初回のみコンテナをビルドします（ビルド可能なLinux環境とネット接続が必要）。
既存の同じ `Singularity.def` で作ったイメージは再利用できます。
コードは実行時にマウントするため、EGNN更新だけなら再ビルド不要です。

```bash
singularity build test42.sif Singularity.def
# Apptainerの場合: apptainer build test42.sif Singularity.def
```

`run_test42.pbs` は既存の `sg8` キュー、GPU 1台、CPU 8、32 GB、20時間の設定です。
使用施設に合わせてキュー名・課題番号を変更してください。
以下の `PROJECT_ID` は実際の課題番号に置き換えます。

```bash
# GPUで実データのforward/backwardを確認し、CPUテストを実行
qsub -P PROJECT_ID -l walltime=00:10:00 -v STAGE=check run_test42.pbs

# 短い学習確認（完了してから次の生成確認へ）
qsub -P PROJECT_ID -l walltime=00:10:00 -v UPDATES=2,WIDTH=16,LAYERS=2,BATCH_SIZE=1,TRAIN_DIR=results/egnn_smoke,TIME_BUDGET_HOURS=0.1 run_test42.pbs
qsub -P PROJECT_ID -l walltime=00:10:00 -v STAGE=generate,STEPS=20,TRAIN_DIR=results/egnn_smoke,GENERATED_DIR=results/egnn_smoke_sample,TIME_BUDGET_HOURS=0.1 run_test42.pbs

# 本学習: 同梱62フレームを使用、results/egnn_train に保存
qsub -P PROJECT_ID run_test42.pbs

# 中断後の再開。同じ幅・層数・ノイズ設定などを使用すること
qsub -P PROJECT_ID -v RESUME=1,UPDATES=30000 run_test42.pbs

# 本学習の完了後、構造を生成。既定出力: results/egnn_seed1337
qsub -P PROJECT_ID -v STAGE=generate run_test42.pbs
```

ジョブは自動的には順番待ちしません。学習ログの完了を確認してから対応する
生成ジョブを投入してください。途中保存で終了した場合の終了コードは75です。
短時間確認で生成品質は評価できません。本学習の損失と構造指標を確認してください。
旧 `results/train` のチェックポイントは再利用できません。

主な環境変数: `TRAIN_DIR`, `GENERATED_DIR`, `DATASET_PATH`, `SIF_IMAGE`,
`WIDTH`, `LAYERS`, `BLOCH_SHELLS`, `BATCH_SIZE`, `UPDATES`, `SIGMA_MIN`,
`SIGMA_MAX`, `LEARNING_RATE`, `SEED`。外部ディレクトリを使う場合は
`EXTRA_BIND=/absolute/path:/absolute/path` も指定します。
walltimeを変更した場合は `WALLTIME_HOURS` または `TIME_BUDGET_HOURS` も合わせます。

生成結果は `final.extxyz`, `final.data`, `positions.npy` に保存されます。
評価はCPUで実行できます（施設の計算ノード利用方針に従ってください）。

```bash
singularity exec --bind "$PWD:$PWD" --pwd "$PWD" test42.sif \
  python test42.py evaluate --dataset examples/crystal \
  --sample results/egnn_seed1337/final.extxyz --output results/egnn_evaluation.json
```

## Local checks

```bash
python -m venv .venv
. .venv/bin/activate
pip install torch==2.6.0
pip install -r requirements.txt
python -m pytest -q
```

Tests cover periodicity, global translations, atom permutations, projection
units, empty graphs, backpropagation, a short fit to relative radial
displacements, and prepare/train/generate/evaluate including exact restart.
The former three-atom random-target short-fit test did not meet its loss
threshold with this EGNN; the radial fit checks implementation learnability,
not recovery of arbitrary denoising targets or crystal-generation quality.
