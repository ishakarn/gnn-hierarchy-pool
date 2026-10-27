# GNN Readout Benchmark — Molecular Property Prediction

Study how GNN architecture and graph-level readout interact on MoleculeNet
benchmarks — binary classification (BBBP, BACE, ClinTox; HIV is registry-ready)
and regression (ESOL, FreeSolv, Lipophilicity) — and which modes of information
each model relies on (ModeTrace). The hyperparameter sweep is specified in
[docs/hparam_sweep_protocol.ipynb](docs/hparam_sweep_protocol.ipynb).

## Setup (Unity HPCC)

```bash
module load conda/latest
conda activate graph-pooling
```

All required packages (PyTorch, PyG, RDKit, scikit-learn) are installed in the
`graph-pooling` environment. Run training through SLURM, not on a login node.

## Running

```bash
# one run (add --epochs 2 for a smoke test)
python scripts/run_single.py --config configs/bbbp.yaml --model gin --readout mean --seed 0

# a grid, sequentially
python scripts/run_grid.py --datasets BBBP ESOL --models gin gine \
    --readouts mean sum max attention --seeds 0 1 2
```

On the cluster (submit from the repo root; logs go to `logs/`):

```bash
mkdir -p logs
sbatch scripts/slurm/smoke_test.slurm         # 2 epochs on every dataset + ModeTrace
bash scripts/slurm/submit_base.sh             # one GPU job per dataset -> results/base/<DATASET>/
bash scripts/slurm/submit_base.sh ESOL LIPO   # or a subset
```

`submit_base.sh` runs GIN/GINE × all readouts × seeds 0–2 (edit
`scripts/slurm/run_base.slurm` to change the grid) and queues
`aggregate_results.py` after the jobs. Jobs are requeue-safe: finished runs are
skipped. For one job per run instead, use `scripts/slurm/gen_combos.py` +
`run_grid_array.slurm` (default grid: 200 jobs; cap concurrency, e.g. `--array=1-200%20`).

Each run writes `<DATASET>_<model>_<readout>_seed<N>_metrics.json`;
`aggregate_results.py` collects them into `grid_results.csv` and `grid_summary.csv`.

## Analysis and figures

```bash
python scripts/aggregate_results.py --results_dir results/base
python scripts/analyze_results.py   --results_dir results/base --models gin gine   # stats + plots -> results/base/analysis/
python scripts/plot_results.py      --results_dir results/base --models gin gine --datasets BBBP ESOL
python scripts/plot_datasets.py     --results_dir results/base --modetrace_dir results/modetrace
python scripts/make_dataset_card.py --datasets BACE ESOL
```

Every script scores each dataset on its own metric (ROC-AUC for binary, RMSE for
regression; `--regression_metric mae|r2` to switch). `plot_datasets.py` writes
all datasets side by side to `results/figures/combined/` and one set per dataset
to `results/figures/individual/<DATASET>/`:

| Figure | Shows |
|--------|-------|
| `benchmark_bars` | test metric per readout, bars per model, ± 95% CI over seeds |
| `benchmark_ranking` | model·readout × dataset heatmap, coloured by within-dataset score, with mean rank |
| `modetrace_matched` / `modetrace_ood` | test metric vs p, one panel per perturbation |
| `modetrace_drop` | relative drop at p = 1 (red = worse, blue = better; n/a where a model never reads edge features) |

## Project layout

```
configs/            YAML config per dataset
docs/               protocols (hyperparameter sweep)
src/
  data.py           MoleculeNet loader + dataset registry (task_type per dataset)
  splits.py         seeded scaffold split (cached to data/splits/) + k-fold CV
  tasks.py          binary vs regression: loss, prediction transform, metrics
  models.py         GIN / GINE (and legacy GCN / GAT / GraphSAGE / GCN-P)
  readouts.py       mean / sum / max / attention / selfattention
  train.py          training loop with early stopping
  evaluate.py       predictions and task metrics
  chemprop_model.py Chemprop D-MPNN baseline (binary only)
  results.py        shared helpers for the analysis scripts
  modetrace/        ModeTrace (below)
scripts/            run_single, run_grid, aggregate/analyze/plot, modetrace/, slurm/
```

## Models and readouts

| Model | Key layer | Notes |
|-------|-----------|-------|
| GIN   | GINConv   | MLP inside each layer, BatchNorm, train_eps |
| GINE  | GINEConv  | GIN + edge features (the only edge-aware model) |

GCN, GAT, GraphSAGE and GCN-P remain in `src/models.py` but are no longer part of the study.

| Readout | Implementation |
|---------|----------------|
| mean / sum / max | `global_mean_pool` / `global_add_pool` / `global_max_pool` |
| attention | learnable gate → per-graph softmax → weighted sum of nodes |
| selfattention | single-head self-attention read at the first node; leaks the label on BBBP, do not use |

## Task types

Each dataset's `task_type` in `DATASET_REGISTRY` (`src/data.py`) selects a task from `src/tasks.py`:

| task_type  | Loss | Metrics (result keys) | Early stopping / selection |
|------------|------|-----------------------|----------------------------|
| binary     | BCE  | ROC-AUC (`*_auc`)     | max val AUC |
| regression | MSE on z-scored targets | RMSE, MAE, R² (`*_rmse`, `*_mae`, `*_r2`), original units | min val RMSE |

Scaffold splits are class-stratified for binary datasets only. Multi-task
datasets (ClinTox) report the mean of per-task metrics and are stratified on
the joint label pattern.

To add a dataset: add it to `DATASET_REGISTRY` with its `task_type`, copy a
config YAML (set `dataset:` and `batch_size:`), and add it to
`DATASET_TO_CONFIG` in `scripts/run_grid.py`.

## ModeTrace

`src/modetrace/` is ModeTrace, ported from
[modes-of-information](https://github.com/ishakarn/modes-of-information). It
measures how much a model relies on node features, edge features and topology
by shuffling features at rate `p` and measuring the drop in performance, using
this repo's models, training step and metrics.

| Perturbation | Shuffles | Leaves intact |
|--------------|----------|---------------|
| WNFS / GNFS  | node features within each graph / across the split | topology, edge features |
| WEFS / GEFS  | edge features within each graph / across the split | topology, node features |
| JGFS         | node + edge features across the split | topology only |

Protocols (scaffold 10-fold CV on the MoleculeNet datasets):

- *Matched* (`run_matched.py`): perturb train/val/test at rate `p`, train for a
  fixed number of epochs, log every metric every epoch; `summarize.py` picks the best-val epoch.
- *OOD* (`run_ood.py`): train on clean data, evaluate the best clean-val
  checkpoint on each perturbed test set.

```bash
bash scripts/slurm/submit_modetrace.sh        # GIN/GINE on every dataset: OOD + matched
sbatch scripts/slurm/run_modetrace.slurm ood --datasets ESOL --models gin gine \
    --perturbations WNFS GNFS WEFS GEFS JGFS   # a single job
python scripts/modetrace/summarize.py         # -> results/modetrace/{matched,ood}_summary.csv
```

Hyperparameters default to the ModeTrace values in `src/modetrace/__init__.py`
(2 layers, 100 epochs, BCE `pos_weight`); `--config configs/bbbp.yaml` uses the
benchmark's. Runs are seeded per fold and resume where they left off.
`degradation` in the summaries is signed so that positive = worse.

Differences from the original repo: molecular datasets only; ModeTrace runs this
repo's models (MLP head, configurable readout); molecular datasets keep the first task only, as before;
weights are seeded per fold; a GNFS bug that misassigned features after a
zero-node graph is fixed. Perturbations and fold assignments are otherwise identical.
