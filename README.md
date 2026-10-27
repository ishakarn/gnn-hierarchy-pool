# GNN Readout Benchmark — Molecular Classification

Study how GNN architecture and graph-level readout interact on MoleculeNet binary
classification benchmarks (BBBP, BACE; ClinTox and HIV are registry-ready).

## Setup (Unity HPCC)

```bash
module load conda/latest
conda activate graph-pooling
cd <your_working_directory>
```

All required packages (PyTorch, PyG, RDKit, scikit-learn) are already installed
in the `graph-pooling` environment.

## Quick smoke test

```bash
python scripts/run_single.py \
    --config configs/bbbp.yaml \
    --model gcn --readout mean --seed 0 --epochs 2
```

## Single run

```bash
python scripts/run_single.py \
    --config configs/bbbp.yaml \
    --model gcn --readout mean --seed 0

python scripts/run_single.py \
    --config configs/bace.yaml \
    --model gat --readout attention --seed 1
```

## Full grid (local — sequential)

```bash
python scripts/run_grid.py \
    --datasets BBBP BACE \
    --models gcn gin gat graphsage \
    --readouts mean sum max attention \
    --seeds 0 1 2
```

## Full grid on HPCC (recommended)

Each combination is one 1-GPU preemptable job — 96 jobs for the default grid.

```bash
# 1. Generate the combo list
python scripts/slurm/gen_combos.py

# 2. Preview
wc -l combos.txt          # should be 96
head -5 combos.txt

# 3. Submit job array
N=$(wc -l < combos.txt)
sbatch --array=1-$N scripts/slurm/run_grid_array.slurm

# 4. After all jobs finish, aggregate
# (or add --dependency=afterok:<ARRAY_JOB_ID>)
sbatch scripts/slurm/aggregate_results.slurm
```

Results land in `results/`:
- `<dataset>_<model>_<readout>_seed<N>_metrics.json` — one file per run
- `grid_results.csv` — all runs in one table
- `grid_summary.csv` — mean/std test AUC grouped by dataset/model/readout

## SPECTRA splits

Instead of the scaffold split, runs can use the SPECTRA Tanimoto splits from
[SPECTRA_evaluation](https://github.com/vynkdo/SPECTRA_evaluation), which
vary train–test similarity via a SPECTRA parameter (SP, 0.00–1.00).

**1. Clone SPECTRA_evaluation next to this repo** (the configs default to
`spectra_repo: ../SPECTRA_evaluation`, resolved relative to this repo's root):

```bash
cd <parent_of_this_repo>
git clone https://github.com/vynkdo/SPECTRA_evaluation
```

If it lives elsewhere, edit `spectra_repo` in `configs/<dataset>.yaml` or pass
`--spectra_repo /path/to/SPECTRA_evaluation`.

**2. Run with `--split spectra`:**

```bash
# One run: SP = 0.30, split replicate 0
python scripts/run_single.py --config configs/bbbp.yaml \
    --model gcn --readout mean --seed 0 \
    --split spectra --sp 0.30 --split_idx 0

# Grid over SP values and split replicates
python scripts/run_grid.py --datasets BBBP BACE --models gcn gin \
    --readouts mean attention --seeds 0 \
    --split spectra --sps 0.00 0.10 0.20 0.30 --split_idxs 0 1 2
```

Graphs are built from SPECTRA's curated CSVs (not PyG's MoleculeNet copy), so
split indices line up with molecules. Splits with fewer than `min_test_size`
(default 20) test molecules, or with single-class val/test sets, are skipped:
their metrics file has `"status": "skipped"` and a `skip_reason`. In practice
this keeps roughly SP ≤ 0.30 for BBBP and BACE.

## Project layout

```
gnn_readout_bio/
  configs/          YAML configs per dataset
  src/
    data.py         MoleculeNet loader + curated-CSV loader (SPECTRA)
    splits.py       Bemis-Murcko scaffold split + SPECTRA split loader
    readouts.py     mean / sum / max / attention readouts
    models.py       GCN / GIN / GAT / GraphSAGE
    train.py        training loop with early stopping
    evaluate.py     ROC-AUC computation
    utils.py        seeding, device, config I/O
  scripts/
    run_single.py   one experiment
    run_grid.py     sequential grid (local use)
    slurm/          Unity HPCC job scripts
  results/          output directory
  data/             downloaded datasets + split index files
```

## Models & Readouts

| Model      | Key layer  | Notes                                      |
|------------|------------|--------------------------------------------|
| GCN        | GCNConv    | 3 layers, ReLU, dropout                    |
| GIN        | GINConv    | MLP inside each layer, BatchNorm, train_eps|
| GAT        | GATConv    | 4 heads, head_dim = hidden_dim/heads, ELU  |
| GraphSAGE  | SAGEConv   | 3 layers, ReLU, dropout                    |

| Readout   | Implementation                                               |
|-----------|--------------------------------------------------------------|
| mean      | `global_mean_pool`                                           |
| sum       | `global_add_pool`                                            |
| max       | `global_max_pool`                                            |
| attention | learnable gate → per-graph softmax → weighted sum of nodes   |

## Adding a new dataset (ClinTox / HIV)

1. Add entry to `DATASET_REGISTRY` in `src/data.py`.
2. Copy a config YAML and set `dataset:` and `batch_size:`.
3. Add to `DATASET_TO_CONFIG` in `scripts/run_grid.py` and `scripts/slurm/run_grid_array.slurm`.

## Metric

ROC-AUC throughout (sklearn `roc_auc_score` on sigmoid probabilities).
NaN is returned with a warning if a split contains only one class.
