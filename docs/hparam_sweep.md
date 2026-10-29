# Running the hyperparameter sweep

How to run the sweep described in `docs/hparam_sweep_protocol.ipynb`
(layers 2–6 × dropout 0.0–0.5 × seeds 0–4 = 150 runs per dataset / model / readout,
300 epochs, patience 50).

| File | What it does |
|---|---|
| `scripts/run_sweep.py` | Runs the 150 runs for one (dataset, model, readout), one after another. Skips runs that already have a metrics file. |
| `scripts/slurm/run_sweep.slurm` | Runs `run_sweep.py` as one GPU job. |
| `scripts/slurm/submit_sweep.sh` | Submits one job per (dataset, readout) for a model. |
| `scripts/aggregate_sweep.py` | Collects all results and picks the best config per (dataset, model, readout). |
| `scripts/run_single.py` | One training run. The sweep calls it with `--num_layers`, `--dropout`, `--patience`. |

## 1. Setup (once, on Unity)

```bash
module load python/3.12.3
source ~/venvs/sage/bin/activate      # or your own venv with requirements.txt
cd ~/gnn-hierarchy-pool
mkdir -p logs
```

Every model must use the same split files. Copy the shared ones (seeds 0–2), then create seeds 3–4
**before** submitting jobs, so parallel jobs don't try to write the same file:

```bash
cp -n /work/pi_annagreen_umass_edu/Isha/gnn-hierarchy-pool/data/splits/*.json data/splits/
python -c "from src.data import load_dataset; from src.splits import get_or_create_split; [get_or_create_split(load_dataset(n)[0], './data/splits', n, s) for n in ['BBBP', 'BACE', 'CLINTOX', 'ESOL', 'FREESOLV', 'LIPO'] for s in [3, 4]]"
ls data/splits | wc -l                # expect 30
```

## 2. Submit the sweep

```bash
bash scripts/slurm/submit_sweep.sh chemprop --dry-run     # print the sbatch commands only
bash scripts/slurm/submit_sweep.sh chemprop               # submit
```

- Models: `chemprop` (BBBP, BACE, CLINTOX × max, mean, sum, attention = 12 jobs),
  `gin` / `gine` (all 6 datasets × max, mean, sum, attention, multihead).
- Only some datasets: `bash scripts/slurm/submit_sweep.sh gine BBBP ESOL`
- Only some readouts: `SWEEP_READOUTS="max mean" bash scripts/slurm/submit_sweep.sh gine`
- `multihead` is not implemented yet, so for GIN/GINE use
  `SWEEP_READOUTS="max mean sum attention"` until it is.

Each job uses `gpu-preempt`, 1 GPU, 8 h, and `--requeue`. A Chemprop run takes about 1.5 min, so one job is about 4 h.

## 3. Check on jobs

```bash
squeue --me                               # PD = waiting, R = running
tail -f logs/sweep_<model>_<dataset>_<readout>_<JOBID>.out   # live output of one job
ls results/sweep/BBBP/chemprop/mean/*/*_metrics.json | wc -l   # finished runs (150 = done)
```

**Resuming:** if a job is preempted, times out or fails, just submit again with the same command.
Finished runs are skipped, so only the missing ones run.

## 4. Collect results

```bash
python scripts/aggregate_sweep.py
```

This writes to `results/sweep/`:
- `sweep_results.csv`: one row per run.
- `sweep_selected.csv`: the best config per (dataset, model, readout). It is chosen on mean val over seeds,
  needs at least 4 valid seeds, and ties within 0.001 go to fewer layers, then lower dropout. Test is
  reported as mean ± std and 95% CI.

It also prints groups with fewer than 150 runs, the share of runs that hit the epoch cap,
and configs that land on the edge of the grid.

## Run without Slurm

One (dataset, model, readout), on an interactive node or locally:

```bash
python scripts/run_sweep.py --dataset BBBP --model chemprop --readout mean
```

A smaller grid (e.g. a quick test): `--layers 2 3 --dropouts 0.0 --seeds 0 --epochs 2 --output_dir results/test_sweep`

## Output layout

```
results/sweep/<DATASET>/<model>/<readout>/L<layers>_do<dropout>/<DATASET>_<model>_<readout>_seed<N>_metrics.json
```

Checkpoints are deleted after each run to save space. Only the metrics JSON is kept.

## Notes

- Chemprop supports binary datasets only (BBBP, BACE, CLINTOX).
- ClinTox Chemprop AUC is currently pooled over its 2 tasks, not averaged per task (to fix).
- Don't use the `selfattention` readout: it leaks the label on BBBP.
- Chemprop shares weights across message-passing steps, so its parameter count doesn't change with layers.
