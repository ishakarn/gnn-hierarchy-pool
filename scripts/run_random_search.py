"""Random hyperparameter search over a single (dataset, model, readout).

Samples `n_trials` hyperparameter configs from a search-space YAML, runs each
across a fixed list of seeds via run_single.py (as a subprocess, same pattern
as run_grid.py), and ranks trials by mean validation AUC.

Search-space YAML format (see configs/bbbp_search_space.yaml for an example):

    dataset: BBBP
    model: gcn
    readout: mean
    n_trials: 30
    random_search_seed: 42     # controls which hyperparams get sampled
    seeds: [0, 1, 2]            # every trial is run across these seeds

    search_space:
      hidden_dim:   {type: choice,      values: [64, 128, 256]}
      num_layers:   {type: int_uniform, low: 2, high: 5}       # inclusive
      dropout:      {type: uniform,     low: 0.1, high: 0.5}
      lr:           {type: loguniform,  low: 1.0e-4, high: 1.0e-2}
      weight_decay: {type: loguniform,  low: 1.0e-6, high: 1.0e-3}
      batch_size:   {type: choice,      values: [16, 32, 64]}

Any hyperparameter not listed in `search_space` falls back to the base
dataset config (configs/bbbp.yaml / configs/bace.yaml).

Usage:
    python run_random_search.py --search_config configs/bbbp_search_space.yaml
"""
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import json
import math
import os
import random
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import yaml

from run_grid import DATASET_TO_CONFIG
from src.utils import load_config, merge_configs


# --------------------------------------------------------------------------
# Sampling
# --------------------------------------------------------------------------

def _sample_value(name, spec, rng):
    """Draw one value for a single search_space entry."""
    kind = spec.get('type')
    if kind == 'choice':
        return rng.choice(spec['values'])
    if kind == 'uniform':
        return round(rng.uniform(spec['low'], spec['high']), 6)
    if kind == 'loguniform':
        low, high = spec['low'], spec['high']
        if low <= 0 or high <= 0:
            raise ValueError(f"search_space['{name}']: loguniform bounds must be > 0")
        sampled_log = rng.uniform(math.log(low), math.log(high))
        return round(math.exp(sampled_log), 8)
    if kind == 'int_uniform':
        # random.Random.randint is inclusive on both ends
        return rng.randint(int(spec['low']), int(spec['high']))
    raise ValueError(f"search_space['{name}']: unknown type '{kind}'")


def sample_hyperparams(search_space, rng):
    return {name: _sample_value(name, spec, rng) for name, spec in search_space.items()}


# --------------------------------------------------------------------------
# GAT / num_heads validation
# --------------------------------------------------------------------------

def _validate_gat_compatibility(search_space, base_cfg, model):
    """GATModel asserts hidden_dim % num_heads == 0. If hidden_dim is being
    searched and the fixed model is GAT, validate every candidate value up
    front rather than letting a random trial crash partway through the run.
    """
    if model.lower() != 'gat' or 'hidden_dim' not in search_space:
        return

    spec = search_space['hidden_dim']
    num_heads = base_cfg.get('num_heads', 4)

    if spec.get('type') != 'choice':
        raise ValueError(
            "GAT requires hidden_dim % num_heads == 0. To validate this up front, "
            "'hidden_dim' in search_space must use type: choice with explicit "
            f"values (got type: '{spec.get('type')}')."
        )

    bad = [v for v in spec['values'] if v % num_heads != 0]
    if bad:
        raise ValueError(
            f"GAT requires hidden_dim divisible by num_heads={num_heads} "
            f"(set via 'num_heads' in the base config). Invalid hidden_dim "
            f"choices in search_space: {bad}"
        )


# --------------------------------------------------------------------------
# Running one trial (all seeds)
# --------------------------------------------------------------------------

def _run_single_seed(config_path, model, readout, seed, output_dir, extra_args):
    """Invoke run_single.py for one (trial config, seed). Returns True on success."""
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'run_single.py'),
        '--config',     config_path,
        '--model',      model,
        '--readout',    readout,
        '--seed',       str(seed),
        '--output_dir', output_dir,
    ] + extra_args
    result = subprocess.run(cmd, capture_output=False, text=True)
    return result.returncode == 0


def run_trial(trial_id, dataset, model, readout, seeds, trial_cfg, base_output_dir, extra_args):
    """Run one sampled hyperparameter config across all seeds.

    Each trial gets its own output subdirectory. This is necessary because
    run_single.py names its metrics/checkpoint files
    "{dataset}_{model}_{readout}_seed{seed}" — identical across trials here,
    since only the hyperparameters (not dataset/model/readout/seed) vary.
    Separating trials by directory prevents them from overwriting each other.
    """
    trial_dir = os.path.join(base_output_dir, trial_id)
    os.makedirs(trial_dir, exist_ok=True)

    config_path = os.path.join(trial_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        yaml.safe_dump(trial_cfg, f, sort_keys=False)

    seed_results = []
    for seed in seeds:
        run_name = f"{dataset.upper()}_{model}_{readout}_seed{seed}"
        metrics_path = os.path.join(trial_dir, f"{run_name}_metrics.json")

        if os.path.exists(metrics_path):
            print(f"  [skip] {trial_id}/seed{seed} — metrics already exist")
            with open(metrics_path) as f:
                seed_results.append(json.load(f))
            continue

        print(f"  [run]  {trial_id}/seed{seed}")
        ok = _run_single_seed(config_path, model, readout, seed, trial_dir, extra_args)
        if not ok:
            print(f"  [ERROR] {trial_id}/seed{seed} — training failed, skipping this seed")
            continue

        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                seed_results.append(json.load(f))
        else:
            print(f"  [WARN] {trial_id}/seed{seed} — expected metrics file missing: {metrics_path}")

    return seed_results


# --------------------------------------------------------------------------
# Summarization
# --------------------------------------------------------------------------

def summarize_trial(trial_id, sampled_hyperparams, seed_results):
    row = {'trial': trial_id, 'n_seeds_completed': len(seed_results)}
    row.update(sampled_hyperparams)

    df = pd.DataFrame(seed_results)
    for metric in ('train_auc', 'val_auc', 'test_auc'):
        if metric in df.columns and len(df):
            vals = pd.to_numeric(df[metric], errors='coerce')
            row[f'{metric}_mean'] = round(vals.mean(), 4)
            row[f'{metric}_std'] = round(vals.std(), 4) if len(vals) > 1 else 0.0
        else:
            row[f'{metric}_mean'] = None
            row[f'{metric}_std'] = None
    return row

def _to_native(value):
    """Convert a numpy scalar (as returned by indexing into a pandas
    DataFrame row, e.g. numpy.int64/numpy.float64) to a plain Python type.
    yaml.safe_dump can't represent numpy scalars, only built-in types.
    """
    if hasattr(value, 'item'):
        return value.item()
    return value

# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Random hyperparameter search over a single (dataset, model, readout)."
    )
    p.add_argument('--search_config', required=True,
                    help='YAML file describing the search space.')
    p.add_argument('--output_dir', default='./results/random_search',
                    help='Directory to store per-trial configs, checkpoints, and metrics.')
    p.add_argument('--epochs', type=int, default=None,
                    help='Override epochs in every trial config (e.g. for a smoke test).')
    args = p.parse_args()

    search_cfg = load_config(args.search_config)

    dataset = search_cfg['dataset'].upper()
    model = search_cfg['model']
    readout = search_cfg['readout']
    n_trials = search_cfg['n_trials']
    seeds = search_cfg.get('seeds', [0, 1, 2])
    rs_seed = search_cfg.get('random_search_seed', 0)
    search_space = search_cfg.get('search_space', {})

    if dataset not in DATASET_TO_CONFIG:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {list(DATASET_TO_CONFIG.keys())}")
    if not search_space:
        raise ValueError("search_config must define a non-empty 'search_space'.")

    base_config_path = DATASET_TO_CONFIG[dataset]
    base_cfg = load_config(base_config_path)

    _validate_gat_compatibility(search_space, base_cfg, model)

    extra_args = ['--epochs', str(args.epochs)] if args.epochs else []

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Random search: {n_trials} trial(s) x {len(seeds)} seed(s) "
          f"on {dataset}/{model}/{readout}")
    print(f"Search space: {list(search_space.keys())}\n")

    rng = random.Random(rs_seed)
    trial_rows = []

    for i in range(n_trials):
        trial_id = f"trial{i:03d}"
        sampled = sample_hyperparams(search_space, rng)
        trial_cfg = merge_configs(base_cfg, sampled)

        print(f"[{i + 1}/{n_trials}] {trial_id}  {sampled}")
        seed_results = run_trial(
            trial_id, dataset, model, readout, seeds, trial_cfg, args.output_dir, extra_args
        )

        if not seed_results:
            print(f"  [ERROR] {trial_id} — no successful seeds, excluding from summary")
            continue

        trial_rows.append(summarize_trial(trial_id, sampled, seed_results))

    if not trial_rows:
        print("\nNo trials completed successfully — nothing to summarize.")
        return

    results_df = pd.DataFrame(trial_rows).sort_values('val_auc_mean', ascending=False)
    trials_csv = os.path.join(args.output_dir, 'random_search_trials.csv')
    results_df.to_csv(trials_csv, index=False)

    print(f"\nAll trials  -> {trials_csv}\n")
    print(results_df.to_string(index=False))

    # ---- Best config ----
    best_row = results_df.iloc[0]
    best_trial_id = best_row['trial']
    best_hparams = {k: _to_native(best_row[k]) for k in search_space.keys()}
    best_cfg = merge_configs(base_cfg, best_hparams)

    best_config_path = os.path.join(args.output_dir, 'best_config.yaml')
    with open(best_config_path, 'w') as f:
        yaml.safe_dump(best_cfg, f, sort_keys=False)

    print(
        f"\nBest trial: {best_trial_id}  "
        f"(val_auc = {best_row['val_auc_mean']:.4f} +/- {best_row['val_auc_std']:.4f}, "
        f"test_auc = {best_row['test_auc_mean']:.4f})"
    )
    print(f"Best config saved -> {best_config_path}")
    print(
        f"Reproduce directly with:\n"
        f"  python run_single.py --config {best_config_path} "
        f"--model {model} --readout {readout} --seed <seed>"
    )


if __name__ == '__main__':
    main()
