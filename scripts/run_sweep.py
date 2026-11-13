"""Resumable depth/dropout sweep with persistent data for scaffold or SPECTRA."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
from itertools import product

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_grid import DATASET_TO_CONFIG, _run_name, _run_single
from src.run_identity import model_tag

LAYERS = [2, 3, 4, 5, 6]
DROPOUTS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
SEEDS = [0, 1, 2, 3, 4]


def run_dir(output_dir, dataset, model, readout, layers, dropout):
    return os.path.join(output_dir, dataset, model, readout, f"L{layers}_do{dropout}")


def finished(path, expected):
    if not path.exists():
        return False
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (value.get("status") == "ok"
            and all(value.get(k) == v for k, v in expected.items())
            and math.isfinite(value.get("val_auc" if value.get("task_type") == "binary" else "val_rmse", float("nan"))))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', required=True, choices=list(DATASET_TO_CONFIG))
    p.add_argument('--model', required=True,
                   choices=['gin', 'gine', 'chemprop', 'attentivefp', 'jknet'])
    p.add_argument('--config', help='Dataset config override, including audited scaffold splits')
    p.add_argument('--readout', required=True,
                   choices=['max', 'mean', 'sum', 'attention', 'multihead'])
    p.add_argument('--layers', nargs='+', type=int, default=LAYERS)
    p.add_argument('--dropouts', nargs='+', type=float, default=DROPOUTS)
    p.add_argument('--seeds', nargs='+', type=int, default=SEEDS)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--output_dir', default='./results/sweep')
    p.add_argument('--jk_mode', choices=['cat', 'max', 'lstm'], default=None)
    p.add_argument('--split', choices=['scaffold', 'spectra'], default=None)
    p.add_argument('--sps', nargs='+', type=float)
    p.add_argument('--split_idxs', nargs='+', type=int, default=[0, 1, 2])
    p.add_argument('--graph_cache_dir', default='./data/cache/graphs')
    p.add_argument('--fresh_process', action='store_true',
                   help='Use subprocesses instead of retaining Python/CUDA and graphs')
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    if args.model == 'chemprop' and args.readout == 'multihead':
        p.error('Chemprop does not support multihead')
    if any(n < 1 for n in args.layers) or any(not 0 <= d < 1 for d in args.dropouts):
        p.error('Layers must be positive; dropout must be in [0, 1)')
    cfg_path = args.config or DATASET_TO_CONFIG[args.dataset]
    cfg = yaml.safe_load(Path(cfg_path).read_text())
    if cfg['dataset'].upper() != args.dataset:
        p.error('Config dataset does not match --dataset')
    split = args.split or cfg.get('split', 'scaffold')
    mode = args.jk_mode or cfg.get('jk_mode', 'cat')
    tag = model_tag(args.model, mode)
    if split == 'spectra':
        if not args.sps:
            p.error('SPECTRA sweeps require explicit --sps')
        if any(not 0 <= sp <= 1 or abs(sp * 20 - round(sp * 20)) > 1e-8 for sp in args.sps):
            p.error('SPECTRA values must be 0.00-1.00 in steps of 0.05')
        if any(k not in (0, 1, 2) for k in args.split_idxs):
            p.error('SPECTRA replicates must be 0, 1 or 2')
        # Audit once before fitting: no repeated featurization for unusable splits.
        import pandas as pd
        from src.splits import load_spectra_split, SpectraSplitSkipped
        from src.utils import resolve_spectra_path
        labels = pd.read_csv(resolve_spectra_path(cfg, 'curated_csv'))[cfg['label_cols']].values
        splits = []
        for sp, rep in product(args.sps, args.split_idxs):
            try:
                load_spectra_split(resolve_spectra_path(cfg, 'spectra_split_dir'), sp, rep,
                                   labels=labels, min_test_size=cfg.get('min_test_size', 20))
                splits.append((sp, rep))
            except SpectraSplitSkipped as error:
                print(f"[excluded] SP={sp:.2f} split={rep}: {error}")
    else:
        if args.sps:
            p.error('--sps requires --split spectra')
        splits = [(None, None)]
    configs = list(product(splits, args.layers, args.dropouts, args.seeds))
    print(f"Sweep {args.dataset}/{tag}/{args.readout}: {len(configs)} eligible fits", flush=True)
    failed, skipped = [], 0
    for (sp, rep), layers, dropout, seed in configs:
        base = Path(args.output_dir)
        if sp is not None:
            base = base / f"SP_{sp:.2f}" / f"split{rep}"
        folder = run_dir(str(base), args.dataset, tag, args.readout, layers, dropout)
        name = _run_name(args.dataset, args.model, args.readout, seed, sp, rep, jk_mode=mode)
        path = Path(folder) / f"{name}_metrics.json"
        expected = dict(dataset=args.dataset, model=tag, readout=args.readout, seed=seed,
                        num_layers=layers, dropout=dropout, epochs=args.epochs,
                        patience=args.patience, split=split)
        if args.model == 'jknet':
            expected['jk_mode'] = mode
        if sp is not None:
            expected.update(sp=round(sp, 2), split_idx=rep)
        if finished(path, expected):
            skipped += 1
            continue
        extra = ['--num_layers', str(layers), '--dropout', str(dropout),
                 '--epochs', str(args.epochs), '--patience', str(args.patience),
                 '--split', split]
        if args.dry_run:
            print(f"{name} L{layers} do{dropout} -> {folder}")
            continue
        ok = _run_single(args.dataset, args.model, args.readout, seed, folder, extra,
                         sp=sp, split_idx=rep, config=cfg_path,
                         in_process=not args.fresh_process, jk_mode=mode,
                         graph_cache_dir=args.graph_cache_dir)
        if not ok or not finished(path, expected):
            failed.append(str(path))
            continue
        for suffix in ('.pt', '.ckpt'):
            (Path(folder) / f"{name}_best{suffix}").unlink(missing_ok=True)
    print(f"Finished: {len(configs)-skipped-len(failed)} new, {skipped} resumed, {len(failed)} failed")
    if failed:
        for path in failed:
            print("[FAILED]", path)
        raise SystemExit(1)


if __name__ == '__main__':
    main()
