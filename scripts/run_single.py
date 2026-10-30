"""Run one experiment: one dataset / model / readout / seed combination (binary or regression)."""

import os
import sys

# Allow `src` imports when called from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from src.utils import set_seed, get_device, load_config, merge_configs, resolve_spectra_path
from src.data import load_dataset, load_curated_csv, feature_dims
from src.splits import get_or_create_split, load_spectra_split, SpectraSplitSkipped
from src.models import build_model, MODEL_REGISTRY
from src.readouts import READOUTS
from src.train import train
from src.evaluate import save_metrics
from src.tasks import build_task


VALID_MODELS  = list(MODEL_REGISTRY.keys()) + ['chemprop']
VALID_READOUTS = list(READOUTS)


def parse_args():
    p = argparse.ArgumentParser(
        description="Train one GNN+readout combination on a MoleculeNet dataset."
    )
    p.add_argument('--config',   required=True,
                   help='Path to YAML config (e.g. configs/bbbp.yaml)')
    p.add_argument('--model',    required=True, choices=VALID_MODELS)
    p.add_argument('--readout',  required=True, choices=VALID_READOUTS)
    p.add_argument('--seed',     type=int, default=0)
    p.add_argument('--epochs',   type=int, default=None,
                   help='Override epochs in config (useful for smoke tests)')
    p.add_argument('--output_dir', default='./results',
                   help='Directory to save checkpoints and metric files')

    # ---- Sweep overrides ----
    p.add_argument('--num_layers', type=int,   default=None,
                   help='Override num_layers in config (Chemprop: message-passing depth)')
    p.add_argument('--dropout',    type=float, default=None,
                   help='Override dropout in config')
    p.add_argument('--patience',   type=int,   default=None,
                   help='Override early-stopping patience in config')

    # ---- SPECTRA splits ----
    p.add_argument('--split', choices=['scaffold', 'spectra'], default=None,
                   help='Override `split` in config (default: scaffold)')
    p.add_argument('--sp', type=float, default=None,
                   help='SPECTRA parameter, e.g. 0.30 (required with --split spectra)')
    p.add_argument('--split_idx', type=int, default=0,
                   help='SPECTRA split replicate (0, 1, 2)')
    p.add_argument('--spectra_repo', default=None,
                   help='Override `spectra_repo` in config (path to SPECTRA_evaluation)')
    p.add_argument('--min_test_size', type=int, default=None,
                   help='Skip SPECTRA splits with fewer test molecules (default 20)')

    # ---- AttentiveFP ----
    p.add_argument('--head', choices=['mlp', 'linear'], default=None,
                   help='AttentiveFP classifier head (default mlp; linear = PyG original)')
    return p.parse_args()


def main():
    args = parse_args()

    cfg = load_config(args.config)
    cfg = merge_configs(cfg, {
        'readout': args.readout,
        'epochs':  args.epochs,
        'split':   args.split,
        'spectra_repo': args.spectra_repo,
        'min_test_size': args.min_test_size,
        'num_layers':    args.num_layers,
        'dropout':       args.dropout,
        'patience':      args.patience,
        'head':    args.head,
    })
    split_type = cfg.get('split', 'scaffold')
    if split_type == 'spectra' and args.sp is None:
        sys.exit('[run_single] split: spectra requires --sp')

    set_seed(args.seed)
    device = get_device()
    print(f"[run_single] device={device}  model={args.model}  readout={args.readout}  seed={args.seed}")

    # ---- Output paths ----
    # head only applies to AttentiveFP; tag non-default heads so runs don't collide
    head = cfg.get('head', 'mlp') if args.model == 'attentivefp' else None
    model_tag = f"{args.model}-{head}" if head == 'linear' else args.model
    if split_type == 'spectra':
        run_name = (f"{cfg['dataset']}_{model_tag}_{args.readout}"
                    f"_sp{args.sp:.2f}_split{args.split_idx}_seed{args.seed}")
    else:
        run_name = f"{cfg['dataset']}_{model_tag}_{args.readout}_seed{args.seed}"
    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_ext        = '.ckpt' if args.model == 'chemprop' else '.pt'
    checkpoint_path = os.path.join(args.output_dir, f"{run_name}_best{ckpt_ext}")
    metrics_path    = os.path.join(args.output_dir, f"{run_name}_metrics.json")

    run_info = {
        'run_name':  run_name,
        'dataset':   cfg['dataset'],
        'model':     args.model,
        'readout':   args.readout,
        'seed':      args.seed,
        'split':     split_type,
    }
    if split_type == 'spectra':
        run_info.update({'sp': round(args.sp, 2), 'split_idx': args.split_idx})
    if head is not None:
        run_info['head'] = head
    if args.readout == 'afp':
        run_info['num_timesteps'] = cfg.get('num_timesteps', 2)

    # ---- Data ----
    if split_type == 'spectra':
        # Graphs come from SPECTRA's curated CSV so split indices line up
        dataset, meta = load_curated_csv(resolve_spectra_path(cfg, 'curated_csv'),
                                         cfg['label_cols'])
    else:
        dataset, meta = load_dataset(cfg['dataset'], root=cfg.get('data_root', './data'))
    cfg['task_type'] = run_info['task_type'] = meta['task_type']
    source = ' (curated CSV)' if split_type == 'spectra' else ''
    print(f"Dataset: {cfg['dataset']}{source}  |  {len(dataset)} molecules  |  "
          f"{meta['num_tasks']} task(s)  |  {meta['task_type']}")

    # ---- Split ----
    if split_type == 'spectra':
        labels = [g.y.view(-1).tolist() for g in dataset]
        try:
            train_idx, val_idx, test_idx = load_spectra_split(
                resolve_spectra_path(cfg, 'spectra_split_dir'),
                sp=args.sp,
                split_idx=args.split_idx,
                labels=labels,
                min_test_size=cfg.get('min_test_size', 20),
            )
        except SpectraSplitSkipped as e:
            # Record the skip so aggregation can see it, then exit cleanly
            run_info.update({'status': 'skipped', 'skip_reason': str(e)})
            save_metrics(run_info, metrics_path)
            print(f"[run_single] SKIPPED {run_name}: {e}")
            return
    else:
        train_idx, val_idx, test_idx = get_or_create_split(
            dataset,
            split_dir=cfg.get('split_dir', './data/splits'),
            dataset_name=cfg['dataset'],
            seed=args.seed,
            train_frac=cfg.get('train_frac', 0.8),
            val_frac=cfg.get('val_frac',   0.1),
            test_frac=cfg.get('test_frac', 0.1),
            task_type=cfg['task_type'],
        )
    print(f"Split ({split_type})  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    # ---- Model ----
    if args.model == 'chemprop' and cfg['task_type'] != 'binary':
        raise SystemExit(f"Chemprop baseline supports binary datasets only; {cfg['dataset']} is {cfg['task_type']}")
    if args.model == 'chemprop':
        from src.chemprop_model import build_chemprop, train_chemprop
        model = build_chemprop(cfg, cfg['readout'], num_tasks=meta['num_tasks'])
    else:
        num_node_features, num_edge_features = feature_dims(dataset)
        model = build_model(args.model, num_node_features, cfg, num_tasks=meta['num_tasks'],
                            num_edge_features=num_edge_features)
        model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}  |  readout={args.readout}  |  params={n_params:,}")

    # ---- Train ----
    print(f"\nTraining {run_name} for up to {cfg['epochs']} epochs ...")
    train_fn = train_chemprop if args.model == 'chemprop' else train
    results = train_fn(
        model, dataset,
        train_idx, val_idx, test_idx,
        cfg, device, checkpoint_path,
    )

    # ---- Save ----
    results.pop('history', None)   # keep metrics file compact
    results.update(run_info)
    results.update({
        'status':    'ok',
        'n_train':   len(train_idx),
        'n_val':     len(val_idx),
        'n_test':    len(test_idx),
        'checkpoint': checkpoint_path,
        # config snapshot
        'hidden_dim': cfg['hidden_dim'],
        'num_layers': cfg['num_layers'],
        'dropout':    cfg['dropout'],
        'lr':         cfg['lr'],
        'batch_size': cfg['batch_size'],
        'epochs':     cfg['epochs'],
        'patience':   cfg.get('patience', 20),
    })
    save_metrics(results, metrics_path)

    metric = build_task(cfg).primary
    print(
        f"\n{'='*60}\n"
        f"  {run_name}\n"
        f"  train_{metric}={results[f'train_{metric}']:.4f}  "
        f"val_{metric}={results[f'val_{metric}']:.4f}  "
        f"test_{metric}={results[f'test_{metric}']:.4f}  "
        f"(best epoch {results['best_epoch']})\n"
        f"  Saved → {metrics_path}\n"
        f"{'='*60}"
    )


if __name__ == '__main__':
    main()
