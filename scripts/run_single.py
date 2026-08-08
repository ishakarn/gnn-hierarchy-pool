"""Run one experiment: one dataset / model / readout / seed combination."""

import os
import sys

# Allow `src` imports when called from any working directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from src.utils import set_seed, get_device, load_config, merge_configs
from src.data import load_dataset
from src.splits import get_or_create_split
from src.models import build_model, MODEL_REGISTRY
from src.readouts import get_readout   # noqa: imported for validation
from src.train import train
from src.evaluate import save_metrics


VALID_MODELS  = list(MODEL_REGISTRY.keys())
VALID_READOUTS = ['mean', 'sum', 'max', 'attention', 'selfattention']


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
    return p.parse_args()


def main():
    args = parse_args()

    cfg = load_config(args.config)
    cfg = merge_configs(cfg, {
        'readout': args.readout,
        'epochs':  args.epochs,
    })

    set_seed(args.seed)
    device = get_device()
    print(f"[run_single] device={device}  model={args.model}  readout={args.readout}  seed={args.seed}")

    # ---- Data ----
    dataset, meta = load_dataset(cfg['dataset'], root=cfg.get('data_root', './data'))
    print(f"Dataset: {cfg['dataset']}  |  {len(dataset)} molecules  |  {meta['num_tasks']} task(s)")

    # ---- Scaffold split ----
    train_idx, val_idx, test_idx = get_or_create_split(
        dataset,
        split_dir=cfg.get('split_dir', './data/splits'),
        dataset_name=cfg['dataset'],
        seed=args.seed,
        train_frac=cfg.get('train_frac', 0.8),
        val_frac=cfg.get('val_frac',   0.1),
        test_frac=cfg.get('test_frac', 0.1),
    )
    print(f"Split  train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

    # ---- Model ----
    num_node_features = dataset[0].x.shape[1]
    model = build_model(args.model, num_node_features, cfg, num_tasks=meta['num_tasks'])
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model}  |  readout={args.readout}  |  params={n_params:,}")

    # ---- Output paths ----
    run_name = f"{cfg['dataset']}_{args.model}_{args.readout}_seed{args.seed}"
    os.makedirs(args.output_dir, exist_ok=True)
    checkpoint_path = os.path.join(args.output_dir, f"{run_name}_best.pt")
    metrics_path    = os.path.join(args.output_dir, f"{run_name}_metrics.json")

    # ---- Train ----
    print(f"\nTraining {run_name} for up to {cfg['epochs']} epochs ...")
    results = train(
        model, dataset,
        train_idx, val_idx, test_idx,
        cfg, device, checkpoint_path,
    )

    # ---- Save ----
    results.pop('history', None)   # keep metrics file compact
    results.update({
        'run_name':  run_name,
        'dataset':   cfg['dataset'],
        'model':     args.model,
        'readout':   args.readout,
        'seed':      args.seed,
        'split':     'scaffold',
        'checkpoint': checkpoint_path,
        # config snapshot
        'hidden_dim': cfg['hidden_dim'],
        'num_layers': cfg['num_layers'],
        'dropout':    cfg['dropout'],
        'lr':         cfg['lr'],
        'batch_size': cfg['batch_size'],
    })
    save_metrics(results, metrics_path)

    print(
        f"\n{'='*60}\n"
        f"  {run_name}\n"
        f"  train_auc={results['train_auc']:.4f}  "
        f"val_auc={results['val_auc']:.4f}  "
        f"test_auc={results['test_auc']:.4f}  "
        f"(best epoch {results['best_epoch']})\n"
        f"  Saved → {metrics_path}\n"
        f"{'='*60}"
    )


if __name__ == '__main__':
    main()
