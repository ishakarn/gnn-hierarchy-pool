"""
Generate a dataset card for a MoleculeNet classification dataset.

Computes dataset statistics, molecular descriptors, graph structure metrics,
and split diagnostics across all seeds.  Saves a JSON card and diagnostic plots.

Binary datasets are broken down by class label. Regression datasets are broken
down by target tercile (low / mid / high), and the card reports target
statistics instead of class balance. Multi-task datasets use the first task
for the breakdowns (per-task positive ratios are listed in the card).

Usage:
    python scripts/make_dataset_card.py --datasets BACE
    python scripts/make_dataset_card.py --datasets BACE BBBP ESOL --seeds 0 1 2 3 4 5 6 7 8 9
"""

import os
import sys
import json
import argparse
import warnings
from collections import defaultdict, Counter

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ── helpers ───────────────────────────────────────────────────────────────────

DESCRIPTORS = [
    'MolWt', 'HeavyAtomCount', 'NumHDonors', 'NumHAcceptors',
    'NumRotatableBonds', 'RingCount', 'TPSA', 'MolLogP',
    'FractionCSP3', 'NumAromaticRings',
]

READOUT_COLORS = ['#4C72B0', '#55A868', '#C44E52', '#8172B2',
                  '#CCB974', '#64B5CD', '#E377C2', '#7F7F7F']


def _target(data):
    """First-task target as a float (None if missing)."""
    v = data.y.view(-1)[0].item() if data.y is not None else float('nan')
    return None if v != v else float(v)


def _label(data):
    """First-task class label (None if missing)."""
    t = _target(data)
    return None if t is None else round(t)


TERCILES = ['low', 'mid', 'high']


def _terciles(targets):
    """Target tercile ('low' / 'mid' / 'high') per molecule (None if missing)."""
    vals = np.array([v for v in targets if v is not None])
    q1, q2 = np.quantile(vals, [1 / 3, 2 / 3])
    return [None if v is None else ('low' if v <= q1 else 'mid' if v <= q2 else 'high')
            for v in targets]


def _group_order(keys):
    """Class labels sorted numerically; terciles in low -> high order."""
    keys = list(keys)
    if keys and isinstance(keys[0], str):
        return [k for k in TERCILES if k in keys]
    return sorted(keys)


def _group_name(key):
    return f'{key} target' if isinstance(key, str) else f'Label {int(key)}'


def _group_key(key):
    return key if isinstance(key, str) else str(int(key))


def _group_colors(keys):
    return ['#4C72B0', '#8172B2', '#C44E52'] if keys and isinstance(keys[0], str) else ['#C44E52', '#4C72B0']


def _rdkit_descriptors(smiles_list):
    """Compute RDKit descriptors for every molecule. Returns dict of lists."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    calc = {
        'MolWt':              Descriptors.MolWt,
        'HeavyAtomCount':     Descriptors.HeavyAtomCount,
        'NumHDonors':         rdMolDescriptors.CalcNumHBD,
        'NumHAcceptors':      rdMolDescriptors.CalcNumHBA,
        'NumRotatableBonds':  rdMolDescriptors.CalcNumRotatableBonds,
        'RingCount':          rdMolDescriptors.CalcNumRings,
        'TPSA':               Descriptors.TPSA,
        'MolLogP':            Descriptors.MolLogP,
        'FractionCSP3':       rdMolDescriptors.CalcFractionCSP3,
        'NumAromaticRings':   rdMolDescriptors.CalcNumAromaticRings,
    }
    result = {k: [] for k in calc}
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi) if smi else None
        for name, fn in calc.items():
            try:
                result[name].append(fn(mol) if mol else float('nan'))
            except Exception:
                result[name].append(float('nan'))
    return result


def _scaffold(smi):
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    mol = Chem.MolFromSmiles(smi) if smi else None
    if mol is None:
        return ''
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def _bxp(ax, data_by_label, xlabel='', ylabel='', title=''):
    """Box plot grouped by label (class or target tercile)."""
    labels_present = _group_order(data_by_label.keys())
    data   = [np.array([v for v in data_by_label[l] if not np.isnan(v)])
              for l in labels_present]
    colors = _group_colors(labels_present)
    bps = ax.boxplot(data, patch_artist=True, widths=0.5,
                     medianprops=dict(color='black', linewidth=1.5))
    for patch, c in zip(bps['boxes'], colors[:len(labels_present)]):
        patch.set_facecolor(c)
        patch.set_alpha(0.75)
    ax.set_xticks(range(1, len(labels_present) + 1))
    ax.set_xticklabels([_group_name(l) for l in labels_present], fontsize=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight='bold')
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)


# ── main computation ──────────────────────────────────────────────────────────

def compute_card(dataset_name, data_root, seeds, split_dir, config_path):
    from src.data import load_dataset, get_smiles
    from src.splits import get_or_create_split

    print(f"\n{'='*60}")
    print(f"  Dataset card: {dataset_name}")
    print(f"{'='*60}")

    dataset, meta = load_dataset(dataset_name, root=data_root)
    smiles_list   = get_smiles(dataset)
    n             = len(dataset)

    # ---- labels (class for binary, target tercile for regression) ----
    task_type = meta['task_type']
    targets   = [_target(d) for d in dataset]
    if task_type == 'regression':
        all_labels = _terciles(targets)
        n_pos = n_neg = None
    else:
        all_labels = [_label(d) for d in dataset]
        valid_labels = [l for l in all_labels if l is not None]
        counts = Counter(valid_labels)
        n_pos  = int(counts.get(1.0, counts.get(1, 0)))
        n_neg  = int(counts.get(0.0, counts.get(0, 0)))

    # ---- graph stats ----
    print("Computing graph statistics...")
    num_nodes, num_edges, avg_deg, max_deg, density = [], [], [], [], []
    nodes_by_label, edges_by_label = defaultdict(list), defaultdict(list)

    for i, data in enumerate(dataset):
        nn  = data.x.shape[0]
        ne  = data.edge_index.shape[1] // 2   # undirected
        deg = data.edge_index.shape[1] / max(nn, 1)
        den = (2 * ne) / max(nn * (nn - 1), 1)
        num_nodes.append(nn)
        num_edges.append(ne)
        avg_deg.append(deg)
        density.append(den)

        # per-node max degree
        from torch import zeros
        deg_vec = zeros(nn)
        deg_vec.scatter_add_(0, data.edge_index[0], zeros(data.edge_index.shape[1]).fill_(1))
        max_deg.append(int(deg_vec.max().item()) if nn > 0 else 0)

        lbl = all_labels[i]
        if lbl is not None:
            nodes_by_label[lbl].append(nn)
            edges_by_label[lbl].append(ne)

    # ---- scaffolds ----
    print("Computing scaffolds...")
    scaffolds = [_scaffold(s) for s in smiles_list]
    scaffold_counts = Counter(scaffolds)
    n_unique_scaffolds = len(scaffold_counts)
    scaffold_sizes = sorted(scaffold_counts.values(), reverse=True)

    # ---- RDKit descriptors ----
    print("Computing RDKit descriptors...")
    descs = _rdkit_descriptors(smiles_list)

    descs_by_label = {name: defaultdict(list) for name in DESCRIPTORS}
    for i, lbl in enumerate(all_labels):
        if lbl is None:
            continue
        for name in DESCRIPTORS:
            v = descs[name][i]
            if not np.isnan(v):
                descs_by_label[name][lbl].append(v)

    # ---- split stats per seed ----
    print(f"Computing split stats for seeds {seeds}...")
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    split_stats = {}
    for seed in seeds:
        tr, vl, te = get_or_create_split(
            dataset,
            split_dir=split_dir,
            dataset_name=dataset_name,
            seed=seed,
            train_frac=cfg.get('train_frac', 0.8),
            val_frac=cfg.get('val_frac', 0.1),
            test_frac=cfg.get('test_frac', 0.1),
        )
        def pos_ratio(idxs):
            lbs = [all_labels[i] for i in idxs if all_labels[i] is not None]
            return sum(1 for l in lbs if l == 1.0) / max(len(lbs), 1)

        def target_stat(idxs, fn):
            return round(float(fn([targets[i] for i in idxs if targets[i] is not None])), 4)

        split_stats[seed] = {'train_size': len(tr), 'val_size': len(vl), 'test_size': len(te)}
        for split, idxs in [('train', tr), ('val', vl), ('test', te)]:
            if task_type == 'regression':
                split_stats[seed][f'{split}_target_mean'] = target_stat(idxs, np.mean)
                split_stats[seed][f'{split}_target_std']  = target_stat(idxs, np.std)
            else:
                split_stats[seed][f'{split}_pos_ratio'] = round(pos_ratio(idxs), 4)

    # ---- assemble card ----
    def _agg(arr):
        a = np.array([v for v in arr if not np.isnan(v)])
        return {'mean': round(float(np.mean(a)), 4),
                'std':  round(float(np.std(a)),  4),
                'min':  round(float(np.min(a)),  4),
                'max':  round(float(np.max(a)),  4)}

    card = {
        'dataset': dataset_name,
        'n_molecules': n,
        'n_tasks': meta['num_tasks'],
        'task_type': task_type,
    }
    if task_type == 'regression':
        card['target_stats'] = _agg([v for v in targets if v is not None])
        card['tercile_edges'] = [round(float(q), 4) for q in
                                 np.quantile([v for v in targets if v is not None], [1 / 3, 2 / 3])]
    else:
        card.update({
            'n_positive': n_pos,
            'n_negative': n_neg,
            'positive_ratio': round(n_pos / max(n_pos + n_neg, 1), 4),
        })
        if meta['num_tasks'] > 1:
            ys = np.stack([d.y.view(-1).numpy() for d in dataset])
            card['per_task_positive_ratio'] = [round(float(np.nanmean(ys[:, t])), 4)
                                               for t in range(ys.shape[1])]
    card.update({
        'n_unique_scaffolds': n_unique_scaffolds,
        'scaffold_size_stats': _agg(scaffold_sizes),
        'graph_stats': {
            'num_nodes': _agg(num_nodes),
            'num_edges': _agg(num_edges),
            'avg_degree': _agg(avg_deg),
            'max_degree': _agg(max_deg),
            'density': _agg(density),
        },
        'descriptor_stats': {
            name: {
                'overall': _agg(descs[name]),
                'by_label': {
                    _group_key(lbl): _agg(vals)
                    for lbl, vals in descs_by_label[name].items()
                }
            }
            for name in DESCRIPTORS
        },
        'split_stats': {str(s): v for s, v in split_stats.items()},
    })
    if task_type == 'regression':
        # Pearson r between each descriptor and the target
        corr = {}
        for name in DESCRIPTORS:
            pairs = [(d, t) for d, t in zip(descs[name], targets) if t is not None and not np.isnan(d)]
            if len(pairs) > 2:
                d_arr, t_arr = np.array(pairs).T
                corr[name] = round(float(np.corrcoef(d_arr, t_arr)[0, 1]), 4)
        card['descriptor_target_corr'] = corr

    return card, {
        'num_nodes': num_nodes, 'num_edges': num_edges,
        'avg_deg': avg_deg, 'density': density,
        'nodes_by_label': nodes_by_label, 'edges_by_label': edges_by_label,
        'scaffold_sizes': scaffold_sizes,
        'descs': descs, 'descs_by_label': descs_by_label,
        'split_stats': split_stats,
        'all_labels': all_labels,
        'targets': targets,
        'task_type': task_type,
        'n_pos': n_pos, 'n_neg': n_neg,
    }


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_overview(card, raw, out_dir, dataset_name):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{dataset_name} — Dataset Overview', fontsize=14, fontweight='bold')

    regression = raw['task_type'] == 'regression'
    seeds_sorted = sorted(raw['split_stats'].keys())
    splits = ['train', 'val', 'test']
    split_colors = ['#4C72B0', '#55A868', '#C44E52']

    ax = axes[0, 0]
    if regression:
        # Target distribution with tercile edges
        ax.hist([v for v in raw['targets'] if v is not None], bins=40, color='#4C72B0', alpha=0.85)
        for q in card['tercile_edges']:
            ax.axvline(q, color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel('Target', fontsize=10)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Target Distribution (dashed: tercile edges)', fontsize=12, fontweight='bold')
    else:
        # Class balance
        bars = ax.bar(['Negative (0)', 'Positive (1)'],
                      [card['n_negative'], card['n_positive']],
                      color=['#C44E52', '#4C72B0'], alpha=0.85, width=0.5)
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 10,
                    f'{int(b.get_height())}', ha='center', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title('Class Balance', fontsize=12, fontweight='bold')
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    # Positive ratio (binary) / mean target (regression) per split across seeds
    ax = axes[0, 1]
    stat = 'target_mean' if regression else 'pos_ratio'
    for split, color in zip(splits, split_colors):
        vals = [raw['split_stats'][s][f'{split}_{stat}'] for s in seeds_sorted]
        ax.plot(seeds_sorted, vals, marker='o', label=split.capitalize(),
                color=color, linewidth=1.8)
    overall = card['target_stats']['mean'] if regression else card['positive_ratio']
    ax.axhline(overall, color='gray', linestyle='--',
               linewidth=1, label='Overall')
    ax.set_xlabel('Seed', fontsize=10)
    if regression:
        ax.set_ylabel('Mean Target', fontsize=10)
        ax.set_title('Mean Target per Split across Seeds', fontsize=12, fontweight='bold')
    else:
        ax.set_ylabel('Positive Class Ratio', fontsize=10)
        ax.set_title('Positive Ratio per Split across Seeds', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_xticks(seeds_sorted)

    # Scaffold size distribution
    ax = axes[1, 0]
    sizes = raw['scaffold_sizes']
    bins  = [1, 2, 3, 5, 10, 20, 50, max(sizes) + 1]
    counts, edges = np.histogram(sizes, bins=bins)
    labels = [f'{int(edges[i])}–{int(edges[i+1])-1}' for i in range(len(counts))]
    ax.bar(range(len(counts)), counts, color='#8172B2', alpha=0.82)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=9)
    ax.set_xlabel('Scaffold Size (# molecules)', fontsize=10)
    ax.set_ylabel('# Scaffolds', fontsize=10)
    ax.set_title(f'Scaffold Size Distribution\n'
                 f'({card["n_unique_scaffolds"]} unique scaffolds)', fontsize=12, fontweight='bold')
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    # Split sizes per seed
    ax = axes[1, 1]
    for split, color in zip(splits, split_colors):
        sizes_s = [raw['split_stats'][s][f'{split}_size'] for s in seeds_sorted]
        ax.plot(seeds_sorted, sizes_s, marker='s', label=split.capitalize(),
                color=color, linewidth=1.8)
    ax.set_xlabel('Seed', fontsize=10)
    ax.set_ylabel('# Molecules', fontsize=10)
    ax.set_title('Split Sizes across Seeds', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    ax.set_xticks(seeds_sorted)

    plt.tight_layout()
    path = os.path.join(out_dir, 'overview.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def plot_graph_structure(raw, out_dir, dataset_name):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    by = 'Target Tercile' if raw['task_type'] == 'regression' else 'Label'
    fig.suptitle(f'{dataset_name} — Graph Structure by {by}', fontsize=14, fontweight='bold')

    _bxp(axes[0, 0], raw['nodes_by_label'],
         ylabel='# Atoms', title=f'Num Atoms by {by}')
    _bxp(axes[0, 1], raw['edges_by_label'],
         ylabel='# Bonds (undirected)', title=f'Num Edges by {by}')

    # Avg degree and density don't have by-label breakdown in raw — compute
    from collections import defaultdict
    deg_by_label, den_by_label = defaultdict(list), defaultdict(list)
    for i, lbl in enumerate(raw['all_labels']):
        if lbl is not None:
            deg_by_label[lbl].append(raw['avg_deg'][i])
            den_by_label[lbl].append(raw['density'][i])

    _bxp(axes[1, 0], deg_by_label,
         ylabel='Avg Degree', title=f'Average Degree by {by}')
    _bxp(axes[1, 1], den_by_label,
         ylabel='Graph Density', title=f'Density by {by}')

    plt.tight_layout()
    path = os.path.join(out_dir, 'graph_structure.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def plot_chemistry(raw, out_dir, dataset_name):
    props = ['MolWt', 'NumHDonors', 'NumHAcceptors',
             'RingCount', 'TPSA', 'MolLogP']
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    by = 'Target Tercile' if raw['task_type'] == 'regression' else 'Label'
    fig.suptitle(f'{dataset_name} — Molecular Properties by {by}', fontsize=14, fontweight='bold')

    for ax, name in zip(axes.flat, props):
        _bxp(ax, raw['descs_by_label'][name], ylabel=name, title=name)

    plt.tight_layout()
    path = os.path.join(out_dir, 'chemistry.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def plot_descriptor_corr(raw, out_dir, dataset_name):
    # Build matrix of descriptors
    descs = raw['descs']
    names = DESCRIPTORS
    mat   = np.array([descs[n] for n in names], dtype=float)  # [D, N]

    # Mask NaN columns
    valid = ~np.any(np.isnan(mat), axis=0)
    mat   = mat[:, valid]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        corr = np.corrcoef(mat)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)))
    ax.set_yticks(range(len(names)))
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_title(f'{dataset_name} — Descriptor Correlation', fontsize=13, fontweight='bold')

    for i in range(len(names)):
        for j in range(len(names)):
            val = corr[i, j]
            color = 'white' if abs(val) > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=7, color=color)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Pearson r')
    plt.tight_layout()
    path = os.path.join(out_dir, 'descriptor_corr.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


def plot_atom_type_freq(dataset, labels, out_dir, dataset_name):
    """Atom type frequency by label / target tercile (optional — skip if slow)."""
    from collections import defaultdict
    atom_counts = defaultdict(lambda: defaultdict(int))
    for data, lbl in zip(dataset, labels):
        if lbl is None:
            continue
        # x[:, 0] is typically the atomic number in PyG MoleculeNet featurization
        for row in data.x.tolist():
            atom_num = int(row[0])
            atom_counts[lbl][atom_num] += 1

    common = Counter()
    for lbl in atom_counts:
        common.update(atom_counts[lbl])
    top_atoms = [a for a, _ in common.most_common(15)]

    labels_present = _group_order(atom_counts.keys())
    fig, ax = plt.subplots(figsize=(12, 5))
    width = 0.7 / len(labels_present)
    x = np.arange(len(top_atoms))

    for k, (lbl, color) in enumerate(zip(labels_present, _group_colors(labels_present))):
        counts = [atom_counts[lbl].get(a, 0) for a in top_atoms]
        ax.bar(x + (k - (len(labels_present) - 1) / 2) * width, counts, width, label=_group_name(lbl),
               color=color, alpha=0.82)

    ax.set_xticks(x)
    ax.set_xticklabels([f'AtomNum {a}' for a in top_atoms], rotation=35, ha='right', fontsize=9)
    ax.set_ylabel('Total Count', fontsize=11)
    ax.set_title(f'{dataset_name} — Atom Type Frequency by Group\n'
                 f'(feature dim 0 = atomic number)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    path = os.path.join(out_dir, 'atom_type_freq.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {path}")


# ── entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Generate dataset diagnostic card.')
    p.add_argument('--datasets',   nargs='+', default=['BACE'])
    p.add_argument('--seeds',      nargs='+', type=int, default=list(range(10)))
    p.add_argument('--data_root',  default='./data')
    p.add_argument('--split_dir',  default='./data/splits')
    p.add_argument('--configs_dir', default='./configs')
    p.add_argument('--out_root',   default='./results/dataset_cards')
    args = p.parse_args()

    for dataset in args.datasets:
        dataset = dataset.upper()
        config_path = os.path.join(args.configs_dir, f'{dataset.lower()}.yaml')
        out_dir     = os.path.join(args.out_root, dataset)
        os.makedirs(out_dir, exist_ok=True)

        card, raw = compute_card(
            dataset_name=dataset,
            data_root=args.data_root,
            seeds=args.seeds,
            split_dir=args.split_dir,
            config_path=config_path,
        )

        # Save JSON card
        card_path = os.path.join(out_dir, 'card.json')
        with open(card_path, 'w') as f:
            json.dump(card, f, indent=2)
        print(f"\n  Saved card → {card_path}")

        # Plots
        print("  Generating plots...")
        from src.data import load_dataset
        dataset_obj, _ = load_dataset(dataset, root=args.data_root)

        plot_overview(card, raw, out_dir, dataset)
        plot_graph_structure(raw, out_dir, dataset)
        plot_chemistry(raw, out_dir, dataset)
        plot_descriptor_corr(raw, out_dir, dataset)
        plot_atom_type_freq(dataset_obj, raw['all_labels'], out_dir, dataset)

        print(f"\n  Done. Outputs in {out_dir}/")

        # Print summary
        print(f"\n  {dataset} summary:")
        if card['task_type'] == 'regression':
            t = card['target_stats']
            print(f"    {card['n_molecules']} molecules, target "
                  f"{t['mean']:.3f} ± {t['std']:.3f} (range {t['min']:.3f} to {t['max']:.3f})")
        else:
            print(f"    {card['n_molecules']} molecules, "
                  f"{card['n_positive']} pos / {card['n_negative']} neg "
                  f"({card['positive_ratio']:.1%} positive)")
        print(f"    {card['n_unique_scaffolds']} unique scaffolds")
        g = card['graph_stats']
        print(f"    Atoms: {g['num_nodes']['mean']:.1f} ± {g['num_nodes']['std']:.1f}  "
              f"Bonds: {g['num_edges']['mean']:.1f} ± {g['num_edges']['std']:.1f}")


if __name__ == '__main__':
    main()
