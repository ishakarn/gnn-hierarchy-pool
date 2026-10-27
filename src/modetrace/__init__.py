"""ModeTrace: how much a model relies on node features, edge features and topology.

Ported from github.com/ishakarn/modes-of-information; runs this repo's models,
training step and metrics. perturbations/ = feature shuffles, data.py = scaffold
10-fold CV, protocols.py = matched and OOD protocols, summary.py = fold summaries.
Entry points: scripts/modetrace/run_matched.py, run_ood.py, summarize.py.
"""

PERTURBATIONS = ("none", "WNFS", "GNFS", "WEFS", "GEFS", "JGFS")

P_VALUES  = (0.0, 0.25, 0.50, 0.75, 1.00)
NUM_FOLDS = 10

# Overridable with --config (e.g. configs/bbbp.yaml for benchmark hyperparameters)
DEFAULTS = dict(
    hidden_dim   = 128,
    num_layers   = 2,
    dropout      = 0.3,
    lr           = 1e-3,
    weight_decay = 1e-5,
    epochs       = 100,
    batch_size   = 64,
    pos_weight   = True,    # BCE pos_weight = n_neg / n_pos (binary only)
)
