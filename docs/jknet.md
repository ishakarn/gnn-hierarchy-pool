# JKNet integration

JKNet is a GCN backbone with Jumping Knowledge over the node representations from
every message-passing layer. This happens before graph pooling. The pooled output
uses the existing two-layer MLP prediction head. All five study readouts work:
max, mean, sum, attention and multihead. Binary, multitask binary and regression
heads use the existing task machinery.

Modes: cat (default), max, lstm. Cat concatenates all layer representations and
projects back to hidden_dim; max/LSTM return hidden_dim directly. This keeps the
readout width fixed when depth changes. Uses PyG's JumpingKnowledge implementation:
https://pytorch-geometric.readthedocs.io/en/2.6.1/generated/torch_geometric.nn.models.JumpingKnowledge.html
Paper: https://arxiv.org/abs/1806.03536

## Single run, grid and small sweep

Run from the repository root in the existing torch/PyG/RDKit environment:

    python scripts/run_single.py --config configs/bbbp.yaml --model jknet --readout mean --jk_mode cat --seed 0

    python scripts/run_grid.py --datasets BBBP BACE CLINTOX --models jknet --jk_mode max --readouts max mean sum attention multihead --seeds 0 1 2 3 4 --split spectra --sps 0.00 0.25 0.50 0.75 1.00 --split_idxs 0 1 2 --skip_existing --output_dir results/jknet_grid

    python scripts/run_sweep.py --dataset BBBP --model jknet --jk_mode lstm --readout mean --split spectra --sps 0.00 0.25 --split_idxs 0 1 2 --output_dir results/jknet_lstm_sweep

Non-default modes use model tags jknet-max and jknet-lstm in paths and metrics;
cat uses jknet. Modes never overwrite one another or mix during selection.

For the previously prepared strictly disjoint scaffold splits, pass the audited
configuration instead of the legacy dataset config:

    python scripts/run_sweep.py --dataset BBBP --model jknet --readout mean --config /work/pi_annagreen_umass_edu/helly/afp-scaffold/gnn-hierarchy-pool/configs/unity/scaffold80/bbbp.yaml --output_dir results/jknet_scaffold80

## Audited Slurm sweep

The default is an audit/dry run. It does not submit training jobs.

    python scripts/sweep_jobs.py prepare --model jknet --jk_mode cat --split spectra

    python scripts/sweep_jobs.py prepare --model jknet --jk_mode cat --split scaffold --config_dir /work/pi_annagreen_umass_edu/helly/afp-scaffold/gnn-hierarchy-pool/configs/unity/scaffold80

Defaults match the prior workflow: BBBP/BACE/ClinTox; five readouts; layers 2-6;
dropout 0.0-0.5 in increments of 0.1; training seeds 0-4; 300 epochs; patience 50.
The SPECTRA values default to the requested 0.00, 0.25, 0.50, 0.75, 1.00,
with all three supplied split replicates. Missing or unusable splits are audited
and excluded. The current supplied files yield 375 SPECTRA tasks/11,250 fits and
75 scaffold tasks/2,250 fits. Actual split partitions are reused; none are rerolled.

To submit from the repository root:

    export VENV=/work/pi_annagreen_umass_edu/helly/afp-scaffold/.venv
    python scripts/sweep_jobs.py prepare --model jknet --split spectra --submit

A CPU preparation job builds shared graph caches before reserving GPUs. Each
array task runs one dataset/readout/depth/SP/replicate: 30 independent fits.
One Python/CUDA process and one in-memory dataset are reused throughout the task.
The concurrency default is four GPUs. Arrays larger than 500 tasks are chained.
A dependent CPU job verifies expected result identities and collects the results.
All jobs use an immutable manifest with source, configuration and input hashes.
Editing that source after submission causes a clear error, so prepare a separate
checkout for concurrent experiments.

For another JK mode, use --jk_mode max or --jk_mode lstm. Run one mode at a time
to retain a four-GPU total cap; each submission has its own cap.
The older entry points also accept JKNet:
    bash scripts/slurm/submit_sweep.sh jknet --dry-run
    sbatch scripts/slurm/run_sweep.slurm BBBP jknet mean --jk_mode max
    python scripts/slurm/gen_combos.py --models jknet --jk_mode lstm --readouts mean multihead

The default data cache is data/cache/graphs. It is keyed by CSV contents, label
columns, featurizer source and library versions. Graphs stay on CPU; standard
minibatch transfers still occur during training. Each fit resets random seeds
and constructs a fresh model/optimizer. --fresh_process opts out of in-process
reuse for debugging while retaining the shared graph cache.

## Results, selection and recovery

Standalone sweeps can be collected with:

    python scripts/aggregate_sweep.py --results_dir results/jknet_lstm_sweep --min_seeds 5

Slurm manifests can be collected or a failed task restarted using the same scripts:

    python scripts/sweep_jobs.py collect /absolute/path/to/manifest.json
    sbatch --array=<failed_task_indices>%4 scripts/slurm/run_sweep_array.slurm /absolute/path/to/manifest.json

Completed fits with matching settings are skipped. Incomplete fits restart.
Selection uses mean validation over seeds, with ties within 0.001 preferring
fewer layers then lower dropout. Each SPECTRA value/replicate and JK mode is kept
separate. Test results are reported only. Seed uncertainty is conditional on the
split; sweep_summary.csv reports variation across the actual split replicates.
ClinTox keeps both tasks and reports mean task ROC-AUC.

## Verification

    python -m unittest discover -s tests -p test_jknet.py -v

Real-data smoke checks are in tests/smoke_jknet.py. Run through
tests/jknet_check.slurm on a compute node with VENV and JKNET_REFERENCE_ROOT set.
They cover the exact single/grid/sweep and Slurm manifest entry points, data
reuse, resume, both split types and split-aware collection.
