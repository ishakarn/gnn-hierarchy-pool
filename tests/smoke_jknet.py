"""Real-data CLI checks; run on a CPU/GPU compute node, never a login node."""
import argparse
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.data_cache import DatasetCache
from src.run_identity import run_name
from scripts.run_single import main as single


def execute(arguments):
    result = subprocess.run([sys.executable] + arguments, cwd=ROOT,
                            text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    print(result.stdout, flush=True)
    if result.returncode:
        raise RuntimeError(f"CLI failed: {arguments}")
    return result.stdout


def main():
    reference = Path(os.environ["JKNET_REFERENCE_ROOT"])
    root = ROOT / "results/integration_smoke" / os.environ.get("SLURM_JOB_ID", "manual")
    root.mkdir(parents=True, exist_ok=True)
    graph_cache = str(ROOT / "data/cache/graphs")
    cache = DatasetCache(graph_cache)
    for mode, readout in [("cat", ro) for ro in ("max", "mean", "sum", "attention", "multihead")] + [
            ("max", "mean"), ("lstm", "mean")]:
        single(["--config", "configs/bace.yaml", "--model", "jknet",
                "--jk_mode", mode, "--readout", readout, "--split", "spectra",
                "--sp", "0.00", "--split_idx", "0", "--seed", "0",
                "--num_layers", "2", "--epochs", "2", "--patience", "1",
                "--output_dir", str(root / "single")], data_cache=cache)
    if cache.builds + cache.disk_loads != 1:
        raise RuntimeError("Single-run loop loaded BACE more than once.")
    single(["--config", "configs/clintox.yaml", "--model", "jknet", "--readout", "mean",
            "--split", "spectra", "--sp", "0.00", "--epochs", "2", "--patience", "1",
            "--num_layers", "2", "--output_dir", str(root / "clintox")], data_cache=cache)
    single(["--config", str(reference / "configs/unity/scaffold80/bace.yaml"),
            "--model", "jknet", "--readout", "mean", "--num_layers", "2",
            "--epochs", "2", "--patience", "1", "--output_dir", str(root / "scaffold")],
           data_cache=cache)
    # Resume and all mode-specific paths use the same CLI as a Slurm task.
    command = ["scripts/run_sweep.py", "--dataset", "BACE", "--config", "configs/bace.yaml",
               "--model", "jknet", "--jk_mode", "max", "--readout", "mean",
               "--split", "spectra", "--sps", "0", "--split_idxs", "0",
               "--layers", "2", "3", "--dropouts", "0", ".1", "--seeds", "0", "1",
               "--epochs", "2", "--patience", "1", "--graph_cache_dir", graph_cache,
               "--output_dir", str(root / "sweep")]
    first = execute(command)
    if first.count("[data_cache] disk load: bace.csv") != 1:
        raise RuntimeError("Sweep did not load preprocessed BACE exactly once.")
    second = execute(command)
    if "8 resumed, 0 failed" not in second or "[data_cache]" in second:
        raise RuntimeError("Resume retrained or reloaded a completed sweep.")
    execute(["scripts/aggregate_sweep.py", "--results_dir", str(root / "sweep"), "--min_seeds", "2"])
    execute(["scripts/run_grid.py", "--datasets", "BACE", "--models", "jknet",
             "--jk_mode", "lstm", "--readouts", "mean", "multihead", "--seeds", "0",
             "--split", "spectra", "--sps", "0", "--split_idxs", "0",
             "--epochs", "2", "--output_dir", str(root / "grid"), "--skip_existing"])
    for split, extra in (("spectra", []), ("scaffold", [
            "--config_dir", str(reference / "configs/unity/scaffold80")])):
        text = execute(["scripts/sweep_jobs.py", "prepare", "--model", "jknet", "--split", split] + extra)
        expected = "375 tasks, 11250 fits" if split == "spectra" else "75 tasks, 2250 fits"
        if expected not in text:
            raise RuntimeError(f"Unexpected audited {split} grid.")
    # Exercise the exact manifest/prewarm/task/collector entry points without sbatch.
    import sweep_jobs
    args = argparse.Namespace(model="jknet", jk_mode="lstm", split="spectra",
        output_dir=str(root / "manifest"), config_dir="configs", datasets=["BACE"],
        readouts=["mean"], layers=[2], dropouts=[0.], seeds=[0, 1], epochs=2,
        patience=1, sps=[0.], split_idxs=[0], graph_cache_dir=graph_cache)
    plan = sweep_jobs.plan(args)
    manifest = Path(plan["output_dir"]) / "manifest.json"
    for dataset, cfg in plan["configs"].items():
        path = Path(plan["output_dir"]) / "configs" / f"{dataset.lower()}.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        plan["input_sha256"][str(path)] = sweep_jobs.sha(path)
    sweep_jobs.save(manifest, plan)
    sweep_jobs.prewarm(manifest)
    sweep_jobs.run(manifest, 0)
    sweep_jobs.collect(manifest)
    report = dict(status="passed", checks=[
        "JK modes and five readouts on real BACE", "two-task ClinTox training",
        "audited scaffold split training", "one BACE graph load across seven fits",
        "one graph load across eight sweep fits", "resume skips all eight fits and graph loading",
        "SPECTRA grid CLI", "SPECTRA/scaffold full Slurm dry-run counts",
        "exact manifest prewarm/task/collector workflow"])
    (root / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
