"""Plan/submit persistent Slurm sweeps with audited scaffold or SPECTRA inputs."""
import argparse
import csv
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.run_identity import model_tag, run_name
from run_sweep import LAYERS, DROPOUTS, SEEDS, run_dir


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_hashes():
    files = list((ROOT / "src").rglob("*.py"))
    files += [ROOT / "scripts" / p for p in (
        "run_single.py", "run_grid.py", "run_sweep.py", "aggregate_sweep.py", "sweep_jobs.py")]
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(files)}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def load(path):
    value = json.loads(Path(path).read_text())
    if source_hashes() != value["source_sha256"]:
        raise RuntimeError("Source changed since this sweep was prepared.")
    for file, expected in value["input_sha256"].items():
        if sha(file) != expected:
            raise RuntimeError(f"Audited input changed: {file}")
    return value


def plan(args):
    from src.splits import load_spectra_split, SpectraSplitSkipped
    tag = model_tag(args.model, args.jk_mode)
    output = Path(args.output_dir or f"results/{args.split}_{tag}").resolve()
    tasks, audit, hashes, configs = [], [], {}, {}
    for dataset in args.datasets:
        path = Path(args.config_dir) / f"{dataset.lower()}.yaml"
        cfg = yaml.safe_load(path.read_text())
        if cfg["dataset"].upper() != dataset:
            raise ValueError(f"Wrong dataset in {path}")
        cfg.update(split=args.split, jk_mode=args.jk_mode)
        for key in ("data_root", "split_dir", "spectra_repo"):
            if key in cfg:
                cfg[key] = str(Path(cfg[key]).resolve())
        configs[dataset] = cfg
        config = str(output / "configs" / f"{dataset.lower()}.yaml")
        if args.split == "spectra":
            csv_path = Path(cfg["spectra_repo"]) / cfg["curated_csv"]
            split_dir = Path(cfg["spectra_repo"]) / cfg["spectra_split_dir"]
            hashes[str(csv_path)] = sha(csv_path)
            with csv_path.open() as handle:
                rows = list(csv.DictReader(handle))
            labels = [[float(row[c]) for c in cfg["label_cols"]] for row in rows]
            valid = []
            for sp, rep in product(args.sps, args.split_idxs):
                split_path = split_dir / f"SP_{sp:.2f}/data_{rep}.json"
                record = dict(dataset=dataset, sp=sp, split_idx=rep)
                try:
                    parts = load_spectra_split(str(split_dir), sp, rep, labels=labels,
                                              min_test_size=cfg.get("min_test_size", 20))
                    flat = sum([list(p) for p in parts], [])
                    if not all(parts) or len(flat) != len(set(flat)) or any(
                            type(i) is not int or not 0 <= i < len(rows) for i in flat):
                        raise ValueError(f"Invalid or overlapping split indices: {split_path}")
                    hashes[str(split_path)] = sha(split_path)
                    record.update(eligible=True, sizes=[len(p) for p in parts],
                                  split_sha256=hashes[str(split_path)])
                    valid.append((sp, rep))
                except SpectraSplitSkipped as error:
                    record.update(eligible=False, reason=str(error))
                audit.append(record)
        else:
            # Reuse prepared splits; never generate them from parallel GPU jobs.
            for seed in args.seeds:
                split_path = Path(cfg.get("split_dir", "./data/splits")) / f"{dataset}_scaffold_seed{seed}.json"
                if not split_path.exists():
                    raise ValueError(f"Prepare shared scaffold splits before submitting: {split_path}")
                split = json.loads(split_path.read_text())
                parts = [split[k] for k in ("train", "val", "test")]
                flat = sum(parts, [])
                if not all(parts) or len(flat) != len(set(flat)):
                    raise ValueError(f"Invalid scaffold partitions: {split_path}")
                hashes[str(split_path)] = sha(split_path)
                audit.append(dict(dataset=dataset, seed=seed, eligible=True,
                                  sizes=[len(p) for p in parts], split_sha256=hashes[str(split_path)]))
            valid = [(None, None)]
        for (sp, rep), readout, layers in product(valid, args.readouts, args.layers):
            tasks.append(dict(dataset=dataset, config=config, sp=sp, split_idx=rep,
                              readout=readout, layers=layers))
    if not tasks:
        raise ValueError("No eligible tasks.")
    return dict(model=args.model, model_tag=tag, jk_mode=args.jk_mode, split=args.split,
                output_dir=str(output), graph_cache_dir=str(Path(args.graph_cache_dir).resolve()),
                configs=configs, audit=audit, tasks=tasks, input_sha256=hashes,
                source_sha256=source_hashes(), layers=args.layers, dropouts=args.dropouts,
                seeds=args.seeds, epochs=args.epochs, patience=args.patience,
                expected_runs=len(tasks) * len(args.dropouts) * len(args.seeds))


def prewarm(manifest):
    from src.data_cache import DatasetCache
    value = load(manifest)
    cache = DatasetCache(value["graph_cache_dir"])
    for cfg in value["configs"].values():
        if value["split"] == "spectra":
            cache.load_curated_csv(str(Path(cfg["spectra_repo"]) / cfg["curated_csv"]), cfg["label_cols"])
        else:
            cache.load_dataset(cfg["dataset"], cfg.get("data_root", "./data"))
    save(Path(value["output_dir"]) / "cache.ready", dict(source_sha256=value["source_sha256"]))
    print("Prepared graphs before reserving production GPUs.", flush=True)


def run(manifest, index):
    value = load(manifest)
    ready = json.loads((Path(value["output_dir"]) / "cache.ready").read_text())
    if ready["source_sha256"] != value["source_sha256"]:
        raise RuntimeError("Graph preparation used different source.")
    task = value["tasks"][index]
    command = [sys.executable, str(ROOT / "scripts/run_sweep.py"),
               "--dataset", task["dataset"], "--model", value["model"],
               "--jk_mode", value["jk_mode"], "--config", task["config"],
               "--split", value["split"], "--readout", task["readout"],
               "--layers", str(task["layers"]), "--dropouts", *map(str, value["dropouts"]),
               "--seeds", *map(str, value["seeds"]), "--epochs", str(value["epochs"]),
               "--patience", str(value["patience"]), "--output_dir", value["output_dir"],
               "--graph_cache_dir", value["graph_cache_dir"]]
    if task["sp"] is not None:
        command += ["--sps", str(task["sp"]), "--split_idxs", str(task["split_idx"])]
    subprocess.run(command, cwd=ROOT, check=True)


def collect(manifest):
    import pandas as pd
    from aggregate_sweep import select_configs
    value = load(manifest)
    root = Path(value["output_dir"])
    expected, rows = set(), []
    for task in value["tasks"]:
        for dropout, seed in product(value["dropouts"], value["seeds"]):
            base = root
            if task["sp"] is not None:
                base = base / f"SP_{task['sp']:.2f}" / f"split{task['split_idx']}"
            folder = Path(run_dir(str(base), task["dataset"], value["model_tag"],
                                  task["readout"], task["layers"], dropout))
            name = run_name(task["dataset"], value["model"], task["readout"], seed,
                            task["sp"], task["split_idx"], value["jk_mode"])
            path = folder / f"{name}_metrics.json"
            if path in expected:
                raise RuntimeError("Duplicate run in manifest.")
            expected.add(path)
            row = json.loads(path.read_text())
            identity = dict(dataset=task["dataset"], model=value["model_tag"], readout=task["readout"],
                            seed=seed, num_layers=task["layers"], dropout=dropout,
                            epochs=value["epochs"], patience=value["patience"], split=value["split"],
                            status="ok")
            if value["model"] == "jknet":
                identity["jk_mode"] = value["jk_mode"]
            if task["sp"] is not None:
                identity.update(sp=task["sp"], split_idx=task["split_idx"])
            if any(row.get(k) != v for k, v in identity.items()):
                raise RuntimeError(f"Result identity mismatch: {path}")
            rows.append(row)
    if len(rows) != value["expected_runs"]:
        raise RuntimeError("Missing results.")
    frame = pd.DataFrame(rows)
    selected = select_configs(frame, min_seeds=len(value["seeds"]))
    groups = ["dataset", "model", "readout"] + (
        ["sp", "split_idx"] if value["split"] == "spectra" else [])
    if len(selected) != frame.groupby(groups).ngroups:
        raise RuntimeError("Missing finite validation results for selection.")
    frame.to_csv(root / "sweep_results.csv", index=False)
    selected.to_csv(root / "sweep_selected.csv", index=False)
    if value["split"] == "spectra":
        selected.groupby(["dataset", "model", "readout", "sp"]).agg(
            n_split_replicates=("split_idx", "nunique"), test_mean=("test_mean", "mean"),
            test_std_across_splits=("test_mean", "std")).reset_index().to_csv(
                root / "sweep_summary.csv", index=False)
    save(root / "production.complete", dict(runs=len(rows), selected=len(selected)))
    print(f"Collected {len(rows)} fits; selected {len(selected)} configurations using validation only.")


def submit(value, concurrency):
    root = Path(value["output_dir"])
    manifest = root / "manifest.json"
    if manifest.exists():
        raise RuntimeError(f"Manifest already exists; retain it and resubmit only failed array tasks: {manifest}")
    root.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(exist_ok=True)
    for dataset, cfg in value["configs"].items():
        path = root / "configs" / f"{dataset.lower()}.yaml"
        path.parent.mkdir(exist_ok=True)
        path.write_text(yaml.safe_dump(cfg, sort_keys=False))
        value["input_sha256"][str(path)] = sha(path)
    save(manifest, value)
    jobs = []
    def sbatch(arguments, kind):
        job = subprocess.check_output(["sbatch", "--parsable"] + arguments, cwd=ROOT, text=True).strip().split(";")[0]
        jobs.append(dict(kind=kind, id=job))
        save(root / "submitted_jobs.json", jobs)
        return job
    previous = sbatch(["scripts/slurm/sweep_prepare.slurm", str(manifest)], "prepare")
    arrays = []
    for start in range(0, len(value["tasks"]), 500):
        end = min(start + 499, len(value["tasks"]) - 1)
        previous = sbatch([f"--array={start}-{end}%{concurrency}", "--job-name=" + value["model_tag"],
                           "--dependency=afterok:" + previous,
                           "scripts/slurm/run_sweep_array.slurm", str(manifest)], "sweep")
        arrays.append(previous)
    sbatch(["--dependency=afterok:" + ":".join(arrays),
            "scripts/slurm/sweep_collect.slurm", str(manifest)], "collect")
    print(json.dumps(jobs, indent=2))


def main():
    os.chdir(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--model", choices=["jknet", "gin", "gine", "attentivefp"], default="jknet")
    p.add_argument("--jk_mode", choices=["cat", "max", "lstm"], default="cat")
    p.add_argument("--split", choices=["scaffold", "spectra"], required=True)
    p.add_argument("--datasets", nargs="+", choices=["BBBP", "BACE", "CLINTOX"], default=["BBBP", "BACE", "CLINTOX"])
    p.add_argument("--readouts", nargs="+", choices=["max", "mean", "sum", "attention", "multihead"],
                   default=["max", "mean", "sum", "attention", "multihead"])
    p.add_argument("--layers", nargs="+", type=int, default=LAYERS)
    p.add_argument("--dropouts", nargs="+", type=float, default=DROPOUTS)
    p.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--patience", type=int, default=50)
    p.add_argument("--sps", nargs="+", type=float, default=[0., .25, .5, .75, 1.])
    p.add_argument("--split_idxs", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--config_dir", default="configs")
    p.add_argument("--graph_cache_dir", default="./data/cache/graphs")
    p.add_argument("--output_dir")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--submit", action="store_true", help="Submit; default only audits and prints the plan")
    for command in ("prewarm", "collect"):
        p = sub.add_parser(command)
        p.add_argument("manifest")
    p = sub.add_parser("run")
    p.add_argument("manifest")
    p.add_argument("task", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        if (args.concurrency < 1 or any(n < 1 for n in args.layers)
                or any(not 0 <= d < 1 for d in args.dropouts)
                or any(not 0 <= sp <= 1 or abs(sp * 20 - round(sp * 20)) > 1e-8 for sp in args.sps)
                or any(rep not in (0, 1, 2) for rep in args.split_idxs)):
            parser.error("Invalid sweep grid or concurrency.")
        value = plan(args)
        print(f"{len(value['tasks'])} tasks, {value['expected_runs']} fits; max {args.concurrency} GPUs")
        excluded = [r for r in value["audit"] if not r["eligible"]]
        for record in excluded:
            print("[excluded]", record)
        if args.submit:
            submit(value, args.concurrency)
        else:
            print("Audit only. Add --submit to prepare graphs on CPU, run the GPU arrays and collect results.")
    elif args.command == "prewarm":
        prewarm(args.manifest)
    elif args.command == "run":
        run(args.manifest, args.task)
    else:
        collect(args.manifest)


if __name__ == "__main__":
    main()
