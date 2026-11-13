"""Verify SPECTRA metadata using the normal real-data training entry point."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_single import main

folder = ROOT / "results/integration_smoke" / ("metadata-" + os.environ.get("SLURM_JOB_ID", "manual"))
main(["--config", "configs/bace.yaml", "--model", "jknet", "--readout", "mean",
      "--split", "spectra", "--sp", "0.00", "--split_idx", "0", "--seed", "0",
      "--num_layers", "2", "--epochs", "2", "--patience", "1", "--output_dir", str(folder)])
value = json.loads((folder / "BACE_jknet_mean_sp0.00_split0_seed0_metrics.json").read_text())
assert value["split_method"] == "spectra_tanimoto"
assert "spectra_tanimoto/bace" in value["split_dir"]
assert value["n_train"] + value["n_val"] + value["n_test"] == 1513
for part in ("train", "val", "test"):
    assert value[f"{part}_frac"] == value[f"n_{part}"] / 1513
assert value["jk_mode"] == "cat" and value["status"] == "ok"
(folder / "verification.json").write_text(json.dumps(
    dict(status="passed", checks=["Actual SPECTRA fractions, split method and path; JK mode metadata"]),
    indent=2) + "\n")
print("PASS: SPECTRA metadata reports actual curated-data partitions.")
