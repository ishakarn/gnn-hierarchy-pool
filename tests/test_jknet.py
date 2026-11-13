import copy
from itertools import product
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
from torch_geometric.utils import from_smiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.models import build_model, JKNetModel
from src.data_cache import DatasetCache
from src.run_identity import run_name
from aggregate_sweep import select_configs


class JKNetTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.batch = Batch.from_data_list([from_smiles(s) for s in ("CCO", "CCN", "[He]")])
        self.cfg = dict(hidden_dim=16, num_layers=3, dropout=.2, readout="mean")

    def test_modes_readouts_depths_and_gradients(self):
        for mode, readout, depth in product(
                ("cat", "max", "lstm"), ("max", "mean", "sum", "attention", "multihead"), (2, 6)):
            with self.subTest(mode=mode, readout=readout, depth=depth):
                cfg = dict(self.cfg, jk_mode=mode, readout=readout, num_layers=depth)
                model = build_model("jknet", 9, cfg, num_tasks=2)
                output = model(self.batch)
                self.assertEqual(tuple(output.shape), (3, 2))
                output.square().sum().backward()
                self.assertTrue(all(c.lin.weight.grad is not None for c in model.convs))
                model.eval()
                whole = model(self.batch).detach()
                separate = torch.cat([model(Batch.from_data_list([g]))
                                      for g in self.batch.to_data_list()]).detach()
                torch.testing.assert_close(whole, separate, rtol=1e-4, atol=1e-5)

    def test_jk_is_node_layer_aggregation_before_readout(self):
        model = build_model("jknet", 9, dict(self.cfg, jk_mode="max", dropout=0))
        model.eval()
        x, states = self.batch.x.float(), []
        for conv in model.convs:
            x = torch.relu(conv(x, self.batch.edge_index))
            states.append(x)
        expected = torch.stack(states).max(dim=0).values
        torch.testing.assert_close(model.encode(self.batch), expected)
        self.assertEqual(tuple(expected.shape), (7, 16))

    def test_node_permutation_invariance(self):
        for mode in ("cat", "max", "lstm"):
            model = build_model("jknet", 9, dict(self.cfg, jk_mode=mode, readout="attention")).eval()
            permutation = torch.tensor([2, 0, 1, 5, 3, 4, 6])
            inverse = torch.argsort(permutation)
            shuffled = self.batch.clone()
            shuffled.x = self.batch.x[permutation]
            shuffled.batch = self.batch.batch[permutation]
            shuffled.edge_index = inverse[self.batch.edge_index]
            torch.testing.assert_close(model(self.batch), model(shuffled), rtol=1e-4, atol=1e-5)

    def test_invalid_mode_and_depth(self):
        for kwargs in (dict(jk_mode="bad"), dict(num_layers=0)):
            with self.assertRaises(ValueError):
                JKNetModel(9, 16, 2, .1, "mean", **kwargs) if "num_layers" not in kwargs else JKNetModel(9, 16, 0, .1, "mean")

    def test_cache_memory_disk_and_csv_invalidation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            csv = root / "tiny.csv"
            csv.write_text("smiles,label\nCCO,0\nCCN,1\n")
            cache = DatasetCache(root / "cache")
            first, meta = cache.load_curated_csv(csv, ["label"])
            self.assertIs(first, cache.load_curated_csv(csv, ["label"])[0])
            self.assertEqual(cache.builds, 1)
            other = DatasetCache(root / "cache")
            with patch("src.data_cache.load_curated_csv", side_effect=AssertionError("Re-featurized cached data")):
                restored, _ = other.load_curated_csv(csv, ["label"])
            self.assertEqual(other.disk_loads, 1)
            for a, b in zip(first, restored):
                torch.testing.assert_close(a.x, b.x)
                torch.testing.assert_close(a.y, b.y)
                self.assertEqual(a.smiles, b.smiles)
            csv.write_text("smiles,label\nCCO,1\nCCN,1\n")
            changed, _ = cache.load_curated_csv(csv, ["label"])
            self.assertEqual(cache.builds, 2)
            self.assertEqual(changed[0].y.item(), 1)

    def test_mode_and_split_identity_selection(self):
        self.assertNotEqual(run_name("BBBP", "jknet", "mean", 0, jk_mode="max"),
                            run_name("BBBP", "jknet", "mean", 0))
        rows = []
        for model, sp, rep in product(("jknet", "jknet-max", "jknet-lstm"), (0., .25), (0, 1)):
            for layer, val, test in ((2, .6, .99), (3, .8, .51)):
                for seed in range(5):
                    rows.append(dict(dataset="BBBP", model=model, split="spectra", readout="mean",
                        sp=sp, split_idx=rep, num_layers=layer, dropout=.1, seed=seed,
                        val_auc=val, test_auc=test, best_epoch=2, task_type="binary", status="ok"))
        selected = select_configs(pd.DataFrame(rows), min_seeds=5)
        self.assertEqual(len(selected), 12)
        self.assertTrue(selected.num_layers.eq(3).all())
        self.assertTrue(np.allclose(selected.test_mean, .51))
        self.assertTrue(selected["n"].eq(5).all())
        self.assertTrue(select_configs(pd.DataFrame(rows).query("seed < 4"), min_seeds=5).empty)


if __name__ == "__main__":
    unittest.main()
