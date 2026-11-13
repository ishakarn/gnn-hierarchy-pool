"""Reuse molecular graphs in memory; share curated-CSV preprocessing on disk."""
import hashlib
import json
from pathlib import Path
import uuid

from src.data import load_curated_csv, load_dataset


class DatasetCache:
    def __init__(self, cache_dir="./data/cache/graphs"):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.memory = {}
        self.disk_loads = 0
        self.builds = 0

    def load_dataset(self, name, root="./data"):
        key = ("moleculenet", name.upper(), str(Path(root).resolve()))
        if key not in self.memory:
            self.memory[key] = load_dataset(name, root=root)
            self.builds += 1
        return self.memory[key]

    def load_curated_csv(self, csv_path, label_cols):
        import rdkit
        import torch
        import torch_geometric
        csv_path = Path(csv_path).resolve()
        identity = dict(
            csv_sha256=hashlib.sha256(csv_path.read_bytes()).hexdigest(),
            label_cols=list(label_cols),
            featurizer_sha256=hashlib.sha256(
                (Path(__file__).parent / "data.py").read_bytes()).hexdigest(),
            torch=torch.__version__, pyg=torch_geometric.__version__,
            rdkit=rdkit.__version__)
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        if key in self.memory:
            return self.memory[key]
        path = self.cache_dir / f"{key}.pt" if self.cache_dir else None
        if path is not None and path.exists():
            # Trusted local artifact, keyed by input, featurizer and library versions.
            value = torch.load(path, map_location="cpu", weights_only=False)
            if value["identity"] != identity:
                raise ValueError(f"Graph cache identity mismatch: {path}")
            result = value["graphs"], value["meta"]
            self.disk_loads += 1
            print(f"[data_cache] disk load: {csv_path.name}", flush=True)
        else:
            result = load_curated_csv(str(csv_path), label_cols)
            self.builds += 1
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
                try:
                    torch.save(dict(identity=identity, graphs=result[0], meta=result[1]), temp)
                    temp.replace(path)
                finally:
                    temp.unlink(missing_ok=True)
            print(f"[data_cache] built once: {csv_path.name}", flush=True)
        self.memory[key] = result
        return result
