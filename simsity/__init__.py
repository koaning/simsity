import datetime as dt
import itertools as it
from pathlib import Path
from typing import Iterable, Protocol

import srsly
from hnswlib import Index
from tqdm import tqdm


class Transformer(Protocol):
    def transform(self):
        pass


class SimSityIndex:
    def __init__(self, path: Path, encoder: Transformer) -> None:
        self.path = Path(path)
        self.metadata = srsly.read_json(self.path / "metadata.json")
        self.index = Index(space=self.metadata["space"], dim=self.metadata["dim"])
        self.index.load_index(str(self.path / "index.bin"))
        self.db = {
            i: k["data"] for i, k in enumerate(srsly.read_jsonl(self.path / "db.jsonl"))
        }
        self.encoder = encoder

    def query(self, query, n=10):
        arr = self.encoder.transform(query)
        labels, distances = self.index.knn_query(arr, k=n)
        out = [self.db[int(label)] for label in labels[0]]
        return out, list(distances[0])


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def create_index(
    data: Iterable, encoder: Transformer, path: Path, space="cosine", pbar=True, batch_size=500
):
    path = Path(path)
    index = None 
    batches = batch(data, batch_size)
    if pbar:
        batches, batches_copy = it.tee(batches)
        total = sum(1 for _ in batches_copy)
        batches = tqdm(batches, desc="indexing", total=total)
    for b in batches:
        encoded = encoder.transform(b)
        if not index:
            dim = encoded.shape[1]
            index = Index(space=space, dim=dim)
            index.init_index(max_elements=len(data))
        index.add_items(encoded)
    path.mkdir(parents=True, exist_ok=True)
    if (path / "db.jsonl").exists():
        (path / "db.jsonl").unlink()
    srsly.write_jsonl(
        path / "db.jsonl", ({"data": item} for i, item in enumerate(data))
    )
    index.save_index(str(path / "index.bin"))
    srsly.write_json(
        path / "metadata.json",
        {"created": str(dt.datetime.now())[:19], "dim": dim, "space": space},
    )
    return SimSityIndex(path=path, encoder=encoder)


def load_index(path, encoder):
    return SimSityIndex(path=path, encoder=encoder)
