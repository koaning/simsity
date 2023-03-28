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
    """Object for easy querying."""

    def __init__(self, index, encoder, db) -> None:
        self.index = index
        self.encoder = encoder
        self.db = db

    def query(self, query, n=10):
        """
        Query using approximate nearest neighbors

        The object handles the encoder/data from disk.
        """
        arr = self.encoder.transform(query)
        return self.query_vector(query=arr, n=n)
    
    def query_vector(self, query, n=10):
        labels, distances = self.index.knn_query(query, k=n)
        out = [self.db[int(label)] for label in labels[0]]
        return out, list(distances[0])


def batch(iterable, n=1):
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx : min(ndx + n, length)]


def create_index(
    data: Iterable,
    encoder: Transformer,
    path: Path = None,
    space="cosine",
    pbar=True,
    batch_size=500,
):
    """
    Creates a simple ANN index. Uses hnswlib under the hood.
    You need to provide a scikit-learn compatible encoder for the data manually.
    """
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
    if path:
        path = Path(path)
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
    db = {i: k for i, k in enumerate(data)}
    return SimSityIndex(index=index, encoder=encoder, db=db)


def load_index(path, encoder):
    """Load in a simsity index from a path. Must supply same encoder."""
    path = Path(path)
    metadata = srsly.read_json(path / "metadata.json")
    index = Index(space=metadata["space"], dim=metadata["dim"])
    index.load_index(str(path / "index.bin"))
    db = {i: k for i, k in enumerate(srsly.read_jsonl(path / "db.jsonl"))}
    return SimSityIndex(index=index, encoder=encoder, db=db)
