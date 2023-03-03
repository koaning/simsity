import datetime as dt
import srsly
from hnswlib import Index

from pathlib import Path 
from typing import Iterable, Protocol

class Transformer(Protocol):
    def transform(self):
        pass

class SimSityIndex:
    def __init__(self, path: Path, encoder: Transformer) -> None:
        self.path = Path(path)
        self.metadata = srsly.read_json(self.path / "metadata.json")
        self.index = Index(space = self.metadata['space'], dim = self.metadata['dim'])
        self.index.load_index(str(self.path / "index.bin"))
        self.db = {i: k['data'] for i, k in enumerate(srsly.read_jsonl(self.path / "db.jsonl"))}
        self.encoder = encoder
    
    def query(self, query, n=10):
        arr = self.encoder.transform(query)
        labels, distances = self.index.knn_query(arr, k=n)
        out = [self.db[int(label)] for label in labels[0]]
        return out, list(distances[0])

def create_index(data: Iterable, encoder: Transformer, path: Path, dim:int, space="cosine"):
    path = Path(path)
    index = Index(space = space, dim = dim)
    index.init_index(max_elements = len(data))
    encoded = encoder.transform(data)
    index.add_items(encoded)
    if (path / "db.jsonl").exists():
        (path / "db.jsonl").unlink()
    srsly.write_jsonl(path / "db.jsonl", ({'data': item} for i, item in enumerate(data)))
    index.save_index(str(path / "index.bin"))
    srsly.write_json(path / "metadata.json", {
        "created": str(dt.datetime.now())[:19],
        "dim": dim,
        "space": space
    })
    return SimSityIndex(path=path, encoder=encoder)

def load_index(path, encoder):
    return SimSityIndex(path=path, encoder=encoder)



from simsity.datasets import fetch_recipes

# Fetch data
df_recipes = fetch_recipes()
recipes = df_recipes['text']

# Create an indexer and encoder
from embetter.text import SentenceEncoder
encoder = SentenceEncoder()

dim = encoder.transform(["yo"]).shape[1]
index = create_index(recipes, encoder, "demoo", dim=dim)
    
index = load_index("demoo", encoder=encoder)
index.query("pork")
index.index.element_count