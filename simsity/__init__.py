from hnswlib import Index
from tinydb import TinyDB 
from BetterJSONStorage import BetterJSONStorage

from pathlib import Path 
import itertools as it
from typing import Iterable, Callable, Protocol, Union, Protocol

class Transformer(Protocol):
    def transform(self):
        pass

def encode(encoder, item):
    if hasattr(encoder, "transform"):
        return encoder.transform([item])[0]
    return encoder(item)


def create_index(data: Iterable, encoder: Union[Callable, Transformer], path: Path, dim:int, space="cosine"):
    batch = it.islice(data, 1)
    encoded = encode(encoder, batch)
    index = Index(space = space, dim = dim)
    index.init_index(max_elements = 10_000)
    with TinyDB(path, access_mode="r+", storage=BetterJSONStorage) as db:
        for i, item in enumerate(data):
            db.insert({'i': i, 'data': item})
            


def load_index():
    pass

class Index:
