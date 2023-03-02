from hnswlib import Index
from tinydb import TinyDB 
from BetterJSONStorage import BetterJSONStorage

from pathlib import Path 
from typing import Iterable, Callable, Protocol, Union, Protocol

class Transformer(Protocol):
    def transform(self):
        pass

def encode(encoder, item):
    if hasattr(encoder, "transform"):
        return encoder.transform([item])[0]
    return encoder(item)


def create_index(data: Iterable, encoder: Union[Callable, Transformer], path: Path, space="cosine"):
    first_item = next(data)
    encoded = encode(encoder, first_item)
    index = Index(space = space, dim = encoded.shape[0])
    index.init_index(max_elements = 10_000)
    with TinyDB(path, access_mode="r+", storage=BetterJSONStorage) as db:
        for i, item in enumerate(data):
            db.insert({'i': i, 'data': item})
            


def load_index():
    pass

class Index:
