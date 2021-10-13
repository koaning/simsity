import json
import pathlib 

import pandas as pd 
from joblib import dump, load


class Service:
    def __init__(self, encoder, indexer, storage={}) -> None:
        self.encoder = encoder
        self.indexer = indexer
        self.storage = storage
        self.__trained = False
    
    def train_from_csv(self, path, text_col="text"):
        df = pd.read_csv(path)
        texts = list(df[text_col])
        for text in texts:
            self.storage[len(self.storage) + 1] = {"text": text}
        data = self.encoder.fit_transform(texts)
        self.indexer.index(data)
        self.__trained = True
        return self

    def query(self, text, n_neighbors=10):
        data = self.encoder.transform([text])
        idx, dist = self.indexer.query(data, n_neighbors=n_neighbors)
        return [{'text': self.storage[idx[0][i]]['text'], 'dist': dist[0][i]} for i in range(len(idx[0]))]
    
    def save(self, path):
        if not self.__trained:
            raise Exception("Cannot save, Service is not trained.")
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)
        storage_path = pathlib.Path(path) / "storage.json"
        storage_path.write_text(json.dumps(self.storage))
        dump(self.encoder, pathlib.Path(path) / "encoder.joblib")
        dump(self.indexer, pathlib.Path(path) / "indexer.joblib")

    @classmethod
    def load(cls, path):
        if not pathlib.Path(path).exists():
            raise FileNotFoundError(f"{path} does not exist")
        storage_path = pathlib.Path(path) / "storage.json"
        storage = {int(k): v for k, v in json.loads(storage_path.read_text()).items()}
        encoder = load(pathlib.Path(path) / "encoder.joblib")
        decoder = load(pathlib.Path(path) / "indexer.joblib")
        return cls(encoder, decoder, storage)
