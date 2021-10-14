import os

from joblib import dump
from pynndescent import NNDescent


class PyNNDescentIndexer:
    def __init__(self, metric="euclidean", n_neighbors=10) -> None:
        self.metric = metric
        self.n_neighbors = n_neighbors
        self.model = None
    
    def index(self, data):
        self.model = NNDescent(data, metric=self.metric, n_neighbors=self.n_neighbors)
        self.model.prepare()
    
    def query(self, query, n_neighbors=1):
        return self.model.query(query, n_neighbors)
    
    def save(self, folder_path):
        dump(self, os.path.join(folder_path, 'indexer.joblib'))
