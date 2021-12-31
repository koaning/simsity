from typing import Tuple, List
from abc import ABC, abstractmethod


class Indexer(ABC):
    """
    An indexer ABC.
    """

    @abstractmethod
    def index(self, data) -> None:
        """
        Index the given data.

        Arguments:
            data: The data to index.
        """
        pass

    @abstractmethod
    def save(self, path) -> None:
        """
        Save the indexer in a path.

        Arguments:
            path: string or pathlib.Path to folder.
        """
        pass

    @abstractmethod
    def load(self, path) -> "Indexer":
        """
        Load the indexer in a path.

        Arguments:
            path: string or pathlib.Path to folder.
        """
        pass

    def query(self, query, n_neighbors=1) -> Tuple[List, List]:
        """
        Query the index.

        Arguments:
            query: The query to query the index with.
            n_neighbors: The number of neighbors to return.
        """
        if not self.model:
            raise RuntimeError("Index not yet built.")
        idx, dist = self.model.query(query, n_neighbors)
        return list(idx[0]), list(dist[0])
