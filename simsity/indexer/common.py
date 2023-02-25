from typing import Tuple, List
from abc import ABC, abstractmethod, abstractclassmethod


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

    @abstractclassmethod
    def load(self, path) -> "Indexer":
        """
        Load the indexer in a path.

        Arguments:
            path: string or pathlib.Path to folder.
        """
        pass

    def query(self, item, n_neighbors=1) -> Tuple[List, List]:
        """
        Query the index.

        Arguments:
            query: The query to query the index with.
            n_neighbors: The number of neighbors to return.
        """
        pass
