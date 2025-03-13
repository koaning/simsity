

import numpy as np
import polars as pl
from typing import Callable, List, Tuple, Any, Union, Optional, Sequence

def dot_product(query, matrix):
    return query @ matrix.T

class Index:
    """
    Index class for similarity search using vector embeddings.
    """
    def __init__(self, inputs: Sequence[Any], encoder: Callable, distance_func: Callable = dot_product):
        """
        Initialize an index with a sequence of inputs and encoder function.

        Args:
            inputs: Sequence of items to be indexed (e.g., list of strings)
            encoder: Function to convert items to embeddings (can be vectorized)
            distance_func: Function to compute distance between embeddings
        """
        self.encoder = encoder
        self.distance_func = distance_func

        # Generate embeddings for all inputs using vectorized encoding
        # The encoder is expected to handle a list of inputs
        embeddings = encoder(inputs)

        # Create polars DataFrame
        self.data = pl.DataFrame({
            "item": inputs,
            "embedding": embeddings
        })

    def to_disk(self, path: str = "simsity_index.parquet") -> None:
        """
        Save the index to disk as a parquet file.

        Args:
            path: Path to save the index
        """
        self.data.write_parquet(path)

    def query_vector(self, query_vector: np.ndarray, k: int = 5, return_index=False):
        """
        Query the index using a vector.

        Args:
            query_vector: Query vector
            k: Number of results to return

        Returns:
            Tuple of (items, distances)
        """
        embeddings = self.data["embedding"].to_numpy(allow_copy=False)        

        distances = self.distance_func(query_vector, embeddings)
        idx = np.argpartition(distances, -k)[-k:]
        idx = idx[np.argsort(distances[idx])[::-1]]    
        score = distances[idx]
        if return_index:
            return idx, score
        out = self.data.with_row_index().filter(pl.col("index").is_in(idx))["item"].to_list()
        return out, score

    def query(self, query_item: Any, k: int = 5, return_index = False) -> Tuple[List[Any], List[float]]:
        """
        Query the index using an item.

        Args:
            query_item: Item to query
            k: Number of results to return

        Returns:
            Tuple of (items, distances)
        """
        # Encode the query item (wrap in a list for vectorized encoders)
        query_vectors = self.encoder([query_item])
        query_vector = query_vectors[0]  # Extract the single result

        # Use query_vector function
        return self.query_vector(query_vector, k=k, return_index=return_index)


def create_index(inputs: Sequence[Any], encoder: Callable, distance_func: Callable = dot_product) -> Index:
    """
    Create an index from a sequence of inputs.

    Args:
        inputs: Sequence of items to index (e.g., list of strings)
        encoder: Function to convert items to embeddings (can be vectorized)
        distance_func: Function to compute distance between embeddings

    Returns:
        Index object
    """
    return Index(inputs, encoder, distance_func)

def load_index(path: str, encoder: Callable, distance_func: Callable = dot_product) -> Index:
    """
    Load an index from disk.

    Args:
        path: Path to the index file
        encoder: Function to convert items to embeddings
        distance_func: Function to compute distance between embeddings

    Returns:
        Index object
    """
    data = pl.read_parquet(path)

    # Create Index instance without inputs (will be replaced)
    index = Index([], encoder, distance_func)

    # Replace the data with loaded data
    index.data = data

    return index
