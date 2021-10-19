from datasketch import MinHashLSHForest, MinHash


class MinHashIndexer:
    """
    An indexer based on the MinHashLSHForest from datasketch.

    Arguments:
        num_perm: The number of permutations to build.
    """

    def __init__(self, num_perm=128) -> None:
        self.num_perm = num_perm
        self.forest = MinHashLSHForest(num_perm=num_perm)

    def index(self, data):
        """
        Index the given data.

        Arguments:
            data: The data to index.
        """
        for i, d in enumerate(data):
            self.forest.add(i, d)
        self.forest.index()

    def query(self, query: MinHash, n_neighbors=1):
        """
        Query the index.

        Arguments:
            query: The query to query the index with.
            n_neighbors: The number of neighbors to return.
        """
        idx = self.forest.query(query, k=n_neighbors)
        if len(idx) == 0:
            return list(range(n_neighbors)), [9999] * n_neighbors
        return idx, [0] * len(idx)
