from simsity.error import NotInstalled
from .pynn import PyNNDescentIndexer
from .common import Indexer


try:
    from .lshforest import MinHashIndexer
except ModuleNotFoundError:
    MinHashIndexer = NotInstalled("MinHashIndexer", "minhash")


__all__ = ["PyNNDescentIndexer", "MinHashIndexer", "Indexer"]
