from simsity.error import NotInstalled
from .annoy import AnnoyIndexer
from .common import Indexer

try:
    from .pynn import PyNNDescentIndexer
except ModuleNotFoundError:
    PyNNDescentIndexer = NotInstalled("PyNNDescentIndexer", "pynn")

try:
    from .nmslib import NMSlibIndexer
except ModuleNotFoundError:
    NMSlibIndexer = NotInstalled("NMSlibIndexer", "nms")


__all__ = ["PyNNDescentIndexer", "Indexer", "AnnoyIndexer", "NMSlibIndexer"]
