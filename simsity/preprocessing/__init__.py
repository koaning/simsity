from simsity.error import NotInstalled

from ._identity import Identity
from ._columnlist import ColumnLister

try:
    from ._minhash import SparseMinHasher
except ModuleNotFoundError:
    MinHashIndexer = NotInstalled("SparseMinHasher", "minhash")


__all__ = ["Identity", "ColumnLister", "SparseMinHasher"]
