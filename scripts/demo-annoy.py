from simsity.service import Service
from simsity.datasets import fetch_clinc
from simsity.indexer import PyNNDescentIndexer
from simsity.preprocessing import ColumnLister

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD

# The encoder defines how we encode the data going in.
encoder = make_pipeline(
    ColumnLister(column="text"), CountVectorizer(), TruncatedSVD(n_components=10)
)

# The indexer handles the nearest neighbor lookup.
# indexer = AnnoyIndexer(metric="euclidean", n_trees=10)
indexer = PyNNDescentIndexer()

# The service combines the two into a single object.
service_clinc = Service(
    encoder=encoder,
    indexer=indexer,
)

# We can now train the service using this data.
df_clinc = fetch_clinc()

# Important for later: we're only passing the 'text' column to encode
service_clinc.train_from_dataf(df_clinc, features=["text"])

# Save the entire system
service_clinc.save("/tmp/simple-model")

# You can also load the model now.
reloaded = Service.load("/tmp/simple-model")
