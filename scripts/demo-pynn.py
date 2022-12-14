"""
This demo is kept around so that we can confirm that optional dependencies
do not break the rest of the toolkit. This script only contains tools that
come with the base install of simsity.
"""

import pandas as pd

from simsity.service import Service
from simsity.indexer import PyNNDescentIndexer
from simsity.preprocessing import ColumnGrabber
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer


df = pd.read_csv("tests/data/clinc-data.csv").head(100)

# The encoder defines how we encode the data going in.
encoder = make_pipeline(ColumnGrabber(column="text"), CountVectorizer())
encoder.fit(df)

# The indexer handles the nearest neighbor lookup.
indexer = PyNNDescentIndexer(metric="euclidean")

# The service combines the two into a single object.
service = Service(indexer=indexer, encoder=encoder)

# We can now build the service using this data.
service.index(df)

# And use it
idx, dists = service.query(df.iloc[2], n_neighbors=3)