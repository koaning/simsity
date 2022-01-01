"""
This demo is kept around so that we can confirm that optional dependencies
do not break the rest of the toolkit. This script only contains tools that
come with the base install of simsity.
"""

import pandas as pd

from simsity.service import Service
from simsity.indexer import PyNNDescentIndexer
from simsity.preprocessing import ColumnLister
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer


df = pd.read_csv("tests/data/clinc-data.csv").head(100)

# The encoder defines how we encode the data going in.
encoder = make_pipeline(ColumnLister(column="text"), CountVectorizer())

# The indexer handles the nearest neighbor lookup.
indexer = PyNNDescentIndexer(metric="euclidean")

# The service combines the two into a single object.
service = Service(indexer=indexer, encoder=encoder)

# We can now train the service using this data.
# Important for later: we're only passing the 'text' column to encode
service.train_from_dataf(df, features=["text"])
service.query(text="where is my phone", n_neighbors=3, out="dataframe")

# Save the entire system
service.save("/tmp/simple-pynn")

# You can also load the model now.
reloaded = Service.load("/tmp/simple-pynn")
