<img src="docs/icon.png" width=150 height=150 align="right">

# simsity

> Simsity is a Super Simple Similarities Service[tm]. <br>
> It's all about building a neighborhood. Literally! <br>
> The reason why this library is hard to find is why it exists. 

This repository contains
simple tools to help in similarity retreival scenarios by making a convient
wrapper around encoding strategies as well as nearest neighbor approaches. 
Typical usecases include early stage bulk labelling and duplication discovery.

## Warning

Alpha software. Expect things to break. Do not use in production.

## Quickstart

This is the basic setup for this package.

```python
import pandas as pd

from simsity.service import Service
from simsity.indexer import PyNNDescentIndexer
from simsity.preprocessing import Identity, ColumnLister


# The Indexer handles the nearest neighbor search
# The Encoder handles the encoding of the datapoints
service = Service(
    indexer=PyNNDescentIndexer(metric="euclidean"),
    encoder=CountVectorizer()
)

# The encoder defines how we encode the data going in.
encoder = make_pipeline(
    ColumnLister(column="text"),
    CountVectorizer()
)

# The indexer handles the nearest neighbor lookup.
indexer = PyNNDescentIndexer(metric="euclidean", n_neighbors=2)

# The service combines the two into a single object.
service_clinc = Service(
    encoder=encoder,
    indexer=indexer,
)

# We can now train the service.
df_clinc = pd.read_csv("tests/data/clinc-data.csv")

# Important for later: we're only passing the 'text' column to encode
service_clinc.train_from_dataf(df_clinc, features=["text"])

# Query the datapoints
# Note that the keyword argument here refers to 'text'-column
service.query(text="give me directions", n_neighbors=20)
```

If you'd like you can also save and load the service on disk. 

```python
# Save the entire system
service.save("/tmp/simple-model")

# You can also load the model now.
reloaded = Service.load("/tmp/simple-model")
```

You could even run it as a webservice if you were so inclined. 

```python
reloaded.serve(host='0.0.0.0', port=8080)
```

You can now POST to http://0.0.0.0:8080/query with payload:

```
{"query": {"text": "hello there"}, "n_neighbors": 20}
```

Note that the query content here refers to `"text"`-column once again.
