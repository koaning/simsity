<img src="docs/icon.png" width=150 height=150 align="right">

# simsity

> Simsity is a Super Simple Similarities Service[tm]. <br>
> It's all about building a neighborhood. Literally! <br>

<br> 

This repository contains simple tools to help in similarity retrieval scenarios
by making a convenient wrapper around encoding strategies as well as nearest neighbor
approaches. Typical usecases include early stage bulk labelling and duplication discovery.

## Install 

You can install simsity via pip. 

```
python -m pip install simsity
```

## Quickstart

This is the basic setup for this package.

```python
from simsity.service import Service
from simsity.datasets import fetch_clinc
from simsity.indexer import PyNNDescentIndexer
from simsity.preprocessing import Identity, ColumnLister

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer

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

# We can now train the service using this data.
df_clinc = fetch_clinc()

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

## Examples 

Check the `examples` folder for some interesting use-cases and tool integrations.

In particular: 

- [benchmark.ipynb](https://github.com/koaning/simsity/blob/main/examples/benchmark.ipynb) demonstrates an example on how you might benchmark simsity
- [votes-example.ipynb](https://github.com/koaning/simsity/blob/main/examples/votes-example.ipynb) demonstrates how to label similar data using pigeon and simsity
- [text-widget-example.ipynb](https://github.com/koaning/simsity/blob/main/examples/text-widget-example.ipynb) demonstrates how to add interactivity with ipywidgets
