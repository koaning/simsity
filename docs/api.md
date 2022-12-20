# API Details

## Service

This object represents a nearest neighbor lookup service. You can
pass it an encoder and a method to index the data.

Arguments:
    - **encoder**: A scikit-learn compatible encoder for the input.
    - **indexer**: A compatible indexer for the nearest neighbor search.

## Indexer Features

The table below shows the features of each indexer.

| name                 | support sparse | support save | incremental_index |
|----------------------|----------------|--------------|-------------------|
| `AnnoyIndexer`       | no             | yes          | no                |
| `PynnDescentIndexer` | yes            | yes          | no                |

These can be loaded via:

```python
from simsity.datasets import fetch_clinc, fetch_recipe, fetch_voters
```

## Data Loaders

This library has a few data loaders so that you can play with data.

- `fetch_clinc` loads an intent detection dataset with 150+ intents
- `fetch_recipe` loads a dataset with recipes titles
- `fetch_voters` loads a dataset with voter information that contain typos

They can be loaded via:

```python
from simsity.datasets import fetch_clinc, fetch_recipe, fetch_voters
```
