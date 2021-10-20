An interesting use-case for simsity is to use it as a tool that
explores deduplication.

## Example

Let's consider the `voters` dataset.

```python
from simsity.datasets import fetch_voters

df = fetch_voters()
```

| name               | suburb     | postcode   |
|:-------------------|:-----------|:-----------|
| khimerc thomas     | charlotte  | 2826g      |
| lucille richardst  | kannapolis | 28o81      |
| reb3cca bauerboand | raleigh    | 27615      |
| maleda mccloud     | goldsboro  | 2753o      |
| belida stovall     | morrisvill | 27560      |

This dataset contains information about "voters" and the concern is that
some of these rows may represent the same person. The persons name might occur
in different spellings and the postcodes may contain typos, but they could
still refer to the same person. In other words; there may be duplicates in this
dataframe that we cannot remove with `.drop_duplicates()`. So how might we go
about finding these?

## Similarity Service!

Let's build a similarity service, but now we'll use encoders from
the [dirty_cat](https://dirty-cat.github.io/stable/) package. These
encoders are designed to handle dirty categorical data, which would
be perfect for our use-case here.

```python
service = Service(
    indexer=PyNNDescentIndexer(metric="euclidean", n_jobs=10),
    encoder=GapEncoder()
)

service.train_from_dataf(df)
```

Note that in this example, we've not specified any `features=`-parameters
in our `service.train_from_dataf` call. In this case the service will assume
all columns in the dataframe are relevant to encode.

## Query

If we now want to construct a query, we will need to use all columns that
we've encoded in our query call. That means that we need to query with a
`"name"`, `"suburb"` and `"postcode"` keyword argument.

```python
service.query(name="khimerc thmas",
              suburb="chariotte",
              postcode="28273",
              n_neighbors=10, out="dataframe")
```

This is the dataframe that we get out.

| name            | suburb    | postcode   |    dist |
|:----------------|:----------|:-----------|--------:|
| chimerc thmas   | chaflotte | 28269      | 3.43277 |
| quianna pope    | charlotte | 28213      | 3.65635 |
| chimerc thomas  | charlotte | 28269      | 3.93795 |
| khimerc thomas  | charlotte | 2826g      | 3.99429 |
| quianha pope    | charlotre | 28213      | 5.47086 |
| kendel beachum  | charlotte | 28226      | 6.22856 |
| mariq simpsony  | charlotte | 28269      | 6.36486 |
| quiarina pope   | charlotte | 28113      | 6.57162 |
| andrean polchow | waxhaw    | 28173      | 7.21047 |
| maria simpson   | charlotte | 28269      | 8.2501  |

It certainly seems like the first few rows may indeed contain
duplicates that we're interested in detecting.

It deserves mentioning, once again, that the quality of our
retreival depends a lot on our choice of index and encoding.
But experimenting with this is exactly what this library
makes easy.

## HTTP

As always, you can easily turn this service into an API.

```python
service.serve(host='0.0.0.0', port=8080)
```

But also here you'd need to mind the parameters that you send
to the server. They need to correspond with the column names
in the dataframe. This would be an appropriate payload for `https://0.0.0.0:8080/query`.

```
{
    "query": {
        "name": "khimerc thmas",
        "suburb": "chariotte",
        "postcode": "28273
    },
    "n_neighbors": 5
}
```
