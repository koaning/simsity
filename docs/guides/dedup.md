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
from simsity.service import Service
from simsity.indexer import AnnoyIndexer
from dirty_cat import GapEncoder

# Set up
indexer = AnnoyIndexer(n_trees=50)
encoder = GapEncoder().fit(df)
service = Service(indexer=indexer, encoder=encoder)

# Index
service.index(df)
```

## Query

If we now want to construct a query, we will need to send a pandas row.
The encoder assumes pandas, so we need to make sure our query is compatible.

```python
# Query as a dictionary
dict_in = dict(name="khimerc thmas", suburb="chariotte", postcode="28273")
# Single row from dataframe
q_in = pd.DataFrame([dict_in]).iloc[0]

idx, dists = service.query(q_in, n_neighbors=10)
df.iloc[idx].assign(dist=dists)
```

This is the dataframe that we get out.

| name              | suburb      | postcode   |    dist |
|:------------------|:------------|:-----------|--------:|
| chimerc thmas     | chaflotte   | 28269      | 3.14833 |
| chimerc thomas    | charlotte   | 28269      | 3.95177 |
| khimerc thomas    | charlotte   | 2826g      | 3.98925 |
| angelique deas    | charlotte   | 28278      | 4.76251 |
| barbara dambrosio | charlotte   | 28277      | 5.46748 |
| kendel beachum    | charlotte   | 28226      | 5.6414  |
| mariq simpsony    | charlotte   | 28269      | 6.76645 |
| herber oxendine   | charlotte   | 28247      | 8.22691 |
| steven twamley    | chapel hill | 27514      | 8.97374 |
| herbert oxendin   | chsrlotte   | 28277      | 9.53476 |

It certainly seems like we have some duplicates in here! So
we may be able to use retreival/embedding tricks for that use-case.

It deserves mentioning, once again, that the quality of our
retreival depends a lot on our choice of index and encoding.
But experimenting with this is exactly what this library
makes easy.
