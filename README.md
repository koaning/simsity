
![landing](https://user-images.githubusercontent.com/1019791/222645884-fd88cd66-3dd0-4b6e-98f4-65586040e538.png)

# simsity

> Simsity is a Super Simple Similarities Service[tm]. <br>
> It's all about building a neighborhood. Literally! <br>

This simple library is partially inspired by [this blogpost by Max Woolfe](https://minimaxir.com/2025/02/embeddings-parquet/). You don't always need a full fledged vector database. Polars and numpy might be all you need. And for those moments, `simsity` is all you need to build a neighborhood!

## Install

You can install simsity via pip.

```
uv pip install simsity
```

The goal of simsity is to be minimal, to make rapid prototyping very easy and to be "just enough" for medium sized datasets. You will mainly interact with these two functions. 

```python
from simsity import create_index, load_index
```

As their names imply, you can use these to create an index or to load one from disk. 

## Quickstart

```python
from simsity import create_index, load_index
from simsity.datasets import fetch_recipes

# Let's fetch some demo data
recipes = fetch_recipes()["text"].to_list()

# Let's use model2vec for embeddings 
from model2vec import StaticModel
model = StaticModel.from_pretrained("minishlab/potion-base-8M")

# Populate the ANN vector index and use it. 
index = create_index(recipes, model.encode)
texts, dists = index.query("pork")

# You can also query using vectors
v_pork = model.encode(["pork"])[0]
texts, dists = index.query_vector(v_pork)
```

You can also provide a path and then you'll be able to store/load everything.

```python
# Make an index with a path
index = create_index(recipes, encoder, path="demo")

# Load an index from a path
reloaded_index = load_index(path="demo", encoder=encoder)
texts, dists = reloaded_index.query("pork")
```

That's it! Happy hacking!
