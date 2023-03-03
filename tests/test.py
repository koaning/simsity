from embetter.text import SentenceEncoder

from simsity import create_index, load_index
from simsity.datasets import fetch_recipes

# Fetch data
df_recipes = fetch_recipes()
recipes = df_recipes['text']

# Create an indexer and encoder
encoder = SentenceEncoder()
index = create_index(recipes, encoder, "demo")
texts, dists = index.query("pork")
for text in texts:
    assert "prok" in text
assert index.index.element_count == 6118
