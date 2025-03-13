from model2vec import StaticModel

model = StaticModel.from_pretrained("minishlab/potion-base-8M")

from simsity import create_index, load_index
from simsity.datasets import fetch_recipes
import pytest

# Fetch data
df_recipes = fetch_recipes()
recipes = df_recipes["text"][:1000]



def check_output(texts):
    for text in texts:
        assert "pork" in text


def test_base_usage(tmpdir):
    # Make an index with a path
    index = create_index(recipes, model.encode)
    out1, _ = index.query("pork")
    check_output(out1)

    # Save an index to a path
    tmpfile = str(tmpdir / "index.parquet")
    index.to_disk(path=tmpfile)

    # Load an index from a path
    loader_index = load_index(path=tmpfile, encoder=model.encode)
    out2, _ = loader_index.query("pork")
    check_output(out2)
    assert out1 == out2

def test_top_k():
    index = create_index(recipes, model.encode)
    k_values = [1, 5, 10]
    for k in k_values:
        results, scores = index.query("pork", k=k)
        assert len(results) == k
        assert len(scores) == k
        # Check scores are in descending order (highest similarity first)
        assert all(scores[i] >= scores[i+1] for i in range(len(scores)-1))

def test_invalid_k():
    index = create_index(recipes, model.encode)
    with pytest.raises(ValueError):
        index.query("test", k=0)
    with pytest.raises(ValueError):
        index.query("test", k=-1)
    with pytest.raises(ValueError):
        index.query("test", k=len(recipes) + 1)  # k larger than dataset
