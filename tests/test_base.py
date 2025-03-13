from model2vec import StaticModel

model = StaticModel.from_pretrained("minishlab/potion-base-8M")

from simsity import create_index, load_index
from simsity.datasets import fetch_recipes

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
