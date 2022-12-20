import pytest
import pathlib
from mktestdocs import check_docstring, grab_code_blocks

from simsity.datasets import fetch_clinc, fetch_voters, fetch_recipes


@pytest.mark.parametrize(
    "func", [fetch_clinc, fetch_voters, fetch_recipes], ids=lambda d: d.__name__
)
def test_function_docstrings(func):
    """Test the docstring code of some functions."""
    check_docstring(obj=func)


@pytest.mark.parametrize(
    "fpath", ["README.md", "docs/guides/index.md", "docs/guides/dedup.md"]
)
def test_quickstart_docs_file(fpath):
    """Test the quickstart files."""
    grab_code_blocks(pathlib.Path(fpath).read_text())
