build:
	python nbs/build.py

clean:
	rm -rf nbs/__pycache__

work:
	uv run marimo edit nbs/__init__.py

install:
	uv pip install -e . pytest marimo model2vec

pypi: check
	uv build
	uv publish

check: build
	pytest
