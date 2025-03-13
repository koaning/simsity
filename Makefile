build:
	python nbs/build.py

clean:
	rm -rf nbs/__pycache__

work:
	uv run marimo edit nbs/__init__.py
