black:
	black simsity tests setup.py
	black simsity tests setup.py --check

flake:
	flake8 simsity tests setup.py

test:
	python tests/test.py

install:
	python -m pip install -e ".[dev]"
	python -m pip install black flake8 interrogate pyright

pypi:
	python setup.py sdist
	python setup.py bdist_wheel --universal
	twine upload dist/*

pyright:
	pyright simsity tests
	
clean:
	rm -rf **/.ipynb_checkpoints **/.pytest_cache **/__pycache__ **/**/__pycache__ .ipynb_checkpoints .pytest_cache

check: clean black flake pyright test clean
