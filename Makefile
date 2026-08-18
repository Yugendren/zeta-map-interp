SHELL := /bin/bash

.PHONY: test datagen

test:
	.venv/bin/python -m unittest discover -s tests -p 'test_*.py'

datagen:
	.venv/bin/python -c "import sys; sys.path.insert(0, 'src'); from zetamap.dyck import generate_dataset; generate_dataset(11, 'data/dyck_n11.jsonl')"
