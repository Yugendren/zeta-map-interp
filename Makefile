SHELL := /bin/bash

.PHONY: test datagen datagen-n11 datagen-n12 datagen-n13

test:
	.venv/bin/python -m unittest discover -s tests -p 'test_*.py'

datagen: datagen-n11 datagen-n12 datagen-n13

datagen-n11:
	.venv/bin/python -c "import sys; sys.path.insert(0, 'src'); from zetamap.dyck import generate_dataset; generate_dataset(11, 'data/dyck_n11.jsonl')"

datagen-n12:
	.venv/bin/python -c "import sys; sys.path.insert(0, 'src'); from zetamap.dyck import generate_dataset; generate_dataset(12, 'data/dyck_n12.jsonl')"

datagen-n13:
	.venv/bin/python -c "import sys; sys.path.insert(0, 'src'); from zetamap.dyck import generate_dataset; generate_dataset(13, 'data/dyck_n13.jsonl')"
