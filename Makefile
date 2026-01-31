.PHONY: db-build, preprocess

db-build:
	python3.12 -m src.data.build_db

preprocess:
	python3.12 -m src.preprocessing.run_preprocessing