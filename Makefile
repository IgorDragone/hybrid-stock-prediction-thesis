.PHONY: db-build

db-build:
	python3.12 -m src.data.build_db
