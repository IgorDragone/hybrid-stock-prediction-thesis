PY=python3.12
APP=src/app/streamlit_app.py

.PHONY: help db-build run-app test clean
.PHONY: lint format

help:
	@echo "make db-build   Build database"
	@echo "make run-app    Start Streamlit app"
	@echo "make test       Run pytest"
	@echo "make lint       Run ruff linter"
	@echo "make format     Run ruff formatter"
	@echo "make clean      Remove caches"

db-build:
	$(PY) -m src.db_buiding.build_db

run-app:
	streamlit run $(APP)

test:
	pytest -q

lint:
	ruff check .

format:
	ruff format .

clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache .ipynb_checkpoints
