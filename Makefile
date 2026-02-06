.PHONY: db-build

db-build:
	python3.12 -m src.data.build_db

run-app:
	streamlit run src/app/streamlit_app.py