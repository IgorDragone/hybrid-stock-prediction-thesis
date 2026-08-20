# Hybrid Stock Prediction Thesis

[![CI](https://github.com/IgorDragone/hybrid-stock-prediction-thesis/actions/workflows/ci.yml/badge.svg)](https://github.com/IgorDragone/hybrid-stock-prediction-thesis/actions/workflows/ci.yml)

An end-to-end equity ranking and portfolio construction project built for my final thesis in Computer Engineering at the University of La Laguna.

The project combines:
- technical features from price history
- fundamental features from company financial statements
- macroeconomic context from FRED
- walk-forward model evaluation and portfolio backtesting
- a Streamlit UI for model comparison and portfolio recommendations

The goal is not to predict exact prices, but to rank stocks cross-sectionally and test whether a hybrid feature set can generate better portfolio decisions than simpler baselines.

## What This Project Covers

This repository includes the full research workflow:
- data acquisition and database construction
- preprocessing and feature engineering
- leakage-aware target construction
- expanding walk-forward validation with embargo
- model training, backtesting, and robustness analysis
- model artifact storage for reuse in the UI
- an interactive application for model comparison and portfolio creation

It is designed as both:
- a thesis project with reproducible experimental notebooks
- a portfolio project that showcases practical data science, machine learning, and applied finance engineering

## Core Ideas

The modeling setup is cross-sectional and monthly:
- each month, stocks are scored and ranked
- the model predicts relative performance, not just direction
- portfolio decisions are derived from top-ranked names under explicit guardrails

The project currently includes:
- `baseline_mom`: simple momentum baseline
- `ridge`: linear benchmark with engineered features
- `hgb`: non-linear HistGradientBoosting model
- `buy_hold_eqw`: equal-weight benchmark

The research workflow compares:
- 3M vs 1M target horizons
- overlay on vs off
- fundamentals-only vs technicals-only vs combined features
- strategy performance vs SPY
- feature importance and rolling IC stability

## Repository Structure

```text
hybrid-stock-prediction-thesis/
├── configs/                  # universe and project configuration files
├── data/                     # raw and processed data folders
├── notebooks/                # thesis workflow notebooks
│   ├── 01_db_build.ipynb
│   ├── 02_preprocessing_pipeline.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_robustness.ipynb
│   ├── params.py
│   └── run_spec.yaml
├── src/
│   ├── app/                  # Streamlit UI
│   ├── config/               # paths, manifest/config I/O, universe loading
│   ├── db_building/          # prices, fundamentals, macro, technicals, DB assembly
│   ├── modeling/             # splits, baselines, models, backtests, registry
│   └── preprocessing/        # cleaning, feature engineering, target construction
├── tests/                    # focused regression tests for the pipeline
├── Makefile
├── requirements.txt
└── README.md
```

## Project Workflow

### 1. Database Build
`01_db_build.ipynb`

Builds a daily long-format panel by combining:
- adjusted prices
- technical indicators
- quarterly fundamentals with effective-date lagging
- macro series with publication-aware lagging
- sector metadata from the selected universe

Outputs are saved under:

```text
data/processed/datasets/<dataset_id>/stages/
```

### 2. Preprocessing Pipeline
`02_preprocessing_pipeline.ipynb`

Runs the full transformation pipeline:
- daily cleaning
- domain-aware clipping
- forward filling where appropriate
- end-of-month sampling
- percentile-ranked feature engineering
- target construction and model-ready panel creation

This notebook also writes:
- `config.yaml`: the run snapshot
- `manifest.json`: stage-by-stage dataset summary

### 3. Modeling and Backtesting
`03_modeling.ipynb`

Runs:
- expanding walk-forward splits
- baseline and ML model scoring
- top-K long-only backtests
- equal-weight benchmark comparison
- artifact saving to `data/processed/models/`

Saved model artifacts include:
- `config.json`
- `metrics.json`
- optional serialized model
- OOS scores for UI-side filtering and subset comparisons

### 4. Robustness and Diagnostics
`04_robustness.ipynb`

Analyzes:
- target horizon sensitivity
- macro overlay impact
- feature set ablations
- SPY benchmark comparison
- Ridge coefficients
- HGB permutation importance
- rolling 12M Information Coefficient
- macro regime and top-K composition analysis

## Methodology Highlights

### Leakage Control

The project explicitly handles leakage through:
- effective-date lagging of quarterly fundamentals
- lagged macro alignment
- end-of-month panel construction
- forward-return target construction
- expanding walk-forward validation with embargo

### Modeling Target

The main target is cross-sectional excess performance:
- forward returns are computed at 1M, 3M, and 6M horizons
- the main supervised target is the demeaned 3M forward return
- the model learns relative ranking strength rather than absolute market direction

### Backtesting Logic

The backtest is monthly and long-only:
- rank the universe by model score
- select the top `K`
- optionally scale exposure down under high macro stress
- track return, CAGR, Sharpe, max drawdown, turnover, and hit rate

## Streamlit Application

The UI in `src/app/` provides:
- model comparison
- metrics and equity curve visualization
- subset-based comparisons on user-selected tickers
- portfolio recommendation generation from saved models
- basic portfolio persistence inside the app

Run it with:

```bash
make run-app
```

or:

```bash
streamlit run src/app/streamlit_app.py
```

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/IgorDragone/hybrid-stock-prediction-thesis.git
cd hybrid-stock-prediction-thesis
```

### 2. Create and activate a virtual environment

```bash
python3.12 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Alpha Vantage

Fundamental-data downloads require a free Alpha Vantage API key. Copy the example file and replace the placeholder locally:

```bash
cp .env.example .env
set -a
source .env
set +a
```

The `.env` file is ignored by Git and must never be committed. The application reads `ALPHAVANTAGE_API_KEY` from the process environment, so load the file in each new shell session or export the variable through your preferred environment manager.

## Typical Commands

Run tests:

```bash
make test
```

Lint the project:

```bash
make lint
```

Format the project:

```bash
make format
```

## Data and Artifacts

Raw market and fundamentals data are not intended to be fully tracked in GitHub.

In general:
- source code and configuration are versioned
- large processed artifacts are kept out of the repository
- small metadata files such as dataset configs and manifests are kept for reproducibility

Model artifacts in `data/processed/models/` are handled selectively:
- lightweight metadata is tracked
- large derived artifacts are not

## Tests

The `tests/` folder focuses on pipeline correctness rather than generic unit test coverage. Current tests cover key assumptions such as:
- effective-date handling for fundamentals
- end-of-month sampling
- percentile-rank feature behavior
- target sanity
- walk-forward split constraints
- backtest mechanics

## Tech Stack

- Python
- pandas
- NumPy
- scikit-learn
- SciPy
- matplotlib
- seaborn
- Streamlit
- yfinance
- PyYAML
- pytest
- Ruff

## Why This Project Matters

This repository is meant to demonstrate the combination of:
- data engineering for messy financial data
- leakage-aware ML experimentation
- model evaluation beyond standard predictive metrics
- practical backtesting logic
- reproducible research workflows
- product-oriented delivery through a lightweight UI

It sits at the intersection of:
- data science
- machine learning
- quantitative finance
- applied AI engineering

## Author

**Igor Dragone**  
Computer Engineering, University of La Laguna  
Thesis project, 2025/2026

## Disclaimer

This project is for academic and educational purposes only. Its outputs are not financial advice, investment recommendations, or a guarantee of future performance. Backtest results are hypothetical and may not reflect transaction costs, taxes, liquidity constraints, or live-market execution.
