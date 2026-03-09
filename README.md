# Neural Additive Models for Interpretable Credit Risk with Uncertainty Quantification

Empirical comparison of Neural Additive Models (NAMs) with conformal prediction against XGBoost+SHAP for credit risk scoring on the Taiwan Credit Card Default dataset.

## Quick Start

### Using uv (recommended)

```bash
# Install dependencies
uv sync

# Run full pipeline (with hyperparameter search)
uv run python scripts/run_pipeline.py

# Run with default config (skip search, faster)
uv run python scripts/run_pipeline.py --skip-search
```

### Using pip

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run full pipeline (with hyperparameter search)
python scripts/run_pipeline.py

# Run with default config (skip search, faster)
python scripts/run_pipeline.py --skip-search
```

## Project Structure

```
├── configs/default.yaml          # All hyperparameters and settings
├── src/
│   ├── data/                     # Download, preprocess, split
│   ├── models/                   # NAM and XGBoost implementations
│   ├── conformal/                # MAPIE wrapper and calibration metrics
│   ├── evaluation/               # Metrics, statistical tests, comparison tables
│   └── visualization/            # Shape functions, SHAP, calibration plots
├── notebooks/                    # Step-by-step Jupyter notebooks (01-05)
├── scripts/                      # Standalone training and pipeline scripts
├── tests/                        # Unit tests
└── results/                      # Output figures, tables, logs
```

## Dataset

Taiwan Credit Card Default (UCI ML Repository, ID 350): 30,000 instances, 23 features, binary target (default/no default).

## Running Tests

```bash
uv run pytest tests/ -v

# Or with pip
pytest tests/ -v
```
