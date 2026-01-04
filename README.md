# Financial Behavior Risk Analysis Pipeline

## 📋 Overview

A comprehensive, reproducible ML pipeline for financial behavior risk assessment. Processes survey data to predict financial risk levels using clean baselines and optional (quality-gated) synthetic augmentation.

### Key Features
- **Modular Architecture** - Separate packages for config, data, models, CV
- **Config Validation** - Schema validation with forbidden key detection
- **Data Integrity Checks** - NaN/infinite value detection, mutual exclusivity validation
- **Quality Gates** - Synthetic data must pass 4 mandatory tests before use
- **Dataset Fingerprinting** - Hash-based tracking for reproducibility
- **Comprehensive Test Suite** - Pytest-based validation

---

## ⚠️ CRITICAL RULES (NON-NEGOTIABLE)

### Forbidden Targets
**`Behavior_Risk_Level` is FORBIDDEN** - Circular label derived from features.

### Allowed Targets (LOCKED)
| Target | Type | Use Case |
|--------|------|----------|
| `Risk_Score` | Continuous | **Primary** - Regression |
| `Save_Money_Yes` | Binary | **Secondary** - Classification |

### Hard Removals (FOREVER)
- ❌ Retrain loops on enriched data
- ❌ Exponential dataset growth (`max_size`, `target_total`)
- ❌ Final 50/50 balancing
- ❌ GAN retraining on synthetic output
- ❌ Iterative augmentation patterns

---

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt

# For testing
pip install pytest pytest-cov
```

### Run Baseline Experiments (Recommended)

```bash
# Regression baseline (Risk_Score) - PRIMARY
python run_baseline.py --config configs/baseline_regression.yaml --dataset data.csv

# Classification baseline (Save_Money_Yes) - SECONDARY
python run_baseline.py --config configs/baseline_classification.yaml --dataset data.csv

# Run ALL baselines at once
python run_baseline.py --all-baselines --dataset data.csv
```

### Run Augmentation Experiment (Quality-Gated)

```bash
python run_augmentation_experiment.py --config configs/augmentation_experiment.yaml --dataset data.csv
```

### Run Tests

```bash
pytest -v                          # All tests
pytest tests/test_config.py -v     # Specific file
pytest --cov=experiments --cov-report=term-missing  # With coverage
```

---

## 🗂️ Project Structure

```
financial_behavior_risk/
├── run_baseline.py                # Baseline experiments (augmentation DISABLED)
├── run_augmentation_experiment.py # Augmentation experiments (quality-gated)
├── run_experiment.py              # DEPRECATED - delegates to run_baseline.py
│
├── experiments/                   # Modular experiment pipeline
│   ├── __init__.py               
│   ├── config_schema.py           # Config validation + forbidden keys
│   ├── io.py                      # Load/save, run directories, data profiles
│   ├── data.py                    # Dataset loading, preprocessing, validation
│   ├── models.py                  # Model building (Ridge, XGB, etc.)
│   └── cv.py                      # Cross-validation with integrity checks
│
├── configs/                       # Experiment configurations
│   ├── baseline_regression.yaml   # Risk_Score (PRIMARY)
│   ├── baseline_classification.yaml # Save_Money_Yes (SECONDARY)
│   └── augmentation_experiment.yaml # Synthetic testing
│
├── tests/                         # Pytest test suite
│   ├── conftest.py                # Fixtures
│   ├── test_config.py             # Config validation tests
│   ├── test_data_integrity.py     # Data preprocessing tests
│   ├── test_cv_and_leakage.py     # CV integrity tests
│   ├── test_augmentation_policy.py # Policy enforcement tests
│   ├── test_reproducibility.py    # Reproducibility tests
│   └── test_artifacts.py          # Output artifact tests
│
├── DataAugmentation/              # Synthetic data generation
│   ├── quality_gates.py           # 4 mandatory quality gates
│   ├── cluster_enrichment.py      # Cluster-aware enrichment
│   └── ...
│
├── FirstProcessing/               # Raw data preprocessing
├── EDA/                           # Exploratory data analysis
└── runs/                          # Experiment outputs
```

---

## 🔒 Config Validation

The pipeline validates configs at startup and **raises errors** (not silent fixes):

### Required Keys
```yaml
experiment:
  name: string
  seed: integer
data:
  target_column: string
  target_type: regression|classification
model:
  type: ridge|lasso|xgboost_reg|logistic_regression|...
cross_validation:
  n_splits: integer >= 2
```

### Forbidden Keys (Will Error)
```yaml
# These patterns are BLOCKED:
max_size: ...          # Iterative growth
target_total: ...      # Forced balancing
iterative: ...         # Iterative augmentation
iterative_growth: ...
retrain_on_synthetic: ...
final_balancing: ...
```

### Baseline vs Augmentation Mode
```bash
# Baseline mode: augmentation.enabled MUST be false
# If true, raises ConfigValidationError (not silent disable)
python run_baseline.py --config configs/with_aug_enabled.yaml
# → ERROR: "Augmentation is enabled in a baseline config..."

# Augmentation mode: augmentation.enabled can be true
python run_augmentation_experiment.py --config configs/augmentation_experiment.yaml
# → OK (with quality gates)
```

---

## 📊 Output per Run

Each run creates a timestamped folder in `runs/`:

| File | Description |
|------|-------------|
| `config.yaml` | Experiment configuration |
| `metrics.json` | CV metrics with **fold-level scores** |
| `model.joblib` | Trained model |
| `data_profile.json` | **Dataset fingerprint** (hash, rows, features) |
| `cv_distribution.png` | Score distribution plot |

### metrics.json Format
```json
{
  "cv_results": {
    "mae": {
      "mean": 0.1234,
      "std": 0.0123,
      "all": [0.12, 0.13, 0.11, ...]  // Fold-level scores for CI/boxplots
    }
  }
}
```

### data_profile.json Format
```json
{
  "dataset_path": "data/survey.csv",
  "dataset_hash": "a1b2c3d4e5f6",
  "total_rows": 500,
  "feature_count": 45,
  "features_used": ["Debt_Level", "Impulse_Buying_Frequency", ...],
  "target_column": "Risk_Score",
  "timestamp": "2026-01-04T12:34:56"
}
```

---

## 🔬 Quality Gates for Synthetic Data

Synthetic data must pass **ALL 4 gates** or is rejected:

| Gate | Description | Threshold |
|------|-------------|-----------|
| **Memorization** | No near-duplicates of real samples | < 5% duplicates |
| **Two-Sample** | Discriminator can't distinguish real/synthetic | AUC < 0.75 |
| **Utility** | Augmented performance ≥ real-only | Not worse by > 2% |
| **Stability** | Variance not increased | < 20% increase |

### Synthetic Ratio Limits
- Minimum: 15%
- Maximum: 30%
- **Never** exceed 50%

---

## 🧪 Data Integrity Checks

The pipeline enforces these at runtime:

1. **NaN Detection** - Errors if features or target contain NaN
2. **Infinite Values** - Errors if numeric columns have ±∞
3. **Save_Money Consistency** - Mutual exclusivity (Yes=1 ⟹ No=0)
4. **CV Split Integrity** - Train/val indices are disjoint
5. **Forbidden Features** - Behavior_Risk_Level blocked from features

---

## 📋 Validation Protocol

- **Repeated K-Fold CV:** 5 folds × 10 repeats = 50 evaluations
- **Regression metrics:** MAE, RMSE, Spearman, R²
- **Classification metrics:** Macro-F1, Accuracy, Precision, Recall
- **Stratification:** Used only for classification (enforced)

---

## 🔄 Migration from Old Scripts

### Old (Deprecated)
```bash
python run_experiment.py --config configs/baseline.yaml
# Shows deprecation warning, delegates to run_baseline.py
```

### New (Recommended)
```bash
# For baselines
python run_baseline.py --config configs/baseline_regression.yaml

# For augmentation experiments
python run_augmentation_experiment.py --config configs/augmentation_experiment.yaml
```

---

## 📝 Changelog

### v2.0 (January 2026) - Modular Architecture
- **Split entrypoints**: `run_baseline.py` vs `run_augmentation_experiment.py`
- **Config validation**: Schema checking with forbidden key detection
- **Modular packages**: `experiments/` with io, data, models, cv modules
- **Data integrity**: NaN/infinite checks, Save_Money consistency
- **Dataset fingerprinting**: Hash + metadata in `data_profile.json`
- **Fold-level scores**: Full scores saved in `metrics.json` for CI/boxplots
- **CV integrity**: Train/val disjoint assertion in runtime

### v1.0 (January 2026) - Clean Baselines
- Quality gates for synthetic data
- Forbidden target blocking
- Repeated K-Fold CV

---

## ⚠️ Known Limitations

- **Old/ folder**: Contains deprecated experiments - DO NOT USE
- **Ridge/Lasso**: No random_state (deterministic solvers)
- **Folder name**: Has space ("Procesare Dataset") - consider renaming for CI

---

## 🎯 Scientific Validity

1. **Deterministic seeds** - All randomness is seeded
2. **No circular labels** - Behavior_Risk_Level forbidden
3. **Explicit errors** - Config issues raise, never silently fixed
4. **Quality gates** - Synthetic must prove utility
5. **Dataset fingerprinting** - Know exactly what data was used
6. **Fold-level metrics** - Full distribution available for analysis
7. **Test coverage** - Automated verification of all invariants

