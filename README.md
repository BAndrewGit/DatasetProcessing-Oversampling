# Dataset Processing Pipeline - Financial Behavior Risk Analysis

## 📋 Project Overview

This project implements a comprehensive data processing and analysis pipeline for **financial behavior risk assessment**. The system processes survey data about financial habits, spending patterns, and economic behaviors to predict and analyze financial risk levels.

### Key Features
- **Data Preprocessing & Normalization** - Translation, encoding, and feature engineering
- **Advanced Exploratory Data Analysis (EDA)** - Statistical analysis with multiple visualization techniques
- **Dimensionality Reduction** - PCA analysis for feature extraction
- **Clustering Analysis** - K-Means and GMM clustering for behavioral segmentation
- **Reproducible Experiments** - Single entrypoint with YAML configs for consistent results
- **Synthetic Data Quality Gates** - Rigorous validation before using any synthetic data
- **Comprehensive Test Suite** - Pytest-based tests for reproducibility verification

---

## ⚠️ CRITICAL: Clean Baseline Rules (NON-NEGOTIABLE)

### Forbidden Targets
**`Behavior_Risk_Level` is FORBIDDEN as a training target** - This is a circular label derived from features.

### Allowed Targets (LOCKED)
| Target | Type | Use Case |
|--------|------|----------|
| `Risk_Score` | Continuous | **Primary** - Regression |
| `Save_Money_Yes` | Binary | **Secondary** - Classification |

### Hard Removals (FOREVER)
- Retrain loops on enriched data
- Exponential dataset growth
- Final 50/50 balancing
- GAN retraining on synthetic output

---

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or minimal dependencies
pip install -r requirements_minimal.txt
```

### Run Clean Baselines

```bash
# Run regression baseline (Risk_Score target) - PRIMARY
python run_experiment.py --config configs/baseline_regression.yaml --dataset path/to/data.csv

# Run classification baseline (Save_Money target) - SECONDARY
python run_experiment.py --config configs/baseline_classification.yaml --dataset path/to/data.csv

# Run ALL baselines at once
python run_experiment.py --all-baselines --dataset path/to/data.csv
```

### Run Augmentation Experiment (Tests if synthetic helps)

```bash
# Test whether synthetic data improves real-only performance
python augmentation_experiment.py --config configs/augmentation_experiment.yaml --dataset path/to/data.csv
```

### Run Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest -v

# Run with coverage
pytest --cov=. --cov-report=html
```

---

## 📊 Output per Run

Each experiment run produces a folder in `runs/` containing:

| File | Description |
|------|-------------|
| `config.yaml` | Experiment configuration used |
| `metrics.json` | All CV metrics (mean ± std) |
| `model.joblib` | Trained model (sklearn format) |
| `cv_distribution.png` | Cross-validation score distribution |

### Validation Protocol
- **Repeated K-Fold CV:** 5 folds × 10 repeats = 50 evaluations
- **Regression metrics:** MAE, RMSE, Spearman correlation, R²
- **Classification metrics:** Macro-F1, Accuracy, Precision, Recall

---

## 🗂️ Project Structure

```
Procesare Dataset/
├── run_experiment.py              # MAIN ENTRYPOINT - reproducible experiments
├── augmentation_experiment.py     # Controlled synthetic augmentation testing
├── pytest.ini                     # Pytest configuration
│
├── configs/                       # Experiment configurations (YAML)
│   ├── baseline_regression.yaml         # Risk_Score regression (PRIMARY)
│   ├── baseline_classification.yaml     # Save_Money classification (SECONDARY)
│   ├── augmentation_experiment.yaml     # Synthetic augmentation testing
│   ├── default.yaml                     # Default config template
│   └── smote_experiment.yaml            # SMOTE augmentation config
│
├── tests/                         # Test suite (pytest)
│   ├── conftest.py                      # Fixtures and test utilities
│   ├── test_config.py                   # Config validation tests
│   ├── test_data_integrity.py           # Data preprocessing tests
│   ├── test_cv_and_leakage.py           # CV and leakage detection tests
│   ├── test_augmentation_policy.py      # Augmentation policy tests
│   ├── test_reproducibility.py          # Reproducibility verification
│   └── test_artifacts.py                # Output artifact tests
│
├── runs/                          # Output folder for experiment runs
│
├── FirstProcessing/               # Initial data processing pipeline
│   ├── main.py                          # Entry point for preprocessing
│   ├── preprocessing.py                 # Data normalization (RO→EN translation)
│   ├── risk_calculation.py              # Risk scoring and clustering
│   ├── encoder.py                       # Feature encoding utilities
│   ├── data_generation.py               # Feature engineering
│   └── file_operations.py               # File I/O and Excel formatting
│
├── EDA/                           # Exploratory Data Analysis
│   ├── V1/                              # Basic EDA (legacy)
│   │   ├── mainEDA.py
│   │   ├── data_loading.py
│   │   ├── preprocessing.py
│   │   ├── visualization.py
│   │   └── model_training.py
│   │
│   └── V2/                              # Advanced EDA (CURRENT)
│       ├── mainEDA2.py                  # Main workflow with PCA + clustering
│       ├── config.py                    # Configuration settings
│       ├── data_loader.py               # Data loading and preparation
│       ├── plot_generator.py            # Comprehensive plotting
│       ├── utils.py                     # Utility functions
│       ├── PCA/                         # Principal Component Analysis
│       │   ├── pca_transformer.py
│       │   └── pca_visualizer.py
│       └── clustering/                  # Clustering analysis
│           ├── kmeans_clustering.py
│           ├── gmm_clustering.py
│           ├── cluster_comparison.py
│           └── cluster_visualizer.py
│
├── DataAugmentation/              # Synthetic data generation
│   ├── __init__.py                      # Module exports
│   ├── base.py                          # Base augmentation class
│   ├── quality_gates.py                 # Synthetic data quality gates
│   ├── cluster_enrichment.py            # Cluster-aware enrichment
│   ├── smote_tomek.py                   # SMOTE-Tomek (DEPRECATED)
│   ├── CTGan_Augmentation.py            # CTGAN augmentation
│   └── WC_GAN.py                        # Wasserstein GAN
│
├── Old/                           # Deprecated experiments (DO NOT USE)
│
├── scaler/                        # Saved preprocessing models
│   └── robust_scaler.pkl
│
├── requirements.txt               # Full Python dependencies
└── requirements_minimal.txt       # Minimal dependencies
```

---

## 🔬 Synthetic Data Quality Gates (Sprint 2)

Before any synthetic data is used, it must pass **ALL** quality gates:

### Gate 1: Memorization Test
Synthetic samples must not be near-duplicates of real samples.

### Gate 2: Two-Sample Test
A classifier trying to distinguish real vs synthetic must have AUC < 0.75.

### Gate 3: Utility Test
Training on real+synthetic must improve (or not hurt) real-only test performance.

### Gate 4: Stability Test
Variance across CV folds must not increase by more than 20%.

### Quality Gates Logic
```
For each CV fold:
  1. Generate synthetic data INSIDE the fold
  2. Run all 4 quality gates
  3. If ALL gates pass → use augmented training data
  4. If ANY gate fails → use real-only training data
  
Final verdict:
  - "useful" if improvement > 1% AND stability not degraded
  - "not_useful" otherwise (this is valid science!)
```

### Synthetic Ratio Limits
- Minimum: 15%
- Maximum: 30%
- **Never** exceed these bounds

---

## 📋 Workflow

### Step 1: Data Preprocessing

```bash
python -m FirstProcessing.main
```

**Purpose:** Transform raw survey data into ML-ready format

**What it does:**
- Translation: Romanian → English
- Normalization: Standardize categorical values
- Feature Engineering: Age grouping, income categorization, product lifetime
- Risk Calculation: Weighted scoring (15+ features), GMM clustering, outlier detection

**Output:** `encoded_data.csv` / `encoded_data.xlsx`

---

### Step 2: Exploratory Data Analysis

```bash
python -m EDA.V2.mainEDA2
```

**Features:**
- Univariate/Bivariate analysis
- Correlation heatmaps
- PCA (80% variance threshold)
- K-Means and GMM clustering
- Cluster comparison metrics

**Configuration** (`EDA/V2/config.py`):
```python
PCA_VARIANCE_THRESHOLD = 0.80
CLUSTERING_K_RANGE = (2, 11)
TARGET = "Risk_Score"
DPI = 300
```

---

### Step 3: Run Baseline Experiments

```bash
# Primary target: Risk_Score (regression)
python run_experiment.py --config configs/baseline_regression.yaml --dataset data.csv

# Secondary target: Save_Money_Yes (classification)
python run_experiment.py --config configs/baseline_classification.yaml --dataset data.csv
```

**Models Available:**

| Regression | Classification |
|------------|----------------|
| Ridge | Logistic Regression |
| Lasso | Random Forest |
| XGBoost Regressor | XGBoost Classifier |
| LightGBM Regressor | LightGBM Classifier |
| Random Forest Regressor | |

---

### Step 4: Test Synthetic Augmentation (Optional)

```bash
python augmentation_experiment.py --config configs/augmentation_experiment.yaml --dataset data.csv
```

**Methods available:**
- `jitter` - Gaussian noise injection (default for regression)
- `smote` - SMOTE oversampling (for classification)
- `cluster` - Cluster-aware enrichment (max 20% per cluster)

**Output:**
- Verdict: "useful" or "not_useful"
- Comparison metrics: real-only vs augmented
- Quality gate results per fold

---

## 🧪 Test Suite

The project includes a comprehensive test suite:

| Test File | Purpose |
|-----------|---------|
| `test_config.py` | Validates forbidden target blocking, config hashing |
| `test_data_integrity.py` | Ensures proper feature/target separation |
| `test_cv_and_leakage.py` | Verifies CV returns expected metrics |
| `test_augmentation_policy.py` | Confirms baseline forces augmentation OFF |
| `test_reproducibility.py` | Same seed = same results |
| `test_artifacts.py` | Checks all output files are created |

```bash
# Run all tests
pytest -v

# Run specific test file
pytest tests/test_reproducibility.py -v

# Run with coverage report
pytest --cov=. --cov-report=term-missing
```

---

## 📁 Data Format

### Input (Raw Survey)
- Format: CSV/Excel (Romanian language)
- ~22 survey questions covering demographics, financial behaviors, savings, etc.

### Processed (Encoded)
- Format: CSV/Excel (English, encoded)
- 80+ one-hot encoded features
- Target variables:
  - `Risk_Score` (continuous) - **USE THIS**
  - `Behavior_Risk_Level` (binary) - **FORBIDDEN AS TARGET**
- Metadata: `Confidence`, `Cluster`, `Outlier`

---

## ⚙️ Configuration Reference

### `configs/baseline_regression.yaml`
```yaml
experiment:
  name: "clean_baseline_regression"
  seed: 42

data:
  target_column: "Risk_Score"
  target_type: "regression"

preprocessing:
  ignored_columns: ["Behavior_Risk_Level"]

augmentation:
  enabled: false  # OFF for baseline

cross_validation:
  n_splits: 5
  n_repeats: 10
```

### `configs/augmentation_experiment.yaml`
```yaml
augmentation:
  enabled: true
  synthetic_ratio: 0.15  # 15%
  max_ratio: 0.30        # Never exceed 30%
  method: "jitter"       # jitter, smote, or cluster

quality_gates:
  memorization_threshold: 0.05
  max_discriminator_auc: 0.75
```

---

## 📚 Dependencies

**Core:**
- pandas, numpy, scipy
- scikit-learn, xgboost, lightgbm
- imbalanced-learn (SMOTE)
- pyyaml, joblib

**Visualization:**
- matplotlib, seaborn

**Testing:**
- pytest, pytest-cov

**Optional (GANs):**
- torch, sdv (CTGAN)

See `requirements.txt` for complete list.

---

## 📝 Changelog

### Sprint 2 (January 2026) - Controlled Synthetic Augmentation
- Added `DataAugmentation/quality_gates.py` - 4 mandatory quality gates
- Added `DataAugmentation/cluster_enrichment.py` - Cluster-aware generation
- Added `augmentation_experiment.py` - Controlled augmentation testing
- Added `tests/` - Comprehensive pytest suite
- Added `pytest.ini` - Test configuration
- Fixed Ridge/Lasso random_state compatibility
- Updated all configs to use Risk_Score (not Behavior_Risk_Level)

### Sprint 1 (January 2026) - Clean Baselines
- Created `run_experiment.py` - Single reproducible entrypoint
- Added FORBIDDEN_TARGETS blocking
- Implemented Repeated K-Fold CV (5×10)
- Created baseline configs for regression and classification

---

## ⚠️ Known Limitations

- **Old folder:** Contains deprecated experiments - DO NOT USE
- **CPU-Only:** GANs configured for CPU (`cuda=False`)
- **Language:** Raw survey data must be in Romanian for FirstProcessing

---

## 🎯 Scientific Validity

This project follows strict reproducibility principles:

1. **Deterministic seeds** - numpy/sklearn/torch seeded everywhere
2. **No circular labels** - Behavior_Risk_Level forbidden as target
3. **Quality gates** - Synthetic data must prove utility before use
4. **Repeated CV** - 50 evaluations (5 folds × 10 repeats) for stable metrics
5. **Test coverage** - Automated tests verify reproducibility

**Important:** If synthetic augmentation does not help, that result is reported as "not_useful" - this is **valid science**, not a failure.

