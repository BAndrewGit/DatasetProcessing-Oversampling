# Financial Behavior Risk Analysis Pipeline

A reproducible machine learning pipeline for analyzing financial behavior from survey data.  
The system trains and evaluates models to predict **financial risk (continuous)** and **savings propensity (binary)** using clean baselines and optional, strictly quality-gated synthetic augmentation. The project is designed for research use: deterministic runs, explicit configuration, leakage prevention, and full experiment traceability.

---

## Methodological foundation

### The learning problem

This pipeline addresses **prediction on self-reported behavioral data**, not financial default prediction or credit scoring. The dataset consists of cross-sectional survey responses describing financial attitudes, spending patterns, and self-assessed outcomes. The data is noisy, subjective, and contains no observed default events or externally validated financial outcomes.

The prediction targets are:
1. **Risk_Score (continuous)** – A latent proxy variable reflecting aggregated behavioral tendencies
2. **Save_Money_Yes (binary)** – A self-reported behavioral outcome

### What this pipeline does NOT do

- **Does not predict financial default or insolvency** – No ground-truth default events exist in the data
- **Does not produce credit scores** – Risk_Score is a behavioral tendency metric, not a creditworthiness assessment
- **Does not make causal claims** – Models learn correlational patterns under uncertainty
- **Does not discover "true risk"** – The target is noisy, subjective, and measurement-limited

### Rejected approach: Circular supervision

A binary label `Behavior_Risk_Level` exists in some datasets but is **explicitly excluded from all training, validation, and model selection** because:
- It was derived via clustering on the same features used as model inputs
- Using it as supervision creates tautological learning (predicting a label computed from the predictors)
- Any model trained on it would learn artifacts of the clustering algorithm, not behavioral patterns

This variable is preserved only for:
- Exploratory analysis
- Error stratification
- Post-hoc interpretation

**It may never influence model training.**

---

## Target definitions and justifications

### Target 1: Risk_Score (Primary – Continuous Regression)

**Definition:**  
Risk_Score is a weighted aggregation of behavioral features, treated as a **latent proxy variable** for financial behavioral risk tendency. It is not a ground-truth measure of insolvency or default probability.

**Measurement properties:**
- **Type:** Continuous (assumed interval-scale by convention)
- **Validity:** Subjective and noisy (derived from self-reported survey data)
- **Interpretation:** Reflects aggregated behavioral tendencies, not causal outcomes

**Why regression despite survey noise:**
- Behavioral tendencies exist on a continuum; dichotomization discards information
- Rank ordering (Spearman correlation) is as important as absolute error (MAE, RMSE)
- Monotonic interpretability prioritized over precise numerical accuracy
- The goal is **relative risk assessment**, not exact point prediction

**Justification for use:**
- Risk_Score is computed **independently** of the model (no circularity)
- It aggregates multiple behavioral dimensions into a single metric
- Regression allows evaluation of rank consistency and ordinal relationships
- Null or weak results are acceptable outcomes (indicating limited predictability from survey data)

**Explicit limitations:**
- Does not represent objective financial risk (no default events observed)
- Measurement error is high (self-reported data, social desirability bias)
- Causal interpretation is not supported
- Model predictions are correlational, not prescriptive

**Evaluation priorities:**
1. **Rank consistency** (Spearman correlation) – Can the model preserve relative ordering?
2. **Cross-validation stability** – Do predictions generalize across folds?
3. **Error distribution** – Are errors systematic or random?
4. **Absolute accuracy** (MAE, RMSE) – Secondary to rank-based metrics

---

### Target 2: Save_Money_Yes (Secondary – Binary Classification)

**Definition:**  
Save_Money_Yes is a **self-reported behavioral outcome** indicating whether the respondent reports actively saving money. It is a direct survey response, not a derived or clustered label.

**Measurement properties:**
- **Type:** Binary (0 = No, 1 = Yes)
- **Source:** Direct survey question (lower abstraction than Risk_Score)
- **Validity:** Subject to self-report bias, but observationally grounded

**Why binary classification:**
- The outcome is discrete and directly observed in the survey
- Lower ambiguity compared to abstract risk constructs
- Complements the regression task by focusing on a specific behavioral outcome

**Justification for use:**
- Independent of model inputs (direct survey response, not computed from features)
- Behaviorally meaningful (savings behavior is a tangible outcome)
- Lower measurement error than abstract risk assessments
- Allows evaluation of class-specific performance (precision/recall by group)

**Rejected approaches:**
- **No forced class balancing** – Natural class distribution preserved to reflect survey population
- **No synthetic oversampling** – Augmentation tested separately with strict quality gates
- **No threshold tuning on validation data** – Default threshold (0.5) used consistently

**Evaluation priorities:**
1. **Macro-F1** – Balanced performance across both classes
2. **Recall** – Ability to identify savers (minority class in many surveys)
3. **Precision** – Avoiding false positives
4. **Accuracy** – Secondary (can be misleading with class imbalance)

---

### Why two separate tasks (not multi-task learning)

The regression and classification tasks are **intentionally decoupled** to preserve experimental control:

1. **Different noise profiles:**
   - Risk_Score is aggregate (high measurement noise)
   - Save_Money_Yes is direct response (lower abstraction)

2. **Different supervision strength:**
   - Regression target is weakly validated (no external ground truth)
   - Classification target is self-reported (observable behavior)

3. **Risk of inductive bias leakage:**
   - Shared representations could transfer circular patterns
   - Task coupling obscures which target drives performance

4. **Interpretability:**
   - Separate models allow independent analysis of each outcome
   - Failure of one task does not invalidate the other

**Multi-task learning is avoided** to prevent confounding effects and maintain clear attribution of predictive signals.

---

### Multi-task ablation experiment (Sprint 4)

While the default approach keeps tasks separate, a **controlled ablation experiment** tests whether shared representations can support both targets under strict experimental control.

**Design:**
1. **Risk-only model** – MLP trunk + risk head (baseline)
2. **Savings-only model** – MLP trunk + savings head (baseline)
3. **Multi-task model** – Shared trunk + both heads (ablation)

**Architecture constraints:**
- **Conservative trunk:** 1-2 hidden layers maximum
- **Mandatory regularization:** Dropout + weight decay
- **Early stopping:** Validation-based (no test set peeking)
- **No synthetic data:** Real data only
- **Same CV protocol:** Repeated K-fold (5×10)

**Evaluation criteria:**
- Multi-task model must **not catastrophically degrade either task**
- Marginal gains or improved stability reported cautiously
- Negative results (multi-task worse) are valid outcomes
- Comparison is fair: same architecture, same data, same protocol

**Framing:**
- This is an **experiment**, not a recommendation
- Tests whether shared representations help despite intentional task separation
- Single-task baselines remain the primary approach
- Performance improvement is optional; experimental integrity is mandatory

**Usage:**
```bash
# Run multi-task ablation
python run_multitask_experiment.py --config configs/multitask_experiment.yaml
```

---

### Summary table

| Target                | Type       | Role          | Validation        | Primary Metric |
|-----------------------|------------|---------------|-------------------|----------------|
| `Risk_Score`          | Continuous | Primary       | Noisy, subjective | Spearman ρ     |
| `Save_Money_Yes`      | Binary     | Secondary     | Self-reported     | Macro-F1       |
| `Behavior_Risk_Level` | Binary     | **FORBIDDEN** | Circular          | N/A (excluded) |

---

## Hard rules (non-negotiable)

- No retraining on synthetic-enriched data  
- No iterative or exponential dataset growth  
- No forced 50/50 class balancing  
- No GAN retraining on synthetic outputs  
- No use of `Behavior_Risk_Level` as target or feature  

Violations raise errors; nothing is silently corrected.

---

## Expected input

- CSV file with numeric and one-hot encoded features  
- At least one valid target column (`Risk_Score` or `Save_Money_Yes`)  
- Optional `Save_Money_No` column (must be mutually exclusive with `Save_Money_Yes`)  
- `Behavior_Risk_Level` may exist but will be excluded automatically  

---

## Project Structure

The project is organized into logical subfolders for easy navigation:

```
Procesare Dataset/
├── runners/                    # Experiment execution scripts
│   ├── run_baseline.py         # Baseline experiments (no augmentation)
│   ├── run_multitask_experiment.py    # Multi-task learning
│   ├── run_domain_transfer_experiment.py  # Domain transfer (ADV+GMSC)
│   ├── run_analysis.py         # Interpretability and paper artifacts
│   └── augmentation_experiment.py  # Synthetic data experiments
│
├── experiments/                # Core ML modules
│   ├── config_schema.py        # Configuration validation
│   ├── cv.py                   # Cross-validation
│   ├── data.py                 # Data loading/preprocessing
│   ├── models.py               # Model builders
│   ├── multitask.py            # Multi-task architecture
│   └── domain_transfer.py      # Domain transfer architecture
│
├── analysis/                   # Interpretability and analysis
│   ├── interpretability.py     # Feature importance, SHAP
│   ├── stability.py            # CV stability analysis
│   ├── error_analysis.py       # Error distribution
│   ├── what_if.py              # Partial dependence, sensitivity
│   └── paper_artifacts.py      # LaTeX tables, diagrams
│
├── configs/                    # YAML configurations
│   ├── baseline/               # Baseline experiment configs
│   │   ├── regression.yaml
│   │   └── classification.yaml
│   ├── augmentation/           # Augmentation experiment configs
│   │   ├── experiment.yaml
│   │   └── smote.yaml
│   ├── multitask/              # Multi-task configs
│   │   └── experiment.yaml
│   └── transfer/               # Domain transfer configs
│       └── domain_transfer.yaml
│
├── DataAugmentation/           # Synthetic data generation
│   ├── quality_gates.py        # Mandatory quality checks
│   ├── smote_tomek.py          # SMOTE-Tomek
│   └── cluster_enrichment.py   # Cluster-aware enrichment
│
├── FirstProcessing/            # Data preprocessing pipeline
│   ├── main.py                 # Main preprocessing runner
│   ├── encoder.py              # Feature encoding
│   └── risk_calculation.py     # Risk score computation
│
├── EDA/                        # Exploratory data analysis
│   ├── V1/                     # Basic EDA
│   └── V2/                     # Advanced EDA with clustering
│
├── tests/                      # Test suite
│   ├── test_analysis.py        # Analysis module tests
│   ├── test_domain_transfer.py # Domain transfer tests
│   ├── test_multitask.py       # Multi-task tests
│   └── ...                     # Other tests
│
├── data/                       # Datasets (not in git)
│   ├── raw/                    # Original survey CSV
│   ├── processed/              # Encoded datasets
│   └── gmsc/                   # GMSC auxiliary dataset
│
└── runs/                       # Experiment outputs
```

---

## Data organization

Datasets are organized in a `data/` folder by processing stage:

```
data/
├── raw/               # Original survey CSV (unmodified)
├── processed/         # Encoded dataset from FirstProcessing
├── gmsc/              # Give Me Some Credit dataset (auxiliary)
└── experiments/       # Custom datasets (optional)
```

### Data pipeline

```
1. FirstProcessing/main.py
   Input:  data/raw/survey_responses.csv
   Output: data/processed/encoded_dataset.csv

2. runners/run_baseline.py
   Input:  data/processed/encoded_dataset.csv
   Output: runs/experiment_TIMESTAMP/
```

**First-time setup:**
```bash
# Place your raw survey CSV in: data/raw/survey_responses.csv
# Generate encoded dataset
python FirstProcessing/main.py
# Output: data/processed/encoded_dataset.csv
```

**Note:** CSV files in the `data/` folder are excluded from git (see `.gitignore`).

---

## Output per run

Each experiment creates a folder in `runs/` containing:

| File                  | Description                               |
|-----------------------|-------------------------------------------|
| `config.yaml`         | Exact configuration used                  |
| `metrics.json`        | CV metrics (mean, std, fold-level values) |
| `model.joblib`        | Final trained model                       |
| `data_profile.json`   | Dataset hash, size, features              |
| `cv_distribution.png` | CV score distributions                    |

---

## Installation

```bash
pip install -r requirements.txt
pip install torch  # For multi-task experiments
pip install pytest pytest-cov
```

---

## Usage

### 1. Baseline experiments (single-task)

```bash
# Risk regression
python runners/run_baseline.py --config configs/baseline/regression.yaml

# Savings classification
python runners/run_baseline.py --config configs/baseline/classification.yaml
```

### 2. Multi-task ablation experiment

```bash
# Run all three experiments (risk-only, savings-only, multi-task)
python runners/run_multitask_experiment.py --config configs/multitask/experiment.yaml

# Output: Comparison table showing:
# - Risk-only baseline
# - Savings-only baseline
# - Multi-task model (shared trunk)
```

### 3. Domain transfer experiment

```bash
# Run ADV-only vs ADV+GMSC transfer comparison
python runners/run_domain_transfer_experiment.py --config configs/transfer/domain_transfer.yaml
```

### 4. Synthetic data experiments

```bash
python runners/augmentation_experiment.py --config configs/augmentation/experiment.yaml
```

### 5. Analysis and interpretability

```bash
python runners/run_analysis.py \
  --runs runs/experiment_1/ runs/experiment_2/ \
  --dataset data/processed/1_encoded.csv \
  --output analysis_output/
```

---

## Running baseline experiments

### Risk regression (primary task)

```bash
python runners/run_baseline.py \
  --config configs/baseline/regression.yaml \
  --dataset path/to/data.csv
```

### Savings classification (secondary task)

```bash
python runners/run_baseline.py \
  --config configs/baseline/classification.yaml \
  --dataset path/to/data.csv
```

### Run all baselines

```bash
python runners/run_baseline.py --all-baselines --dataset path/to/data.csv
```

Baseline runs **never** allow synthetic augmentation.

---

## Running augmentation experiments (optional)

This mode tests whether synthetic data improves **real-only performance**. If any quality gate fails, the pipeline automatically falls back to real-only data.

```bash
python runners/augmentation_experiment.py \
  --config configs/augmentation/experiment.yaml \
  --dataset path/to/data.csv
```

---

## Available models

Models are selected **only via YAML configuration**.

### Regression
- Ridge
- Lasso
- Random Forest Regressor
- XGBoost Regressor
- LightGBM Regressor

### Classification
- Logistic Regression
- Random Forest
- XGBoost Classifier
- LightGBM Classifier

---

## Domain Transfer Learning (ADV + GMSC)

This module leverages the **Give Me Some Credit (GMSC)** dataset as auxiliary supervision for improving ADV risk representation learning. The approach uses domain-specific adapters with a shared latent trunk, enabling knowledge transfer without dataset merging.

### Architecture

```
ADV features  -> ADV adapter  -┐
                               ├-> Shared trunk -> Risk head (ADV regression)
GMSC features -> GMSC adapter -┘              -> Risk head (GMSC classification)
                                              -> Savings head (ADV only)
```

### Key Design Principles

- **NO dataset merging**: Separate feature adapters per domain
- **Loss masking**: GMSC batches do NOT update the savings head
- **Mixed-domain batches**: Fixed ratio (default 70% ADV / 30% GMSC)
- **Evaluation on ADV only**: GMSC is auxiliary supervision, not test data
- **Optional latent alignment**: CORAL or MMD regularization

### Running Domain Transfer Experiments

```bash
python run_domain_transfer_experiment.py \
  --config configs/domain_transfer_experiment.yaml \
  --adv path/to/adv_data.csv \
  --gmsc path/to/gmsc_data.csv
```

### Ablation Study

The experiment compares:
1. **ADV-only** (baseline): No GMSC transfer
2. **ADV + GMSC transfer**: With domain alignment

Results are evaluated on ADV test data only, measuring:
- Risk regression: MAE, RMSE, Spearman ρ, R²
- Savings classification: Macro-F1, Accuracy

### Configuration Options

```yaml
data:
  adv_ratio: 0.70       # 70% ADV per batch
  gmsc_ratio: 0.30      # 30% GMSC per batch

domain_alignment:
  enabled: true
  method: "coral"       # Options: coral, mmd
  weight: 0.1

model:
  gmsc_risk_weight: 0.5 # Auxiliary supervision weight
```

---

## Validation protocol

- Repeated Cross-Validation  
  - Regression: `RepeatedKFold`  
  - Classification: `RepeatedStratifiedKFold`  
- Default: 5 folds × 10 repeats (50 evaluations)

### Metrics

**Regression**
- MAE
- RMSE
- Spearman correlation
- R²

**Classification**
- Macro-F1
- Accuracy
- Precision
- Recall

---

## Evaluation framework

### Success criteria

Under this methodological framing, **success is not defined by high predictive accuracy alone**. The following criteria determine experimental validity:

#### 1. Stability across cross-validation folds
- Consistent metrics across 50 evaluations (5 folds × 10 repeats)
- Standard deviation < 20% of mean performance
- No single fold drives aggregate results

#### 2. Rank consistency over point accuracy
- **For Risk_Score:** Spearman ρ prioritized over MAE/RMSE
  - Can the model preserve relative ordering of risk?
  - Absolute error is expected due to survey noise
- **For Save_Money_Yes:** Macro-F1 prioritized over accuracy
  - Balanced performance across both classes
  - No exploitation of class imbalance

#### 3. Acceptance of negative or null results
- **Weak correlations (ρ < 0.3) are valid outcomes**
  - Indicates limited predictability from survey features
  - Does not invalidate the experimental design
- **Negative synthetic augmentation effects are informative**
  - Confirms that data quality gates work as intended
  - Shows synthetic data does not always help

#### 4. Non-circularity verification
- `Behavior_Risk_Level` never appears in feature sets
- Preprocessing logs confirm exclusion
- Tests enforce forbidden target blocking

#### 5. Reproducibility
- Same seed → same metrics (within floating-point precision)
- Dataset fingerprinting ensures correct data version
- Full experiment provenance (config, data hash, random state)

### What failure looks like

The following outcomes indicate **methodological failure**, not valid results:

| Problem                          | Indicator                                     |
|----------------------------------|-----------------------------------------------|
| Circular supervision leaked      | Near-perfect metrics (R² > 0.95, F1 > 0.98)   |
| Overfitting to synthetic data    | Train >> validation performance               |
| Data leakage                     | Unstable metrics across folds (σ > 50% mean)  |
| Invalid target used              | `Behavior_Risk_Level` in feature importance   |

### Performance expectations

Realistic performance ranges for self-reported behavioral data:

**Risk_Score regression:**
- Spearman ρ: **0.15 – 0.45** (weak to moderate)
- MAE: **0.3 – 0.6** (normalized scale)
- R²: **0.05 – 0.25** (low explained variance is expected)

**Save_Money_Yes classification:**
- Macro-F1: **0.50 – 0.70** (above random baseline)
- Accuracy: **0.55 – 0.75** (modest improvement over majority class)
- Recall (savers): **0.40 – 0.65** (difficult class to predict)

**Results outside these ranges require scrutiny:**
- Too high → Check for circularity or leakage
- Too low → Check for data issues or invalid preprocessing

### Interpretation guidelines

#### For Risk_Score:
- **ρ < 0.20:** Weak signal, features have limited predictive value
- **ρ = 0.20–0.40:** Moderate signal, some rank ordering preserved
- **ρ > 0.40:** Strong signal for survey data (rare, requires validation)

#### For Save_Money_Yes:
- **Macro-F1 < 0.55:** Minimal improvement over random guessing
- **Macro-F1 = 0.55–0.70:** Moderate predictive capability
- **Macro-F1 > 0.70:** Strong signal (verify class balance and overfitting)

---

## Analysis and Interpretability

The `analysis/` module provides comprehensive tools for transforming trained models into publication-ready results.

### Running Analysis

```bash
python run_analysis.py \
  --runs runs/experiment_1/ runs/experiment_2/ \
  --dataset data/processed/1_encoded.csv \
  --output analysis_output/
```

### Analysis Components

#### 1. Interpretability (`analysis/interpretability.py`)
- **Permutation importance**: Global feature importance for any model
- **SHAP values**: TreeSHAP, KernelSHAP for local explanations
- **Actionable features**: Classifies features as actionable vs non-actionable

#### 2. Stability Analysis (`analysis/stability.py`)
- **CV stability**: Variance analysis across repeated folds
- **Model comparison**: Statistical tests between model variants
- **Unstable feature detection**: Identifies brittle features

#### 3. Error Analysis (`analysis/error_analysis.py`)
- **Segment analysis**: Error distribution by cluster/persona
- **Failure cases**: Identifies systematic failure patterns
- **Cluster balance**: Verifies no cluster dominates performance

#### 4. What-If Analysis (`analysis/what_if.py`)
- **Partial dependence**: Feature effect curves
- **Sensitivity analysis**: Perturbation-based sensitivity
- **Economic plausibility**: Validates monotonicity expectations

#### 5. Paper Artifacts (`analysis/paper_artifacts.py`)
- **LaTeX tables**: Ablation and comparison tables
- **Methodology diagrams**: Architecture visualizations
- **Final figures**: Publication-ready plots

### Output Structure

```
analysis_output/
├── analysis_results.json       # Complete results
├── stability/
│   ├── model_stability_comparison.csv
│   └── stability_*.pdf
├── experiment_name/
│   ├── interpretability/
│   │   ├── feature_importance.pdf
│   │   └── feature_importance.csv
│   ├── errors/
│   │   ├── error_distribution.pdf
│   │   └── failure_cases.csv
│   └── what_if/
│       ├── partial_dependence.pdf
│       └── sensitivity_tornado.pdf
└── paper_artifacts/
    ├── diagram_multitask.pdf
    ├── diagram_domain_transfer.pdf
    ├── table_ablation.tex
    └── results_summary.json
```

### Acceptance Criteria

The analysis module enforces:
- ✅ All predictions are explainable via post-hoc methods
- ✅ Performance claims backed by repeated CV statistics
- ✅ No forbidden/circular labels used
- ✅ Reproducible from fixed seeds
- ✅ Negative results clearly reported

---

### Reporting requirements

All experimental results must report:

1. **Mean ± std** across all CV folds
2. **Distribution plots** of fold-level scores
3. **Rank-based metrics** (Spearman) alongside error metrics (MAE, RMSE)
4. **Class-wise performance** (precision/recall per class) for classification
5. **Null model comparison** (mean predictor, majority class baseline)

**Negative results must be reported** – they are scientifically valuable outcomes that demonstrate:
- Limits of predictability from survey data
- Effectiveness of quality gates (synthetic data rejection)
- Boundaries of the method's applicability

---

## Synthetic data quality gates

Synthetic data is used **only if all gates pass per fold**:

| Gate         | Requirement                   |
|--------------|-------------------------------|
| Memorization | < 5% near-duplicates          |
| Two-sample   | Real vs synthetic AUC < 0.75  |
| Utility      | Performance not worse by > 2% |
| Stability    | Variance increase < 20%       |

Synthetic ratio limits:
- Recommended: 15%
- Maximum: 30%
- Never exceed 50%

---

## Data integrity checks (runtime enforced)

- No NaN values in features or targets  
- No infinite values in numeric columns  
- `Save_Money_Yes` / `Save_Money_No` mutual exclusivity  
- Train/validation indices are disjoint  
- `Behavior_Risk_Level` blocked from features  

---

## Configuration validation

Configs are validated at startup. Errors are raised if:

- Required keys are missing  
- Target type is invalid  
- Augmentation is enabled in baseline mode  
- Forbidden keys are present anywhere in the config  

Forbidden keys include:
- `max_size`
- `target_total`
- `iterative`
- `iterative_growth`
- `retrain_on_synthetic`
- `final_balancing`

---

## Project structure

```
financial_behavior_risk/
├── data/                          # Dataset storage
│   ├── raw/                       # Original survey data
│   ├── processed/                 # Encoded datasets
│   └── experiments/               # Custom datasets
├── FirstProcessing/               # One-time data encoding
│   ├── main.py                    # Generate encoded dataset
│   └── ...
├── experiments/                   # Experiment logic
│   ├── config_schema.py           # Config validation
│   ├── data.py                    # Data loading
│   ├── cv.py                      # Cross-validation
│   └── models.py                  # Model builders
├── tests/                         # Test suite
├── configs/                       # YAML configurations
├── runs/                          # Experiment outputs
├── run_baseline.py                # Baseline experiment runner
└── run_augmentation_experiment.py # Augmentation experiment runner
```

---

## Testing

```bash
pytest -v
pytest --cov=experiments --cov-report=term-missing
```

---

## Scientific guarantees

- Deterministic seeds for all runs  
- No circular labels  
- No silent configuration changes  
- Synthetic data must prove utility  
- Full dataset fingerprinting per run  
- Fold-level metrics available for analysis  
- Automated tests enforce all invariants  
