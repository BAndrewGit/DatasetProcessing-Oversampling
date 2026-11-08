# Financial Behavior ML Pipeline

**A research-oriented machine learning pipeline for exploring financial habits, comparing models, and producing explainable predictions.**

## Overview

This Python project studies financial behavior using survey responses. It explores two questions:

- **Behavioral risk score:** Can a model estimate a score that summarizes self-reported financial habits?
- **Savings behavior:** Can a model predict whether a respondent reports saving money?

The project goes beyond training a single model: it covers data preparation, experiments, model comparison, validation, interpretation, and model export.

> **Important:** The risk score reflects survey-based behavior. It is **not** a credit score and does not predict loan defaults or insolvency.

## Key Features

- **Data preparation:** Cleans, checks, and encodes survey responses for machine learning.
- **Model comparison:** Evaluates linear models, Random Forest, XGBoost, and LightGBM against baseline approaches.
- **Deep learning:** Experiments with PyTorch neural networks that can learn the two prediction tasks separately or together (multi-task learning).
- **Transfer learning:** Tests whether information from an additional financial dataset can help the main model without merging the datasets.
- **Reliable evaluation:** Uses repeated cross-validation and checks for data leakage, duplicate records, and suspiciously strong results.
- **Explainable results:** Investigates which inputs influence predictions using feature importance, SHAP, error analysis, and what-if experiments.
- **Reproducible experiments:** Uses configuration files, fixed random seeds, saved metrics, and model export tools.

## How It Works

```text
Survey responses
      |
      v
Data cleaning and preparation
      |
      v
Train and compare ML models
      |
      v
Evaluate performance and check for data leakage
      |
      v
Explain results and select a model
      |
      v
Export model and supporting files
```

The repository also contains experiments with synthetic data generation. **These were evaluated but excluded from the final training pipeline** after the research process; they are retained for transparency and comparison.

## Technologies

**Python**, Pandas, scikit-learn, **PyTorch**, XGBoost, LightGBM, SHAP, Matplotlib, YAML, and pytest.

## Evaluation

Models are compared using metrics suited to each task:

- **Risk score:** MAE (prediction error) and Spearman correlation (how well the model preserves relative ranking).
- **Savings behavior:** Macro-F1, precision, and recall (performance across both classes).
- **Reliability:** Results across multiple training/validation splits, rather than a single favorable run.

As one recorded example, the model export in the companion application contains the following comparison:

| Model | Risk MAE (lower is better) | Spearman correlation (higher is better) |
| --- | ---: | ---: |
| Multi-task neural network | 0.1841 | 0.4774 |
| Hybrid transfer model | 0.2117 | 0.2383 |

*These values are taken from saved experiment metadata (April 2026), not from a new independent benchmark. They refer to survey-based evaluation, not to validated predictions from bank statements.*

## Repository Structure

| Folder | Purpose |
| --- | --- |
| `FirstProcessing/` | Data cleaning and preparation |
| `EDA/` | Data exploration and visualizations |
| `experiments/` | Machine learning and deep learning code |
| `runners/` | Scripts to launch experiments |
| `configs/` | Experiment settings |
| `analysis/` | Model explanations and result analysis |
| `DataAugmentation/` | Archived and experimental synthetic data methods |
| `tests/` | Automated checks |

## Getting Started

Install dependencies in a Python environment:

```bash
python -m pip install -r requirements.txt
```

Prepare a compatible, encoded survey dataset, then run a shorter experimental pipeline:

```bash
python runners/run_pipeline.py --test-mode --data data/processed/1_encoded.csv
```

For automated checks:

```bash
python -m pytest -q
```

**Setup note:** The survey datasets are not included. The current repository snapshot also references a shared model-contract module that is not present in the archive, so a clean installation requires restoring that dependency. The dependency file may also need to be converted from UTF-16 to UTF-8 before installation.

## Scope and Limitations

- Results are based on **self-reported survey data**, which can be incomplete or biased.
- The project measures associations, **not cause and effect**.
- Synthetic oversampling is an archived research path, **not** a demonstrated improvement to the final model.
- Exporting a model does not establish real-world predictive accuracy for bank transactions.

## Related Project

[BankStatementConvertor](https://github.com/BAndrewGit/BankStatementConvertor) is a companion application that processes bank statements and integrates an exported PyTorch model for experimental inference.
