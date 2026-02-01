# Cross-validation utilities

import numpy as np
from sklearn.model_selection import RepeatedKFold, RepeatedStratifiedKFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    f1_score, accuracy_score, precision_score, recall_score
)
from scipy.stats import spearmanr

from .models import build_model


def _validate_cv_split(train_idx, val_idx, X, y):
    """
    Validate CV split integrity.

    Assertions:
    - Train/val indices are disjoint
    - No NaN/infinite values in split data
    """
    # Check indices are disjoint
    train_set = set(train_idx)
    val_set = set(val_idx)
    if not train_set.isdisjoint(val_set):
        overlap = train_set.intersection(val_set)
        raise ValueError(f"CV LEAK: Train/val indices overlap! {len(overlap)} shared indices")

    # Get split data
    X_train = X.iloc[train_idx]
    X_val = X.iloc[val_idx]
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]

    # Check for NaN in this split
    if X_train.isnull().any().any():
        raise ValueError("CV split contains NaN in X_train")
    if X_val.isnull().any().any():
        raise ValueError("CV split contains NaN in X_val")
    if y_train.isnull().any():
        raise ValueError("CV split contains NaN in y_train")
    if y_val.isnull().any():
        raise ValueError("CV split contains NaN in y_val")

    # Check for infinite values in numeric columns
    numeric_train = X_train.select_dtypes(include=[np.number])
    numeric_val = X_val.select_dtypes(include=[np.number])

    if not np.isfinite(numeric_train.values).all():
        raise ValueError("CV split contains infinite values in X_train")
    if not np.isfinite(numeric_val.values).all():
        raise ValueError("CV split contains infinite values in X_val")

    return True


def run_repeated_cv_regression(model, X, y, config):
    """
    Run repeated K-fold cross-validation for regression.

    Returns dict with metrics: mae, rmse, spearman, r2
    Each metric has: mean, std, all (list of fold scores)
    """
    cv_config = config.get('cross_validation', {})
    seed = config['experiment']['seed']
    n_splits = cv_config.get('n_splits', 5)
    n_repeats = cv_config.get('n_repeats', 10)

    print(f"Running {n_splits}-fold x {n_repeats} repeats CV (regression)...")

    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    mae_scores = []
    rmse_scores = []
    spearman_scores = []
    r2_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X)):
        # Validate split integrity
        _validate_cv_split(train_idx, val_idx, X, y)

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model_clone = build_model(config)
        model_clone.fit(X_train, y_train)
        y_pred = model_clone.predict(X_val)

        mae_scores.append(mean_absolute_error(y_val, y_pred))
        rmse_scores.append(np.sqrt(mean_squared_error(y_val, y_pred)))

        # Handle constant arrays for Spearman correlation
        if np.std(y_val) > 1e-8 and np.std(y_pred) > 1e-8:
            try:
                spearman_scores.append(spearmanr(y_val, y_pred)[0])
            except Exception:
                spearman_scores.append(0.0)
        else:
            spearman_scores.append(0.0)

        r2_scores.append(r2_score(y_val, y_pred))

    return {
        'mae': {'mean': np.mean(mae_scores), 'std': np.std(mae_scores), 'all': mae_scores},
        'rmse': {'mean': np.mean(rmse_scores), 'std': np.std(rmse_scores), 'all': rmse_scores},
        'spearman': {'mean': np.mean(spearman_scores), 'std': np.std(spearman_scores), 'all': spearman_scores},
        'r2': {'mean': np.mean(r2_scores), 'std': np.std(r2_scores), 'all': r2_scores},
        'n_folds': n_splits,
        'n_repeats': n_repeats
    }


def run_repeated_cv_classification(model, X, y, config):
    """
    Run repeated stratified K-fold cross-validation for classification.

    Returns dict with metrics: macro_f1, accuracy, precision, recall
    Each metric has: mean, std, all (list of fold scores)
    """
    cv_config = config.get('cross_validation', {})
    seed = config['experiment']['seed']
    n_splits = cv_config.get('n_splits', 5)
    n_repeats = cv_config.get('n_repeats', 10)

    print(f"Running {n_splits}-fold x {n_repeats} repeats CV (classification)...")

    # Use stratified CV for classification
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    f1_scores = []
    acc_scores = []
    prec_scores = []
    rec_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        # Validate split integrity
        _validate_cv_split(train_idx, val_idx, X, y)

        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model_clone = build_model(config)
        model_clone.fit(X_train, y_train)
        y_pred = model_clone.predict(X_val)

        f1_scores.append(f1_score(y_val, y_pred, average='macro'))
        acc_scores.append(accuracy_score(y_val, y_pred))
        prec_scores.append(precision_score(y_val, y_pred, average='macro', zero_division=0))
        rec_scores.append(recall_score(y_val, y_pred, average='macro', zero_division=0))

    return {
        'macro_f1': {'mean': np.mean(f1_scores), 'std': np.std(f1_scores), 'all': f1_scores},
        'accuracy': {'mean': np.mean(acc_scores), 'std': np.std(acc_scores), 'all': acc_scores},
        'precision': {'mean': np.mean(prec_scores), 'std': np.std(prec_scores), 'all': prec_scores},
        'recall': {'mean': np.mean(rec_scores), 'std': np.std(rec_scores), 'all': rec_scores},
        'n_folds': n_splits,
        'n_repeats': n_repeats
    }

