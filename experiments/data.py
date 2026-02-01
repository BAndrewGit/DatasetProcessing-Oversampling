# Data loading and preprocessing utilities

import numpy as np
import pandas as pd

FORBIDDEN_TARGETS = ["Behavior_Risk_Level"]

# Features used to calculate Risk_Score - must be excluded to avoid leakage
# when predicting Risk_Score (they ARE the formula components)
RISK_SCORE_COMPONENTS = [
    'Debt_Level',
    'Impulse_Buying_Frequency',
    'Essential_Needs_Percentage',
    'Savings_Goal_Emergency_Fund',
    'Bank_Account_Analysis_Frequency'
]


def load_dataset(config, dataset_path=None):
    """Load dataset from path or interactive file dialog."""
    path = dataset_path or config['data'].get('dataset_path')

    if path is None:
        from tkinter import Tk, filedialog
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        print("Select the dataset file...")
        path = filedialog.askopenfilename(
            filetypes=[("CSV files", "*.csv")],
            parent=root
        )
        root.destroy()
        if not path:
            raise ValueError("No dataset selected")

    print(f"Loading dataset: {path}")
    df = pd.read_csv(path)

    return df, path


def preprocess_data(df, config):
    """
    Preprocess dataset: extract features and target.

    Returns:
        X: DataFrame of features
        y: Series of target values
    """
    target = config['data']['target_column']

    # Validate target is not forbidden
    if target in FORBIDDEN_TARGETS:
        raise ValueError(
            f"BLOCKED: '{target}' is a forbidden target (circular label). "
            f"Use 'Risk_Score' for regression or 'Save_Money_Yes' for classification."
        )

    # Drop auxiliary columns
    cols_to_drop = config['preprocessing'].get('columns_to_drop', [])
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    # Get ignored columns (not dropped, just not used as features)
    ignored = config['preprocessing'].get('ignored_columns', [])
    ignored = [c for c in ignored if c in df.columns and c != target]

    # Verify target exists
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in dataset. Available: {list(df.columns)}")

    # IMPORTANT: Exclude Risk_Score formula components when predicting Risk_Score
    # This prevents data leakage (Risk_Score is calculated from these 5 features)
    leakage_cols = []
    if target == 'Risk_Score':
        leakage_cols = [c for c in RISK_SCORE_COMPONENTS if c in df.columns]
        if leakage_cols:
            print(f"EXCLUDED (leakage prevention): {leakage_cols}")

    # Build feature set excluding target, ignored columns, and leakage columns
    feature_cols = [c for c in df.columns
                    if c != target
                    and c not in ignored
                    and c not in leakage_cols]

    X = df[feature_cols].copy()
    y = df[target].copy()

    # Attach metadata about excluded leakage columns so downstream integrity checks
    # can still validate the original values (tests rely on catching NaNs in excluded cols).
    try:
        X._preprocess_meta = {"excluded_columns": leakage_cols, "excluded_df": df[leakage_cols].copy() if leakage_cols else pd.DataFrame()}
    except Exception:
        # Fallback: pandas may not allow arbitrary attributes in some versions; use attrs
        X.attrs["_preprocess_meta"] = {"excluded_columns": leakage_cols, "excluded_df": df[leakage_cols].copy() if leakage_cols else pd.DataFrame()}

    # Print what we're ignoring
    if ignored:
        print(f"IGNORED columns (not used in training): {ignored}")

    # Verify no forbidden columns in features
    for col in X.columns:
        if col in FORBIDDEN_TARGETS:
            raise ValueError(f"BLOCKED: Forbidden column '{col}' found in features!")

    return X, y


def validate_data_integrity(X, y, config):
    """
    Validate data integrity before training.

    Checks:
    - No NaN/infinite values
    - Save_Money_Yes/No mutual exclusivity (for classification)
    - Correct data types
    """
    errors = []

    # Check for NaN in features
    nan_cols = X.columns[X.isnull().any()].tolist()
    # Additionally check any excluded/leakage columns that were attached as metadata on X
    meta = getattr(X, "_preprocess_meta", None) or X.attrs.get("_preprocess_meta") if hasattr(X, 'attrs') else None
    excluded_nan_cols = []
    if meta and isinstance(meta, dict):
        excluded_df = meta.get("excluded_df")
        if excluded_df is not None and not excluded_df.empty:
            excluded_nan_cols = excluded_df.columns[excluded_df.isnull().any()].tolist()
    # merge lists but keep message concise
    if nan_cols:
        errors.append(f"NaN values found in features: {nan_cols}")
    if excluded_nan_cols:
        errors.append(f"NaN values found in excluded/leakage columns: {excluded_nan_cols}")

    # Check for NaN in target
    if y.isnull().any():
        errors.append(f"NaN values found in target ({y.name}): {y.isnull().sum()} missing")

    # Check for infinite values in features
    numeric_cols = X.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if not np.isfinite(X[col]).all():
            errors.append(f"Infinite values found in feature: {col}")

    # Check for infinite values in target (if numeric)
    if np.issubdtype(y.dtype, np.number):
        if not np.isfinite(y).all():
            errors.append(f"Infinite values found in target: {y.name}")

    # Classification-specific checks
    target_type = config['data'].get('target_type', 'regression')
    if target_type == 'classification':
        # Check Save_Money mutual exclusivity if applicable
        target = config['data']['target_column']
        if target == 'Save_Money_Yes' and 'Save_Money_No' in X.columns:
            # This shouldn't happen if config is correct, but check anyway
            errors.append("Save_Money_No found in features for Save_Money_Yes classification. Add to ignored_columns.")

    if errors:
        raise ValueError("Data integrity check failed:\n  - " + "\n  - ".join(errors))

    return True


def validate_save_money_consistency(df):
    """
    Validate Save_Money_Yes and Save_Money_No are mutually exclusive.
    Call this before preprocessing if dataset has both columns.
    """
    if 'Save_Money_Yes' not in df.columns or 'Save_Money_No' not in df.columns:
        return True

    both_one = (df['Save_Money_Yes'] == 1) & (df['Save_Money_No'] == 1)
    both_zero = (df['Save_Money_Yes'] == 0) & (df['Save_Money_No'] == 0)

    if both_one.sum() > 0:
        raise ValueError(f"Data error: {both_one.sum()} rows have both Save_Money_Yes=1 AND Save_Money_No=1")

    if both_zero.sum() > 0:
        raise ValueError(f"Data error: {both_zero.sum()} rows have both Save_Money_Yes=0 AND Save_Money_No=0")

    return True

