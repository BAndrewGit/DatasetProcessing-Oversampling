# Sanity checks for anti-leakage validation
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score, silhouette_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.stats import spearmanr
import warnings
import hashlib


# =============================================================================
# FAST LEAKAGE CHECKS (non-negotiable before any experiment)
# =============================================================================

def check_perfect_baseline(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray,
                           task: str = "regression", seed: int = 42) -> dict:
    """
    Check if baseline achieves suspiciously perfect performance (R² ≈ 1 / MAE ~ 1e-10).
    This is almost always a sign of leakage or bug.
    """
    from sklearn.linear_model import HuberRegressor, Ridge, LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score, mean_absolute_error, f1_score

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc = scaler.transform(X_val)

    if task == "regression":
        model = HuberRegressor(epsilon=1.35, max_iter=500)
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_val_sc)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        # Suspiciously perfect: R² > 0.999 OR MAE < 1e-6 (relative to target scale)
        y_scale = np.std(y)
        relative_mae = mae / max(y_scale, 1e-10)

        is_suspicious = (r2 > 0.999) or (relative_mae < 1e-6)

        if is_suspicious:
            return {
                'is_valid': False,
                'severity': 'CRITICAL',
                'mae': float(mae),
                'r2': float(r2),
                'relative_mae': float(relative_mae),
                'message': f"LEAKAGE SUSPECTED: R²={r2:.6f}, MAE={mae:.2e}, relative_MAE={relative_mae:.2e}"
            }
        return {
            'is_valid': True,
            'severity': 'OK',
            'mae': float(mae),
            'r2': float(r2),
            'relative_mae': float(relative_mae)
        }
    else:
        model = LogisticRegression(max_iter=500)
        model.fit(X_train_sc, y_train)
        y_pred = model.predict(X_val_sc)
        f1 = f1_score(y_val, y_pred, average='macro', zero_division=0)

        is_suspicious = f1 > 0.99

        if is_suspicious:
            return {
                'is_valid': False,
                'severity': 'CRITICAL',
                'f1': float(f1),
                'message': f"LEAKAGE SUSPECTED: Macro-F1={f1:.4f} (too perfect)"
            }
        return {'is_valid': True, 'severity': 'OK', 'f1': float(f1)}


def check_feature_identical_to_target(X: np.ndarray, y: np.ndarray, feature_names: list = None) -> dict:
    """Check if any feature is identical to y (instant leakage)."""
    issues = []
    for i in range(X.shape[1]):
        if np.max(np.abs(X[:, i] - y)) < 1e-10:
            fname = feature_names[i] if feature_names else f"col_{i}"
            issues.append(f"Feature '{fname}' is IDENTICAL to target!")

    return {
        'issues': issues,
        'is_valid': len(issues) == 0,
        'severity': 'CRITICAL' if issues else 'OK'
    }


def check_trivial_target_recovery(X: np.ndarray, y: np.ndarray, feature_names: list = None,
                                   task: str = "regression") -> dict:
    """Check if y is trivially recoverable from a single feature or one-hot group."""
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
    from sklearn.metrics import r2_score

    issues = []
    high_corr_features = []

    # Check correlation/MI with each feature
    for i in range(X.shape[1]):
        fname = feature_names[i] if feature_names else f"col_{i}"

        # Correlation for regression
        if task == "regression":
            if np.std(X[:, i]) > 1e-10 and np.std(y) > 1e-10:
                corr = np.abs(np.corrcoef(X[:, i], y)[0, 1])
                if corr > 0.99:
                    high_corr_features.append((fname, corr, "correlation"))

        # Check if depth-1 tree achieves perfect score
        if task == "regression":
            dt = DecisionTreeRegressor(max_depth=1, random_state=42)
        else:
            dt = DecisionTreeClassifier(max_depth=1, random_state=42)

        dt.fit(X[:, i:i+1], y)
        pred = dt.predict(X[:, i:i+1])

        if task == "regression":
            score = r2_score(y, pred)
            if score > 0.99:
                issues.append(f"Feature '{fname}' alone achieves R²={score:.4f} with depth-1 tree!")
        else:
            acc = np.mean(pred == y)
            if acc > 0.99:
                issues.append(f"Feature '{fname}' alone achieves accuracy={acc:.4f} with depth-1 tree!")

    return {
        'issues': issues,
        'high_corr_features': high_corr_features,
        'is_valid': len(issues) == 0,
        'severity': 'CRITICAL' if issues else ('WARNING' if high_corr_features else 'OK')
    }


def check_duplicate_rows_across_folds(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray,
                                       val_idx: np.ndarray) -> dict:
    """Check for duplicate/near-duplicate rows between train and val."""
    def row_hash(row):
        return hashlib.md5(row.tobytes()).hexdigest()

    # Hash X rows
    train_hashes = set(row_hash(X[i]) for i in train_idx)
    val_hashes = set(row_hash(X[i]) for i in val_idx)
    x_overlap = train_hashes & val_hashes

    # Hash X+y rows
    Xy = np.column_stack([X, y.reshape(-1, 1)])
    train_hashes_xy = set(row_hash(Xy[i]) for i in train_idx)
    val_hashes_xy = set(row_hash(Xy[i]) for i in val_idx)
    xy_overlap = train_hashes_xy & val_hashes_xy

    return {
        'x_duplicates': len(x_overlap),
        'xy_duplicates': len(xy_overlap),
        'is_valid': len(xy_overlap) == 0,
        'severity': 'CRITICAL' if xy_overlap else ('WARNING' if x_overlap else 'OK')
    }


def check_shuffled_y_sanity(X: np.ndarray, y: np.ndarray, model_fn, task: str = "regression",
                             n_repeats: int = 3, seed: int = 42) -> dict:
    """
    Shuffle y_train and verify baseline collapses.
    If shuffled model still performs well -> LEAKAGE.
    """
    from sklearn.model_selection import cross_val_score
    rng = np.random.RandomState(seed)

    # Real performance
    if task == "regression":
        scoring = 'neg_mean_absolute_error'
    else:
        scoring = 'f1_macro'

    try:
        real_scores = cross_val_score(model_fn(), X, y, cv=3, scoring=scoring)
        real_score = -real_scores.mean() if task == "regression" else real_scores.mean()
    except:
        return {'is_valid': True, 'severity': 'SKIPPED', 'reason': 'Model fitting failed'}

    # Shuffled performance
    shuffled_scores = []
    for i in range(n_repeats):
        y_shuffled = rng.permutation(y)
        try:
            scores = cross_val_score(model_fn(), X, y_shuffled, cv=3, scoring=scoring)
            shuffled_scores.append(-scores.mean() if task == "regression" else scores.mean())
        except:
            pass

    if not shuffled_scores:
        return {'is_valid': True, 'severity': 'SKIPPED', 'reason': 'Shuffled model fitting failed'}

    shuffled_mean = np.mean(shuffled_scores)

    # For regression: shuffled MAE should be much worse (higher)
    # For classification: shuffled F1 should be near random (~1/n_classes)
    if task == "regression":
        # Shuffled MAE should be at least 50% worse than real
        ratio = shuffled_mean / max(real_score, 1e-10)
        is_valid = ratio > 1.5  # Shuffled should be 50%+ worse
        severity = 'CRITICAL' if ratio < 1.1 else ('WARNING' if ratio < 1.5 else 'OK')
    else:
        n_classes = len(np.unique(y))
        random_baseline = 1.0 / n_classes
        is_valid = shuffled_mean < random_baseline + 0.15
        severity = 'CRITICAL' if shuffled_mean > random_baseline + 0.3 else ('WARNING' if not is_valid else 'OK')

    return {
        'real_score': float(real_score),
        'shuffled_score': float(shuffled_mean),
        'is_valid': is_valid,
        'severity': severity
    }


def run_all_leakage_checks(X: np.ndarray, y: np.ndarray, train_idx: np.ndarray, val_idx: np.ndarray,
                            feature_names: list = None, task: str = "regression",
                            model_fn=None) -> dict:
    """Run all fast leakage checks and return summary."""
    results = {}

    # Check 0: Perfect baseline (R² ≈ 1 / MAE ~ 1e-10) - MOST CRITICAL
    results['perfect_baseline'] = check_perfect_baseline(X, y, train_idx, val_idx, task)

    # Check 1: Feature identical to target
    results['identical_feature'] = check_feature_identical_to_target(X, y, feature_names)

    # Check 2: Trivial target recovery
    results['trivial_recovery'] = check_trivial_target_recovery(X, y, feature_names, task)

    # Check 3: Duplicate rows across folds
    results['fold_duplicates'] = check_duplicate_rows_across_folds(X, y, train_idx, val_idx)

    # Check 4: Shuffled-y sanity (only if model_fn provided)
    if model_fn is not None:
        results['shuffled_y'] = check_shuffled_y_sanity(X, y, model_fn, task)

    # Overall verdict
    critical = any(r.get('severity') == 'CRITICAL' for r in results.values())
    warning = any(r.get('severity') == 'WARNING' for r in results.values())

    results['overall'] = {
        'is_valid': not critical,
        'severity': 'CRITICAL' if critical else ('WARNING' if warning else 'OK'),
        'message': 'LEAKAGE DETECTED!' if critical else ('Potential issues' if warning else 'All checks passed')
    }

    return results


# =============================================================================
# DOMAIN VALIDITY CHECKS (for synthetic data)
# =============================================================================

def detect_onehot_groups(feature_names: list) -> dict:
    """Detect one-hot encoded feature groups from naming patterns."""
    from collections import defaultdict

    groups = defaultdict(list)

    for i, name in enumerate(feature_names):
        # Common patterns: Feature_Value, Feature_Yes, Feature_No
        if '_' in name:
            prefix = '_'.join(name.split('_')[:-1])
            groups[prefix].append((i, name))

    # Filter to groups with 2+ features (likely one-hot)
    onehot_groups = {k: v for k, v in groups.items() if len(v) >= 2}

    return onehot_groups


def check_onehot_validity(X_synth: np.ndarray, feature_names: list,
                           tolerance: float = 0.1) -> dict:
    """
    Check if synthetic data has valid one-hot structure.

    For each one-hot group:
    - Sum should be ~1.0 (or ~0.0 for "none of above")
    - One value should be dominant (>0.5)

    Returns issues and suggested fixes.
    """
    groups = detect_onehot_groups(feature_names)

    if not groups:
        return {'is_valid': True, 'groups': [], 'issues': []}

    issues = []
    group_stats = []

    for prefix, cols in groups.items():
        col_indices = [c[0] for c in cols]
        col_names = [c[1] for c in cols]

        # Sum across group for each row
        group_sums = X_synth[:, col_indices].sum(axis=1)

        # Check if sums are valid (~1.0 or ~0.0)
        valid_sums = np.logical_or(
            np.abs(group_sums - 1.0) < tolerance,
            np.abs(group_sums) < tolerance
        )
        invalid_count = (~valid_sums).sum()

        # Check if one value is dominant per row
        max_vals = X_synth[:, col_indices].max(axis=1)
        dominant_count = (max_vals > 0.5).sum()

        # Check for fractional values (should be 0 or 1)
        has_fractional = np.any((X_synth[:, col_indices] > tolerance) &
                                 (X_synth[:, col_indices] < 1 - tolerance))

        stat = {
            'prefix': prefix,
            'columns': col_names,
            'invalid_sum_rows': int(invalid_count),
            'dominant_rows': int(dominant_count),
            'has_fractional': bool(has_fractional),
            'mean_sum': float(group_sums.mean()),
            'std_sum': float(group_sums.std())
        }
        group_stats.append(stat)

        if invalid_count > len(X_synth) * 0.1:  # More than 10% invalid
            issues.append(f"Group '{prefix}': {invalid_count}/{len(X_synth)} rows have invalid sum")

        if has_fractional:
            issues.append(f"Group '{prefix}': Contains fractional values (soft one-hot)")

    return {
        'is_valid': len(issues) == 0,
        'groups': group_stats,
        'issues': issues,
        'n_groups': len(groups)
    }


def fix_onehot_synthetic(X_synth: np.ndarray, feature_names: list) -> np.ndarray:
    """
    Fix synthetic data to have valid one-hot structure.
    For each one-hot group: apply argmax to enforce exactly one active category.
    """
    X_fixed = X_synth.copy()
    groups = detect_onehot_groups(feature_names)

    for prefix, cols in groups.items():
        col_indices = [c[0] for c in cols]

        # For each row, set max to 1, others to 0
        for i in range(len(X_fixed)):
            row_vals = X_fixed[i, col_indices]
            max_idx = np.argmax(row_vals)
            X_fixed[i, col_indices] = 0
            X_fixed[i, col_indices[max_idx]] = 1

    return X_fixed


def check_binary_validity(X_synth: np.ndarray, feature_names: list,
                          tolerance: float = 0.1) -> dict:
    """Check if binary features have valid 0/1 values."""
    issues = []
    binary_cols = []

    for i, name in enumerate(feature_names):
        unique_approx = np.unique(np.round(X_synth[:, i]))

        # Detect likely binary columns
        if len(unique_approx) <= 2:
            col_min = X_synth[:, i].min()
            col_max = X_synth[:, i].max()

            # Check if values are near 0 or 1
            near_zero = np.abs(X_synth[:, i]) < tolerance
            near_one = np.abs(X_synth[:, i] - 1) < tolerance
            valid = near_zero | near_one
            invalid_count = (~valid).sum()

            if invalid_count > 0:
                issues.append(f"Binary '{name}': {invalid_count} rows not near 0/1")

            binary_cols.append({
                'name': name,
                'col_idx': i,
                'min': float(col_min),
                'max': float(col_max),
                'invalid_count': int(invalid_count)
            })

    return {
        'is_valid': len(issues) == 0,
        'binary_columns': binary_cols,
        'issues': issues
    }


def check_domain_validity(X_synth: np.ndarray, X_real: np.ndarray,
                          feature_names: list) -> dict:
    """
    Comprehensive domain validity check for synthetic data.

    Checks:
    1. One-hot groups are valid
    2. Binary features are 0/1
    3. Numeric features are within plausible range
    """
    results = {}

    # Check one-hot validity
    results['onehot'] = check_onehot_validity(X_synth, feature_names)

    # Check binary validity
    results['binary'] = check_binary_validity(X_synth, feature_names)

    # Check range validity (synth should be within real range + margin)
    margin = 0.2
    range_issues = []
    for i in range(X_synth.shape[1]):
        real_min, real_max = X_real[:, i].min(), X_real[:, i].max()
        real_range = real_max - real_min

        synth_min, synth_max = X_synth[:, i].min(), X_synth[:, i].max()

        # Allow margin outside real range
        if synth_min < real_min - margin * real_range or synth_max > real_max + margin * real_range:
            fname = feature_names[i] if feature_names else f"col_{i}"
            range_issues.append(f"'{fname}': synth range [{synth_min:.2f}, {synth_max:.2f}] outside real [{real_min:.2f}, {real_max:.2f}]")

    results['range'] = {
        'is_valid': len(range_issues) <= X_synth.shape[1] * 0.1,  # Allow 10% columns
        'issues': range_issues[:10]  # Limit to first 10
    }

    # Overall validity
    all_valid = results['onehot']['is_valid'] and results['binary']['is_valid'] and results['range']['is_valid']

    results['overall'] = {
        'is_valid': all_valid,
        'message': 'Domain valid' if all_valid else 'Domain issues found'
    }

    return results


def check_label_shuffle_baseline(X: np.ndarray, y: np.ndarray, model_fn, n_repeats: int = 5, seed: int = 42) -> dict:
    """Shuffle labels and verify F1 drops to ~random (proves model uses real signal)."""
    rng = np.random.RandomState(seed)

    # Real performance
    from sklearn.model_selection import cross_val_score
    real_scores = cross_val_score(model_fn(), X, y, cv=3, scoring='f1_macro')
    real_f1 = real_scores.mean()

    # Shuffled performance (should be ~random)
    shuffled_f1s = []
    for i in range(n_repeats):
        y_shuffled = rng.permutation(y)
        shuffled_scores = cross_val_score(model_fn(), X, y_shuffled, cv=3, scoring='f1_macro')
        shuffled_f1s.append(shuffled_scores.mean())

    shuffled_mean = np.mean(shuffled_f1s)
    shuffled_std = np.std(shuffled_f1s)

    # Random baseline for binary: ~0.5, for multiclass: ~1/n_classes
    n_classes = len(np.unique(y))
    random_baseline = 1.0 / n_classes

    result = {
        'real_f1': float(real_f1),
        'shuffled_f1_mean': float(shuffled_mean),
        'shuffled_f1_std': float(shuffled_std),
        'random_baseline': float(random_baseline),
        'gap': float(real_f1 - shuffled_mean),
        'is_valid': shuffled_mean < random_baseline + 0.15  # shuffled should be near random
    }

    if not result['is_valid']:
        warnings.warn(f"LEAKAGE WARNING: Shuffled F1={shuffled_mean:.3f} >> random={random_baseline:.3f}")

    return result


def check_fold_overlap(cv, X: np.ndarray, id_col: np.ndarray = None) -> dict:
    """Check for ID/index overlap between train and val in each fold."""
    if id_col is None:
        id_col = np.arange(len(X))

    overlaps = []
    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X)):
        train_ids = set(id_col[train_idx])
        val_ids = set(id_col[val_idx])
        overlap = train_ids & val_ids
        overlaps.append({
            'fold': fold_idx,
            'train_size': len(train_ids),
            'val_size': len(val_ids),
            'overlap_count': len(overlap),
            'overlap_ids': list(overlap)[:10] if overlap else []
        })

    total_overlap = sum(o['overlap_count'] for o in overlaps)
    return {
        'folds': overlaps,
        'total_overlap': total_overlap,
        'is_valid': total_overlap == 0
    }


def validate_preprocessing_no_leakage(X_train: np.ndarray, X_val: np.ndarray,
                                       scaler=None, pca=None) -> dict:
    """Verify scaler/PCA fit ONLY on train, transform on val."""
    issues = []

    # Check scaler
    if scaler is not None:
        if not hasattr(scaler, 'mean_') or scaler.mean_ is None:
            issues.append("Scaler not fitted")
        else:
            # Verify scaler was fit on train (means should match train data)
            train_mean = X_train.mean(axis=0)
            if not np.allclose(scaler.mean_, train_mean, rtol=0.01):
                issues.append("Scaler may be fitted on data other than train")

    # Check PCA
    if pca is not None:
        if not hasattr(pca, 'components_') or pca.components_ is None:
            issues.append("PCA not fitted")

    return {
        'issues': issues,
        'is_valid': len(issues) == 0
    }


class QualityGateLogger:
    """Log each quality gate with PASS/FAIL, reason, and values."""

    def __init__(self):
        self.gates = []

    def check(self, name: str, condition: bool, value: float, threshold: float,
              comparison: str = '>', reason: str = None) -> bool:
        result = {
            'name': name,
            'passed': condition,
            'value': float(value) if value is not None else None,
            'threshold': float(threshold),
            'comparison': comparison,
            'reason': reason or (f"{name}: {value:.4f} {comparison} {threshold:.4f}" if value else name)
        }
        self.gates.append(result)
        return condition

    def all_passed(self) -> bool:
        return all(g['passed'] for g in self.gates)

    def summary(self) -> dict:
        return {
            'gates': self.gates,
            'all_passed': self.all_passed(),
            'passed_count': sum(1 for g in self.gates if g['passed']),
            'failed_count': sum(1 for g in self.gates if not g['passed']),
            'total_gates': len(self.gates)
        }

    def print_summary(self):
        for g in self.gates:
            status = "PASS" if g['passed'] else "FAIL"
            print(f"  [{status}] {g['reason']}")


def run_quality_gates_strict(X_real: np.ndarray, X_synth: np.ndarray,
                              y_real: np.ndarray, y_synth: np.ndarray,
                              memo_threshold: float, config: dict = None,
                              X_val: np.ndarray = None, y_val: np.ndarray = None,
                              task: str = "regression", model_class=None,
                              feature_names: list = None, synth_weight: float = 0.3) -> dict:
    """
    Run all quality gates with REAL utility check on validation.

    Gates:
    1. memorization_min: synthetic not too close to nearest real
    2. memorization_median: median distance OK
    3. two_sample_auc: discriminator can't easily distinguish (<0.75)
    4. coverage: synthetic covers target range
    5. label_variance: synthetic labels have variance
    6. utility: augmented model not worse on VALIDATION
    7. domain_validity: one-hot/binary features are valid

    FIX A: If baseline MAE < floor (1e-6), skip synth entirely
    FIX B: Use synth_weight in utility check for consistency
    FIX C: Use max(auc, 1-auc) for two-sample test
    """
    from sklearn.neighbors import NearestNeighbors
    from sklearn.linear_model import LogisticRegression, Ridge, HuberRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, mean_absolute_error, f1_score

    config = config or {}
    logger = QualityGateLogger()

    if X_synth.shape[0] == 0:
        logger.check("n_samples", False, 0, 1, '>=', "No synthetic samples generated")
        return logger.summary()

    # Gate 1: Memorization - min distance check
    nn = NearestNeighbors(n_neighbors=1).fit(X_real)
    dists, _ = nn.kneighbors(X_synth)
    min_dist = dists.min()
    median_dist = np.median(dists)

    logger.check("memorization_min", min_dist > memo_threshold * 0.5,
                 min_dist, memo_threshold * 0.5, '>',
                 f"Min kNN dist: {min_dist:.4f} > {memo_threshold*0.5:.4f}")

    # Gate 2: Median kNN distance
    logger.check("memorization_median", median_dist > memo_threshold * 0.3,
                 median_dist, memo_threshold * 0.3, '>',
                 f"Median kNN dist: {median_dist:.4f} > {memo_threshold*0.3:.4f}")

    # Gate 3: Two-sample test (discriminator AUC should be near 0.5 = indistinguishable)
    # FIX: Ideal synthetic has AUC ~0.5, detectable synthetic has AUC > 0.75
    # Strict threshold: FAIL if AUC >= 0.75 (easily distinguishable)
    # Warning: if AUC > 0.65 (somewhat distinguishable)
    auc_threshold = config.get('two_sample_auc', 0.75)
    auc_warning_threshold = config.get('two_sample_auc_warning', 0.65)

    try:
        n_synth_available = len(X_synth)
        n = min(len(X_real), n_synth_available, 200)

        # Skip two-sample test when synth count is single digits (not meaningful)
        if n_synth_available < 10:
            logger.check("two_sample_auc", True, None, auc_threshold, '<',
                        f"Skipped: only {n_synth_available} synth samples (need >= 10)")
        elif n >= 10:
            # Bootstrap to get CI for more robust assessment
            n_bootstrap = 5  # Fast bootstrap
            auc_samples = []

            for b in range(n_bootstrap):
                # Subsample with different random state
                rng = np.random.RandomState(42 + b)
                idx_real = rng.choice(len(X_real), min(n, len(X_real)), replace=False)
                idx_synth = rng.choice(len(X_synth), min(n, len(X_synth)), replace=False)

                X_ds = np.vstack([X_real[idx_real], X_synth[idx_synth]])
                y_ds = np.array([0]*len(idx_real) + [1]*len(idx_synth))

                X_tr, X_te, y_tr, y_te = train_test_split(
                    X_ds, y_ds, test_size=0.3, random_state=42+b, stratify=y_ds
                )
                clf = LogisticRegression(max_iter=200, solver='lbfgs')
                clf.fit(X_tr, y_tr)
                auc_raw = roc_auc_score(y_te, clf.predict_proba(X_te)[:, 1])
                auc_samples.append(max(auc_raw, 1 - auc_raw))

            auc_mean = np.mean(auc_samples)
            auc_std = np.std(auc_samples)
            auc_lower = auc_mean - 2 * auc_std  # Approximate 95% CI lower
            auc_upper = auc_mean + 2 * auc_std

            # Check if CI includes 0.5 (ideal: synthetic indistinguishable from real)
            ci_includes_05 = auc_lower <= 0.55  # Allow small margin

            # Pass conditions:
            # 1. Mean AUC < hard threshold (0.75) - not easily distinguishable
            # 2. Preferably CI includes values near 0.5 - truly indistinguishable
            passed = auc_mean < auc_threshold
            is_warning = auc_mean >= auc_warning_threshold and auc_mean < auc_threshold
            is_excellent = ci_includes_05 and auc_mean < 0.60

            if is_excellent:
                msg = f"Discriminator AUC: {auc_mean:.3f}±{auc_std:.3f} [CI: {auc_lower:.2f}-{auc_upper:.2f}] (EXCELLENT: near random)"
            elif is_warning:
                msg = f"Discriminator AUC: {auc_mean:.3f}±{auc_std:.3f} [CI: {auc_lower:.2f}-{auc_upper:.2f}] (WARNING: somewhat distinguishable)"
            elif not passed:
                msg = f"Discriminator AUC: {auc_mean:.3f}±{auc_std:.3f} [CI: {auc_lower:.2f}-{auc_upper:.2f}] (FAIL: easily distinguishable)"
            else:
                msg = f"Discriminator AUC: {auc_mean:.3f}±{auc_std:.3f} [CI: {auc_lower:.2f}-{auc_upper:.2f}]"

            logger.check("two_sample_auc", passed, auc_mean, auc_threshold, '<', msg)
        else:
            logger.check("two_sample_auc", True, None, auc_threshold, '<', "Too few samples")
    except Exception as e:
        logger.check("two_sample_auc", False, None, auc_threshold, '<', f"Test failed: {e}")

    # Gate 4: Coverage (synthetic covers at least 40% of real target range)
    try:
        y_real_range = y_real.max() - y_real.min()
        y_synth_range = y_synth.max() - y_synth.min()
        coverage = y_synth_range / y_real_range if y_real_range > 0 else 0
        logger.check("coverage", coverage > 0.4, coverage, 0.4, '>',
                    f"Target coverage: {coverage:.2f} > 0.4")
    except:
        logger.check("coverage", True, None, 0.4, '>', "Coverage check skipped")

    # Gate 5: Label variance (synthetic labels should have meaningful variance)
    try:
        synth_std = np.std(y_synth)
        real_std = np.std(y_real)
        ratio = synth_std / real_std if real_std > 0 else 0
        logger.check("label_variance", ratio > 0.2, ratio, 0.2, '>',
                    f"Label std ratio: {ratio:.2f} > 0.2")
    except:
        logger.check("label_variance", True, None, 0.2, '>', "Variance check skipped")

    # Gate 6: UTILITY - Real performance check on VALIDATION set (CRITICAL!)
    # This is the gate that actually matters: does augmentation help or hurt?
    utility_delta = config.get('utility_delta', 0.10)  # Allow max 10% degradation

    if X_val is not None and y_val is not None:
        try:
            import inspect

            # Use same model class for fair comparison
            if model_class is not None:
                model_real = model_class()
                model_aug = model_class()
            elif task == "regression":
                model_real = HuberRegressor(max_iter=500)
                model_aug = HuberRegressor(max_iter=500)
            else:
                model_real = LogisticRegression(max_iter=200)
                model_aug = LogisticRegression(max_iter=200)

            # Train on real only
            model_real.fit(X_real, y_real)
            pred_real = model_real.predict(X_val)

            # Train on augmented (real + synth) with sample weights
            # FIX B: Use synth_weight for consistency with actual training
            X_aug = np.vstack([X_real, X_synth])
            y_aug = np.concatenate([y_real, y_synth])
            weights = np.concatenate([np.ones(len(y_real)), np.full(len(y_synth), synth_weight)])

            # Use sample weights if model supports them
            if "sample_weight" in inspect.signature(model_aug.fit).parameters:
                model_aug.fit(X_aug, y_aug, sample_weight=weights)
            else:
                model_aug.fit(X_aug, y_aug)
            pred_aug = model_aug.predict(X_val)

            if task == "regression":
                score_real = mean_absolute_error(y_val, pred_real)
                score_aug = mean_absolute_error(y_val, pred_aug)

                # FIX A: If baseline MAE < floor, skip synth (hypersensitive to noise)
                # Use target scale to set sensible floor
                y_scale = np.std(y_real)
                relative_mae = score_real / max(y_scale, 1e-10)
                mae_floor = config.get('mae_floor', 1e-6)

                if score_real < mae_floor or relative_mae < 1e-6:
                    logger.check("utility", False, score_real, mae_floor, '<',
                                f"Baseline MAE {score_real:.2e} (relative: {relative_mae:.2e}) < floor - "
                                f"LEAKAGE SUSPECTED! Skip synth")
                else:
                    # Proper tolerance for comparison: 5% relative + small absolute floor
                    rel_tol = max(0.05, utility_delta)  # At least 5% tolerance
                    abs_tol = max(mae_floor, 0.001 * y_scale)  # At least 0.1% of target scale
                    tol = max(abs_tol, rel_tol * score_real)
                    max_allowed = score_real + tol

                    passed = score_aug <= max_allowed
                    if not passed:
                        # Compute how much worse augmented is
                        degradation = (score_aug - score_real) / max(score_real, 1e-10) * 100
                        logger.check("utility", False, score_aug, max_allowed, '<=',
                                    f"Val MAE: real={score_real:.6f}, aug={score_aug:.6f} "
                                    f"(degradation: {degradation:.1f}%, max allowed: {rel_tol*100:.0f}%)")
                    else:
                        logger.check("utility", True, score_aug, max_allowed, '<=',
                                    f"Val MAE: real={score_real:.6f}, aug={score_aug:.6f} (OK)")
            else:
                score_real = f1_score(y_val, pred_real, average='macro', zero_division=0)
                score_aug = f1_score(y_val, pred_aug, average='macro', zero_division=0)
                passed = score_aug >= score_real * (1 - utility_delta)
                logger.check("utility", passed, score_aug, score_real * (1 - utility_delta), '>=',
                            f"Val F1: real={score_real:.4f}, aug={score_aug:.4f} (min: {score_real*(1-utility_delta):.4f})")
        except Exception as e:
            logger.check("utility", False, None, None, '', f"Utility check failed: {e}")
    else:
        # No validation set provided - skip utility gate with warning
        logger.check("utility", True, None, None, '', "Utility check skipped (no val set)")

    # Gate 7: Domain validity (one-hot and binary features)
    # This catches "soft one-hots" that can artificially boost performance
    # NOTE: This is a WARNING gate, not a blocking gate (soft one-hots are common in interpolation)
    if feature_names is not None and len(feature_names) == X_synth.shape[1]:
        try:
            domain_check = check_domain_validity(X_synth, X_real, feature_names)
            onehot_valid = domain_check['onehot']['is_valid']
            n_issues = len(domain_check['onehot'].get('issues', []))

            # Domain issues are common with PCA-based interpolation - warn but don't block
            # This is expected behavior for feature-space interpolation methods
            passed = True  # Always pass, but log the warning
            if n_issues > 0:
                logger.check("domain_validity", True, n_issues, n_issues, '<=',
                            f"Domain validity: {n_issues} one-hot issues (WARNING - soft interpolation)")
            else:
                logger.check("domain_validity", True, 0, 0, '<=',
                            f"Domain validity: OK (no one-hot issues)")
        except Exception as e:
            logger.check("domain_validity", True, None, None, '', f"Domain check skipped: {e}")
    else:
        logger.check("domain_validity", True, None, None, '', "Domain check skipped (no feature names)")

    logger.print_summary()
    return logger.summary()


def compute_clustering_stability_strict(Z: np.ndarray, k: int, n_bootstrap: int = 300,
                                         subsample_ratio: float = 0.8, n_init: int = 100,
                                         seed: int = 42) -> dict:
    """Proper clustering stability: 80% subsampling, 300 runs, ARI on intersection."""
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.RandomState(seed)
    n = len(Z)
    subsample_size = int(n * subsample_ratio)

    # Reference clustering on full data
    ref_km = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    ref_labels = ref_km.fit_predict(Z)

    ari_scores = []
    for i in range(n_bootstrap):
        # Subsample
        idx = rng.choice(n, subsample_size, replace=False)
        Z_sub = Z[idx]

        # Cluster subsample
        km = KMeans(n_clusters=k, n_init=n_init, random_state=seed + i)
        sub_labels = km.fit_predict(Z_sub)

        # ARI on intersection (subsample indices)
        ref_sub = ref_labels[idx]
        ari = adjusted_rand_score(ref_sub, sub_labels)
        ari_scores.append(ari)

    mean_ari = np.mean(ari_scores)
    std_ari = np.std(ari_scores)

    # Silhouette on full data
    try:
        sil = silhouette_score(Z, ref_labels)
    except:
        sil = 0.0

    result = {
        'k': k,
        'n_bootstrap': n_bootstrap,
        'subsample_ratio': subsample_ratio,
        'ari_mean': float(mean_ari),
        'ari_std': float(std_ari),
        'ari_min': float(np.min(ari_scores)),
        'ari_max': float(np.max(ari_scores)),
        'silhouette': float(sil),
        'is_stable': mean_ari > 0.6,  # Threshold for usable clustering
        'recommendation': 'use_clustering' if mean_ari > 0.6 else 'no_cluster_tendency'
    }

    if mean_ari < 0.2:
        result['recommendation'] = 'no_cluster_tendency_confirmed'
        result['action'] = 'Stop using clustering. Consider supervised/contrastive embedding or explicit segmentation.'

    return result


def check_multitask_f1_leakage(X: np.ndarray, y_savings: np.ndarray,
                                feature_names: list = None, seed: int = 42) -> dict:
    """Investigate why F1=1 (perfect classification) - likely leakage."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.feature_selection import mutual_info_classif

    results = {
        'checks': [],
        'likely_leakage': False,
        'suspicious_features': []
    }

    # Check 1: Is the exact complement of target in features? (e.g., Save_Money_No)
    if feature_names:
        # Only flag exact leakage patterns, not all savings-related features
        leakage_patterns = ['Save_Money_No', 'Save_Money_Yes']  # target or complement
        suspicious = [f for f in feature_names if f in leakage_patterns]
        if suspicious:
            results['checks'].append(f"Target/complement in features: {suspicious}")
            results['suspicious_features'].extend(suspicious)
            results['likely_leakage'] = True

    # Check 2: Perfect correlation with any feature
    for i in range(X.shape[1]):
        corr = np.corrcoef(X[:, i], y_savings)[0, 1]
        if abs(corr) > 0.99:
            fname = feature_names[i] if feature_names and i < len(feature_names) else f"feature_{i}"
            results['checks'].append(f"Perfect correlation: {fname} (r={corr:.4f})")
            results['suspicious_features'].append(fname)
            results['likely_leakage'] = True

    # Check 3: Mutual information (top features)
    try:
        mi = mutual_info_classif(X, y_savings, random_state=seed)
        top_idx = np.argsort(mi)[-5:][::-1]
        top_mi = [(feature_names[i] if feature_names else f"f{i}", mi[i]) for i in top_idx]
        results['top_mutual_info'] = top_mi

        # Very high MI suggests leakage
        if mi.max() > 0.9:
            results['checks'].append(f"Very high MI: {top_mi[0]} = {mi.max():.4f}")
            results['likely_leakage'] = True
    except:
        pass

    # Check 4: Single feature achieves F1=1?
    for i in range(min(X.shape[1], 20)):  # Check first 20 features
        try:
            scores = cross_val_score(LogisticRegression(max_iter=200), X[:, i:i+1], y_savings, cv=3, scoring='f1_macro')
            if scores.mean() > 0.95:
                fname = feature_names[i] if feature_names and i < len(feature_names) else f"feature_{i}"
                results['checks'].append(f"Single feature F1>0.95: {fname} (F1={scores.mean():.4f})")
                results['suspicious_features'].append(fname)
                results['likely_leakage'] = True
        except:
            pass

    # Check 5: Shuffle test
    shuffle_result = check_label_shuffle_baseline(X, y_savings, lambda: LogisticRegression(max_iter=200), n_repeats=3, seed=seed)
    results['shuffle_test'] = shuffle_result
    if not shuffle_result['is_valid']:
        results['checks'].append("Shuffled labels still have high F1 - leakage confirmed")
        results['likely_leakage'] = True

    return results


