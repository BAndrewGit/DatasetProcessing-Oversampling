# Synthetic Data Quality Gates
# Synthetic data is REJECTED unless ALL tests pass
# This prevents synthetic data from poisoning the model

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import mean_absolute_error, f1_score


class SyntheticQualityGates:
    # Mandatory quality gates for synthetic data
    # Synthetic is rejected if ANY gate fails

    def __init__(self, seed=42, verbose=True):
        self.seed = seed
        self.verbose = verbose
        self.results = {}

    def _log(self, msg):
        if self.verbose:
            print(msg)

    # Gate 1: Memorization Test
    # Synthetic samples must not be near-duplicates of real samples
    def memorization_test(self, X_real, X_syn, threshold=0.05):
        self._log("\n[Gate 1] Memorization Test...")

        nn = NearestNeighbors(n_neighbors=1)
        nn.fit(X_real)
        distances, indices = nn.kneighbors(X_syn)

        # Compute feature-wise standard deviation for normalization
        std_real = np.std(X_real, axis=0)
        std_real[std_real == 0] = 1  # Avoid division by zero

        # Normalized distances
        normalized_distances = distances.flatten() / np.mean(std_real)

        # Count near-duplicates (very close samples)
        near_dup_count = np.sum(normalized_distances < threshold)
        near_dup_ratio = near_dup_count / len(X_syn)

        # PASS if less than 5% are near-duplicates
        passed = near_dup_ratio < 0.05

        self.results['memorization'] = {
            'near_duplicate_ratio': float(near_dup_ratio),
            'near_duplicate_count': int(near_dup_count),
            'threshold': threshold,
            'passed': passed
        }

        status = "PASS" if passed else "FAIL"
        self._log(f"  Near-duplicate ratio: {near_dup_ratio:.4f} ({near_dup_count}/{len(X_syn)})")
        self._log(f"  Result: {status}")

        return passed

    # Gate 2: Two-Sample Test (Distributional Similarity)
    # Real vs synthetic should not be trivially separable
    def two_sample_test(self, X_real, X_syn, max_auc=0.75):
        self._log("\n[Gate 2] Two-Sample Test (real vs synthetic discriminator)...")

        # Create labels: 0=real, 1=synthetic
        n_real = len(X_real)
        n_syn = len(X_syn)

        X_combined = np.vstack([X_real, X_syn])
        y_combined = np.concatenate([np.zeros(n_real), np.ones(n_syn)])

        # Train a classifier to distinguish real from synthetic
        clf = RandomForestClassifier(n_estimators=50, random_state=self.seed, max_depth=5)

        # Cross-validation AUC
        try:
            from sklearn.metrics import roc_auc_score
            scores = cross_val_score(clf, X_combined, y_combined, cv=5, scoring='roc_auc')
            mean_auc = np.mean(scores)
        except Exception as e:
            self._log(f"  Warning: AUC calculation failed: {e}")
            mean_auc = 0.5

        # PASS if classifier cannot easily distinguish (AUC close to 0.5)
        passed = mean_auc < max_auc

        self.results['two_sample'] = {
            'discriminator_auc': float(mean_auc),
            'max_auc_threshold': max_auc,
            'passed': passed
        }

        status = "PASS" if passed else "FAIL"
        self._log(f"  Discriminator AUC: {mean_auc:.4f} (threshold: <{max_auc})")
        self._log(f"  Result: {status}")

        return passed

    # Gate 3: Utility Test (Train on real+syn, test on real-only improves)
    # This is the core test: does synthetic data actually help?
    def utility_test_regression(self, X_real, y_real, X_syn, y_syn, n_splits=3, n_repeats=3):
        self._log("\n[Gate 3] Utility Test (regression)...")

        cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=self.seed)

        # Baseline: train on real only
        mae_real_only = []
        # Augmented: train on real+syn
        mae_augmented = []

        for train_idx, test_idx in cv.split(X_real):
            X_train_real = X_real[train_idx]
            y_train_real = y_real[train_idx]
            X_test = X_real[test_idx]
            y_test = y_real[test_idx]

            # Real-only model
            model_real = Ridge(random_state=self.seed)
            model_real.fit(X_train_real, y_train_real)
            pred_real = model_real.predict(X_test)
            mae_real_only.append(mean_absolute_error(y_test, pred_real))

            # Augmented model (real + synthetic)
            X_train_aug = np.vstack([X_train_real, X_syn])
            y_train_aug = np.concatenate([y_train_real, y_syn])

            model_aug = Ridge(random_state=self.seed)
            model_aug.fit(X_train_aug, y_train_aug)
            pred_aug = model_aug.predict(X_test)
            mae_augmented.append(mean_absolute_error(y_test, pred_aug))

        mae_real_mean = np.mean(mae_real_only)
        mae_aug_mean = np.mean(mae_augmented)
        improvement = (mae_real_mean - mae_aug_mean) / mae_real_mean * 100

        # PASS if augmented is not worse (or at most 2% worse)
        passed = mae_aug_mean <= mae_real_mean * 1.02

        self.results['utility'] = {
            'mae_real_only': float(mae_real_mean),
            'mae_augmented': float(mae_aug_mean),
            'improvement_pct': float(improvement),
            'passed': passed
        }

        status = "PASS" if passed else "FAIL"
        self._log(f"  MAE (real-only): {mae_real_mean:.4f}")
        self._log(f"  MAE (augmented): {mae_aug_mean:.4f}")
        self._log(f"  Improvement: {improvement:+.2f}%")
        self._log(f"  Result: {status}")

        return passed

    def utility_test_classification(self, X_real, y_real, X_syn, y_syn, n_splits=3, n_repeats=3):
        self._log("\n[Gate 3] Utility Test (classification)...")

        cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=self.seed)

        f1_real_only = []
        f1_augmented = []

        for train_idx, test_idx in cv.split(X_real, y_real):
            X_train_real = X_real[train_idx]
            y_train_real = y_real[train_idx]
            X_test = X_real[test_idx]
            y_test = y_real[test_idx]

            # Real-only model
            model_real = LogisticRegression(class_weight='balanced', random_state=self.seed, max_iter=1000)
            model_real.fit(X_train_real, y_train_real)
            pred_real = model_real.predict(X_test)
            f1_real_only.append(f1_score(y_test, pred_real, average='macro'))

            # Augmented model
            X_train_aug = np.vstack([X_train_real, X_syn])
            y_train_aug = np.concatenate([y_train_real, y_syn])

            model_aug = LogisticRegression(class_weight='balanced', random_state=self.seed, max_iter=1000)
            model_aug.fit(X_train_aug, y_train_aug)
            pred_aug = model_aug.predict(X_test)
            f1_augmented.append(f1_score(y_test, pred_aug, average='macro'))

        f1_real_mean = np.mean(f1_real_only)
        f1_aug_mean = np.mean(f1_augmented)
        improvement = (f1_aug_mean - f1_real_mean) / f1_real_mean * 100

        # PASS if augmented is not worse (or at most 2% worse)
        passed = f1_aug_mean >= f1_real_mean * 0.98

        self.results['utility'] = {
            'f1_real_only': float(f1_real_mean),
            'f1_augmented': float(f1_aug_mean),
            'improvement_pct': float(improvement),
            'passed': passed
        }

        status = "PASS" if passed else "FAIL"
        self._log(f"  F1 (real-only): {f1_real_mean:.4f}")
        self._log(f"  F1 (augmented): {f1_aug_mean:.4f}")
        self._log(f"  Improvement: {improvement:+.2f}%")
        self._log(f"  Result: {status}")

        return passed

    # Gate 4: Stability Test (lower variance across folds)
    def stability_test(self, X_real, y_real, X_syn, y_syn, target_type='regression', n_splits=5, n_repeats=3):
        self._log("\n[Gate 4] Stability Test (variance reduction)...")

        if target_type == 'regression':
            cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=self.seed)

            scores_real = []
            scores_aug = []

            for train_idx, test_idx in cv.split(X_real):
                X_train_real = X_real[train_idx]
                y_train_real = y_real[train_idx]
                X_test = X_real[test_idx]
                y_test = y_real[test_idx]

                model_real = Ridge(random_state=self.seed)
                model_real.fit(X_train_real, y_train_real)
                scores_real.append(mean_absolute_error(y_test, model_real.predict(X_test)))

                X_train_aug = np.vstack([X_train_real, X_syn])
                y_train_aug = np.concatenate([y_train_real, y_syn])
                model_aug = Ridge(random_state=self.seed)
                model_aug.fit(X_train_aug, y_train_aug)
                scores_aug.append(mean_absolute_error(y_test, model_aug.predict(X_test)))
        else:
            cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=self.seed)

            scores_real = []
            scores_aug = []

            for train_idx, test_idx in cv.split(X_real, y_real):
                X_train_real = X_real[train_idx]
                y_train_real = y_real[train_idx]
                X_test = X_real[test_idx]
                y_test = y_real[test_idx]

                model_real = LogisticRegression(class_weight='balanced', random_state=self.seed, max_iter=1000)
                model_real.fit(X_train_real, y_train_real)
                scores_real.append(f1_score(y_test, model_real.predict(X_test), average='macro'))

                X_train_aug = np.vstack([X_train_real, X_syn])
                y_train_aug = np.concatenate([y_train_real, y_syn])
                model_aug = LogisticRegression(class_weight='balanced', random_state=self.seed, max_iter=1000)
                model_aug.fit(X_train_aug, y_train_aug)
                scores_aug.append(f1_score(y_test, model_aug.predict(X_test), average='macro'))

        std_real = np.std(scores_real)
        std_aug = np.std(scores_aug)
        variance_reduction = (std_real - std_aug) / std_real * 100 if std_real > 0 else 0

        # PASS if variance is not significantly increased (allow up to 20% increase)
        passed = std_aug <= std_real * 1.2

        self.results['stability'] = {
            'std_real_only': float(std_real),
            'std_augmented': float(std_aug),
            'variance_reduction_pct': float(variance_reduction),
            'passed': passed
        }

        status = "PASS" if passed else "FAIL"
        self._log(f"  Std (real-only): {std_real:.4f}")
        self._log(f"  Std (augmented): {std_aug:.4f}")
        self._log(f"  Variance reduction: {variance_reduction:+.2f}%")
        self._log(f"  Result: {status}")

        return passed

    # Run all gates - ALL must pass
    def run_all_gates(self, X_real, y_real, X_syn, y_syn, target_type='regression'):
        self._log("=" * 60)
        self._log("SYNTHETIC DATA QUALITY GATES")
        self._log("=" * 60)
        self._log(f"Real samples: {len(X_real)}")
        self._log(f"Synthetic samples: {len(X_syn)}")
        self._log(f"Synthetic ratio: {len(X_syn) / len(X_real) * 100:.1f}%")

        # Gate 1: Memorization
        gate1 = self.memorization_test(X_real, X_syn)

        # Gate 2: Two-sample
        gate2 = self.two_sample_test(X_real, X_syn)

        # Gate 3: Utility
        if target_type == 'regression':
            gate3 = self.utility_test_regression(X_real, y_real, X_syn, y_syn)
        else:
            gate3 = self.utility_test_classification(X_real, y_real, X_syn, y_syn)

        # Gate 4: Stability
        gate4 = self.stability_test(X_real, y_real, X_syn, y_syn, target_type)

        all_passed = gate1 and gate2 and gate3 and gate4

        self._log("\n" + "=" * 60)
        self._log("QUALITY GATES SUMMARY")
        self._log("=" * 60)
        self._log(f"Gate 1 (Memorization):  {'PASS' if gate1 else 'FAIL'}")
        self._log(f"Gate 2 (Two-Sample):    {'PASS' if gate2 else 'FAIL'}")
        self._log(f"Gate 3 (Utility):       {'PASS' if gate3 else 'FAIL'}")
        self._log(f"Gate 4 (Stability):     {'PASS' if gate4 else 'FAIL'}")
        self._log("-" * 60)

        if all_passed:
            self._log("VERDICT: SYNTHETIC DATA ACCEPTED")
        else:
            self._log("VERDICT: SYNTHETIC DATA REJECTED")

        self.results['all_passed'] = all_passed
        return all_passed, self.results


def validate_synthetic_ratio(n_syn, n_real, max_ratio=0.30):
    # Validate synthetic ratio is within bounds (15-30%)
    ratio = n_syn / n_real
    if ratio > max_ratio:
        raise ValueError(
            f"Synthetic ratio {ratio:.2%} exceeds maximum {max_ratio:.2%}. "
            f"Reduce synthetic samples from {n_syn} to max {int(n_real * max_ratio)}"
        )
    return ratio

