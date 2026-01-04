import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import pandas as pd
import numpy as np
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.metrics import f1_score, silhouette_score, cohen_kappa_score
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import cosine
from scipy.stats import f_oneway
from EDA import select_save_directory
from sklearn.preprocessing import StandardScaler
import warnings

try:
    from DataAugmentation.base import BaseAugmentation
except ImportError:
    import sys
    import os

    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from DataAugmentation.base import BaseAugmentation


# DEPRECATED: This class uses iterative two-phase augmentation.
# This causes experimental drift and non-reproducible results.
# For reproducible experiments, use run_experiment.py with configs/*.yaml
#
# HARD REMOVALS (enforced):
# - NO retrain loops on enriched data
# - NO exponential dataset growth
# - NO final 50/50 balancing
# - NO GAN retraining on synthetic output
#
# Use augmentation_experiment.py for controlled synthetic testing with quality gates.

class SMOTETomekAugmentation(BaseAugmentation):
    # DEPRECATED - Do not use in new experiments
    # Use augmentation_experiment.py instead

    def __init__(self, target_column="Behavior_Risk_Level", random_state=42, step_fraction=0.2):
        import warnings
        warnings.warn(
            "SMOTETomekAugmentation is DEPRECATED. "
            "Use augmentation_experiment.py with quality gates instead.",
            DeprecationWarning,
            stacklevel=2
        )
        super().__init__(target_column=target_column)
        self.random_state = random_state
        self.step_fraction = step_fraction
        self.min_step_fraction = 0.05
        self.history = []

    def compute_validation_metrics(self, X, y, X_prev=None, y_prev=None):
        """Compute validation metrics for the augmented dataset."""
        metrics = {}

        # 1. Weighted F1-score using KNN cross-validation
        knn = KNeighborsClassifier(n_neighbors=min(5, len(y) // 2))
        try:
            from sklearn.model_selection import cross_val_score
            f1_weighted = cross_val_score(knn, X, y, cv=min(5, len(y)), scoring='f1_weighted').mean()
            metrics['f1_weighted'] = f1_weighted
        except Exception as e:
            print(f"Warning: F1 score calculation failed: {e}")
            metrics['f1_weighted'] = 0.0

        # 2. Silhouette Score (cluster coherence)
        try:
            silhouette = silhouette_score(X, y)
            metrics['silhouette'] = silhouette
        except Exception as e:
            print(f"Warning: Silhouette score calculation failed: {e}")
            metrics['silhouette'] = 0.0

        # 3. Cohen's Kappa (if previous iteration available)
        if X_prev is not None and y_prev is not None:
            # Train a classifier on previous data
            knn_prev = KNeighborsClassifier(n_neighbors=min(5, len(y_prev) // 2))
            knn_prev.fit(X_prev, y_prev)
            y_pred = knn_prev.predict(X)
            kappa = cohen_kappa_score(y, y_pred)
            metrics['kappa'] = kappa
        else:
            metrics['kappa'] = 1.0  # First iteration, perfect agreement

        # 4. Cosine Similarity between current and previous datasets
        if X_prev is not None:
            # Compare mean feature vectors
            x_mean = np.mean(X, axis=0)
            x_prev_mean = np.mean(X_prev, axis=0)
            cos_sim = 1 - cosine(x_mean, x_prev_mean)
            metrics['cosine_similarity'] = cos_sim
        else:
            metrics['cosine_similarity'] = 1.0  # First iteration

        # 5. Average KNN Distance
        try:
            nn = NearestNeighbors(n_neighbors=min(5, len(X) - 1))
            nn.fit(X)
            distances, _ = nn.kneighbors(X)
            avg_knn_distance = np.mean(distances)
            metrics['avg_knn_distance'] = avg_knn_distance
        except Exception as e:
            print(f"Warning: KNN distance calculation failed: {e}")
            metrics['avg_knn_distance'] = 0.0

        # 6. ANOVA p-value (between class distributions)
        try:
            classes = np.unique(y)
            if len(classes) > 1:
                # Calculate ANOVA p-value for each feature and take the average
                p_values = []

                if isinstance(X, pd.DataFrame):
                    for col in X.columns:
                        groups = [X.loc[y == cls, col].values for cls in classes]
                        if all(len(g) > 0 for g in groups):
                            _, p_val = f_oneway(*groups)
                            p_values.append(p_val)
                else:  # numpy array
                    for i in range(X.shape[1]):
                        groups = [X[y == cls, i] for cls in classes]
                        if all(len(g) > 0 for g in groups):
                            _, p_val = f_oneway(*groups)
                            p_values.append(p_val)

                metrics['anova_p_value'] = float(np.mean(p_values)) if p_values else 1.0
            else:
                metrics['anova_p_value'] = 1.0
        except Exception as e:
            print(f"Warning: ANOVA test failed: {e}")
            metrics['anova_p_value'] = 1.0

        return metrics

    def is_valid_augmentation(self, metrics, prev_metrics=None):
        """Determine if the augmentation is valid based on metrics."""
        if prev_metrics is None:
            return True, "good"  # First iteration is always valid

        # Define thresholds for significant drops
        thresholds = {
            'f1_weighted': 0.05,  # Max 5% drop in F1
            'silhouette': 0.1,  # Max 10% drop in silhouette
            'kappa': 0.1,  # Max 10% drop in kappa
            'cosine_similarity': 0.05,  # Max 5% drop in cosine similarity
            'avg_knn_distance': 0.2,  # Max 20% increase in KNN distance
            'anova_p_value': -0.1  # p-value can increase (less significant difference)
        }

        # Calculate drops for each metric
        drops = {}
        significant_drops = 0
        total_metrics = 0

        for metric, threshold in thresholds.items():
            if metric in metrics and metric in prev_metrics:
                if metric == 'avg_knn_distance':
                    # For distance, an increase is bad
                    drops[metric] = (metrics[metric] / prev_metrics[metric]) - 1
                    if drops[metric] > threshold:
                        significant_drops += 1
                elif metric == 'anova_p_value':
                    # For p-value, an increase is actually ok (less significant difference)
                    drops[metric] = prev_metrics[metric] - metrics[metric]
                    if drops[metric] > -threshold:  # Reversed logic
                        significant_drops += 1
                else:
                    drops[metric] = prev_metrics[metric] - metrics[metric]
                    if drops[metric] > threshold:
                        significant_drops += 1
                total_metrics += 1

        # Decide if the augmentation is valid
        if significant_drops == 0:
            return True, "good"
        elif significant_drops <= total_metrics // 3:  # Allow up to 1/3 of metrics to degrade
            return True, "tolerated"
        else:
            return False, "repeated"

    def augment(self, df=None, target_count=None, target_total=2000):
        """
        Two-phase augmentation process:
        1. Balance minority class
        2. Expand both classes symmetrically
        """
        if df is None:
            return None, None, None

        X, y = self.prepare_data(df)

        # Initial class distribution
        class_counts = np.bincount(y.astype(int))
        minority_idx = np.argmin(class_counts)
        majority_idx = 1 - minority_idx  # Assuming binary classification

        minority_count = class_counts[minority_idx]
        majority_count = class_counts[majority_idx]

        print(f"Original dataset: {len(df)} samples")
        print(f"Class distribution: {class_counts}")
        print(f"Minority class ({minority_idx}): {minority_count}")
        print(f"Majority class ({majority_idx}): {majority_count}")

        # Phase 1: Balancing minority class
        print("\n--- Phase 1: Balancing minority class ---")
        X_balanced, y_balanced = self._phase1_balancing(X, y, minority_idx, majority_count)

        # Phase 2: Symmetric expansion
        print("\n--- Phase 2: Symmetric expansion ---")
        X_aug, y_aug = self._phase2_expansion(X_balanced, y_balanced, target_total)

        # Final relabeling step
        print("\n--- Final relabeling step ---")
        X_aug, y_aug = self._final_relabeling(X, y, X_aug, y_aug)

        return X_aug, y_aug, self.history

    def _phase1_balancing(self, X, y, minority_class, majority_count):
        """Phase 1: Balance minority class to match majority class count."""
        X_current = X.copy()
        y_current = y.copy()

        # For each iteration, track metrics
        X_prev, y_prev = None, None
        prev_metrics = None
        iteration = 0

        # While minority class count is below majority class count
        while np.bincount(y_current.astype(int))[minority_class] < majority_count:
            iteration += 1
            print(f"\nPhase 1 - Iteration {iteration}")

            # Store current data for comparison
            X_prev, y_prev = X_current.copy(), y_current.copy()

            # Calculate how many minority samples to add
            current_counts = np.bincount(y_current.astype(int))
            current_minority = current_counts[minority_class]
            samples_to_add = min(
                int((majority_count - current_minority) * self.step_fraction),
                majority_count - current_minority
            )

            # Make sure to add at least one sample
            samples_to_add = max(1, samples_to_add)
            target_count = current_minority + samples_to_add

            print(f"  Adding {samples_to_add} samples to class {minority_class}")
            print(f"  Current counts: {current_counts}")
            print(f"  Target count for minority class: {target_count}")

            # Generate synthetic samples using SMOTE-Tomek
            try:
                sampler = SMOTETomek(
                    sampling_strategy={minority_class: target_count},
                    random_state=self.random_state
                )
                X_resampled, y_resampled = sampler.fit_resample(X_current, y_current)

                # Compute validation metrics
                metrics = self.compute_validation_metrics(X_resampled, y_resampled, X_prev, y_prev)
                print(f"  Validation metrics: {metrics}")

                # Check if augmentation is valid
                valid, status = self.is_valid_augmentation(metrics, prev_metrics)

                if valid:
                    # Update data and metrics
                    X_current, y_current = X_resampled, y_resampled
                    prev_metrics = metrics

                    # Record history
                    self.history.append({
                        'phase': 1,
                        'iteration': iteration,
                        'class_distribution': np.bincount(y_current.astype(int)).tolist(),
                        'metrics': metrics,
                        'status': status
                    })

                    print(f"  Status: {status.upper()}")
                    print(f"  New class distribution: {np.bincount(y_current.astype(int))}")
                else:
                    # Try with smaller step size
                    self.step_fraction = max(self.step_fraction / 2, self.min_step_fraction)
                    print(f"  Invalid augmentation. Reducing step fraction to {self.step_fraction}")

                    # Record history
                    self.history.append({
                        'phase': 1,
                        'iteration': iteration,
                        'class_distribution': np.bincount(y_prev.astype(int)).tolist(),
                        'metrics': metrics,
                        'status': status
                    })

                    # If step size is at minimum and still invalid, tolerate it
                    if self.step_fraction == self.min_step_fraction:
                        print("  At minimum step size, tolerating this augmentation")
                        X_current, y_current = X_resampled, y_resampled
                        prev_metrics = metrics

            except Exception as e:
                print(f"  Error in augmentation: {e}")
                # Try with smaller step size
                self.step_fraction = max(self.step_fraction / 2, self.min_step_fraction)
                print(f"  Reducing step fraction to {self.step_fraction}")

                # If at minimum step size, break to avoid infinite loop
                if self.step_fraction == self.min_step_fraction:
                    print("  At minimum step size, ending Phase 1")
                    break

        return X_current, y_current

    def _phase2_expansion(self, X, y, target_total):
        """Phase 2: Expand both classes symmetrically to reach target total."""
        X_current = X.copy()
        y_current = y.copy()

        # Reset step fraction for phase 2
        self.step_fraction = min(0.2, self.step_fraction * 2)

        # For each iteration, track metrics
        X_prev, y_prev = None, None
        prev_metrics = None
        iteration = 0

        # While total count is below target
        while len(y_current) < target_total:
            iteration += 1
            print(f"\nPhase 2 - Iteration {iteration}")

            # Store current data for comparison
            X_prev, y_prev = X_current.copy(), y_current.copy()

            # Calculate how many samples to add for each class
            current_counts = np.bincount(y_current.astype(int))
            samples_to_add = min(
                int((target_total - len(y_current)) * self.step_fraction),
                target_total - len(y_current)
            )

            # Make sure to add at least one sample per class
            samples_to_add = max(2, samples_to_add)
            samples_per_class = samples_to_add // 2

            # Calculate target counts for each class
            class0_count = current_counts[0] + samples_per_class
            class1_count = current_counts[1] + samples_per_class

            print(f"  Adding {samples_per_class} samples to each class")
            print(f"  Current counts: {current_counts}")
            print(f"  Target counts: [{class0_count}, {class1_count}]")

            # Generate synthetic samples using SMOTE
            try:
                sampler = SMOTE(
                    sampling_strategy={0: class0_count, 1: class1_count},
                    random_state=self.random_state
                )
                X_resampled, y_resampled = sampler.fit_resample(X_current, y_current)

                # Compute validation metrics
                metrics = self.compute_validation_metrics(X_resampled, y_resampled, X_prev, y_prev)
                print(f"  Validation metrics: {metrics}")

                # Check if augmentation is valid
                valid, status = self.is_valid_augmentation(metrics, prev_metrics)

                if valid:
                    # Update data and metrics
                    X_current, y_current = X_resampled, y_resampled
                    prev_metrics = metrics

                    # Record history
                    self.history.append({
                        'phase': 2,
                        'iteration': iteration,
                        'class_distribution': np.bincount(y_current.astype(int)).tolist(),
                        'metrics': metrics,
                        'status': status
                    })

                    print(f"  Status: {status.upper()}")
                    print(f"  New class distribution: {np.bincount(y_current.astype(int))}")
                else:
                    # Try with smaller step size
                    self.step_fraction = max(self.step_fraction / 2, self.min_step_fraction)
                    print(f"  Invalid augmentation. Reducing step fraction to {self.step_fraction}")

                    # Record history
                    self.history.append({
                        'phase': 2,
                        'iteration': iteration,
                        'class_distribution': np.bincount(y_prev.astype(int)).tolist(),
                        'metrics': metrics,
                        'status': status
                    })

                    # If step size is at minimum and still invalid, tolerate it
                    if self.step_fraction == self.min_step_fraction:
                        print("  At minimum step size, tolerating this augmentation")
                        X_current, y_current = X_resampled, y_resampled
                        prev_metrics = metrics

            except Exception as e:
                print(f"  Error in augmentation: {e}")
                # Try with smaller step size
                self.step_fraction = max(self.step_fraction / 2, self.min_step_fraction)
                print(f"  Reducing step fraction to {self.step_fraction}")

                # If at minimum step size, break to avoid infinite loop
                if self.step_fraction == self.min_step_fraction:
                    print("  At minimum step size, ending Phase 2")
                    break

        return X_current, y_current

    def _final_relabeling(self, X_orig, y_orig, X_aug, y_aug):
        """Final relabeling step using a trained classifier."""
        # Train classifier on original data
        classifier = LogisticRegression(max_iter=1000, random_state=self.random_state)
        classifier.fit(X_orig, y_orig)

        # Apply to augmented data to ensure consistent labels
        final_proba = classifier.predict_proba(X_aug)
        confidence = np.max(final_proba, axis=1)

        # Only relabel instances with high confidence
        relabel_mask = confidence > 0.8
        y_relabeled = y_aug.copy()

        if np.any(relabel_mask):
            print(f"Relabeling {np.sum(relabel_mask)} instances with high confidence.")
            y_relabeled[relabel_mask] = classifier.predict(X_aug[relabel_mask])

        # Report changes
        changes = np.sum(y_relabeled != y_aug)
        print(f"Changed {changes} labels during final relabeling.")

        return X_aug, y_relabeled


# DEPRECATED: This script uses iterative augmentation which causes experimental drift.
# Use run_experiment.py with configs/*.yaml for reproducible experiments.

if __name__ == "__main__":
    print("=" * 60)
    print("DEPRECATED: This augmentation script is disabled.")
    print("Reason: Iterative phase-based augmentation causes experimental drift.")
    print("")
    print("Use instead:")
    print("  python run_experiment.py --config configs/smote_experiment.yaml")
    print("=" * 60)
    # Original code disabled - use run_experiment.py instead
