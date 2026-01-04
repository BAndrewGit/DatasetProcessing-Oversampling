import os
os.environ["LOKY_MAX_CPU_COUNT"] = "4"
import json
import pandas as pd
import numpy as np
import gc
import torch
import time
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import Metadata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score
from scipy.spatial.distance import cosine
from scipy.stats import ks_2samp
from EDA.V1.data_loading import load_data
from EDA.V1.file_operations import select_save_directory
from FirstProcessing.file_operations import auto_adjust_column_width
import warnings
warnings.filterwarnings("ignore")

# DEPRECATED: This class uses iterative dataset growth and 50/50 final balancing.
# These practices cause experimental drift and non-reproducible results.
# For reproducible experiments, use run_experiment.py with configs/*.yaml
class WCGANAugmentation:
    def __init__(self, target_column="Behavior_Risk_Level", step_fraction=0.25, max_size=2000,
                 random_state=42, min_confidence=0.8, generation_buffer=1.5):
        self.model = None
        self.metadata = None
        self.target_column = target_column
        self.step_fraction = step_fraction
        self.max_size = max_size
        self.generation_buffer = generation_buffer
        self.random_state = random_state
        self.min_confidence = min_confidence
        self.device = torch.device("cpu")
        self.history = []
        self.total_generated = 0
        self.total_kept_after_confidence = 0
        self.total_kept_after_consistency = 0
        self._has_logged_metadata = False

    def prepare_data(self, df):
        excluded = ['Risk_Score', 'Confidence', 'Outlier', 'Cluster', 'Auto_Label']
        df = df.drop(columns=[col for col in excluded if col in df.columns], errors='ignore')
        return df.drop(columns=[self.target_column]), df[self.target_column]

    def build_metadata(self, df):
        excluded_columns = {
            'Risk_Score', 'Confidence', 'Outlier', 'Cluster', 'Auto_Label'
        }

        # loghează o singură dată, indiferent de când e apelată
        verbose = not self._has_logged_metadata
        if verbose:
            print("Generating metadata with manual column type assignment...")

        columns_metadata = {}
        for col in df.columns:
            if col in excluded_columns:
                continue

            dtype = df[col].dtype
            unique_values = df[col].dropna().unique()

            if dtype == 'float64':
                sdtype = 'numerical'
            elif dtype == 'int64':
                if set(unique_values).issubset({0, 1}):
                    sdtype = 'boolean'
                elif len(unique_values) < 20:
                    sdtype = 'categorical'
                else:
                    sdtype = 'numerical'
            else:
                sdtype = 'categorical'

            columns_metadata[col] = {'sdtype': sdtype}
            if verbose:
                print(f" {col}: {dtype} → SDV type: {sdtype}")

        self._has_logged_metadata = True

        metadata_dict = {'columns': columns_metadata}
        metadata = Metadata.load_from_dict(metadata_dict)
        return metadata_dict, metadata

    def generate_step(self, df_train, n_samples):
        metadata_dict, metadata = self.build_metadata(df_train)
        included_cols = list(metadata_dict['columns'].keys())
        df_train = df_train[included_cols].copy()

        bool_cols = [col for col, meta in metadata_dict["columns"].items() if meta.get("sdtype") == "boolean"]
        df_train[bool_cols] = df_train[bool_cols].astype(bool)

        model = CTGANSynthesizer(metadata, epochs=1000, batch_size=150, cuda=False)
        model.fit(df_train)

        print("Generating samples using freshly trained model...")
        samples = model.sample(n_samples)

        if self.target_column in samples.columns:
            current_dist = samples[self.target_column].value_counts()
            print(f"Generated distribution: {current_dist.to_dict()}")

            if min(current_dist) < n_samples * 0.3:
                print("Attempting to balance classes in generated data...")
                extra_samples = model.sample(n_samples * 2)
                all_samples = pd.concat([samples, extra_samples])

                balanced = []
                target_per_class = n_samples // 2
                for class_val in [0, 1]:
                    class_samples = all_samples[all_samples[self.target_column] == class_val]
                    if len(class_samples) >= target_per_class:
                        balanced.append(class_samples.sample(target_per_class, random_state=self.random_state))
                    else:
                        balanced.append(class_samples)

                samples = pd.concat(balanced)
                print(f"Balanced distribution: {samples[self.target_column].value_counts().to_dict()}")

        return samples

    def relabel_confident(self, original_df, generated_df):
        X_orig, y_orig = self.prepare_data(original_df)
        X_gen, y_gen = self.prepare_data(generated_df)

        clf = LogisticRegression(max_iter=1000, random_state=self.random_state)
        clf.fit(X_orig, y_orig)

        proba = clf.predict_proba(X_gen)
        confidence = np.max(proba, axis=1)
        delta = np.abs(proba[:, 1] - proba[:, 0])

        # Filtrare de outlieri pe baza încrederii și polarizării predicției
        mask = (confidence >= self.min_confidence) & (delta > 0.5)
        print(f"\n Filtering: Kept {np.sum(mask)} / {len(generated_df)} samples (confident & clear)")

        # Aplică mască și relabel doar pe mostrele bune
        X_gen_filtered = X_gen[mask]
        y_gen_filtered = y_gen[mask]
        y_pred_filtered = clf.predict(X_gen_filtered)

        changed = np.sum(y_pred_filtered != y_gen_filtered)
        y_gen_filtered = y_gen_filtered.copy()
        y_gen_filtered[:] = y_pred_filtered  # suprascrie cu predicții

        # Returnează doar mostrele filtrate
        generated_df_filtered = generated_df.iloc[mask].copy()
        generated_df_filtered[self.target_column] = y_gen_filtered.astype(float)


        print(f"Relabeled {np.sum(mask)} samples, with {changed} changes.")
        return generated_df_filtered

    def filter_consistency(self, df):
        if "Save_Money_Yes" not in df.columns or "Save_Money_No" not in df.columns:
            return df  # nimic de verificat

        goal_cols = [col for col in df.columns if col.startswith("Savings_Goal_")]
        obs_cols = [col for col in df.columns if col.startswith("Savings_Obstacle_")]

        mask_goal_valid = ~((df["Save_Money_No"] == 1) & (df[goal_cols].any(axis=1)))
        mask_obs_valid = ~((df["Save_Money_Yes"] == 1) & (df[obs_cols].any(axis=1)))

        consistent_df = df[mask_goal_valid & mask_obs_valid].copy()
        print(f" Consistency filter: Kept {len(consistent_df)} / {len(df)} samples")

        return consistent_df

    def validate_new_samples(self, original_df, augmented_df):
        X_orig, y_orig = self.prepare_data(original_df)
        X_new, y_new = self.prepare_data(augmented_df)

        clf = LogisticRegression(max_iter=1000, random_state=self.random_state)
        clf.fit(X_orig, y_orig)
        y_pred = clf.predict(X_new)

        y_new = y_new.astype(float)
        y_pred = y_pred.astype(float)

        kappa = cohen_kappa_score(y_new, y_pred)
        f1 = f1_score(y_new, y_pred, average='weighted')

        # Select only numeric columns (float/int) and compute means as NumPy arrays
        numeric_cols = X_orig.select_dtypes(include=[np.number]).columns
        mean_orig = X_orig[numeric_cols].mean().values.astype(np.float64)
        mean_new = X_new[numeric_cols].mean().values.astype(np.float64)
        cos_sim = 1 - cosine(mean_orig, mean_new)

        # KS test for distribution shift
        ks_results = {
            col: ks_2samp(X_orig[col], X_new[col])[0]
            for col in numeric_cols
        }

        return {
            'cohen_kappa': kappa,
            'f1_score': f1,
            'cosine_similarity': cos_sim,
            'ks_distance': ks_results
        }

    def save_results(self, df_aug, history, save_dir):
        bool_cols = df_aug.select_dtypes(include=["bool"]).columns
        df_aug[bool_cols] = df_aug[bool_cols].astype(int)
        df_aug.to_csv(os.path.join(save_dir, "augmented_dataset_encoded.csv"), index=False)

        excel_path = os.path.join(save_dir, "augmented_dataset_encoded.xlsx")
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df_aug.to_excel(writer, index=False, sheet_name='Encoded_Data')
            auto_adjust_column_width(writer, 'Encoded_Data')

        with open(os.path.join(save_dir, "augmentation_metrics.json"), 'w') as f:
            json.dump({"augmentation_history": history}, f, indent=2)

        print(f"\nSaved to {save_dir}")

    def augment_incrementally(self, df):
        df_current = df.copy()
        iteration = 0
        start_time = time.time()
        iteration_durations = []

        # Tracking counters
        self.total_generated = 0
        self.total_kept_after_confidence = 0
        self.total_kept_after_consistency = 0

        while len(df_current) < self.max_size:
            iteration += 1
            iter_start = time.time()

            remaining = self.max_size - len(df_current)
            base_step = max(1, int(len(df_current) * self.step_fraction))
            to_generate = min(int(base_step * self.generation_buffer), remaining * 3)

            print(f"\nIteration {iteration}: Attempting to generate {to_generate} samples...")

            generated = self.generate_step(df_current, to_generate)
            self.total_generated += to_generate

            generated = self.relabel_confident(df_current, generated)
            self.total_kept_after_confidence += len(generated)

            generated = self.filter_consistency(generated)
            self.total_kept_after_consistency += len(generated)

            generated = generated.sample(min(len(generated), remaining), random_state=self.random_state)

            metrics = self.validate_new_samples(df_current, generated)
            self.history.append({
                'iteration': iteration,
                'generated': len(generated),
                'new_total': len(df_current) + len(generated),
                'metrics': metrics
            })

            df_current = pd.concat([df_current, generated], ignore_index=True)
            gc.collect()

            elapsed_iter = time.time() - iter_start
            total_elapsed = time.time() - start_time
            iteration_durations.append(elapsed_iter)

            print(
                f"Validation - F1: {metrics['f1_score']:.3f}, Kappa: {metrics['cohen_kappa']:.3f}, CosSim: {metrics['cosine_similarity']:.3f}")
            print(
                f"Time: {elapsed_iter:.2f}s | Total: {total_elapsed:.1f}s | Progress: {len(df_current) / self.max_size * 100:.1f}%")

        # Final balancing loop with re-training until exact 50/50
        print("\nBalancing final dataset to 50/50 distribution...")
        desired_per_class = self.max_size // 2

        for class_val in [0, 1]:
            while True:
                current_count = df_current[self.target_column].value_counts().get(class_val, 0)
                if current_count >= desired_per_class:
                    break  # această clasă este completă

                to_generate = desired_per_class - current_count
                to_generate_with_buffer = int(to_generate * self.generation_buffer)
                print(f"Generating {to_generate_with_buffer} extra samples for class {class_val}...")

                class_subset = df_current[df_current[self.target_column] == class_val]
                extra = self.generate_step(class_subset, to_generate_with_buffer)
                self.total_generated += to_generate_with_buffer

                extra = self.relabel_confident(df_current, extra)
                self.total_kept_after_confidence += len(extra)

                extra = self.filter_consistency(extra)
                self.total_kept_after_consistency += len(extra)

                extra = extra[extra[self.target_column] == class_val]
                extra = extra.sample(min(len(extra), to_generate), random_state=self.random_state)

                df_current = pd.concat([df_current, extra], ignore_index=True)

        # Trim to exact 50/50
        balanced_df = []
        for class_val in [0, 1]:
            class_samples = df_current[df_current[self.target_column] == class_val]
            if len(class_samples) > desired_per_class:
                class_samples = class_samples.sample(desired_per_class, random_state=self.random_state)
            balanced_df.append(class_samples)

        df_current = pd.concat(balanced_df, ignore_index=True)

        # Sync labels
        X_full, y_full = self.prepare_data(df_current)
        df_current[self.target_column] = y_full

        # Final Summary
        final_distribution = df_current[self.target_column].value_counts().to_dict()
        print("\n=== AUGMENTATION SUMMARY ===")
        print(f"Total iterations: {iteration}")
        print(f"Total generated samples (raw): {self.total_generated}")
        print(f"Kept after confidence filter: {self.total_kept_after_confidence}")
        print(f"Kept after consistency filter: {self.total_kept_after_consistency}")
        print(f"Final dataset size: {len(df_current)}")
        print(f"Final class distribution: {final_distribution}")

        return df_current, self.history


# DEPRECATED: This script uses iterative augmentation which causes experimental drift.
# Use run_experiment.py with configs/*.yaml for reproducible experiments.
# The code below is disabled to prevent accidental use.

if __name__ == "__main__":
    print("=" * 60)
    print("DEPRECATED: This augmentation script is disabled.")
    print("Reason: Iterative growth + 50/50 balancing causes experimental drift.")
    print("")
    print("Use instead:")
    print("  python run_experiment.py --config configs/default.yaml")
    print("=" * 60)
    # Original code disabled:
    # augmenter = WCGANAugmentation(...)
    # df_aug, history = augmenter.augment_incrementally(df)
