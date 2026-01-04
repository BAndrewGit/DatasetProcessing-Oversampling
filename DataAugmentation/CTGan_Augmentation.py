import os
import json
import pandas as pd
import numpy as np
import gc
import torch
from sdv.single_table import CTGANSynthesizer
from sdv.metadata import SingleTableMetadata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score
from scipy.spatial.distance import cosine
from FirstProcessing.risk_calculation import calculate_risk_advanced
from EDA.V1.data_loading import load_data
from EDA.V1.file_operations import select_save_directory
from FirstProcessing.file_operations import auto_adjust_column_width

# DEPRECATED: This class uses iterative dataset growth (max_size, step_fraction).
# This causes experimental drift and non-reproducible results.
# For reproducible experiments, use run_experiment.py with configs/*.yaml
class CTGANAugmentation:
    def __init__(self, target_column="Behavior_Risk_Level", step_fraction=0.25, max_size=2000, random_state=42, min_confidence=0.8):
        self.target_column = target_column
        self.step_fraction = step_fraction
        self.max_size = max_size
        self.random_state = random_state
        self.min_confidence = min_confidence
        # Force CPU usage
        self.device = torch.device("cpu")
        self.history = []

    def prepare_data(self, df):
        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]
        return X, y

    def build_metadata(self, df):
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(data=df)
        return metadata

    def generate_step(self, df_train, n_samples):
        metadata = self.build_metadata(df_train)
        # Explicitly set CUDA to False to ensure CPU usage
        model = CTGANSynthesizer(metadata, epochs=300, cuda=False)
        model.fit(df_train)

        # Generate samples with balanced classes using the correct conditions format
        half = n_samples // 2

        # Create proper condition objects for each class
        conditions_0 = [{self.target_column: 0} for _ in range(half)]
        conditions_1 = [{self.target_column: 1} for _ in range(n_samples - half)]

        # Sample from each condition
        samples_0 = model.sample(num_rows=half)
        samples_1 = model.sample(num_rows=n_samples - half)

        # Set the target column values explicitly
        samples_0[self.target_column] = 0
        samples_1[self.target_column] = 1

        # Concatenate the samples
        samples = pd.concat([samples_0, samples_1], ignore_index=True)
        return samples

    def relabel_confident(self, original_df, generated_df):
        X_orig, y_orig = self.prepare_data(original_df)
        X_gen, y_gen = self.prepare_data(generated_df)

        clf = LogisticRegression(max_iter=1000, random_state=self.random_state)
        clf.fit(X_orig, y_orig)

        proba = clf.predict_proba(X_gen)
        confidence = np.max(proba, axis=1)
        relabel_mask = confidence >= self.min_confidence
        y_pred = clf.predict(X_gen)

        changed = np.sum(y_pred[relabel_mask] != y_gen[relabel_mask])
        y_gen_final = y_gen.copy()
        y_gen_final[relabel_mask] = y_pred[relabel_mask]

        print(f"🔁 Relabeled {np.sum(relabel_mask)} samples, with {changed} changes.")
        generated_df[self.target_column] = y_gen_final
        return generated_df

    def validate_new_samples(self, original_df, augmented_df):
        X_orig, y_orig = self.prepare_data(original_df)
        X_new, y_new = self.prepare_data(augmented_df)

        clf = LogisticRegression(max_iter=1000, random_state=self.random_state)
        clf.fit(X_orig, y_orig)
        y_pred = clf.predict(X_new)

        kappa = cohen_kappa_score(y_new, y_pred)
        f1 = f1_score(y_new, y_pred, average='weighted')
        cos_sim = 1 - cosine(X_orig.mean(), X_new.mean())

        return {
            'cohen_kappa': kappa,
            'f1_score': f1,
            'cosine_similarity': cos_sim
        }

    def save_results(self, df_aug, history, save_dir):
        csv_path = os.path.join(save_dir, "augmented_dataset_encoded.csv")
        df_aug.to_csv(csv_path, index=False)

        excel_path = os.path.join(save_dir, "augmented_dataset_encoded.xlsx")
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            df_aug.to_excel(writer, index=False, sheet_name='Encoded_Data')
            auto_adjust_column_width(writer, 'Encoded_Data')

        metrics_path = os.path.join(save_dir, "augmentation_metrics.json")
        with open(metrics_path, 'w') as f:
            json.dump({"augmentation_history": history}, f, indent=2)

        print(f"\nSaved to {save_dir}")

    def augment_incrementally(self, df):
        df_current = df.copy()
        iteration = 0

        while len(df_current) < self.max_size:
            iteration += 1
            remaining = self.max_size - len(df_current)
            step_size = max(1, int(len(df_current) * self.step_fraction))
            to_add = min(step_size, remaining)

            print(f"\nIteration {iteration}: Generating {to_add} samples")
            generated = self.generate_step(df_current, to_add)
            generated = calculate_risk_advanced(generated)

            if self.target_column not in generated.columns:
                print("Generated samples missing target column. Skipping iteration.")
                continue

            print("Generated class distribution:", generated[self.target_column].value_counts().to_dict())

            # Relabel with high-confidence model predictions
            generated = self.relabel_confident(df_current, generated)

            metrics = self.validate_new_samples(df_current, generated)
            self.history.append({
                'iteration': iteration,
                'generated': to_add,
                'new_total': len(df_current) + to_add,
                'metrics': metrics
            })

            print(f"✅ Validation - F1: {metrics['f1_score']:.3f}, Kappa: {metrics['cohen_kappa']:.3f}, CosSim: {metrics['cosine_similarity']:.3f}")
            df_current = pd.concat([df_current, generated], ignore_index=True)

            # Clean memory
            gc.collect()

        return df_current, self.history


# DEPRECATED: This script uses iterative augmentation which causes experimental drift.
# Use run_experiment.py with configs/*.yaml for reproducible experiments.

if __name__ == "__main__":
    print("=" * 60)
    print("DEPRECATED: This augmentation script is disabled.")
    print("Reason: Iterative growth causes experimental drift.")
    print("")
    print("Use instead:")
    print("  python run_experiment.py --config configs/default.yaml")
    print("=" * 60)
    # Original code disabled:
    # augmenter = CTGANAugmentation(...)
    # df_augmented, history = augmenter.augment_incrementally(df)
