# DEPRECATED: Use run_baseline.py or run_augmentation_experiment.py instead
#
# This file is kept for backward compatibility but will be removed in future versions.
#
# For baseline experiments (no augmentation):
#   python run_baseline.py --config configs/baseline_regression.yaml --dataset data.csv
#
# For augmentation experiments:
#   python run_augmentation_experiment.py --config configs/augmentation_experiment.yaml --dataset data.csv

import warnings
import sys

def main():
    warnings.warn(
        "\n" + "=" * 70 + "\n"
        "DEPRECATION WARNING: run_experiment.py is deprecated.\n"
        "Use the following instead:\n"
        "  - run_baseline.py           (for clean baseline experiments)\n"
        "  - run_augmentation_experiment.py (for augmentation experiments)\n"
        "=" * 70,
        DeprecationWarning,
        stacklevel=2
    )

    # Import and delegate to run_baseline for backward compatibility
    from run_baseline import main as baseline_main
    baseline_main()


if __name__ == "__main__":
    main()

