# DEPRECATED: Use run_baseline.py or run_augmentation_experiment.py instead
#
# This file is kept for backward compatibility but will be removed in future versions.
#
# For baseline experiments (no augmentation):
#   python runners/run_baseline.py --config configs/baseline/regression.yaml --dataset data.csv
#
# For augmentation experiments:
#   python runners/augmentation_experiment.py --config configs/augmentation/experiment.yaml --dataset data.csv

import warnings
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    warnings.warn(
        "\n" + "=" * 70 + "\n"
        "DEPRECATION WARNING: run_experiment.py is deprecated.\n"
        "Use the following instead:\n"
        "  - runners/run_baseline.py           (for clean baseline experiments)\n"
        "  - runners/augmentation_experiment.py (for augmentation experiments)\n"
        "=" * 70,
        DeprecationWarning,
        stacklevel=2
    )

    # Import and delegate to run_baseline for backward compatibility
    from runners.run_baseline import main as baseline_main
    baseline_main()


if __name__ == "__main__":
    main()

