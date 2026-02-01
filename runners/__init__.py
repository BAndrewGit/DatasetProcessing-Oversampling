# Experiment Runners Package
# Contains all experiment execution scripts

import os
# Set LOKY_MAX_CPU_COUNT early to silence joblib/loky warnings on Windows
os.environ.setdefault('LOKY_MAX_CPU_COUNT', str(os.cpu_count() or 1))

from . import run_baseline
from . import run_experiment
from . import run_multitask_experiment
from . import run_domain_transfer_experiment
from . import run_analysis
from . import augmentation_experiment

__all__ = [
    'run_baseline',
    'run_experiment',
    'run_multitask_experiment',
    'run_domain_transfer_experiment',
    'run_analysis',
    'augmentation_experiment'
]
