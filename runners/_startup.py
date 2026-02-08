# Startup configuration for silencing warnings
# Import this module FIRST in any runner to suppress common warnings

import os
import warnings
import logging

# Silence TensorFlow warnings - MUST be set before TF is imported
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=info, 2=warning, 3=error only
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Suppress oneDNN messages

# Silence joblib/loky physical core detection warning on Windows
os.environ['LOKY_MAX_CPU_COUNT'] = str(os.cpu_count() or 4)

# Configure logging to suppress TF warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('keras').setLevel(logging.ERROR)

# Suppress common Python warnings
warnings.filterwarnings('ignore', message='.*Could not find the number of physical cores.*')
warnings.filterwarnings('ignore', category=UserWarning, module='joblib')
warnings.filterwarnings('ignore', category=UserWarning, module='loky')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='tensorflow')
warnings.filterwarnings('ignore', category=DeprecationWarning, module='keras')
warnings.filterwarnings('ignore', message='.*tf.losses.*deprecated.*')
warnings.filterwarnings('ignore', message='.*sparse_softmax_cross_entropy.*')


