# Model building utilities

from sklearn.linear_model import LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGBOOST = True
except ImportError:
    XGBClassifier = None
    XGBRegressor = None
    HAS_XGBOOST = False

try:
    from lightgbm import LGBMClassifier, LGBMRegressor
    HAS_LIGHTGBM = True
except ImportError:
    LGBMClassifier = None
    LGBMRegressor = None
    HAS_LIGHTGBM = False


SUPPORTED_MODELS = {
    'regression': ['ridge', 'lasso', 'xgboost_reg', 'lightgbm_reg', 'random_forest_reg'],
    'classification': ['logistic_regression', 'random_forest', 'xgboost_clf', 'lightgbm_clf']
}

# Models that support random_state parameter
MODELS_WITH_RANDOM_STATE = [
    'xgboost_reg', 'lightgbm_reg', 'random_forest_reg',
    'logistic_regression', 'random_forest', 'xgboost_clf', 'lightgbm_clf'
]

# Models that are deterministic (no random_state needed)
DETERMINISTIC_MODELS = ['ridge', 'lasso']


def build_model(config):
    """
    Build and return a model instance based on config.

    Note: Ridge and Lasso are deterministic solvers and don't use random_state.
    Tree-based and ensemble models use random_state for reproducibility.
    """
    model_type = config['model']['type']
    params = config['model'].get('params', {}).get(model_type, {})
    seed = config['experiment']['seed']

    # Regression models
    if model_type == 'ridge':
        # Ridge is deterministic, no random_state
        return Ridge(**params)

    elif model_type == 'lasso':
        # Lasso is deterministic, no random_state
        return Lasso(**params)

    elif model_type == 'xgboost_reg':
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
        return XGBRegressor(random_state=seed, verbosity=0, **params)

    elif model_type == 'lightgbm_reg':
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")
        return LGBMRegressor(random_state=seed, verbose=-1, **params)

    elif model_type == 'random_forest_reg':
        return RandomForestRegressor(random_state=seed, **params)

    # Classification models
    elif model_type == 'logistic_regression':
        return LogisticRegression(random_state=seed, **params)

    elif model_type == 'random_forest':
        return RandomForestClassifier(random_state=seed, **params)

    elif model_type == 'xgboost_clf':
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed. Run: pip install xgboost")
        return XGBClassifier(
            random_state=seed,
            verbosity=0,
            use_label_encoder=False,
            eval_metric='logloss',
            **params
        )

    elif model_type == 'lightgbm_clf':
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")
        return LGBMClassifier(random_state=seed, verbose=-1, **params)

    else:
        raise ValueError(
            f"Unknown model type: '{model_type}'. "
            f"Supported: {SUPPORTED_MODELS}"
        )


def get_model_info(model_type):
    """Get information about a model type."""
    info = {
        'type': model_type,
        'supports_random_state': model_type in MODELS_WITH_RANDOM_STATE,
        'is_deterministic': model_type in DETERMINISTIC_MODELS,
        'task_type': 'regression' if model_type in SUPPORTED_MODELS['regression'] else 'classification'
    }
    return info

