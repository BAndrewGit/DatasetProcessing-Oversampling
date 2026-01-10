# Config schema validation
# Validates config structure, types, and forbidden keys

REQUIRED_KEYS = {
    'experiment': ['name', 'seed'],
    'data': ['target_column', 'target_type'],
    'model': ['type'],
    'preprocessing': [],
    'cross_validation': ['n_splits'],
}

ALLOWED_TARGET_TYPES = ['regression', 'classification']

ALLOWED_MODEL_TYPES = [
    'ridge', 'lasso', 'xgboost_reg', 'lightgbm_reg', 'random_forest_reg',
    'logistic_regression', 'random_forest', 'xgboost_clf', 'lightgbm_clf'
]

ALLOWED_AUGMENTATION_METHODS = ['jitter', 'smote', 'cluster', None]

# Keys that indicate deprecated/forbidden patterns
FORBIDDEN_KEYS = [
    'max_size',
    'target_total',
    'iterative',
    'iterative_growth',
    'retrain_on_synthetic',
    'final_balancing',
    'exponential_growth',
]

FORBIDDEN_TARGETS = ['Behavior_Risk_Level']


class ConfigValidationError(Exception):
    """Raised when config validation fails."""
    pass


def validate_config(config, mode='baseline'):
    """
    Validate experiment configuration.

    Args:
        config: dict - Configuration dictionary
        mode: str - 'baseline', 'augmentation', or 'multitask'

    Raises:
        ConfigValidationError if validation fails
    """
    errors = []

    # Check required top-level keys
    for section, required_keys in REQUIRED_KEYS.items():
        if section not in config:
            errors.append(f"Missing required section: '{section}'")
            continue

        # Skip target validation for multitask mode
        if mode == 'multitask' and section == 'data':
            # Multitask doesn't use target_column/target_type in config
            continue

        # Skip model.type requirement for multitask mode (uses neural network)
        if mode == 'multitask' and section == 'model':
            # Multitask uses hidden_dims instead of type
            continue

        for key in required_keys:
            if key not in config[section]:
                errors.append(f"Missing required key: '{section}.{key}'")

    if errors:
        raise ConfigValidationError("Config validation failed:\n  - " + "\n  - ".join(errors))

    # Validate target_type (skip for multitask)
    if mode != 'multitask':
        target_type = config['data'].get('target_type')
        if target_type not in ALLOWED_TARGET_TYPES:
            errors.append(f"Invalid target_type '{target_type}'. Allowed: {ALLOWED_TARGET_TYPES}")

        # Validate target is not forbidden
        target = config['data'].get('target_column')
        if target in FORBIDDEN_TARGETS:
            errors.append(f"FORBIDDEN target '{target}'. Use 'Risk_Score' or 'Save_Money_Yes'.")

    # Validate model type (skip for multitask - uses neural network)
    if mode != 'multitask':
        model_type = config['model'].get('type')
        if model_type not in ALLOWED_MODEL_TYPES:
            errors.append(f"Invalid model type '{model_type}'. Allowed: {ALLOWED_MODEL_TYPES}")

    # Validate augmentation method if present
    aug_config = config.get('augmentation', {})
    if aug_config.get('enabled'):
        method = aug_config.get('method')
        if method not in ALLOWED_AUGMENTATION_METHODS:
            errors.append(f"Invalid augmentation method '{method}'. Allowed: {ALLOWED_AUGMENTATION_METHODS}")

    # Check for forbidden keys (deprecated patterns)
    forbidden_found = _find_forbidden_keys(config)
    if forbidden_found:
        errors.append(f"FORBIDDEN keys detected (deprecated patterns): {forbidden_found}")

    # Mode-specific validation
    if mode == 'baseline':
        if aug_config.get('enabled'):
            errors.append(
                "BLOCKED: Augmentation is enabled in a baseline config. "
                "Baseline experiments must have augmentation.enabled = false. "
                "Use run_augmentation_experiment.py for augmentation experiments."
            )

    # Validate types
    if not isinstance(config['experiment'].get('seed'), int):
        errors.append("experiment.seed must be an integer")

    if not isinstance(config['cross_validation'].get('n_splits'), int):
        errors.append("cross_validation.n_splits must be an integer")

    n_splits = config['cross_validation'].get('n_splits', 5)
    if n_splits < 2:
        errors.append("cross_validation.n_splits must be >= 2")

    # Validate augmentation ratio bounds
    if aug_config.get('enabled'):
        ratio = aug_config.get('synthetic_ratio', 0.15)
        max_ratio = aug_config.get('max_ratio', 0.30)
        if ratio > max_ratio:
            errors.append(f"synthetic_ratio ({ratio}) exceeds max_ratio ({max_ratio})")
        if max_ratio > 0.50:
            errors.append(f"max_ratio ({max_ratio}) exceeds absolute maximum 0.50")

    if errors:
        raise ConfigValidationError("Config validation failed:\n  - " + "\n  - ".join(errors))

    return True


def _find_forbidden_keys(config, prefix=''):
    """Recursively find forbidden keys in config."""
    found = []
    if isinstance(config, dict):
        for key, value in config.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if key in FORBIDDEN_KEYS:
                found.append(full_key)
            found.extend(_find_forbidden_keys(value, full_key))
    return found


def validate_baseline_config(config):
    """Validate config specifically for baseline experiments."""
    return validate_config(config, mode='baseline')


def validate_augmentation_config(config):
    """Validate config specifically for augmentation experiments."""
    return validate_config(config, mode='augmentation')

