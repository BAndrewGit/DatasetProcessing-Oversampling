import json

import joblib
import numpy as np
import torch
import yaml
from sklearn.preprocessing import StandardScaler

from experiments.inference_hybrid import HybridInferenceEngine
from experiments.multitask import MultiTaskModel


def test_hybrid_inference_predict_and_explain_contract(tmp_path):
    feature_columns = [
        "Debt_Level",
        "Impulse_Buying_Frequency",
        "Essential_Needs_Percentage",
        "Savings_Goal_Emergency_Fund",
        "Income_Category",
    ]

    X_fit = np.array(
        [
            [1.0, 2.0, 0.2, 1.0, 5.0],
            [2.0, 3.0, 0.4, 0.0, 4.0],
            [3.0, 1.0, 0.3, 1.0, 6.0],
            [2.0, 2.0, 0.5, 0.0, 3.0],
        ]
    )

    scaler = StandardScaler().fit(X_fit)
    joblib.dump(scaler, tmp_path / "scaler.pkl")

    model = MultiTaskModel(input_dim=len(feature_columns), hidden_dims=[8, 4], dropout=0.0, activation="relu")
    torch.save(model.state_dict(), tmp_path / "model.pt")

    metadata = {
        "model_family": "multitask",
        "model_config": {
            "input_dim": len(feature_columns),
            "hidden_dims": [8, 4],
            "dropout": 0.0,
            "activation": "relu",
        },
    }
    with open(tmp_path / "model_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    with open(tmp_path / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, indent=2)

    thresholds = {"saving_probability_threshold": 0.5, "top_k_factors": 3}
    with open(tmp_path / "thresholds.json", "w", encoding="utf-8") as f:
        json.dump(thresholds, f, indent=2)

    rules = {
        "thresholds": {
            "discretionary_spending_high": 3.0,
            "credit_dependency_high": 3.0,
            "essential_income_ratio_high": 0.3,
            "emergency_buffer_low": 0.0,
        }
    }
    with open(tmp_path / "bank_mapping_rules.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(rules, f, sort_keys=False)

    engine = HybridInferenceEngine(str(tmp_path))

    features = {
        "Debt_Level": 4.0,
        "Impulse_Buying_Frequency": 4.0,
        "Essential_Needs_Percentage": 2.0,
        "Savings_Goal_Emergency_Fund": 0.0,
        "Income_Category": 3.0,
    }

    pred = engine.predict(features)
    exp = engine.explain(features)

    for key in ["risk_score", "saving_probability", "top_factors", "alerts", "confidence"]:
        assert key in pred

    assert isinstance(pred["top_factors"], list)
    assert isinstance(pred["alerts"], list)
    assert 0.0 <= pred["saving_probability"] <= 1.0
    assert 0.0 <= pred["confidence"] <= 1.0

    for key in ["risk_score", "top_factors", "alerts", "confidence"]:
        assert key in exp

