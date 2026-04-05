# Frozen inference utilities for multitask/transfer hybrid models.

import json
import os
from typing import Dict, Any, List

import joblib
import numpy as np
import pandas as pd
import torch
import yaml

from experiments.multitask import MultiTaskModel
from experiments.domain_transfer import DomainTransferModel


class HybridInferenceEngine:
    """Load exported bundle and provide predict()/explain() for app integration."""

    def __init__(self, bundle_dir: str, device: str = "cpu"):
        self.bundle_dir = bundle_dir
        self.device = torch.device(device)

        self.metadata = self._load_json("model_metadata.json")
        self.thresholds = self._load_json("thresholds.json")
        self.feature_columns = self._load_json("feature_columns.json")
        self.rules = self._load_yaml("bank_mapping_rules.yaml")

        self.scaler = joblib.load(os.path.join(bundle_dir, "scaler.pkl"))
        self.model = self._load_model()

    def _load_json(self, name: str) -> Dict[str, Any]:
        path = os.path.join(self.bundle_dir, name)
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _load_yaml(self, name: str) -> Dict[str, Any]:
        path = os.path.join(self.bundle_dir, name)
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _load_model(self):
        family = self.metadata.get("model_family", "hybrid_transfer")
        model_cfg = self.metadata.get("model_config", {})
        model_path = os.path.join(self.bundle_dir, "model.pt")

        if family == "multitask":
            model = MultiTaskModel(
                input_dim=int(model_cfg["input_dim"]),
                hidden_dims=list(model_cfg.get("hidden_dims", [64, 32])),
                dropout=float(model_cfg.get("dropout", 0.3)),
                activation=str(model_cfg.get("activation", "relu")),
            )
            state = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state)
            model.to(self.device)
            model.eval()
            return model

        # Default: domain transfer/hybrid family.
        model = DomainTransferModel(
            adv_input_dim=int(model_cfg["adv_input_dim"]),
            gmsc_input_dim=int(model_cfg.get("gmsc_input_dim", 10)),
            config=model_cfg,
        )
        state = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return model

    def _prepare_input(self, features: Dict[str, Any]) -> pd.DataFrame:
        row = {c: float(features.get(c, 0.0)) for c in self.feature_columns}
        return pd.DataFrame([row])[self.feature_columns]

    def _forward(self, scaled_input: np.ndarray):
        x = torch.tensor(scaled_input, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            family = self.metadata.get("model_family", "hybrid_transfer")
            if family == "multitask":
                risk_pred, savings_logits = self.model(x)
            else:
                out = self.model.forward(x, domain=0)
                risk_pred = out["risk"]
                savings_logits = out["savings"]

            risk_score = float(risk_pred.squeeze().cpu().item())
            saving_probability = float(torch.sigmoid(savings_logits.squeeze()).cpu().item())
        return risk_score, saving_probability

    def _risk_attributions(self, scaled_input: np.ndarray, top_k: int = 5) -> List[Dict[str, float]]:
        x = torch.tensor(scaled_input, dtype=torch.float32, device=self.device, requires_grad=True)
        family = self.metadata.get("model_family", "hybrid_transfer")

        if family == "multitask":
            risk_pred, _ = self.model(x)
        else:
            out = self.model.forward(x, domain=0)
            risk_pred = out["risk"]

        risk_pred.squeeze().backward()
        grads = x.grad.detach().cpu().numpy()[0]
        vals = scaled_input[0]
        contrib = grads * vals

        idx = np.argsort(np.abs(contrib))[::-1][:top_k]
        return [
            {
                "feature": self.feature_columns[i],
                "contribution": float(contrib[i]),
                "abs_contribution": float(abs(contrib[i])),
            }
            for i in idx
        ]

    def _rule_alerts(self, raw: Dict[str, Any]) -> List[str]:
        t = self.rules.get("thresholds", {})
        alerts = []

        if float(raw.get("Impulse_Buying_Frequency", 0.0)) >= float(t.get("discretionary_spending_high", 3.0)):
            alerts.append("cheltuieli discretionare mari")

        if float(raw.get("Debt_Level", 0.0)) >= float(t.get("credit_dependency_high", 3.0)):
            alerts.append("dependenta de credit")

        essentials = float(raw.get("Essential_Needs_Percentage", 0.0))
        income = max(float(raw.get("Income_Category", 1.0)), 1.0)
        if essentials / income >= float(t.get("essential_income_ratio_high", 0.45)):
            alerts.append("cheltuieli esentiale prea mari raportat la venit")

        if float(raw.get("Savings_Goal_Emergency_Fund", 0.0)) <= float(t.get("emergency_buffer_low", 0.0)):
            alerts.append("lipsa buffer financiar")

        return alerts

    def predict(self, features: Dict[str, Any]) -> Dict[str, Any]:
        X = self._prepare_input(features)
        scaled = self.scaler.transform(X)

        risk_score, saving_probability = self._forward(scaled)
        top_factors = self._risk_attributions(scaled, top_k=int(self.thresholds.get("top_k_factors", 5)))
        alerts = self._rule_alerts(features)

        save_threshold = float(self.thresholds.get("saving_probability_threshold", 0.5))
        confidence = float(np.clip(abs(saving_probability - save_threshold) / 0.5, 0.0, 1.0))

        return {
            "risk_score": risk_score,
            "saving_probability": saving_probability,
            "top_factors": top_factors,
            "alerts": alerts,
            "confidence": confidence,
        }

    def explain(self, features: Dict[str, Any]) -> Dict[str, Any]:
        out = self.predict(features)
        return {
            "risk_score": out["risk_score"],
            "top_factors": out["top_factors"],
            "alerts": out["alerts"],
            "confidence": out["confidence"],
        }


