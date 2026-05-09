# Production hybrid runner: compares multitask vs hybrid-transfer and exports inference bundle.

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Dict, Tuple

import pandas as pd
import torch
import joblib

RUNNERS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(RUNNERS_DIR)
sys.path.insert(0, RUNNERS_DIR)
sys.path.insert(0, PROJECT_ROOT)

from experiments.data import RISK_SCORE_COMPONENTS
from experiments.save_model import export_inference_bundle as _export_inference_bundle
from run_multitask_experiment import run_multitask_experiment
from run_domain_transfer_experiment import (
    load_adv_data,
    load_gmsc_data,
    run_domain_transfer_cv,
    load_config,
    set_seeds,
)


def _load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _safe_metric(d: dict, name: str):
    if not d:
        return None, None
    node = d.get(name, {})
    return node.get("mean"), node.get("std")


def _feature_columns(dataset_path: str):
    df = pd.read_csv(dataset_path)
    exclude = {
        "Risk_Score",
        "Save_Money_Yes",
        "Save_Money_No",
        "Behavior_Risk_Level",
        "Confidence",
        "Outlier",
        "Cluster",
        "Auto_Label",
    }
    exclude.update(RISK_SCORE_COMPONENTS)
    return [c for c in df.columns if c not in exclude]


def _choose_final(multitask_res: dict, hybrid_res: dict):
    mt_risk_mae, mt_risk_mae_std = _safe_metric(multitask_res, "risk_mae")
    mt_risk_sp, mt_risk_sp_std = _safe_metric(multitask_res, "risk_spearman")
    mt_risk_r2, _ = _safe_metric(multitask_res, "risk_r2")
    mt_sav_f1, _ = _safe_metric(multitask_res, "savings_macro_f1")

    hy_risk_mae, hy_risk_mae_std = _safe_metric(hybrid_res, "risk_mae")
    hy_risk_sp, _ = _safe_metric(hybrid_res, "risk_spearman")
    hy_risk_r2, _ = _safe_metric(hybrid_res, "risk_r2")
    hy_sav_f1, _ = _safe_metric(hybrid_res, "savings_macro_f1")

    if None in [mt_risk_mae, mt_risk_sp, hy_risk_mae, hy_risk_sp]:
        return "multitask", {"reason": "missing_metrics"}

    spearman_ok = hy_risk_sp >= (mt_risk_sp - 0.03)
    mae_ok = hy_risk_mae <= mt_risk_mae
    risk_r2_ok = True
    if mt_risk_r2 is not None and hy_risk_r2 is not None:
        risk_r2_ok = hy_risk_r2 >= (mt_risk_r2 - 0.03)
    stability_ok = (hy_risk_mae_std or 0.0) <= (mt_risk_mae_std or 0.0) * 1.2
    savings_ok = (hy_sav_f1 or 0.0) >= (mt_sav_f1 or 0.0) * 0.95

    use_hybrid = spearman_ok and mae_ok and risk_r2_ok and stability_ok and savings_ok
    return (
        "hybrid_transfer" if use_hybrid else "multitask",
        {
            "spearman_ok": spearman_ok,
            "mae_ok": mae_ok,
            "risk_r2_ok": risk_r2_ok,
            "stability_ok": stability_ok,
            "savings_ok": savings_ok,
            "multitask_risk_mae": mt_risk_mae,
            "multitask_risk_spearman": mt_risk_sp,
            "multitask_risk_r2": mt_risk_r2,
            "hybrid_risk_mae": hy_risk_mae,
            "hybrid_risk_spearman": hy_risk_sp,
            "hybrid_risk_r2": hy_risk_r2,
        },
    )


def select_final_model(results_multitask: dict, results_hybrid: dict) -> Tuple[str, Dict, str]:
    """Select final family and expose winner checkpoint path for pipeline hooks."""
    multitask_metrics = results_multitask.get("metrics", results_multitask)
    hybrid_metrics = results_hybrid.get("metrics", results_hybrid)

    final_model_family, selection_report = _choose_final(multitask_metrics, hybrid_metrics)

    multitask_ckpt = results_multitask.get("checkpoint_path")
    hybrid_ckpt = results_hybrid.get("checkpoint_path")
    winner_checkpoint_path = hybrid_ckpt if final_model_family == "hybrid_transfer" else multitask_ckpt

    return final_model_family, selection_report, winner_checkpoint_path


def export_inference_bundle(winner_checkpoint: dict, output_dir: str) -> dict:
    """Export inference bundle from a selected winner checkpoint payload."""
    model_path = winner_checkpoint.get("checkpoint_path")
    scaler_path = winner_checkpoint.get("scaler_path")
    if not model_path or not scaler_path:
        raise ValueError("winner_checkpoint must include checkpoint_path and scaler_path")

    return _export_inference_bundle(
        bundle_dir=output_dir,
        model_src_path=model_path,
        scaler_src_path=scaler_path,
        feature_columns=winner_checkpoint.get("feature_columns", []),
        thresholds=winner_checkpoint.get("thresholds", {}),
        bank_mapping_rules=winner_checkpoint.get("bank_mapping_rules", {}),
        metadata=winner_checkpoint.get("metadata", {}),
    )


def _metric_verdict(multitask_val, hybrid_val, lower_is_better: bool, rel_tol: float = 0.01):
    if multitask_val is None or hybrid_val is None:
        return "missing"
    threshold = abs(multitask_val) * rel_tol
    delta = hybrid_val - multitask_val
    if lower_is_better:
        if delta < -threshold:
            return "improved"
        if delta > threshold:
            return "worse"
    else:
        if delta > threshold:
            return "improved"
        if delta < -threshold:
            return "worse"
    return "neutral"


def _build_comparison(multitask_res: dict, hybrid_res: dict, selection_report: dict, final_family: str):
    metric_specs = {
        "risk_mae": True,
        "risk_rmse": True,
        "risk_spearman": False,
        "risk_r2": False,
        "savings_macro_f1": False,
        "savings_accuracy": False,
    }
    metrics = {}
    for metric, lower_is_better in metric_specs.items():
        mt = multitask_res.get(metric, {})
        hy = hybrid_res.get(metric, {})
        metrics[metric] = {
            "multitask": mt,
            "hybrid": hy,
            "verdict": _metric_verdict(mt.get("mean"), hy.get("mean"), lower_is_better),
        }

    return {
        "final_model_family": final_family,
        "selection_report": selection_report,
        "metrics": metrics,
    }


def run_hybrid_production(
    multitask_config: str,
    transfer_config: str,
    dataset_path: str,
    gmsc_path: str,
    output_dir: str = "runs",
    source_pipeline_run: str = None,
    pipeline_run_id: str = None,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    decision_dir = os.path.join(output_dir, f"final_decision_{timestamp}")
    os.makedirs(decision_dir, exist_ok=True)

    # 1) Train multitask-only baseline family.
    multitask_run_dir = run_multitask_experiment(
        multitask_config,
        dataset_path,
        output_dir=output_dir,
        multitask_only=True,
    )
    multitask_results = _load_json(os.path.join(multitask_run_dir, "ablation_results.json"))
    multitask_final = multitask_results.get("multitask", {})
    multitask_grad_logs = multitask_results.get("grad_logs", [])

    # 2) Train hybrid transfer family ONLY (no ADV-only branch).
    cfg = load_config(transfer_config)
    cfg.setdefault("experiment", {})["output_dir"] = output_dir
    seed = cfg.get("experiment", {}).get("seed", 42)
    set_seeds(seed)

    adv_X, adv_y_risk, adv_y_savings, _ = load_adv_data(cfg, dataset_path)
    gmsc_X, gmsc_y, _ = load_gmsc_data(cfg, gmsc_path)

    hybrid_raw = run_domain_transfer_cv(
        adv_X=adv_X,
        adv_y_risk=adv_y_risk,
        adv_y_savings=adv_y_savings,
        gmsc_X=gmsc_X,
        gmsc_y=gmsc_y,
        config=cfg,
        seed=seed,
        include_adv_only=False,
    )
    hybrid_final = hybrid_raw.get("transfer", {})

    hybrid_run_dir = os.path.join(output_dir, f"hybrid_transfer_only_{timestamp}")
    os.makedirs(hybrid_run_dir, exist_ok=True)

    saved = hybrid_raw.get("saved_models", {})
    transfer_models = saved.get("transfer_models", [])
    adv_scalers = saved.get("adv_scalers", [])
    if not transfer_models or not adv_scalers:
        raise RuntimeError("Hybrid training completed without saved model/scaler artifacts.")

    transfer_model_path = os.path.join(hybrid_run_dir, "transfer_model.pth")
    adv_scaler_path = os.path.join(hybrid_run_dir, "adv_scaler.joblib")
    torch.save(transfer_models[-1].state_dict(), transfer_model_path)
    joblib.dump(adv_scalers[-1], adv_scaler_path)

    with open(os.path.join(hybrid_run_dir, "final_hybrid_results.json"), "w", encoding="utf-8") as f:
        json.dump(hybrid_final, f, indent=2)

    # 3) Compare and choose final family.
    mt_model_path = os.path.join(multitask_run_dir, "multitask_model.pth")
    mt_scaler_path = os.path.join(multitask_run_dir, "scaler.joblib")
    hybrid_model_path = transfer_model_path
    hybrid_scaler_path = adv_scaler_path

    features = json.load(open(os.path.join(multitask_run_dir, "feature_columns.json")))
    rules = {
        "thresholds": {
            "discretionary_spending_high": 3.0,
            "credit_dependency_high": 3.0,
            "essential_income_ratio_high": 0.45,
            "emergency_buffer_low": 0.0,
        }
    }
    thresholds = {
        "saving_probability_threshold": 0.5,
        "top_k_factors": 5,
    }

    multitask_candidate = {
        "metrics": multitask_final,
        "checkpoint_path": mt_model_path,
        "scaler_path": mt_scaler_path,
        "feature_columns": features,
        "thresholds": thresholds,
        "bank_mapping_rules": rules,
        "source_run": multitask_run_dir,
    }
    hybrid_candidate = {
        "metrics": hybrid_final,
        "checkpoint_path": hybrid_model_path,
        "scaler_path": hybrid_scaler_path,
        "feature_columns": features,
        "thresholds": thresholds,
        "bank_mapping_rules": rules,
        "source_run": hybrid_run_dir,
    }

    final_family, selection_report, winner_checkpoint_path = select_final_model(
        multitask_candidate,
        hybrid_candidate,
    )

    with open(os.path.join(decision_dir, "final_multitask_results.json"), "w", encoding="utf-8") as f:
        json.dump(multitask_final, f, indent=2)
    with open(os.path.join(decision_dir, "final_hybrid_results.json"), "w", encoding="utf-8") as f:
        json.dump(hybrid_final, f, indent=2)

    comparison_payload = _build_comparison(multitask_final, hybrid_final, selection_report, final_family)
    with open(os.path.join(decision_dir, "final_comparison.json"), "w", encoding="utf-8") as f:
        json.dump(comparison_payload, f, indent=2)

    # Save gradient diagnostics for auditability.
    with open(os.path.join(decision_dir, "final_multitask_grad_logs.json"), "w", encoding="utf-8") as f:
        json.dump(multitask_grad_logs, f, indent=2)

    hybrid_training_logs = hybrid_raw.get('_last_fold_data', {}).get('training_logs', [])
    with open(os.path.join(decision_dir, "final_hybrid_grad_logs.json"), "w", encoding="utf-8") as f:
        json.dump(hybrid_training_logs, f, indent=2)

    # 4) Export frozen inference bundle.
    bundle_dir = os.path.join(decision_dir, "inference_bundle")

    metrics_summary = {
        "multitask": {
            "risk_mae": multitask_final.get("risk_mae", {}).get("mean"),
            "risk_spearman": multitask_final.get("risk_spearman", {}).get("mean"),
            "risk_r2": multitask_final.get("risk_r2", {}).get("mean"),
            "savings_macro_f1": multitask_final.get("savings_macro_f1", {}).get("mean"),
        },
        "hybrid_transfer": {
            "risk_mae": hybrid_final.get("risk_mae", {}).get("mean"),
            "risk_spearman": hybrid_final.get("risk_spearman", {}).get("mean"),
            "risk_r2": hybrid_final.get("risk_r2", {}).get("mean"),
            "savings_macro_f1": hybrid_final.get("savings_macro_f1", {}).get("mean"),
        },
    }

    if final_family == "hybrid_transfer":
        winner_payload = dict(hybrid_candidate)
        metadata = {
            "family": "hybrid_transfer",
            "model_family": "hybrid_transfer",
            "model_config": hybrid_raw.get("_model_config", {}),
            "run_id": pipeline_run_id or os.path.basename(decision_dir),
            "features_used": features,
            "metrics_summary": metrics_summary,
            "source_pipeline_run": source_pipeline_run,
            "source_run": hybrid_run_dir,
            "winner_checkpoint_path": winner_checkpoint_path,
            "selection_report": selection_report,
        }
    else:
        winner_payload = dict(multitask_candidate)
        mcfg = _load_json(os.path.join(multitask_run_dir, "config.json"))
        metadata = {
            "family": "multitask",
            "model_family": "multitask",
            "model_config": {
                "input_dim": len(features),
                "hidden_dims": mcfg.get("model", {}).get("hidden_dims", [64, 32]),
                "dropout": mcfg.get("model", {}).get("dropout", 0.3),
                "activation": mcfg.get("model", {}).get("activation", "relu"),
            },
            "run_id": pipeline_run_id or os.path.basename(decision_dir),
            "features_used": features,
            "metrics_summary": metrics_summary,
            "source_pipeline_run": source_pipeline_run,
            "source_run": multitask_run_dir,
            "winner_checkpoint_path": winner_checkpoint_path,
            "selection_report": selection_report,
        }

    winner_payload["metadata"] = metadata
    bundle_paths = export_inference_bundle(winner_payload, bundle_dir)

    comparison = {
        "decision_dir": decision_dir,
        "multitask_run": multitask_run_dir,
        "hybrid_run": hybrid_run_dir,
        "final_model_family": final_family,
        "winner_checkpoint_path": winner_checkpoint_path,
        "selection_report": selection_report,
        "bundle": bundle_paths,
    }

    with open(os.path.join(decision_dir, "comparison_report.json"), "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)

    return comparison


def main():
    parser = argparse.ArgumentParser(description="Run production hybrid selection + export inference bundle")
    parser.add_argument("--multitask-config", required=True)
    parser.add_argument("--transfer-config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--gmsc", required=True)
    parser.add_argument("--output", default="runs")
    args = parser.parse_args()

    result = run_hybrid_production(
        multitask_config=args.multitask_config,
        transfer_config=args.transfer_config,
        dataset_path=args.dataset,
        gmsc_path=args.gmsc,
        output_dir=args.output,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()







