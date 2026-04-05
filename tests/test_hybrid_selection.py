from runners.run_hybrid_production import _choose_final, select_final_model


def _metric(mean, std=0.01):
    return {"mean": mean, "std": std}


def test_choose_final_prefers_hybrid_when_all_risk_first_criteria_hold():
    multitask = {
        "risk_mae": _metric(0.20, 0.02),
        "risk_spearman": _metric(0.45, 0.03),
        "savings_macro_f1": _metric(0.90, 0.02),
    }
    hybrid = {
        "risk_mae": _metric(0.19, 0.02),
        "risk_spearman": _metric(0.43, 0.03),
        "savings_macro_f1": _metric(0.89, 0.02),
    }

    family, report = _choose_final(multitask, hybrid)
    assert family == "hybrid_transfer"
    assert report["mae_ok"] is True
    assert report["spearman_ok"] is True
    assert report["stability_ok"] is True


def test_choose_final_falls_back_to_multitask_on_spearman_drop():
    multitask = {
        "risk_mae": _metric(0.20, 0.02),
        "risk_spearman": _metric(0.45, 0.03),
        "savings_macro_f1": _metric(0.90, 0.02),
    }
    hybrid = {
        "risk_mae": _metric(0.19, 0.02),
        "risk_spearman": _metric(0.30, 0.03),
        "savings_macro_f1": _metric(0.90, 0.02),
    }

    family, report = _choose_final(multitask, hybrid)
    assert family == "multitask"
    assert report["spearman_ok"] is False


def test_select_final_model_returns_winner_checkpoint_path():
    multitask = {
        "metrics": {
            "risk_mae": _metric(0.1845, 0.02),
            "risk_spearman": _metric(0.4847, 0.03),
            "risk_r2": _metric(0.30, 0.04),
            "savings_macro_f1": _metric(0.88, 0.02),
        },
        "checkpoint_path": "mt.pth",
    }
    hybrid = {
        "metrics": {
            "risk_mae": _metric(0.2119, 0.01),
            "risk_spearman": _metric(0.3017, 0.03),
            "risk_r2": _metric(0.22, 0.04),
            "savings_macro_f1": _metric(0.82, 0.02),
        },
        "checkpoint_path": "hy.pth",
    }

    family, report, winner = select_final_model(multitask, hybrid)
    assert family == "multitask"
    assert report["mae_ok"] is False
    assert report["spearman_ok"] is False
    assert report["savings_ok"] is False
    assert winner == "mt.pth"


