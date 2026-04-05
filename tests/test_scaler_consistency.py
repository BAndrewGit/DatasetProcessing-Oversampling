import joblib
import numpy as np
import pandas as pd
from typing import Any
from sklearn.preprocessing import StandardScaler

from FirstProcessing.risk_calculation import fit_and_save_scaler, apply_existing_scaler
from experiments.save_model import save_sklearn_model


def test_fit_and_save_scaler_roundtrip_preserves_transformed_values(tmp_path):
    df = pd.DataFrame(
        {
            "Debt_Level": [1.0, 2.0, 3.0, 4.0, 5.0],
            "Impulse_Buying_Frequency": [5.0, 4.0, 3.0, 2.0, 1.0],
            "keep_col": [10, 11, 12, 13, 14],
        }
    )
    cols = ["Debt_Level", "Impulse_Buying_Frequency"]
    scaler_path = tmp_path / "std_scaler.joblib"

    expected_scaler: Any = StandardScaler().fit(df[cols])
    expected_transformed = expected_scaler.transform(df[cols])

    out_df = fit_and_save_scaler(df.copy(), cols, str(scaler_path))

    assert scaler_path.exists()
    np.testing.assert_allclose(out_df[cols].to_numpy(), expected_transformed, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(out_df["keep_col"].to_numpy(), df["keep_col"].to_numpy(), rtol=0, atol=0)

    reloaded_scaler: Any = joblib.load(scaler_path)
    roundtrip = reloaded_scaler.transform(df[cols])
    np.testing.assert_allclose(roundtrip, expected_transformed, rtol=1e-10, atol=1e-12)


def test_apply_existing_scaler_matches_original_transform(tmp_path):
    train_df = pd.DataFrame(
        {
            "Debt_Level": [1.0, 2.0, 3.0, 4.0],
            "Impulse_Buying_Frequency": [2.0, 1.0, 0.0, -1.0],
        }
    )
    infer_df = pd.DataFrame(
        {
            "Debt_Level": [2.5, 3.5],
            "Impulse_Buying_Frequency": [0.5, -0.5],
            "untouched": [99, 100],
        }
    )
    cols = ["Debt_Level", "Impulse_Buying_Frequency"]
    scaler: Any = StandardScaler().fit(train_df[cols])
    scaler_path = tmp_path / "exported_scaler.joblib"
    joblib.dump(scaler, scaler_path)

    expected = scaler.transform(infer_df[cols])
    out_df = apply_existing_scaler(infer_df.copy(), cols, str(scaler_path))

    np.testing.assert_allclose(out_df[cols].to_numpy(), expected, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(out_df["untouched"].to_numpy(), infer_df["untouched"].to_numpy(), rtol=0, atol=0)


def test_save_sklearn_model_scaler_export_import_is_consistent(tmp_path):
    X = np.array(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ]
    )
    scaler = StandardScaler().fit(X)

    scaler_path = tmp_path / "saved_scaler.joblib"
    save_sklearn_model(scaler, str(scaler_path))
    loaded: Any = joblib.load(scaler_path)

    np.testing.assert_allclose(loaded.mean_, scaler.mean_, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(loaded.scale_, scaler.scale_, rtol=1e-12, atol=1e-12)

    original_tx = scaler.transform(X)
    loaded_tx = loaded.transform(X)
    np.testing.assert_allclose(loaded_tx, original_tx, rtol=1e-10, atol=1e-12)


