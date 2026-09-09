from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "stage2_valuation_model.joblib"
)


def test_stage2_model_exists():
    """
    The promoted Stage 2 model artifact should exist
    after training.
    """

    assert MODEL_PATH.exists()


def test_stage2_model_can_be_loaded():
    """
    The saved Stage 2 model should be loadable.
    """

    model = joblib.load(MODEL_PATH)

    assert model is not None


def test_stage2_model_has_predict():
    """
    Stage 2 is a regression model and must expose predict().
    """

    model = joblib.load(MODEL_PATH)

    assert hasattr(model, "predict")


def test_stage2_model_prediction_is_numeric():
    """
    The Stage 2 model should produce a numeric loan valuation.
    """

    model = joblib.load(MODEL_PATH)

    sample = pd.DataFrame(
        {
            "no_of_dependents": [2],
            "education": ["Graduate"],
            "self_employed": ["No"],
            "income_annum": [10000000],
            "loan_term": [10],
            "cibil_score": [750],

            "residential_assets_value": [20000000],
            "commercial_assets_value": [5000000],
            "luxury_assets_value": [5000000],
            "bank_asset_value": [10000000],

            # Engineered Stage 2 features
            "total_assets": [40000000],
            "assets_to_income_ratio": [4.0],
            "income_per_dependent": [5000000],
            "bank_asset_ratio": [0.25],
            "luxury_asset_ratio": [0.125],
            "cibil_band": ["Excellent"],
        }
    )

    prediction = model.predict(sample)

    assert len(prediction) == 1

    assert isinstance(
        float(prediction[0]),
        float,
    )


def test_stage2_prediction_is_positive():
    """
    A loan valuation should not be negative.
    """

    model = joblib.load(MODEL_PATH)

    sample = pd.DataFrame(
        {
            "no_of_dependents": [2],
            "education": ["Graduate"],
            "self_employed": ["No"],
            "income_annum": [10000000],
            "loan_term": [10],
            "cibil_score": [750],

            "residential_assets_value": [20000000],
            "commercial_assets_value": [5000000],
            "luxury_assets_value": [5000000],
            "bank_asset_value": [10000000],

            "total_assets": [40000000],
            "assets_to_income_ratio": [4.0],
            "income_per_dependent": [5000000],
            "bank_asset_ratio": [0.25],
            "luxury_asset_ratio": [0.125],
            "cibil_band": ["Excellent"],
        }
    )

    prediction = float(
        model.predict(sample)[0]
    )

    assert prediction >= 0