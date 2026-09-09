from pathlib import Path

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "stage1_approval_model.joblib"
)


def test_stage1_model_exists():
    assert MODEL_PATH.exists()


def test_stage1_model_can_be_loaded():
    model = joblib.load(MODEL_PATH)

    assert model is not None


def test_stage1_model_has_predict():
    model = joblib.load(MODEL_PATH)

    assert hasattr(model, "predict")


def test_stage1_model_has_predict_proba():
    model = joblib.load(MODEL_PATH)

    assert hasattr(model, "predict_proba")