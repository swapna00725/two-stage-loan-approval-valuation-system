import pandas as pd

from src.monitoring.monitoring import (
    check_data_quality,
    calculate_drift,
    evaluate_drift,
    monitor_predictions,
    should_retrain,
)


# =========================================================
# SAMPLE DATA
# =========================================================

def create_valid_application_data():

    return pd.DataFrame(
        {
            "no_of_dependents": [1, 2],
            "education": [
                "Graduate",
                "Not Graduate",
            ],
            "self_employed": [
                "No",
                "Yes",
            ],
            "income_annum": [
                5000000,
                8000000,
            ],
            "loan_amount": [
                10000000,
                15000000,
            ],
            "loan_term": [
                10,
                15,
            ],
            "cibil_score": [
                700,
                750,
            ],
            "residential_assets_value": [
                10000000,
                15000000,
            ],
            "commercial_assets_value": [
                2000000,
                3000000,
            ],
            "luxury_assets_value": [
                1000000,
                2000000,
            ],
            "bank_asset_value": [
                5000000,
                7000000,
            ],
        }
    )


# =========================================================
# TEST 1 — VALID DATA QUALITY
# =========================================================

def test_valid_data_passes_quality_check():

    df = create_valid_application_data()

    result = check_data_quality(df)

    assert result["row_count"] == 2

    assert result["total_nulls"] == 0

    assert result["quality_status"] == "PASS"

    assert result["quality_issues"] == []


# =========================================================
# TEST 2 — NULL DETECTION
# =========================================================

def test_missing_values_are_detected():

    df = create_valid_application_data()

    df.loc[0, "income_annum"] = None

    result = check_data_quality(df)

    assert result["total_nulls"] == 1

    assert (
        result["quality_status"]
        == "WARNING"
    )

    assert (
        "Missing values detected"
        in result["quality_issues"]
    )


# =========================================================
# TEST 3 — INVALID CIBIL DETECTION
# =========================================================

def test_invalid_cibil_is_detected():

    df = create_valid_application_data()

    df.loc[0, "cibil_score"] = 1000

    result = check_data_quality(df)

    assert (
        result["quality_status"]
        == "WARNING"
    )

    assert (
        "Invalid CIBIL score detected"
        in result["quality_issues"]
    )


# =========================================================
# TEST 4 — DRIFT CALCULATION
# =========================================================

def test_drift_is_calculated():

    reference = pd.DataFrame(
        {
            "income_annum": [
                100,
                100,
            ]
        }
    )

    recent = pd.DataFrame(
        {
            "income_annum": [
                120,
                120,
            ]
        }
    )

    result = calculate_drift(
        reference,
        recent,
    )

    assert (
        result["income_annum"][
            "reference_mean"
        ]
        == 100
    )

    assert (
        result["income_annum"][
            "recent_mean"
        ]
        == 120
    )

    assert (
        result["income_annum"][
            "drift_percentage"
        ]
        == 20
    )


# =========================================================
# TEST 5 — DRIFT THRESHOLD
# =========================================================

def test_drift_threshold_detects_drift():

    drift_results = {
        "income_annum": {
            "reference_mean": 100,
            "recent_mean": 130,
            "drift_percentage": 30,
        }
    }

    result = evaluate_drift(
        drift_results,
        threshold=20,
    )

    assert result[
        "drift_detected"
    ] is True

    assert (
        "income_annum"
        in result["drifted_features"]
    )


# =========================================================
# TEST 6 — NO DRIFT
# =========================================================

def test_no_drift_below_threshold():

    drift_results = {
        "income_annum": {
            "reference_mean": 100,
            "recent_mean": 105,
            "drift_percentage": 5,
        }
    }

    result = evaluate_drift(
        drift_results,
        threshold=20,
    )

    assert result[
        "drift_detected"
    ] is False

    assert (
        result["drifted_features"]
        == []
    )


# =========================================================
# TEST 7 — PREDICTION MONITORING
# =========================================================

def test_prediction_monitoring():

    predictions = pd.DataFrame(
        {
            "approval_decision": [
                "Approved",
                "Approved",
                "Rejected",
            ]
        }
    )

    result = monitor_predictions(
        predictions
    )

    assert result["status"] == "PASS"

    assert (
        result["prediction_counts"][
            "Approved"
        ]
        == 2
    )

    assert (
        result["prediction_counts"][
            "Rejected"
        ]
        == 1
    )


# =========================================================
# TEST 8 — RETRAINING TRIGGER
# =========================================================

def test_retraining_triggered_by_drift():

    result = should_retrain(
        drift_detected=True
    )

    assert (
        result["retrain_required"]
        is True
    )

    assert (
        "Feature drift detected"
        in result["reasons"]
    )


# =========================================================
# TEST 9 — NO RETRAINING
# =========================================================

def test_retraining_not_required():

    result = should_retrain(
        drift_detected=False,
        model_performance_drop=False,
        sufficient_new_data=False,
    )

    assert (
        result["retrain_required"]
        is False
    )

    assert result["reasons"] == []