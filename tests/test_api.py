from fastapi.testclient import TestClient

from src.serving.app import app


client = TestClient(app)


# =========================================================
# TEST 1 — HEALTH ENDPOINT
# =========================================================


def test_health_endpoint():

    response = client.get(
        "/health"
    )

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"

    assert (
        data["stage1_model_version"]
        == "1.0.0"
    )

    assert (
        data["stage2_model_version"]
        == "1.0.0"
    )


# =========================================================
# TEST 2 — MODEL INFO
# =========================================================


def test_model_info_endpoint():

    response = client.get(
        "/model-info"
    )

    assert response.status_code == 200

    data = response.json()

    assert "stage1" in data

    assert "stage2" in data

    assert (
        data["stage1"]["model"]
        == "RandomForestClassifier"
    )

    assert (
        data["stage2"]["model"]
        == "LinearRegression"
    )


# =========================================================
# TEST 3 — VALID PREDICTION
# =========================================================


def test_predict_endpoint():

    application = {
        "no_of_dependents": 2,
        "education": "Graduate",
        "self_employed": "No",
        "income_annum": 10000000,
        "loan_amount": 20000000,
        "loan_term": 10,
        "cibil_score": 750,
        "residential_assets_value": 20000000,
        "commercial_assets_value": 5000000,
        "luxury_assets_value": 5000000,
        "bank_asset_value": 10000000,
    }

    response = client.post(
        "/predict",
        json=application,
    )

    assert response.status_code == 200

    data = response.json()

    assert (
        data["approval_decision"]
        in {"Approved", "Rejected"}
    )

    assert (
        0.0
        <= data["approval_probability"]
        <= 1.0
    )

    assert (
        data["stage1_model_version"]
        == "1.0.0"
    )


# =========================================================
# TEST 4 — INVALID CIBIL
# =========================================================


def test_invalid_cibil_is_rejected():

    application = {
        "no_of_dependents": 2,
        "education": "Graduate",
        "self_employed": "No",
        "income_annum": 10000000,
        "loan_amount": 20000000,
        "loan_term": 10,
        "cibil_score": 1000,
        "residential_assets_value": 20000000,
        "commercial_assets_value": 5000000,
        "luxury_assets_value": 5000000,
        "bank_asset_value": 10000000,
    }

    response = client.post(
        "/predict",
        json=application,
    )

    assert response.status_code == 422


# =========================================================
# TEST 5 — NEGATIVE INCOME
# =========================================================


def test_negative_income_is_rejected():

    application = {
        "no_of_dependents": 2,
        "education": "Graduate",
        "self_employed": "No",
        "income_annum": -100,
        "loan_amount": 20000000,
        "loan_term": 10,
        "cibil_score": 750,
        "residential_assets_value": 20000000,
        "commercial_assets_value": 5000000,
        "luxury_assets_value": 5000000,
        "bank_asset_value": 10000000,
    }

    response = client.post(
        "/predict",
        json=application,
    )

    assert response.status_code == 422


# =========================================================
# TEST 6 — MISSING FIELD
# =========================================================


def test_missing_field_is_rejected():

    application = {
        "no_of_dependents": 2,
        "education": "Graduate",
        "self_employed": "No",
        "income_annum": 10000000,
        "loan_amount": 20000000,
        "loan_term": 10,
        # cibil_score intentionally missing
        "residential_assets_value": 20000000,
        "commercial_assets_value": 5000000,
        "luxury_assets_value": 5000000,
        "bank_asset_value": 10000000,
    }

    response = client.post(
        "/predict",
        json=application,
    )

    assert response.status_code == 422


# =========================================================
# TEST 7 — ZERO LOAN AMOUNT
# =========================================================


def test_zero_loan_amount_is_rejected():

    application = {
        "no_of_dependents": 2,
        "education": "Graduate",
        "self_employed": "No",
        "income_annum": 10000000,
        "loan_amount": 0,
        "loan_term": 10,
        "cibil_score": 750,
        "residential_assets_value": 20000000,
        "commercial_assets_value": 5000000,
        "luxury_assets_value": 5000000,
        "bank_asset_value": 10000000,
    }

    response = client.post(
        "/predict",
        json=application,
    )

    assert response.status_code == 422