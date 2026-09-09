import pandas as pd

from src.preprocessing.preprocessing import (
    CATEGORICAL_COLUMNS,
    NUMERICAL_COLUMNS,
    get_preprocessor,
)


def test_preprocessor_can_fit_and_transform():

    data = {
        # -------------------------------------------------
        # Raw numerical features
        # -------------------------------------------------
        "no_of_dependents": [2, 0, 3],
        "income_annum": [10000000, 5000000, 15000000],
        "loan_amount": [20000000, 5000000, 25000000],
        "loan_term": [10, 5, 15],
        "cibil_score": [750, 620, 800],

        "residential_assets_value": [
            20000000,
            5000000,
            30000000,
        ],

        "commercial_assets_value": [
            5000000,
            2000000,
            10000000,
        ],

        "luxury_assets_value": [
            5000000,
            1000000,
            8000000,
        ],

        "bank_asset_value": [
            10000000,
            2000000,
            15000000,
        ],

        # -------------------------------------------------
        # Engineered numerical features
        # -------------------------------------------------
        "total_assets": [
            40000000,
            10000000,
            63000000,
        ],

        "loan_to_income_ratio": [
            2.0,
            1.0,
            1.6667,
        ],

        "assets_to_income_ratio": [
            4.0,
            2.0,
            4.2,
        ],

        "income_per_dependent": [
            5000000,
            5000000,
            5000000,
        ],

        "requested_loan_per_term": [
            2000000,
            1000000,
            1666666.67,
        ],

        "bank_asset_ratio": [
            0.25,
            0.20,
            0.2381,
        ],

        "luxury_asset_ratio": [
            0.125,
            0.10,
            0.1270,
        ],

        "asset_coverage_ratio": [
            2.0,
            2.0,
            2.52,
        ],

        # -------------------------------------------------
        # Categorical features
        # -------------------------------------------------
        "education": [
            "Graduate",
            "Not Graduate",
            "Graduate",
        ],

        "self_employed": [
            "No",
            "Yes",
            "No",
        ],

        "cibil_band": [
            "Excellent",
            "Average",
            "Excellent",
        ],
    }

    X = pd.DataFrame(data)

    preprocessor = get_preprocessor()

    transformed = preprocessor.fit_transform(X)

    assert transformed is not None
    assert transformed.shape[0] == 3
    assert transformed.shape[1] > 0


def test_column_definitions():

    assert "income_annum" in NUMERICAL_COLUMNS
    assert "cibil_score" in NUMERICAL_COLUMNS
    assert "loan_amount" in NUMERICAL_COLUMNS

    assert "education" in CATEGORICAL_COLUMNS
    assert "self_employed" in CATEGORICAL_COLUMNS

    # Engineered features must also be included
    assert "total_assets" in NUMERICAL_COLUMNS
    assert "assets_to_income_ratio" in NUMERICAL_COLUMNS
    assert "income_per_dependent" in NUMERICAL_COLUMNS
    assert "bank_asset_ratio" in NUMERICAL_COLUMNS
    assert "luxury_asset_ratio" in NUMERICAL_COLUMNS