import pandas as pd

from src.feature_engineering.feature_engineering import (
    create_features,
    get_stage1_features,
    get_stage2_features,
)


def sample_data():
    return pd.DataFrame(
        {
            "loan_id": [1, 2],
            "no_of_dependents": [2, 0],
            "education": [" Graduate", " Not Graduate"],
            "self_employed": [" No", " Yes"],
            "income_annum": [10000000, 5000000],
            "loan_amount": [20000000, 5000000],
            "loan_term": [10, 5],
            "cibil_score": [750, 620],
            "residential_assets_value": [20000000, 5000000],
            "commercial_assets_value": [5000000, 2000000],
            "luxury_assets_value": [5000000, 1000000],
            "bank_asset_value": [10000000, 2000000],
            "loan_status": [" Approved", " Rejected"],
        }
    )


def test_feature_creation():
    df = sample_data()

    result = create_features(df)

    expected_features = [
        "total_assets",
        "loan_to_income_ratio",
        "assets_to_income_ratio",
        "income_per_dependent",
        "requested_loan_per_term",
        "bank_asset_ratio",
        "luxury_asset_ratio",
        "asset_coverage_ratio",
        "cibil_band",
    ]

    for feature in expected_features:
        assert feature in result.columns


def test_total_assets():
    df = sample_data()

    result = create_features(df)

    assert result.loc[0, "total_assets"] == 40000000
    assert result.loc[1, "total_assets"] == 10000000


def test_stage1_keeps_requested_loan_amount():
    df = sample_data()

    result = get_stage1_features(df)

    assert "loan_amount" in result.columns
    assert "loan_status" not in result.columns
    assert "loan_id" not in result.columns


def test_stage2_does_not_use_loan_amount():
    df = sample_data()

    result = get_stage2_features(df)

    assert "loan_amount" not in result.columns
    assert "loan_status" not in result.columns
    assert "loan_id" not in result.columns

def test_stage2_excludes_target_and_target_derived_features():
    df = sample_data()

    result = get_stage2_features(df)

    forbidden_features = [
        "loan_amount",
        "loan_to_income_ratio",
        "requested_loan_per_term",
        "asset_coverage_ratio",
    ]

    for feature in forbidden_features:
        assert feature not in result.columns