import pandas as pd

from src.feature_engineering.feature_engineering import (
    get_stage2_features,
)
from src.preprocessing.preprocessing import (
    get_stage2_preprocessor,
)


def test_stage2_preprocessor_works_without_loan_amount():

    df = pd.DataFrame(
        {
            "loan_id": [1, 2],
            "no_of_dependents": [2, 0],
            "education": ["Graduate", "Not Graduate"],
            "self_employed": ["No", "Yes"],
            "income_annum": [10000000, 5000000],
            "loan_amount": [20000000, 5000000],
            "loan_term": [10, 5],
            "cibil_score": [750, 620],
            "residential_assets_value": [20000000, 5000000],
            "commercial_assets_value": [5000000, 2000000],
            "luxury_assets_value": [5000000, 1000000],
            "bank_asset_value": [10000000, 2000000],
            "loan_status": ["Approved", "Rejected"],
        }
    )

    X = get_stage2_features(df)

    assert "loan_amount" not in X.columns

    preprocessor = get_stage2_preprocessor()

    transformed = preprocessor.fit_transform(X)

    assert transformed is not None
    assert transformed.shape[0] == 2
    assert transformed.shape[1] > 0