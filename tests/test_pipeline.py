import pytest

from src.pipeline.pipeline import (
    LoanApprovalValuationPipeline,
)


@pytest.fixture
def pipeline():
    """
    Create a reusable pipeline instance for tests.
    """
    return LoanApprovalValuationPipeline()


@pytest.fixture
def valid_application():
    """
    Representative valid raw loan application.
    """

    return {
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


# =========================================================
# TEST 1 — PIPELINE INITIALIZATION
# =========================================================


def test_pipeline_initializes(pipeline):

    assert pipeline is not None

    assert pipeline.stage1_model is not None

    assert pipeline.stage2_model is not None


# =========================================================
# TEST 2 — VALID APPLICATION RETURNS RESULT
# =========================================================


def test_pipeline_returns_result(
    pipeline,
    valid_application,
):

    result = pipeline.predict(
        valid_application
    )

    assert result is not None

    assert isinstance(result, dict)


# =========================================================
# TEST 3 — APPROVAL DECISION EXISTS
# =========================================================


def test_pipeline_contains_approval_decision(
    pipeline,
    valid_application,
):

    result = pipeline.predict(
        valid_application
    )

    assert "approval_decision" in result

    assert result["approval_decision"] in {
        "Approved",
        "Rejected",
    }


# =========================================================
# TEST 4 — APPROVAL PROBABILITY IS VALID
# =========================================================


def test_approval_probability_is_valid(
    pipeline,
    valid_application,
):

    result = pipeline.predict(
        valid_application
    )

    probability = result[
        "approval_probability"
    ]

    assert probability is not None

    assert 0.0 <= probability <= 1.0


# =========================================================
# TEST 5 — APPROVED APPLICATION HAS VALUATION
# =========================================================


def test_approved_application_has_valuation(
    pipeline,
    valid_application,
):

    result = pipeline.predict(
        valid_application
    )

    if result["approval_decision"] == "Approved":

        assert (
            result["recommended_loan_amount"]
            is not None
        )

        assert (
            result["recommended_loan_amount"]
            >= 0
        )


# =========================================================
# TEST 6 — REJECTED APPLICATION DOES NOT HAVE
#          STAGE 2 VALUATION
# =========================================================


def test_rejected_application_has_no_valuation(
    pipeline,
    valid_application,
    monkeypatch,
):

    def fake_stage1_predict(features):

        return ["Rejected"]

    monkeypatch.setattr(
        pipeline.stage1_model,
        "predict",
        fake_stage1_predict,
    )

    result = pipeline.predict(
        valid_application
    )

    assert result[
        "approval_decision"
    ] == "Rejected"

    assert result[
        "recommended_loan_amount"
    ] is None

    assert result[
        "stage2_model_version"
    ] is None


# =========================================================
# TEST 7 — MISSING INPUT IS REJECTED
# =========================================================


def test_missing_required_field_raises_error(
    pipeline,
    valid_application,
):

    invalid_application = (
        valid_application.copy()
    )

    del invalid_application[
        "cibil_score"
    ]

    with pytest.raises(ValueError):

        pipeline.predict(
            invalid_application
        )


# =========================================================
# TEST 8 — MODEL VERSIONS ARE RETURNED
# =========================================================


def test_model_versions_are_returned(
    pipeline,
    valid_application,
):

    result = pipeline.predict(
        valid_application
    )

    assert (
        result["stage1_model_version"]
        == "1.0.0"
    )

    if result["approval_decision"] == "Approved":

        assert (
            result["stage2_model_version"]
            == "1.0.0"
        )