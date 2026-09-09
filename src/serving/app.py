from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.pipeline.pipeline import (
    LoanApprovalValuationPipeline,
)


# =========================================================
# APPLICATION
# =========================================================

app = FastAPI(
    title="Two Stage Loan Approval & Valuation System",
    description=(
        "Production-style ML API for loan approval "
        "and loan valuation."
    ),
    version="1.0.0",
)


# =========================================================
# PIPELINE
# =========================================================

pipeline = LoanApprovalValuationPipeline()


# =========================================================
# REQUEST SCHEMA
# =========================================================

class LoanApplication(BaseModel):

    no_of_dependents: int = Field(
        ...,
        ge=0,
        description="Number of dependents",
    )

    education: str = Field(
        ...,
        description="Education level",
    )

    self_employed: str = Field(
        ...,
        description="Employment status",
    )

    income_annum: float = Field(
        ...,
        gt=0,
        description="Annual income",
    )

    loan_amount: float = Field(
        ...,
        gt=0,
        description="Requested loan amount",
    )

    loan_term: int = Field(
        ...,
        gt=0,
        description="Loan term",
    )

    cibil_score: float = Field(
        ...,
        ge=0,
        le=900,
        description="CIBIL score",
    )

    residential_assets_value: float = Field(
        ...,
        ge=0,
    )

    commercial_assets_value: float = Field(
        ...,
        ge=0,
    )

    luxury_assets_value: float = Field(
        ...,
        ge=0,
    )

    bank_asset_value: float = Field(
        ...,
        ge=0,
    )


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/health")
def health_check():

    return {
        "status": "healthy",
        "service": (
            "two-stage-loan-approval-valuation"
        ),
        "stage1_model_version": (
            pipeline.STAGE1_VERSION
        ),
        "stage2_model_version": (
            pipeline.STAGE2_VERSION
        ),
    }


# =========================================================
# MODEL INFORMATION
# =========================================================

@app.get("/model-info")
def model_info():

    return {
        "stage1": {
            "model": "RandomForestClassifier",
            "version": pipeline.STAGE1_VERSION,
        },
        "stage2": {
            "model": "LinearRegression",
            "version": pipeline.STAGE2_VERSION,
        },
    }


# =========================================================
# PREDICTION
# =========================================================

@app.post("/predict")
def predict(application: LoanApplication):

    try:

        result = pipeline.predict(
            application.model_dump()
        )

        return result

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed: "
                f"{str(exc)}"
            ),
        )