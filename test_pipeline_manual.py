from src.pipeline.pipeline import (
    LoanApprovalValuationPipeline,
)


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


pipeline = LoanApprovalValuationPipeline()

result = pipeline.predict(application)

print("\n" + "=" * 60)
print("TWO STAGE LOAN APPROVAL & VALUATION")
print("=" * 60)

for key, value in result.items():

    print(
        f"{key:30}: {value}"
    )

print("=" * 60)