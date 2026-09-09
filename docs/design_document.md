# Two Stage Loan Approval & Valuation System

## MLOps Design Document

**Version:** 1.0.0\
**Purpose:** Academic / portfolio-oriented MLOps implementation

## 1. Introduction

The Two Stage Loan Approval & Valuation System is an end-to-end machine
learning application designed to support two important steps in a loan
decision workflow. The project combines data ingestion, feature
engineering, preprocessing, model training and evaluation, model
persistence, API serving, Docker deployment, automated testing,
monitoring, drift detection, batch inference, benchmarking,
configuration management, and continuous integration.

The system contains two sequential machine learning stages:

1.  **Stage 1 -- Loan Approval Classification:** determines whether a
    loan application is Approved or Rejected using a
    RandomForestClassifier.
2.  **Stage 2 -- Loan Valuation Regression:** for approved applications,
    estimates a recommended loan amount using LinearRegression.

The models are exposed through FastAPI. The application is containerized
using Docker, while GitHub Actions automatically runs tests and verifies
the Docker image.

## 2. Problem Definition and Objectives

### Problem Definition

The system addresses two related questions:

-   Should the loan application be approved or rejected?
-   If approved, what loan amount can be recommended based on applicant
    characteristics?

The workflow is:

``` text
Loan Application
      |
      v
Data Validation
      |
      v
Feature Engineering & Preprocessing
      |
      v
Stage 1: Approval Classification
      |
   +--+--+
   |     |
Approved Rejected
   |     |
   v     v
Stage 2  Stop
Valuation
   |
   v
Recommended Loan Amount
   |
   v
FastAPI Response
```

### Objectives

-   Automate initial loan approval classification.
-   Estimate a recommended loan amount for approved applications.
-   Maintain consistency between training and inference preprocessing.
-   Expose the models through a REST API.
-   Package the application using Docker.
-   Validate functionality through automated tests.
-   Support batch inference.
-   Monitor recent application data for quality and feature drift.
-   Produce evaluation, benchmark, and monitoring artifacts.
-   Separate operational configuration from source code.
-   Use GitHub Actions for continuous integration.

### Intended Users

Potential users include bank loan officers, financial institutions,
credit analysts, loan processing teams, and authorized internal
applications. The system is intended as decision support and would
require institutional policies, validation, security controls, and human
oversight before real-world financial deployment.

## 3. Data and Feature Design

### Dataset

The supplied `loan_approval_dataset.csv` contains 4,269 rows and 13
columns. The supplied dataset has 0 missing values and 0 duplicate rows.

Main attributes include:

  Feature                      Description
  ---------------------------- -------------------------
  `loan_id`                    Application identifier
  `no_of_dependents`           Number of dependents
  `education`                  Education level
  `self_employed`              Employment status
  `income_annum`               Annual income
  `loan_amount`                Requested loan amount
  `loan_term`                  Loan term
  `cibil_score`                CIBIL credit score
  `residential_assets_value`   Residential asset value
  `commercial_assets_value`    Commercial asset value
  `luxury_assets_value`        Luxury asset value
  `bank_asset_value`           Bank asset value
  `loan_status`                Approval target

`loan_id` is treated as an identifier rather than a predictive feature.

### Feature Engineering

The system generates:

-   `total_assets`
-   `loan_to_income_ratio`
-   `assets_to_income_ratio`
-   `income_per_dependent`
-   `requested_loan_per_term`
-   `bank_asset_ratio`
-   `luxury_asset_ratio`
-   `asset_coverage_ratio`
-   `cibil_band`

CIBIL bands are:

    CIBIL Score Band
  ------------- -----------
          ≤ 549 Poor
       550--649 Fair
       650--749 Good
          ≥ 750 Excellent

Categorical variables use one-hot encoding, while numerical variables
use the preprocessing pipeline.

### Training/Serving Consistency

The same feature engineering and preprocessing logic is reused during
inference so that serving features remain consistent with training
features.

## 4. Two-Stage Model Design

### Stage 1 -- Approval Classification

Stage 1 is a binary classification task with `loan_status` as the
target.

Selected model: **RandomForestClassifier**

Metrics:

-   Accuracy
-   Precision
-   Recall
-   F1-score
-   ROC-AUC

Latest evaluation:

  Metric         Score
  ----------- --------
  Accuracy      1.0000
  Precision     1.0000
  Recall        1.0000
  F1-score      1.0000
  ROC-AUC       1.0000

Artifact:

`artifacts/models/stage1_approval_model.joblib`

The near-perfect results should be interpreted cautiously because the
supplied dataset may contain highly deterministic relationships.
Representative real-world validation would be required before production
use.

### Stage 2 -- Loan Valuation

Stage 2 runs for approved applications. Target: `loan_amount`.

Selected model: **LinearRegression**

  Model                            MAE      RMSE        R²
  -------------------------- --------- --------- ---------
  Linear Regression            \~2.63M   \~3.45M   \~0.851
  Random Forest Regression     \~2.61M   \~3.48M   \~0.849

Linear Regression was selected because the Random Forest candidate did
not improve the overall promotion criteria.

Artifact:

`artifacts/models/stage2_valuation_model.joblib`

### Target Leakage Prevention

Stage 2 excludes `loan_amount` because it is the target. It also
excludes target-dependent engineered features:

-   `loan_to_income_ratio`
-   `requested_loan_per_term`
-   `asset_coverage_ratio`

This prevents indirect leakage from the value being predicted.

## 5. Model Evaluation and Artifact Management

Important artifacts include:

``` text
artifacts/
├── models/
│   ├── stage1_approval_model.joblib
│   └── stage2_valuation_model.joblib
├── reports/
│   ├── stage1_evaluation.json
│   ├── stage2_evaluation.json
│   └── docker_benchmark.json
├── monitoring/
└── logs/
```

Saved artifacts allow a specific trained model to be reused without
retraining whenever the service starts.

## 6. Model Serving

The serving layer is implemented using FastAPI in `src/serving/app.py`.

  Endpoint        Method   Purpose
  --------------- -------- -----------------------------------
  `/health`       GET      Service health and model versions
  `/model-info`   GET      Deployed model information
  `/predict`      POST     Loan prediction

Example successful response:

``` json
{
  "approval_decision": "Approved",
  "approval_probability": 0.9966666666666667,
  "recommended_loan_amount": 30538981.3308796,
  "stage1_model_version": "1.0.0",
  "stage2_model_version": "1.0.0"
}
```

Pydantic validation checks constraints such as positive income, positive
loan amount, positive loan term, CIBIL range 0--900, and non-negative
asset values.

## 7. Docker Deployment

The application is containerized with Docker and includes Python 3.11,
dependencies, source code, trained models, configuration, FastAPI, and
Uvicorn.

Build:

``` bash
docker build -t loan-approval-valuation:1.0 .
```

Run:

``` bash
docker run -p 8000:8000 loan-approval-valuation:1.0
```

The API is exposed on port 8000. The same Dockerfile is validated in
GitHub Actions.

## 8. Testing Strategy

The project uses pytest for API, ingestion, preprocessing, training,
monitoring, and drift-related tests.

Latest complete test execution:

**41 passed, 2 warnings**

Run locally:

``` bash
pytest -v
```

## 9. Batch Inference

Batch prediction is supported with:

``` bash
python -m src.utils.batch_inference   --input data/recent_batch/sample_applications.csv   --output data/recent_batch/predictions.csv
```

Latest validation processed 5 input applications and produced 5
predictions successfully.

## 10. Monitoring and Drift Detection

Monitoring includes:

1.  Data-quality checks
2.  Missing-value checks
3.  Feature-distribution comparison
4.  Drift detection
5.  Prediction-distribution inspection
6.  Retraining recommendation

Configured drift threshold: **20%**

Latest monitoring result:

-   Recent rows: 5
-   Missing values: 0
-   Data quality: PASS
-   Feature drift detected: YES
-   Retraining required: YES
-   Reason: Feature drift detected

Output:

`artifacts/monitoring/monitoring_report.json`

The current implementation recommends retraining; it does not
automatically retrain and redeploy a model.

## 11. Performance Benchmarking

The Dockerized API was benchmarked using 100 requests.

  Metric                             Result
  --------------------- -------------------
  Total requests                        100
  Successful requests                   100
  Failed requests                         0
  Success rate                         100%
  Average latency                 160.85 ms
  P50 latency                     149.63 ms
  P95 latency                     219.55 ms
  Maximum latency                 296.52 ms
  Throughput              6.22 requests/sec

Report:

`artifacts/reports/docker_benchmark.json`

These results are environment-specific benchmark results rather than
universal production capacity guarantees.

## 12. Configuration Management

Project configuration is maintained in:

`configs/config.yaml`

It can contain project metadata, data paths, artifact paths, model
settings, model versions, monitoring thresholds, and API settings. This
separates operational parameters from application code.

## 13. Continuous Integration

GitHub Actions workflow:

`.github/workflows/main.yml`

Pipeline:

``` text
Code Push
    |
    v
Checkout Repository
    |
    v
Setup Python 3.11
    |
    v
Install Dependencies
    |
    v
Run Pytest
    |
    v
Build Docker Image
    |
    v
Verify Docker Image
    |
    v
CI Success
```

The current CI pipeline successfully validates the Python environment,
dependencies, automated tests, Docker image build, and Docker image
verification.

## 14. End-to-End MLOps Lifecycle

``` text
Problem Definition
       |
       v
Data Ingestion
       |
       v
Feature Engineering
       |
       v
Preprocessing
       |
       v
Model Training & Evaluation
       |
       v
Model Artifact Storage
       |
       v
FastAPI Serving
       |
       v
Docker Deployment
       |
       v
Monitoring
       |
       v
Drift Detection
       |
       v
Retraining Recommendation
```

GitHub Actions provides automation across testing and container
verification.

## 15. Trade-offs and Design Decisions

**Random Forest for Stage 1:** selected for strong classification
performance and ability to model non-linear relationships. The trade-off
is greater complexity and the need to investigate near-perfect results.

**Linear Regression for Stage 2:** selected because the Random Forest
candidate did not improve the promotion criteria. It is simple and
interpretable, but may not capture complex non-linear relationships.

**File-based artifacts:** simple and appropriate for this academic
implementation, but production systems would normally use managed
storage and a model registry.

**Lightweight monitoring:** easy to understand and implement, but
production systems would require richer statistical monitoring,
alerting, and historical tracking.

**Docker deployment:** provides reproducibility and portability without
the complexity of Kubernetes, but does not yet provide cloud-native
orchestration or autoscaling.

## 16. Security, Reliability and Governance

Before real financial deployment, the system would require
authentication, authorization, role-based access control, encryption,
secure secret management, audit logging, data privacy controls, model
governance, fairness assessment, human review, institutional
credit-policy integration, and stronger reliability controls.

Loan decisions can materially affect individuals, so model predictions
should not replace required institutional, regulatory, or human decision
processes.

## 17. Limitations

1.  The dataset contains only 4,269 records.
2.  Stage 1 has near-perfect evaluation scores and requires further
    validation.
3.  The supplied dataset may contain deterministic relationships.
4.  Monitoring is lightweight.
5.  Retraining is recommended rather than automatic.
6.  No production database or feature store is implemented.
7.  No model registry is implemented.
8.  Enterprise authentication is not implemented.
9.  Deployment is containerized but not cloud-native.
10. Human underwriting workflows are outside scope.

## 18. Future Enhancements

Potential improvements include:

-   MLflow experiment tracking
-   Model registry
-   Automated retraining
-   Automated model promotion and rollback
-   Cloud deployment
-   Kubernetes orchestration
-   Feature-store integration
-   Database integration
-   Advanced drift monitoring
-   Model performance monitoring
-   SHAP-based explainability
-   Authentication and role-based authorization
-   Real-time dashboards
-   Automated alerting
-   Human-in-the-loop review
-   Fairness and bias analysis

## 19. Repository Structure

``` text
loan_approval_n_valuation_systems/
|
├── .github/
│   └── workflows/
│       └── main.yml
├── artifacts/
│   ├── logs/
│   ├── models/
│   ├── reports/
│   └── monitoring/
├── configs/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── recent_batch/
├── docs/
│   ├── design_document.md
│   └── images/
├── notebooks/
├── src/
│   ├── feature_engineering/
│   ├── ingestion/
│   ├── monitoring/
│   ├── pipeline/
│   ├── preprocessing/
│   ├── serving/
│   ├── training/
│   └── utils/
├── tests/
├── .dockerignore
├── .gitignore
├── Dockerfile
├── README.md
└── requirements.txt
```

## 20. Conclusion

The Two Stage Loan Approval & Valuation System demonstrates how a
machine learning solution can be developed as an end-to-end MLOps
application rather than remaining a standalone notebook experiment.

The system combines data processing, feature engineering, two-stage
machine learning, model artifacts, FastAPI serving, Docker, automated
testing, monitoring, drift detection, batch inference, benchmarking,
configuration management, and GitHub Actions CI.

Stage 1 performs approval classification using Random Forest, while
Stage 2 estimates a recommended loan amount using Linear Regression for
approved applications. The project provides a strong academic and
portfolio demonstration of modularity, reproducibility, deployment
readiness, monitoring, configuration management, and continuous
integration.

Future work can extend the system toward production through stronger
security and governance, automated retraining, model management, cloud
infrastructure, advanced monitoring, explainability, fairness analysis,
and human-in-the-loop decision processes.

## 21. Demonstration Artifacts

Supporting screenshots are maintained under `docs/images/`, including:

-   `architecture.png`
-   `project_structure.png`
-   `pytest_results.png`
-   `swagger_ui.png`
-   `prediction_response.png`
-   `docker_prediction_response.png`
-   `benchmark_results.png`
-   `api_performance.png`
-   `monitoring_report.png`
