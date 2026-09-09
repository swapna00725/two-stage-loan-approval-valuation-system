"""
API performance benchmark for the
Two Stage Loan Approval & Valuation System.
"""

import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import requests


# =========================================================
# CONFIGURATION
# =========================================================

API_URL = "http://127.0.0.1:8000/predict"

NUM_REQUESTS = 100

REPORT_PATH = Path(
    "artifacts/reports/docker_benchmark.json"
)


# =========================================================
# SAMPLE APPLICATION
# =========================================================

APPLICATION = {
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
# PERCENTILE CALCULATION
# =========================================================

def calculate_percentile(
    values,
    percentile,
):
    """
    Calculate percentile without
    requiring NumPy.
    """

    sorted_values = sorted(values)

    index = (
        percentile
        / 100
        * (len(sorted_values) - 1)
    )

    lower = int(index)

    upper = min(
        lower + 1,
        len(sorted_values) - 1,
    )

    weight = index - lower

    return (
        sorted_values[lower]
        + weight
        * (
            sorted_values[upper]
            - sorted_values[lower]
        )
    )


# =========================================================
# BENCHMARK
# =========================================================

def run_benchmark():

    latencies = []

    successful_requests = 0

    failed_requests = 0

    # -----------------------------------------------------
    # Benchmark wall-clock start
    # -----------------------------------------------------

    benchmark_start = time.perf_counter()

    print("=" * 60)

    print(
        "API PERFORMANCE BENCHMARK"
    )

    print("=" * 60)

    print(
        f"Requests       : {NUM_REQUESTS}"
    )

    print(
        f"Endpoint       : {API_URL}"
    )

    print("-" * 60)

    # -----------------------------------------------------
    # Send requests
    # -----------------------------------------------------

    for i in range(NUM_REQUESTS):

        start = time.perf_counter()

        try:

            response = requests.post(
                API_URL,
                json=APPLICATION,
                timeout=10,
            )

            elapsed = (
                time.perf_counter()
                - start
            )

            elapsed_ms = (
                elapsed * 1000
            )

            if response.status_code == 200:

                successful_requests += 1

                latencies.append(
                    elapsed_ms
                )

            else:

                failed_requests += 1

        except requests.RequestException:

            failed_requests += 1

    # -----------------------------------------------------
    # Benchmark wall-clock end
    # -----------------------------------------------------

    benchmark_end = time.perf_counter()

    total_wall_time_seconds = (
        benchmark_end
        - benchmark_start
    )

    # -----------------------------------------------------
    # No successful requests
    # -----------------------------------------------------

    if not latencies:

        print(
            "No successful requests."
        )

        return

    # =====================================================
    # METRICS
    # =====================================================

    average_latency = (
        statistics.mean(latencies)
    )

    median_latency = (
        statistics.median(latencies)
    )

    p95_latency = (
        calculate_percentile(
            latencies,
            95,
        )
    )

    max_latency = max(
        latencies
    )

    success_rate = (
        successful_requests
        / NUM_REQUESTS
        * 100
    )

    # -----------------------------------------------------
    # Throughput
    # -----------------------------------------------------

    throughput = (
        successful_requests
        / total_wall_time_seconds
    )

    # =====================================================
    # CONSOLE OUTPUT
    # =====================================================

    print()

    print(
        f"Successful     : "
        f"{successful_requests}"
    )

    print(
        f"Failed         : "
        f"{failed_requests}"
    )

    print(
        f"Success rate   : "
        f"{success_rate:.2f}%"
    )

    print()

    print(
        f"Average latency: "
        f"{average_latency:.2f} ms"
    )

    print(
        f"p50 latency    : "
        f"{median_latency:.2f} ms"
    )

    print(
        f"p95 latency    : "
        f"{p95_latency:.2f} ms"
    )

    print(
        f"Max latency    : "
        f"{max_latency:.2f} ms"
    )

    print()

    print(
        f"Throughput     : "
        f"{throughput:.2f} requests/sec"
    )

    print("=" * 60)

    # =====================================================
    # REPORT
    # =====================================================

    report = {

        "benchmark_type": "Dockerized API",

        "timestamp_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "endpoint": API_URL,

        "total_requests": NUM_REQUESTS,

        "successful_requests": (
            successful_requests
        ),

        "failed_requests": (
            failed_requests
        ),

        "success_rate_percent": round(
            success_rate,
            2,
        ),

        "latency_ms": {

            "average": round(
                average_latency,
                2,
            ),

            "p50": round(
                median_latency,
                2,
            ),

            "p95": round(
                p95_latency,
                2,
            ),

            "max": round(
                max_latency,
                2,
            ),
        },

        "total_wall_time_seconds": round(
            total_wall_time_seconds,
            4,
        ),

        "throughput_requests_per_second": round(
            throughput,
            2,
        ),

        "status": (
            "PASS"
            if failed_requests == 0
            else "PARTIAL"
        ),
    }

    # -----------------------------------------------------
    # Create report directory
    # -----------------------------------------------------

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -----------------------------------------------------
    # Save JSON report
    # -----------------------------------------------------

    with REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
        )

    print()

    print(
        f"Report saved  : "
        f"{REPORT_PATH}"
    )


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    run_benchmark()