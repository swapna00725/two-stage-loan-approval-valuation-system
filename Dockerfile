# =========================================================
# TWO STAGE LOAN APPROVAL & VALUATION SYSTEM
# Dockerfile
# =========================================================

FROM python:3.11-slim

# ---------------------------------------------------------
# Environment settings
# ---------------------------------------------------------

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ---------------------------------------------------------
# Working directory
# ---------------------------------------------------------

WORKDIR /app

# ---------------------------------------------------------
# Install Python dependencies
# ---------------------------------------------------------

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------
# Copy application source code
# ---------------------------------------------------------

COPY src ./src

# ---------------------------------------------------------
# Copy trained model artifacts
# ---------------------------------------------------------

COPY artifacts/models ./artifacts/models

# ---------------------------------------------------------
# Copy configuration
# ---------------------------------------------------------

COPY configs ./configs

# ---------------------------------------------------------
# Create required runtime directories
# ---------------------------------------------------------

RUN mkdir -p artifacts/logs \
    artifacts/reports \
    artifacts/monitoring

# ---------------------------------------------------------
# Expose FastAPI port
# ---------------------------------------------------------

EXPOSE 8000

# ---------------------------------------------------------
# Start FastAPI application
# ---------------------------------------------------------

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000"]