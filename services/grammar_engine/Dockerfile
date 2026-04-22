# MLAF Grammar Engine — FastAPI + SWI-Prolog
# Multi-stage build for minimal production image

FROM python:3.11-slim AS base

# Install SWI-Prolog (required by pyswip)
RUN apt-get update && \
    apt-get install -y --no-install-recommends swi-prolog && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ app/
COPY prolog/ prolog/

# Create data directories
RUN mkdir -p data/raw data/processed data/custom data/splits logs models

# Expose port
EXPOSE 8300

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8300/health')" || exit 1

# Run with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8300", "--workers", "2"]
