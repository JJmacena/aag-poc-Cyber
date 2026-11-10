# Agent Profiler - FastAPI Service
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Install additional dependencies for profiler
RUN pip install --no-cache-dir \
    fastapi==0.109.0 \
    uvicorn==0.27.0 \
    pydantic==2.5.3 \
    pydantic-settings==2.1.0 \
    sentence-transformers==3.0.0 \
    faiss-cpu==1.8.0 \
    torch==2.1.2 \
    numpy==1.26.2 \
    pyarrow==14.0.2 \
    pandas==2.1.4 \
    scikit-learn==1.3.2 \
    httpx==0.26.0

# Copy source code
COPY src/ /app/src/
COPY pyproject.toml /app/pyproject.toml

# Install package in development mode
RUN pip install --no-cache-dir -e .

# Create directories
RUN mkdir -p /data /models

# Expose port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8002/health || exit 1

# Default command
CMD ["python", "-m", "aag_profiler.api"]