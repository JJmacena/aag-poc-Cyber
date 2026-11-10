# Behavioral Engine - FastAPI Service
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# Copy source code
COPY src/ /app/src/
COPY pyproject.toml /app/pyproject.toml

# Install package in development mode
RUN pip install --no-cache-dir -e .

# Create directories for data and models
RUN mkdir -p /data /models

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Default command - will be overridden by docker-compose entrypoint
CMD ["python", "-m", "aag_be.api"]